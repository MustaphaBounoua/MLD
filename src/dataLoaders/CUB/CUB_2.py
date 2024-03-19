import io
import json
import os
import pickle
from collections import Counter, OrderedDict
from collections import defaultdict
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
from nltk.tokenize import sent_tokenize, word_tokenize
from torch.utils.data import Dataset
from torchvision import transforms, models, datasets
from torchnet.dataset import TensorDataset, ResampleDataset
import nltk
#nltk.download('punkt')
import argparse
maxSentLen = 32

class OrderedCounter(Counter, OrderedDict):
    """Counter that remembers the order elements are first encountered."""

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)


class CUBSentences(Dataset):

    def __init__(self,  split, transform=None, root_data_dir= "data/data_cub", **kwargs):
        """split: 'trainval' or 'test' """

        super().__init__()
        #self.data_dir = os.path.join(root_data_dir, 'cub')
        self.data_dir = root_data_dir
        self.split = split
        self.max_sequence_length = kwargs.get('max_sequence_length', 32)
        self.min_occ = kwargs.get('min_occ', 3)
        self.transform = transform
        os.makedirs(os.path.join(root_data_dir, "lang_emb"), exist_ok=True)

        self.gen_dir = os.path.join(self.data_dir, "oc_{}_msl_{}".
                                    format(self.min_occ, self.max_sequence_length))

        if split == 'train':
            self.raw_data_path = os.path.join(self.data_dir, 'text_trainvalclasses.txt')
        elif split == 'test':
            self.raw_data_path = os.path.join(self.data_dir, 'text_testclasses.txt')
        else:
            raise Exception("Only train or test split is available")

        os.makedirs(self.gen_dir, exist_ok=True)
        self.data_file = 'cub.{}.s{}'.format(split, self.max_sequence_length)
        self.vocab_file = 'cub.vocab'

        if not os.path.exists(os.path.join(self.gen_dir, self.data_file)):
            print("Data file not found for {} split at {}. Creating new... (this may take a while)".
                  format(split.upper(), os.path.join(self.gen_dir, self.data_file)))
            self._create_data()

        else:
            self._load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sent = self.data[str(idx)]['idx']
        if self.transform is not None:
            sent = self.transform(sent)
        return sent, self.data[str(idx)]['length']

    @property
    def vocab_size(self):
        return len(self.w2i)

    @property
    def pad_idx(self):
        return self.w2i['<pad>']

    @property
    def eos_idx(self):
        return self.w2i['<eos>']

    @property
    def unk_idx(self):
        return self.w2i['<unk>']

    def get_w2i(self):
        return self.w2i

    def get_i2w(self):
        return self.i2w

    def _load_data(self, vocab=True):
        with open(os.path.join(self.gen_dir, self.data_file), 'rb') as file:
            self.data = json.load(file)

        if vocab:
            self._load_vocab()

    def _load_vocab(self):
        if not os.path.exists(os.path.join(self.gen_dir, self.vocab_file)):
            self._create_vocab()
        with open(os.path.join(self.gen_dir, self.vocab_file), 'r') as vocab_file:
            vocab = json.load(vocab_file)
        self.w2i, self.i2w = vocab['w2i'], vocab['i2w']

    def _create_data(self):
        if self.split == 'train' and not os.path.exists(os.path.join(self.gen_dir, self.vocab_file)):
            self._create_vocab()
        else:
            self._load_vocab()

        with open(self.raw_data_path, 'r') as file:
            text = file.read()
            sentences = sent_tokenize(text)

        data = defaultdict(dict)
        pad_count = 0

        for i, line in enumerate(sentences):
            words = word_tokenize(line)

            tok = words[:self.max_sequence_length - 1]
            tok = tok + ['<eos>']
            length = len(tok)
            if self.max_sequence_length > length:
                tok.extend(['<pad>'] * (self.max_sequence_length - length))
                pad_count += 1
            idx = [self.w2i.get(w, self.w2i['<exc>']) for w in tok]

            id = len(data)
            data[id]['tok'] = tok
            data[id]['idx'] = idx
            data[id]['length'] = length

        print("{} out of {} sentences are truncated with max sentence length {}.".
              format(len(sentences) - pad_count, len(sentences), self.max_sequence_length))
        with io.open(os.path.join(self.gen_dir, self.data_file), 'wb') as data_file:
            data = json.dumps(data, ensure_ascii=False)
            data_file.write(data.encode('utf8', 'replace'))

        self._load_data(vocab=False)

    def _create_vocab(self):

        assert self.split == 'train', "Vocablurary can only be created for training file."

        with open(self.raw_data_path, 'r') as file:
            text = file.read()
            sentences = sent_tokenize(text)

        occ_register = OrderedCounter()
        w2i = dict()
        i2w = dict()

        special_tokens = ['<exc>', '<pad>', '<eos>']
        for st in special_tokens:
            i2w[len(w2i)] = st
            w2i[st] = len(w2i)

        texts = []
        unq_words = []

        for i, line in enumerate(sentences):
            words = word_tokenize(line)
            occ_register.update(words)
            texts.append(words)

        for w, occ in occ_register.items():
            if occ > self.min_occ and w not in special_tokens:
                i2w[len(w2i)] = w
                w2i[w] = len(w2i)
            else:
                unq_words.append(w)

        assert len(w2i) == len(i2w)

        print("Vocablurary of {} keys created, {} words are excluded (occurrence <= {})."
              .format(len(w2i), len(unq_words), self.min_occ))

        vocab = dict(w2i=w2i, i2w=i2w)
        with io.open(os.path.join(self.gen_dir, self.vocab_file), 'wb') as vocab_file:
            data = json.dumps(vocab, ensure_ascii=False)
            vocab_file.write(data.encode('utf8', 'replace'))

        with open(os.path.join(self.gen_dir, 'cub.unique'), 'wb') as unq_file:
            pickle.dump(np.array(unq_words), unq_file)

        with open(os.path.join(self.gen_dir, 'cub.all'), 'wb') as a_file:
            pickle.dump(occ_register, a_file)

        self._load_vocab()




def to_tensor(data):
   # return torch.Tensor(data).long()
    return F.one_hot(torch.Tensor(data).long(),1590).float() 

    
def getDataSet_Image(data_augment= False, resize = False):
    #transforms.Resize([64, 64])
        if resize ==True:
            tx_test = transforms.Compose([ transforms.Resize([64, 64]), 
                                      transforms.ToTensor(),])
        else:
            tx_test = transforms.Compose([# transforms.Resize([64, 64]), 
                                      transforms.ToTensor(),])
        if data_augment:
            print("data_augment")
            tx_train = transforms.Compose([transforms.Resize([64, 64]), 
                           #  transforms.TrivialAugmentWide(),
                              #             transforms.AutoAugment(),
                                          transforms.ToTensor(), 
                           transforms.GaussianBlur(kernel_size=(11,11), sigma=(0.05, 2)),
                         #   transforms.RandomCrop(64, padding=4),
                          #  transforms.ColorJitter(brightness=1, contrast=1, saturation=1),
                            transforms.RandomHorizontalFlip(),  
                                           
                                ])
        else:
            tx_train = tx_test
        from src.dataLoaders.CUB.preprocess import DalleTransformerPreprocessor

        transform_train = DalleTransformerPreprocessor(size=128, phase='train')
        transform_test = DalleTransformerPreprocessor(size=128, phase='test')

        tr= datasets.ImageFolder('data/data_cub/train', transform=tx_test)
        ts = datasets.ImageFolder('data/data_cub/test', transform=tx_test)
        return tr , ts   
    
from src.unimodal_vae.text_optimus.utils import BucketingDataLoader  
from src.unimodal_vae.text_optimus.pytorch_transformers import (WEIGHTS_NAME, AdamW, WarmupLinearSchedule,
                                  BertConfig, BertForLatentConnector, BertTokenizer,
                                  GPT2Config, GPT2ForLatentConnector, GPT2Tokenizer,
                                  OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                                  RobertaConfig, RobertaForMaskedLM, RobertaTokenizer)
def getDataSet_Sentence():

    MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2ForLatentConnector, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForLatentConnector, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer)
}
    tx = transforms.Compose([transforms.Lambda(to_tensor) ])
      #  tx = transforms.Compose([transforms.ToTensor()])
    encoder_config_class, encoder_model_class, encoder_tokenizer_class = MODEL_CLASSES["bert"]
    decoder_config_class, decoder_model_class, decoder_tokenizer_class = MODEL_CLASSES["gpt2"]

    tokenizer_encoder = encoder_tokenizer_class.from_pretrained( "bert-base-cased", do_lower_case=False)
    tokenizer_decoder = decoder_tokenizer_class.from_pretrained( "gpt2", do_lower_case=False)
    special_tokens_dict = {'pad_token': '<PAD>', 'bos_token': '<BOS>', 'eos_token': '<EOS>'}
    num_added_toks = tokenizer_decoder.add_special_tokens(special_tokens_dict)

    args = argparse.Namespace()
    args.dataset ="cub"
    args.use_philly =False
    args.block_size =False
    t_data = BucketingDataLoader('/home/******/work/mld/data/data_cub/text_trainvalclasses.txt',256,maxSentLen,[tokenizer_encoder,tokenizer_decoder],args= args, shuffle=False).dataset
    s_data = BucketingDataLoader('/home/******/work/mld/data/data_cub/text_testclasses.txt',256,maxSentLen,[tokenizer_encoder,tokenizer_decoder],args=args,shuffle=False).dataset

    return t_data, s_data
    


def getDataSet_CUB(train_dataset):
        # load base datasets
        t1, s1 = getDataSet_Image()
        t2, s2 = getDataSet_Sentence()
        
     
        train = TensorDataset([
            ResampleDataset(t1, resampler, size=len(t1) * 10),t2])
        
        test = TensorDataset([
            ResampleDataset(s1, resampler, size=len(s1) * 10),s2])
        if train_dataset :
            return train
        else:
            return test
        

    
    
def resampler(dataset, idx):
    return idx // 10





class CubDataset(Dataset):
    """
    Dataset which resamples a given dataset.

    """

    def __init__(self ,train = True):
        super(CubDataset, self).__init__()
        self.dataset = getDataSet_CUB(train_dataset= train)
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        
        if idx < 0 or idx >= len(self.dataset):
            raise IndexError('out of range')
        element = self.dataset[idx]
        image = element[0][0]
        sentence = element[1]
        sentence ["bert_token"] = sentence["bert_token"] + [None] * (32 - len(sentence ["bert_token"]))
        sentence ["gpt2_token"] = sentence["gpt2_token"] + [None] * (32 - len(sentence ["gpt2_token"]))


        return {
            "image":image,
            "sentence":sentence
        }, [element[0][1], None]
    
    
    
class CubDatasetImage(Dataset):
    """
    Dataset which resamples a given dataset.

    """

    def __init__(self ,train = True,data_augment = False,resize= False):
        super(CubDatasetImage, self).__init__()
        t1, s1 = getDataSet_Image(data_augment=data_augment,resize=resize)
        if train:
            self.dataset = t1
        else: 
            self.dataset = s1
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        
        if idx < 0 or idx >= len(self.dataset):
            raise IndexError('out of range')
        element = self.dataset[idx]
        return element[0]
        # return {
        #     "image":element[0],
           
        # }, [element[1], ]
    
    
    
class CubDatasetText(Dataset):
    """
    Dataset which resamples a given dataset.

    """

    def __init__(self ,train = True):
        super(CubDatasetText, self).__init__()
        t1, s1 = getDataSet_Sentence()
        if train:
            self.dataset = t1
        else: 
            self.dataset = s1
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        
        if idx < 0 or idx >= len(self.dataset):
            raise IndexError('out of range')
        element = self.dataset[idx]
       
        return {
            "sentence":element[0],
           
        }, [element[1], ]
    





import io
def save_tensor(x,file_path):
    torch.save(x, file_path)
    buffer = io.BytesIO()
    torch.save(x, buffer)
    
def read_tensor(file_path):
    return torch.load(file_path)

class Dataset_latent(Dataset):
    def __init__(self, folder = "/home/******/Documents/code/mld/"+"data/cub_2/"):
            self.imgs = read_tensor(folder+"image.pt")  
            self.sentences = read_tensor(folder+"sentence.pt")  
            

    def __getitem__(self, index):
        
        return {"image": self.imgs[index], "sentence":self.sentences[index]}

    def __len__(self):
        return self.imgs.size(0) 