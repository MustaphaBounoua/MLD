
import sys
import random

import torch
import torch.utils.data as data
from torchvision import datasets, transforms 
from torch.utils.data import random_split, DataLoader
import warnings
from PIL import Image
import os
import os.path
import gzip
import numpy as np
import torch
import codecs
import json
import pytorch_lightning as pl


digit_text_english = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

DATA_FOLDER = "data/data_mnistsvhntext/"
#DATA_FOLDER ="/home/bounoua/work/mld/data/data_mnistsvhntext/"
data_multiplications = 20


alphabet_file = DATA_FOLDER+"alphabet.json"



class VisionDataset(data.Dataset):
    _repr_indent = 4

    def __init__(self, root):
        # if isinstance(root, torch._six.string_classes):
        #     root = os.path.expanduser(root)
        self.root = DATA_FOLDER

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __repr__(self):
        head = "Dataset " + self.dataset
        body = ["Number of datapoints: {}".format(self.__len__())]
        if self.root is not None:
            body.append("Root location: {}".format(self.root))
        body += self.extra_repr().splitlines()
        if hasattr(self, 'transform') and self.transform is not None:
            body += self._format_transform_repr(self.transform,
                                                "Transforms: ")
        if hasattr(self, 'target_transform') and self.target_transform is not None:
            body += self._format_transform_repr(self.target_transform,
                                                "Target transforms: ")
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)

    def _format_transform_repr(self, transform, head):
        lines = transform.__repr__().splitlines()
        return (["{}{}".format(head, lines[0])] +
                ["{}{}".format(" " * len(head), line) for line in lines[1:]])

    def extra_repr(self):
        return ""



class SVHNMNIST(VisionDataset):
    
    training_file_svhn_idx = 'train-ms-svhn-idx.pt';
    training_file_mnist_idx = 'train-ms-mnist-idx.pt';
    training_file_mnist = 'training.pt';
    training_file_svhn = 'train_32x32.mat';
    test_file_svhn_idx = 'test-ms-mnist-idx.pt';
    test_file_mnist_idx = 'test-ms-svhn-idx.pt';
    test_file_mnist = 'test.pt';
    test_file_svhn = 'test_32x32.mat';
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    @property
    def train_labels_mnist(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets_mnist

    @property
    def train_labels_svhn(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets_svhn

    @property
    def test_labels_mnist(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets_mnist

    @property
    def test_labels_svhn(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets_svhn

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data_mnist

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data_mnist

    def __init__(self, train=True, transform=None, target_transform=None, with_text = True):
        super(SVHNMNIST, self).__init__(DATA_FOLDER)
        self.dataset = 'MNIST_SVHN';
        self.dataset_mnist = 'MNIST';
        self.dataset_svhn = 'SVHN';
        self.len_sequence = 8
        self.transform = transform
        self.target_transform = target_transform
        self.with_text =with_text
        self.train = train  # training set or test set
        with open(alphabet_file) as al_file:
            alphabet = str(''.join(json.load(al_file)))
        self.alphabet =  alphabet

        self.dir_svhn = os.path.join(self.root, self.dataset_svhn)
        

        if not self._check_exists_mnist():
            raise RuntimeError('Dataset MNIST not found.')
        if not self._check_exists_svhn():
            raise RuntimeError('Dataset SVHN not found.')


        if self.train:
            id_file_svhn = self.training_file_svhn_idx;
            id_file_mnist = self.training_file_mnist_idx;
            data_file_svhn = self.training_file_svhn;
            data_file_mnist = self.training_file_mnist;
        else:
            id_file_svhn = self.test_file_svhn_idx;
            id_file_mnist = self.test_file_mnist_idx;
            data_file_svhn = self.test_file_svhn;
            data_file_mnist = self.test_file_mnist;

        # import here rather than at top of file because this is
        # an optional dependency for torchvision
        import scipy.io as sio

        
        # reading(loading) mat file as array
        loaded_mat = sio.loadmat(os.path.join(self.dir_svhn, data_file_svhn))

        self.data_svhn = loaded_mat['X']
        # loading from the .mat file gives an np array of type np.uint8
        # converting to np.int64, so that we have a LongTensor after
        # the conversion from the numpy array
        # the squeeze is needed to obtain a 1D tensor
        self.labels_svhn = loaded_mat['y'].astype(np.int64).squeeze()

        # the svhn dataset assigns the class label "10" to the digit 0
        # this makes it inconsistent with several loss functions
        # which expect the class labels to be in the range [0, C-1]
        np.place(self.labels_svhn, self.labels_svhn == 10, 0)
        self.data_svhn = np.transpose(self.data_svhn, (3, 2, 0, 1))
        samples_svhn = self.data_svhn.shape[0];
        channels_svhn = self.data_svhn.shape[1];
        width_svhn = self.data_svhn.shape[2];
        height_svhn = self.data_svhn.shape[3];

        self.data_mnist, self.labels_mnist = torch.load(os.path.join(self.processed_folder, data_file_mnist));

        # # get transformed indices
        self.labels_svhn = torch.LongTensor(self.labels_svhn);
        mnist_l, mnist_li = self.labels_mnist.sort()
        svhn_l, svhn_li = self.labels_svhn.sort()
        self.mnist_idx, self.svhn_idx = rand_match_on_idx(mnist_l, mnist_li,
                                                          svhn_l, svhn_li,
                                                          max_d=10000,
                                                          dm=data_multiplications)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        idx_mnist = self.mnist_idx[index];
        idx_svhn = self.svhn_idx[index];
        img_svhn, target_svhn = self.data_svhn[idx_svhn], int(self.labels_svhn[idx_svhn]);
        img_mnist, target_mnist = self.data_mnist[idx_mnist], int(self.labels_mnist[idx_mnist]);

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img_svhn = Image.fromarray(np.transpose(img_svhn, (1, 2, 0)))
        img_mnist = Image.fromarray(img_mnist.numpy(), mode='L')

        if self.transform is not None:
            if self.transform[0] is not None:
                img_mnist = self.transform[0](img_mnist);
                img_svhn = self.transform[1](img_svhn);

        if target_mnist == target_svhn:
            text_target = create_text_from_label_mnist(self.len_sequence, target_mnist, self.alphabet)
        else:
            print(target_svhn)
            print(target_mnist)
            print('targets do not match...exit')
            sys.exit();

        
    
        if self.target_transform is not None:
            target = self.target_transform(target_mnist)
        else:
            target = target_mnist;
        if self.with_text:
            batch = {'mnist': img_mnist, 'svhn': img_svhn, 'label': text_target}
            return batch, target
        else:
            batch = {'mnist': img_mnist, 'svhn': img_svhn}
            return batch, target

    def __len__(self):
        return len(self.mnist_idx)

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.dataset, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.dataset_mnist, 'processed')

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists_mnist(self):
   
        return (os.path.exists(os.path.join(self.processed_folder,
                                            self.training_file_mnist)) and
                os.path.exists(os.path.join(self.processed_folder,
                                            self.test_file_mnist)))

    def _check_exists_svhn(self):
        return (os.path.exists(os.path.join(self.dir_svhn,
                                            self.training_file_svhn)) and
                os.path.exists(os.path.join(self.dir_svhn,
                                            self.test_file_svhn)))

    @staticmethod
    def extract_gzip(gzip_path, remove_finished=False):
        print('Extracting {}'.format(gzip_path))
        with open(gzip_path.replace('.gz', ''), 'wb') as out_f, \
                gzip.GzipFile(gzip_path) as zip_f:
            out_f.write(zip_f.read())
        if remove_finished:
            os.unlink(gzip_path)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")



def rand_match_on_idx(l1, idx1, l2, idx2, max_d=10000, dm=10):
    """
    l*: sorted labels
    idx*: indices of sorted labels in original list
    """
    _idx1, _idx2 = [], []
    for l in l1.unique():  # assuming both have same idxs
        l_idx1, l_idx2 = idx1[l1 == l], idx2[l2 == l]
        n = min(l_idx1.size(0), l_idx2.size(0), max_d)
        l_idx1, l_idx2 = l_idx1[:n], l_idx2[:n]
        for _ in range(dm):
            _idx1.append(l_idx1[torch.randperm(n)])
            _idx2.append(l_idx2[torch.randperm(n)])
    return torch.cat(_idx1), torch.cat(_idx2)


def get_transform_mnist():
        transform_mnist = transforms.Compose([transforms.ToTensor(),
                                              transforms.ToPILImage(),
                                              transforms.Resize(size=(28, 28), interpolation=transforms.InterpolationMode.BICUBIC),
                                              transforms.ToTensor()])
        return transform_mnist


def get_transform_svhn():
        transform_svhn = transforms.Compose([transforms.ToTensor()])
        return transform_svhn


def get_data_set_svhn_mnist(with_text = False):
        transform_mnist = get_transform_mnist()
        transform_svhn = get_transform_svhn()
        
        transforms = [ transform_mnist, transform_svhn ]
        
        train = SVHNMNIST(with_text = with_text,
                          train=True,
                          transform=transforms)
        test = SVHNMNIST(with_text = with_text,
                         train=False,
                         transform=transforms)
        return  train, test

        
class DataModuleSVHNMNIST(pl.LightningDataModule):
    def __init__(self, batch_size,with_text = False,shuffle=True,num_workers=12, drop_last=True):
        super().__init__()
          
          
        # Defining batch size of our data
        self.batch_size = batch_size
        self.with_text = with_text
        self.suffle = shuffle
        self.num_worker = num_workers
        self.drop_last = drop_last

        # Defining transforms to be applied on the data
        self.transforms = [ get_transform_mnist(), get_transform_svhn() ]

 
        
        
  
    def prepare_data(self):
    
        pass
  
    def setup(self, stage=None):
        
          # Loading our data after applying the transforms
        self.train_data = SVHNMNIST(with_text = self.with_text,
                          train=True,
                          transform=self.transforms)
        self.test_data = SVHNMNIST(with_text = self.with_text,
                         train=False,
                         transform=self.transforms)
        self.val_data = self.test_data
  
  
  
    def train_dataloader(self):
        
          # Generating train_dataloader
        return DataLoader(self.train_data, 
                          batch_size = self.batch_size,shuffle=self.suffle,num_workers=self.num_worker, drop_last=self.drop_last)
  
    def val_dataloader(self):
        
          # Generating val_dataloader
        return DataLoader(self.val_data,
                          batch_size = self.batch_size, shuffle=False,num_workers=self.num_worker, drop_last=self.drop_last)
  
    def test_dataloader(self):
        
        # Generating test_dataloader
        return DataLoader(self.test_data,
                          batch_size =  self.batch_size, shuffle=False,num_workers=self.num_worker, drop_last=self.drop_last)

















def char2Index(alphabet, character):
    return alphabet.find(character)


def one_hot_encode(len_seq, alphabet, seq):
    X = torch.zeros(len_seq, len(alphabet))
    if len(seq) > len_seq:
        seq = seq[:len_seq];
    for index_char, char in enumerate(seq):
        if char2Index(alphabet, char) != -1:
            X[index_char, char2Index(alphabet, char)] = 1.0
    return X


def create_text_from_label_mnist(len_seq, label, alphabet):
    text = digit_text_english[label];
    sequence = len_seq * [' '];
    start_index = random.randint(0, len_seq - 1 - len(text));
    sequence[start_index:start_index + len(text)] = text;
    sequence_one_hot = one_hot_encode(len_seq, alphabet, sequence);
    return sequence_one_hot


def seq2text(alphabet, seq):
    decoded = []
    for j in range(len(seq)):
        decoded.append(alphabet[seq[j]])
    return decoded


def tensor_to_text(alphabet, gen_t):
    gen_t = gen_t.cpu().data.numpy()
    gen_t = np.argmax(gen_t, axis=-1)
    decoded_samples = []
    for i in range(len(gen_t)):
        decoded = seq2text(alphabet, gen_t[i])
        decoded_samples.append(decoded)
    return decoded_samples;

