
import os
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import json
from src.utils import set_subset,set_paths_fid
from src.abstract.generative_model import GenerativeModel
from abc import ABC, abstractmethod
from src.eval.representation import train_clf_lr_all_subsets, test_clf_lr_all_subsets
from src.eval.coherence import  test_Clip , test_gen_base ,test_celebA
from src.logger.utils import log_results_train_step,log_results_eval_step,flatten_dict ,log_modalities, log_cond_modalities
from src.eval.sample_quality import compute_fad, compute_fid




class MG(pl.LightningModule, GenerativeModel,ABC):
    """ Abstract Base multimodal generative model 

    """

    def __init__(self, latent_dim, modalities_list,train_loader,test_loader, 
                 model_name, 
                 subsampling_strategy = "powerset",
                 beta = 1.0 ,
                 nb_samples = 8 ,
                 batch_size = 64 ,
                 num_train_lr = 500 ,
                 eval_epoch = 20 ,
                 do_evaluation= True ,
                 do_fd = True,
                 log_epoch = 1 ,
                 n_fd = 5000,
                 lr = 0.001 ,
                 do_class = True ,
                 nb_batchs = 50,
                 train_batch_size = 64,
                 dataset =None,
                 limit_clip = 3000,
                 ):
        super(MG, self).__init__()
        self.dataset = dataset
        self.do_class = do_class
        self.lr = lr
        self.eval_epoch = eval_epoch
        self.do_fd = do_fd
        self.log_epoch = log_epoch
        self.n_fd = n_fd
        self.train_batch_size =train_batch_size

        self.train_loader= train_loader
        self.test_loader = test_loader
        self.modelName = model_name
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.modalities_list = modalities_list
        self.num_train_lr = num_train_lr
        self.encoders = nn.ModuleList(
            [modality.enc for modality in self.modalities_list])
        self.decoders = nn.ModuleList(
            [modality.dec for modality in self.modalities_list])
        
        self.do_evaluation = do_evaluation
        self.beta = beta
        # Subsampling starategy used for training option possible (powerset/unimodal)
        self.subsampling_strategy = subsampling_strategy
        
        self.subset_list = set_subset(self.modalities_list, strategy=self.subsampling_strategy)
        
        self.limit_clip = limit_clip
        self.modelName = "None"
        self.nb_samples = nb_samples
        
        self.hparams["modalities"] = ','.join([mod.name for mod in modalities_list])
        self.hparams["modalities_weights"] = ','.join([str(mod.reconstruction_weight) for mod in modalities_list])
        self.hparams["reconstruction_dist"] = ','.join([mod.likelihood_name for mod in modalities_list])
        
        self.save_hyperparameters(ignore= ["modalities_list","train_loader","test_loader","aes"])

        self.modalities_list_dict = {mod.name : mod for mod in modalities_list}
        self.subset_list_dict = { ','.join(np.array([mod.name for mod in self.modalities_list])[s]) : s for s in self.subset_list}
        self.nb_batchs = nb_batchs
   
        self.final_results = None
        self.logdir = None
    
    

    def forward(self, x):
        # Encode x into param
        # in case of gaussian posterior -> generate mu and var
        encodings = self.encode(x)
        posterior = self.posterior(encodings)
        
        mu, logvar = posterior["joint"]
        
        z = self.reparam(mu, logvar)
        posterior["joint_z"] = z
        # Decode z and reconstruct x
        return self.decode(z), posterior


    def encode(self, x):

        encodings = {}
        for idx, modality in enumerate(self.modalities_list):
            if modality.name in x.keys():
                mod_data = x[modality.name]
                mu_, logvar_ = self.encoders[idx](mod_data)
                encodings[modality.name] = [mu_, logvar_]

        return encodings


    def decode(self, z):
        decodings = {}
        for idx, modality in enumerate(self.modalities_list):
            decodings[modality.name] = self.decoders[idx](z)
        return decodings
        #return [decoder(z) for decoder in self.decoders]


    def reparam(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)  # return z sample




    def sample(self, N):
        self.eval()
        with torch.no_grad():
            z = torch.randn(N, self.latent_dim, device=self.device)
            output = self.decode(z)
            return output

    
    
    
    
    def Kl_div_gaussian(self, mu,log_var):  
        return - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) 

    
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # geting data
        x = batch[0]
        # x.shape = (nb_modalities,batch_size, modality_input_size)
        results = self.compute_loss(x) 
        #self.log({"results": results})
        log_results_train_step(self.logger,results,self.global_step )
        
        return results


    def on_train_epoch_end(self):

        if self.current_epoch % self.log_epoch ==0 :
            self.do_sampling_and_cond_gen()
        elif self.current_epoch % (self.log_epoch * 2) ==0:
            self.do_sampling_and_cond_gen()
        if (self.current_epoch % self.eval_epoch )== 0 and self.do_evaluation and self.current_epoch != (self.trainer.max_epochs -1)  and  self.current_epoch !=0 :
            eval_res = self.evaluation()
            
            with open(os.path.join(self.logger.log_dir,"results{}.json".format(self.current_epoch)),"w") as f:
                json.dump(eval_res,f)
            log_results_eval_step(self.logger,eval_res,self.global_step )
        elif ( self.current_epoch == (self.trainer.max_epochs -1)   ):# and self.current_epoch !=0:
            self.final_eval()
          
        
     

   
    

      
    
    
    def do_sampling_and_cond_gen(self,step_log= None):
        self.eval()
        print("Doing Sampling")
        if step_log == None:
            step_log = self.current_epoch
        output = self.sample(self.nb_samples)
        log_modalities(self.logger, output, self.modalities_list,step_log,nb_samples=self.nb_samples)
        test_batch = next(iter(self.test_loader))
        if len(test_batch) ==2:
            test_batch = test_batch[0]
        for k, m_key in enumerate(test_batch.keys()):
            test_batch[m_key] = test_batch[m_key][:8].to(self.device) 
        print("Doing Cross gen")
        output_cond = self.conditional_gen_all_subsets(test_batch,N=8)
       # log_modalities(self.logger, test_batch, self.modalities_list, step_log,prefix="real/" ,nb_samples=self.nb_samples)
        
        log_cond_modalities(self.logger, output_cond, self.modalities_list, step_log,nb_samples=self.nb_samples)
        
    
    
    def final_eval(self):
        #self.n_fd = 5000
        #self.limit_clip = 5000
        self.do_fd = True
        self.nb_batchs = 6
        self.n_fd = 3000
       # self.n_fd = 5000
        # self.nb_batchs = 1
        print("Running Final evaluation with all the data and 5000 samples for FID")
        results = self.evaluation()
        log_results_eval_step(self.logger , results,self.global_step )
        
        self.final_results = results
         
        with open(os.path.join(self.logger.log_dir,"results_epoch_{}_final.json".format(self.current_epoch)),"w") as f:
            json.dump(results,f)
        
    def evaluation(self):
        print("Evaluation using {} batchs".format(self.nb_batchs))
        if self.logger != None:
            self.logdir = self.logger.log_dir
            self.paths_fid = set_paths_fid(folder =self.logdir ,subsets=self.subset_list_dict)
        else:
            self.paths_fid = set_paths_fid(folder ="trained_models/temp",subsets=self.subset_list_dict)
        if self.do_class:
            print("Class eval")
            clf_lr =  train_clf_lr_all_subsets(model = self,subsets_dict = self.subset_list_dict , 
                                            d_loader = self.train_loader ,
                                            batch_size =self.train_batch_size, 
                                            class_dim =self.latent_dim, 
                                            device=self.device,
                                            num_training_samples_lr = self.num_train_lr
                                            )
            print("Done training classifiers")
            lr_eval = test_clf_lr_all_subsets( clf_lr, model = self, subsets= self.subset_list_dict ,
                                                d_loader = self.test_loader, 
                                                batch_size =self.train_batch_size ,
                                                device = self.device, 
                                                nb_batchs = self.nb_batchs )
            print("Done testing classification")
            
        print("Coherence eval")
        if self.dataset =="CUB":
            cohrence = test_Clip(
                                        model = self,
                                        modalities_list= self.modalities_list,
                                        d_loader=self.test_loader,
                                        batch_size = self.train_batch_size ,
                                        num_samples_fid = self.n_fd,
                                        device=self.device,
                                        do_fd = self.do_fd,
                                        limit_clip= self.limit_clip,
                                        path_fid = self.paths_fid,
                                        nb_batchs = self.nb_batchs
            ) 
            res = {"Coherence": cohrence }


        elif self.dataset =="celebA":
            cohrence, fid = test_celebA(
                                        model = self, subset_list = self.subset_list,
                                        modalities_list= self.modalities_list,
                                        d_loader=self.test_loader,
                                        batch_size = self.train_batch_size ,
                                        num_samples_fid = self.n_fd,
                                        device=self.device,
                                        do_fd = self.do_fd,
                                        path_fid = self.paths_fid,
                                        nb_batchs = self.nb_batchs
            ) 
            res = {"Coherence": cohrence, "fid":fid }
        else:
            cohrence,fid = test_gen_base(model = self, subset_list = self.subset_list,
                                        modalities_list= self.modalities_list,
                                        d_loader=self.test_loader,
                                        batch_size = self.train_batch_size ,
                                        num_samples_fid = self.n_fd,
                                        device=self.device,
                                        do_fd = self.do_fd,
                                        path_fid = self.paths_fid,
                                        nb_batchs = self.nb_batchs)
            coh=flatten_dict( cohrence )
            coh ["random"] =  cohrence["random"]
        
            res = {
            "Coherence": coh,
            "fid": fid}
            if self.do_class:
                res["Accuracy"]= {"latentclass": lr_eval } 
            if self.do_fd:
               # fids = compute_fid(path_list=self.paths_fid,modalities_dict=self.modalities_list_dict,subset_dict=self.subset_list_dict,device =self.device)
                fads = compute_fad(path_list=self.paths_fid,modalities_dict=self.modalities_list_dict,subset_dict=self.subset_list_dict,device =self.device)
               # res["FID"] = fids
                res["FAD"] = fads
        print(str(res))
        return res





    def test_step(self, batch, batch_idx):
        # this is the test loop
        x = batch[0]
        # x.shape = (nb_modalities,batch_size, input_size)
        results = self.compute_loss(x)
        log_results_train_step(self.logger, results, self.global_step, prefix="test/" )
        
        return results


    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x = batch[0]
        # x.shape = (nb_modalities,batch_size, input_size)
        results = self.compute_loss(x)
        log_results_train_step(self.logger, results, self.global_step, prefix="test/" )
        
        return results

  
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr , betas=(0.9,0.999) )
        return optimizer

    
    
    
   
    @abstractmethod
    def compute_loss(self, x):

        pass
    
    
  
    def compute_reconstruction_error(self, x, reconstruction):
        pass
    

    def compute_KLD(self,posterior):
        pass
    
    
  
    def elbo_objectif(self, reconstruction_error, posterior, beta):
        pass

           
    
    def conditional_gen_all_subsets(self, x,N=None): raise NotImplementedError
    
    
    
    def gen_latent(self,x): raise NotImplementedError
    
    
    
    def conditional_gen_latent_subsets(self, x): raise NotImplementedError




