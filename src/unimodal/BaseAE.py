import torch
import pytorch_lightning as pl
from src.logger.utils import log_cond_modalities, log_modalities
from src.eval_metrics.fd.FD import get_inception_net ,populate_metrics_step_fid , init_metric_fd
MODEL_STR = "AE"

class AE(pl.LightningModule):

    def __init__(self, modality,test_samp = None, regularization = None , alpha = 0.0 , lr =0.001, decay =0.0 ,train_samp = None):

        super(AE, self).__init__()
        self.lr = lr
        self.decay = decay
        self.modality = modality
        self.latent_dim =  self.modality.latent_dim
        self.encoder = self.modality.enc
        self.decoder = self.modality.dec
        self.regularization = regularization
        self.train_samp = train_samp
        self.alpha = alpha
        self.test_samp = test_samp
        self.save_hyperparameters(ignore= ["modality","encoder","train_samp","test_samp","decoder"])
      
    def training_step(self, x) :
  
        self.train()
        if isinstance(x,dict):
            x = x[0][self.modality.name]
        
        batch_size = x.size(0)
        recon ,z  = self.forward(x)  
        
        regularization = 0.0
        if self.regularization != None:
            if self.regularization == "l1":
                regularization = torch.abs(z).sum() 
            elif self.regularization == "l2":
                regularization = torch.square(z).sum()
                 
        recon_loss = self.reconstruction_loss(x,recon)
        total_loss= recon_loss + self.alpha * regularization
        
        self.logger.experiment.add_scalar("loss/train", total_loss/batch_size, self.global_step)
        return{"loss":total_loss / batch_size, "recon_loss": recon_loss.detach() / batch_size, "regularization": regularization / batch_size} 




    def test_step(self, x, batch_idx):
            if isinstance(x,dict):
                x = x[0][self.modality.name]
           
            batch_size = x.size(0)
            recon ,z  = self.forward(x)  
            
            regularization = 0.0
            if self.regularization != None:
                if self.regularization == "l1":
                    regularization = torch.abs(z).sum() 
                elif self.regularization == "l2":
                    regularization = torch.square(z).sum()
                    
            recon_loss = self.reconstruction_loss(x,recon)
            total_loss= recon_loss + self.alpha * regularization
            
            self.logger.experiment.add_scalar("loss/test", total_loss/batch_size, self.global_step)
            
            return{"loss":total_loss/batch_size} 


    def validation_step(self, x, batch_idx):
        if isinstance(x,dict):
            x = x[0][self.modality.name]
   
        batch_size = x.size(0)
        recon ,z  = self.forward(x)  
        
        regularization = 0.0
        if self.regularization != None:
            if self.regularization == "l1":
                regularization = torch.abs(z).sum() 
            elif self.regularization == "l2":
                regularization = torch.square(z).sum()
                 
        recon_loss = self.reconstruction_loss(x,recon)
        total_loss= recon_loss + self.alpha * regularization
        self.logger.experiment.add_scalar("loss/test", total_loss/batch_size, self.global_step)
        return{"loss":total_loss/batch_size}



    def on_train_epoch_end(self):
        
        if self.current_epoch % 20 ==0:
            self.encoder.eval()
            self.decoder.eval()
            test_batch = self.test_samp.to(self.device) 
            train_batch = self.train_samp.to(self.device)  

            print("Doing reconstruction")
            with torch.no_grad():
                recon, z = self.forward(test_batch)
                recon_train, z_train = self.forward(train_batch)

            if self.modality.name =="attributes" or  self.modality.name =="mask" :

                f_1_test =self.modality.get_f_1_score(recon,test_batch)
                f_1_train =self.modality.get_f_1_score(recon_train,train_batch)
                self.logger.experiment.add_scalar("eval/f_1_test", f_1_test, self.global_step)
                self.logger.experiment.add_scalar("eval/f_1_train", f_1_train, self.global_step)

            print("test : std  : " + str(z.std().detach()) + "  mean : " +str(z.mean().detach()) )
            print("train : std  : " + str(z_train.std().detach()) + "  mean : " +str(z_train.mean().detach()) )
            
            log_modalities(self.logger, {self.modality.name:test_batch}, [self.modality], self.current_epoch ,prefix="real_test/" ,nb_samples=8)
            log_modalities(self.logger, {self.modality.name:recon }, [self.modality], self.current_epoch ,prefix="recon_test/" ,nb_samples=8)
            
            log_modalities(self.logger, {self.modality.name:train_batch}, [self.modality], self.current_epoch ,prefix="real_train/" ,nb_samples=8)
            log_modalities(self.logger, {self.modality.name:recon_train }, [self.modality], self.current_epoch ,prefix="recon_train/" ,nb_samples=8)
      
          

    def encode(self, x):
        return self.encoder(x)


    def decode(self, z):
        return self.decoder(z)


    def forward(self, x):
        
        z = self.encode(x)
        return self.decode(z) ,z


    
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr= self.lr , betas=(0.9,0.999), weight_decay=self.decay
                                     ,amsgrad=False 
                                    )
        return optimizer
    
    def reconstruction_loss(self, x,recon):
        return  - self.modality.calc_log_prob(x,recon)

    # def reconstruction_loss(self, x,recon):
    #     return  - self.modality.log_prob(x,recon) 

    def kld_loss(self,mu,logvar, batch_size):
        return - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) /batch_size

    def elbo_objectif(self,reconstruction_loss,KLD, beta = 1.0):
        return reconstruction_loss + beta * KLD
    