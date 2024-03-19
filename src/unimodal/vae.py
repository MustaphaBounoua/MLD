import torch
import pytorch_lightning as pl
from src.logger.utils import log_cond_modalities, log_modalities
MODEL_STR = "AE"

class AE(pl.LightningModule):

    def __init__(self, modality,test_loader = None, regularization = None , alpha = 1.0 , lr =0.001, decay =0.0 ,train_loader = None):

        super(AE, self).__init__()
        self.lr = lr
        self.decay = decay
        self.modality = modality
        self.latent_dim =  self.modality.latent_dim
        self.encoder = self.modality.enc
        self.decoder = self.modality.dec
        self.regularization = regularization
        self.test_loader = test_loader
        self.alpha = 5
        self.train_loader = train_loader
        self.save_hyperparameters(ignore= ["modality","encoder","test_loader","decoder"])

    def training_step(self, x) :
        
        self.train()
        #print(x.shape)
        #x = x[0][self.modality.name]

        batch_size = x.size(0)
        recon ,mu,logvar  = self.forward(x)  

        recon_loss = self.reconstruction_loss(x,recon) /x.size(0) 
        kld = self.kld_loss(mu,logvar,x.size(0))
        if self.global_step >= 300:
            b = self.alpha
        else:
            b = self.alpha * self.global_step / 300
        total_loss = recon_loss + self.alpha * kld
        
        self.logger.experiment.add_scalar("loss/train", total_loss, self.global_step)
        self.logger.experiment.add_scalar("kld/train", kld, self.global_step)

        return{"loss":total_loss } 




    def test_step(self, x, batch_idx):
        
        batch_size = x.size(0)
        recon ,mu,logvar  = self.forward(x)  
        recon_loss = self.reconstruction_loss(x,recon) 
        
        kld = self.kld_loss(mu,logvar,x.size(0))
        total_loss = recon_loss + self.alpha * kld

        self.logger.experiment.add_scalar("loss/test", total_loss/batch_size, self.global_step)
        
        return{"loss":total_loss/batch_size} 


    def validation_step(self, x, batch_idx):
      
        batch_size = x.size(0)
        recon ,mu,logvar  = self.forward(x)  
        recon_loss = self.reconstruction_loss(x,recon) 
        
        kld = self.kld_loss(mu,logvar,x.size(0))
        total_loss = recon_loss + self.alpha * kld

        self.logger.experiment.add_scalar("loss/test", total_loss/batch_size, self.global_step)
        
        return{"loss":total_loss/batch_size} 



    def on_train_epoch_end(self):
        
        if self.current_epoch % 10 ==0:
            self.encoder.eval()
            self.decoder.eval()
            test_batch = next(iter(self.test_loader)).to(self.device) 
            train_batch = next(iter(self.train_loader)).to(self.device) 
            print("Doing reconstruction")
            with torch.no_grad():
                recon, mu,sig = self.forward(test_batch)
                recon_train, mu,sig = self.forward(train_batch)
                out_sampling = self.decode(torch.randn_like(sig))
            

            log_modalities(self.logger, {self.modality.name:test_batch}, [self.modality], self.current_epoch ,prefix="real_test/" ,nb_samples=8)
            log_modalities(self.logger, {self.modality.name:recon }, [self.modality], self.current_epoch ,prefix="recon_test/" ,nb_samples=8)
            
            log_modalities(self.logger, {self.modality.name:train_batch}, [self.modality], self.current_epoch ,prefix="real_train/" ,nb_samples=8)
            log_modalities(self.logger, {self.modality.name:recon_train }, [self.modality], self.current_epoch ,prefix="recon_train/" ,nb_samples=8)
            
            log_modalities(self.logger, {self.modality.name:out_sampling}, [self.modality], self.current_epoch ,prefix="sampling/" ,nb_samples=8)
          
    def encode(self, x):
        return self.encoder(x)


    def decode(self, z):
        return self.decoder(z)


    def forward(self, x):
        ## Encode x into param 
        # in case of gaussian posterior -> generate mu and var
        mu, logvar = self.encode(x)
      
        z = self.reparam(mu, logvar)
        ## Decode z and reconstruct x       
        return self.decode(z), mu, logvar


    def reparam(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr= self.lr , betas=(0.9,0.999), weight_decay=self.decay
                                     ,amsgrad=True 
                                    )
        return optimizer
    
    def reconstruction_loss(self, x,recon):
        return  - self.modality.calc_log_prob(x,recon)


    def kld_loss(self,mu,logvar, batch_size):
        return - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) /batch_size

    def elbo_objectif(self,reconstruction_loss,KLD, beta = 1.0):
        return reconstruction_loss + beta * KLD
    