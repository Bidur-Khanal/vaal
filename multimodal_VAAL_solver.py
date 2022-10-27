import torch
import torch.nn as nn
import torch.optim as optim

import os
import numpy as np
from sklearn.metrics import accuracy_score
from pathlib import Path
import sampler
import copy





class multi_modal_VAAL_Solver:
    def __init__(self, args, test_dataloader):
        self.args = args
        self.test_dataloader = test_dataloader
        self.device = torch.device('cuda:'+args.gpu_id if torch.cuda.is_available() else 'cpu')
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.dir_checkpoint = Path('./checkpoints/')
        self.sampler = sampler.AdversarySampler_multimodal(self.args.budget)


    def read_data(self, dataloader, labels=True):
        if labels:
            while True:
                for img, label, depth in dataloader:
                    yield img, depth, label
        else:
            while True:
                for img, _, depth in dataloader:
                    yield img, depth


    def train(self, current_split, querry_dataloader, val_dataloader, vae, discriminator, unlabeled_dataloader):
        self.args.train_iterations = (self.args.num_images* self.args.query_train_epochs) // self.args.batch_size
        #self.args.train_iterations = int(((self.args.budget*current_split+ self.args.initial_budget) * self.args.query_train_epochs)/self.args.batch_size)
        
        if not (os.path.exists(str(self.dir_checkpoint)+'/'+self.args.expt + '/'+ 'vae_checkpoint'+str(current_split)+'.pth') and os.path.exists(str(self.dir_checkpoint)+'/'+self.args.expt + '/'+ 'discriminator_checkpoint'+str(current_split)+'.pth')):

            labeled_data = self.read_data(querry_dataloader)
            unlabeled_data = self.read_data(unlabeled_dataloader, labels=False)

            optim_vae = optim.Adam(vae.parameters(), lr=self.args.alpha1)
            optim_discriminator = optim.Adam(discriminator.parameters(), lr=self.args.alpha2)

            vae.train()
            discriminator.train()

        
            
            for iter_count in range(self.args.train_iterations):
                
                labeled_imgs, labeled_depths, labels = next(labeled_data)
                unlabeled_imgs, unlabeled_depths = next(unlabeled_data)
            
                labeled_imgs = labeled_imgs.to(device=self.args.device, dtype=torch.float32)
                labeled_depths = labeled_depths.to(device=self.args.device, dtype=torch.float32)

                unlabeled_imgs = unlabeled_imgs.to(device=self.args.device, dtype=torch.float32)
                unlabeled_depths = unlabeled_depths.to(device=self.args.device, dtype=torch.float32)
                labels = labels.to(device=self.args.device, dtype=torch.long)


                # VAE step
                for count in range(self.args.num_vae_steps):
                    recon, depth_recon, z, mu, logvar = vae(labeled_imgs)
                
                    unsup_loss = self.vae_loss(labeled_imgs, labeled_depths, recon, depth_recon, mu, logvar, self.args.beta)
                    unlab_recon, unlab_depth_recon, unlab_z, unlab_mu, unlab_logvar = vae(unlabeled_imgs)

                    
                    transductive_loss = self.vae_loss(unlabeled_imgs, unlabeled_depths,
                            unlab_recon, unlab_depth_recon, unlab_mu, unlab_logvar, self.args.beta)
                
                    labeled_preds = discriminator(mu)
                    unlabeled_preds = discriminator(unlab_mu)
                    
                    lab_real_preds = torch.ones(labeled_imgs.size(0))
                    unlab_real_preds = torch.ones(unlabeled_imgs.size(0))
                        

                    lab_real_preds = lab_real_preds.to(device=self.args.device)
                    unlab_real_preds = unlab_real_preds.to(device=self.args.device)


                    dsc_loss = self.bce_loss(labeled_preds[:,0], lab_real_preds) + \
                            self.bce_loss(unlabeled_preds[:,0], unlab_real_preds)
                    total_vae_loss = unsup_loss + transductive_loss + self.args.adversary_param * dsc_loss
                    optim_vae.zero_grad()
                    total_vae_loss.backward()
                    optim_vae.step()

                    # sample new batch if needed to train the adversarial network
                    if count < (self.args.num_vae_steps - 1):
                        labeled_imgs, labeled_depths, _ = next(labeled_data)
                        unlabeled_imgs, unlabeled_depths = next(unlabeled_data)

                        labeled_imgs = labeled_imgs.to(device=self.args.device, dtype=torch.float32)
                        labeled_depths = labeled_depths.to(device=self.args.device, dtype=torch.float32)

                        unlabeled_imgs = unlabeled_imgs.to(device=self.args.device, dtype=torch.float32)
                        unlabeled_depths = unlabeled_depths.to(device=self.args.device, dtype=torch.float32)
                        labels = labels.to(device=self.args.device, dtype=torch.long)

                # Discriminator step
                for count in range(self.args.num_adv_steps):
                    with torch.no_grad():
                        _, _, _, mu, _ = vae(labeled_imgs)
                        _, _, _, unlab_mu, _ = vae(unlabeled_imgs)
                    
                    labeled_preds = discriminator(mu)
                    unlabeled_preds = discriminator(unlab_mu)
                    
                    lab_real_preds = torch.ones(labeled_imgs.size(0))
                    unlab_fake_preds = torch.zeros(unlabeled_imgs.size(0))

                    
                    lab_real_preds = lab_real_preds.to(device=self.args.device)
                    unlab_fake_preds = unlab_fake_preds.to(device=self.args.device)
                    
                    dsc_loss = self.bce_loss(labeled_preds[:,0], lab_real_preds) + \
                            self.bce_loss(unlabeled_preds[:,0], unlab_fake_preds)

                    optim_discriminator.zero_grad()
                    dsc_loss.backward()
                    optim_discriminator.step()

                    # sample new batch if needed to train the adversarial network
                    if count < (self.args.num_adv_steps - 1):
                        labeled_imgs, labeled_depths, _ = next(labeled_data)
                        unlabeled_imgs, unlabeled_depths = next(unlabeled_data)

                        labeled_imgs = labeled_imgs.to(device=self.args.device, dtype=torch.float32)
                        labeled_depths = labeled_depths.to(device=self.args.device, dtype=torch.float32)
                        unlabeled_imgs = unlabeled_imgs.to(device=self.args.device, dtype=torch.float32)
                        unlabeled_depths = unlabeled_depths.to(device=self.args.device, dtype=torch.float32)
                        labels = labels.to(device=self.args.device, dtype=torch.long)

                    

                if iter_count % 100 == 0:
                    print('Current vae model loss: {:.4f}'.format(total_vae_loss.item()))
                    print('Current discriminator model loss: {:.4f}'.format(dsc_loss.item()))


            Path(str(self.dir_checkpoint)+'/'+self.args.expt+'/multi_modal_VAAL').mkdir(parents=True, exist_ok=True)
            torch.save(vae.state_dict(), str(self.dir_checkpoint)+'/'+self.args.expt + '/'+ 'vae_checkpoint'+str(current_split)+'.pth')
            torch.save(discriminator.state_dict(), str(self.dir_checkpoint)+'/'+self.args.expt + '/'+ 'discriminator_checkpoint'+str(current_split)+'.pth')

        else:
            # load the checkpoint models
            discriminator.load_state_dict(torch.load(str(self.dir_checkpoint)+'/'+self.args.expt + '/'+ 'discriminator_checkpoint'+str(current_split)+'.pth'))
            vae.load_state_dict(torch.load(str(self.dir_checkpoint)+'/'+self.args.expt + '/'+ 'vae_checkpoint'+str(current_split)+'.pth'))


        return vae, discriminator


    def sample_for_labeling(self, vae, discriminator, unlabeled_dataloader, unlabeled_indices):
       
        querry_indices = self.sampler.sample(vae, 
                                             discriminator, 
                                             unlabeled_dataloader, unlabeled_indices,
                                             self.args.device)

        return querry_indices
                


    def vae_loss(self, x, depth,recon, depth_recon, mu, logvar, beta):
        MSE1 = self.mse_loss(recon, x)
        MSE2 = self.mse_loss(depth_recon,depth)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD = KLD * beta
        if self.args.adaptive_mse:
            mse_2 = self.args.mse_gamma2 * np.exp(-2 * (self.args.split_step* self.args.budget)/(self.args.num_images-self.args.num_val))
            MSE = self.args.mse_gamma1*MSE1 + mse_2*MSE2
        else:
            MSE = self.args.mse_gamma1*MSE1 + self.args.mse_gamma2*MSE2
        return  MSE + KLD

