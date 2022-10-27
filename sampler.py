import torch

import numpy as np

class AdversarySampler:
    def __init__(self, budget):
        self.budget = budget


    def sample(self, vae, discriminator, data, unlabeled_indices, device):
        all_preds = []

        for images, _,_  in data:
            images = images.to(device)

            with torch.no_grad():
                _, _, mu, _ = vae(images)
                preds = discriminator(mu)

            preds = preds.cpu().data
            all_preds.extend(preds)
          

        all_preds = torch.stack(all_preds)
        all_preds = all_preds.view(-1)
        # need to multiply by -1 to be able to use torch.topk 
        all_preds *= -1

        # select the points which the discriminator thinks are the most likely to be unlabeled
        _, querry_indices = torch.topk(all_preds, int(self.budget))
        querry_pool_indices = list(np.asarray(unlabeled_indices)[querry_indices])

        return querry_pool_indices


class AdversarySampler_multimodal:
    def __init__(self, budget):
        self.budget = budget


    def sample(self, vae, discriminator, data, unlabeled_indices, device):
        all_preds = []

        for images, _,_  in data:
            images = images.to(device)

            with torch.no_grad():
                _, _, _, mu, _ = vae(images)
                preds = discriminator(mu)

            preds = preds.cpu().data
            all_preds.extend(preds)
          

        all_preds = torch.stack(all_preds)
        all_preds = all_preds.view(-1)
        # need to multiply by -1 to be able to use torch.topk 
        all_preds *= -1

        # select the points which the discriminator thinks are the most likely to be unlabeled
        _, querry_indices = torch.topk(all_preds, int(self.budget))
        querry_pool_indices = list(np.asarray(unlabeled_indices)[querry_indices])

        return querry_pool_indices


class AdversarySampler_multimodal2:
    def __init__(self, budget):
        self.budget = budget


    def sample(self, vae_image, discriminator_image, vae_depth, discriminator_depth, data, unlabeled_indices, device):
        all_preds_images = []
        all_preds_depths = []

        for images, _,_  in data:
            images = images.to(device)

            with torch.no_grad():
                _, _, mu_image, _ = vae_image(images)
                preds_image = discriminator_image(mu_image)

                _, _, mu_depth, _ = vae_depth(images)
                preds_depth = discriminator_depth(mu_depth)


            preds_image = preds_image.cpu().data
            all_preds_images.extend(preds_image)

            preds_depth = preds_depth.cpu().data
            all_preds_depths.extend(preds_depth)
          

        all_preds_images = torch.stack(all_preds_images)
        all_preds_images = all_preds_images.view(-1)
        
        all_preds_depths = torch.stack(all_preds_depths)
        all_preds_depths = all_preds_depths.view(-1)
        
        all_preds = torch.add(all_preds_images, all_preds_depths)
        # need to multiply by -1 to be able to use torch.topk 
        all_preds *= -1
        
        # select the points which the discriminator thinks are the most likely to be unlabeled
        _, querry_indices = torch.topk(all_preds, int(self.budget))
        querry_pool_indices = list(np.asarray(unlabeled_indices)[querry_indices])

        return querry_pool_indices

class AdversarySampler_depth:
    def __init__(self, budget):
        self.budget = budget


    def sample(self, vae_depth, discriminator_depth, data, unlabeled_indices, device):
        all_preds_depths = []

        for images, _,_  in data:
            images = images.to(device)

            with torch.no_grad():
                
                _, _, mu_depth, _ = vae_depth(images)
                preds_depth = discriminator_depth(mu_depth)

            preds_depth = preds_depth.cpu().data
            all_preds_depths.extend(preds_depth)
          
        all_preds_depths = torch.stack(all_preds_depths)
        all_preds_depths = all_preds_depths.view(-1)
        # need to multiply by -1 to be able to use torch.topk 
        all_preds_depths *= -1

        
        # select the points which the discriminator thinks are the most likely to be unlabeled
        _, querry_indices = torch.topk(all_preds_depths, int(self.budget))
        querry_pool_indices = list(np.asarray(unlabeled_indices)[querry_indices])

        return querry_pool_indices
        
