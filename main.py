import torch
from torchvision import datasets, transforms
import torch.utils.data.sampler  as sampler
import torch.utils.data as data

import numpy as np
import argparse
import random
import os

from custom_datasets import *
import model
import vgg
from VAAL_solver import VAAL_Solver
from utils import *
import arguments
from evaluate import evaluate
from unet import UNet
from task_solver import train_task



def main(args):
    # if args.dataset == 'cifar10':
    #     test_dataloader = data.DataLoader(
    #             datasets.CIFAR10(args.data_path, download=True, transform=cifar_transformer(), train=False),
    #         batch_size=args.batch_size, drop_last=False)

    #     train_dataset = CIFAR10(args.data_path)

    #     args.num_images = 50000
    #     args.num_val = 5000
    #     args.budget = 2500
    #     args.initial_budget = 5000
    #     args.num_classes = 10

    if args.dataset == 'liver-seg':
        test_dataloader = data.DataLoader(
                datasets.ImageFolder(args.data_path, transform=imagenet_transformer()),
            drop_last=False, batch_size=args.batch_size)

        train_dataset =  LiverSegDataset("/home/bidur/vaal/data/liver_seg_dataset", scale = None, flip = True, resize= args.resize)
        test_dataset =  LiverSegDataset("/home/bidur/vaal/data/liver_seg_dataset", train = False, scale = None, flip = True, resize= args.resize)

        args.num_val = 1890
        args.num_images = 18900
        args.budget = 800
        args.initial_budget = 800
        args.num_classes = 5

    else:
        raise NotImplementedError

    all_indices = set(np.arange(args.num_images))
    random.seed(args.seed)  #every time set the same seed
    val_indices = random.sample(all_indices, args.num_val)
    all_indices = np.setdiff1d(list(all_indices), val_indices)


    initial_indices = random.sample(list(all_indices), args.initial_budget)
    sampler = data.sampler.SubsetRandomSampler(initial_indices)
    val_sampler = data.sampler.SubsetRandomSampler(val_indices)

    # dataset with labels available
    querry_dataloader = data.DataLoader(train_dataset, sampler=sampler, 
            batch_size=args.batch_size, drop_last=True)
    val_dataloader = data.DataLoader(train_dataset, sampler=val_sampler,
            batch_size=args.batch_size, drop_last=False)
    

            
    args.device = torch.device('cuda:'+args.gpu_id if torch.cuda.is_available() else 'cpu')
    solver = VAAL_Solver(args, test_dataloader)

    splits = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

    current_indices = list(initial_indices)

    accuracies = []
    
    for split in splits:


        # need to retrain all the models on the new images
        # re initialize and retrain the models
        # task_model = vgg.vgg16_bn(num_classes=args.num_classes)


        task_model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
        task_model.to(device=args.device)
        train_task(args, net=task_model, train_loader = querry_dataloader, val_lodaer = val_dataloader,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  amp=args.amp)


        ## all unlabeled train samples
        unlabeled_indices = np.setdiff1d(list(all_indices), current_indices)
        unlabeled_sampler = data.sampler.SubsetRandomSampler(unlabeled_indices)
        unlabeled_dataloader = data.DataLoader(train_dataset, 
                sampler=unlabeled_sampler, batch_size=args.batch_size, drop_last=False)


        #### initilaize the VAAL models
        vae = model.VAE(args.latent_dim)
        discriminator = model.Discriminator(args.latent_dim)

        # train the models on the current data
        vae, discriminator = solver.train(querry_dataloader,
                                               val_dataloader,
                                               vae, 
                                               discriminator,
                                               unlabeled_dataloader)

        sampled_indices = solver.sample_for_labeling(vae, discriminator, unlabeled_dataloader)
        current_indices = list(current_indices) + list(sampled_indices)
        sampler = data.sampler.SubsetRandomSampler(current_indices)
        querry_dataloader = data.DataLoader(train_dataset, sampler=sampler, 
                batch_size=args.batch_size, drop_last=True)

    torch.save(accuracies, os.path.join(args.out_path, args.log_name))

if __name__ == '__main__':
    args = arguments.get_args()
    main(args)

