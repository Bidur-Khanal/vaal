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
import wandb



def main(args):

    # (Initialize logging)
    experiment = wandb.init(project='U-Net-active-learning')
    

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

        scale = tuple(float(i) for i in args.scale.split(","))
        if min(scale) == 0:
            scale = None

        train_dataset =  LiverSegDataset("/home/bidur/vaal/data/liver_seg_dataset", scale = scale, flip = True, resize= args.resize)
        test_dataset =  LiverSegDataset("/home/bidur/vaal/data/liver_seg_dataset", train = False, scale = scale, flip = True, resize= args.resize)

        args.num_val = 1890
        args.num_images = 18900
        args.budget = 850
        args.initial_budget = 850
        args.num_classes = 5

    else:
        raise NotImplementedError

    # save the hyper-parameters in wandb
    experiment.config.update(vars(args))
    

    all_indices = set(np.arange(args.num_images))
    random.seed(args.seed)  #every time set the same seed
    val_indices = random.sample(all_indices, args.num_val)
    all_indices = np.setdiff1d(list(all_indices), val_indices)


    initial_indices = random.sample(list(all_indices), args.initial_budget)[0:50]
    sampler = data.sampler.SubsetRandomSampler(initial_indices)
    val_sampler = data.sampler.SubsetRandomSampler(val_indices[0:50])

    # dataset with labels available
    querry_dataloader = data.DataLoader(train_dataset, sampler=sampler, 
            batch_size=args.batch_size, drop_last=True)
    val_dataloader = data.DataLoader(train_dataset, sampler=val_sampler,
            batch_size=args.batch_size, drop_last=False)
    test_dataloader = data.DataLoader(test_dataset,
            batch_size=args.batch_size, drop_last=False)
    
 
            
    args.device = torch.device('cuda:'+args.gpu_id if torch.cuda.is_available() else 'cpu')
    solver = VAAL_Solver(args, test_dataloader)

    splits = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

    current_indices = list(initial_indices)
    for i, split in enumerate(splits):


        # need to retrain all the models on the new images
        # re initialize and retrain the models
        # task_model = vgg.vgg16_bn(num_classes=args.num_classes)
        experiment.log({
                    'split': split,
        
                })

        task_model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
        task_model.to(device=args.device)
        train_task(args, net=task_model, train_loader = querry_dataloader, val_loader = val_dataloader, test_loader= test_dataloader,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  amp=args.amp, wandb_log= experiment)


        ## all unlabeled train samples
        unlabeled_indices = np.setdiff1d(list(all_indices), current_indices)
        unlabeled_sampler = data.sampler.SubsetRandomSampler(unlabeled_indices)
        unlabeled_dataloader = data.DataLoader(train_dataset, 
                sampler=unlabeled_sampler, batch_size=args.batch_size, drop_last=False)


        if args.method == 'VAAL':
            #### initilaize the VAAL models
            vae = model.VAE(args.latent_dim)
            discriminator = model.Discriminator(args.latent_dim)

            # train the models on the current data
            vae, discriminator = solver.train(split,querry_dataloader,
                                                val_dataloader,
                                                vae, 
                                                discriminator,
                                                unlabeled_dataloader)
            sampled_indices = solver.sample_for_labeling(vae, discriminator, unlabeled_dataloader)

        elif args.method == "RandomSampling":
            random.shuffle(unlabeled_indices)
            arg = np.random.randint(len(unlabeled_indices), size=len(unlabeled_indices))
            sampled_indices = unlabeled_indices[arg][:args.budget]

        current_indices = list(current_indices) + list(sampled_indices)
        sampler = data.sampler.SubsetRandomSampler(current_indices)
        querry_dataloader = data.DataLoader(train_dataset, sampler=sampler, 
                batch_size=args.batch_size, drop_last=True)

if __name__ == '__main__':
    args = arguments.get_args()
    main(args)

