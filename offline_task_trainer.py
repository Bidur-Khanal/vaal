import os
from multimodal_VAAL_solver import multi_modal_VAAL_Solver
import torch
from torchvision import datasets, transforms
import torchvision.models as torch_models
import torch.utils.data.sampler  as sampler
import torch.utils.data as data
import sampler as query_Sampler
import numpy as np
import argparse
import random
from pathlib import Path
from custom_datasets import *
import model
import multi_modal_model
import vgg
from VAAL_solver import VAAL_Solver
from utils import *
import arguments
from evaluate import evaluate
from unet import UNet
from task_solver import train_task

from multi_label_classification_task_solver import train_multilabel_classifier
import wandb
import torch.backends.cudnn as cudnn


## Set Seed
def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    cudnn.deterministic = True



def main(args):

    # (Initialize logging)
    # experiment = wandb.init(project='U-Net-active-learning-final-RC-2-classes')
    experiment = wandb.init(project='U-Net-active-learning-final-RC-nogallbladder-no-less-than-3-classes')


    if args.dataset == 'liver-seg':
        
        scale = tuple(float(i) for i in args.scale.split(","))
        if min(scale) == 0:
            scale = None

        train_dataset =  LiverSegDataset("/home/bidur/vaal/data/liver_seg_dataset", scale = scale, flip = False, resize= args.resize,train_pth_file = 'train_files.npy', test_pth_file = 'test_files.npy')
        test_dataset =  LiverSegDataset("/home/bidur/vaal/data/liver_seg_dataset", train = False, scale = scale, flip = False, resize= args.resize,train_pth_file = 'train_files.npy', test_pth_file = 'test_files.npy')
        
        args.num_val = 1890
        args.num_images = 18900
        args.budget = 850
        args.initial_budget = 850
        #args.budget = 500
        #args.initial_budget = 200
        args.num_classes = 5


    elif args.dataset == 'liver-seg-gallbladder-removed':
        
        scale = tuple(float(i) for i in args.scale.split(","))
        if min(scale) == 0:
            scale = None

        train_dataset =  LiverSegDataset_Gallbladder_Removed("/home/bidur/vaal/data/liver_seg_dataset", scale = scale, flip = False, resize= args.resize,train_pth_file = 'train_files_filtered_gallbladder.npy', test_pth_file = 'test_files_filtered_gallbladder.npy')
        test_dataset =  LiverSegDataset_Gallbladder_Removed("/home/bidur/vaal/data/liver_seg_dataset", train = False, scale = scale, flip = False, resize= args.resize,train_pth_file = 'train_files_filtered_gallbladder.npy', test_pth_file = 'test_files_filtered_gallbladder.npy')
        
        
        args.num_val = 1819
        args.num_images = 18191
        args.budget = 818
        #args.budget = 818*2
        args.initial_budget = 818
        #args.budget = 500
        #args.initial_budget = 200
        args.num_classes = 4


    elif args.dataset == 'liver-seg-gallbladder-removed-class-no-less-than-3':
       
        scale = tuple(float(i) for i in args.scale.split(","))
        if min(scale) == 0:
            scale = None

        train_dataset =  LiverSegDataset_Gallbladder_Removed("/home/bidur/vaal/data/liver_seg_dataset", scale = scale, flip = False, resize= args.resize,train_pth_file = 'train_files_filtered_gallbladder_no_less_than_3_classes.npy', test_pth_file = 'test_files_filtered_gallbladder_no_less_than_3_classes.npy')
        test_dataset =  LiverSegDataset_Gallbladder_Removed("/home/bidur/vaal/data/liver_seg_dataset", train = False, scale = scale, flip = False, resize= args.resize,train_pth_file = 'train_files_filtered_gallbladder_no_less_than_3_classes.npy', test_pth_file = 'test_files_filtered_gallbladder_no_less_than_3_classes.npy')
        
        
        args.num_val = 1548
        args.num_images = 15482
        #args.budget = 774
        #args.budget = 200
        args.budget = 500
        #args.initial_budget = 774
        args.initial_budget = 500

        args.num_classes = 4

    
    elif args.dataset == 'liver-seg-gallbladder-2-classes':
       
        scale = tuple(float(i) for i in args.scale.split(","))
        if min(scale) == 0:
            scale = None

        train_dataset =  LiverSegDataset_2_classes("/home/bidur/vaal/data/liver_seg_dataset", scale = scale, flip = False, resize= args.resize,train_pth_file = 'train_files_filtered_gallbladder_2classes.npy', test_pth_file = 'test_files_filtered_gallbladder_2classes.npy')
        test_dataset =  LiverSegDataset_2_classes("/home/bidur/vaal/data/liver_seg_dataset", train = False, scale = scale, flip = False, resize= args.resize,train_pth_file = 'train_files_filtered_gallbladder_2classes.npy', test_pth_file = 'test_files_filtered_gallbladder_2classes.npy')
        
        
        args.num_val = 1548
        args.num_images = 15482
        #args.budget = 200
        args.budget = 100
        # args.budget = 774
        #args.budget = 818*2
        #args.initial_budget = 774
        #args.initial_budget = 200
        args.initial_budget = 100
        args.num_classes = 2

    elif args.dataset == 'liver-seg-small':
    
        scale = tuple(float(i) for i in args.scale.split(","))
        if min(scale) == 0:
            scale = None

        train_dataset =  LiverSegDataset("/home/bidur/vaal/data/liver_seg_dataset", scale = scale, flip = False, resize= args.resize, train_pth_file= 'train_files_curated.npy')
        test_dataset =  LiverSegDataset("/home/bidur/vaal/data/liver_seg_dataset", train = False, scale = scale, flip = False, resize= args.resize, test_pth_file= "test_files_curated.npy")

        args.num_val = 500
        args.num_images = 2000
        #args.budget = 0
        args.budget = 100
        args.initial_budget = 200
        args.num_classes = 5

    elif args.dataset == 'classification-liver-seg-gallbladder-removed-class-no-less-than-3':

        scale = tuple(float(i) for i in args.scale.split(","))
        if min(scale) == 0:
            scale = None

        train_dataset = LiverSegDataset_Classification("data/liver_seg_dataset", train = True, scale = scale, flip = False, resize= args.resize,train_pth_file = "train_files_classification_filtered_gallbladder_no_less_than_3_classes.npy" )
        test_dataset = LiverSegDataset_Classification("data/liver_seg_dataset", train = False, scale = scale, flip = False, resize= args.resize,test_pth_file = "test_files_classification_filtered_gallbladder_no_less_than_3_classes.npy" )

        
        
        args.num_val = 1548
        args.num_images = 15482
        #args.budget = 774
        #args.budget = 200
        args.budget = 500
        #args.initial_budget = 774
        args.initial_budget = 500

        args.num_classes = 4

    elif args.dataset == 'classification-liver-seg-gallbladder-removed':
        
        scale = tuple(float(i) for i in args.scale.split(","))
        if min(scale) == 0:
            scale = None

        train_dataset =  LiverSegDataset_Classification("/home/bidur/vaal/data/liver_seg_dataset", scale = scale, flip = False, resize= args.resize,train_pth_file = 'train_classification_files_filtered_gallbladder.npy', test_pth_file = 'test_classification_files_filtered_gallbladder.npy')
        test_dataset =  LiverSegDataset_Classification("/home/bidur/vaal/data/liver_seg_dataset", train = False, scale = scale, flip = False, resize= args.resize,train_pth_file = 'train_classification_files_filtered_gallbladder.npy', test_pth_file = 'test_classification_files_filtered_gallbladder.npy')
        
        
        args.num_val = 1819
        args.num_images = 18191
        #args.budget = 818
        args.budget = 818*2
        args.initial_budget = 818
        #args.budget = 500
        #args.initial_budget = 200
        args.num_classes = 4

    else:
        raise NotImplementedError

    # save the hyper-parameters in wandb
    experiment.config.update(vars(args))
    

    all_indices = set(np.arange(args.num_images))
    random.seed(0)  #every time set the same seed
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
    test_dataloader = data.DataLoader(test_dataset,
            batch_size=args.batch_size, drop_last=False)
    
 
            
    args.device = torch.device('cuda:'+args.gpu_id if torch.cuda.is_available() else 'cpu')
   

    #splits = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    splits = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55]
    
    current_indices = list(initial_indices)
    for i, split in enumerate(splits):


        # need to retrain all the models on the new images
        # re initialize and retrain the models
        # task_model = vgg.vgg16_bn(num_classes=args.num_classes)
        experiment.log({
                    'split': split,
        
                })

        fix_seed(args.seed)

        if args.task_type == "segmentation":
            task_model = UNet(n_channels=3, n_classes=args.num_classes, bilinear=args.bilinear)
            task_model.to(device=args.device)
            train_task(args, net=task_model, train_loader = querry_dataloader, val_loader = val_dataloader, test_loader= test_dataloader,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.lr,
                    amp=args.amp, wandb_log= experiment, split = split)

        elif args.task_type == "classification":
           
            task_model = torch_models.resnet18(pretrained= False, num_classes = args.num_classes)
            task_model.to(device=args.device)
            train_multilabel_classifier(args, net=task_model, train_loader = querry_dataloader, val_loader = val_dataloader, test_loader= test_dataloader,
              epochs = args.epochs,
              batch_size = args.batch_size,
              learning_rate=args.lr,
              wandb_log= experiment, split = split)


        ## all unlabeled train samples
        unlabeled_indices = np.setdiff1d(list(all_indices), current_indices)
        unlabeled_sampler = data.sampler.SubsetRandomSampler(unlabeled_indices)
        unlabeled_dataloader = data.DataLoader(train_dataset, 
                sampler=unlabeled_sampler, batch_size=args.batch_size, drop_last=False)

        if split == splits[-1]:
            break

        if args.method == 'VAAL':
            #### initilaize the VAAL models
            discriminator = model.Discriminator(args.latent_dim)
            vae = model.VAE(args.latent_dim)

            # load the checkpoint models
            discriminator.load_state_dict(torch.load('./checkpoints/'+'/'+args.expt + 
                                        '/'+ 'discriminator_checkpoint'+str(split)+'.pth'))
            vae.load_state_dict(torch.load('./checkpoints/'+'/'+args.expt + 
                                        '/'+ 'vae_checkpoint'+str(split)+'.pth'))
 
            # send model to gpu
            discriminator = discriminator.to(device = args.device)
            vae = vae.to(device = args.device)
            
            VAAL_sampler = query_Sampler.AdversarySampler(args.budget)


            sampled_indices = VAAL_sampler.sample(vae, 
                                             discriminator, 
                                             unlabeled_dataloader, unlabeled_indices,
                                             args.device)


        elif args.method == 'multimodal_VAAL':
            
            #### initilaize the VAAL models
            vae = multi_modal_model.VAE(args.latent_dim)
            discriminator = multi_modal_model.Discriminator(args.latent_dim)

            # load the checkpoint models
            discriminator.load_state_dict(torch.load('./checkpoints/'+'/'+args.expt + 
                                        '/'+ 'discriminator_checkpoint'+str(split)+'.pth'))
            vae.load_state_dict(torch.load('./checkpoints/'+'/'+args.expt + 
                                        '/'+ 'vae_checkpoint'+str(split)+'.pth'))

            # send the model to gpu
            vae = vae.to(device = args.device)
            discriminator = discriminator.to(device = args.device)

            

            multimodal_VAAL_sampler = query_Sampler.AdversarySampler_multimodal(args.budget)
            sampled_indices = multimodal_VAAL_sampler.sample(vae, 
                                             discriminator, 
                                             unlabeled_dataloader, unlabeled_indices,
                                             args.device)

        elif args.method == "RandomSampling":
            
            random.seed(args.random_sampling_seed)
            random.shuffle(unlabeled_indices)
            sampled_indices = unlabeled_indices[:args.budget]

        current_indices = list(current_indices) + list(sampled_indices)
        sampler = data.sampler.SubsetRandomSampler(current_indices)
        querry_dataloader = data.DataLoader(train_dataset, sampler=sampler, 
                batch_size=args.batch_size, drop_last=True)

if __name__ == '__main__':
    args = arguments.get_args()
    fix_seed(args.seed)
    main(args)

