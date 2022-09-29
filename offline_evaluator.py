import logging
from multimodal_VAAL_solver import multi_modal_VAAL_Solver
import torch
from torchvision import datasets, transforms
import torch.utils.data.sampler  as sampler
import torch.utils.data as data
import torch.nn as nn
import numpy as np
import random
from pathlib import Path
from custom_datasets import *
import model
import multi_modal_model
from VAAL_solver import VAAL_Solver
from utils import *
import arguments
from evaluate import evaluate, evaluate_classwise
from unet import UNet
import wandb
import torch.backends.cudnn as cudnn
import warnings
warnings.filterwarnings("ignore")

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
    experiment = wandb.init(project='U-Net-active-learning-final-RC-eval')
    
    if args.dataset == 'liver-seg':
        test_dataloader = data.DataLoader(
                datasets.ImageFolder(args.data_path, transform=imagenet_transformer()),
            drop_last=False, batch_size=args.batch_size)

        scale = tuple(float(i) for i in args.scale.split(","))
        if min(scale) == 0:
            scale = None

        train_dataset =  LiverSegDataset("/home/bidur/vaal/data/liver_seg_dataset", scale = scale, flip = True, resize= args.resize,train_pth_file = 'train_files.npy', test_pth_file = 'test_files.npy')
        test_dataset =  LiverSegDataset("/home/bidur/vaal/data/liver_seg_dataset", train = False, scale = scale, flip = True, resize= args.resize,train_pth_file = 'train_files.npy', test_pth_file = 'test_files.npy')
        
        args.num_val = 1890
        args.num_images = 18900
        args.budget = 850
        args.initial_budget = 850
        #args.budget = 500
        #args.initial_budget = 200
        args.num_classes = 5

    elif args.dataset == 'liver-seg-small':
        
        test_dataloader = data.DataLoader(
                datasets.ImageFolder(args.data_path, transform=imagenet_transformer()),
            drop_last=False, batch_size=args.batch_size)

        scale = tuple(float(i) for i in args.scale.split(","))
        if min(scale) == 0:
            scale = None

        train_dataset =  LiverSegDataset("/home/bidur/vaal/data/liver_seg_dataset", scale = scale, flip = True, resize= args.resize, train_pth_file= 'train_files_curated.npy')
        test_dataset =  LiverSegDataset("/home/bidur/vaal/data/liver_seg_dataset", train = False, scale = scale, flip = True, resize= args.resize, test_pth_file= "test_files_curated.npy")

        args.num_val = 500
        args.num_images = 2000
        #args.budget = 0
        args.budget = 100
        args.initial_budget = 200
        args.num_classes = 5
   

    else:
        raise NotImplementedError

    # save the hyper-parameters in wandb
    experiment.config.update(vars(args))
    

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
    test_dataloader = data.DataLoader(test_dataset,
            batch_size=args.batch_size, drop_last=False)
    
 
            
    args.device = torch.device('cuda:'+args.gpu_id if torch.cuda.is_available() else 'cpu')
    args.evaluation_only = True
   

    splits = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    for i, split in enumerate(splits):


        experiment.log({
                    'split': split,
        
                })

        fix_seed(0)

        task_model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
        task_model.to(device=args.device)
        test_task(args, net=task_model, train_loader = querry_dataloader, val_loader = val_dataloader, test_loader= test_dataloader,
                   wandb_log= experiment, split = split)


        if split == splits[-1]:
            break


def test_task(args,net, train_loader, val_loader, test_loader,wandb_log = None, split = 1):


    scale = tuple(float(i) for i in args.scale.split(","))
    if min(scale) == 0:
        scale = None

    dir_checkpoint = Path('./checkpoints/')
    val_score = evaluate_classwise(net, val_loader, args.device)
    for key, value in val_score.items():
            wandb_log.log({'Val Dice Class '+str(key): value})
    
    net.load_state_dict(torch.load(str(dir_checkpoint)+'/'+args.expt + '/'+ 'checkpoint'+str(split)+'.pth'))
    
    val_score = evaluate_classwise(net, val_loader, args.device)
    for key, value in val_score.items():
            wandb_log.log({'Val Dice Class '+str(key): value})
            
    test_score = evaluate_classwise(net, test_loader, args.device)
    for key, value in test_score.items():
            wandb_log.log({'Test Dice Class '+str(key): value})




if __name__ == '__main__':
    args = arguments.get_args()
    fix_seed(0)
    main(args)

