import logging
import sys
from pathlib import Path
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import random
import numpy as np
import warnings
from sklearn.metrics import average_precision_score
warnings.filterwarnings("ignore")


def train_multilabel_classifier(args,net, train_loader, val_loader, test_loader,
              epochs: int = 100,
              batch_size: int = 16,
              learning_rate: float = 0.1,
              save_checkpoint: bool = True, wandb_log = None, split = 1):


    dir_checkpoint = Path('./checkpoints/')
    scale = tuple(float(i) for i in args.scale.split(","))
    if min(scale) == 0:
        scale = None

    if not os.path.exists(str(dir_checkpoint)+'/'+args.expt + '/'+ 'checkpoint'+str(split)+'.pth'):

        best_val_mAP = 0.0

        
        # 4. Set up the optimizer, the loss, the learning rate scheduler 
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, weight_decay=5e-4, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
        criterion = nn.BCEWithLogitsLoss()
        global_step = 0
        

        # 5. Begin training
        for epoch in range(1, epochs+1):
            net.train()
            epoch_loss = 0
            with tqdm(total=len(train_loader)*args.batch_size, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
                for images, labels, depths in train_loader:
                    
                    images = images.to(device=args.device, dtype=torch.float32)
                    labels = labels.to(device=args.device, dtype=torch.double)
                    
                    outputs = net(images)
                    optimizer.zero_grad()
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    outputs_no_sig = outputs
                    outputs = torch.sigmoid(outputs) 
                    
                    epoch_loss += loss.item()
                    pbar.update(images.shape[0])
                    global_step += 1
                    wandb.log({
                        'train loss': loss.item(),
                        'step': global_step,
                        'epoch': epoch
                    })
                    pbar.set_postfix(**{'loss (batch)': loss.item()})

            ### Evaluation round
            histograms = {}
            for tag, value in net.named_parameters():
                tag = tag.replace('/', '.')
                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())


            val_mAP = evaluate(net,val_loader, args.device)
            scheduler.step()

            logging.info('Validation mAP: {}'.format(val_mAP))
            wandb_log.log({
                'learning rate': optimizer.param_groups[0]['lr'],
                'validation mAP': val_mAP,
                'step': global_step,
                'epoch': epoch,
                **histograms
            })       

            if best_val_mAP < val_mAP:
                best_val_mAP = val_mAP
                if save_checkpoint:
                    Path(str(dir_checkpoint)+'/'+args.expt).mkdir(parents=True, exist_ok=True)
                    torch.save(net.state_dict(), str(dir_checkpoint)+'/'+args.expt + '/'+ 'checkpoint'+str(split)+'.pth')
                    logging.info(f'Checkpoint {epoch} saved!')
            

    net.load_state_dict(torch.load(str(dir_checkpoint)+'/'+args.expt + '/'+ 'checkpoint'+str(split)+'.pth'))
    test_mAP = evaluate(net,test_loader, args.device)
    wandb_log.log({'test mAP': test_mAP})
    


def evaluate(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    mAP = 0.0
    all_targets = []
    all_predictions = []

    # iterate over the validation set
    for image, labels, depths in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        labels = labels.to(device=device, dtype=torch.double)
       
        with torch.no_grad():
            outputs = net(image)
            outputs_no_sig = outputs
            outputs = torch.sigmoid(outputs)

            all_targets.extend(labels.detach().cpu().numpy())
            all_predictions.extend(outputs.detach().cpu().numpy())   


    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)  
    mAP = average_precision_score(all_targets, all_predictions)

    net.train()
    return mAP