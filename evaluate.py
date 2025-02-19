import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import wandb
#from torchmetrics.functional import dice_score as ds
from dice_score import multiclass_dice_coeff, dice_coeff, perclass_dice_coeff


def evaluate(net, dataloader, device, ignore_background = False):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    for image, mask_true, depth in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        temp_mask_true = mask_true
        mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)

        
            # convert to one-hot format
            if net.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                if ignore_background:
                    # compute the Dice score, ignore the background
                    dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)
                else:
                    # compute the Dice score, don't ignore background
                    dice_score += multiclass_dice_coeff(mask_pred, mask_true, reduce_batch_first=False)        
    net.train()
    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score
    return dice_score / num_val_batches





def evaluate_classwise(net, dataloader, device, ignore_background = False, log_masks = None):
    net.eval()
    num_val_batches = len(dataloader)
    classwise_dice_score = {0:0,1:0,2:0,3:0,4:0}
    eachbatch_dice = []
    classwise_perbatch_dice_scores = {0:[],1:[],2:[],3:[],4:[]}
    # iterate over the validation set
    for image, mask_true, depth in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)

        
            # convert to one-hot format
            if net.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, don't ignore background
                class_dice_score = perclass_dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
                if ignore_background:
                    # compute the Dice score, ignore the background
                    dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)
                else:
                    # compute the Dice score, don't ignore background
                    dice_score += multiclass_dice_coeff(mask_pred, mask_true, reduce_batch_first=False)   
                
                batchwise_dice = 0
                
                for key, value in class_dice_score.items():
                    classwise_dice_score[key] = classwise_dice_score[key] + value
                    classwise_perbatch_dice_scores[key].append(value) 
                    batchwise_dice += value 

                eachbatch_dice.append(batchwise_dice)

        if log_masks is not None:
            log_masks.log({
            'images': wandb.Image(image[0].cpu()),
            'masks': {
                'true': wandb.Image(mask_true[0].float().cpu()),
                'pred': wandb.Image(mask_pred.argmax(dim=1)[0].float().cpu()),
            },
        })       
    STD = np.std(eachbatch_dice) #/np.sqrt(num_val_batches)
    
    class_wise_STD = []

    ### classwise SME
    for key, value in classwise_perbatch_dice_scores.items():
        #class_wise_SME.append(np.std(value)/np.sqrt(num_val_batches)) 
        class_wise_STD.append(np.std(value)) 


    net.train()
    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return classwise_dice_score, STD

    classwise_dice_score = {key: value / num_val_batches for key, value in classwise_dice_score.items()}
    return dice_score, classwise_dice_score, STD, class_wise_STD, classwise_perbatch_dice_scores

