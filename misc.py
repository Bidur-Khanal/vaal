import numpy as np
import torch
from torchmetrics.functional import dice_score as ds
from dice_score import multiclass_dice_coeff, dice_coeff
import torch.nn.functional as F


pred = torch.tensor([[[[0.85, 0.05, 0.05, 0.05],
                      [0.05, 0.85, 0.05, 0.05],
                      [0.05, 0.05, 0.85, 0.05],
                      [0.05, 0.05, 0.05, 0.85]]]])
print(pred.shape)
pred = pred.permute(0,3,1,2)
print(pred.shape)

target = torch.tensor([[[0, 1, 3, 2]]])
print(target.shape)
target = target.permute(0, 2,1)
print(target.shape)

library_dice = ds(pred, target, bg = True)
print ("dice score from the library: ", library_dice)



dice_score = 0.
_,sh = torch.max(pred, dim = 1)
print(sh.shape)
sh = sh.permute(0,2,1)
sh_one = F.one_hot(sh, 4)
sh_one = sh_one.permute(0,3,1,2)
print(sh_one)
print (sh_one.shape)

print (target.shape)
target_one = F.one_hot(target, 4)
target_one = target_one.permute(0,3,1,2)
print(target_one.shape)


dice_score += multiclass_dice_coeff(sh_one, target_one, reduce_batch_first=False)
print ("dice score computed using the unet code: ", dice_score)


