import os
import matplotlib.pyplot as plt
import torch
import torch.utils.data as data
import sampler as query_Sampler
import numpy as np
import random
from custom_datasets import *
import model
import multi_modal_model
from utils import *
import arguments
from sklearn.manifold import TSNE
import seaborn as sns
import torch.nn as nn


def main(args):



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


    elif args.dataset == 'liver-seg-gallbladder-removed-class-no-less-than-3-small':
       
        scale = tuple(float(i) for i in args.scale.split(","))
        if min(scale) == 0:
            scale = None

        train_dataset =  LiverSegDataset_Gallbladder_Removed("/home/bidur/vaal/data/liver_seg_dataset", scale = scale, flip = False, resize= args.resize,train_pth_file = 'train_files_filtered_gallbladder_no_less_than_3_classes_curated_3000.npy', test_pth_file = 'test_files_filtered_gallbladder_no_less_than_3_classes_curated_3000.npy')
        test_dataset =  LiverSegDataset_Gallbladder_Removed("/home/bidur/vaal/data/liver_seg_dataset", train = False, scale = scale, flip = False, resize= args.resize,train_pth_file = 'train_files_filtered_gallbladder_no_less_than_3_classes_curated_3000.npy', test_pth_file = 'test_files_filtered_gallbladder_no_less_than_3_classes_curated_3000.npy')
        
        
        args.num_val = 500
        args.num_images = 3000
        args.budget = 200
        args.initial_budget = 200
        args.num_classes = 4


    elif args.dataset == 'liver-seg-gallbladder-removed-class-no-less-than-3-rendered':
       
        scale = tuple(float(i) for i in args.scale.split(","))
        if min(scale) == 0:
            scale = None

        train_dataset =  LiverSegDataset_Gallbladder_Removed("/home/bidur/vaal/data/liver_seg_dataset", scale = scale, flip = False, resize= args.resize, translated_input = False, train_pth_file = 'train_files_filtered_gallbladder_no_less_than_3_classes.npy', test_pth_file = 'test_files_filtered_gallbladder_no_less_than_3_classes.npy')
        test_dataset =  LiverSegDataset_Gallbladder_Removed("/home/bidur/vaal/data/liver_seg_dataset", train = False, scale = scale, flip = False, resize= args.resize,translated_input = False, train_pth_file = 'train_files_filtered_gallbladder_no_less_than_3_classes.npy', test_pth_file = 'test_files_filtered_gallbladder_no_less_than_3_classes.npy')
        
        
        args.num_val = 1548
        args.num_images = int(15482/3)
        #args.budget = 774
        #args.budget = 200
        args.budget = 200
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


    elif args.dataset == 'rendered-classification-liver-seg-gallbladder-removed-class-no-less-than-3':

        scale = tuple(float(i) for i in args.scale.split(","))
        if min(scale) == 0:
            scale = None

        train_dataset = LiverSegDataset_Classification("data/liver_seg_dataset", train = True, scale = scale, flip = False, resize= args.resize,translated_input = False, train_pth_file = "train_files_classification_filtered_gallbladder_no_less_than_3_classes.npy" )
        test_dataset = LiverSegDataset_Classification("data/liver_seg_dataset", train = False, scale = scale, flip = False, resize= args.resize,translated_input = False,test_pth_file = "test_files_classification_filtered_gallbladder_no_less_than_3_classes.npy" )

        
        
        args.num_val = 1548
        args.num_images = int(15482/3)
        #args.budget = 774
        #args.budget = 200
        args.budget = 200
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
        args.budget = 500
        args.initial_budget = 500
        #args.budget = 500
        #args.initial_budget = 200
        args.num_classes = 4



    elif args.dataset == 'classification-liver-seg-gallbladder-removed-small':
        
        scale = tuple(float(i) for i in args.scale.split(","))
        if min(scale) == 0:
            scale = None

        train_dataset =  LiverSegDataset_Classification("/home/bidur/vaal/data/liver_seg_dataset", scale = scale, flip = False, resize= args.resize,train_pth_file = 'train_classification_files_filtered_gallbladder_curated_3000.npy', test_pth_file = 'test_classification_files_filtered_gallbladder_curated_3000.npy')
        test_dataset =  LiverSegDataset_Classification("/home/bidur/vaal/data/liver_seg_dataset", train = False, scale = scale, flip = False, resize= args.resize,train_pth_file = 'train_classification_files_filtered_gallbladder_curated_3000.npy', test_pth_file = 'test_classification_files_filtered_gallbladder_curated_3000.npy')
        
        
        args.num_val = 500
        args.num_images = 3000
        args.budget = 200
        args.initial_budget = 200
        args.num_classes = 4

    elif args.dataset == 'rendered-classification-liver-seg-gallbladder-removed':
        
        scale = tuple(float(i) for i in args.scale.split(","))
        if min(scale) == 0:
            scale = None

        train_dataset =  LiverSegDataset_Classification("/home/bidur/vaal/data/liver_seg_dataset", scale = scale, flip = False, resize= args.resize,translated_input = False,train_pth_file = 'train_classification_files_filtered_gallbladder.npy', test_pth_file = 'test_classification_files_filtered_gallbladder.npy')
        test_dataset =  LiverSegDataset_Classification("/home/bidur/vaal/data/liver_seg_dataset", train = False, scale = scale, flip = False, resize= args.resize,translated_input = False,train_pth_file = 'train_classification_files_filtered_gallbladder.npy', test_pth_file = 'test_classification_files_filtered_gallbladder.npy')
        
        
        args.num_val = 1819
        args.num_images = int(18191/3)
        #args.budget = 818
        args.budget = 200
        args.initial_budget = 500
        #args.budget = 500
        #args.initial_budget = 200
        args.num_classes = 4

    else:
        raise NotImplementedError

    
    all_indices = set(np.arange(args.num_images))
    whole_sampler = data.sampler.SubsetRandomSampler(list(all_indices))
    whole_dataloader = data.DataLoader(train_dataset, sampler = whole_sampler,
            batch_size=args.batch_size, drop_last=True)
    random.seed(0)  #every time set the same seed
    val_indices = random.sample(all_indices, args.num_val)
    all_indices = np.setdiff1d(list(all_indices), val_indices)

    

    initial_indices = random.sample(list(all_indices), args.initial_budget)
    sampler = data.sampler.SubsetRandomSampler(initial_indices)
    val_sampler = data.sampler.SubsetRandomSampler(val_indices)

    # dataset with labels available
    querry_dataloader = data.DataLoader(train_dataset, sampler=sampler, 
            batch_size=args.batch_size, drop_last=True)
            
    args.device = torch.device('cuda:'+args.gpu_id if torch.cuda.is_available() else 'cpu')
   

    splits = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    #splits = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55]
    
    current_indices = list(initial_indices)
    for i, split in enumerate(splits):

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

        old_indices = list(current_indices)
        current_indices = list(current_indices) + list(sampled_indices)
        

        if split == 0.25:
           
            if args.method == "RandomSampling":
                tsne(old_indices, sampled_indices, unlabeled_indices, whole_dataloader, vae = None, disc= None, method = "RandomSampling")
            elif args.method == "VAAL":
                tsne(old_indices, sampled_indices, unlabeled_indices, whole_dataloader, vae = vae, disc = discriminator, method = "VAAL")
            elif args.method == "multimodal_VAAL":
                tsne(old_indices, sampled_indices, unlabeled_indices, whole_dataloader, vae = vae, disc = discriminator, method = "multimodal_VAAL")

            break

        
def extract_features (dataloader, model_name = "inception_v3", vae = None, disc = None):
    features_data = []
    

    if model_name == "inception_v3":
        model = torchvision.models.inception_v3(pretrained=True)
        model.fc = nn.Identity()
        model.to(args.device)
        model.eval()

    elif (model_name == "multimodalvae") or (model_name == "vae"):
        if  disc:
            model = vae
            model.eval()

            model_disc = disc
            #model_disc.net = nn.Sequential(*[model_disc.net[i] for i in range(4)])
            model_disc.eval()
        else:
            model = vae
            model.eval()
            
    
    # put features and labels into arrays
    for batch_ix, (batch_image, batch_mask, batch_depth) in enumerate(dataloader):
        batch_image = batch_image.to(args.device)
        with torch.no_grad():
            if disc:
                if model_name =="multimodalvae":
                    _,_,_,batch_feature,_ = model(batch_image)
                elif model_name =="vae":
                    _,_,batch_feature,_ = model(batch_image)

                batch_feature = model_disc(batch_feature)
            else:
                if model_name == "inception_v3":
                    batch_feature = model(batch_image)
                elif model_name =="multimodalvae":
                    _,_,_,batch_feature,_ = model(batch_image)
                elif model_name =="vae":
                    _,_,batch_feature,_ = model(batch_image)
        features_data.append(batch_feature.flatten().cpu().numpy())
    
    return torch.Tensor(features_data)

def tsne(old_indices, sampled_indices, unlabeled_indices, whole_dataloader, vae = None, disc= None, method = "RandomSampling"):
    

    # whole_features= extract_features(whole_dataloader)
    
    if method == "VAAL":
        whole_features = extract_features(whole_dataloader, model_name= "vae", vae = vae, disc= disc) 
    elif method == "multimodal_VAAL" :
        whole_features = extract_features(whole_dataloader, model_name= "multimodalvae", vae = vae,disc= disc) 
    else:
        whole_features= extract_features(whole_dataloader)


    labels = ["unlabelled" for i in range (len(unlabeled_indices))]
    tsne = TSNE(n_components=2,n_iter=300)
    tsne_results = tsne.fit_transform(whole_features)
    print(len(tsne_results))
    tsne_X = tsne_results[:,0][unlabeled_indices]
    tsne_Y = tsne_results[:,1][unlabeled_indices]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(
        x=tsne_X, y=tsne_Y,
        hue= labels,
        data=tsne_results,
        legend="full",
        alpha=0.3)

    tsne_X = tsne_results[:,0][old_indices]
    tsne_Y = tsne_results[:,1][old_indices]
    labels = ["labelled" for i in range (len(old_indices))]

    sns.scatterplot(
        x=tsne_X, y=tsne_Y,
        hue= labels,
        data=tsne_results,
        legend="full",
        color=".2",
        palette="rocket",
        alpha=0.8)
    
    tsne_X = tsne_results[:,0][sampled_indices]
    tsne_Y = tsne_results[:,1][sampled_indices]
    labels = ["selected" for i in range (len(sampled_indices))]

    sns.scatterplot(
        x=tsne_X, y=tsne_Y,
        hue= labels,
        data=tsne_results,
        legend="full",
        color=".5",
        palette="viridis",
        alpha=0.8)
    

    save_name = "VAAL_disc.png"
    plt.savefig(save_name)
    plt.close(fig)



if __name__ == '__main__':
    args = arguments.get_args()
    main(args)

