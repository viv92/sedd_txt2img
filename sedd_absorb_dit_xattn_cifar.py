'''
### Program implementing 

## Features:
1. 

## Todos / Questions:
1.

'''

import os
import cv2
import math 
from copy import deepcopy 
from matplotlib import pyplot as plt 
import numpy as np
import torch
torch.set_float32_matmul_precision('high') # use TF32 precision for speeding up matmul
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import json 
from torch.utils.data import DataLoader

from torchvision.datasets import CIFAR10
from torchvision.utils import save_image, make_grid

# import T5 
from transformers import T5Tokenizer, T5ForConditionalGeneration
# import VQVAE for loading the pretrained weights
from fsq_transformer import FSQ_Transformer, init_transformer, patch_seq_to_img, img_to_patch_seq
# import DiT 
from utils_sedd_dit_xattn import *



# noise schedule
def logLinearNoise(t, eps = 1e-3):
    """
    Log Linear noise schedule built so that 1 - 1/e^(n(t)), i.e., the flip probability interpolates between 0 and ~1
    when t goes from 0 to 1. Used for absorbing

    Total noise sigma is -log(1 - (1 - eps) * t), so the sigma (derivative of total noise) will be (1 - eps) / (1 - (1 - eps) * t)
    """
    total_noise = -torch.log1p(-(1 - eps) * t)
    rate_noise = (1 - eps) / (1 - (1 - eps) * t)
    return total_noise, rate_noise 


# perturbation function for forward diffusion process (absorption / mask case)
def perturb(x, sigma, mask_token): # x.shape: [batch_size, seq_len]
    sigma = sigma.unsqueeze(-1) # sigma.shape: [batch_size, 1]
    flip_prob = 1 - (-sigma).exp()
    flip_indices = torch.rand(*x.shape, device=x.device) < flip_prob 
    x_perturb = torch.where(flip_indices, mask_token, x) # fill the mask_token at flip_indices; fill the original token at other indices
    return x_perturb # x_perturb.shape: [b, seqlen]


# loss function 
def score_entropy_loss(log_score, sigma, x, x0, mask_token): # log_score.shape: [b, seqlen, vocab_size]
    flipped_indices = x == mask_token # flipped_indices is a boolean tensor with shape: [b, seqlen]

    # calculate exp(sigma) - 1 with high precision
    esigm1 = torch.where(
        sigma < 0.5,
        torch.expm1(sigma),
        torch.exp(sigma) - 1
    )

    # since ratio = p(y) / p(x) =
    # for unflipped indices = exp(-sigma) / exp(-sigma) = 1
    # for flipped indices = (1 - exp(-sigma)) / exp(-sigma) = exp(sigma) - 1 
    ratio = 1 / esigm1.expand_as(x)[flipped_indices] # ratio.shape: [b * num_flipped_tokens_in_each_sequence]
    flipped_tokens = x0[flipped_indices].unsqueeze(-1) # flipped_tokens.shape: [b * num_flipped_tokens_in_each_sequence, 1]

    ## prepare loss terms (equation 5 in the SEDD paper)

    # negative_term
    # torch.gather gathers the log_scores at the flipped indices (along seq_len_dim) and at the flipped token values (along the vocab dim) to give a 1-D tensor of shape [b * num_flipped_tokens_in_each_sequence]
    log_scores_for_flipped_indices = log_score[flipped_indices] # shape: [b * num_flipped_tokens_in_each_sequence, vocab_size]
    neg_term = ratio * torch.gather(log_scores_for_flipped_indices, -1, flipped_tokens).squeeze(-1)

    #positive term
    # sum all scores along the vocab dim, except for the mask_token
    pos_term = log_scores_for_flipped_indices[:, :-1].exp().sum(dim=-1) # shape: [b * num_flipped_tokens_in_each_sequence]

    # constant term
    const = ratio * (ratio.log() - 1)

    entropy = torch.zeros(*x.shape, device=x.device)
    entropy[flipped_indices] += pos_term - neg_term + const
    return entropy


# utility function to expand dims of x to match dims of y 
def unsqueeze_as(x, y):
    while len(x.shape) < len(y.shape):
        x = x.unsqueeze(-1)
    return x 

# utility function to sample from categorical distribution - TODO why not just use multinomial?
def sample_categorical(probs):
    gumbel_norm = 1e-10 - (torch.rand_like(probs) + 1e-10).log()
    return (probs / gumbel_norm).argmax(dim=-1)

# utility function to calculate staggered score - corresponds to the LHS term in the product in equation 19 of the SEDD paper 
def get_stag_score(score, dsigma):
    '''
    score.shape: [b, seqlen, vocab_size]
    dsigma.shape: [b, 1]
    '''
    extra_const = (1 - (dsigma).exp()) * score.sum(dim=-1) # extra_const.shape: [b=1, seqlen]
    stag_score = score * dsigma.exp().unsqueeze(-1)
    stag_score[..., -1] += extra_const # add extra_const to the score values for mask token transitions
    return stag_score

# utility function to calculate staggered probability = marginal probability but using dsigma = exp(dsigma * Q) - corresponds to the RHS term in the product in equation 19 of the SEDD paper 
def get_stag_prob(x, dsigma, vocab_size, mask_token):
    dsigma = unsqueeze_as(dsigma, x.unsqueeze(-1)) # dsigma.shape: [1, seqlen, 1]
    stag_prob = (-dsigma).exp() * F.one_hot(x, num_classes=vocab_size) # stag_prob.shape: [1, seqlen, vocab_size]
    stag_prob += torch.where(
        x == mask_token,
        1 - (-dsigma).squeeze(-1).exp(),
        0
    ).unsqueeze(-1)
    return stag_prob


## sampling function
def get_sample(net, seq_len, mask_token, vocab_size, num_sampling_steps, sample_batch_size, sample_condition, cfg_scale, device, eps=1e-5):
    # x_T
    x = mask_token * torch.ones((sample_batch_size, seq_len), dtype=torch.int64, device=device)

    timesteps = torch.linspace(1, eps, num_sampling_steps + 1, device=device)
    dt = (1 - eps) / num_sampling_steps

    for i in range(num_sampling_steps):
        t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)
        curr_sigma = logLinearNoise(t)[0]
        next_sigma = logLinearNoise(t - dt)[0]
        dsigma = curr_sigma - next_sigma

        # get conditioned score
        log_score_cond = net(x, curr_sigma.squeeze(-1), sample_condition)
        score_cond = log_score_cond.exp().clone().detach()

        # get unconditioned score
        log_score = net(x, curr_sigma.squeeze(-1), None)
        score = log_score.exp().clone().detach()

        # apply cfg 
        score = score_cond + cfg_scale * (score_cond - score)

        # calculate staggered score - corresponds to the LHS term in the product in equation 19 of the SEDD paper 
        stag_score = get_stag_score(score, dsigma)

        # calculate staggered probability = marginal probability but using dsigma = exp(dsigma * Q) - corresponds to the RHS term in the product in equation 19 of the SEDD paper 
        stag_prob = get_stag_prob(x, dsigma, vocab_size, mask_token)

        # sampling probability for reverse diffusion process - equation 19 of SEDD paper
        probs = stag_score * stag_prob
        # x_(t-1)
        x =  sample_categorical(probs)
        
    ## final sampling step: going from x_1 to x_0

    t = timesteps[-1] * torch.ones(x.shape[0], 1, device=device)
    sigma = logLinearNoise(t)[0]

    log_score_cond = net(x, sigma.squeeze(-1), sample_condition)
    score_cond = log_score_cond.exp().clone().detach()

    log_score = net(x, sigma.squeeze(-1), None)
    score = log_score.exp().clone().detach()

    # apply cfg
    score = score_cond + cfg_scale * (score_cond - score)

    stag_score = get_stag_score(score, sigma)

    stag_prob = get_stag_prob(x, sigma, vocab_size, mask_token)

    # added cfg 
    probs = stag_score * stag_prob

    # truncate probabilities - avoid mask prob
    probs = probs[..., :-1]
    x =  sample_categorical(probs)
        
    return x


# fetch dataset - using data loader
def cifar10_dl(img_size, batch_size):
    tf = torchvision.transforms.Compose([
        # torchvision.transforms.Resize(int(img_size * 1.25)),  # args.image_size + 1/4 *args.image_size
        # torchvision.transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # equivalent to transforming pixel values from range [0,1] to [-1,1]
    ])
    dataset = CIFAR10(
        "./dataset_cifar",
        train=True,
        download=True,
        transform=tf,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=16)
    return dataloader

# utility function to freeze model
def freeze(model):
    for p in model.parameters():
        p.requires_grad_(False) 

# utility function to load model weights from checkpoint - loads to the device passed as 'device' argument
def load_ckpt(checkpoint_path, model, optimizer=None, scheduler=None, device=torch.device('cpu'), mode='eval'):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    if mode == 'eval':
        model.eval() 
        return model
    else:
        model.train()
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if scheduler is not None:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            return model, optimizer, scheduler
        else:
            return model, optimizer
        
# utility function to save a checkpoint (model_state, optimizer_state, scheduler_state) - saves on cpu (to save gpu memory)
def save_ckpt(device, checkpoint_path, model, optimizer, scheduler=None):
    # transfer model to cpu
    model = model.to('cpu')
    # prepare dicts for saving
    save_dict = {'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}
    if scheduler is not None:
        save_dict['scheduler_state_dict'] = scheduler.state_dict()
    torch.save(save_dict, checkpoint_path)
    # load model back on original device 
    model = model.to(device)
        

# convert tensor to img
def to_img(x):
    x = 0.5 * x + 0.5 # transform img from range [-1, 1] -> [0, 1]
    x = x.clamp(0, 1) # clamp img to be strictly in [-1, 1]
    x = x.permute(0,2,3,1) # [b,c,h,w] -> [b,h,w,c]
    return x 

# function to save a generated img
def save_img_generated(x_g, save_path):
    gen_img = x_g.detach().cpu().numpy()
    gen_img = np.uint8( gen_img * 255 )
    # bgr to rgb 
    gen_img = gen_img[:, :, ::-1]
    cv2.imwrite(save_path, gen_img)
        


### main
if __name__ == '__main__':
    # hyperparams for vqvae (FSQ_Transformer)
    num_quantized_values = [7, 5, 5, 5, 5] # L in fsq paper
    latent_dim = len(num_quantized_values)
    img_size = 32 # cifar 
    img_channels = 3 
    img_shape = torch.tensor([img_channels, img_size, img_size])
    resize_shape = (img_size, img_size)
    img_latent_dim = latent_dim # as used in the pretrained VQVAE 

    patch_size = 4 # necessary that img_size % patch_size == 0
    assert img_size % patch_size == 0
    patch_dim = img_channels * (patch_size**2)
    seq_len = (img_size // patch_size) ** 2 # equal to num latents per item
    
    # hyperparams for FSQ Transformer
    d_model_fsq = patch_dim * 2
    n_heads_fsq = 8
    assert d_model_fsq % n_heads_fsq == 0
    d_k_fsq = d_model_fsq / n_heads_fsq 
    d_v_fsq = d_k_fsq 
    n_layers_fsq = 6
    d_ff_fsq = d_model_fsq * 4
    dropout_fsq = 0.1

    # hyperparams for sedd (dit)
    d_model = 1024
    n_layers = 6
    n_heads = 8 # TODO: might be better to keep this low when modeling image token sequences (since small seqlen)
    d_k = d_model // n_heads
    d_v = d_k 
    d_ff = d_model * 4
    dropout = 0.1

    # get vocab size and mask token 
    # TODO: for a new model, can try to pad the vocab size to nearest power of 2 for speed up
    vocab_size = 1
    for n in num_quantized_values:
        vocab_size *= n 
    mask_token = vocab_size - 1

    # hyperparams for training 
    diffusion_start_time_eps = 1e-3
    batch_size = 512
    gradient_accumulation_steps = 1 # why does the loss curve become flat (instead of going down) on increasing this ?
    lr = 3e-4
    num_epochs = 50 # 10
    train_steps_done = 10670
    random_seed = 101010

    # hyperparams for sampling
    num_sampling_steps = 1024
    total_images_in_dataset = 50000
    num_train_steps_per_epoch = total_images_in_dataset // batch_size
    total_train_steps = num_train_steps_per_epoch * num_epochs
    sampling_freq = total_train_steps // 30 # 20
    sample_batch_size = 16
    p_uncond = 0.1 
    cfg_scale = 1.5
    plot_freq = total_train_steps // 5 # 3

    results_dir = './results_cifar_xattn/'
    ckpts_dir = './ckpts/'
    ckpt_path = ckpts_dir + 'cifar_dit_xattn.pt'
    resume_training_from_ckpt = True 

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    if not os.path.exists(ckpts_dir):
        os.makedirs(ckpts_dir)


    # vqvae ckpt path 
    vqvae_ckpt_path = '/home/vivswan/experiments/fsq/ckpts/FSQ_Transformer_cifar|patchSize=4.pt' # path to pretrained vqvae 

    # set random seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)

    # device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # load img dataset
    dataloader = cifar10_dl(img_size, batch_size)

    # load pretrained VQVAE in eval mode 
    # init transformer encoder
    encoder_transformer = init_transformer(patch_dim, seq_len, d_model_fsq, d_k_fsq, d_v_fsq, n_heads_fsq, n_layers_fsq, d_ff_fsq, dropout_fsq, latent_dim, device)
    # init transformer decoder
    decoder_transformer = init_transformer(latent_dim, seq_len, d_model_fsq, d_k_fsq, d_v_fsq, n_heads_fsq, n_layers_fsq, d_ff_fsq, dropout_fsq, patch_dim, device)
    # init FSQ_Transformer 
    vqvae_model = FSQ_Transformer(device, num_quantized_values, encoder_transformer, decoder_transformer, seq_len).to(device)
    vqvae_model = load_ckpt(vqvae_ckpt_path, vqvae_model, device=device, mode='eval')

    # init dit
    x_seq_len = seq_len 
    # condition_seq_len = 1 # just the class label
    max_seq_len = seq_len + 1 # [t, x]
    # x_dim = 1
    condition_dim = 1 
    dit = init_dit(max_seq_len, x_seq_len, d_model, condition_dim, vocab_size, d_k, d_v, n_heads, n_layers, d_ff, dropout, device).to(device)

    # freeze vqvae
    freeze(vqvae_model)

    # optimizer and loss criterion
    optimizer = torch.optim.AdamW(params=dit.parameters(), lr=lr)

    # load ckpt
    if resume_training_from_ckpt:
        dit, optimizer = load_ckpt(ckpt_path, dit, optimizer, device=device, mode='train')

    # torch compile model for speed up 
    dit = torch.compile(dit)

    # train

    train_step = train_steps_done
    epoch = 0
    losses = []

    pbar = tqdm(total=num_epochs)
    while epoch < num_epochs:

        # fetch minibatch
        pbar2 = tqdm(dataloader)
        for imgs, labels in pbar2:
        
            labels = torch.tensor(labels.tolist()).float().to(device)
            imgs = imgs.to(device)

            with torch.no_grad():
                # convert img to sequence of patches
                x = img_to_patch_seq(imgs, patch_size, seq_len) # x.shape: [b, seq_len, patch_dim]
                # obtain img latent embeddings using pre-trained VQVAE
                z_e = vqvae_model.encode(x) # z_e.shape: [b, seq_len,  img_latent_dim]
                img_latents, _, _, _, _, target_idx = vqvae_model.quantize(z_e) # target_idx.shape: [b * img_latent_seqlen]
                target_idx = target_idx.view(-1, seq_len) # [b, seqlen] 

            x = target_idx # x.shape: [b, seq_len] 
            condition = labels.unsqueeze(-1).unsqueeze(-1) # condition.shape: [b, condition_seqlen = 1, condition_dim = 1]

            # for sampling 
            sample_condition = condition[0].unsqueeze(0).expand(sample_batch_size, -1, -1) # shape: [sample_batch_size, 1, 1]
            sample_label = sample_condition[0].squeeze().int().item()
            if p_uncond == 1:
                sample_condition = None

            # set condition = None with prob p_uncond
            if np.random.rand() < p_uncond: # TODO: explore the effect of no CFG versus CFG only during training versus CFG during training and sampling
                condition = None

            # sample diffusion time ~ uniform(eps, 1)
            t = (1 - diffusion_start_time_eps) * torch.rand(x.shape[0], device=device) + diffusion_start_time_eps

            # get noise from noise schedule
            sigma, dsigma = logLinearNoise(t)

            # perturb the data
            x_perturb = perturb(x, sigma, mask_token)

            # # use bfloat16 precision for speed up 
            # with torch.autocast(device_type='cuda', dtype=torch.bfloat16):

            # get score
            log_score = dit(x_perturb, sigma, condition)

            # calculate loss 
            loss = score_entropy_loss(log_score, sigma.unsqueeze(-1), x_perturb, x, mask_token)
            loss = (dsigma.unsqueeze(-1) * loss).sum(dim=-1).mean()

            # adjustment for gradient accumulation 
            loss_scaled = loss / gradient_accumulation_steps

            loss_scaled.backward()

            # gradient cliping - helps to prevent unnecessary divergence 
            torch.nn.utils.clip_grad_norm_(dit.parameters(), max_norm=1.0)

            # # calculate max_grad_norm 
            # for p in model.parameters(): 
            #     grad_norm = p.grad.norm().item()
            #     if max_grad_norm < grad_norm:
            #         max_grad_norm = grad_norm
            #         print('max_grad_norm: ', max_grad_norm)

            if (train_step + 1) % gradient_accumulation_steps == 0:
                # gradient step
                optimizer.step()
                optimizer.zero_grad()

            if len(losses) == 0:
                loss_ema = loss.item()
            else:
                loss_ema = 0.999 * (losses[-1]) + 0.001 * loss.item()
            losses.append(loss_ema)

            pbar2.update(1)
            pbar2.set_description('loss: {:.2f}'.format(loss.item()))


            # sample
            if (train_step+1) % sampling_freq == 0: ## sample 
                
                # put model in eval mode to avoid dropout
                dit.eval()

                with torch.no_grad():

                    # get sample tokens corresponding to indices of codebook 
                    x_sample = get_sample(dit, seq_len, mask_token, vocab_size, num_sampling_steps, sample_batch_size, sample_condition, cfg_scale, device) # x_sample.shape: [b, seqlen]
                    x_sample = x_sample.flatten() # shape: [b * seqlen]

                    # get codebook vectors indexed by x_sample
                    sampled_img_latents = vqvae_model.codebook[x_sample] # sampled_img_latents.shape: [b, seqlen, latent_dim]
                    gen_img_patch_seq = vqvae_model.decode(sampled_img_latents.float())

                    # convert patch sequence to img 
                    gen_imgs = patch_seq_to_img(gen_img_patch_seq, patch_size, img_channels) # [b,c,h,w]

                    # save img

                    gen_imgs = (gen_imgs * 0.5 + 0.5).clamp(0,1)
                    grid = make_grid(gen_imgs, nrow=4)
                    save_image(grid, f"{results_dir}trainStep={train_step}_caption={sample_label}.png")

                # put model back to train mode 
                dit.train()


            if (train_step+1) % plot_freq == 0: ## save ckpt and plot losses

                # save ckpt 
                save_ckpt(device, ckpt_path, dit, optimizer)

                fig = plt.figure()
                plt.plot(losses)
                plt.title('final_loss:{:.3f}'.format(losses[-1]))
                fig.savefig(results_dir + 'loss_curve_iter_' + str(train_step) + '.png')

                losses = []

            train_step += 1

        epoch += 1
        pbar.update(1)
        pbar2.close()


    pbar.close()
