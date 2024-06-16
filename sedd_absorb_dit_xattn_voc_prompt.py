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
    # gen_img = gen_img[:, :, ::-1]
    cv2.imwrite(save_path, gen_img)
        


### main
if __name__ == '__main__':
    # hyperparams for vqvae (FSQ_Transformer)
    num_quantized_values = [7, 5, 5, 5, 5] # L in fsq paper
    latent_dim = len(num_quantized_values)
    img_size = 128 # voc 
    img_channels = 3 
    img_shape = torch.tensor([img_channels, img_size, img_size])
    resize_shape = (img_size, img_size)
    img_latent_dim = latent_dim # as used in the pretrained VQVAE 

    patch_size = 16 # necessary that img_size % patch_size == 0
    assert img_size % patch_size == 0
    patch_dim = img_channels * (patch_size**2)
    seq_len = (img_size // patch_size) ** 2 # equal to num latents per item
    
    # hyperparams for FSQ Transformer
    d_model_fsq = 768 # patch_dim * 1 - should match d_model_t5
    n_heads_fsq = 8
    assert d_model_fsq % n_heads_fsq == 0
    d_k_fsq = d_model_fsq / n_heads_fsq 
    d_v_fsq = d_k_fsq 
    n_layers_fsq = 6
    d_ff_fsq = d_model_fsq * 4
    dropout_fsq = 0.1

    # hyperparams for T5 (T5 decoder implements the consistency model backbone)
    d_model_t5 = 768 # d_model for T5 (required for image latents projection)
    max_seq_len_t5 = 512 # required to init T5 Tokenizer
    # dropout = 0. # TODO: check if we can set the dropout in T5 decoder

    # hyperparams for sedd (dit)
    d_model = 1024
    n_layers = 6
    n_heads = 2 # might be better to keep this low when modeling image token sequences (since small seqlen)
    d_k = d_model // n_heads
    d_v = d_k 
    d_ff = d_model * 4
    dropout = 0.1

    # get vocab size and mask token 
    vocab_size = 1
    for n in num_quantized_values:
        vocab_size *= n 
    mask_token = vocab_size - 1

    # pad vocab to increase vocab size to a nice number 
    rem = vocab_size % 256 
    vocab_size += rem     

    # hyperparams for sampling
    num_prompts = 100 
    num_sampling_steps = 1024
    sample_batch_size = 1
    p_uncond = 0.1 
    cfg_scale = 1.5
    diffusion_start_time_eps = 1e-3
    random_seed = 10

    results_dir = './prompt_voc_xattn_rotaryDisabledXattn/'
    ckpts_dir = './ckpts/'
    ckpt_path = ckpts_dir + 'voc_dit_xattn_rotaryDisabledXattn.pt'
    resume_training_from_ckpt = True    

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    if not os.path.exists(ckpts_dir):
        os.makedirs(ckpts_dir)

    # vqvae ckpt path 
    vqvae_ckpt_path = '/home/vivswan/experiments/fsq/ckpts/FSQ_Transformer|img_size:128|patch_size:16|patch_dim:768|seq_len:64|d_model:768|n_heads:8|dropout:0.1|batch_size:256|lr:0.0003.pt' # path to pretrained vqvae 

    # t5 model (for encoding captions) 
    t5_model_name = 't5-base'

    # set random seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)

    # device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # load pretrained VQVAE in eval mode 
    # init transformer encoder
    encoder_transformer = init_transformer(patch_dim, seq_len, d_model_fsq, d_k_fsq, d_v_fsq, n_heads_fsq, n_layers_fsq, d_ff_fsq, dropout_fsq, latent_dim, device)
    # init transformer decoder
    decoder_transformer = init_transformer(latent_dim, seq_len, d_model_fsq, d_k_fsq, d_v_fsq, n_heads_fsq, n_layers_fsq, d_ff_fsq, dropout_fsq, patch_dim, device)
    # init FSQ_Transformer 
    vqvae_model = FSQ_Transformer(device, num_quantized_values, encoder_transformer, decoder_transformer, seq_len).to(device)
    vqvae_model = load_ckpt(vqvae_ckpt_path, vqvae_model, device=device, mode='eval')
    
    # init T5 tokenizer and transformer model
    t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name, model_max_length=max_seq_len_t5)
    t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name).to(device)
    # delete t5_decoder to save ram 
    del t5_model.decoder 

    # init dit
    x_seq_len = seq_len 
    # condition_seq_len = 1 # just the class label
    max_seq_len = seq_len + 1 # [t, x]
    # x_dim = 1
    condition_dim = d_model_t5
    dit = init_dit(max_seq_len, x_seq_len, d_model, condition_dim, vocab_size, d_k, d_v, n_heads, n_layers, d_ff, dropout, device).to(device)

    # freeze vqvae, t5_encoder and ema_net
    freeze(vqvae_model)
    freeze(t5_model.encoder)

    # load ckpt
    dit = load_ckpt(ckpt_path, dit, device=device, mode='eval')

    # # torch compile model for speed up TODO: unable to load ckpt if using torch.compile
    # dit = torch.compile(dit)

    # loop
    for ep in (pbar := tqdm(range(num_prompts))):

        # input prompt 
        prompt = input('Enter prompt: ')

        # convert prompt to tokens 
        cap_tokens_dict = t5_tokenizer([prompt], return_tensors='pt', padding=True, truncation=True)
        cap_tokens_dict = cap_tokens_dict.to(device)


        with torch.no_grad():

            # extract cap tokens and attn_mask from cap_tokens_dict
            cap_tokens, cap_attn_mask = cap_tokens_dict.input_ids, cap_tokens_dict.attention_mask
            # feed cap_tokens to t5 encoder to get encoder output
            enc_out = t5_model.encoder(input_ids=cap_tokens, attention_mask=cap_attn_mask).last_hidden_state # enc_out.shape: [batch_size, cap_seqlen, d_model_t5]

            # get sample tokens corresponding to indices of codebook 
            x_sample = get_sample(dit, seq_len, mask_token, vocab_size, num_sampling_steps, sample_batch_size, enc_out, cfg_scale, device) # x_sample.shape: [b, seqlen]
            x_sample = x_sample.flatten() # shape: [b * seqlen]

            # get codebook vectors indexed by x_sample
            sampled_img_latents = vqvae_model.codebook[x_sample] # sampled_img_latents.shape: [b, seqlen, latent_dim]
            gen_img_patch_seq = vqvae_model.decode(sampled_img_latents.float())

            # convert patch sequence to img 
            gen_imgs = patch_seq_to_img(gen_img_patch_seq, patch_size, img_channels) # [b,c,h,w]

            # save generated img
            gen_imgs = (gen_imgs * 0.5 + 0.5).clamp(0,1)
            # bgr to rgb 
            gen_imgs = torch.flip(gen_imgs, dims=(1,))
            grid = make_grid(gen_imgs, nrow=2)
            save_image(grid, f"{results_dir}_caption={prompt}.png")
            


        
