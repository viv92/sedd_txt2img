# sedd_txt2img

A text-to-image model based on discrete diffusion.


Most of the current text-to-image models model the PDF of continuous image data using a continuous diffusion process defined by a SDE or its corresponding probability flow ODE.


This is an implementation of a text-to-image model that models the PMF of discretized image tokens using [Score Entropy Discrete Diffusion](https://arxiv.org/abs/2310.16834).
We use [Finite Scalar Quantization](https://arxiv.org/abs/2309.15505) for quantizing images into discrete tokens.

## Acknowledgements

This repository builds heavily off of [SEDD](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion)
