import torch

CUDA = torch.cuda.is_available()
# CUDA = False

if CUDA:
    torch.cuda._lazy_init()

Tensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor


