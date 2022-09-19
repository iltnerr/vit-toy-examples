print("Script based on https://towardsdatascience.com/implementing-visualttransformer-in-pytorch-184f9f16f632")
print("This script is not meant to be used as a solution, but rather as a working example to comprehensively examine building blocks of a vision transformer implemenation and how to train it.")

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn, Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary

from building_blocks import PatchEmbedding


##### Hyperparameters ######################################################################################

patch_size = 16

##### Data #################################################################################################

img = Image.open('../data/cat.jpg')

fig = plt.figure()
plt.imshow(img)

# resize to imagenet size
transform = Compose([Resize((224, 224)), ToTensor()])
x = transform(img)
x = x.unsqueeze(0) # add batch dim
print("x.shape: ", x.shape)

##### Patch Embeddings #####################################################################################

patches = rearrange(x, 'b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size)

print("Patch Embedding shape: ", PatchEmbedding()(x).shape)