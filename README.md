# Vit

This is a simple program for training and testing vit.
Key requirements: torch, torchvision and timm.


## Dataset

I put 5 categories of the cub classification data set for simple training.
You can train on your dataset by setting file directory with the same structure standard.


## Train

The num-worker is set to zero for using cpu and I suggest you increase the number when switching to gpu.

## Test

I put 5 pictures and one checkpoint  for testing the model, you should change the class-dict when you use your own dataset.


