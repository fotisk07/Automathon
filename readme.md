# Automathon 

[![License](https://img.shields.io/github/license/valentingol/cliconfig?color=999)](https://stringfixer.com/fr/MIT_license)

![PythonVersion](https://img.shields.io/badge/python-3.8%20%7E%203.10-informational)
![PytorchVersion](https://img.shields.io/badge/Pytorch-1.8%20%7E%201.12%20%7c%202.0-blue)
[![Torch_logo](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Wandb_logo](https://img.shields.io/badge/Weights_&_Biases-FFBE00?style=for-the-badge&logo=WeightsAndBiases&logoColor=white)](https://wandb.ai/site)

Below is my solution for the Automathon Superposed MNIST competition. 

Given an input image of two superimposed digits images from the MNIST dataset, the goal is to reconstruct those two images of the two digits.




## Installation
Install requirements for this repo just by running

```
pip install requirements.txt
```

## Model

The model used is a simple Autoencoder 

![picture 1](assets/cd25adae56e054d7dbd7aa74babe56972916683052bb6dc6a7ec779c990899f4.png)  

However, some tricks are being used to increase performance. 

The first thing to do is to use a custom loss function that takes into account the symmetry of the two images. In addition, the model only has to predict one image, since the other can simply be obtained by substracting the predicted one from the input one.


## Training 

I used the [cliconfig package](https://github.com/valentingol/cliconfig) by [valentingol](https://github.com/valentingol?tab=repositories) to keep track of my configurations. 

I also used Weights and Biases to keep track of my training metrics.

![picture 2](assets/eb011d62ab0df8c07ca0a726e0dcf5ad4f2700d9959760cd8a48636b6852aaee.png)  





 