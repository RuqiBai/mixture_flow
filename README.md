# mixture_flow

This repo contains Ruqi's Final project for ECE 69500 Inference and Learning in Generative Models in Purdue.

## Introduction
Normalizing flows are bijective functions that transform an unknown distribution to a gaussian distribution. With these bijective transformations, we could presever the exact likelihood. So that we could 1) sampling new data by sampling the gaussian; 2) estimate the likelihood of an unknown sample. In this project, we explore the possibility that build a gaussian mixture model using a normalizing flow. We assume the data and labels are all observed.

## Approach
Suppose we have observed data multinomial(N, p) and p_Y(y), we could build a generative model: X -> Y. Our proposal is as follow: instead of transforming to one multivariate gaussian, we tranform our data Y to N gaussian depending on the correspoding label X. With this method, we could not only build a generative model which could sample from assigned label, but use bayes rule to estimate p(x|y) and apply maximum likelihood estimation to classify unseen samples. In this project, we choose CIFAR10 as our dataset. For transformation, we use invertible-Resnet(https://arxiv.org/abs/1811.00995) to build our flow model.

## Main Results
1) First, we random initialize all the parameters including the means \mu and covariates \sigma. With epochs of training, the accuracy freeze at around 20%. The main reason is that while minimizing the KL divergence between data distribution p(y|x) and model distribuition q(y|x), the N gaussian tends to overlap. See figure below:
![myplot0](https://user-images.githubusercontent.com/51713050/116952913-e3de5e00-ac59-11eb-815d-0604f441c4c7.png =250x)
![myplot3](https://user-images.githubusercontent.com/51713050/116952749-69add980-ac59-11eb-8604-304933d16555.png)
![myplot](https://user-images.githubusercontent.com/51713050/116952392-8dbceb00-ac58-11eb-87dd-2d28c4d139e8.png)
![myplot7](https://user-images.githubusercontent.com/51713050/116952764-76cac880-ac59-11eb-8bb6-48da5d1784bf.png)

## Installation
The code is running with env: pytorch 1.7, torchvision 0.8.2 , numpy 1.19.2 and matplotlib 3.3.2. (Not tested in other version.) 
## Evaluation
```
python main.py
```
