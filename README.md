# mixture_flow

This repo contains Ruqi's Final project for ECE 69500 Inference and Learning in Generative Models in Purdue.

## Introduction
Normalizing flows are bijective functions that transform an unknown distribution to a gaussian distribution. With these bijective transformations, we could presever the exact likelihood. So that we could 1) sampling new data by sampling the gaussian; 2) estimate the likelihood of an unknown sample. In this project, we explore the possibility that build a gaussian mixture model using a normalizing flow. We assume the data and labels are all observed.

## Approach
Suppose we have observed data multinomial(N, p) and p_Y(y), we could build a generative model: X -> Y. Our proposal is as follow: instead of transforming to one multivariate gaussian, we tranform our data Y to N gaussian depending on the correspoding label X. With this method, we could not only build a generative model which could sample from assigned label, but use bayes rule to estimate p(x|y) and apply maximum likelihood estimation to classify unseen samples. In this project, we choose CIFAR10 as our dataset. For transformation, we use invertible-Resnet(https://arxiv.org/abs/1811.00995) to build our flow model.

## Main Results
1) First, we random initialize all the parameters including the means \mu and covariates \sigma. With epochs of training, the accuracy freeze at around 20%. The main reason is that while minimizing the KL divergence between data distribution p(y|x) and model distribuition q(y|x), the N gaussian tends to overlap. See figure below (We random pick two dimension from the 3\*32\*32 dimension):
![myplot0](https://user-images.githubusercontent.com/51713050/116952913-e3de5e00-ac59-11eb-815d-0604f441c4c7.png)
![myplot3](https://user-images.githubusercontent.com/51713050/116952749-69add980-ac59-11eb-8604-304933d16555.png)
![myplot](https://user-images.githubusercontent.com/51713050/116952392-8dbceb00-ac58-11eb-87dd-2d28c4d139e8.png)
![myplot7](https://user-images.githubusercontent.com/51713050/116952764-76cac880-ac59-11eb-8bb6-48da5d1784bf.png)

2) Then, we try to initialize the means and covariates to make them separate to each other and then fix these value. The accuracy freeze at around 10%.
![myplot](https://user-images.githubusercontent.com/51713050/116953860-53554d00-ac5c-11eb-87b7-1f03cafb9e10.png)

3) Using the same hyperparameter to initialize mu and sigma, but make them learnable. With training, the gaussian starts to close to each other:
![myplot_fix](https://user-images.githubusercontent.com/51713050/116956277-e09ba000-ac62-11eb-9bc4-faa1a596fdc2.png)

4) Finally, we try to not only maximize the likelihood of p(x|y), but also minimize the likelihood of p(x|!y). With a tuning parameter, we could control the learning task between the maximization and minimization. If the tuning parameter is too large (=1), the optimization will focus on minimizing p(x|!y), which lead to the diverge. If we shrink the tuning parameter to (=0.1). With training, the gaussians learn to smaller their covariance. And if we tune the tuning parameter to make it even smaller (=0.01), the constraint doesn't help for the problem. 
![image](https://user-images.githubusercontent.com/51713050/116958211-ca441300-ac67-11eb-9d10-45acf2d6ec6c.png)

## Installation
The code is running with env: pytorch 1.7, torchvision 0.8.2 , numpy 1.19.2 and matplotlib 3.3.2. (Not tested in other version.) 
## Evaluation
```
python main.py
```
## Summary
I will continue to try to make this idea work. Currently, it seems that the learning is very sensitive to hyperparameters like the learning rate, tuning parameter and how to initialize the gaussian parameters. While my greedy search for these parameter, I will review more literature and think this problem mathematically.
