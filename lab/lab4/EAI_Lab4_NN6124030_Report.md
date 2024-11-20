<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
<script type="text/x-mathjax-config"> MathJax.Hub.Config({ tex2jax: {inlineMath: [['$', '$']]}, messageStyle: "none" });</script>

# Lab 4 - Model Pruning

## Sparsity Training Accuracy over Epochs

![](images/sparse_train_acc_0.png)

![](images/sparse_train_acc_1.png)

![](images/sparse_train_acc_2.png)

## Sparsity Regularization

### Scaling Factor Distribution of $\lambda = 0$

![](images/bn_scale_dist_0.png)

### Scaling Factor Distribution of $\lambda = 10^{-5}$

![](images/bn_scale_dist_1.png)

### Scaling Factor Distribution of $\lambda = 10^{-4}$

![](images/bn_scale_dist_2.png)

## Test Accuracy with Prune Ratio 50%

Take $\lambda = 10^{-4}$ for example

## Test Accuracy with Prune Ratio 90%

Take $\lambda = 10^{-4}$ for example

## Fine Tuning Accuracy over Epochs

Take $\lambda = 10^{-4}$ and prune ratio 90% for example

## Challenge and Solution
