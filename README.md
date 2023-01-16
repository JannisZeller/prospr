# prosper

This repo contains my attempt to implement ProsPr (neural network pruning) with tensorflow. I mainly follow the maths described by [Alizadeh et al. (2022)](https://arxiv.org/pdf/2202.08132.pdf). I do neither have the resources nor the time to do benchmarks on the datasets and networks mentioned in the paper, instead I stick to the MNIST dataset and use a simple Dense-Network and LeNet-5 [LeCun et al. (1998)](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) for classification. I reuse code from [Géron (2019)](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/).

Note that I implement an alternative version to the algorithm presented by [Alizadeh et al. (2022, p. 5)](https://arxiv.org/pdf/2202.08132.pdf). They first perform a "prune loop" (Algorithm 1, lines 3 to 8) containing $M$ steps iterating through batches $m=1,...,M$ and aferwards calculate the meta gradient $$g = \frac{\partial}{\partial c} \mathcal L(\mathbf{w}_M) \, .$$
Therefore it is necessairy to keep track of all variables and itermediate steps $1,..., M$. Instead, I exploit the structure of equation (10) on page 4, i. e.: 
```math
g = \frac{\partial}{\partial c} \mathcal L(\mathbf{w}_M) = \frac{\partial \mathcal L(\mathbf{w}_M)}{\partial \mathbf{w}_M} \Bigg[ \prod_{m=1}^M \underbrace{\frac{\partial \mathbf{w}_m}{\partial \mathbf{w}_{m-1}}}_{\gamma_m} \Bigg] \cdot \mathbf{w}_{\textsf{init}} \, .
```
Within the "prune loop" at after each batch update, i. e. after setting 
```math
\mathbf{w}_{m+1} = \mathbf w_m - \alpha \frac{\partial \mathcal L(\mathbf{w}_m, \mathcal D_m)}{\partial \mathbf{w}_m} \, ,
```
I directly calculate the gradient $\gamma_m$ and therefore keep track only of the product (or the sum, using the logarithm-trick to prevent underflow). This means I use autodiff to calculate 
```math
\gamma_m = \mathbf{1} - \alpha \frac{\partial^2 \mathcal L(\mathbf{w}_m, \mathcal D_m)}{\partial \mathbf{w}_m^2} \, .
```
With this approach I do not encounter issues with collapsing or exploding gradients.

I hope that my implementation is correct, but I do not have the possibility to check it thoroughly. At least I definitely beat a random-pruning approach.