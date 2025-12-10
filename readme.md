# mw-samadi
Code of Fair PCA algorithm, introduced in the paper "The Price of Fair PCA: One Extra Dimension" by Samadi S, Tantipongpipat U, Morgenstern J, Singh M, and Vempala S. 32nd Conference on Neural Information Processing Systems (NIPS 2018). Please cite this paper (https://papers.nips.cc/paper/8294-the-price-of-fair-pca-one-extra-dimension.pdf) if you plan to use the code. 

All codes in this project are contributed and maintained by Samira Samadi and Uthaipon (Tao) Tantipongpipat. For questions, you may contact Samira or Tao at s.samadi@gmail.com or uthaipon@gmail.com.

The data sets used in this paper belong to 

-- Gary B. Huang, Manu Ramesh, Tamara Berg, and Erik Learned-Miller. Labeled faces in the wild: A database for studying face recognition in unconstrained environments. Technical Report 07-49, University of Massachusetts, Amherst, October 2007.

-- I-Cheng Yeh and Che-hui Lien. The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. Expert Systems with Applications, 36(2): 2473–2480, 2009.

The data folder required to run this code is available here https://drive.google.com/file/d/1oj6BwlPHZap4qYPGdGKObBzEXcPiuY9Y/view?usp=sharing 
## MM
The method is based on the algorithm proposed in the paper"Fair principal component analysis (PCA): minorization-maximization algorithms for Fair PCA, Fair Robust PCA and Fair Sparse PCA" by Prabhu, B., & 
Petre,S.,&Astha Saini (TMLR 2025)
# LocalCompositeNewton.jl

<!-- [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://GillesBareilles.github.io/LocalCompositeNewton.jl/stable) -->
<!-- [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://GillesBareilles.github.io/LocalCompositeNewton.jl/dev) -->
[![Build Status](https://github.com/GillesBareilles/LocalCompositeNewton.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/GillesBareilles/LocalCompositeNewton.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/GillesBareilles/LocalCompositeNewton.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/GillesBareilles/LocalCompositeNewton.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)

*Setup* is as follow:
```julia
using Pkg
Pkg.update()
Pkg.Registry.add(RegistrySpec(url = "https://github.com/GillesBareilles/OptimRegistry.jl"))
Pkg.add(url = "https://github.com/GillesBareilles/LocalCompositeNewton.jl", rev="master")
```
# Experiments
Experiments are executed with the commands:
```julia
using LocalCompositeNewton

# FPCA expriments
LocalCompositeNewton.fpca()

```
# Ackonnowledgements
We acknowledge and thank all original paper authors, dataset providers, and the developers of the LocalCompositeNewton.jl library. This implementation aims to facilitate research and application in the field of fair machine learning. Please adhere to academic norms and properly cite all related works when using this code.
