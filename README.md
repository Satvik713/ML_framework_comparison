# ML_framework_comparison
In this project I am trying to compare the accuracies and run time of different algorithms written in different frameworks. For example- our first step is to write a mnist dataset classifier in different machine learning frameworks like Pytorch, tinygrad, tensorflow etc. Then we compare these framework on different metrics. Second step of the project is to understand how these frameworks leverage hardware to accelerate the processes and in the process understand why one of these frameworks takes less time for the same task than all the other frameworks. 

Currently, we have implemented a very simple neural network for classification of digits from mnist dataset. There are three cases- torch implementation of a neural network, tinygrad implementation of a neural network and another tinygrad implemenatation of a neural network but with JIT.

*What is JIT?*  

**current stats of a very simple neural network for MNIST classification** -
1. mnist_torch.py - accuracy = 0.844, Time for test set evaluation - 344 ms
2. nn.py - accuracy = 0.844, Time for test set evaluation - 2046.22 ms
3. nn_with_jit.py - accuracy = 0.922, Time for test set evaluation - 1360.21 ms


*Next step is to compare these frameworks on Convolutional neural networks (CNNs)*
