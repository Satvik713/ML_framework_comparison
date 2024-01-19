# ML_framework_comparison
In this project I am trying to compare the accuracies and run time of different algorithms written in different frameworks. For example- our first step is to write a mnist dataset classifier in different machine learning frameworks like Pytorch, tinygrad, tensorflow etc. Then we compare these framework on different metrics. Second step of the project is to understand how these frameworks leverage hardware to accelerate the processes and in the process understand why one of these frameworks takes less time for the same task than all the other frameworks. 

Currently, we have implemented a very simple neural network for classification of digits from mnist dataset. There are three cases- torch implementation of a neural network, tinygrad implementation of a neural network and another tinygrad implemenatation of a neural network but with JIT.

*What is JIT?*  
