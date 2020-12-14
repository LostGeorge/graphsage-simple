# Reference PyTorch GraphSAGE Implementation
### Author: William L. Hamilton


Basic reference PyTorch implementation of [GraphSAGE](https://github.com/williamleif/GraphSAGE).
This reference implementation is not as fast as the TensorFlow version for large graphs, but the code is easier to read and it performs better (in terms of speed) on small-graph benchmarks.
The code is also intended to be simpler, more extensible, and easier to work with than the TensorFlow version.

Currently, only supervised versions of GraphSAGE-mean and GraphSAGE-GCN are implemented. 

#### Requirements

pytorch >0.2 is required.

#### Running examples

Execute `python -m graphsage.model` to run the Cora example.
It assumes that CUDA is not being used, but modifying the run functions in `model.py` in the obvious way can change this.
There is also a pubmed example (called via the `run_pubmed` function in model.py).

## BATS Exercises - George

### Exercise 1
Reading though Inductive Representation Learning on Large Graphs by Hamilton et al., I can best summarize the GraphSAGE framework
as essentially an extensible methodology to train feature representations on a graph inductively, or without needing
to see the graph structure of the test data. This is opposed to transductive frameworks for graph algorithms,
which require both the training and test data to be present, using the training labels and all the input features
to predict the testing labels. The extensibility of the inductive framework and importance of getting good low dimensional
feature representations of nodes is exemplified in the three different varying test applications, where GraphSAGE has remarkable
performance improvements.

GraphSAGE is similar to Graph Convolutional Network (GCN), which uses alternating mean aggregation
and linear neural network layers to generate a lower dimensional representation of a graph, but 
GraphSAGE is unique in it's flexibility in terms of being able to be applied to an unsupervised
setting, and the flexibility in choosing the aggregator and sampling neighbors. This power of this
framework is especially exemplified in the ability for GraphSAGE to capture local graph structure (in terms of clustering coefficients) within
its output feature representations, despite the algorithm only accounting for neighors through sampling.

### Exercise 2
- With a 2-layer graph neural network and random seed 1, the validation F1-Score is 0.838.
- With a 3-layer graph neutral network and random seed 1, the 
validation F1-Score is 0.872.

For these, make sure `MeanAggregator` is uncommented and `MaxPoolAggregator` is commented out. To run 3-layer, do `python3 -m graphsage.model 3`. These are also with the provided learning rate 0.7.

### Exercise 3
The implmentation of `MaxPoolAggregator` can be found in `aggregators.py`. For this, I assumed that GCN would always be true and CUDA false for simplicity reasons.

I am fairly sure my implementation is at least close to correct, but my results were fairly poor. I get a validation F1-Score of 0.34 for the 2 layer neural network with the max pool aggregator and learning_rate 0.05 (had to change this to prevent nan).
Some of the considerations I made:
- I initialized W_pool as glorot uniform, and used the default initialization for the bias b.
- I had to use a sigmoid (or any similar logistic) activation function, since when I tried standard ReLU or Leaky ReLU, my loss would always be nan (like 99% of the time) for any choice of learning rate. I think this is due to either the loss being too close to zero or unbounded above, so maybe batch normalization here would have helped.
    - Something that's weird in this implementation is that the algorithm in the paper does have normalization at the end of each encoder, but that is not implemented here.
    - So, I added normalization at the end of the encoder myself, and the results were better. Now with leaky ReLU as the activation function and learning rate 0.01 with Adam optimizer, the 2 layer neural network with MaxPoolAggregator achieves F1-Score of 0.808. Still worse than MeanAggregator, but I suspect that the added complexity here makes it overfit.
    - I still do get nan as the loss occasionally, to which I am unsure why.
