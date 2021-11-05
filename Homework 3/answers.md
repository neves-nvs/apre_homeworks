# 2

    Confusion matrix
    [[419  25]
    [ 18 221]]
    Confusion matrix - Early Stopping
    [[389  55]
    [  5 234]]

2 reason for the only slight observed differences are:

 - The use of k-fold cross validation where the model is repeatedly reffited on parts of the dataset;
 
 - Early stopping being meant to stop a single model when it starts having increased generalized error;
 
 These make them not very suited to be used together
 

# 3

4 strategies that can be used to minimize the observed error of the multi-layer perceptron regressor are:
 - increase the training sample;

 - early stopping: prevent overfitting by stopping the training when the testing error rate starts increasing;

 - change the complexity of the network structure and parameters by adding/removing nodes and ensuring the weights keep small, since this indicate a less complex model and therefore more stable and less prone to error from outliers in the input

 - noise: Add noise to the input statistically.
