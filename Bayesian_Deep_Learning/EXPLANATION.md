# EXPLANATION

In this folder I have implemented a bayesian approach for the weights of the last layer of a Deep Learning model. Taking a Bayesian approach to the weights of a model reduces the risk of overfitting and could therefore improve the accuracy of the predictions. In order to obtain the weights, I have sampled from their probability distribution with the package emcee (MCMC).

I have decided to explore a Bayesian approach only for the weights of the last layer due to the next reasons:
1) The original model has hundreds of thousands of parameters. The last layer has 2010 parameters. Running a MCMC in a lower dimensional space is computationally much cheaper.
2) The parameters of the last layer are fast. Evaluating the likelihood of a set of parameters when only the weights of the last layer are perturbed allows to reuse some computation (the matrix operations and non-linearities of all other layers).
3) Since only the weights of the last layer are modified, this is equivalent to a softmax regression problem (linear model). Therefore, the cost function is CONVEX. This is important for the MCMC chain to explore all the probability distribution without forgetting about some modes.

In order to run this experiment, the next scripts are run:
1) First, the model's parameters are fitted in a standard manner (non-bayesian). This is done with the model_training.py script.
2) Second, the features_extraction.py script will use the weights fitted to extract the hidden representation (hidden units of the last layer) for each data item of the training and validation sets. We will face the problem as a linear model from now onwards. 
3) Third (optional), we can use the prediction_checking.py script to validate that the performance of the new linear model is the same as the original Deep Learning model as expected (accuracy around 62%).
4) Then, in the Bayesian_inference.py script we will use the emcee package to obtain samples from the probability distribution of the weights according to the the log-likelihood of the data (we use a flat prior, assuming nothing about the weights).
5) Finally, in the Bayesian_prediction.py script we will average the predictions obtained with these samples to estimate the genre of each song. 

Results:
The performance of this approach is similar to the standard model (non-bayesian). The Bayesian approach is used to reduce both overfitting and underfitting. It assumes that the model fitted is sensible but there is not enough data to constrain it. If this is not the case, the averaged predictions will be close to the one of a single fit, which is our case. 
