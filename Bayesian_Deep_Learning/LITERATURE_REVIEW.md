# LITERATURE REVIEW

In this document I briefly review the state of the art of Bayesian Deep Learning. Deep Learning models are very data hungry and prune to overfitting. Being Bayesian about a model prevents overfitting and allows to estimate uncertainty on the weights and on the prediction of new data. Exact inference is generally intractable. Approximate methods such as Markov chain Monte Carlo and Variational Inference are commonly used.

Bayesian neural networks is a topic with a long history. In the nineties, David Mackay did a Laplace approximation to the weights of a neural net, while Radford Neal introduced a Markov chain Monte Carlo method to efficiently sample from its probability distribution and Geoffrey Hinton performed variational inference on the weights of a neural network. These methods do not scale well to large datasets and they have not been used for some time. They require processing the whole dataset for a single iteration. In 2011, Max Welling proposed to add noise to the standard update of the weights when fitting the parameters of a neural network with stochastic gradient descent resulting in Bayesian Inference. It shows than optimization and Bayesian Inference are not that different. This allows learning from small minibatches and therefore scale better to large datasets. 

There has been a lot of work on Bayesian neural networks in the last few years. Some active research areas include scalable Markov chain Monte Carlo methods, which follow from the above mentioned work of Max Welling; and stochastic variational inference that scale traditional variational inference methods to larger datasets by using a noisy estimator obtained from a subsample of the dataset. Also in 2016, Gal and Ghaharamani published an intepretation of Dropout as approximate Bayesian Inference. The posterior distribution of a neural network is clearly multimodal and the use Dropout proposed by Gal and Ghaharamani allows to obtain samples from different modes.

Capturing the whole posterior distribution is hard. When using a multivariate Gaussian to approximate the posteorior distribution, which is usually the case, it is impossible to identify the probability mass of different modes. MCMC methods are not guaranteed to converge to the right probability distribution in a finite amount of time and the chain could be trapped in a single mode or a subset of them. Obtaining independent samples from the right posterior distribution is often not possible. However, the above mentioned approximations usually perform better than non-Bayesian approaches and they are therefore used.

REFERENCES:

@article{mackay1992practical,
  title={A practical Bayesian framework for backpropagation networks},
  author={MacKay, David JC},
  journal={Neural computation},
  volume={4},
  number={3},
  pages={448--472},
  year={1992},
  publisher={MIT Press}
}

@inproceedings{neal1993bayesian,
  title={Bayesian learning via stochastic dynamics},
  author={Neal, Radford M},
  booktitle={Advances in neural information processing systems},
  pages={475--482},
  year={1993}
}

@inproceedings{hinton1993keeping,
  title={Keeping the neural networks simple by minimizing the description length of the weights},
  author={Hinton, Geoffrey E and Van Camp, Drew},
  booktitle={Proceedings of the sixth annual conference on Computational learning theory},
  pages={5--13},
  year={1993},
  organization={ACM}
}

@inproceedings{welling2011bayesian,
  title={Bayesian learning via stochastic gradient Langevin dynamics},
  author={Welling, Max and Teh, Yee W},
  booktitle={Proceedings of the 28th International Conference on Machine Learning (ICML-11)},
  pages={681--688},
  year={2011}
}

@article{hoffman2013stochastic,
  title={Stochastic variational inference},
  author={Hoffman, Matthew D and Blei, David M and Wang, Chong and Paisley, John},
  journal={The Journal of Machine Learning Research},
  volume={14},
  number={1},
  pages={1303--1347},
  year={2013},
  publisher={JMLR. org}
}

@inproceedings{gal2016dropout,
  title={Dropout as a Bayesian approximation: Representing model uncertainty in deep learning},
  author={Gal, Yarin and Ghahramani, Zoubin},
  booktitle={international conference on machine learning},
  pages={1050--1059},
  year={2016}
}
