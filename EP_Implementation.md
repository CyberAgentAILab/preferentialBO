This note provides the details of the implementation of Gaussian process regression (GPR) for preferential learning using expectation propagation (EP).


We assume that $f$ is a sample path of GP with zero mean and some stationary kernel $k: \mathcal{X} \times \mathcal{X} \rightarrow \mathbb{R}$.
Following [2, 3], we consider that the duel is determined as follows:

$$
    \boldsymbol{x}\_{i, w} \succ \boldsymbol{x}\_{i, l}
    \Leftrightarrow
    f(\boldsymbol{x}\_{i, w}) + \epsilon_w > f(\boldsymbol{x}\_{i, l}) + \epsilon_l,
$$

where $\boldsymbol{x}\_{i, w}$ and $\boldsymbol{x}\_{i, l}$ are a winner and a loser of $i$-th duel, respectively, and i.i.d. additive noise $\epsilon_w$ and $\epsilon_l$ follow the normal distribution $\mathcal{N}(0, \sigma^2_{\rm noise})$.
Therefore, the training data $\mathcal{D}_t$ can be written as, 

$$
    \mathcal{D}_t = \\{ \boldsymbol{x}\_{i, w} \succ \boldsymbol{x}\_{i, l} \\}\_{i=1}^t \equiv \\{v_i < 0\\}\_{i=1}^t,
$$

where $v_i = f(\boldsymbol{x}\_{i, l}) + \epsilon_l - f(\boldsymbol{x}\_{i, w}) - \epsilon_w$.
For brevity, we denote $\\{v_i < 0\\}\_{i=1}^t$ as $\boldsymbol{v}_t < \boldsymbol{0}$, where $\boldsymbol{v}_t = (v_1, \dots, v_t)^\top$.
Then, we approximate the posterior 

$$
p\left(\boldsymbol{f}\_{\rm tes} \mid \boldsymbol{v}_t < \boldsymbol{0} \right) 
    = \frac{\Pr\left(\boldsymbol{v}_t < \boldsymbol{0} \mid \boldsymbol{f}\_{\rm tes}\right)  p\left(\boldsymbol{f}\_{\rm tes}\right)}{\Pr(\boldsymbol{v}_t < \boldsymbol{0})},
$$

where $\boldsymbol{f}\_{\rm tes} = \bigl( f(\boldsymbol{x}\_{1, {\rm tes}}), \dots, f(\boldsymbol{x}\_{m, {\rm tes}}) \bigr)^\top \in \mathbb{R}^m$ is an output vector on arbitrary inputs.



Prior work [2] apply EP to function values on the duels, i.e., $\bigl( f(\boldsymbol{x}\_{1, w}), f(\boldsymbol{x}\_{1, l}), \dots, f(\boldsymbol{x}\_{t, w}), f(\boldsymbol{x}\_{t, l}) \bigr)$.
However, this formulation requires two-dimensional local moment matching for $\bigl( f(\boldsymbol{x}\_{i, w}), f(\boldsymbol{x}\_{i, l}) \bigr)$.
Therefore, the implementation is highly cumbersome (See Appendix A in [2]).


Alternatively, we apply EP to $\boldsymbol{v}_t | \boldsymbol{v}_t < \boldsymbol{0}$ directly.
Then, since $\boldsymbol{v}_t | \boldsymbol{v}_t < \boldsymbol{0}$ follows a truncated multivariate normal distribution with a simple truncation, we can implement EP using one dimensional local moment matching, which is simpler than that of [2].
We have referred to Appendix B.2 of [4] for the subsequent derivation, although [4] contains some mathematical typos for the update of the mean parameters, which is fixed in the following derivation.


Let $\boldsymbol{v} = (v_1, \dots, v_t)^\top \sim \mathcal{N}(\boldsymbol{\mu}_0, \boldsymbol{\Sigma}_0)$ and we denote a probability density function (PDF) of the normal distribution as $\mathcal{N}(\boldsymbol{v} | \boldsymbol{\mu}_0, \boldsymbol{\Sigma}_0)$.
Then, we consider the EP approximation for the truncated normal distribution $p(\boldsymbol{v} | \boldsymbol{v} < \boldsymbol{0})$ as follows:

```math
\begin{align}
    p(\boldsymbol{v} | \boldsymbol{v} < \boldsymbol{0}) 
    &\propto \mathcal{N}(\boldsymbol{v} | \boldsymbol{\mu}_0, \boldsymbol{\Sigma}_0) \prod_{i=1}^t \mathbb{I}\\{ v_i < 0 \\} \\
    &\approx \mathcal{N}(\boldsymbol{v} | \boldsymbol{\mu}_0, \boldsymbol{\Sigma}_0) \prod_{i=1}^t \mathcal{N}( v_i | \tilde{\mu}_i, \tilde{\sigma}\_{i}^2 ) \\
    &= \mathcal{N}(\boldsymbol{v} | \boldsymbol{\mu}_0, \boldsymbol{\Sigma}_0) \mathcal{N}( \boldsymbol{v} | \tilde{\boldsymbol{\mu}}, \tilde{\boldsymbol{\Sigma}} ) \\
    &\propto \mathcal{N}(\boldsymbol{v} |  \boldsymbol{\mu}, \boldsymbol{\Sigma})
\end{align}
```

where $\mathbb{I}\\{ v_i < 0 \\}$ is an indicator function, which is $1$ if $v_i < 0$ and $0$ otherwise,  $\tilde{\boldsymbol{\mu}} \in \mathbb{R}^t = (\tilde{\mu}_1, \dots, \tilde{\mu}_t)$, $\tilde{\boldsymbol{\Sigma}} \in \mathbb{R}^{t \times t}$ is a diagonal matrix, whose $(i,i)$-th element is $\tilde{\sigma}_i^2$, and resulting mean $\boldsymbol{\mu}$ and covariance matrix $\boldsymbol{\Sigma}$ are computed as follows:

$$
\begin{align}
    \boldsymbol{\mu} &= \boldsymbol{\Sigma} \left( \tilde{\boldsymbol{\Sigma}}^{-1} \tilde{\boldsymbol{\mu}} + \boldsymbol{\Sigma}_0^{-1} \boldsymbol{\mu} \right) ,\\
    \boldsymbol{\Sigma} &= \left(\boldsymbol{\Sigma}_0^{-1} + \tilde{\boldsymbol{\Sigma}}^{-1} \right)^{-1}.
\end{align}
$$

Then, $\tilde{\boldsymbol{\mu}}$ and $\tilde{\boldsymbol{\Sigma}}$ are iteratively refined.
These parameters are initialized as $\tilde{\mu}_i = 0$ and $\tilde{\sigma}\_{i}^2 = \infty$ for all $i$, which corresponds to $\boldsymbol{\mu} = \boldsymbol{\mu}_0$ and $\boldsymbol{\Sigma} = \boldsymbol{\Sigma}_0$.


Let us consider the update of $\tilde{\mu}_j$ and $\tilde{\sigma}\_{j}^2$.
First, we consider the cavity distribution

```math
\begin{align}
    q_{\backslash j}(\boldsymbol{v}) 
    &= \mathcal{N}(\boldsymbol{v} | \bar{\boldsymbol{\mu}}\_{\backslash j}, \bar{\boldsymbol{\Sigma}}\_{\backslash j}) \\
    &\propto \mathcal{N}(\boldsymbol{v} | \boldsymbol{\mu}_0, \boldsymbol{\Sigma}_0) \prod_{i=1, i \neq j}^t \mathcal{N}( v_i | \tilde{\mu}_i, \tilde{\sigma}\_{i}^2 )  
,
\end{align}
```

where

```math
\begin{align}
    \bar{\mu}\_{\backslash j, j} 
    &=
    \mathbb{E}\_{q_{\backslash j}}[v_j] = \bar{\sigma}\_{\backslash j, j}^2 \left( \mu_j / \sigma_{j}^2 - \tilde{\mu}_j / \tilde{\sigma}\_{j}^2 \right) ,\\
    \bar{\sigma}\_{\backslash j, j}^2 
    &= 
    \mathbb{V}\_{q_{\backslash j}} [v_j] = \left( 1 / \sigma_{j}^2 - 1 / \tilde{\sigma}\_{j}^2 \right)^{-1},
\end{align}
```

where $\mathbb{E}$ and $\mathbb{V}$ are the expectation and variance with respect to the distribution in the subscript.
Then, $\tilde{\mu}_j$ and $\tilde{\sigma}\_{j}^2$ are updated by the moment matching between a truncated normal distribution 

$$
r_j(v_j) = 
\frac{\mathcal{N} \left( v_j | \bar{\mu}\_{\backslash j, j}, \bar{\sigma}\_{\backslash j, j}^2 \right) \mathbb{I}\\{ v_j < 0 \\}}{\Phi (\beta)},
$$ 

where $\beta = - \bar{\mu}\_{\backslash j, j} / \bar{\sigma}\_{\backslash j, j}$ and $\Phi ( \cdot )$ is a cumulative distribution function of the standard normal distribution, and the approximated normal distribution

$$
\begin{align}
    \hat{r}_j(v_j) 
    &= \mathcal{N} \left( v_j | \mathbb{E}\_{\hat{r}_j} \bigl[ v_j \bigr], \mathbb{V}\_{\hat{r}_j} \bigl[ v_j \bigr] \right) \\
    &\propto
    \mathcal{N} \left( v_j | \bar{\mu}\_{\backslash j, j}, \bar{\sigma}\_{\backslash j, j}^2 \right)
    \mathcal{N} \left( v_j | \tilde{\mu}_j, \tilde{\sigma}\_{j}^2 \right).
\end{align}
$$

Specifically, we set $\tilde{\mu}_j$ and $\tilde{\sigma}\_{j}^2$ using the following equations:

$$
\begin{align}
    \bar{\mu}\_{\backslash j, j} + \frac{\phi(\beta)}{\Phi(\beta)} \bar{\sigma}\_{\backslash j, j} 
    = \mathbb{E}\_{r_j}\bigl[v_j\bigr] 
    &= \mathbb{E}\_{\hat{r}_j} \bigl[ v_j \bigr]
    = \mathbb{V}\_{\hat{r}_j}\bigl[v_j\bigr] \left( \frac{\bar{\mu}\_{\backslash j, j}}{\bar{\sigma}\_{\backslash j, j}^2} + \frac{\tilde{\mu}_j}{\tilde{\sigma}\_{j}^2} \right),
    \\
     \left( 1 - \frac{\beta \phi(\beta)}{\Phi(\beta)} - \left( \frac{ \phi(\beta)}{\Phi(\beta)} \right)^2 \right) \bar{\sigma}\_{\backslash j, j}^2 
    = \mathbb{V}\_{r_j}\bigl[v_j\bigr] 
    &= \mathbb{V}\_{\hat{r}_j} \bigl[ v_j \bigr]
    = \left( \frac{1}{\bar{\sigma}\_{\backslash j, j}^2} + \frac{1}{\tilde{\sigma}\_{j}^2} \right)^{-1},
\end{align}
$$

where $\phi$ is PDF of the standard normal distribution.
Note that the expectation and variance of the truncated normal $r_j(v_j)$ and the normal $\hat{r}_j(v_j)$ can be obtained analytically.
Consequently, we obtain the update rule of $\tilde{\mu}_j$ and $\tilde{\sigma}\_{j}^2$  as follows:

$$
\begin{align}
    \tilde{\mu}_j &\leftarrow \bar{\mu}\_{\backslash j, j} + \left( \beta + \frac{ \phi(\beta)}{\Phi(\beta)} \right)^{-1} \bar{\sigma}\_{\backslash j, j}, \\
    \tilde{\sigma}\_{j}^2 &\leftarrow \left( \left( \frac{\beta \phi(\beta)}{\Phi(\beta)} + \left( \frac{ \phi(\beta)}{\Phi(\beta)} \right)^2 \right)^{-1} -1 \right) \bar{\sigma}\_{\backslash j, j}^2 .
\end{align}
$$

This update rule is repeated until convergence for all $j=1, \dots, t$.


The prediction of $\boldsymbol{f}\_{\rm tes}$ can be performed as follows:

$$
\begin{align}
    \boldsymbol{f}\_{\rm tes} | \mathcal{D}_t &\sim \mathcal{N}(\boldsymbol{\mu}\_{\rm tes}, \boldsymbol{\Sigma}\_{\rm tes}), \\
    \boldsymbol{\mu}\_{\rm tes} &= \boldsymbol{\Sigma}\_{{\rm tes}, \boldsymbol{v}} \left( \boldsymbol{\Sigma}_0 + \tilde{\boldsymbol{\Sigma}} \right)^{-1} \tilde{\boldsymbol{\mu}},\\
    \boldsymbol{\Sigma}\_{\rm tes} &= \boldsymbol{\Sigma}\_{{\rm tes}, {\rm tes}} - \boldsymbol{\Sigma}\_{{\rm tes}, \boldsymbol{v}} \left( \boldsymbol{\Sigma}_0 + \tilde{\boldsymbol{\Sigma}} \right)^{-1} \boldsymbol{\Sigma}\_{{\rm tes}, \boldsymbol{v}}^\top,
\end{align}
$$

where $\boldsymbol{\Sigma}\_{{\rm tes}, {\rm tes}}$ and $\boldsymbol{\Sigma}\_{{\rm tes}, \boldsymbol{v}}$ are the covariance matrices, whose definitions are shown in Proposition 3.1 in our paper [1].
Note that $\boldsymbol{\Sigma}_0$ corresponds to $\boldsymbol{\Sigma}\_{\boldsymbol{v}, \boldsymbol{v}}$ in our paper [1].

# Other formulation

We described the EP for the truncated multivariate normal distribution.
On the other hand, using $z_i = f(\boldsymbol{x}\_{i, l}) - f(\boldsymbol{x}\_{i, w})$, we can apply EP to the distribution of $\boldsymbol{z}_t = (z_1, \dots, z_t)^\top$:

```math
\begin{align}
    p(\boldsymbol{z}_t | \boldsymbol{v}_t < \boldsymbol{0}) 
    &\propto \mathcal{N}(\boldsymbol{z}_t | \boldsymbol{\mu}_0, \boldsymbol{\Sigma}_0 - 2\sigma_{\rm noise}\boldsymbol{I}) \prod_{i=1}^t \Pr( z_i + \epsilon_l - \epsilon_w < 0 | z_i) \\
    &= \mathcal{N}(\boldsymbol{z}_t | \boldsymbol{\mu}_0, \boldsymbol{\Sigma}_0 - 2\sigma_{\rm noise}\boldsymbol{I}) \prod_{i=1}^t \Phi\left( -\frac{z_i}{\sqrt{2}\sigma_{\rm noise}} \right) \\
\end{align}
```

where $\boldsymbol{I}$ is the identity matrix.
Then, the derivation of the EP procedure is slightly changed.
However, the derivation for this formulation is essentially same as the GP classification model [5].
See [5] for the detailed derivation.

# Reference

- [1]: [Takeno, S., Nomura, M., and Karasuyama, M., Towards Practical Preferential Bayesian Optimization with Skew Gaussian Processes, arXiv:2302.01513, 2023.](https://arxiv.org/abs/2302.01513) 
- [2]: [Chu, W. and Ghahramani, Z. Extensions of Gaussian processes for ranking: semisupervised and active learning. In Proceedings of the Learning to Rank workshop at the 18th Conference on Neural Information Processing Systems, pp. 29–34, 2005.](http://www.gatsby.ucl.ac.uk/~chuwei/paper/gprl.pdf)
- [3]: [Chu, W. and Ghahramani, Z. Preference learning with Gaussian processes. In Proceedings of the 22nd international conference on Machine learning, pp. 137–144, 2005.](https://icml.cc/Conferences/2005/proceedings/papers/018_Preference_ChuGhahramani.pdf)
- [4]: [Hernandez-Lobato, J. M., Hoffman, M. W., Ghahramani, Z., Predictive Entropy Search for Efficient Global Optimization of Black-box Functions, Advances in neural information processing systems 27, pp. 918-926, 2014.](https://papers.nips.cc/paper/2014/hash/069d3bb002acd8d7dd095917f9efe4cb-Abstract.html)
- [5]: [Rasmussen, C. E. and Williams, C. K. I. Gaussian Processes for Machine Learning (Adaptive Computation and Machine Learning). The MIT Press, 2005](https://gaussianprocess.org/gpml/)
