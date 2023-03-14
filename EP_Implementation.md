This note provides the details of the implementation of Gaussian process regression (GPR) for preferential learning using expectation propagation (EP).


We assume that $f$ is a sample path of GP with zero mean and some stationary kernel $k: \mathcal{X} \times \mathcal{X} \rightarrow \mathbb{R}$.
Following [2, 3], we consider that the duel is determined as follows:
$$
    \bm{x}_{i, w} \succ \bm{x}_{i, l}
    \Leftrightarrow
    f(\bm{x}_{i, w}) + \epsilon_w > f(\bm{x}_{i, l}) + \epsilon_l,
$$
where $\bm{x}_{i, w}$ and $\bm{x}_{i, l}$ are a winner and a loser of $i$-th duel, respectively, and i.i.d. additive noise $\epsilon_w$ and $\epsilon_l$ follow the normal distribution $\mathcal{N}(0, \sigma^2_{\rm noise})$.
Therefore, the training data $\mathcal{D}_t$ can be written as, 
$$
    \mathcal{D}_t = \{ \bm{x}_{i, w} \succ \bm{x}_{i, l} \}_{i=1}^t \equiv \{v_i < 0\}_{i=1}^t,
$$
where $v_i \coloneqq f(\bm{x}_{i, l}) + \epsilon_l - f(\bm{x}_{i, w}) - \epsilon_w$.
For brevity, we denote $\{v_i < 0\}_{i=1}^t$ as $\bm{v}_t < \bm{0}$, where $\bm{v}_t \coloneqq (v_1, \dots, v_t)^\top$.
Then, we approximate the posterior 
$$
p\left(\bm{f}_{\rm tes} \mid \bm{v}_t < \bm{0} \right) 
    = \frac{\Pr\left(\bm{v}_t < \bm{0} \mid \bm{f}_{\rm tes}\right)  p\left(\bm{f}_{\rm tes}\right)}{\Pr(\bm{v}_t < \bm{0})},
$$
where $\bm{f}_{\rm tes} \coloneqq \bigl( f(\bm{x}_{1, {\rm tes}}), \dots, f(\bm{x}_{m, {\rm tes}}) \bigr)^\top \in \mathbb{R}^m$ is an output vector on arbitrary inputs.



Prior work [2] apply EP to function values on the duels, i.e., $\bigl( f(\bm{x}_{1, w}), f(\bm{x}_{1, l}), \dots, f(\bm{x}_{t, w}), f(\bm{x}_{t, l}) \bigr)$.
However, this formulation requires two-dimensional local moment matching for $\bigl( f(\bm{x}_{i, w}), f(\bm{x}_{i, l}) \bigr)$.
Therefore, the implementation is highly cumbersome (See Appendix A in [2]).


Alternatively, we apply EP to $\bm{v}_t | \bm{v}_t < \bm{0}$ directly.
Then, since $\bm{v}_t | \bm{v}_t < \bm{0}$ follows a truncated multivariate normal distribution with a simple truncation, we can implement EP using one dimensional local moment matching, which is simpler than that of [2].
We have referred to Appendix B.2 of [4] for the subsequent derivation, although [4] contains some mathematical typos for the update of the mean parameters, which is fixed in the following derivation.


Let $\bm{v} \coloneqq (v_1, \dots, v_t)^\top \sim \mathcal{N}(\bm{\mu}_0, \bm{\Sigma}_0)$ and we denote a probability density function (PDF) of the normal distribution as $\mathcal{N}(\bm{v} | \bm{\mu}_0, \bm{\Sigma}_0)$.
Then, we consider the EP approximation for the truncated normal distribution $p(\bm{v} | \bm{v} < \bm{0})$ as follows:
$$
\begin{aligned}
    p(\bm{v} | \bm{v} < \bm{0}) 
    &\propto \mathcal{N}(\bm{v} | \bm{\mu}_0, \bm{\Sigma}_0) \prod_{i=1}^t \mathbb{I}\{ v_i < 0 \} \\
    &\approx \mathcal{N}(\bm{v} | \bm{\mu}_0, \bm{\Sigma}_0) \prod_{i=1}^t \mathcal{N}( v_i | \tilde{\mu}_i, \tilde{\sigma}_{i}^2 ) \\
    &= \mathcal{N}(\bm{v} | \bm{\mu}_0, \bm{\Sigma}_0) \mathcal{N}( \bm{v} | \tilde{\bm{\mu}}, \tilde{\bm{\Sigma}} ) \\
    &\propto \mathcal{N}(\bm{v} |  \bm{\mu}, \bm{\Sigma})
\end{aligned}
$$
where $\mathbb{I}\{ v_i < 0 \}$ is an indicator function, which is $1$ if $v_i < 0$ and $0$ otherwise,  $\tilde{\bm{\mu}} \in \mathbb{R}^t \coloneqq (\tilde{\mu}_1, \dots, \tilde{\mu}_t)$, $\tilde{\bm{\Sigma}} \in \mathbb{R}^{t \times t}$ is a diagonal matrix, whose $(i,i)$-th element is $\tilde{\sigma}_i^2$, and resulting mean $\bm{\mu}$ and covariance matrix $\bm{\Sigma}$ are computed as follows:
$$
\begin{aligned}
    \bm{\mu} &= \bm{\Sigma} \left( \tilde{\bm{\Sigma}}^{-1} \tilde{\bm{\mu}} + \bm{\Sigma}_0^{-1} \bm{\mu} \right) ,\\
    \bm{\Sigma} &= \left(\bm{\Sigma}_0^{-1} + \tilde{\bm{\Sigma}}^{-1} \right)^{-1}.
\end{aligned}
$$
Then, $\tilde{\bm{\mu}}$ and $\tilde{\bm{\Sigma}}$ are iteratively refined.
These parameters are initialized as $\tilde{\mu}_i = 0$ and $\tilde{\sigma}_{i}^2 = \infty$ for all $i$, which corresponds to $\bm{\mu} = \bm{\mu}_0$ and $\bm{\Sigma} = \bm{\Sigma}_0$.


Let us consider the update of $\tilde{\mu}_j$ and $\tilde{\sigma}_{j}^2$.
First, we consider the cavity distribution
$$
\begin{aligned}
    q_{\backslash j}(\bm{v}) 
    &= \mathcal{N}(\bm{v} | \bm{\mu}_0, \bm{\Sigma}_0) \prod_{i=1, i \neq j}^t \mathcal{N}( v_i | \tilde{\mu}_i, \tilde{\sigma}_{i}^2 ) \\
    &= \mathcal{N}(\bm{v} | \bar{\bm{\mu}}_{\backslash j}, \bar{\bm{\Sigma}}_{\backslash j}),
\end{aligned}
$$
where
$$
\begin{aligned}
    \bar{\mu}_{\backslash j, j} 
    &\coloneqq
    \mathbb{E}_{q_{\backslash j}}[v_j] = \bar{\sigma}_{\backslash j, j}^2 \left( \mu_j / \sigma_{j}^2 - \tilde{\mu}_j / \tilde{\sigma}_{j}^2 \right) ,\\
    \bar{\sigma}_{\backslash j, j}^2 
    &\coloneqq 
    \mathbb{V}_{q_{\backslash j}} [v_j] = \left( 1 / \sigma_{j}^2 - 1 / \tilde{\sigma}_{j}^2 \right)^{-1},
\end{aligned}
$$
where $\mathbb{E}$ and $\mathbb{V}$ are the expectation and variance with respect to the distribution in the subscript.
Then, $\tilde{\mu}_j$ and $\tilde{\sigma}_{j}^2$ are updated by the moment matching of a truncated normal distribution 
$$
r_j(v_j) = 
\frac{\mathcal{N} \left( v_j | \bar{\mu}_{\backslash j, j}, \bar{\sigma}_{\backslash j, j}^2 \right) \mathbb{I}\{ v_j < 0 \}}{\Phi (\beta)},
$$ 
where $\beta = - \bar{\mu}_{\backslash j, j} / \bar{\sigma}_{\backslash j, j}$ and $\Phi ( \cdot )$ is a cumulative distribution function of the standard normal distribution, and the approximated normal distribution
$$
\hat{r}_j(v_j) \propto
\mathcal{N} \left( v_j | \bar{\mu}_{\backslash j, j}, \bar{\sigma}_{\backslash j, j}^2 \right)
\mathcal{N} \left( v_j | \tilde{\mu}_j, \tilde{\sigma}_{j}^2 \right).
$$
Specifically, we set $\tilde{\mu}_j$ and $\tilde{\sigma}_{j}^2$ using the following equations:
$$
\begin{aligned}
    \bar{\mu}_{\backslash j, j} + \frac{\phi(\beta)}{\Phi(\beta)} \bar{\sigma}_{\backslash j, j} 
    = \mathbb{E}_{r_j}\bigl[v_j\bigr] 
    &= \mathbb{E}_{\hat{r}_j} \bigl[ v_j \bigr]
    = \mathbb{V}_{\hat{r}_j}\bigl[v_j\bigr] \left( \frac{\bar{\mu}_{\backslash j, j}}{\bar{\sigma}_{\backslash j, j}^2} + \frac{\tilde{\mu}_j}{\tilde{\sigma}_{j}^2} \right),
    \\
     \left( 1 - \frac{\beta \phi(\beta)}{\Phi(\beta)} - \left( \frac{ \phi(\beta)}{\Phi(\beta)} \right)^2 \right) \bar{\sigma}_{\backslash j, j}^2 
    = \mathbb{V}_{r_j}\bigl[v_j\bigr] 
    &= \mathbb{V}_{\hat{r}_j} \bigl[ v_j \bigr]
    = \left( \frac{1}{\bar{\sigma}_{\backslash j, j}^2} + \frac{1}{\tilde{\sigma}_{j}^2} \right)^{-1},
\end{aligned}
$$
where $\phi$ is PDF of the standard normal distribution.
Note that the expectation and variance of the truncated normal $r_j(v_j)$ and the normal $\hat{r}_j(v_j)$ can be obtained analytically.
Consequently, we obtain the update rule of $\tilde{\mu}_j$ and $\tilde{\sigma}_{j}^2$  as follows:
$$
\begin{aligned}
    \tilde{\mu}_j &\leftarrow \bar{\mu}_{\backslash j, j} + \left( \beta + \frac{ \phi(\beta)}{\Phi(\beta)} \right)^{-1} \bar{\sigma}_{\backslash j, j}, \\
    \tilde{\sigma}_{j}^2 &\leftarrow \left( \left( \frac{\beta \phi(\beta)}{\Phi(\beta)} + \left( \frac{ \phi(\beta)}{\Phi(\beta)} \right)^2 \right)^{-1} -1 \right) \bar{\sigma}_{\backslash j, j}^2 .
\end{aligned}
$$
This update rule is repeated until convergence for all $j=1, \dots, t$.


The prediction of $\bm{f}_{\rm tes}$ can be performed as follows:
$$
\begin{aligned}
    \bm{f}_{\rm tes} | \mathcal{D}_t &\sim \mathcal{N}(\bm{\mu}_{\rm tes}, \bm{\Sigma}_{\rm tes}), \\
    \bm{\mu}_{\rm tes} &= \bm{k}_{\bm{v}, {\rm tes}}^\top \left( \bm{\Sigma}_0 + \tilde{\bm{\Sigma}} \right)^{-1} \tilde{\bm{\mu}},\\
    \bm{\Sigma}_{\rm tes} &= \bm{K}_{\rm tes} - \bm{k}_{\bm{v}, {\rm tes}}^\top \left( \bm{\Sigma}_0 + \tilde{\bm{\Sigma}} \right)^{-1} \bm{k}_{\bm{v}, {\rm tes}},
\end{aligned}
$$
where $\bm{k}_{\bm{v}, {\rm tes}} \in \mathbb{R}^{t \times m}$ is a prior covariance matrix, whose $(i, j)$-th element is ${\rm Cov}\bigl(v_i, f (\bm{x}_{j, {\rm tes}}) \bigr)$, and $\bm{K}_{{\rm tes}} \in \mathbb{R}^{m \times m}$ is a kernel matrix, whose $(i, j)$-th element is $k\bigl(f (\bm{x}_{i, {\rm tes}}), f (\bm{x}_{j, {\rm tes}}) \bigr)$.
# Other formulation

We described the EP for the truncated multivariate normal distribution.
On the other hand, using $z_i \coloneqq f(\bm{x}_{i, l}) - f(\bm{x}_{i, w})$, we can formulate the estimation of GPR for the preferential learning as follows:
$$
\begin{aligned}
    p(\bm{z} | \bm{v}_t < \bm{0}) 
    &\propto \mathcal{N}(\bm{z} | \bm{\mu}_0, \bm{\Sigma}_0 - 2\sigma_{\rm noise}\bm{I}) \prod_{i=1}^t \Pr( z_i + \epsilon_l - \epsilon_w < 0 | z_i) \\
    &= \mathcal{N}(\bm{z} | \bm{\mu}_0, \bm{\Sigma}_0 - 2\sigma_{\rm noise}\bm{I}) \prod_{i=1}^t \Phi\left( -\frac{z_i}{\sqrt{2}\sigma_{\rm noise}} \right) \\
\end{aligned}
$$
where $\bm{I}$ is the identity matrix.
Then, the derivation of the EP procedure is slightly changed.
However, the derivation for this formulation is essentially same as the GP classification model [5].
See [5] for the detailed derivation.
# Reference

- [1]: [Takeno, S., Nomura, M., and Karasuyama, M., Towards Practical Preferential Bayesian Optimization with Skew Gaussian Processes, arXiv:2302.01513, 2023.](https://arxiv.org/abs/2302.01513) 
- [2]: [Chu, W. and Ghahramani, Z. Extensions of Gaussian processes for ranking: semisupervised and active learning. In Proceedings of the Learning to Rank workshop at the 18th Conference on Neural Information Processing Systems, pp. 29–34, 2005.](http://www.gatsby.ucl.ac.uk/~chuwei/paper/gprl.pdf)
- [3]: [Chu, W. and Ghahramani, Z. Preference learning with Gaussian processes. In Proceedings of the 22nd international conference on Machine learning, pp. 137–144, 2005.](https://icml.cc/Conferences/2005/proceedings/papers/018_Preference_ChuGhahramani.pdf)
- [4]: [Hernandez-Lobato, J. M., Hoffman, M. W., Ghahramani, Z., Predictive Entropy Search for Efficient Global Optimization of Black-box Functions, Advances in neural information processing systems 27, pp. 918-926, 2014.](https://papers.nips.cc/paper/2014/hash/069d3bb002acd8d7dd095917f9efe4cb-Abstract.html)
- [5]: [Rasmussen, C. E. and Williams, C. K. I. Gaussian Processes for Machine Learning (Adaptive Computation and Machine Learning). The MIT Press, 2005](https://gaussianprocess.org/gpml/)