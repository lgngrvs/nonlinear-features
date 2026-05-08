

Hi Claude!

This is a research replication repository for *Do Sparse Autoencoders Capture Concept Manifolds?*, a recently-released paper on on nonlinear feature geometry. We're going to replicate similar results to the paper on a different model (Gemma 12B with the Gemma Scope 2 Sparse Autoencoders dataset) and on a toy synthetic setup.


Here's the fundamental idea (insert the paper)


## Experiment details.

### 1. Reproduce the synthetic task

The original paper contains a synthetic task where it tests how sparse autoencoders behave with data from manifold superposition.

#### Data generation pipleine
(Define the list of manifolds, parameters, and free values)

Take this list of manifolds and isotropically rescale: take 50k points from the raw embedding, and then rescale the points in the raw embedding $\gamma_i(\theta_i)$ by subtracting the mean, and dividing by their standard deviation so that all the points have mean ~1 and std deviation ~1. This should always be done before sampling, so that when the autoencoder is learning on the data from these manifolds, each contributes approximately the same amount to the loss. This will produce a rescaled embedding $\tilde \gamma_i(theta_i)$.

Then for each manifold type, create an up embedding matrix in R^k -> R^d (where k is the ambient embedding dimension, e.g. 3 for a sphere) by sampling a gaussian random matrix of that shape and then taking the Q component of its QR-decomposition, transposed to obtain orthonormal rows.

##### Sampling from the manifolds

Define To generate the data proper, we use $$x = \sum_{i \in S} \gamma(\theta_i) V_i + \epsilon$$

where $\epsilon ~ N(0, \sigma_\epsilonI_d)$ with $\sigma_epsilon = 10^{-5}$ and $|S|$ is defined as a hyperparameter for how many manifolds to include in the sample, with $i \in S$ sampled at random without replacement.

#### Training

Create a dataset of 2 million datapoints with $|S| = 4$. 

Set up training for a TopK sparse autoencoder $$ x = W_{dec} TopK(W_enc x)$$, with hidden size 512. (Since ambient dimension is 128, this yields an expansion size of 4.) 

We want to train SAEs with active dictionary size k = 3, 4, 6, 8, 10, 14, 16, 20, 25. (This is how many nonzero indices to have in the TopK.)

#### Evaluation






future directions not explored yet:
2. Rediscover SAE clusters
- temperature, days of the week, and colors --- establish a pipeline
- use gemma 2

3. extensions
- come up with some hypothesis structures: what kinds of features might be in there that they haven't discovered?
- line breaking? can we discover line breaking?
	- they have cross-layer transcoders in the gemma 2 release!? why is no one talking about this
- CLTs?

