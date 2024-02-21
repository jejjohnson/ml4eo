## Prior Predictive Checks


We want to sample something from the prior predictive distribution given by:

$$
y_n \sim p(y)
$$

This is the same as the posterior predictive distribution but without any observations.
So first, we simulate the parameters

$$
\boldsymbol{\theta}_n \sim p(\boldsymbol{\theta})
$$

Then according to the prior, we pass it through the likelihood

$$
y_n \sim p(y|\boldsymbol{\theta}_n)
$$

The result is a simulation from the joint distribution

$$
y_n,\boldsymbol{\theta}_n \sim p(y,\boldsymbol{\theta})
$$

and therefore $y_n \sim p(y)$.