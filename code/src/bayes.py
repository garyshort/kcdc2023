import numpy as np


def bayes_update(prior, likelihoods):

    for likelihood in likelihoods:
        # Calculate the likelihood ratio
        likelihood_ratio = likelihood[0] / likelihood[1]

        # Update the prior
        prior = (likelihood_ratio * prior) / \
            (likelihood_ratio * prior + (1 - prior))

    return prior


# Initial belief
prior = 0.5

# Vector of evidence. Each tuple represents a piece of evidence.
# The first value in the tuple is the likelihood of observing the evidence
# given that the hypothesis is true. # The second value is the likelihood of
# observing the evidence given that the hypothesis is false.
evidence = [(0.7, 0.3), (0.9, 0.2), (0.8, 0.4)]

# Update the belief based on the evidence
posterior = bayes_update(prior, evidence)
print(f"Updated belief: {posterior}")
