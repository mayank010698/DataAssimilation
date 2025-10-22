"""
This file implements a particle filter, at step t = 0, we start from a set of particles drawn from the prior and then based on that we produce proposal from new state, the new state particles are reweighed according to the oracles and then used to update the particles. Our goal is to replace the particle filter by an approximate distribution, say a generative model, whose posterior is suitably modified (using either control / guidance ) at inference time.
"""



