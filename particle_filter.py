"""
This file implements a particle filter, at step t = 0, we start from a set of particles drawn from the prior and then based on that we produce proposal from new state, the new state particles are reweighed according to the oracles and then used to update the particles. Our goal is to replace the particle filter by an approximate distribution, say a generative model, whose posterior is suitably modified (using either control / guidance ) at inference time.
"""

import torch
from matplotlib import pyplot as plt
N = 3 #dimension of the state



torch.manual_seed(0)


from model import MLP

model = MLP()
weights = "/Users/mayankshrivastava/Desktop/DataAssimilation/train_model_20251022_170846.pt"
model.load_state_dict(torch.load(weights))

model.eval()

@torch.no_grad()
def fm_transport_once(model, x0, o_t=None, steps=16):
    """
    Integrate dx/ds = v_theta(x, s, o_t), s in [0,1], using RK4 with 'steps' substeps.
    x0: [B, D]
    returns: x1_pred [B, D]
    """
    B = x0.shape[0]
    x = x0
    h = 1.0 / steps
    s = torch.zeros(B, device=x0.device)

    for i in range(steps):
        s0 = s
        s1 = s + 0.5*h
        s2 = s + 0.5*h
        s3 = s + h

        k1 = model(x,          s0, o_t)
        k2 = model(x + 0.5*h*k1, s1, o_t)
        k3 = model(x + 0.5*h*k2, s2, o_t)
        k4 = model(x + h*k3,    s3, o_t)

        x = x + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        s = s3
    return x

def simulate_flow(particle):
    return fm_transport_once(model,particle)
    

def likelihood(x_t,o_t):
    #gaussian likelihood o_t ~ N(x_t , \sigma^2)
    sigma2 = torch.tensor(1)
    return 1/torch.sqrt(2*torch.pi*sigma2)*torch.exp(-torch.linalg.norm(o_t-x_t)**2/2/sigma2)

num_particles = 10

particle_history = []
weights_history = []
cur_particles = torch.randn(num_particles,N)
weights = 1/num_particles*torch.ones(num_particles)

time_steps = 100

for i in range(time_steps):
    particle_history.append(cur_particles.clone())
    weights_history.append(weights.clone())
    new_particles = []
    new_weights = []
    for j in range(num_particles):
        
        #propagate  x_t ~ p(x_t | x_{t-1})
        ip = cur_particles[j].unsqueeze(0)
        
        new_particle = simulate_flow(ip)
        new_particle = new_particle.squeeze()
        
        
        o_t = new_particle + 1e-1*torch.randn(3)
        
        #reweigh according to o_t 
        
        new_particles.append(new_particle)
        
        weights[j] *= likelihood(new_particle,o_t)
        
        #check ESS
        ess = 1
        
    #normalize
    
    weights = weights/torch.sum(weights)
    
    
    cur_particles = torch.vstack(new_particles)


plt.figure(figsize=(6,6))
plt.ylim(0,1)
for i in range(time_steps):
    print(weights_history[i].shape)
    plt.plot(weights_history[i],label=f"Iter:{i}")
    
plt.legend()

plt.savefig("Weight Skew")

      
        
        

        
        
        
        
            
            
        
        
        
        
        
    
    
    