import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

updates_data = torch.load('/home/omar/weiper/Weiper/final_layer_updates_epoch1.pt')
noise_data = torch.load('/home/omar/weiper/Weiper/random_noise_vectors.pt')

updates = updates_data['update_w']  
if isinstance(updates, list):
    updates = torch.cat([u.cpu() for u in updates], dim=0)
else:
    updates = updates.cpu()

if isinstance(noise_data, list):
    noise = torch.cat([n.cpu() for n in noise_data], dim=0)
else:
    noise = noise_data.cpu()
    
updates = updates.view(-1, updates.size(-1))
noise = noise.view(-1, noise.size(-1))

print("Updates shape:", updates.shape)
print("Noise shape:", noise.shape)

updates_norm = F.normalize(updates, dim=1)
noise_norm = F.normalize(noise, dim=1)

cosine_sim_matrix = torch.mm(updates_norm, noise_norm.t())
cosine_similarities = cosine_sim_matrix.flatten().detach().cpu().numpy()

print("Cosine Similarity Stats:")
print("  Mean:", cosine_similarities.mean())
print("  Min: ", cosine_similarities.min())
print("  Max: ", cosine_similarities.max())
