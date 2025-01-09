import numpy as np

gradients = np.load("final_gradients.npy")  # shape [N_steps, D]
mean_grad = gradients.mean(axis=0)
std_grad = gradients.std(axis=0)

mean_magnitude = np.linalg.norm(mean_grad)
std_magnitude = std_grad.mean()  # average std across parameters

print("Mean gradient norm:", mean_magnitude)
print("Average gradient std (sigma):", std_magnitude)
sigma = std_magnitude
