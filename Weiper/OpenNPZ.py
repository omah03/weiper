import numpy as np

# Suppose your file is named "id_scores_layer_1_2_3_4.npz"
file_path = "id_scores_layer_1_2_3_4.npz"

data = np.load(file_path)  # load the npz
print("Keys in this NPZ file:", data.files)

# Let's say the array is under the key "scores"
scores_array = data["scores"]

print("Shape of 'scores' array:", scores_array.shape)
