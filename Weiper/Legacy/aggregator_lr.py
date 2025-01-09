# aggregator_trainer_lr.py
import numpy as np
from sklearn.linear_model import LogisticRegression

# If you have multiple OOD scores per sample, set INPUT_DIM to that number.
# If only one score dimension, INPUT_DIM = 1.
INPUT_DIM = 1  
OUTPUT_MODEL_PATH = "aggregator_lr.npz"

def main():
    print("[DEBUG] Loading validation scores from val_scores.npz")
    data = np.load("val_scores.npz")
    scores_val_id = data["scores_val_id"]     # shape [N_id, D] - D=INPUT_DIM
    scores_val_ood = data["scores_val_ood"]   # shape [N_ood, D]

    labels_id = np.zeros(scores_val_id.shape[0], dtype=np.float32)
    labels_ood = np.ones(scores_val_ood.shape[0], dtype=np.float32)

    X = np.concatenate([scores_val_id, scores_val_ood], axis=0).astype(np.float32)
    Y = np.concatenate([labels_id, labels_ood], axis=0).astype(np.float32)

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]

    # Ensure X has shape [N, D]
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    print("[DEBUG] Training logistic regression aggregator...")
    clf = LogisticRegression(solver='lbfgs')
    clf.fit(X, Y)

    print("[DEBUG] Training completed. Saving model parameters.")
    np.savez(OUTPUT_MODEL_PATH, coef=clf.coef_, intercept=clf.intercept_)
    print("[DEBUG] Logistic regression aggregator saved to:", OUTPUT_MODEL_PATH)

if __name__ == "__main__":
    main()
