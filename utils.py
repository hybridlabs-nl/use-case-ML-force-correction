import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
import pandas as pd

def plot_forces(df):
    "Plots the forces from the dataframe."
    fig, axes = plt.subplots(3, 2, figsize=(15, 8))
    axes = axes.flatten()

    for i, col in enumerate(df.columns):
        axes[i].plot(range(len(df[col])), df[col], label=col)
        axes[i].set_title(f"Column: {col}")
        axes[i].set_xlabel("time (s)")
        axes[i].set_ylabel(col)
        axes[i].legend()
    plt.tight_layout()
    plt.show()


def compute_mi_scores(df, target_labels, predictors_labels):
    """Computes mutual information scores for the given target and predictor labels."""
    mi_scores = {}
    for target_col in target_labels:
        mi = mutual_info_regression(df[predictors_labels], df[target_col], random_state=42)
        mi_scores[target_col] = mi
    return pd.DataFrame(mi_scores, index=predictors_labels)