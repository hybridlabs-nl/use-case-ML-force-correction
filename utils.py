import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
import pandas as pd

def plot_forces(df):
    "Plots the forces from the dataframe."
    if df.shape[1] == 6:
        n_rows, n_cols = 3, 2
    else:
        n_rows, n_cols = 3, 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 8))
    axes = axes.flatten()

    for i, col in enumerate(df.columns):
        axes[i].plot(range(len(df[col])), df[col], label=col)
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


def compare_forces(df1, df2, title=("Target", "Predicted")):
    """Plots the forces from two dataframes."""
    if df1.shape != df2.shape:
        raise ValueError("DataFrames must have the same shape for comparison.")

    if df1.shape[1] == 6:
        n_rows, n_cols = 3, 2
    else:
        n_rows, n_cols = 3, 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 8))
    axes = axes.flatten()

    for i, col in enumerate(df1.columns):
        axes[i].plot(range(len(df1[col])), df1[col], label=f'{col} ({title[0]})', color='blue')
        axes[i].plot(range(len(df2[col])), df2[col], label=f'{col} ({title[1]})', color='red', alpha=0.8)
        axes[i].set_xlabel("time (s)")
        axes[i].set_ylabel(col)
        axes[i].legend()
    plt.tight_layout()
    plt.show()


def compare_histograms(df1, df2, title=("Target", "Predicted"), bins=30):
    """Plots histograms of the forces from two dataframes."""
    if df1.shape != df2.shape:
        raise ValueError("DataFrames must have the same shape for comparison.")

    if df1.shape[1] == 6:
        n_rows, n_cols = 3, 2
    else:
        n_rows, n_cols = 3, 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 8))
    axes = axes.flatten()

    for i, col in enumerate(df1.columns):
        axes[i].hist(df1[col], bins=bins, alpha=0.5, label=f'{col} ({title[0]})', color='blue', density=True)
        axes[i].hist(df2[col], bins=bins, alpha=0.5, label=f'{col} ({title[1]})', color='red', density=True)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Density")
        axes[i].legend()
    plt.tight_layout()
    plt.show()
