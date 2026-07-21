import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation


def plot_forces(df):
    "Plots the forces from the dataframe."

    n_cols = 2
    n_rows = df.shape[1] // n_cols

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
        mi = mutual_info_regression(
            df[predictors_labels], df[target_col], random_state=42
        )
        mi_scores[target_col] = mi
    return pd.DataFrame(mi_scores, index=predictors_labels)


def compare_forces(df1, df2, title=("Target", "Predicted")):
    """Plots the forces from two dataframes."""
    if df1.shape != df2.shape:
        raise ValueError("DataFrames must have the same shape for comparison.")

    n_cols = 2
    n_rows = df1.shape[1] // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 8))
    axes = axes.flatten()

    for i, col in enumerate(df1.columns):
        axes[i].plot(
            range(len(df1[col])), df1[col], label=f"{col} ({title[0]})", color="blue"
        )
        axes[i].plot(
            range(len(df2[col])),
            df2[col],
            label=f"{col} ({title[1]})",
            color="red",
            alpha=0.8,
        )
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
        axes[i].hist(
            df1[col],
            bins=bins,
            alpha=0.5,
            label=f"{col} ({title[0]})",
            color="blue",
            density=True,
        )
        axes[i].hist(
            df2[col],
            bins=bins,
            alpha=0.5,
            label=f"{col} ({title[1]})",
            color="red",
            density=True,
        )
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Density")
        axes[i].legend()
    plt.tight_layout()
    plt.show()


def rotate_force(pos, Faero_est, Taero_est, i_trans=True):
    """
    Rotate aerodynamic forces and moments from the rotating nacelle frame
    to the fixed tower-base frame.

    Parameters
    ----------
    pos : (N, 3) array_like
        Roll, pitch, yaw angles [degrees].
        Columns: [roll, pitch, yaw].

    Faero_est : (N, 3) array_like
        Aerodynamic forces in the rotating frame.

    Taero_est : (N, 3) array_like
        Aerodynamic torques.

    i_trans : bool, default=True
        If True, transport the moment from tower top to tower base.

    Returns
    -------
    F_out : (N, 3) ndarray
        Forces in the fixed frame.

    T_out : (N, 3) ndarray
        Moments in the fixed frame (including transport if enabled).
    """

    pos = np.asarray(pos, dtype=float)
    F = np.asarray(Faero_est, dtype=float)
    T = np.asarray(Taero_est, dtype=float)

    # SciPy expects [yaw, pitch, roll] for 'ZYX'
    angles = pos[:, [2, 1, 0]]

    # One rotation object per time step
    R = Rotation.from_euler("ZYX", angles, degrees=True)

    # Rotate forces
    F_out = R.apply(F)

    if i_trans:
        # Tower-top position relative to tower base
        h = 0.81
        r0 = np.array([0.0, 0.0, h])

        # Rotate position vector for each sample
        r_fixed = R.apply(r0)

        # Transport moment: M = r × F
        T_out = T + np.cross(r_fixed, F_out)
    else:
        T_out = T.copy()

    return F_out, T_out
