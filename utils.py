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


def correct_force(acc_meas_tt, acc_meas_tb, Fmeas):
    """
    Correct measured forces using regression-based force correction.

    Parameters
    ----------
    acc_meas_tt : (N, 3) array_like
        Accelerations measured at the tower top at rotating frame.

    acc_meas_tb : (N, 3) array_like
        Accelerations measured at the tower base at rotating frame.

    Fmeas : pandas.DataFrame
        Measured forces and moments in the rotating frame.

    Returns
    -------
    Faero_est : (N, 6) ndarray
        Corrected aerodynamic forces and moments in the rotating frame.

    Fcorr : (N, 6) ndarray
        Force correction applied to the measurements in the rotating frame.
    """
    # force correction coefficients from regression
    coeff = np.array([
        [0,       0,       0,       0,       0,       0],
        [-1.7379, -0.0422,  0.0284,  0.0037, -0.0670,  0.0011],
        [0.0085, -1.0294,  0.0159,  0.0335, -0.0071,  0.0080],
        [0.1019, -0.1970, -1.0387,  0.0056,  0.0021,  0.0062],
        [-0.0958,  0.0115, -0.0731,  0.0003, -0.0330, -0.0053],
        [-0.0184,  0.1101,  0.0268, -0.0013, -0.0052,  0.0285],
        [-0.2792,  0.1793, -0.5547, -0.0042, -0.0126, -0.0061]
    ])

    X = np.column_stack([
        np.ones(acc_meas_tt.shape[0]),
        acc_meas_tt,
        acc_meas_tb,
    ])

    Fcorr = X @ coeff
    Faero_est = Fmeas - Fcorr

    g = 9.81
    M = 1.7
    Fz_ofs_comp = np.array([0, 0, -M * g])
    Faero_est[:, 0:3] = Faero_est[:, 0:3] + Fz_ofs_comp

    return Faero_est, Fcorr