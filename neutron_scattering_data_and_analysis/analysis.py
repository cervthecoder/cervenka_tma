import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Rectangle, ConnectionPatch
from matplotlib import rc
from matplotlib.pyplot import figure
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C



plt.rcParams['text.usetex'] = True

# Load the data from the ODS file
file_path = 'neutron_scattering_py.ods'
sheet_name = 'Sheet2'

data = pd.read_excel(file_path, engine='odf', sheet_name=sheet_name)

# Extract relevant columns for plotting
Q = pd.to_numeric(data.iloc[:, 0], errors='coerce')  # First column for Q
charmm = pd.to_numeric(data.iloc[:, 1], errors='coerce')  # Second column for CHARMM
prosecco = pd.to_numeric(data.iloc[:, 2], errors='coerce')  # Third column for ProseCCo
experiment = pd.to_numeric(data.iloc[:, 3], errors='coerce')  # Fourth column for Experimental data

# Drop rows with NaN values
valid_data = pd.DataFrame({'Q': Q, 'charmm': charmm, 'prosecco': prosecco, 'experiment': experiment}).dropna()

# Extract valid data after dropping NaNs
Q = valid_data['Q']
charmm = valid_data['charmm']
prosecco = valid_data['prosecco']
experiment = valid_data['experiment']


# Assuming you already have your dataset as arrays X and y
X = np.array(Q).reshape(-1, 1)  # Feature array (shape: [n_samples, n_features])
y = np.array(experiment)  # Target array (shape: [n_samples])

# Define the kernel: 
# C(1.0) -> signal variance
# RBF(length_scale) -> smoothness/scale of the signal
# WhiteKernel() -> estimate the noise variance

kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-10, 1e1))

# Create the GaussianProcessRegressor model
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

# Fit the model to your dataset
gp.fit(X, y)

# Extract the learned kernel hyperparameters after fitting
kernel_ = gp.kernel_

# Print the estimated noise variance
noise_variance = gp.kernel_.k2.noise_level
print(f"Estimated noise standard deviation: {noise_variance**(1/2)}")

y_pred, y_std = gp.predict(X, return_std=True)

# Comment out all intermediate plots so that only the last plot is executed:
"""
# Plot the curves
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(Q, charmm, label='CHARMM36', linestyle='-', color='#C92B76', linewidth=1.5)
ax.plot(Q, prosecco, label='ProseCCo75', linestyle='-', color='#2ECC2E', linewidth=1.5)
ax.plot(Q, experiment, label='Neutron Scattering', linestyle='-', color='#4B0092', linewidth=1.5)
ax.legend()
# Hide top and right spines
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)

# Labeling the plot
ax.set_xlabel(r'Q (\AA$^{-1}$)', fontsize=14, fontweight='bold')
ax.set_ylabel(r'S$_{HTMA-HW}$(Q)', fontsize=14, fontweight='bold')
ax.set_title(r'Structural factor (Correlation of H$_{TMA}$-H$_{W}$)', fontsize=14, fontweight='bold')
ax.legend(loc='center right', fontsize=12)
ax.set_xlim(0, 10)

# Customize axis tick parameters
ax.tick_params(axis='both', which='major', labelsize=12, width=2, direction='in', length=8, labelcolor='black')

# Adding a zoomed inset
ax_inset = inset_axes(ax, width='40%', height='60%', loc='lower center')
ax_inset.plot(Q, charmm, linestyle='-' , color='#C92B76')
ax_inset.plot(Q, prosecco, linestyle='-', color='#2ECC2E')
ax_inset.plot(Q, experiment, linestyle='-', color='#4B0092')
ax_inset.set_xlim(0.3, 1.3)
ax_inset.set_ylim(-8.7, -5)
ax_inset.set_xticks([])
ax_inset.set_yticks([])

# Add a rectangle to indicate the inset area on the main plot
rect = Rectangle((0.3, -8.7), 1, 3.7, edgecolor='black', facecolor='none', linestyle='--', linewidth=1.5)
ax.add_patch(rect)

# Add connecting lines between the rectangle and the inset
con1 = ConnectionPatch(xyA=(1.3, -8.7), xyB=(1.3, -8.7), coordsA="data", coordsB="data",
                       axesA=ax, axesB=ax_inset, color="black", linestyle="--", linewidth=1.5)
con2 = ConnectionPatch(xyA=(0.3, -5), xyB=(0.3, -5), coordsA="data", coordsB="data",
                       axesA=ax, axesB=ax_inset, color="black", linestyle="--", linewidth=1.5)
ax.add_artist(con1)
ax.add_artist(con2)

plt.savefig('plot_rdf_neutron_fixed.png', dpi=300, bbox_inches='tight')
# Show the plot
plt.show()

rmse_charmm   = np.sqrt((charmm - experiment)**2)
rmse_prosecco = np.sqrt((prosecco - experiment)**2)
#calculate RMSE for all data in bad model fit region
print("Average RMSE CHARMM <2.5 \\AA$^{-1}$:", np.sum((charmm[:42] - experiment[:42])**2)/len(Q[:42]))
print("Average RMSE ProsECCo <2.5 \\AA$^{-1}$:",np.sum((prosecco[:42] - experiment[:42])**2)/len(Q[:42]))

plt.plot(Q,rmse_charmm, label = 'CHARMM36')
plt.plot(Q,rmse_prosecco, label = 'ProsECCo75')
plt.xlabel(r'Q [\AA$^{-1}$]')
plt.ylabel('RMSE')
plt.legend()
# plt.xlim(Q[8],2.5)
plt.show()

plt.plot(Q, experiment - prosecco)

rmse_charmm   = np.sqrt((charmm - experiment)**2)
rmse_prosecco = np.sqrt((prosecco - experiment)**2)
#calculate RMSE for all data in bad model fit region
print("Average RMSE CHARMM <2.5 \\AA$^{-1}$:", np.sum((charmm[:42] - experiment[:42])**2)/len(Q[:42]))
print("Average RMSE ProsECCo <2.5 \\AA$^{-1}$:",np.sum((prosecco[:42] - experiment[:42])**2)/len(Q[:42]))

plt.plot(Q,rmse_charmm, label = 'CHARMM36')
plt.plot(Q,rmse_prosecco, label = 'ProsECCo75')
plt.xlabel(r'Q [\AA$^{-1}$]')
plt.ylabel('RMSE')
plt.legend()
# plt.xlim(Q[8],2.5)
plt.show()

plt.plot(Q, experiment - prosecco)

# Main figure
plt.figure(figsize=(8, 6))

# Plot the observations
plt.text(2, -10, r'$\sigma_{noise} = 0.037$', fontsize=24)
plt.plot(X, y, 'k.', markersize=10, label="Experiment", alpha = 0.5)
plt.plot(X, y_pred, 'k-', label="GP Mean Prediction", alpha = 0.5)
#plt.fill_between(X.ravel(), y_pred - 1.96 * y_std, y_pred + 1.96 * y_std, alpha=0.2, label="95% Credibility Interval")
plt.plot(Q,charmm, linestyle = '--', color = 'b', label = 'CHARMM36')
plt.plot(Q,prosecco, linestyle = '--', color = 'r', label = 'ProsECCo75')

# Labels and legend
plt.xlim(0,10)
plt.ylim(-11,1)
plt.xlabel(r'Q (\AA$^{-1}$)', fontsize = 16)
plt.ylabel('S(Q)', fontsize = 16)
plt.legend(loc="best")

# Create an inset plot
ax_inset = plt.gca().inset_axes([0.3, 0.3, 0.5, 0.5])

# Deviation from GP mean
deviation = y - y_pred

# Plot the deviation
ax_inset.plot(X, deviation, 'k.', markersize=5, label="Exp - GP Mean")
# Plot the credibility interval (error bars showing deviation range)
ax_inset.fill_between(X.ravel(), -1.96 * y_std, 1.96 * y_std, color='gray', alpha=0.2, label="95% Credibility Interval")
#ax_inset.plot(Q,charmm - experiment, label = 'CHARMM36 - Exp')
ax_inset.plot(Q,charmm - experiment, linestyle = '--', color = 'b')
ax_inset.plot(Q,prosecco - experiment, linestyle = '--', color = 'r')
ax_inset.axhline(0, color='k', linestyle='--', linewidth=1)  # Zero line for reference

# Inset labels and title
#ax_inset.set_xlim(Q[8],4)
ax_inset.set_title("Deviation from Experiment")
ax_inset.set_xlabel(r'Q (\AA$^{-1}$)')
ax_inset.set_ylabel('Deviation')
ax_inset.legend(loc="best")
ax_inset.set_xlim(0, 10)
ax_inset.set_ylim(-1, 1)

# Show the plot
plt.tight_layout()
plt.savefig('unc_analysis', dpi = 600)
plt.show()
"""

# Main figure with stacked subplots and shared x-axis
fig, (ax_main, ax_bottom) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

# Set all spines' linewidth and customize tick parameters for both subplots
for ax in [ax_main, ax_bottom]:
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.tick_params(axis='both', which='major', labelsize=12, width=2, direction='in', length=8, labelcolor='black')

# Top plot (Main data)
ax_main.plot(X, y, 'k.', markersize=10, label="Experiment", alpha=0.5)
ax_main.plot(X, y_pred, 'k-', label="GP Mean Prediction", alpha=0.5)
ax_main.plot(Q, charmm, linestyle='-', color='#0072B2', label='CHARMM36')
ax_main.plot(Q, prosecco, linestyle='-', color='#D55E00', label='prosECCo75')
ax_main.set_xlim(0, 10)
ax_main.set_ylim(-9.5, 1)
ax_main.set_ylabel(r"$\mathbf{\Delta\Delta} \textbf{S}_{\textbf{H}_{\textbf{non}}}\textbf{(Q)}$", fontsize=16)
# Adding label "a)" following the provided style
ax_main.text(-0.12, 1.05, r"$\textbf{a)}$", transform=ax_main.transAxes, fontsize=26, va="top", ha="left")

# Inset plot (inside top subplot) remains unchanged
ax_inset = ax_main.inset_axes([0.25, 0.1, 0.7, 0.7])
ax_inset.plot(Q, charmm, linestyle='-', color='#0072B2', label='CHARMM36', linewidth=2)
ax_inset.plot(Q, prosecco, linestyle='-', color='#D55E00', label='prosECCo75', linewidth=2)
ax_inset.plot(X, y, 'k.', markersize=5, alpha=0.5, label='Experiment', linewidth=2)
ax_inset.plot(X, y_pred, 'k-', alpha=0.5, label="GP Mean Prediction")
ax_inset.set_xlim(0.4, 1)
ax_inset.set_ylim(-9, -4)
ax_inset.legend(loc="upper center", bbox_to_anchor=(0.43, 1), prop={'weight': 'bold', 'size': 16})
ax_inset.set_xticklabels([])
ax_inset.set_yticklabels([])
ax_inset.set_xticks([])
ax_inset.set_yticks([])
ax_main.indicate_inset_zoom(ax_inset, edgecolor='black', linewidth=1.5)

# Bottom plot (Deviation plot)
deviation = y - y_pred
ax_bottom.plot(X, deviation, 'k.', markersize=5, alpha=0.5)
ax_bottom.fill_between(X.ravel(), -1.96 * y_std, 1.96 * y_std, color='gray', alpha=0.2, label=r"GP 95\% Credibility Interval")
ax_bottom.plot(Q, charmm - experiment, linestyle='-', color='#0072B2')
ax_bottom.plot(Q, prosecco - experiment, linestyle='-', color='#D55E00')
ax_bottom.axhline(0, color='k', linestyle='--', linewidth=1)
ax_bottom.text(3.33, -0.7, r'$\sigma_{noise} = 0.037$', fontsize=24)
ax_bottom.set_xlim(0, 10)
ax_bottom.set_ylim(-1, 1)
ax_bottom.set_xlabel(r"\textbf{Q} \textbf{(Å$^{\textbf{-1}}$)}", fontsize=16)
ax_bottom.set_ylabel(r'\textbf{Deviation}', fontsize=16)
ax_bottom.legend(loc="best", prop={'weight': 'bold', 'size': 16})
# Adding label "b)" following the provided style
ax_bottom.text(-0.12, 1.15, r"$\textbf{b)}$", transform=ax_bottom.transAxes, fontsize=26, va="top", ha="left")

# Adjust layout and show/save the plot
plt.tight_layout()
plt.savefig('unc_analysis_stacked_shared_x', dpi=600)
plt.show()


