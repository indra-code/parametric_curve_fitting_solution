import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.signal import find_peaks
from scipy import optimize
import matplotlib.pyplot as plt
# ------------------------------------------------------------
# STEP 1: Load the data
# ------------------------------------------------------------
df = pd.read_csv('xy_data.csv')   # columns: x, y
x = df['x'].values
y = df['y'].values

# ------------------------------------------------------------
# STEP 2: Estimate approximate direction using PCA
# ------------------------------------------------------------
XY = np.vstack([x, y]).T
mean_xy = XY.mean(axis=0)
XYc = XY - mean_xy

pca = PCA(n_components=1)
pca.fit(XYc)
u = pca.components_[0] 
theta0 = np.arctan2(u[1], u[0])

# ------------------------------------------------------------
# STEP 3: Map x,y onto a 1D parameter t using PCA projection
# ------------------------------------------------------------
s = XYc.dot(u)
order = np.argsort(s)
smin, smax = s[order[0]], s[order[-1]]

# scale s → t ∈ [6, 60]
a = (60.0 - 6.0) / (smax - smin)
b = 6.0 - a * smin
t = a * s + b

# ------------------------------------------------------------
# STEP 4: Estimate θ and X using simple linear regression
# ------------------------------------------------------------
A = np.vstack([t, np.ones_like(t)]).T
ax, X0 = np.linalg.lstsq(A, x, rcond=None)[0]
ay, b_y = np.linalg.lstsq(A, y - 42.0, rcond=None)[0]
theta_init = np.arctan2(ay, ax) 

# ------------------------------------------------------------
# STEP 5: Define residuals function for later use
# ------------------------------------------------------------
def residuals(params):
    th, M, X = params
    xp = t * np.cos(th) - np.exp(M * np.abs(t)) * np.sin(0.3 * t) * np.sin(th) + X
    yp = 42 + t * np.sin(th) + np.exp(M * np.abs(t)) * np.sin(0.3 * t) * np.cos(th)
    rx = x - xp
    ry = y - yp
    return rx, ry

# ------------------------------------------------------------
# STEP 6: Estimate M by examining the residual envelope
# ------------------------------------------------------------
rx, ry = residuals([theta_init, 0.0, X0])
R = np.sqrt(rx ** 2 + ry ** 2)
peaks, _ = find_peaks(R, distance=5)
tp = t[peaks]
Rp = R[peaks]
mask = Rp > np.percentile(Rp, 60)
tp = tp[mask]
Rp = Rp[mask]
M_est, c = np.polyfit(tp, np.log(np.clip(Rp, 1e-9, None)), 1)

# ------------------------------------------------------------
# STEP 7: Refining theta using residual directions
# ------------------------------------------------------------
idx = np.abs(np.sin(0.3 * t)) > 0.5
angles = np.arctan2(ry[idx], -rx[idx])
theta_ref = np.angle(np.mean(np.exp(1j * angles)))  # circular mean

# ------------------------------------------------------------
# STEP 8: Define the loss function
# ------------------------------------------------------------
def loss(params):
    th, M, X = params
    xp = t * np.cos(th) - np.exp(M * np.abs(t)) * np.sin(0.3 * t) * np.sin(th) + X
    yp = 42 + t * np.sin(th) + np.exp(M * np.abs(t)) * np.sin(0.3 * t) * np.cos(th)
    return np.sum(np.abs(xp - x) + np.abs(yp - y))

# ------------------------------------------------------------
# STEP 9: Defining bounds from problem statement
# ------------------------------------------------------------
bnds = [
    (np.deg2rad(1e-4), np.deg2rad(50.0 - 1e-4)),  
    (-0.05, 0.05),                                
    (0.0, 100.0)                                  
]
num_candidates = 10
init_guess = [theta_ref, M_est, X0]
pop = np.tile(init_guess, (num_candidates, 1)) + np.random.normal(
    scale=[np.deg2rad(0.5), 0.001, 0.5],  
    size=(num_candidates, 3)
)
pop = np.clip(pop, 
              [bnds[0][0], bnds[1][0], bnds[2][0]], 
              [bnds[0][1], bnds[1][1], bnds[2][1]])
# ------------------------------------------------------------
# STEP 10: Using manual estimates as the initial seed for Differential Evolution
# ------------------------------------------------------------


res = optimize.differential_evolution(
    lambda p: loss(p),
    bounds=bnds,
    maxiter=400,
    popsize=10,
    init=pop
)

# ------------------------------------------------------------
# STEP 11: Local refinement using L-BFGS-B
# ------------------------------------------------------------
res2 = optimize.minimize(
    lambda p: loss(p),
    res.x,
    bounds=bnds,
    method='L-BFGS-B'
)

theta_opt, M_opt, X_opt = res2.x

# ------------------------------------------------------------
# STEP 12: Print results
# ------------------------------------------------------------
print(f"theta (degrees) = {np.rad2deg(theta_opt):.4f}")
print(f"M = {M_opt:.6f}")
print(f"X = {X_opt:.4f}")

def compute_validation_score(theta, M, X, n_samples=1000):
    t_uniform = np.linspace(6, 60, n_samples)
    x_pred = t_uniform * np.cos(theta) - np.exp(M * np.abs(t_uniform)) * np.sin(0.3 * t_uniform) * np.sin(theta) + X
    y_pred = 42 + t_uniform * np.sin(theta) + np.exp(M * np.abs(t_uniform)) * np.sin(0.3 * t_uniform) * np.cos(theta)
    total_dist = 0
    for xp, yp in zip(x_pred, y_pred):
        distances = np.abs(x - xp) + np.abs(y - yp)
        total_dist += np.min(distances)
    
    return total_dist / n_samples

score = compute_validation_score(theta_opt, M_opt, X_opt)
print(f"\nValidation L1 distance (per point): {score:.4f}")
# Optional: display initial vs final comparison
print("ax, ay:", ax, ay)
print("theta_init (deg):", np.rad2deg(theta_init))
print("theta_init +/- 180 (deg):", np.rad2deg(theta_init) - 180, np.rad2deg(theta_init) + 180)
print("theta_ref (deg):", np.rad2deg(theta_ref))
# ------------------------------------------------------------
t_plot = np.linspace(6, 60, 1000)
x_fit = t_plot * np.cos(theta_opt) - np.exp(M_opt * np.abs(t_plot)) * np.sin(0.3 * t_plot) * np.sin(theta_opt) + X_opt
y_fit = 42 + t_plot * np.sin(theta_opt) + np.exp(M_opt * np.abs(t_plot)) * np.sin(0.3 * t_plot) * np.cos(theta_opt)

plt.figure(figsize=(12, 10))

# Plot 1: Curve fit
plt.subplot(2, 2, 1)
plt.scatter(x, y, alpha=0.6, s=20, label='Data points', c='blue')
plt.plot(x_fit, y_fit, 'r-', linewidth=2, label='Fitted curve')
plt.legend()
plt.axis('equal')
plt.title(f'Curve Fit: θ={np.rad2deg(theta_opt):.2f}°, M={M_opt:.4f}, X={X_opt:.2f}')
plt.grid(True, alpha=0.3)

# Plot 2: Residuals
plt.subplot(2, 2, 2)
rx_final, ry_final = residuals([theta_opt, M_opt, X_opt])
R_final = np.sqrt(rx_final**2 + ry_final**2)
plt.scatter(t, R_final, alpha=0.6)
plt.xlabel('t')
plt.ylabel('Residual magnitude')
plt.title('Final Residuals vs t')
plt.grid(True, alpha=0.3)

# Plot 3: X-residuals
plt.subplot(2, 2, 3)
plt.scatter(t, rx_final, alpha=0.6)
plt.xlabel('t')
plt.ylabel('x residual')
plt.title('X-component residuals')
plt.grid(True, alpha=0.3)

# Plot 4: Y-residuals
plt.subplot(2, 2, 4)
plt.scatter(t, ry_final, alpha=0.6)
plt.xlabel('t')
plt.ylabel('y residual')
plt.title('Y-component residuals')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('curve_fitting_results.png', dpi=300, bbox_inches='tight')
plt.show()
