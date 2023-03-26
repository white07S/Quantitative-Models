import numpy as np
import pandas as pd
import time
from numba import jit
import matplotlib.pyplot as plt
import seaborn as sns


# Parameters for the Vasicek and CIR models
n_simulations = 1000
n_steps = 250
dt = 1/252
kappa = 0.3
theta = 0.02
sigma = 0.02
r0 = 0.01


@jit(nopython=True)
def vasicek_simulations_jit(r0, kappa, theta, sigma, dt, n_steps, n_simulations):
    rates = np.zeros((n_simulations, n_steps + 1))
    rates[:, 0] = r0
    dW = np.random.normal(0, np.sqrt(dt), (n_simulations, n_steps))

    for t in range(1, n_steps + 1):
        dr = kappa * (theta - rates[:, t-1]) * dt + sigma * dW[:, t-1]
        rates[:, t] = rates[:, t-1] + dr

    return rates


@jit(nopython=True)
def cir_simulations_jit(r0, kappa, theta, sigma, dt, n_steps, n_simulations):
    rates = np.zeros((n_simulations, n_steps + 1))
    rates[:, 0] = r0
    dW = np.random.normal(0, np.sqrt(dt), (n_simulations, n_steps))

    for t in range(1, n_steps + 1):
        dr = kappa * (theta - rates[:, t-1]) * dt + sigma * np.sqrt(rates[:, t-1]) * dW[:, t-1]
        rates[:, t] = np.maximum(rates[:, t-1] + dr, 0)

    return rates

vasicek_rates_jit = vasicek_simulations_jit(r0, kappa, theta, sigma, dt, n_steps, n_simulations)

cir_rates_jit = cir_simulations_jit(r0, kappa, theta, sigma, dt, n_steps, n_simulations)

def autocorrelation(x, lag=1):
    return pd.Series(x).autocorr(lag)

vasicek_autocorr = [autocorrelation(vasicek_rates_jit[:, i]) for i in range(n_steps + 1)]
cir_autocorr = [autocorrelation(cir_rates_jit[:, i]) for i in range(n_steps + 1)]

# Plot autocorrelation
fig, ax = plt.subplots()
ax.plot(vasicek_autocorr, label='Vasicek Model')
ax.plot(cir_autocorr, label='CIR Model')
ax.set_title('Autocorrelation of Interest Rates')
ax.set_xlabel('Lag')
ax.set_ylabel('Autocorrelation')
ax.legend()

# Distribution of interest rates at the end of the simulation
vasicek_end_rates = vasicek_rates_jit[:, -1]
cir_end_rates = cir_rates_jit[:, -1]

# Plot distribution of interest rates
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), sharex=True, sharey=True)

sns.histplot(vasicek_end_rates, kde=True, ax=ax1)
ax1.set_title('Vasicek Model - Interest Rate Distribution')
ax1.set_xlabel('Interest Rates')
ax1.set_ylabel('Frequency')

sns.histplot(cir_end_rates, kde=True, ax=ax2)
ax2.set_title('CIR Model - Interest Rate Distribution')
ax2.set_xlabel('Interest Rates')
ax2.set_ylabel('Frequency')

# Evolution of interest rates over time
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), sharex=True, sharey=True)

sns.kdeplot(data=vasicek_rates_jit, ax=ax1, legend=False, bw_adjust=0.5)
ax1.set_title('Vasicek Model - Interest Rate Evolution')
ax1.set_xlabel('Time Steps')
ax1.set_ylabel('Interest Rates')

sns.kdeplot(data=cir_rates_jit, ax=ax2, legend=False, bw_adjust=0.5)
ax2.set_title('CIR Model - Interest Rate Evolution')
ax2.set_xlabel('Time Steps')
ax2.set_ylabel('Interest Rates')

plt.show()




# Convert simulations to Pandas DataFrames
vasicek_rates_df = pd.DataFrame(vasicek_rates_jit)
cir_rates_df = pd.DataFrame(cir_rates_jit)

# Calculate mean and standard deviation for each model
vasicek_mean = vasicek_rates_df.mean(axis=0)
vasicek_std = vasicek_rates_df.std(axis=0)

cir_mean = cir_rates_df.mean(axis=0)
cir_std = cir_rates_df.std(axis=0)

# Plot Vasicek and CIR simulations along with mean and standard deviation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(vasicek_rates_df.T, alpha=0.1)
ax1.plot(vasicek_mean, color='red', label='Mean')
ax1.plot(vasicek_mean + vasicek_std, color='blue', linestyle='--', label='Mean ± 1 SD')
ax1.plot(vasicek_mean - vasicek_std, color='blue', linestyle='--')
ax1.set_title('Vasicek Model')
ax1.set_xlabel('Time Steps')
ax1.set_ylabel('Interest Rates')
ax1.legend()

ax2.plot(cir_rates_df.T, alpha=0.1)
ax2.plot(cir_mean, color='red', label='Mean')
ax2.plot(cir_mean + cir_std, color='blue', linestyle='--', label='Mean ± 1 SD')
ax2.plot(cir_mean - cir_std, color='blue', linestyle='--')
ax2.set_title('Cox-Ingersoll-Ross (CIR) Model')
ax2.set_xlabel('Time Steps')
ax2.set_ylabel('Interest Rates')
ax2.legend()

plt.show()

