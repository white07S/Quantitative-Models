# Quantitative-Models
### Interest Rate Models: Vasicek and CIR Model Simulations

This repository contains Python code to simulate and analyze two widely used interest rate models: the Vasicek Model and the Cox-Ingersoll-Ross (CIR) Model. Both models are stochastic processes that describe the evolution of interest rates over time. By employing the Numba library for Just-In-Time (JIT) compilation, we significantly improve the performance of our simulations.

## Vasicek Model
The Vasicek Model is a mean-reverting, single-factor model for interest rates. It is described by the following stochastic differential equation (SDE):

$$ \begin{cases}
dr(t) = kappa * (theta - r(t)) * dt + sigma * dW(t)
\end{cases}$$

where:
`r(t)` is the interest rate at time `t`
`kappa` is the speed of mean reversion
`theta` is the long-term mean of the interest rate
`sigma` is the volatility of the interest rate
`dW(t)` is a standard Wiener process
This model assumes that interest rates are normally distributed and can assume negative values.

## CIR Model
The CIR Model is an extension of the Vasicek Model that prevents negative interest rates. The model is described by the following SDE:

$$ \begin{cases}
dr(t) = kappa * (theta - r(t)) * dt + sigma * sqrt(r(t)) * dW(t)
\end{cases}$$

Similar to the Vasicek Model, the CIR Model has a mean-reverting, single-factor structure. However, the key difference is the square root term `sqrt(r(t))`, which ensures that the model's interest rates remain non-negative.

## Simulation and Analysis
We generate Monte Carlo simulations for both the Vasicek Model and the CIR Model using the following parameters:

1. n_simulations: number of simulations
2. n_steps: number of time steps
3. dt: time step size
4. kappa: speed of mean reversion
5. theta: long-term mean of the interest rate
6. sigma: volatility of the interest rate
7. r0: initial interest rate

## Visualizations
We perform an in-depth analysis of the simulated interest rate paths and provide the following visualizations:

1. Interest Rate Paths: We plot the individual interest rate paths for both models, illustrating the mean-reverting behavior of the interest rates.

2. Mean and Standard Deviation: We overlay the mean and standard deviation of the interest rate paths on the individual paths to demonstrate the convergence of the interest rates to their long-term mean.

3. Autocorrelation: We calculate the autocorrelation of interest rates for both models, which provides insights into the persistence of interest rate shocks over time.

4. Interest Rate Distribution: We estimate the distribution of interest rates at the end of the simulation for both models. For the Vasicek Model, we observe a normal distribution, while the CIR Model exhibits a non-central chi-squared distribution.

5. Interest Rate Evolution: We visualize the evolution of interest rate distributions over time using kernel density estimation. This representation highlights the convergence of interest rates to their long-term mean.
