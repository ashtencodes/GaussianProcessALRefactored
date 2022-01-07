import numpy as np
import math

def parabola(noise):
    grid_bounds = [(-1, 1), (-1, 1)]
    grid_size = 21
    x_1 = np.linspace(grid_bounds[0][0], grid_bounds[0][1], grid_size)
    x_2 = x_1
    grid_diff = x_1[1] - x_1[0]
    x_plot, y_plot = np.meshgrid(x_1, x_2)
    z_plot = x_plot**2 + y_plot**2
    x_vec = np.array([x_plot.ravel()])
    y_vec = np.array([y_plot.ravel()])
    x_true = np.concatenate((x_vec.transpose(),y_vec.transpose()), axis=1)
    y_true = x_true[:, 0]**2 + x_true[:, 1]**2
    n_samples = len(y_true)
    if noise == "noise":
        y_obs = y_true + np.random.rand(n_samples) * math.sqrt(0.02)
        noise_name = 'noise'
    else:
        y_obs = y_true
        noise_name = 'none'
    func_name = 'parabola'

def townsend(noise):
    grid_bounds = [(-2.5, 2.5), (-2.5, 2.5)]
    grid_size = 51
    x_1 = np.linspace(grid_bounds[0][0], grid_bounds[0][1], grid_size)
    x_2 = x_1
    grid_diff = x_1[1] - x_1[0]
    x_plot, y_plot = np.meshgrid(x_1, x_2)
    z_plot = -(np.cos((x_plot - 0.1) * y_plot)) ** 2 - x_plot * np.sin(3 * x_plot + y_plot)
    x_vec = np.array([x_plot.ravel()])
    y_vec = np.array([y_plot.ravel()])
    x_true = np.concatenate((x_vec.transpose(), y_vec.transpose()), axis=1)
    y_true = -(np.cos((x_true[:, 0] - 0.1) * x_true[:, 1])) ** 2 - x_true[:, 0] * np.sin(
        3 * x_true[:, 0] + x_true[:, 1])
    n_samples = len(y_true)
    func_name = 'townsend'
    if noise == "noise":
        y_obs = y_true + np.random.rand(n_samples) * math.sqrt(0.02)
        noise_name = 'noise'
    else:
        y_obs = y_true
        noise_name = 'none'
    func_name = 'townsend'