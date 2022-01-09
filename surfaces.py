import numpy as np
import math
from spiral import *
from explore import *
from LAMP import *
from initialize import *

def parabola(strategy, dxInput, noise, gpflag="local"):
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
    dx = int(dxInput)

    if strategy == "spiral":
        exploreSpiral(y_obs, x_true, y_true ,grid_bounds, grid_diff, grid_size, n_samples, func_name, noise_name, 3.0, dx)
    elif strategy == "snake":
        exploreSnake(y_obs, x_true, y_true, grid_diff, grid_bounds, grid_size, n_samples, func_name, noise_name, 3.0, dx)
    elif strategy == "gpal":
        exploreGPAL(func_name, y_obs, x_true, y_true, grid_diff, n_samples, gpflag, 3.0)


def townsend(strategy, dxInput, noise, gpflag="local"):
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
    if noise == "noise":
        y_obs = y_true + np.random.rand(n_samples) * math.sqrt(0.02)
        noise_name = 'noise'
    else:
        y_obs = y_true
        noise_name = 'none'
    func_name = 'townsend'
    dx = int(dxInput)

    if strategy == "spiral":
        exploreSpiral(y_obs, x_true, y_true ,grid_bounds, grid_diff, grid_size, n_samples, func_name, noise_name, 3.0, dx)
    elif strategy == "snake":
        exploreSnake(y_obs, x_true, y_true, grid_diff, grid_bounds, grid_size, n_samples, func_name, noise_name, 3.0, dx)
    elif strategy == "gpal":
        exploreGPAL(func_name, y_obs, x_true, y_true, grid_diff, n_samples, gpflag, 3.0)

def lunar(strategy, dxInput, r_dispIn, gpflag="local"):
    x_true, x_true_doub, y_obs = parseTif()

    r_disp = int(r_dispIn)
    r_NN = np.sqrt(3) * 0.25
    r_con = 3 * r_NN

    x_center_all = np.mean(x_true, 0)
    x_disp = np.sqrt((x_true[:, 0] - x_center_all[0]) ** 2 + (x_true[:, 1] - x_center_all[1]) ** 2 + (
                x_true[:, 2] - x_center_all[2]) ** 2)
    i_min = np.argmin(x_disp)
    x_center = x_true[i_min, :]

    x_true = x_true_doub - x_center

    y_obs = y_obs[np.argwhere(x_true[:, 0] >= -r_disp / 2)[:, 0]]
    x_true = x_true[np.argwhere(x_true[:, 0] >= -r_disp / 2)[:, 0]]
    y_obs = y_obs[np.argwhere(x_true[:, 1] >= -r_disp / 2)[:, 0]]
    x_true = x_true[np.argwhere(x_true[:, 1] >= -r_disp / 2)[:, 0]]
    y_obs = y_obs[np.argwhere(x_true[:, 0] <= r_disp / 2)[:, 0]]
    x_true = x_true[np.argwhere(x_true[:, 0] <= r_disp / 2)[:, 0]]
    y_obs = y_obs[np.argwhere(x_true[:, 1] <= r_disp / 2)[:, 0]]
    x_true = x_true[np.argwhere(x_true[:, 1] <= r_disp / 2)[:, 0]]

    n_samples = len(y_obs)

    dx = int(dxInput)

    if strategy == "spiral":
        exploreLAMPSpiral(dx, r_disp, n_samples, x_true, y_obs, r_con)
    elif strategy == "snake":
        exploreLAMPSnake(dx, n_samples, r_disp, x_true, y_obs, r_con)
    elif strategy == "gpal":
        exploreLAMPGPAL(n_samples, x_true, y_obs, gpflag, r_disp)