import math
import torch
import gpytorch
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import random
import os
import cv2
import re
from osgeo import gdal
import sklearn.metrics
import pickle

# demonstrate using Gaussian Processes to sample a 2D value space over 1D parameter space in a reinforcement learning framework
# generate a toy surface to sample and navigate
# sample the space iteratively
#   initialization:
#       random initial conditions (position and value)
#   enter exploration phase:
#       maximize uncertainty (stdev) in next measurement within motion constraints
#   enter exploitation phase:
#       maximize value (exp) in next measurement within motion constraints

# %% function defs


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def nearest_neighbor(x_sample, x_list, y_list):
    disp = np.sqrt((x_sample[0]-x_list[:, 0])**2+(x_sample[1]-x_list[:, 1])**2)
    i_min = np.argmin(disp)
    y_sample = y_list[i_min]
    return y_sample

# define the simplest form of GP model, exact inference


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def unique_sample(i_sample, i_set, i_train, i_max, x):
    if i_sample <= i_max and i_sample >= 0:
        if i_sample not in i_train:
            i_new = i_sample
        else:
            i_set_unique = set(i_set)-set(i_train)
            if not i_set_unique:
                return "empty"
            i_set_unique = list(i_set_unique)
            x_start = x[i_sample, :]
            x_disp = np.sqrt((x[i_set_unique, 0]-x_start[0])**2 + (
                x[i_set_unique, 1]-x_start[1])**2 + (x[i_set_unique, 2]-x_start[2])**2)
            # disp_i = np.abs(np.array(i_set_unique)-np.array(i_sample))
            i_new = i_set_unique[np.argmin(x_disp)]
    elif i_sample > i_max:
        i_new = unique_sample(i_sample-1, i_set, i_train, i_max)
    else:
        i_new = unique_sample(i_sample+1, i_set, i_train, i_max)
    return i_new


def sample_disp_con(x, x_start, r_disp):
    # x_start = x[i_start,:]
    x_disp = np.sqrt((x[:, 0]-x_start[0])**2 + (x[:, 1]-x_start[1])**2)
    i_con = np.argwhere(x_disp <= r_disp)
    i_con = np.sort(i_con)
    return list(i_con[:, 0])

fig = plt.figure(figsize=(18, 12))

dx = 2
r_vec = np.linspace(grid_bounds[0][0],grid_bounds[0][1],int((grid_size-1)/dx+1))
i_spiral = make_spiral(r_vec,r_vec)
x_spiral = r_vec[i_spiral[:,0]]
y_spiral = r_vec[i_spiral[:,1]]
i_train = []
i_seq = list(range(0,n_samples))

training_iter = 100
var_iter_local = []
var_iter_global = []
rmse_local_obs = []
rmse_global_obs = []
rmse_local_true = []
rmse_global_true = []
RBF_lengthscale = []
RBF_noise = []
covar_global = []
covar_trace = []
covar_totelements = []
covar_nonzeroelements = []
AIC = []
BIC = []