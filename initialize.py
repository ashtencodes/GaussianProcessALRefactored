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

def parseTif():
    # %% parse geotiff files
    file_name1 = 'tifs/Shoemaker_5mDEM.tif'
    # You need to multiply 0.5 for each pixel value to get the actual elevation.
    Aimg = gdal.Open(file_name1)
    A = Aimg.GetRasterBand(1).ReadAsArray()

    file_name2 = 'tifs/Shoemaker_280mIceExposures.tif'
    Bimg = gdal.Open(file_name2)
    B = Bimg.GetRasterBand(1).ReadAsArray()

    file_name3 = 'tifs/Shoemaker_250mLAMP-OnOffRatio.tif'
    Cimg = gdal.Open(file_name3)
    C = Cimg.GetRasterBand(1).ReadAsArray()

    # make DEMs and other maps
    # to build a DEM, each index in row and column is 5 m
    (n_y, n_x) = np.shape(A)
    spacing = 5.0
    x_vec_grid5 = np.array(range(n_x)) * spacing
    y_vec_grid5 = np.array(range(n_y)) * spacing
    x_mat5, y_mat5 = np.meshgrid(x_vec_grid5, y_vec_grid5)
    z_mat5 = A / 2
    z_mat5 = np.where(z_mat5 == 32767 / 2, np.nan, z_mat5)
    z_min5 = min(z_mat5[~np.isnan(z_mat5)])
    z_max5 = max(z_mat5[~np.isnan(z_mat5)])

    # unravel grid data
    x_DEM5 = x_mat5.ravel()
    y_DEM5 = y_mat5.ravel()
    z_DEM5 = z_mat5.ravel()

    #  parse ice data distance 280 m
    (n_y, n_x) = np.shape(B)
    spacing = 280.0
    x_vec_grid280 = np.array(range(n_x)) * spacing
    y_vec_grid280 = np.array(range(n_y)) * spacing
    x_mat280, y_mat280 = np.meshgrid(x_vec_grid280, y_vec_grid280)
    z_mat280 = z_mat5[::56, ::56]
    z_mat280 = z_mat280[0:n_y, 0:n_x]

    # unravel grid data
    x_DEM280 = x_mat280.ravel()
    y_DEM280 = y_mat280.ravel()
    z_DEM280 = z_mat280.ravel()
    ice_DEM280 = B.ravel()

    #  parse LAMP data distance 250m
    (n_y, n_x) = np.shape(C)
    spacing = 250.0
    x_vec_grid250 = np.array(range(n_x)) * spacing
    y_vec_grid250 = np.array(range(n_y)) * spacing
    x_mat250, y_mat250 = np.meshgrid(x_vec_grid250, y_vec_grid250)
    z_mat250 = z_mat5[::50, ::50]
    # unravel grid data
    x_DEM250 = x_mat250.ravel()
    y_DEM250 = y_mat250.ravel()
    z_DEM250 = z_mat250.ravel()

    C = np.where(C == -9999, np.nan, C)
    c_min = min(C[~np.isnan(C)])
    c_max = max(C[~np.isnan(C)])
    c_DEM250 = C.ravel()
    # let's make LAMP data the elevation
    LAMP_DEM280 = np.zeros(len(x_DEM280))
    x_list = np.array([x_DEM250, y_DEM250]).transpose()
    for i in range(len(x_DEM280)):
        x_sample = np.array([x_DEM280[i], y_DEM280[i]])
        LAMP_DEM280[i] = nearest_neighbor(x_sample, x_list, c_DEM250)
    # %% clean up data
    # training data input is DEM position
    x_true = np.array([x_DEM250 / 1000, y_DEM250 / 1000, z_DEM250 / 1000]).transpose()
    # training data output is LAMP
    y_obs = np.double(c_DEM250)

    # get rid of elevation nan values
    y_obs = y_obs[~np.isnan(x_true[:, 2])]
    x_true = x_true[~np.isnan(x_true[:, 2]), :]
    # get rid of LAMP data
    x_true = x_true[~np.isnan(y_obs), :]
    y_obs = y_obs[~np.isnan(y_obs)]

    x_true_doub = x_true
    y_obs_doub = y_obs

    for i in range(x_true.shape[0]):
        y_obs_doub[i] = np.float64(y_obs[i])
        for j in range(x_true.shape[1]):
            x_true_doub[i, j] = np.float64(x_true[i, j])

    return x_true, x_true_doub, y_obs