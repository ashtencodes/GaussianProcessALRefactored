from initialize import *
from surfaces import *


def GPtrain(x_train, y_train, training_iter):
    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(x_train, y_train, likelihood)
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},  # Includes GaussianLikelihood parameters
    ], lr=0.1)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # train GP moel
    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(x_train)
        # Calc loss and backprop gradients
        loss = -mll(output, y_train)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
            i + 1, training_iter, loss.item(),
            model.covar_module.base_kernel.lengthscale.item(),
            model.likelihood.noise.item()
        ))
        optimizer.step()
    return likelihood, model, optimizer, output, loss


def GPeval(x_test, model, likelihood):
    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()

    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(x_test))

    f_preds = model(x_test)
    y_preds = likelihood(model(x_test))
    f_mean = f_preds.mean
    f_var = f_preds.variance
    f_covar = f_preds.covariance_matrix

    with torch.no_grad():
        # Get upper and lower confidence bounds
        lower, upper = observed_pred.confidence_region()

    return observed_pred, lower, upper


def plotGP(fig, x_true, i_sample, y_obs, i_train, observed_pred_global, x_test_global, x_test_local, upper_local, lower_local, var_iter_local, var_iter_global, rmse_local_true, rmse_global_true, RBF_lengthscale, RBF_noise, covar_trace, covar_totelements, covar_nonzeroelements, AIC, BIC):
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    rover = ax1.scatter3D(x_true[i_sample, 0], x_true[i_sample, 1],
                          y_obs[i_sample], s=100, color='black', marker='*', zorder=1)
    rover_path = ax1.plot3D(
        x_true[i_train, 0], x_true[i_train, 1], y_obs[i_train], color='black')
    # color by LAMP
    surf = ax1.plot_trisurf(x_true[:, 0], x_true[:, 1], y_obs, cmap='inferno',
                            linewidth=0, alpha=0.25, vmax=max(y_obs), vmin=min(y_obs))
    points_pred = ax1.plot_trisurf(
        x_test_global[:, 0], x_test_global[:, 1], observed_pred_global.mean.numpy(), color='grey', alpha=0.25)
    # Shade between the lower and upper confidence bounds
    for i_test in range(len(x_test_local)):
        ax1.plot(x_test_local[i_test, 0].numpy() * np.array([1, 1]), x_test_local[i_test, 1].numpy() * np.array(
            [1, 1]), np.array([lower_local[i_test].numpy(), upper_local[i_test].numpy()]), 'gray')
    # for i_test in range(len(x_test_global)):
    #     ax2.plot(x_test_global[i_test,0].numpy()*np.array([1, 1]), x_test_global[i_test,1].numpy()*np.array([1, 1]), np.array([lower_global[i_test].numpy(), upper_global[i_test].numpy()]),'gray')
    # ax.legend(['Observed', 'Ice', 'Predictions w/ '+str(len(set(i_train)))+' Samples'],loc=1)
    ax1.view_init(20, 20)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('surface')
    ax1.set_title('rover on surface ' + str(x_true[i_sample, :]))
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(range(0, len(var_iter_local)),
             var_iter_local, color='blue', marker='.')
    ax2.plot(range(0, len(var_iter_global)),
             var_iter_global, color='black', marker='*')
    ax2.set_xlabel('number of samples')
    ax2.set_title('variance of samples')
    ax2.legend(['local', 'global'], loc='upper right')
    ax3 = fig.add_subplot(2, 3, 3)
    local_rms = ax3.plot(range(0, len(rmse_local_true)),
                         rmse_local_true, color='blue', marker='.', label='local')
    global_rms = ax3.plot(range(0, len(rmse_global_true)),
                          rmse_global_true, color='black', marker='*', label='global')
    ax3.set_xlabel('number of samples')
    ax3.legend(['local', 'global'], loc='upper right')
    ax3.set_title('rmse of learned model')
    ax4 = fig.add_subplot(2, 3, 4)
    lengthscale = ax4.plot(range(0, len(RBF_lengthscale)),
                           RBF_lengthscale, color='pink', marker='.', label='lengthscale')
    ax4a = ax4.twinx()
    noise = ax4a.plot(range(0, len(RBF_noise)),
                      RBF_noise, color='green', marker='.', label='noise')
    ax4.set_xlabel('number of samples')
    ax4.set_ylabel('lengthscale', color='pink')
    ax4a.set_ylabel('noise', color='green')
    ax4.set_title('hyperparameters of learned model')
    ax5 = fig.add_subplot(2, 3, 5)
    trace = ax5.plot(range(0, len(covar_trace)),
                     covar_trace, color='pink', marker='.', label='trace')
    ax5.set_xlabel('number of samples')
    ax5.set_ylabel('trace', color='pink')
    ax5.set_xlabel('number of samples')
    ax5.set_ylabel('trace', color='pink')
    ax5a = ax5.twinx()
    tot_elements = ax5a.plot(range(0, len(covar_totelements)),
                             covar_totelements, color='green', marker='.', label='total_elements')
    nonzero_elements = ax5a.plot(range(0, len(covar_nonzeroelements)),
                                 covar_nonzeroelements, color='blue', marker='.', label='nonzero_elements')
    ax5a.set_ylabel('elements', color='green')
    ax5a.legend(['total', 'nonzero'], loc='lower right')
    ax6 = fig.add_subplot(2, 3, 6)
    AIC_plot = ax6.plot(range(0, len(AIC)), AIC, color='blue', marker='.', label='AIC')
    BIC_plot = ax6.plot(range(0, len(BIC)), BIC, color='black', marker='*', label='BIC')
    ax6.set_xlabel('number of samples')
    ax6.legend(['AIC', 'BIC'], loc='lower right')
    ax6.set_title('information criteria of learned model')

    return ax1, ax2, ax3, ax4, ax4a, ax5, ax5a, ax6

def plotLAMPGP(fig, x_true, i_sample, i_train, y_obs, x_test_global, observed_pred_global, x_test_local, lower_local, upper_local, var_iter_global, var_iter_local, rmse_local, rmse_global, RBF_lengthscale, RBF_noise, covar_trace, covar_totelements, covar_nonzeroelements, AIC, BIC):
    ax1 = fig.add_subplot(2, 4, 1, projection='3d')
    # color by values
    rover = ax1.scatter3D(x_true[i_sample, 0], x_true[i_sample, 1], x_true[i_sample, 2], s=100, color='black',
                          marker='*', zorder=1)
    rover_path = ax1.plot3D(x_true[i_train, 0], x_true[i_train, 1], x_true[i_train, 2], color='black')
    LAMP_surf = ax1.plot_trisurf(x_true[:, 0], x_true[:, 1], x_true[:, 2], color='grey', linewidth=1, alpha=0.25)
    LAMP = ax1.scatter3D(x_true[:, 0], x_true[:, 1], x_true[:, 2], c=y_obs, s=15, cmap=cm.inferno, alpha=0.25,
                         vmax=max(y_obs), vmin=min(y_obs))
    # ice = ax1.scatter3D(x_ice[:,0], x_ice[:,1], x_ice[:,2], color='green', s=10)
    ax1.view_init(20, 20)
    # ax.legend(['DEM', 'LAMP', 'Current Position'],loc=4)
    ax1.set_title('LAMP data on DEM; position' + str(x_true[i_sample, :]))
    ax1.set_xlabel('x [km]')
    ax1.set_ylabel('y [km]')
    ax1.set_zlabel('z [km]')
    set_axes_equal(ax1)
    # plot the surface
    ax2 = fig.add_subplot(2, 4, 2, projection='3d')
    # color by LAMP
    LAMP = ax2.plot_trisurf(x_true[:, 0], x_true[:, 1], y_obs, cmap='inferno', linewidth=0, alpha=0.25, vmax=max(y_obs),
                            vmin=min(y_obs))
    # ice = ax2.scatter3D(x_ice[:,0], x_ice[:,1], x_ice[:,3], color='green', s=10)
    # ice = ax.scatter3D(x_DEM280, y_DEM280, C_DEM280, color = 'green')
    points_pred = ax2.plot_trisurf(x_test_global[:, 0], x_test_global[:, 1], observed_pred_global.mean.numpy(),
                                   color='grey', alpha=0.25)
    # Shade between the lower and upper confidence bounds
    for i_test in range(len(x_test_local)):
        ax2.plot(x_test_local[i_test, 0].numpy() * np.array([1, 1]), x_test_local[i_test, 1].numpy() * np.array([1, 1]),
                 np.array([lower_local[i_test].numpy(), upper_local[i_test].numpy()]), 'gray')
    # for i_test in range(len(x_test_global)):
    #     ax2.plot(x_test_global[i_test,0].numpy()*np.array([1, 1]), x_test_global[i_test,1].numpy()*np.array([1, 1]), np.array([lower_global[i_test].numpy(), upper_global[i_test].numpy()]),'gray')
    # ax.legend(['Observed', 'Ice', 'Predictions w/ '+str(len(set(i_train)))+' Samples'],loc=1)
    ax2.view_init(20, 20)
    ax2.set_xlabel('x [km]')
    ax2.set_ylabel('y [km]')
    ax2.set_zlabel('LAMP data')
    fig.colorbar(LAMP, ax=ax2)
    # set_axes_equal(ax2)
    # ax2.set_zlim3d([0, 2.5])
    ax2.set_title('LAMP measured vs predictions')
    ax3 = fig.add_subplot(2, 4, 3)
    ax3.plot(range(0, len(var_iter_local)), var_iter_local, color='blue', marker='.')
    ax3.plot(range(0, len(var_iter_global)), var_iter_global, color='black', marker='*')
    ax3.set_xlabel('number of samples')
    ax3.set_title('variance of samples')
    ax3.legend(['local', 'global'], loc='upper right')
    ax4 = fig.add_subplot(2, 4, 4)
    local_rms = ax4.plot(range(0, len(rmse_local)), rmse_local, color='blue', marker='.', label='local')
    global_rms = ax4.plot(range(0, len(rmse_global)), rmse_global, color='black', marker='*', label='global')
    ax4.set_xlabel('number of samples')
    ax4.legend(['local', 'global'], loc='upper right')
    ax4.set_title('rmse of learned model')
    ax5 = fig.add_subplot(2, 4, 5)
    lengthscale = ax5.plot(range(0, len(RBF_lengthscale)),
                           RBF_lengthscale, color='pink', marker='.', label='lengthscale')
    ax5a = ax5.twinx()
    noise = ax5a.plot(range(0, len(RBF_noise)),
                      RBF_noise, color='green', marker='.', label='noise')
    ax5.set_xlabel('number of samples')
    ax5.set_ylabel('lengthscale', color='pink')
    ax5a.set_ylabel('noise', color='green')
    ax5.set_title('hyperparameters of learned model')
    ax6 = fig.add_subplot(2, 4, 6)
    trace = ax6.plot(range(0, len(covar_trace)),
                     covar_trace, color='pink', marker='.', label='trace')
    ax6.set_xlabel('number of samples')
    ax6.set_ylabel('trace', color='pink')
    ax6.set_xlabel('number of samples')
    ax6.set_ylabel('trace', color='pink')
    ax6a = ax6.twinx()
    tot_elements = ax6a.plot(range(0, len(covar_totelements)),
                             covar_totelements, color='green', marker='.', label='total_elements')
    nonzero_elements = ax6a.plot(range(0, len(covar_nonzeroelements)),
                                 covar_nonzeroelements, color='blue', marker='.', label='nonzero_elements')
    ax6a.set_ylabel('elements', color='green')
    ax6a.legend(['total', 'nonzero'], loc='lower right')
    ax7 = fig.add_subplot(2, 4, 7)
    AIC_plot = ax7.plot(range(0, len(AIC)), AIC, color='blue', marker='.', label='AIC')
    BIC_plot = ax7.plot(range(0, len(BIC)), BIC, color='black', marker='*', label='BIC')
    ax7.set_xlabel('number of samples')
    ax7.legend(['AIC', 'BIC'], loc='lower right')
    ax7.set_title('information criteria of learned model')

    return ax1, ax2, ax3, ax4, ax5, ax5a, ax6, ax6a, ax7