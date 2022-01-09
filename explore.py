from GaussianProcess import *
from surfaces import *
from output import *
from initialize import *


def exploreSpiral(y_obs, x_true, y_true ,grid_bounds, grid_diff, grid_size, n_samples, func_name, noise_name, r_disp, dx):
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

    fig = plt.figure(figsize=(18, 12))

    r_vec = np.linspace(grid_bounds[0][0], grid_bounds[0][1], int((grid_size - 1) / dx + 1))
    i_spiral = make_spiral(r_vec, r_vec)
    x_spiral = r_vec[i_spiral[:, 0]]
    y_spiral = r_vec[i_spiral[:, 1]]
    i_train = []
    i_seq = list(range(0, n_samples))

    for i in range(len(i_spiral)):
        # find where the spiral coordinate lies along x_true
        i_sample = np.where((x_true[:, 0] == x_spiral[i]) & (x_true[:, 1] == y_spiral[i]))
        if not i_sample[0].size:
            print(x_spiral[i])
            print(y_spiral[i])
            continue
        i_train.append(int(i_sample[0]))

        x_train = torch.from_numpy(x_true[i_train])
        y_train = torch.from_numpy(y_obs[i_train])

        x_train = x_train.float()
        y_train = y_train.float()

        likelihood, model, optimizer, output, loss = GPtrain(x_train, y_train, training_iter)

        RBF_lengthscale.append(model.covar_module.base_kernel.lengthscale.item())
        RBF_noise.append(model.likelihood.noise.item())

        # Test points are regularly spaced centered along the last index bounded by index displacement
        i_con = sample_disp_con(x_true, x_true[i_train[-1]], grid_diff * 5)
        # if local_flag==1:
        x_test_local = torch.from_numpy(x_true[i_con, :])  # x_test is constrained to motion displacement
        # else:
        x_test_global = torch.from_numpy(x_true[i_seq, :])  # x_test is the entire dataset

        x_test_local = x_test_local.float()
        x_test_global = x_test_global.float()

        # Get into evaluation (predictive posterior) mode
        observed_pred_local, lower_local, upper_local = GPeval(x_test_local, model, likelihood)
        with torch.no_grad():
            f_preds = model(x_test_local)
            y_preds = likelihood(model(x_test_local))
            f_mean = f_preds.mean
            f_var_local = f_preds.variance
            f_covar = f_preds.covariance_matrix
        var_iter_local.append(max(f_var_local.numpy()))
        mse_local_true = sklearn.metrics.mean_squared_error(y_true[i_con], observed_pred_local.mean.numpy())
        rmse_local_true.append(math.sqrt(mse_local_true))
        mse_local_obs = sklearn.metrics.mean_squared_error(y_true[i_con], observed_pred_local.mean.numpy())
        rmse_local_obs.append(math.sqrt(mse_local_obs))

        observed_pred_global, lower_global, upper_global = GPeval(x_test_global, model, likelihood)
        with torch.no_grad():
            f_preds = model(x_test_global)
            y_preds = likelihood(model(x_test_global))
            f_mean = f_preds.mean
            f_var_global = f_preds.variance
            f_covar = f_preds.covariance_matrix
        var_iter_global.append(max(f_var_global.numpy()))
        mse_global_true = sklearn.metrics.mean_squared_error(y_true[i_seq], observed_pred_global.mean.numpy())
        rmse_global_true.append(math.sqrt(mse_global_true))
        mse_global_obs = sklearn.metrics.mean_squared_error(y_obs[i_seq], observed_pred_global.mean.numpy())
        rmse_global_obs.append(math.sqrt(mse_global_obs))

        # evaluate covariance properties
        covar_global.append(f_covar)
        covar_trace.append(np.trace(f_covar.detach().numpy()))
        covar_totelements.append(np.size(f_covar.detach().numpy()))
        covar_nonzeroelements.append(np.count_nonzero(f_covar.detach().numpy()))
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        # now evaluate information criteria
        # akaike information criterion
        AIC_sample = 2 * np.log(covar_nonzeroelements[-1]) - 2 * np.log(mse_global_true)
        AIC.append(AIC_sample)
        # BIC calculated from https://en.wikipedia.org/wiki/Bayesian_information_criterion#Gaussian_special_case
        BIC_sample = np.size(i_train) * np.log(covar_nonzeroelements[-1]) - 2 * np.log(mse_global_true)
        BIC.append(BIC_sample)

        # plot real surface and the observed measurements
        ax1, ax2, ax3, ax4, ax4a, ax5, ax5a, ax6 = plotGP(fig, x_true, i_sample, y_obs, i_train, observed_pred_global, x_test_global, x_test_local, upper_local, lower_local, var_iter_local, var_iter_global, rmse_local_true, rmse_global_true, RBF_lengthscale, RBF_noise, covar_trace, covar_totelements, covar_nonzeroelements, AIC, BIC)
        plt.show()

        # fig.tight_layout()
        fig.savefig(image_path + str(len(set(i_train))) + '.png')
        fig.clear()
    createVideo("Spiral", func_name, dx, noise_name, r_disp, "")
    outputMetric(i_train, x_true, y_true, observed_pred_global, x_test_global, rmse_global_true, rmse_global_obs, plt, rmse_local_true, var_iter_global, AIC, BIC, RBF_lengthscale, RBF_noise, covar_trace)

def exploreSnake(y_obs, x_true, y_true, grid_diff, grid_bounds, grid_size, n_samples, func_name, noise_name, r_disp, dx):
    i_train = []
    i_seq = list(range(0, n_samples))
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

    fig = plt.figure(figsize=(12, 10))

    x = np.linspace(grid_bounds[0][0], grid_bounds[0][1], int((grid_size - 1) / dx + 1))
    y_plus = np.linspace(
        grid_bounds[0][0], grid_bounds[0][1], int((grid_size - 1) / dx + 1))
    y_minus = np.linspace(
        grid_bounds[0][1], grid_bounds[0][0], int((grid_size - 1) / dx + 1))
    y_status = 'forward'

    for x_i in x:
        i_x = np.argwhere(x_true[:, 0] == x_i)[:, 0]
        if y_status == 'forward':
            y = y_plus
            y_status = 'reverse'
        else:
            y = y_minus
            y_status = 'forward'
        for y_j in y:
            i_y = np.argwhere(np.abs(x_true[:, 1] - y_j) < 1e-5)[:, 0]

            # find where the spiral coordinate lies along x_true
            i_sample = list(set(i_x) & set(i_y))

            if not i_sample[0].size:
                continue
            i_train.append(int(i_sample[0]))

            x_train = torch.from_numpy(x_true[i_train])
            y_train = torch.from_numpy(y_obs[i_train])

            x_train = x_train.float()
            y_train = y_train.float()

            likelihood, model, optimizer, output, loss = GPtrain(
                x_train, y_train, training_iter)

            RBF_lengthscale.append(model.covar_module.base_kernel.lengthscale.item())
            RBF_noise.append(model.likelihood.noise.item())

            # Test points are regularly spaced centered along the last index bounded by index displacement
            i_con = sample_disp_con(x_true, x_true[i_train[-1]], grid_diff * 4)
            # if local_flag==1:
            # x_test is constrained to motion displacement
            x_test_local = torch.from_numpy(x_true[i_con, :])
            # else:
            # x_test is the entire dataset
            x_test_global = torch.from_numpy(x_true[i_seq, :])

            x_test_local = x_test_local.float()
            x_test_global = x_test_global.float()

            # Get into evaluation (predictive posterior) mode
            observed_pred_local, lower_local, upper_local = GPeval(
                x_test_local, model, likelihood)
            with torch.no_grad():
                f_preds = model(x_test_local)
                y_preds = likelihood(model(x_test_local))
                f_mean = f_preds.mean
                f_var_local = f_preds.variance
                f_covar = f_preds.covariance_matrix
            var_iter_local.append(max(f_var_local.numpy()))
            mse_local_true = sklearn.metrics.mean_squared_error(
                y_true[i_con], observed_pred_local.mean.numpy())
            rmse_local_true.append(math.sqrt(mse_local_true))
            mse_local_obs = sklearn.metrics.mean_squared_error(
                y_true[i_con], observed_pred_local.mean.numpy())
            rmse_local_obs.append(math.sqrt(mse_local_obs))

            observed_pred_global, lower_global, upper_global = GPeval(
                x_test_global, model, likelihood)
            with torch.no_grad():
                f_preds = model(x_test_global)
                y_preds = likelihood(model(x_test_global))
                f_mean = f_preds.mean
                f_var_global = f_preds.variance
                f_covar = f_preds.covariance_matrix
            var_iter_global.append(max(f_var_global.numpy()))
            mse_global_true = sklearn.metrics.mean_squared_error(
                y_true[i_seq], observed_pred_global.mean.numpy())
            rmse_global_true.append(math.sqrt(mse_global_true))
            mse_global_obs = sklearn.metrics.mean_squared_error(
                y_obs[i_seq], observed_pred_global.mean.numpy())
            rmse_global_obs.append(math.sqrt(mse_global_obs))

            # evaluate covariance properties
            covar_global.append(f_covar)
            covar_trace.append(np.trace(f_covar.detach().numpy()))
            covar_totelements.append(np.size(f_covar.detach().numpy()))
            covar_nonzeroelements.append(np.count_nonzero(f_covar.detach().numpy()))
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

            # now evaluate information criteria
            # akaike information criterion
            AIC_sample = 2 * np.log(covar_nonzeroelements[-1]) - 2 * np.log(mse_global_true)
            AIC.append(AIC_sample)
            # BIC calculated from https://en.wikipedia.org/wiki/Bayesian_information_criterion#Gaussian_special_case
            BIC_sample = np.size(i_train) * np.log(covar_nonzeroelements[-1]) - 2 * np.log(mse_global_true)
            BIC.append(BIC_sample)

            # plot real surface and the observed measurements
            ax1, ax2, ax3, ax4, ax4a, ax5, ax5a, ax6 = plotGP(fig, x_true, i_sample, y_obs, i_train,
                                                              observed_pred_global, x_test_global, x_test_local,
                                                              upper_local, lower_local, var_iter_local, var_iter_global,
                                                              rmse_local_true, rmse_global_true, RBF_lengthscale,
                                                              RBF_noise, covar_trace, covar_totelements,
                                                              covar_nonzeroelements, AIC, BIC)
            plt.show()

            # fig.tight_layout()
            fig.savefig(image_path + str(len(set(i_train))) + '.png')
            fig.clear()

    createVideo("Snake", func_name, dx, noise_name, r_disp, "")
    outputMetric()

def exploreGPAL(func_name, y_obs, x_true, y_true, grid_diff, n_samples, flagIn, r_disp):
    local_flag = flagIn
    if local_flag == "local":
        explore_name = 'local'
    elif local_flag == "global":
        explore_name = 'global'
    elif local_flag == "NN":
        explore_name = "NN"

    # %% randomly initialize location
    i_0 = random.randrange(n_samples)
    i_train = []
    i_train.append(i_0)
    i_seq = list(range(0, n_samples))

    # randomly sample next 10 data points with a displacement constraint of 10int
    r_NN = np.sqrt(2) * (grid_diff + 1e-5)
    r_con = 3 * r_NN
    for i in range(10):
        i_sample_set = sample_disp_con(x_true, x_true[i_train[-1]], r_NN)
        i_sample = i_sample_set[random.randrange(len(i_sample_set))]
        i_train.append(int(i_sample))
    i_train = list(set(i_train))
    # %% hyperparameters for exploration training
    training_iter = 100
    sample_iter = int(n_samples / 3)
    var_iter = []
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

    # initialize animation
    fig = plt.figure(figsize=(18, 12))

    for j in range(sample_iter):
        # define training data
        x_train = torch.from_numpy(x_true[i_train])
        y_train = torch.from_numpy(y_obs[i_train])

        # train model with GPyTorch model, which optimizes hyperparameters
        likelihood, model, optimizer, output, loss = GPtrain(x_train, y_train, training_iter)
        # store optimal hyperparameters
        RBF_lengthscale.append(model.covar_module.base_kernel.lengthscale.item())
        RBF_noise.append(model.likelihood.noise.item())

        # Test points are regularly spaced centered along the last index bounded by index displacement
        i_con = sample_disp_con(x_true, x_true[i_train[-1]], r_con)
        x_test_local = torch.from_numpy(x_true[i_con, :])
        x_test_global = torch.from_numpy(x_true[i_seq, :])

        # Evaluate RMS for local
        observed_pred_local, lower_local, upper_local = GPeval(x_test_local, model, likelihood)
        with torch.no_grad():
            f_preds = model(x_test_local)
            y_preds = likelihood(model(x_test_local))
            f_mean = f_preds.mean
            f_var_local = f_preds.variance
            f_covar = f_preds.covariance_matrix
        var_iter_local.append(max(f_var_local.numpy()))
        mse_local_true = sklearn.metrics.mean_squared_error(y_true[i_con], observed_pred_local.mean.numpy())
        rmse_local_true.append(math.sqrt(mse_local_true))
        mse_local_obs = sklearn.metrics.mean_squared_error(y_true[i_con], observed_pred_local.mean.numpy())
        rmse_local_obs.append(math.sqrt(mse_local_obs))
        # and global
        observed_pred_global, lower_global, upper_global = GPeval(x_test_global, model, likelihood)
        with torch.no_grad():
            f_preds = model(x_test_global)
            y_preds = likelihood(model(x_test_global))
            f_mean = f_preds.mean
            f_var_global = f_preds.variance
            f_covar = f_preds.covariance_matrix
        var_iter_global.append(max(f_var_global.numpy()))
        mse_global_true = sklearn.metrics.mean_squared_error(y_true[i_seq], observed_pred_global.mean.numpy())
        rmse_global_true.append(math.sqrt(mse_global_true))
        mse_global_obs = sklearn.metrics.mean_squared_error(y_obs[i_seq], observed_pred_global.mean.numpy())
        rmse_global_obs.append(math.sqrt(mse_global_obs))

        # evaluate covariance properties
        covar_global.append(f_covar)
        covar_trace.append(np.trace(f_covar.detach().numpy()))
        covar_totelements.append(np.size(f_covar.detach().numpy()))
        covar_nonzeroelements.append(np.count_nonzero(f_covar.detach().numpy()))
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        # now evaluate information criteria
        # akaike information criterion
        AIC_sample = 2 * np.log(covar_nonzeroelements[-1]) - 2 * np.log(mse_global_true)
        AIC.append(AIC_sample)
        # BIC calculated from https://en.wikipedia.org/wiki/Bayesian_information_criterion#Gaussian_special_case
        BIC_sample = np.size(i_train) * np.log(covar_nonzeroelements[-1]) - 2 * np.log(mse_global_true)
        BIC.append(BIC_sample)

        # plot real surface and the observed measurements
        ax1, ax2, ax3, ax4, ax4a, ax5, ax5a, ax6 = plotGP(fig, x_true, i_sample, y_obs, i_train, observed_pred_global, x_test_global, x_test_local, upper_local, lower_local, var_iter_local, var_iter_global, rmse_local_true, rmse_global_true, RBF_lengthscale, RBF_noise, covar_trace, covar_totelements, covar_nonzeroelements, AIC, BIC)
        plt.show()

        # fig.tight_layout()
        fig.savefig(image_path + str(len(set(i_train))) + '.png')
        fig.clear()

        if local_flag == "local":
            # waypoint within r_con with maximum variance, nearest neighbor along the way
            uncertainty = upper_local - lower_local
            i_max = np.argmax(uncertainty)
            x_max = x_test_local[i_max, :].numpy()
            i_NN = sample_disp_con(x_true, x_true[i_train[-1]], r_NN)
            dx_NN = np.sqrt((x_true[i_NN, 0] - x_max[0]) ** 2 + (x_true[i_NN, 1] - x_max[1]) ** 2)
            i_dx = np.argsort(dx_NN)
        else:
            # waypoint within entire space with max variance, nearest neighbor
            uncertainty = upper_global - lower_global
            i_max = np.argmax(uncertainty)
            x_max = x_test_global[i_max, :].numpy()
            i_NN = sample_disp_con(x_true, x_true[i_train[-1]], r_NN)
            dx_NN = np.sqrt((x_true[i_NN, 0] - x_max[0]) ** 2 + (x_true[i_NN, 1] - x_max[1]) ** 2)
            i_dx = np.argsort(dx_NN)

        i_sample = []
        j = 0
        while not np.array(i_sample).size:
            i_sample = unique_sample(i_NN[i_dx[j]], i_con, i_train, n_samples - 1, x_true)
            j = j + 1
            print(i_sample)
        i_train.append(int(i_sample))

    createVideo("GPAL", func_name, "noise", r_disp, explore_name)
    outputMetric()
