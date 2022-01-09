import GaussianProcess
from output import *
from initialize import *
from spiral import *

def exploreLAMPSpiral(dxInput, r_disp, n_samples, x_true, y_obs, r_con):
    dx = dxInput
    r_vec = np.linspace(-r_disp / 2, r_disp / 2, int(4 * r_disp / dx + 1))
    i_spiral = make_spiral(r_vec, r_vec)
    x_spiral = r_vec[i_spiral[:, 0]]
    y_spiral = r_vec[i_spiral[:, 1]]
    i_train = []
    i_seq = list(range(0, n_samples))

    training_iter = 100
    var_iter_local = []
    var_iter_global = []
    rmse_local = []
    rmse_global = []
    RBF_lengthscale = []
    RBF_noise = []
    covar_global = []
    covar_trace = []
    covar_totelements = []
    covar_nonzeroelements = []
    AIC = []
    BIC = []

    # initialize animation
    fig = plt.figure(figsize=(12, 10))
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    for i in range(len(i_spiral)):
        # find where the spiral coordinate lies along x_true
        i_sample = np.where((x_true[:, 0] == x_spiral[i]) & (x_true[:, 1] == y_spiral[i]))
        if not i_sample[0].size:
            continue
        i_train.append(int(i_sample[0]))

        x_train = torch.from_numpy(x_true[i_train])
        y_train = torch.from_numpy(y_obs[i_train])

        x_train = x_train.float()
        y_train = y_train.float()

        likelihood, model, optimizer, output, loss = GaussianProcess.GPtrain(x_train, y_train, training_iter)

        RBF_lengthscale.append(model.covar_module.base_kernel.lengthscale.item())
        RBF_noise.append(model.likelihood.noise.item())

        # Test points are regularly spaced centered along the last index bounded by index displacement
        i_con = sample_disp_con(x_true, x_true[i_train[-1]], r_con)
        # if local_flag==1:
        x_test_local = torch.from_numpy(x_true[i_con, :])  # x_test is constrained to motion displacement
        # else:
        x_test_global = torch.from_numpy(x_true[i_seq, :])  # x_test is the entire dataset

        x_test_local = x_test_local.float()
        x_test_global = x_test_global.float()

        # Get into evaluation (predictive posterior) mode
        observed_pred_local, lower_local, upper_local = GaussianProcess.GPeval(x_test_local, model, likelihood)
        with torch.no_grad():
            f_preds = model(x_test_local)
            y_preds = likelihood(model(x_test_local))
            f_mean = f_preds.mean
            f_var_local = f_preds.variance
            f_covar = f_preds.covariance_matrix
        var_iter_local.append(max(f_var_local.numpy()))
        mse_local = sklearn.metrics.mean_squared_error(y_obs[i_con], observed_pred_local.mean.numpy())
        rmse_local.append(math.sqrt(mse_local))

        observed_pred_global, lower_global, upper_global = GaussianProcess.GPeval(x_test_global, model, likelihood)
        with torch.no_grad():
            f_preds = model(x_test_global)
            y_preds = likelihood(model(x_test_global))
            f_mean = f_preds.mean
            f_var_global = f_preds.variance
            f_covar = f_preds.covariance_matrix
        var_iter_global.append(max(f_var_global.numpy()))
        mse_global = sklearn.metrics.mean_squared_error(y_obs[i_seq], observed_pred_global.mean.numpy())
        rmse_global.append(math.sqrt(mse_global))

        # evaluate covariance properties
        covar_global.append(f_covar)
        covar_trace.append(np.trace(f_covar.detach().numpy()))
        covar_totelements.append(np.size(f_covar.detach().numpy()))
        covar_nonzeroelements.append(np.count_nonzero(f_covar.detach().numpy()))
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        # now evaluate information criteria
        # akaike information criterion
        AIC_sample = 2 * np.log(covar_nonzeroelements[-1]) - 2 * np.log(mse_global)
        AIC.append(AIC_sample)
        # BIC calculated from https://en.wikipedia.org/wiki/Bayesian_information_criterion#Gaussian_special_case
        BIC_sample = np.size(i_train) * np.log(covar_nonzeroelements[-1]) - 2 * np.log(mse_global)
        BIC.append(BIC_sample)

        # plot real surface and the observed measurements
        ax1, ax2, ax3, ax4, ax5, ax5a, ax6, ax6a, ax7 = GaussianProcess.plotLAMPGP(fig, x_true, i_sample, i_train, y_obs, x_test_global, observed_pred_global, x_test_local, lower_local, upper_local, var_iter_global, var_iter_local, rmse_local, rmse_global, RBF_lengthscale, RBF_noise, covar_trace, covar_totelements, covar_nonzeroelements, AIC, BIC)
        plt.show()

        # fig.tight_layout()
        fig.savefig(image_path + str(len(set(i_train))) + '.png')
        fig.clear()

    rmse_global_true = rmse_global
    rmse_global_obs = rmse_global
    rmse_local_true = rmse_local

    createVideo("Spiral", "LAMP", "noise", r_disp, dx)
    outputMetricLAMP(i_train, x_true, observed_pred_global, x_test_global, rmse_global, plt, rmse_local, var_iter_global, RBF_lengthscale, RBF_noise, covar_trace, AIC, BIC)


def exploreLAMPSnake(dxInput, n_samples, r_disp, x_true, y_obs, r_con):
    # %% snake
    dx = dxInput
    # start from x_min y_min
    i_train = []
    i_seq = list(range(0, n_samples))
    x_min = -r_disp / 2
    y_min = -r_disp / 2
    x_max = r_disp / 2
    y_max = r_disp / 2

    training_iter = 100
    var_iter_local = []
    var_iter_global = []
    rmse_local = []
    rmse_global = []
    RBF_lengthscale = []
    RBF_noise = []
    covar_global = []
    covar_trace = []
    covar_totelements = []
    covar_nonzeroelements = []
    AIC = []
    BIC = []

    x = np.linspace(x_min, x_max, int(4 * r_disp / dx + 1))
    y_plus = np.linspace(y_min, y_max, int(4 * r_disp / dx + 1))
    y_minus = np.linspace(y_max, y_min, int(4 * r_disp / dx + 1))
    y_status = 'forward'

    # initialize animation
    fig = plt.figure(figsize=(12, 10))
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    for x_i in x:
        i_x = np.argwhere(x_true[:, 0] == x_i)[:, 0]
        if y_status == 'forward':
            y = y_plus
            y_status = 'reverse'
        else:
            y = y_minus
            y_status = 'forward'
        for y_j in y:
            i_y = np.argwhere(x_true[:, 1] == y_j)[:, 0]

            i_sample = list(set(i_x) & set(i_y))
            if not i_sample:
                continue
            i_train.append(i_sample[0])

            x_train = torch.from_numpy(x_true[i_train])
            y_train = torch.from_numpy(y_obs[i_train])

            x_train = x_train.float()
            y_train = y_train.float()

            likelihood, model, optimizer, output, loss = GaussianProcess.GPtrain(x_train, y_train, training_iter)

            RBF_lengthscale.append(model.covar_module.base_kernel.lengthscale.item())
            RBF_noise.append(model.likelihood.noise.item())

            # Test points are regularly spaced centered along the last index bounded by index displacement
            i_con = sample_disp_con(x_true, x_true[i_train[-1]], r_con)
            # if local_flag==1:
            x_test_local = torch.from_numpy(x_true[i_con, :])  # x_test is constrained to motion displacement
            # else:
            x_test_global = torch.from_numpy(x_true[i_seq, :])  # x_test is the entire dataset

            x_test_local = x_test_local.float()
            x_test_global = x_test_global.float()

            # Get into evaluation (predictive posterior) mode
            observed_pred_local, lower_local, upper_local = GaussianProcess.GPeval(x_test_local, model, likelihood)
            with torch.no_grad():
                f_preds = model(x_test_local)
                y_preds = likelihood(model(x_test_local))
                f_mean = f_preds.mean
                f_var_local = f_preds.variance
                f_covar = f_preds.covariance_matrix
            var_iter_local.append(max(f_var_local.numpy()))
            mse_local = sklearn.metrics.mean_squared_error(y_obs[i_con], observed_pred_local.mean.numpy())
            rmse_local.append(math.sqrt(mse_local))

            observed_pred_global, lower_global, upper_global = GaussianProcess.GPeval(x_test_global, model, likelihood)
            with torch.no_grad():
                f_preds = model(x_test_global)
                y_preds = likelihood(model(x_test_global))
                f_mean = f_preds.mean
                f_var_global = f_preds.variance
                f_covar = f_preds.covariance_matrix
            var_iter_global.append(max(f_var_global.numpy()))
            mse_global = sklearn.metrics.mean_squared_error(y_obs[i_seq], observed_pred_global.mean.numpy())
            rmse_global.append(math.sqrt(mse_global))

            # evaluate covariance properties
            covar_global.append(f_covar)
            covar_trace.append(np.trace(f_covar.detach().numpy()))
            covar_totelements.append(np.size(f_covar.detach().numpy()))
            covar_nonzeroelements.append(np.count_nonzero(f_covar.detach().numpy()))
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

            # now evaluate information criteria
            # akaike information criterion
            AIC_sample = 2 * np.log(covar_nonzeroelements[-1]) - 2 * np.log(mse_global)
            AIC.append(AIC_sample)
            # BIC calculated from https://en.wikipedia.org/wiki/Bayesian_information_criterion#Gaussian_special_case
            BIC_sample = np.size(i_train) * np.log(covar_nonzeroelements[-1]) - 2 * np.log(mse_global)
            BIC.append(BIC_sample)

            # plot real surface and the observed measurements
            ax1, ax2, ax3, ax4, ax5, ax5a, ax6, ax6a, ax7 = GaussianProcess.plotLAMPGP(fig, x_true, i_sample, i_train, y_obs, x_test_global, observed_pred_global, x_test_local, lower_local, upper_local, var_iter_global, var_iter_local, rmse_local, rmse_global, RBF_lengthscale, RBF_noise, covar_trace, covar_totelements, covar_nonzeroelements, AIC, BIC)
            plt.show()

            # fig.tight_layout()
            fig.savefig(image_path + str(len(set(i_train))) + '.png')
            fig.clear()

    createVideo("Snake", "LAMP", dx, "noise", r_disp, "")
    outputMetricLAMP(i_train, x_true, observed_pred_global, x_test_global, rmse_global, plt, rmse_local, var_iter_global, RBF_lengthscale, RBF_noise, covar_trace, AIC, BIC)

def exploreLAMPGPAL(n_samples, x_true, y_obs, flagIn, r_disp):
    local_flag = flagIn
    i_0 = random.randrange(n_samples)
    i_train = []
    i_train.append(i_0)
    i_seq = list(range(0, n_samples))

    # randomly sample next 10 data points with a displacement constraint of 10int
    r_NN = np.sqrt(3) * 0.25
    r_con = r_NN
    # randomly sample next 10 data points with a displacement constraint of 10int
    for i in range(10):
        i_sample_set = sample_disp_con(x_true, x_true[i_train[-1]], r_NN)  # nearest neighbor (within 0.25 km)
        # i_sample_set = sample_disp_con(x_true,x_true[i_train[-1]],r_con) # within 1 km
        i_sample = i_sample_set[random.randrange(len(i_sample_set))]
        i_train.append(int(i_sample))
    i_train = list(set(i_train))

    # %% hyperparameters for exploration training
    training_iter = 100
    sample_iter = int(n_samples / 2)
    var_iter_local = []
    var_iter_global = []
    rmse_local = []
    rmse_global = []
    RBF_lengthscale = []
    RBF_noise = []
    covar_global = []
    covar_trace = []
    covar_totelements = []
    covar_nonzeroelements = []
    AIC = []
    BIC = []

    # initialize animation
    fig = plt.figure(figsize=(12, 10))
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    # %% EXPLORATION PHASE: for loop to append sample and retrain GP model

    if local_flag == "local":
        explore_name = 'local'
    elif local_flag == "NN":
        explore_name = 'NN'
    elif local_flag == "global":
        explore_name = 'global'

    for j in range(sample_iter):
        # define training data
        x_train = torch.from_numpy(x_true[i_train, :])
        y_train = torch.from_numpy(y_obs[i_train])

        x_train = x_train.float()
        y_train = y_train.float()

        likelihood, model, optimizer, output, loss = GaussianProcess.GPtrain(x_train, y_train, training_iter)
        # store optimal hyperparameters
        RBF_lengthscale.append(model.covar_module.base_kernel.lengthscale.item())
        RBF_noise.append(model.likelihood.noise.item())

        # Test points are regularly spaced centered along the last index bounded by index displacement
        i_con = sample_disp_con(x_true, x_true[i_train[-1]], r_con)
        x_test_local = torch.from_numpy(x_true[i_con, :])  # x_test is constrained to motion displacement
        x_test_global = torch.from_numpy(x_true[i_seq, :])  # x_test is the entire dataset

        x_test_local = x_test_local.float()
        x_test_global = x_test_global.float()

        # Evaluate RMS for local
        observed_pred_local, lower_local, upper_local = GaussianProcess.GPeval(x_test_local, model, likelihood)
        with torch.no_grad():
            f_preds = model(x_test_local)
            y_preds = likelihood(model(x_test_local))
            f_mean = f_preds.mean
            f_var_local = f_preds.variance
            f_covar = f_preds.covariance_matrix
        var_iter_local.append(max(f_var_local.numpy()))
        mse_local = sklearn.metrics.mean_squared_error(y_obs[i_con], observed_pred_local.mean.numpy())
        rmse_local.append(math.sqrt(mse_local))
        # and global
        observed_pred_global, lower_global, upper_global = GaussianProcess.GPeval(x_test_global, model, likelihood)
        with torch.no_grad():
            f_preds = model(x_test_global)
            y_preds = likelihood(model(x_test_global))
            f_mean = f_preds.mean
            f_var_global = f_preds.variance
            f_covar = f_preds.covariance_matrix
        var_iter_global.append(max(f_var_global.numpy()))
        mse_global = sklearn.metrics.mean_squared_error(y_obs[i_seq], observed_pred_global.mean.numpy())
        rmse_global.append(math.sqrt(mse_global))

        # evaluate covariance properties
        covar_global.append(f_covar)
        covar_trace.append(np.trace(f_covar.detach().numpy()))
        covar_totelements.append(np.size(f_covar.detach().numpy()))
        covar_nonzeroelements.append(np.count_nonzero(f_covar.detach().numpy()))
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        # now evaluate information criteria
        # akaike information criterion
        AIC_sample = 2 * np.log(covar_nonzeroelements[-1]) - 2 * np.log(mse_global)
        AIC.append(AIC_sample)
        # BIC calculated from https://en.wikipedia.org/wiki/Bayesian_information_criterion#Gaussian_special_case
        BIC_sample = np.size(i_train) * np.log(covar_nonzeroelements[-1]) - 2 * np.log(mse_global)
        BIC.append(BIC_sample)

        # plot real surface and the observed measurements
        ax1, ax2, ax3, ax4, ax5, ax5a, ax6, ax6a, ax7 = GaussianProcess.plotLAMPGP(fig, x_true, i_sample, i_train, y_obs, x_test_global, observed_pred_global, x_test_local, lower_local, upper_local, var_iter_global, var_iter_local, rmse_local, rmse_global, RBF_lengthscale, RBF_noise, covar_trace, covar_totelements, covar_nonzeroelements, AIC, BIC)
        plt.show()

        # fig.tight_layout()
        fig.savefig(image_path + str(len(set(i_train))) + '.png')
        fig.clear()

        ## pick the next point to sample
        # this objective function maximizes the local variance
        # var_max = np.max(f_var_local.numpy())
        # i_max_set = np.argwhere(f_var_local.numpy()==var_max)
        # if len(i_max_set)==1:
        #     i_max = np.argmax(f_var_local.numpy())
        #     i_sample = i_con[i_max]
        # # with a tie breaker using distance
        # else:
        #     x_start = x_true[i_train[-1]]
        #     dx_con = (x_test_local[i_max_set,0]-x_start[0])**2 + (x_test_local[i_max_set,1]-x_start[1])**2 + (x_test_local[i_max_set,2]-x_start[2])**2
        #     i_max_con = np.argmax(dx_con)
        #     i_sample = int(i_max_set[i_max_con])

        # this objective function maximizes a pareto product of
        #      maximizing local variance and minimizing distance

        # waypoint within r_con with maximum variance, nearest neighbor along the way
        if flagIn == "local":
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

    createVideo("GPAL", "LAMP", "noise", r_disp, explore_name)
    outputMetricLAMP(i_train, x_true, observed_pred_global, x_test_global, rmse_global, plt, rmse_local, var_iter_global, RBF_lengthscale, RBF_noise, covar_trace, AIC, BIC)
