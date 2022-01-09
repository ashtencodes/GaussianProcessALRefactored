import os
import cv2
import re
import numpy as np

image_path = "C:\\Users\\ashte\\Desktop\\images\\"

def createVideo(strategy, func_name, noise_name, r_disp, explore_name="nonGPAL", dx=""):
    video_name = str(strategy) + '_' + str(explore_name) + '_dx' + str(dx) + '_' + str(func_name) + '_' + str(noise_name) + '_' + str(r_disp) + '.avi'

    images = []
    int_list = []
    for img in os.listdir(image_path):
        if img.endswith(".png"):
            images.append(img)
            s = re.findall(r'\d+', img)
            try:
                int_list.append(int(s[0]))
            except:
                print("whatever")

    arg_list = np.argsort(int_list)

    frame = cv2.imread(os.path.join(image_path, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(
        video_name, cv2.VideoWriter_fourcc(*'DIVX'), 30, (width, height))

    for i in range(len(arg_list)):
        image = images[arg_list[i]]
        video.write(cv2.imread(os.path.join(image_path, image)))

    cv2.destroyAllWindows()
    video.release()

def outputMetric(i_train, x_true,y_true, observed_pred_global, x_test_global, rmse_global_true, rmse_global_obs, plt, rmse_local_true, var_iter_global, AIC, BIC, RBF_lengthscale, RBF_noise, covar_trace):
    n_end = len(i_train)

    rover_distance = np.zeros(n_end)
    x_disp = np.zeros(n_end - 1)

    for i in range(n_end - 1):
        x_1 = x_true[i_train[i]]
        x_2 = x_true[i_train[i + 1]]
        x_disp = (x_2[0] - x_1[0]) ** 2 + (x_2[1] - x_1[1]) ** 2
        rover_distance[i + 1] = rover_distance[i] + x_disp

    i_min_pred = np.argmin(observed_pred_global.mean.numpy())
    print('rover converged on min at ' + str(x_test_global[i_min_pred].numpy()))

    i_min_real = np.argmin(y_true)
    print('true min at ' + str(x_true[i_min_real]))

    x_1 = x_test_global[i_min_pred].numpy()
    x_2 = x_true[i_min_real]
    x_disp = np.sqrt((x_2[0] - x_1[0]) ** 2 + (x_2[1] - x_1[1]) ** 2)

    print('min error is ' + str(x_disp))

    print('total roving distance is ' + str(rover_distance[-1]))

    # %% calculate convergence value of RMS error and distance until convergence
    v = rmse_global_true
    v0 = np.max(rmse_global_true)
    vf0 = rmse_global_true[-1]
    dv = v0 - vf0
    # band of noise allowable for 2% settling time convergence
    dv_2percent = 0.02 * dv

    # is there even enough data to confirm convergence?
    v_95thresh = v0 - 0.95 * dv
    i_95thresh = np.where(v < v_95thresh)
    i_95thresh = np.array(i_95thresh[0], dtype=int)
    if len(i_95thresh) >= 10:
        for i in range(len(i_95thresh)):
            v_con = v[i_95thresh[i]:-1]
            vf = np.mean(v_con)
            if np.all(v_con <= vf + dv_2percent) and np.all(v_con >= vf - dv_2percent):
                print("convergence roving distance is " +
                      str(rover_distance[i_95thresh[i]]))
                print("total samples is " + str(len(rover_distance)) + " where the convergence index is " +
                      str(i_95thresh[i]))
                print("convergence true rms error is " +
                      str(rmse_global_true[i_95thresh[i]]))
                print("convergence observed rms error is " +
                      str(rmse_global_obs[i_95thresh[i]]))
                print("reduction of error is " +
                      str(max(rmse_global_true) / rmse_global_true[i_95thresh[i]]))
                # plotty plot plot converge wrt rms error and distance!
                fig = plt.figure(figsize=(12, 6))
                ax1 = fig.add_subplot(1, 2, 1)
                local_rms = ax1.plot(range(0, len(
                    rmse_local_true)), rmse_local_true, color='blue', marker='.', label='local')
                global_rms = ax1.plot(range(0, len(
                    rmse_global_true)), rmse_global_true, color='black', marker='*', label='global')
                ax1.plot([0, len(var_iter_global)], np.array(
                    [1, 1]) * (vf + dv_2percent), 'r--')
                ax1.plot([0, len(var_iter_global)], np.array(
                    [1, 1]) * (vf - dv_2percent), 'r--')
                ax1.plot(i_95thresh[i] * np.array([1, 1]), [0, v0], 'r--')
                ax1.set_xlabel('number of samples')
                ax1.set_ylabel('rmse')
                ax1.legend(['local', 'global', 'convergence bounds'],
                           loc='upper right')
                ax1.set_title('rmse of learned model')
                ax2 = fig.add_subplot(1, 2, 2)
                ax2.plot(range(len(rover_distance)), rover_distance, 'k*-')
                ax2.plot(i_95thresh[i] * np.array([1, 1]),
                         [0, max(rover_distance)], 'r--')
                ax2.plot([0, len(rover_distance)],
                         rover_distance[i_95thresh[i]] * np.array([1, 1]), 'r--')
                ax2.set_xlabel('number of samples')
                ax2.set_ylabel('roving distance')
                ax2.set_title('rover distance during exploration')
                plt.show()
                fig.savefig(image_path + 'convergence.png')
                break
    else:
        print("not able to evaluate convergence")
        print("reduction of true error is " +
              str(max(rmse_global_true) / rmse_global_true[-1]))
        print("reduction of observed error is " +
              str(max(rmse_global_obs) / rmse_global_obs[-1]))

    # %% describe some covariance characteristics
    print("the optimal lengthscale at the end of training is " + str(RBF_lengthscale[-1]))
    print("the optimal noise at the end of training is " + str(RBF_noise[-1]))
    print("the added pointwise variance is 0.02")
    print("the final covariance trace is " + str(covar_trace[-1]))

    # let's talk about the information criteria
    print("the ending AIC is " + str(AIC[-1]))
    print("the ending BIC is " + str(BIC[-1]))