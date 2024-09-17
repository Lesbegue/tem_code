import numpy as np
import gpytorch
import math
import logging
import torch
from scipy.signal import find_peaks, peak_widths
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import seaborn as sns
from io import StringIO
from contextlib import redirect_stdout


plot_params = {'legend.fontsize': 18,
               'figure.figsize': (15, 5),
               'xtick.labelsize': '18',
               'ytick.labelsize': '18',
               'axes.titlesize': '24',
               'axes.labelsize': '22'}
plt.rcParams.update(plot_params)

class mtse:
    def __init__(self, space_input=None, space_output=None, Q = 1 , model=None,likelihood=None, aim=None):
        self.t_d = None
        self.sigma_n = None
        self.theta = None
        self.gamma = None
        self.sigma = None
        self.weight = None
        if aim is None:
            self.Q = Q
            self.likelihood = likelihood
            self.x = space_input
            self.y = space_output
            self.post_mean = None
            self.post_cov = None
            self.post_mean_r = None
            self.post_cov_r = None
            self.post_mean_i = None
            self.post_cov_i = None
            self.time_label = None
            self.signal_label = None
            self.Nx = len(self.x)
            self.alpha = 1 / 2 / ((torch.max(self.x) - torch.min(self.x)) / 2) ** 2
            self.time = self.x
            self.w = torch.linspace(0, self.Nx / (torch.max(self.x) - torch.min(self.x)) / 16, 500)
            self.model =  model
            # self.model.init_parameters(method='LS', iters=500)
            self.initialise_params()



    def initialise_params(self):
        self.sigma = None
        self.gamma = None
        self.theta = None
        self.sigma_n = None

    def set_labels(self,data_name):
        self.t_d =  torch.arange(len(data_name))

    def get_parameters(self, output_file=None):
        weight = [
            (self.model.covar_module.covar_module_list[q].task_covar_module.covar_factor) @
            (self.model.covar_module.covar_module_list[q].task_covar_module.covar_factor).T
            for q in range(self.Q)
        ]
        magnitude = [
            self.model.covar_module.covar_module_list[q].data_covar_module.mixture_weights
            for q in range(self.Q)
        ]
        mean = [
            self.model.covar_module.covar_module_list[q].data_covar_module.mixture_means
            for q in range(self.Q)
        ]
        scale = torch.diag(self.model.likelihood.task_noise_covar_factor.detach() @
                           self.model.likelihood.task_noise_covar_factor.detach().T)
        variance = [
            self.model.covar_module.covar_module_list[q].data_covar_module.mixture_scales
            for q in range(self.Q)
        ]
        self.weight = weight
        self.sigma = torch.tensor([torch.sqrt(m) for m in magnitude])
        self.gamma = torch.tensor([2*math.pi**2*v**2 for v in variance])
        self.theta = torch.tensor([m for m in mean])
        self.sigma_n = scale

        return self.weight, self.sigma, self.gamma, self.theta, self.sigma_n

    def compute_moments(self):
        """
        :return:
        """
        #posterior moments for time
        cov_space = Spec_Mix(self.x, self.x, self.t_d, self.weight, self.gamma, self.theta, self.Q, self.sigma) + \
                    1e-5 * torch.eye(self.y.shape[1] * self.Nx) + \
                    torch.diag(torch.cat([self.sigma_n[i].repeat(self.Nx) for i in range(len(self.sigma_n))]))
        cov_time = Spec_Mix(self.time,self.time,self.t_d,self.weight, self.gamma, self.theta, self.Q, self.sigma)
        cov_star = Spec_Mix(self.time,self.x,self.t_d,self.weight, self.gamma, self.theta,self.Q, self.sigma)
        self.post_mean = torch.squeeze(cov_star @ torch.linalg.solve(cov_space, torch.cat((self.y[:, 0], self.y[:, 1]))))
        self.post_cov = cov_time - (cov_star@torch.linalg.solve(cov_space,cov_star.T))


        #posterior moment for frequency
        cov_real, cov_imag = freq_covariances(self.w,self.w,self.alpha,self.gamma,self.theta,self.Q,self.sigma, kernel = 'sm')
        xcov_real, xcov_imag = time_freq_covariances(self.w, self.x, self.weight,self.t_d,self.sp_d,self.alpha,self.gamma,self.theta,self.Q,self.sigma, kernel = 'sm')
        self.post_mean_r = torch.squeeze(xcov_real@torch.linalg.solve(cov_space,torch.cat((self.y[:, 0], self.y[:, 1]))))
        self.post_cov_r = cov_real - (xcov_real@torch.linalg.solve(cov_space,xcov_real.T))
        self.post_mean_i = torch.squeeze(xcov_imag@torch.linalg.solve(cov_space,torch.cat((self.y[:, 0], self.y[:, 1]))))
        self.post_cov_i = cov_imag - (xcov_imag@torch.linalg.solve(cov_space,xcov_imag.T))
        self.post_cov_ri = - (xcov_real @ torch.linalg.solve(cov_space, xcov_imag.T))

        self.post_mean_F = torch.concatenate((self.post_mean_r, self.post_mean_i))
        self.post_cov_F = torch.vstack((torch.hstack((self.post_cov_r,self.post_cov_ri)), torch.hstack((self.post_cov_ri.T,self.post_cov_i))))

        return cov_real, xcov_real, cov_space

    def plot_freq_posterior_real(self):
        plt.figure(figsize=(18, 6))

        # Convert tensors to numpy arrays for plotting
        w_np = self.w.detach().numpy()
        post_mean_r_np = self.post_mean_r.detach().numpy()
        error_bars_np = 2 * torch.sqrt(torch.diag(self.post_cov_r)).detach().numpy()

        # Plot the posterior mean
        plt.plot(w_np, post_mean_r_np, color='blue', label='posterior mean')

        # Fill between the error bars
        plt.fill_between(w_np, post_mean_r_np - error_bars_np, post_mean_r_np + error_bars_np, color='blue', alpha=0.1,
                         label='95% error bars')

        plt.title('Posterior spectrum (real part)')
        plt.xlabel('frequency')
        plt.legend()
        plt.xlim([min(w_np), max(w_np)])
        plt.tight_layout()
        plt.savefig('freq_posterior_real_{}.png'.format(self.sp_d))

    def plot_time_posterior(self, flag=None):
        # Compute NumPy arrays
        time_np = self.time.detach().numpy()
        post_mean_np = self.post_mean.detach().numpy()
        error_bars_np = 2 * torch.sqrt(torch.diag(self.post_cov)).detach().numpy()
        if hasattr(self, 'sigma') and hasattr(self, 'alpha'):
            sigma_np = self.sigma.detach().numpy()
            alpha_np = self.alpha.detach().numpy()
        else:
            sigma_np = alpha_np = None

        for d in range(self.y.shape[1]):
            plt.figure(figsize=(18, 6))
            plt.plot(self.x.detach().numpy(), self.y[:,d].detach().numpy(), '.r', markersize=10, label='observations')
            plt.plot(time_np, post_mean_np[d * len(self.time):(d + 1) * len(self.time)], color='blue',
                     label='posterior mean')
            plt.fill_between(
                time_np,
                post_mean_np[d * len(self.time):(d + 1) * len(self.time)] - error_bars_np[
                                                                            d * len(self.time):(d + 1) * len(
                                                                                self.time)],
                post_mean_np[d * len(self.time):(d + 1) * len(self.time)] + error_bars_np[
                                                                            d * len(self.time):(d + 1) * len(
                                                                                self.time)],
                color='blue',
                alpha=0.1,
                label='95% error bars'
            )
            if flag == 'with_window' and sigma_np is not None and alpha_np is not None:
                plt.plot(time_np, 2 * sigma_np * np.exp(-alpha_np * time_np ** 2))
            plt.title('Observations and posterior interpolation')
            plt.xlabel(self.time_label)
            plt.ylabel(self.signal_label)
            plt.legend()
            plt.xlim([min(self.x.detach().numpy()), max(self.x.detach().numpy())])
            plt.tight_layout()
            plt.savefig('time_posterior_{}_{}.png'.format(d, self.sp_d))

    def plot_freq_posterior_imag(self):
        plt.figure(figsize=(18, 6))
        # Convert tensors to numpy arrays
        w_np = self.w.detach().numpy()
        post_mean_i_np = self.post_mean_i.detach().numpy()
        error_bars_np = 2 * torch.sqrt(torch.diag(self.post_cov_i)).detach().numpy()
        # Plot the posterior mean
        plt.plot(w_np, post_mean_i_np, color='blue', label='posterior mean')
        # Fill between the error bars
        plt.fill_between(
            w_np,
            post_mean_i_np - error_bars_np,
            post_mean_i_np + error_bars_np,
            color='blue',
            alpha=0.1,
            label='95% error bars'
        )
        plt.title('Posterior spectrum (imaginary part)')
        plt.xlabel('frequency')
        plt.legend()
        plt.xlim([min(w_np), max(w_np)])
        plt.tight_layout()
        plt.savefig('freq_posterior_imag_{}.png'.format(self.sp_d))

    def plot_freq_posterior(self):
        self.plot_freq_posterior_real()
        self.plot_freq_posterior_imag()

    def standardize(self,matrix: torch.Tensor) -> torch.Tensor:
        # # Ensure the matrix is a tensor
        # if not isinstance(matrix, torch.Tensor):
        #     matrix = torch.tensor(matrix)
        #
        # # Calculate min and max values
        # min_val = torch.min(matrix)
        # max_val = torch.max(matrix)
        #
        # # Avoid division by zero if all values are the same
        # if max_val == min_val:
        #     return torch.zeros_like(matrix)  # or return matrix if you prefer to keep the original values
        #
        # # Normalize the matrix to the range [0, 1]
        # normalized_matrix = (matrix - min_val) / (max_val - min_val)
        return matrix

    def plot_MT_kernel(self):
        plt.close('all')

        # Compute kernel matrices
        k_tt = (Spec_Mix(self.x, self.x, self.t_d, self.weight, self.gamma, self.theta, self.Q, self.sigma) +
                1e-5 * torch.eye(self.y.shape[1] * self.Nx) +
                torch.diag(torch.tile(torch.pow(self.sigma_n, 2), (self.Nx, 1)).T.flatten()))

        xcov_real, xcov_imag = time_freq_covariances(self.w, self.x, self.weight, self.t_d, self.sp_d, self.alpha,
                                                     self.gamma, self.theta, self.Q, self.sigma, kernel='sm')
        cov_real, cov_imag = freq_covariances(self.w, self.w, self.alpha, self.gamma, self.theta, self.Q, self.sigma,
                                              kernel='sm')

        # Standardize the matrices
        k_tt = self.standardize(k_tt)
        xcov_real = self.standardize(xcov_real)
        xcov_imag = self.standardize(xcov_imag)
        cov_real = self.standardize(cov_real)
        cov_imag = self.standardize(cov_imag)

        # Combine matrices for visualization
        MT_kernel_real = torch.vstack(
            (torch.hstack((k_tt, xcov_real.T)), torch.hstack((xcov_real, cov_real))))
        MT_kernel_imag = torch.vstack(
            (torch.hstack((k_tt, xcov_imag.T)), torch.hstack((xcov_imag, cov_imag))))

        # Convert tensors to NumPy arrays
        MT_kernel_real = MT_kernel_real.detach().numpy()
        MT_kernel_imag = MT_kernel_imag.detach().numpy()

        # Define color map
        colors = [(0, 'blue'), (0.5, 'white'), (1, 'red')]
        cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)

        # Plot real part
        fig, ax3 = plt.subplots(figsize=(5, 5))
        divider = make_axes_locatable(ax3)
        cax = divider.append_axes('right', size='5%', pad=0.1)
        sns.heatmap(MT_kernel_real, ax=ax3, cbar_ax=cax, cmap=cmap, square=True, xticklabels=100, yticklabels=100,
                    vmin=MT_kernel_real.min() - (MT_kernel_real.min() + MT_kernel_real.max()) / 2,
                    vmax=MT_kernel_real.max() - (MT_kernel_real.min() + MT_kernel_real.max()) / 2)
        ax3.set_title('MT kernel (real part)')
        ax3.set_xlabel('time')
        ax3.set_ylabel('frequency')
        plt.tight_layout()
        plt.savefig('MT_kernel_real_{}.png'.format(self.sp_d))


        # Plot imaginary part
        fig, ax4 = plt.subplots(figsize=(5, 5))
        divider = make_axes_locatable(ax4)
        cax = divider.append_axes('right', size='5%', pad=0.1)
        sns.heatmap(MT_kernel_imag, ax=ax4, cbar_ax=cax, cmap=cmap, square=True, xticklabels=100, yticklabels=100,
                    vmin=MT_kernel_imag.min() - (MT_kernel_imag.min() + MT_kernel_imag.max()) / 2,
                    vmax=MT_kernel_imag.max() - (MT_kernel_imag.min() + MT_kernel_imag.max()) / 2)
        ax4.set_title('MT kernel (imag part)')
        ax4.set_xlabel('time')
        ax4.set_ylabel('frequency')
        plt.tight_layout()
        plt.savefig('MT_kernel_imag_{}.png'.format(self.sp_d))
    def rt(self):
        return  self.likelihood,self.model

    def set_sp_d(self, sp_d):
        self.sp_d = sp_d

    def plot_power_spectral_density(self, how_many, flag=None):
        # Posterior moments for frequency
        global peaks, widths
        plt.figure(figsize=(18, 6))
        freqs = len(self.w)
        samples = torch.zeros((freqs, how_many))

        # Create multivariate normal distribution
        mean = self.post_mean_F
        cov = (self.post_cov_F + self.post_cov_F.T) / 2 + torch.eye(2 * freqs)
        mvn = torch.distributions.MultivariateNormal(mean, cov)

        for i in range(how_many):
            sample = mvn.sample()
            samples[:, i] = sample[:freqs] ** 2 + sample[freqs:] ** 2

        # Convert torch tensors to numpy arrays
        w_np = self.w.numpy()
        samples_np = samples.numpy()
        posterior_mean_psd = (self.post_mean_r ** 2 + self.post_mean_i ** 2 +
                              torch.diag(self.post_cov_r + self.post_cov_i)).detach().numpy()

        plt.plot(w_np, samples_np, color='red', alpha=0.35)
        plt.plot(w_np, samples_np[:, 0], color='red', alpha=0.35, label='posterior samples')
        plt.plot(w_np, posterior_mean_psd, color='black', label='(analytical) posterior mean')

        if flag == 'show peaks':
            peaks, _ = find_peaks(posterior_mean_psd, prominence=500000)
            widths = peak_widths(posterior_mean_psd, peaks, rel_height=0.5)
            plt.stem(w_np[peaks], posterior_mean_psd[peaks], markerfmt='ko', label='peaks')

        plt.title('Sample posterior power spectral density')
        plt.xlabel('frequency')
        plt.legend()
        plt.xlim([w_np.min(), w_np.max()])
        plt.tight_layout()
        plt.savefig('psd_{}.png'.format(self.sp_d))
        plt.close()

        if flag == 'show peaks':
            return peaks, widths

    def set_freqspace(self, max_freq, dimension=500):
        self.w = torch.linspace(0, max_freq, dimension)

    def train(self):
        self.model.train()
        self.likelihood.train()
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        training_iter = 201
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.3)
        for i in range(training_iter):
            self.get_parameters()
            optimizer.zero_grad()
            output = self.model(self.x)
            loss = -mll(output, self.y)
            loss.backward()
            # 直接在 print 函数中提取参数
            if i % 50 == 0:
                print(f'Iter {i + 1}/{training_iter} - Loss: {loss.item():.3f} '
                      f'Mean: {self.theta[0].item():.3f} '
                      f'Weight: {self.sigma[0].item():.3f} '
                      f'Scale: {self.gamma[0].item():.3f} '
                      f'Noise: {self.model.likelihood.noise.item():.3f}')
            optimizer.step()
        # Evaluation
        self.model.eval()
        self.likelihood.eval()

    def return_model(self):
        return self.model



def outersum(a,b):
    return torch.outer(a,torch.ones_like(b))+torch.outer(torch.ones_like(a),b)


def corr_matrix(q,weight):
    # Initialize an empty 2D array (matrix) with specified dimensions
    return weight[q]


def Spec_Mix(x,y,t_d,weight, gamma, theta, Q, sigma):
    global sum_sm
    for q in range(Q):
        if q == 0:
            sum_sm = torch.kron(standardize(corr_matrix(q,weight)),sigma[q]**2 * torch.exp(-gamma[q]*outersum(x,-y)**2)*torch.cos(2*torch.pi*theta[q]*outersum(x,-y)))
        else:
            sum_sm += torch.kron(standardize(corr_matrix(q,weight)),sigma[q]**2 * torch.exp(-gamma[q]*outersum(x,-y)**2)*torch.cos(2*torch.pi*theta[q]*outersum(x,-y)))
    return sum_sm


def Spec_Mix_spectral(x, y, alpha, gamma, theta, Q, sigma):
    global sum
    for q in range(Q):
        # 计算 magnitude
        term1 = torch.pi * sigma[q] ** 2
        term2 = torch.sqrt(alpha * (alpha + 2 * gamma[q]))
        magnitude = term1 / term2

        # 计算 outersum(x, -y) 和 outersum(x, y) 的中间结果
        outersum_neg_y = outersum(x, -y)
        outersum_y = outersum(x, y) / 2

        # 计算 term3 和 term4_squared
        term3 = outersum_neg_y ** 2
        term4 = outersum_y - theta[q]
        term4_squared = term4 ** 2

        # 计算指数部分的中间结果
        exp_term1 = -torch.pi ** 2 / (2 * alpha) * term3
        exp_term2 = -2 * torch.pi ** 2 / (alpha + 2 * gamma[q]) * term4_squared
        exp_argument = exp_term1 + exp_term2

        # 计算 exp_value 和 sum_value
        exp_value = torch.exp(exp_argument)
        sum_value = magnitude * exp_value



        # 处理第一个q值
        if q == 0:
            sum = sum_value
        else:
            # 更新 sum
            sum = sum + sum_value

    # 返回最终的 sum
    return sum


def freq_covariances(x, y, alpha, gamma, theta, Q, sigma, kernel = 'sm'):
    global real_cov, imag_cov
    if kernel == 'sm':
        N = len(x)
        #compute kernels
        K = 1/2*(Spec_Mix_spectral(x, y, alpha, gamma, theta,Q, sigma) + Spec_Mix_spectral(x, y, alpha, gamma, -theta,Q, sigma))
        P = 1/2*(Spec_Mix_spectral(x, -y, alpha, gamma, theta,Q, sigma) + Spec_Mix_spectral(x, -y, alpha, gamma, -theta, Q,sigma))
        real_cov = 1/2*(K + P) + 1e-8*torch.eye(N)
        imag_cov = 1/2*(K - P) + 1e-8*torch.eye(N)
    return real_cov, imag_cov

def time_freq_SM_re_q(x, y, weight,alpha, gamma, theta, Q, sigma):
       at = alpha/(torch.pi**2)
       gt = gamma/(torch.pi**2)
       L = 1/at + 1/gt
       return weight*(sigma ** 2) / (torch.sqrt(torch.pi * (at + gt))) * torch.exp(
outersum(-(x - theta) ** 2 / (at + gt), -y ** 2 * torch.pi ** 2 / L)) * torch.cos(
           -torch.outer(2 * torch.pi * (x / at + theta / gt) / L, y))


def time_freq_SM_im_q(x, y, weight, alpha, gamma, theta, Q, sigma):
    at = alpha / (torch.pi ** 2)
    gt = gamma / (torch.pi ** 2)
    L = 1 / at + 1 / gt
    return weight * (sigma ** 2) / (torch.sqrt(torch.pi * (at + gt))) * torch.exp(
        outersum(-(x - theta) ** 2 / (at + gt), -y ** 2 * torch.pi ** 2 / L)) * torch.sin(
        -torch.outer(2 * torch.pi * (x / at + theta / gt) / L, y))
def time_freq_covariances_q(x, t, weight, alpha, gamma, theta, Q, sigma, kernel ='sm'):
    global tf_real_cov, tf_imag_cov
    if kernel == 'sm':
        tf_real_cov = 1/2*(time_freq_SM_re_q(x, t, weight, alpha, gamma, theta, Q, sigma) + time_freq_SM_re_q(x, t, weight,alpha, gamma, theta, Q, sigma))
        tf_imag_cov = 1/2*(time_freq_SM_im_q(x, t, weight, alpha, gamma, theta, Q, sigma) + time_freq_SM_im_q(x, t, weight, alpha, gamma, theta, Q, sigma))
    return tf_real_cov, tf_imag_cov
def time_freq_covariances(x, t, weight, t_d, sp_d ,alpha, gamma, theta, Q, sigma, kernel = 'sm'):
    if kernel == 'sm':
        sum_real = [torch.zeros((len(x), len(t))) for _ in range(len(t_d))]
        sum_imag = [torch.zeros((len(x), len(t))) for _ in range(len(t_d))]
        for d in range(len(t_d)):
            for q in range(Q):
                if q == 0:
                    weight_dot = standardize(corr_matrix(q,weight))
                    real_cov, imag_cov = time_freq_covariances_q(x, t, weight_dot[d][sp_d],alpha, gamma[q], theta[q], Q, sigma[q])
                    sum_real[d] = real_cov
                    sum_imag[d] = imag_cov
                else:
                    weight_dot = standardize(corr_matrix(q,weight))
                    real_cov, imag_cov = time_freq_covariances_q(x, t, weight_dot[d][sp_d],alpha, gamma[q], theta[q], Q, sigma[q])
                    sum_real[d] = sum_real[d] + real_cov
                    sum_imag[d] = sum_imag[d] + imag_cov
        return torch.cat((sum_real[0], sum_real[1]), dim=1), torch.cat((sum_imag[0],sum_imag[1]), dim=1)


def standardize(matrix: torch.Tensor) -> torch.Tensor:
    # # Ensure the matrix is a tensor
    # if not isinstance(matrix, torch.Tensor):
    #     matrix = torch.tensor(matrix)
    #
    # # Calculate min and max values
    # min_val = torch.min(matrix)
    # max_val = torch.max(matrix)
    #
    # # Avoid division by zero if all values are the same
    # if max_val == min_val:
    #     return torch.zeros_like(matrix)  # or return matrix if you prefer to keep the original values
    #
    # # Normalize the matrix to the range [0, 1]
    # normalized_matrix = (matrix - min_val) / (max_val - min_val)
    return matrix