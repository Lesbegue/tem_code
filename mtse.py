import numpy as np
import mogptk
import math
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
    def __init__(self, space_input=None, space_output=None, Q = 1 ,Rq = 1, sp_d = None,model=None, aim=None):
        if aim is None:
            self.sp_d = sp_d
            self.Q = Q
            self.Rq = Rq
            self.offset = np.median(space_input)
            self.x = space_input - self.offset
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
            self.alpha = 1 / 2 / ((np.max(self.x) - np.min(self.x)) / 2) ** 2
            self.time = np.linspace(np.min(self.x), np.max(self.x), 500)
            self.w = np.linspace(0, self.Nx / (np.max(self.x) - np.min(self.x)) / 16, 500)
            self.model =  model
            # self.model.init_parameters(method='LS', iters=500)
            self.initialise_params()



    def initialise_params(self):
        self.sigma = None
        self.gamma = None
        self.theta = None
        self.sigma_n = None

    def set_labels(self,data_name):
        self.t_d =  np.arange(len(data_name))

    def capture_output_and_save(self, output_file=None):
        output = StringIO()
        with redirect_stdout(output):
            self.model.print_parameters()
        captured_output = output.getvalue()

        if output_file is not None:
            with open(output_file, 'w') as f:
                f.write(captured_output)

        lines = captured_output.strip().split('\n')
        data = []
        index = []

        for line in lines[1:]:
            parts = line.split(maxsplit=1)
            index.append(parts[0])
            data.append(parts[1].strip())

        df = pd.DataFrame(data, index=index, columns=['Value'])
        model_name = 'LinearModelOfCoregionalizationKernel'
        weight = eval(df.loc[model_name + '.weight', 'Value'])
        magnitude = [float(df.loc[model_name + '[{}].SpectralKernel.magnitude'.format(i), 'Value']) for i in range(self.Q)]
        mean = [eval(df.loc[model_name + '[{}].SpectralKernel.mean'.format(i), 'Value'])[0] for i in range(self.Q)]
        variance = [eval(df.loc[model_name + '[{}].SpectralKernel.variance'.format(i), 'Value'])[0] for i in range(self.Q)]
        scale = eval(df.loc['GaussianLikelihood.scale', 'Value'])
        self.weight = weight
        self.sigma = np.array([np.sqrt(m) for m in magnitude])
        self.gamma = np.array([2 * math.pi ** 2 * v for v in variance])
        self.theta = np.array([m for m in mean])
        self.sigma_n = scale

        return self.weight, self.sigma, self.gamma, self.theta,self.sigma_n
    def compute_moments(self):
        """
        :return:
        """
        #posterior moments for time
        cov_space = Spec_Mix(self.x,self.x,self.t_d,self.weight,self.gamma,self.theta,self.Q, self.sigma)+ 1e-5*np.eye(len(self.sigma_n)*self.Nx)+ np.diag(np.concatenate([np.repeat(i, self.Nx) for i in np.power(self.sigma_n,2)]))
        cov_time = Spec_Mix(self.time,self.time,self.t_d,self.weight, self.gamma, self.theta, self.Q, self.sigma)
        cov_star = Spec_Mix(self.time,self.x,self.t_d,self.weight, self.gamma, self.theta,self.Q, self.sigma)
        self.post_mean = np.squeeze(cov_star@np.linalg.solve(cov_space,np.concatenate(self.y)))
        self.post_cov = cov_time - (cov_star@np.linalg.solve(cov_space,cov_star.T))

        #posterior moment for frequency
        cov_real, cov_imag = freq_covariances(self.w,self.w,self.alpha,self.gamma,self.theta,self.Q,self.sigma, kernel = 'sm')
        xcov_real, xcov_imag = time_freq_covariances(self.w, self.x, self.weight,self.t_d,self.sp_d,self.alpha,self.gamma,self.theta,self.Q,self.sigma, kernel = 'sm')
        self.post_mean_r = np.squeeze(xcov_real@np.linalg.solve(cov_space,np.concatenate(self.y)))
        self.post_cov_r = cov_real - (xcov_real@np.linalg.solve(cov_space,xcov_real.T))
        self.post_mean_i = np.squeeze(xcov_imag@np.linalg.solve(cov_space,np.concatenate(self.y)))
        self.post_cov_i = cov_imag - (xcov_imag@np.linalg.solve(cov_space,xcov_imag.T))
        self.post_cov_ri = - ((xcov_real@np.linalg.solve(cov_space,xcov_imag.T)))

        self.post_mean_F = np.concatenate((self.post_mean_r, self.post_mean_i))
        self.post_cov_F = np.vstack((np.hstack((self.post_cov_r,self.post_cov_ri)), np.hstack((self.post_cov_ri.T,self.post_cov_i))))

        return cov_real, xcov_real, cov_space
    def plot_time_posterior_mogp(self, flag=None):
        # model.plot_prediction(title='Untrained model');
        self.model.plot_prediction(title='trained model')
        plt.savefig('time_posterior_mogp_{}.png'.format(self.sp_d))
        plt.close()
        # self.model.plot_prediction()

    def plot_freq_posterior_real(self):
        plt.figure(figsize=(18,6))
        plt.plot(self.w,self.post_mean_r, color='blue', label='posterior mean')
        error_bars = 2 * np.sqrt((np.diag(self.post_cov_r)))
        plt.fill_between(self.w, self.post_mean_r - error_bars, self.post_mean_r + error_bars, color='blue',alpha=0.1, label='95% error bars')
        plt.title('Posterior spectrum (real part)')
        plt.xlabel('frequency')
        plt.legend()
        plt.xlim([min(self.w),max(self.w)])
        plt.tight_layout()
        plt.savefig('freq_posterior_real_{}.png'.format(self.sp_d))
        plt.close()

    def plot_time_posterior(self, flag=None):
        # posterior moments for time
        for d in range(len(self.y)):
            plt.figure(figsize=(18, 6))
            plt.plot(self.x, self.y[d], '.r', markersize=10, label='observations')
            plt.plot(self.time, self.post_mean[d*len(self.time):(d+1)*len(self.time)], color='blue', label='posterior mean')
            error_bars = 2 * np.sqrt(np.diag(self.post_cov))
            plt.fill_between(self.time, self.post_mean[d*len(self.time):(d+1)*len(self.time)] - error_bars[d*len(self.time):(d+1)*len(self.time)], self.post_mean[d*len(self.time):(d+1)*len(self.time)] + error_bars[d*len(self.time):(d+1)*len(self.time)], color='blue', alpha=0.1,
                             label='95% error bars')
            if flag == 'with_window':
                plt.plot(self.time, 2 * self.sigma * np.exp(-self.alpha * self.time ** 2))
            plt.title('Observations and posterior interpolation')
            plt.xlabel(self.time_label)
            plt.ylabel(self.signal_label)
            plt.legend()
            plt.xlim([min(self.x), max(self.x)])
            plt.tight_layout()
            plt.savefig('time_posterior_{}_{}.png'.format(d,self.sp_d))
            plt.close()
    def plot_freq_posterior_imag(self):
        plt.figure(figsize=(18,6))
        plt.plot(self.w,self.post_mean_i, color='blue', label='posterior mean')
        error_bars = 2 * np.sqrt((np.diag(self.post_cov_i)))
        plt.fill_between(self.w, self.post_mean_i - error_bars, self.post_mean_i + error_bars, color='blue',alpha=0.1, label='95% error bars')
        plt.title('Posterior spectrum (imaginary part)')
        plt.xlabel('frequency')
        plt.legend()
        plt.xlim([min(self.w),max(self.w)])
        plt.tight_layout()
        plt.savefig('freq_posterior_imag_{}.png'.format(self.sp_d))
        plt.close()

    def plot_freq_posterior(self):
        self.plot_freq_posterior_real()
        self.plot_freq_posterior_imag()

    def normalize(self, matrix):
        mean_val = np.mean(matrix)
        std_val = np.std(matrix)
        normalized_matrix = (matrix - mean_val) / std_val
        return normalized_matrix
    def plot_MT_kernel(self):
        plt.close('all')
        k_tt = Spec_Mix(self.x,self.x,self.t_d,self.weight,self.gamma,self.theta,self.Q, self.sigma)+ 1e-5*np.eye(len(self.sigma_n)*self.Nx)+ np.diag(np.concatenate([np.repeat(i, self.Nx) for i in np.power(self.sigma_n,2)]))
        xcov_real, xcov_imag = time_freq_covariances(self.w, self.x, self.weight, self.t_d, self.sp_d, self.alpha,
                                                     self.gamma, self.theta, self.Q, self.sigma, kernel='sm')
        cov_real, cov_imag = freq_covariances(self.w, self.w, self.alpha, self.gamma, self.theta, self.Q, self.sigma,
                                              kernel='sm')

        k_tt = self.normalize(k_tt)
        xcov_real = self.normalize(xcov_real)
        xcov_imag = self.normalize(xcov_imag)
        cov_real = self.normalize(cov_real)
        cov_imag = self.normalize(cov_imag)
        MT_kernel_real = np.vstack(
            (np.hstack((k_tt, xcov_real.T)), np.hstack((xcov_real, cov_real))))
        MT_kernel_imag = np.vstack(
            (np.hstack((k_tt, xcov_imag.T)), np.hstack((xcov_imag, cov_imag))))
        fig,ax3 = plt.subplots(figsize = (5,5))
        colors = [(0,'blue'),
                  (0.5,'white'),
                  (1,'red')
                  ]
        cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
        divider = make_axes_locatable(ax3)
        cax = divider.append_axes('right', size='5%', pad=0.1)
        ax = sns.heatmap(MT_kernel_real, ax = ax3, cbar_ax=cax,  cmap=cmap,square=True, xticklabels= 100,yticklabels=100,
                         vmin=MT_kernel_real.min()-(MT_kernel_real.min()+MT_kernel_real.max())/2,
                         vmax=MT_kernel_real.max()-(MT_kernel_real.min()+MT_kernel_real.max())/2)
        ax.set_title('MT kernel (real part)')
        ax.set_xlabel('time')
        ax.set_ylabel('frequency')
        plt.tight_layout()
        plt.savefig('MT_kernel_real_{}.png'.format(self.sp_d))
        plt.close()
        # imaginary part
        fig, ax4 = plt.subplots(figsize=(5, 5))
        colors = [(0, 'blue'),
                  (0.5, 'white'),
                  (1, 'red')
                  ]
        cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
        divider = make_axes_locatable(ax4)
        cax = divider.append_axes('right', size='5%', pad=0.1)
        ax = sns.heatmap(MT_kernel_imag, ax=ax4, cbar_ax=cax, cmap=cmap, square=True, xticklabels=100,yticklabels=100,
                         vmin=MT_kernel_imag.min() - (MT_kernel_imag.min() + MT_kernel_imag.max()) / 2,
                         vmax=MT_kernel_imag.max() - (MT_kernel_imag.min() + MT_kernel_imag.max()) / 2)
        ax.set_title('MT kernel (imag part)')
        ax.set_xlabel('time')
        ax.set_ylabel('frequency')
        plt.tight_layout()
        plt.savefig('MT_kernel_imag_{}.png'.format(self.sp_d))
        plt.close()

    def plot_power_spectral_density(self, how_many, flag=None):
        #posterior moments for frequency
        plt.figure(figsize=(18,6))
        freqs = len(self.w)
        samples = np.zeros((freqs,how_many))
        for i in range(how_many):
            sample = np.random.multivariate_normal(self.post_mean_F,(self.post_cov_F+self.post_cov_F.T)/2 + 1e-5*np.eye(2*freqs))
            samples[:,i] = sample[0:freqs]**2 + sample[freqs:]**2
        plt.plot(self.w,samples, color='red', alpha=0.35)
        plt.plot(self.w,samples[:,0], color='red', alpha=0.35, label='posterior samples')
        posterior_mean_psd = self.post_mean_r**2 + self.post_mean_i**2 + np.diag(self.post_cov_r + self.post_cov_i)
        plt.plot(self.w,posterior_mean_psd, color='black', label = '(analytical) posterior mean')
        if flag == 'show peaks':
            peaks, _  = find_peaks(posterior_mean_psd, prominence=500000)
            widths = peak_widths(posterior_mean_psd, peaks, rel_height=0.5)
            plt.stem(self.w[peaks],posterior_mean_psd[peaks], markerfmt='ko', label='peaks')
        plt.title('Sample posterior power spectral density')
        plt.xlabel('frequency')
        plt.legend()
        plt.xlim([min(self.w),max(self.w)])
        plt.tight_layout()
        plt.savefig('psd_{}.png'.format(self.sp_d))
        plt.close()
        if flag == 'show peaks':
            return peaks, widths
    def plot_spectrum(self, flag=None):
        self.model.plot_spectrum()
    def set_freqspace(self, max_freq, dimension=500):
        self.w = np.linspace(0, max_freq, dimension)
    def train(self):
        self.model.train(method='Adam', lr=2, iters=50, plot=True, error='MAE', verbose=True)
    def return_model(self):
        return self.model

def outersum(a,b):
    return np.outer(a,np.ones_like(b))+np.outer(np.ones_like(a),b)


def corr_matrix(q, t_d, weight):
    # Initialize an empty 2D array (matrix) with specified dimensions
    matrix = np.zeros((len(t_d), len(t_d)))
    # Populate the matrix with dot products
    for i in range(len(t_d)):
        for j in range(len(t_d)):
            matrix[i, j] = np.dot(weight[i][q], weight[j][q])
    return matrix


def Spec_Mix(x,y,t_d,weight, gamma, theta, Q, sigma):
    for q in range(Q):
        if q == 0:
            sum_sm = np.kron(normalize(corr_matrix(q,t_d,weight)),sigma[q]**2 * np.exp(-gamma[q]*outersum(x,-y)**2)*np.cos(2*np.pi*theta[q]*outersum(x,-y)))
        else:
            sum_sm = sum_sm + np.kron(normalize(corr_matrix(q,t_d,weight)),sigma[q]**2 * np.exp(-gamma[q]*outersum(x,-y)**2)*np.cos(2*np.pi*theta[q]*outersum(x,-y)))
    return sum_sm


def Spec_Mix_spectral(x, y, alpha, gamma, theta, Q, sigma):
    for q in range(Q):
        if q == 0:
            magnitude = np.pi * sigma[q] ** 2 / (np.sqrt(alpha * (alpha + 2 * gamma[q])))
            sum = magnitude * np.exp(
                -np.pi ** 2 / (2 * alpha) * outersum(x, -y) ** 2 - 2 * np.pi * 2 / (alpha + 2 * gamma[q]) * (
                        outersum(x, y) / 2 - theta[q]) ** 2)
        else:
            magnitude = np.pi * sigma[q] ** 2 / (np.sqrt(alpha * (alpha + 2 * gamma[q])))
            sum = sum + magnitude * np.exp(
                -np.pi ** 2 / (2 * alpha) * outersum(x, -y) ** 2 - 2 * np.pi * 2 / (alpha + 2 * gamma[q]) * (
                        outersum(x, y) / 2 - theta[q]) ** 2)
    return sum
def freq_covariances(x, y, alpha, gamma, theta, Q, sigma, kernel = 'sm'):
    if kernel == 'sm':
        N = len(x)
        #compute kernels
        K = 1/2*(Spec_Mix_spectral(x, y, alpha, gamma, theta,Q, sigma) + Spec_Mix_spectral(x, y, alpha, gamma, -theta,Q, sigma))
        P = 1/2*(Spec_Mix_spectral(x, -y, alpha, gamma, theta,Q, sigma) + Spec_Mix_spectral(x, -y, alpha, gamma, -theta, Q,sigma))
        real_cov = 1/2*(K + P) + 1e-8*np.eye(N)
        imag_cov = 1/2*(K - P) + 1e-8*np.eye(N)
    return real_cov, imag_cov

def time_freq_SM_re_q(x, y, weight,alpha, gamma, theta, Q, sigma):
       at = alpha/(np.pi**2)
       gt = gamma/(np.pi**2)
       L = 1/at + 1/gt
       return weight*(sigma ** 2) / (np.sqrt(np.pi * (at + gt))) * np.exp(
outersum(-(x - theta) ** 2 / (at + gt), -y ** 2 * np.pi ** 2 / L)) * np.cos(
           -np.outer(2 * np.pi * (x / at + theta / gt) / (1 / at + 1 / gt), y))


def time_freq_SM_im_q(x, y, weight, alpha, gamma, theta, Q, sigma):
    at = alpha / (np.pi ** 2)
    gt = gamma / (np.pi ** 2)
    L = 1 / at + 1 / gt
    return weight * (sigma ** 2) / (np.sqrt(np.pi * (at + gt))) * np.exp(
        outersum(-(x - theta) ** 2 / (at + gt), -y ** 2 * np.pi ** 2 / L)) * np.sin(
        -np.outer(2 * np.pi * (x / at + theta / gt) / (1 / at + 1 / gt), y))
def time_freq_covariances_q(x, t, weight, alpha, gamma, theta, Q, sigma, kernel ='sm'):
    if kernel == 'sm':
        tf_real_cov = 1/2*(time_freq_SM_re_q(x, t, weight, alpha, gamma, theta, Q, sigma) + time_freq_SM_re_q(x, t, weight,alpha, gamma, theta, Q, sigma))
        tf_imag_cov = 1/2*(time_freq_SM_im_q(x, t, weight, alpha, gamma, theta, Q, sigma) + time_freq_SM_im_q(x, t, weight, alpha, gamma, theta, Q, sigma))
    return tf_real_cov, tf_imag_cov
def time_freq_covariances(x, t, weight, t_d, sp_d ,alpha, gamma, theta, Q, sigma, kernel = 'sm'):
    if kernel == 'sm':
        sum_real = [np.zeros((len(x), len(t))) for _ in range(len(t_d))]
        sum_imag = [np.zeros((len(x), len(t))) for _ in range(len(t_d))]
        for d in range(len(t_d)):
            for q in range(Q):
                if q == 0:
                    weight_dot = normalize(corr_matrix(q,t_d,weight))
                    real_cov, imag_cov = time_freq_covariances_q(x, t, weight_dot[d][sp_d],alpha, gamma[q], theta[q], Q, sigma[q])
                    sum_real[d] = real_cov
                    sum_imag[d] = imag_cov
                else:
                    weight_dot = normalize(corr_matrix(q,t_d,weight))
                    real_cov, imag_cov = time_freq_covariances_q(x, t, weight_dot[d][sp_d],alpha, gamma[q], theta[q], Q, sigma[q])
                    sum_real[d] = sum_real[d] + real_cov
                    sum_imag[d] = sum_imag[d] + imag_cov
        return np.concatenate(sum_real, axis=1), np.concatenate(sum_imag, axis=1)
def normalize(matrix):
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    normalized_matrix = (matrix - min_val) / (max_val - min_val)
    return normalized_matrix