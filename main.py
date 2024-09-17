from mtse import *
from model import *
from get_data import *
from remove_images import delete_png_files_in_directory
directory = '/home/w/PycharmProjects/MTGP_SE'
delete_png_files_in_directory(directory)
dt_options = [
        'accelerometry-walk-climb-drive',
        'brain-wearable-monitoring',
        'consumer-grade-wearables',
        'Heart_Rate',
        'culm',
        'cpap-data-canterbury',
        'emgdb',
        'gait-maturation-db',
        'perg-ioba-datasetb',
        'respiratory-heartrate-dataset',
        'treadmill-exercise-cardioresp',
        'sinus-rhythm-dataset'
    ]
data_id = 11
list = [1,3]
freq,time,signal,data_name = get_data(dt_options[data_id])
print(dt_options[data_id])
# time = torch.tensor(time.values, dtype=torch.float32)
signal = torch.tensor(signal[[signal.columns[0], signal.columns[1]]].values, dtype=torch.float32)
time,signal = sample_and_normalize_data(time, signal, size=1/3)
train_x, train_y = time, signal
num_tasks = 2
Q = 3
likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks, rank= 2)
model = MultitaskGPModel(train_x, train_y, likelihood, num_tasks,Q=Q)
my_mtse = mtse(train_x,train_y, Q,model = model,likelihood = likelihood)
my_mtse.set_labels(data_name)
my_mtse.set_freqspace(freq, dimension=500)
my_mtse.train()
weight, sigma, gamma, theta, sigma_n = my_mtse.get_parameters()
print("weight: ", weight)
print("sigma: ", sigma)
print("gamma: ", gamma)
print("theta: ", theta)
print("sigma_n: ", sigma_n)
for sp_d in [0,1]:
    my_mtse.set_sp_d(sp_d)
    my_mtse.compute_moments()
    my_mtse.plot_freq_posterior_real()
    my_mtse.plot_time_posterior()
    my_mtse.plot_freq_posterior_imag()
    my_mtse.plot_power_spectral_density(15)
    # my_mtse.plot_MT_kernel()
likelihood,model = my_mtse.rt()
test_x = train_x
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    pred_multi = likelihood(model(test_x))

# Plot predictions
# Create two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 18))

colors = ['blue', 'red']

# Plot for Task 1
ax1.plot(test_x, pred_multi.mean[:, 0], label='Mean prediction', color='blue')
# ax1.plot(test_x, train_y[:, 0], linestyle='--', label='True function', color='blue')
lower = pred_multi.confidence_region()[0][:, 0].detach().numpy()
upper = pred_multi.confidence_region()[1][:, 0].detach().numpy()
ax1.fill_between(
test_x,
lower,
upper,
alpha=0.2,
label='Confidence interval',
color='blue'
)
ax1.scatter(train_x, train_y[:, 0], color='black', label='Training data')
ax1.set_title('Task 1')
ax1.legend(loc='upper left', fancybox=True)

# Plot for Task 2
ax2.plot(test_x, pred_multi.mean[:, 1], label='Mean prediction', color='red')
# ax2.plot(test_x, train_y[:, 1], linestyle='--', label='True function', color='red')
lower = pred_multi.confidence_region()[0][:, 1].detach().numpy()
upper = pred_multi.confidence_region()[1][:, 1].detach().numpy()
ax2.fill_between(
test_x,
lower,
upper,
alpha=0.2,
label='Confidence interval',
color='red'
)
ax2.scatter(train_x, train_y[:, 1], color='gray', label='Training data')
ax2.set_title('Task 2')
ax2.legend(loc='upper left', fancybox=True)

# Adjust layout and show plot
fig.tight_layout()
fig.savefig('MTSE_Predictions.png')
# plt.show()





