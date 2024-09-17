import gpytorch
class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_tasks,Q):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_tasks
        )
        self.covar_module = gpytorch.kernels.LCMKernel(
            [gpytorch.kernels.SpectralMixtureKernel(num_mixtures=1, ard_num_dims=1) for i in range(Q)],
            num_tasks=num_tasks, rank=2,
            task_covar_prior= None
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
# Model and likelihood
