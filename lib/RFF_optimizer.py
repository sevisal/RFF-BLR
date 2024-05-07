import torch
import numpy as np
from sklearn.utils import check_random_state


class RFF_optimizer(torch.nn.Module):

    def __init__(self, X, y, lr=1e0, gamma=1., n_components=100, random_state=None, mult_gamma = False, links = False):
        '''
        This class computes and optimizes the gamma parameter of a RFF kernel approximation to a RBF kernel
        using the ELBO as the objective function.
        
        Parameters
        ----------
        X : numpy array
            Input data with shape NxD.
        y : numpy array
            Output data with shape Nx1.
        tau : float
            Precision of the likelihood.
        alphas : numpy array
            Precision of the prior of the weights.
        lr : float
            Learning rate of the optimizer.
        gamma : float
            Initial value of the gamma parameter.
        n_components : int
            Number of random Fourier features.
        random_state : int
            Random seed for reproducibility.
        mult_gamma : bool
            If True, the gamma parameter is a vector of length D.
        links : bool
            If True, concatenate input with hidden layer output.
        '''
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
       
        self.X = torch.from_numpy(X).to(self.device)
        self.y = torch.from_numpy(y).to(self.device)

        random_state = check_random_state(random_state)
        n_features = X.shape[1]
        self.n_data = X.shape[0]
        
        self.mult_gamma = mult_gamma
        self.gamma = gamma
        self.n_components = n_components
        self.links = links
        self.random_weights_unscaled_ = torch.from_numpy(random_state.normal(size=(n_features, self.n_components))).to(self.device)
        self.random_offset_ = torch.from_numpy(random_state.uniform(0, 2 * np.pi, size=self.n_components)).to(self.device)

        #Internally, for the optimization, we work in log scale with gamma
        if mult_gamma:
            if np.isscalar(gamma):
                self.gamma_log = torch.nn.Parameter(torch.from_numpy(np.array(np.log(gamma*np.ones(self.n_components,)))).to(self.device))
            else:
                self.gamma_log = torch.nn.Parameter(torch.from_numpy(np.array(np.log(gamma))).to(self.device))

        else:
            self.gamma_log = torch.nn.Parameter(torch.from_numpy(np.array(np.log(gamma))).to(self.device))

        self.opt = torch.optim.Adam(self.parameters(), lr=lr)
        self.to(self.device)

    def computeRFF(self, X, input_idx):
        '''
        Computes the Random Fourier Features for a given input X
        Parameters
        ----------
        X : numpy array
            Input data with shape NxD.
        input_idx : numpy array
            Indexes of the input data to be used in the RFF.

        Returns
        -------
        torch tensor
            Random Fourier Features of the input data with or without the links.
        '''
        if not torch.is_tensor(X):
            X = torch.from_numpy(X).to(self.device)  
        
        if self.mult_gamma:
            self.random_weights_ = torch.sqrt(2 * self.gamma_log.exp().flatten()[input_idx]) * self.random_weights_unscaled_[:,input_idx[:self.n_components]]
        else:
            self.random_weights_ = torch.sqrt(2 * self.gamma_log.exp().flatten()) * self.random_weights_unscaled_[:,input_idx[:self.n_components]]
        X_rff = torch.cos(X @ self.random_weights_ + self.random_offset_[input_idx[:self.n_components]]) * np.sqrt(2./self.n_components)
        if self.links:
            return torch.cat((X_rff, X[:,input_idx[self.n_components:]]), dim=1)
        else:
            return X_rff

    def forward(self):
        '''
        Defines the RBF kernel and calculate the ELBO.
        Parameters
        ----------
        None.

        Returns
        -------
        float
            ELBO value.
        '''

        Z = self.computeRFF(self.X, self.input_idx)
        trace1 = torch.trace(torch.mm(self.y.T, torch.mm(self.W_mean, Z.T)))
        trace2 = torch.trace(torch.mm(self.WW, torch.mm(Z.T,Z)))
        trace3 = torch.trace(torch.mm(self.b.T, torch.mm(self.W_mean, Z.T)))
        L = -0.5*self.tau * (-2*trace1 + trace2 + trace3)
        return -L

    def get_params(self):
        '''
        Returns the lengthscale and the variance of the RBF kernel in numpy form

        Returns
        -------
        float32 numpy array
            A row vector of length D with the value of each lengthscale in float32.

        '''
        return self.gamma_log.exp().data.cpu().numpy()

    def get_RFF(self, X, input_idx):
        """
        Computes the Random Fourier Features for a given input X
        Parameters
        ----------
        X : numpy array
            Input data with shape NxD.
        input_idx : numpy array
            Indexes of the input data to be used in the RFF.
        """
        return self.computeRFF(X, input_idx).data.cpu().numpy()

    def sgd_step(self, W_mean, WW, tau, b, it, input_idx):
        '''
        Computes "it" steps of the Adam optimizer to optimize our ELBO.
        Parameters
        ----------
        ZAT : numpy array
            Matrix product of Z@A.T with shape NxN'..
        it : integer
            Integer that indicates how many steps of the optimizer has to do.

        Returns
        -------
        None.

        '''
        self.W_mean = torch.from_numpy(W_mean).to(self.device)
        self.WW = torch.from_numpy(WW).to(self.device)
        self.tau = torch.from_numpy(np.array(tau)).to(self.device)
        self.b = torch.from_numpy(np.array(b)).to(self.device)
        self.input_idx = input_idx
        for i in range(it):
            self.opt.zero_grad()
            self.L = self.forward()
            self.L.backward()
            self.opt.step()