import torch
import torch.nn as nn
import numpy as np
from sklearn.utils import check_random_state

class RVFL(nn.Module):
    """
    Random Vector Functional Link (RVFL) model for regression.

    :param int input_size: Number of features in the input data
    :param int RFF_size: Number of random Fourier features
    :param int output_size: Number of output neurons
    :param float gamma: Width of the RFF kernel
    :param bool gamma_param: If True, gamma is a parameter to be learned
    :param bool links: If True, concatenate input with hidden layer output
    :param random_state: Random seed for reproducibility
    """
    def __init__(self, input_size, RFF_size, output_size, gamma, gamma_param=False, links=True, random_state=None):
        super(RVFL, self).__init__()
        self.RFF_size = RFF_size

        self.links = links
        # Randomly initialize weights for input to hidden layer
        if links:
            self.output_weights = nn.Parameter(torch.randn(input_size + RFF_size, output_size))
        else:
            self.output_weights = nn.Parameter(torch.randn(RFF_size, output_size))    
        random_state = check_random_state(random_state)
        self.random_weights_unscaled_ = torch.from_numpy(random_state.normal(size=(input_size, RFF_size)))
        self.random_offset_ = torch.from_numpy(random_state.uniform(0, 2 * np.pi, size=RFF_size))

        if gamma_param:
            self.gamma_log = torch.nn.Parameter(torch.from_numpy(np.array(np.log(gamma))))
        else:
            self.gamma_log = torch.from_numpy(np.array(np.log(gamma)))
        

    def computeRFF(self, X):
        """ 
        Compute random Fourier features of input data X.
        
        :param X: Input data
        :return: Random Fourier features of input data X
        """
        if not torch.is_tensor(X):
            X = torch.from_numpy(X).float()

        self.random_weights_ = torch.sqrt(2 * self.gamma_log.exp().flatten()).float() * self.random_weights_unscaled_.float()
        return torch.cos(X @ self.random_weights_ + self.random_offset_.float()) * np.sqrt(2./self.RFF_size).astype(np.float32)

    def forward(self, X):
        """
        Forward pass of the RVFL model.

        :param X: Input data
        :return: Output of the RVFL model
        """
        if not torch.is_tensor(X):
            X = torch.from_numpy(X).float()

        # Input to hidden layer
        hidden_output = self.computeRFF(X)
        
        if self.links:
            # Concatenate input with hidden layer output
            combined_input = torch.cat((X, hidden_output), dim=1)
        else:
            combined_input = hidden_output
        
        # Output
        output = torch.matmul(combined_input, self.output_weights)
        return output