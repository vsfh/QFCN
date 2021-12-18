import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from fft import FFTConv2d
import numpy as np
from torch.distributions import Categorical


class qf_TimeBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution to each node of
    a graph in isolation.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param out_channels: Desired number of output channels at each node in
        each time step.
        :param kernel_size: Size of the 1D temporal kernel.
        """
        super(qf_TimeBlock, self).__init__()
        self.conv1 = FFTConv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2 = FFTConv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3 = FFTConv2d(in_channels, out_channels, (1, kernel_size))

    def activation_function_CapReLu(self,input_tensor, Cap):
        output_tensor = torch.clamp(input_tensor, min=0, max=Cap)
        return output_tensor

    def get_norm_matrix(self, input_tensor, conv): # conv should be self.conv1 for instance
        """Calculate the norm of A_p and F_q, but also the mu(A) and mu(F)"""
        kernel = conv.weight.data
        h = kernel.shape[2]
        w = kernel.shape[3]
        H = input_tensor.shape[2]
        W = input_tensor.shape[3]
        # 0: prepare matrices
        F = []
        A = []
        # 1: F_q norm calculations (kernel)
        kernels_norms = []
        for q in range(len(kernel)): #for each channels q
            Fq = kernel[q]
            kernels_norms.append(torch.norm(Fq).cpu().numpy())#add the norm of F_q to the list of norms
            Fq = Fq.detach().cpu().numpy().flatten()
            F.append(Fq) #add the column F_q to the matrix F
        kernels_norms = np.array(kernels_norms)
        F = np.array(F).transpose()
        # 2: A_p norm calculations (input)
        input_norms = []
        for i in range(H-h+1):
            for j in range(W-w+1): #for each 'kernel' subregion in the input
                Ap = input_tensor[0,0,0+i:h+i,0+j:w+j] #selected subregion
                norm_p = torch.norm(Ap).cpu()
                input_norms.append(norm_p.detach().numpy().flatten()) #add the norm of A_p to the list of norms
                Ap = Ap.detach().cpu().numpy().flatten() #transform as numpy vector
                A.append(Ap) #add the row A_p to the matrix A
        input_norms = np.array(input_norms)
        A = np.array(A)
        # 3: calculate mu=(Frob norm / Spectral norm)
        F_mu = np.linalg.norm(F, 'fro')/np.linalg.norm(F, 2)
        A_mu = np.linalg.norm(A, 'fro')/np.linalg.norm(A, 2)
        # 4: calculate the matrix of ||Ap||*||Fq||
        norm_matrix_output = np.outer(input_norms, kernels_norms) #we create the matrix of ||Ap||*||Fq||
        return norm_matrix_output, F_mu, A_mu

    def add_gaussian_noise(self, input_tensor, epsilon, norm_matrix):
        #norm_matrix is a matrix of ||Ap||*||Fq||
        #the noise matrix is gaussian values centered on 0 and with std deviation of epsilon
        #output = input +2*noise*||Ap||*||Fq||
        noise = torch.Tensor(input_tensor.cpu().data.new(input_tensor.size()).normal_(0, epsilon))
        norms_tensor = np.reshape(norm_matrix.transpose(),
                                  (1,input_tensor.shape[1],input_tensor.shape[2],input_tensor.shape[3]))
        norms_tensor = torch.from_numpy(norms_tensor) #it was a numpy object
        output_tensor = input_tensor.cpu() + 2*torch.mul(noise, norms_tensor, out=None) #torch.mul : entrywise multiplication noise*||Ap||*||Fq||
        output_tensor = torch.clamp(output_tensor, min=0, max=2) #no value should be less than 0 or more than Cap!
        return output_tensor

    def quantum_sampling(self, input_tensor, ratio):
        output_tensor = torch.zeros_like(input_tensor)
        num_samples = int(ratio*input_tensor[0].numel()) # ratio x number of elements in each tensor of the batch
        for i in range(input_tensor.shape[0]): #for each tensor in the batch
            x = input_tensor[i] # tensor n°i
            x_vec = x.flatten() # vectorize
            probabilities = x_vec #quantum sampling
            m = Categorical(probs=probabilities) # create the torch function to sample with probability distribution = x_vec
            sample_index = m.sample((num_samples,)) #sample num_samples times
            y = torch.zeros(x_vec.shape) # create zeros vector of same length
            y[sample_index] = 1 # this is the mask
            out = x_vec*y # apply mask
            out = out.reshape(x.shape) # reshape to initial tensor
            output_tensor[i] = out
        return output_tensor

    def calculate_average(self, input_tensor):
        average = torch.mean(input_tensor)
        return average.detach().numpy()

    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels)
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features_out=out_channels)
        """


        # Convert into NCHW format for pytorch to perform convolutions.
        X = X.permute(0, 3, 1, 2)
        norm_matrix_1, F_mu_1, A_mu_1= self.get_norm_matrix(X, self.conv1) # norm_matrix will be used in add_gaussian_noise later
        temp = self.conv1(X) + torch.sigmoid(self.conv2(X))
        out = self.activation_function_CapReLu(temp + self.conv3(X),2)
        out = self.add_gaussian_noise(out,0.1,norm_matrix_1)
        out = self.quantum_sampling(out,0.5)
        # Convert back from NCHW to NHWC
        out = out.permute(0, 2, 3, 1)
        return out


class TimeBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution to each node of
    a graph in isolation.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param out_channels: Desired number of output channels at each node in
        each time step.
        :param kernel_size: Size of the 1D temporal kernel.
        """
        super(TimeBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels)
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features_out=out_channels)
        """
        # Convert into NCHW format for pytorch to perform convolutions.
        X = X.permute(0, 3, 1, 2)
        temp = self.conv1(X) + torch.sigmoid(self.conv2(X))
        out = F.relu(temp + self.conv3(X))
        # Convert back from NCHW to NHWC
        out = out.permute(0, 2, 3, 1)
        return out


class STGCNBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution on each node in
    isolation, followed by a graph convolution, followed by another temporal
    convolution on each node.
    """

    def __init__(self, in_channels, spatial_channels, out_channels,
                 num_nodes):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param spatial_channels: Number of output channels of the graph
        convolutional, spatial sub-block.
        :param out_channels: Desired number of output features at each node in
        each time step.
        :param num_nodes: Number of nodes in the graph.
        """
        super(STGCNBlock, self).__init__()
        self.temporal1 = TimeBlock(in_channels=in_channels,
                                   out_channels=out_channels)
        self.Theta1 = nn.Parameter(torch.FloatTensor(out_channels,
                                                     spatial_channels))
        self.temporal2 = TimeBlock(in_channels=spatial_channels,
                                   out_channels=out_channels)
        self.batch_norm = nn.BatchNorm2d(num_nodes)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

    def forward(self, X, A_hat):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features=out_channels).
        """
        t = self.temporal1(X)
        lfs = torch.einsum("ij,jklm->kilm", [A_hat, t.permute(1, 0, 2, 3)])
        # t2 = F.relu(torch.einsum("ijkl,lp->ijkp", [lfs, self.Theta1]))
        t2 = F.relu(torch.matmul(lfs, self.Theta1))
        t3 = self.temporal2(t2)
        return self.batch_norm(t3)
        # return t3


class STGCN(nn.Module):
    """
    Spatio-temporal graph convolutional network as described in
    https://arxiv.org/abs/1709.04875v3 by Yu et al.
    Input should have shape (batch_size, num_nodes, num_input_time_steps,
    num_features).
    """

    def __init__(self, num_nodes, num_features, num_timesteps_input,
                 num_timesteps_output):
        """
        :param num_nodes: Number of nodes in the graph.
        :param num_features: Number of features at each node in each time step.
        :param num_timesteps_input: Number of past time steps fed into the
        network.
        :param num_timesteps_output: Desired number of future time steps
        output by the network.
        """
        super(STGCN, self).__init__()
        self.block1 = STGCNBlock(in_channels=num_features, out_channels=64,
                                 spatial_channels=16, num_nodes=num_nodes)
        self.block2 = STGCNBlock(in_channels=64, out_channels=64,
                                 spatial_channels=16, num_nodes=num_nodes)
        self.last_temporal = TimeBlock(in_channels=64, out_channels=64)
        self.fully = nn.Linear((num_timesteps_input - 2 * 5) * 64,
                               num_timesteps_output)

    def forward(self, A_hat, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        """
        out1 = self.block1(X, A_hat)
        out2 = self.block2(out1, A_hat)
        out3 = self.last_temporal(out2)
        out4 = self.fully(out3.reshape((out3.shape[0], out3.shape[1], -1)))
        return out4

class qf_STGCN(nn.Module):
    """
    Spatio-temporal graph convolutional network as described in
    https://arxiv.org/abs/1709.04875v3 by Yu et al.
    Input should have shape (batch_size, num_nodes, num_input_time_steps,
    num_features).
    """

    def __init__(self, num_nodes, num_features, num_timesteps_input,
                 num_timesteps_output):
        """
        :param num_nodes: Number of nodes in the graph.
        :param num_features: Number of features at each node in each time step.
        :param num_timesteps_input: Number of past time steps fed into the
        network.
        :param num_timesteps_output: Desired number of future time steps
        output by the network.
        """
        super(qf_STGCN, self).__init__()
        self.block1 = STGCNBlock(in_channels=num_features, out_channels=64,
                                 spatial_channels=16, num_nodes=num_nodes)
        self.block2 = STGCNBlock(in_channels=64, out_channels=64,
                                 spatial_channels=16, num_nodes=num_nodes)
        self.last_temporal = qf_TimeBlock(in_channels=64, out_channels=64)
        self.fully = nn.Linear((num_timesteps_input - 2 * 5) * 64,
                               num_timesteps_output)

    def forward(self, A_hat, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        """
        out1 = self.block1(X, A_hat)
        out2 = self.block2(out1, A_hat)
        out3 = self.last_temporal(out2).cuda()
        out4 = self.fully(out3.reshape((out3.shape[0], out3.shape[1], -1)))
        return out4


if __name__ == '__main__':
    net = STGCN(num_nodes=207, num_features=1, num_timesteps_input=20, num_timesteps_output=1).cuda()
    X = torch.randn(32, 207, 20 ,1).cuda()
    adj = torch.randn(207,207).cuda()
    output = net(adj, X)
    pass