import torch

from torch import nn
from torch.autograd import Function

class IdentityFunction(Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        return input

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        return grad_output
    
class IdentityLayer(nn.Module):
    def __init__(self):
        # An identity layer does nothing
        super().__init__()
        self.identity = IdentityFunction.apply

    def forward(self, inp):
        # An identity layer just returns whatever it gets as input.
        return self.identity(inp)
    

class ReLUFunction(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.ident = torch.where(input > 0, 1.0, 0.0)
        
        return torch.max(input, torch.tensor(0))

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.ident
    
class ReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = ReLUFunction.apply
        return


    def forward(self, input):     
        return self.relu(input)
    

class LinearFunction(Function):
    @staticmethod
    def forward(ctx, inp, weight, bias):
        ctx.inp = inp
        ctx.weight = weight
        
        return inp @ weight + bias
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output @ ctx.weight.T
        grad_weight = ctx.inp.T @ grad_output
        grad_bias = torch.ones(grad_output.shape[0]) @ grad_output
        
        return grad_input, grad_weight, grad_bias
    
class Linear(nn.Module):
    def __init__(self, input_units, output_units, use_cuda):
        super().__init__()
        # initialize weights with small random numbers from normal distribution
        self.weight = nn.Parameter(torch.normal(0, 2 / torch.sqrt(torch.tensor(input_units + output_units)), size=(output_units, input_units)))
        self.bias = nn.Parameter(torch.zeros(output_units))
        self.func = LinearFunction.apply
        self.use_cuda = use_cuda

    def forward(self, inp):
        return self.func(inp, self.weight.T, self.bias)

    def __repr__(self):
        return 'Linear({1:d}, {0:d})'.format(*self.weight.shape)
    

class LogSoftmaxFunction(Function):
    @staticmethod
    def forward(ctx, inp):
        inp_minus_max = inp - torch.max(inp, dim=-1, keepdim=True).values
        ctx.inp_minus_max = inp_minus_max

        return inp_minus_max - torch.log(torch.sum(torch.exp(inp_minus_max), keepdim=True, dim=-1))

    @staticmethod
    def backward(ctx, grad_output):
        softmax_inp = torch.exp(ctx.inp_minus_max) / torch.sum(torch.exp(ctx.inp_minus_max), keepdim=True, dim=-1) 
        
        return grad_output - torch.sum(grad_output, dim=-1, keepdim=True) * softmax_inp
    
class LogSoftmax(nn.Module):
    def __init__(self):
        super().__init__()
        self.func = LogSoftmaxFunction.apply

    def forward(self, input):
        return self.func(input)
    

class DropoutFunction(Function):
    @staticmethod
    def forward(ctx, inp, p):
        mask = torch.bernoulli((1 - p) * torch.ones_like(inp)) / (1 - p)
        ctx.mask = mask
        
        return inp * mask

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.mask, None
    
class Dropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p
        self.func = DropoutFunction.apply

    def forward(self, input):
        if self.training:
            return self.func(input, self.p)
        
        return input

class CrossEntropyFunction(Function):
    @staticmethod
    def forward(ctx, activations, target):
        ctx.n_classes = activations.shape[-1]
        ctx.target = target
        
        return (-1) * torch.mean(torch.gather(activations, -1, target.unsqueeze(-1)))

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ((-1) * torch.eye(ctx.n_classes)[ctx.target]) / ctx.target.shape[0], None
    
class CrossEntropy(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.func = CrossEntropyFunction.apply

    def forward(self, activations, target):
        return self.func(activations, target)
    
class Network(nn.Module):
    def __init__(self, input_size=28*28, hidden_layers_size=256, num_layers=5,
                 num_classes=10, activation='relu', dropout_prob=0.1, use_cuda=False):
        super().__init__()
        self.activations = nn.ModuleDict({
            'relu': ReLU(),
        })
        
        self.layers = nn.ModuleList()
        
        for i in range(num_layers - 1):
            self.layers.add_module(f'fc{i}', Linear(input_size, hidden_layers_size, use_cuda=use_cuda))
            if dropout_prob:
                self.layers.add_module(f'dropout{i}', Dropout(dropout_prob))
                
            self.layers.add_module(f'relu{i}', self.activations[activation])
            
            input_size = hidden_layers_size
        
        self.layers.add_module(f'classifier_head', Linear(input_size, num_classes, use_cuda=use_cuda))
        self.layers.add_module(f'logsoftmax', LogSoftmax())

    def forward(self, inp):
        for layer in self.layers:
            inp = layer(inp)
            
        return inp
    
    def predict(self, inp):
        for layer in self.layers:
            if repr(layer) == 'LogSoftmax()':
                inp = torch.argmax(inp, dim=-1)
            else:
                inp = layer(inp)
            
        return inp