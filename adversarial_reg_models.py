import torch
import torch.nn as nn
import numpy as np
import torch.autograd as autograd

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.cuda.FloatTensor

class ConvBlock(nn.Module):
    def __init__(self, in_size, out_size, stride=1, kernel_size=5):
        super(ConvBlock, self).__init__()
        self.pad = (kernel_size - 1)//2
        layers = [nn.Conv2d(in_size, out_size, kernel_size=kernel_size, stride=stride, padding=self.pad, bias=True)]
        layers.append(nn.LeakyReLU(0.2))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

#a basic feed-forward critic
class ConvNetClassifier(nn.Module): #accepts images of size 512x512
    def __init__(self, in_channels=1, n_filters=16, add_l2 = True):
        super(ConvNetClassifier, self).__init__()

        self.conv1 = ConvBlock(in_channels, n_filters)
        self.conv2 = ConvBlock(n_filters, 2*n_filters)
        
        self.conv3 = ConvBlock(2*n_filters, 2*n_filters, stride=2)  
        
        self.conv4 = ConvBlock(2*n_filters, 4*n_filters, stride=2)
        self.conv5 = ConvBlock(4*n_filters, 4*n_filters, stride=2)
        
        self.conv6 = ConvBlock(4*n_filters, 8*n_filters, stride=2)  
        
        self.avg_pool = nn.AvgPool2d(kernel_size=16, stride=1, padding=0)
        self.act = nn.LeakyReLU(0.2)
        
        self.fc1 = nn.Linear(in_features=36992, out_features=512, bias=True)
        self.act1 = nn.LeakyReLU(0.2)
        
        self.fc2 = nn.Linear(in_features=512, out_features=1, bias=True)
        
        self.add_l2 = add_l2
        if self.add_l2:
            self.l2_penalty = nn.Parameter(-12.0 * torch.ones(1))
            

    def forward(self, x): # x = bs x in_channels x img_size x img_size
        inp = x
        x = self.conv1(x) 
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        
        x = self.act(self.avg_pool(x))
        
        z = self.act1(self.fc1(x.view(x.size(0), -1)))
        z = self.fc2(z)
        
        if self.add_l2: #add a small l2 term for coercivity
            l2_term = torch.sum(inp.view(inp.size(0), -1)**2, dim=1).view(inp.size(0),1)
            z_plus_l2 = z + (torch.nn.functional.softplus(self.l2_penalty))*l2_term
        else:
            z_plus_l2 = z

        return z_plus_l2, z # output is of size bs x 1
    
#build a resnet
class ResBlock(nn.Module):
    def __init__(self, in_size, n_filters = 16, downsample = False):
        super(ResBlock, self).__init__()
        layers = [ConvBlock(in_size, n_filters, stride=1, kernel_size=3)]
        if downsample:
            layers.append(nn.Conv2d(n_filters, n_filters, kernel_size=3, stride=2, padding=1, bias=True))
            self.conv = nn.Conv2d(in_size, n_filters, kernel_size=3, stride=2, padding=1, bias=True)
        else:
            layers.append(nn.Conv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1, bias=True))
            self.conv = nn.Conv2d(in_size, n_filters, kernel_size=3, stride=1, padding=1, bias=True)
            
                
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x) + self.conv(x)       
    
class ResNetClassifier(nn.Module): #accepts images of size 512x512
    def __init__(self, in_channels, n_filters = 16, add_l2 = True):
        super(ResNetClassifier, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, n_filters, kernel_size=3, stride=1, padding=1, bias=True)
        
        self.res = ResBlock(n_filters, n_filters)
        
        self.res_down1 = ResBlock(n_filters, 2*n_filters, downsample = True)
        self.res_down2 = ResBlock(2*n_filters, 2*n_filters, downsample = True)
        self.res_down3 = ResBlock(2*n_filters, 4*n_filters, downsample = True)
        self.res_down4 = ResBlock(4*n_filters, 8*n_filters, downsample = True)
        
        self.avg_pool = nn.AvgPool2d(kernel_size=16, stride=1, padding=0)
        self.act = nn.LeakyReLU(0.2)
        
        self.fc1 = nn.Linear(in_features=36992, out_features=512, bias=True)
        self.act1 = nn.LeakyReLU(0.2)
        
        self.fc2 = nn.Linear(in_features=512, out_features=1, bias=True)
        
        
        self.add_l2 = add_l2
        if self.add_l2:
            self.l2_penalty = nn.Parameter(-12.0 * torch.ones(1))
        
    
    def forward(self, x):
        inp = x
        x = self.conv1(x)
        x = self.res(x)
        
        x = self.res_down1(x)
        x = self.res_down2(x)
        x = self.res_down3(x)
        x = self.res_down4(x)
        
        x = self.act(self.avg_pool(x))
        
        z = self.act1(self.fc1(x.view(x.size(0), -1)))
        z = self.fc2(z)
        
        if self.add_l2: #add a small l2 term for coercivity
            l2_term = torch.sum(inp.view(inp.size(0), -1)**2, dim=1).view(inp.size(0),1)
            z_plus_l2 = z + (torch.nn.functional.softplus(self.l2_penalty))*l2_term
        else:
            z_plus_l2 = z
        
        return z_plus_l2, z   
    
####################### LPD model ##########################
class cnn_data_space(nn.Module):
    def __init__(self, n_filters=32, n_in_channels=3, n_out_channels = 1, kernel_size=5):
        super(cnn_data_space, self).__init__()
        self.pad = (kernel_size-1)//2
        self.conv1 = nn.Conv2d(n_in_channels, out_channels=n_filters, kernel_size=kernel_size, stride=1, padding=self.pad, bias=True)
        self.conv2 = nn.Conv2d(n_filters, n_filters, kernel_size=kernel_size, stride=1, padding=self.pad, bias=True)
        self.conv3 = nn.Conv2d(n_filters, out_channels=1, kernel_size=kernel_size, stride=1, padding=self.pad, bias=True)
        
        self.act1 = nn.PReLU(num_parameters=1, init=0.25)
        self.act2 = nn.PReLU(num_parameters=1, init=0.25)
        
        
    def forward(self, h, y, z):
        h = h.type(dtype)
        y = y.type(dtype)
        z = z.type(dtype)
        
        dh1 = torch.cat((h, y, z), dim=1)
        dh2 = self.act1(self.conv1(dh1))
        dh3 = self.act2(self.conv2(dh2))
        dh4 = self.conv3(dh3)
        return h + dh4
    
class cnn_image_space(nn.Module):
    def __init__(self, n_filters=32, n_in_channels=2, n_out_channels = 1, kernel_size=5):
        super(cnn_image_space, self).__init__()
        self.pad = (kernel_size-1)//2
        self.conv1 = nn.Conv2d(n_in_channels, out_channels=n_filters, kernel_size=kernel_size, stride=1, padding=self.pad, bias=True)
        self.conv2 = nn.Conv2d(n_filters, n_filters, kernel_size=kernel_size, stride=1, padding=self.pad, bias=True)
        self.conv3 = nn.Conv2d(n_filters, out_channels=1, kernel_size=kernel_size, stride=1, padding=self.pad, bias=True)
        
        self.act1 = nn.PReLU(num_parameters=1, init=0.25)
        self.act2 = nn.PReLU(num_parameters=1, init=0.25)
        
    def forward(self, x, u):
        x = x.type(dtype)
        u = u.type(dtype)
        
        dx1 = torch.cat((x, u), dim=1)
        dx2 = self.act1(self.conv1(dx1))
        dx3 = self.act2(self.conv2(dx2))
        dx4 = self.conv3(dx3)
        return x + dx4
    
class LPD(nn.Module):
    def __init__(self, fwd_op, adjoint_op, n_filters=32, niter=20, sigma=0.01, tau=0.01): 
        super(LPD, self).__init__()
        
        self.fwd_op = fwd_op
        self.adjoint_op = adjoint_op
        self.niter = niter
        self.n_filters = n_filters
        
        self.sigma = nn.Parameter(sigma * torch.ones(self.niter).to(device).type(dtype))
        self.tau = nn.Parameter(tau * torch.ones(self.niter).to(device).type(dtype))
        self.cnn_image_layers = nn.ModuleList([cnn_image_space(n_filters=self.n_filters).to(device) for i in range(self.niter)])
        self.cnn_data_layers = nn.ModuleList([cnn_data_space(n_filters=self.n_filters).to(device) for i in range(self.niter)])
        
    def forward(self, y, x_init):
        x = x_init
        #h = self.fwd_op(x_init)
        #h = y
        h = torch.zeros_like(y).type(dtype)
        for iteration in range(self.niter):
            h = self.cnn_data_layers[iteration](h, y, self.sigma[iteration] * self.fwd_op(x))
            x = self.cnn_image_layers[iteration](x, self.tau[iteration] * self.adjoint_op(h))
        return x
    
####### gradient penalty loss #######
def compute_gradient_penalty(network, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.cuda.FloatTensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    #validity = net(interpolates)
    out_plus_l2, out = network(interpolates)
    fake = torch.cuda.FloatTensor(np.ones(out.shape)).requires_grad_(False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(outputs=out, inputs=interpolates,
                              grad_outputs=fake, create_graph=True, retain_graph=True,
                              only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty