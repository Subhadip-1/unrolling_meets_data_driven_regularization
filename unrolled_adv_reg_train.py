import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import odl
import os

#from skimage.metrics import peak_signal_noise_ratio as compare_psnr
#from skimage.metrics import structural_similarity as compare_ssim
from skimage.measure import compare_ssim, compare_psnr

#get the device details
lambda_adv_prior =0.1 #other values are: 1e-3, 1e-2, 0.1
torch.cuda.set_device(0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

import mayo_utils, torch_wrapper, adversarial_reg_models

############ dataloaders #######################
print('creating dataloaders...')
output_datapath = './mayo_data_arranged_patientwise/'
transform_to_tensor = [transforms.ToTensor()]
train_dataloader = DataLoader(mayo_utils.mayo_dataset(output_datapath, transforms_=transform_to_tensor, aligned = True, mode = 'train'),\
                              batch_size = 1, shuffle = True)

print('number of minibatches during training = %d'%len(train_dataloader))

eval_dataloader = DataLoader(mayo_utils.mayo_dataset(output_datapath, transforms_=transform_to_tensor, aligned = True, mode = 'test'),\
                              batch_size = 1, shuffle = False)
n_val_images = len(eval_dataloader)
print('number of minibatches during eval = %d'%n_val_images)

############################################
##############specify geometry parameters#################
from simulate_projections_for_train_and_test import img_size, space_range, num_angles, det_shape, noise_std_dev, geom

space = odl.uniform_discr([-space_range, -space_range], [space_range, space_range],\
                              (img_size, img_size), dtype='float32', weighting=1.0)
if(geom=='parallel_beam'):
    geometry = odl.tomo.geometry.parallel.parallel_beam_geometry(space, num_angles=num_angles, det_shape=det_shape)
else:
    geometry = odl.tomo.geometry.conebeam.cone_beam_geometry(space, src_radius=1.5*space_range, \
                                                             det_radius=5.0, num_angles=num_angles, det_shape=det_shape)
    
fwd_op_odl = odl.tomo.RayTransform(space, geometry, impl='astra_cuda')
op_norm = 1.1 * odl.power_method_opnorm(fwd_op_odl)
print('operator norm = {:.4f}'.format(op_norm))

fbp_op_odl = odl.tomo.fbp_op(fwd_op_odl)
adjoint_op_odl = fwd_op_odl.adjoint

fwd_op = torch_wrapper.OperatorModule(fwd_op_odl).to(device)
fbp_op = torch_wrapper.OperatorModule(fbp_op_odl).to(device)
adjoint_op = torch_wrapper.OperatorModule(adjoint_op_odl).to(device)

############# networks #####################
model_path = './trained_models_mayo_neurips/'
os.makedirs(model_path, exist_ok=True)
lpd_generator = adversarial_reg_models.LPD(fwd_op, adjoint_op, n_filters=32, niter=20, sigma=0.01, tau=0.01).to(device) #generator model
###### regularizer model #########
network_name = 'ConvNet' #two options: 'ResNet' or 'ConvNet'

if network_name=='ConvNet':       
    adv_reg = adversarial_reg_models.ConvNetClassifier(in_channels=1, n_filters=16, add_l2 = False).to(device)
else:
    adv_reg = adversarial_reg_models.ResNetClassifier(in_channels=1, n_filters=16, add_l2 = False).to(device)
    
#################### first, train a baseline regularizer using FBP ##################################
log_file_name = model_path + "gan_ar_log_mayo_lambda={:.2e}.txt".format(lambda_adv_prior)
try:
    os.remove(log_file_name)
except OSError:
    pass
    
log_file = open(log_file_name, "w+")

message = 'initializing a baseline regularizer: lambda = {:.2e}...\n'.format(lambda_adv_prior)
log_file.write(message)
print(message)  

ar_optimizer = torch.optim.Adam(adv_reg.parameters(), lr=1e-4, betas=(0.5,0.99))
lambda_gp, num_minibatches, n_epochs = 10.0, 20, 10 #just make a few passes over the data

for epoch in np.arange(n_epochs):
    total_loss, total_diff = 0.0, 0.0
    for idx, batch in enumerate(train_dataloader):
        ############################################
        phantom = batch["phantom"].to(device).requires_grad_(False) #true images
        fbp = batch["fbp"].to(device).requires_grad_(False) #FBP
        
        ####### clip the FBP image in [0, 1] #########
        fbp_image = mayo_utils.cut_image(fbp.cpu().detach().numpy().squeeze(), vmin=0.0, vmax=1.0)
        fbp = torch.from_numpy(fbp_image).view(fbp.size()).to(device)

        #compute losses and update the regularizer
        phantom_out_plus_l2, phantom_out = adv_reg(phantom)
        fbp_out_plus_l2, fbp_out = adv_reg(fbp)
        
        diff_loss = phantom_out.mean()  - fbp_out.mean() 
        gp_loss = adversarial_reg_models.compute_gradient_penalty(adv_reg, phantom.data, fbp.data)
        loss = diff_loss + lambda_gp * gp_loss
        
        #parameter update
        ar_optimizer.zero_grad()
        loss.backward()
        ar_optimizer.step()
        
        total_loss += loss.item()
        total_diff += diff_loss.item()
        
        if(idx % num_minibatches == num_minibatches-1):
            ####### true-vs-FBP net #######
            avg_loss = total_loss/num_minibatches
            avg_diff = total_diff/num_minibatches
            
            total_loss, total_diff = 0.0, 0.0
            ########### print and save the training log ##################
            train_log = "epoch:[{}/{}] batch:[{}/{}], avg_loss = {:.8f}, avg_diff =  {:.8f}\n".\
                  format(epoch+1, n_epochs, idx+1, len(train_dataloader), avg_loss, avg_diff)
            print(train_log)
            log_file.write(train_log)

########### save the baseline regularizer
torch.save(adv_reg.state_dict(),  model_path + "initial_baseline_regularizer_mayo_lambda={:.2e}.pt".format(lambda_adv_prior))     

############ compute variational loss ############
sq_loss = nn.MSELoss(reduction='mean')
def compute_variational_loss(adv_reg, fwd_op, image, sinogram, lambda_adv_prior = lambda_adv_prior):
    data_loss = sq_loss(sinogram, fwd_op(image))
    prior_plus_l2, prior = adv_reg(image)
    return data_loss + lambda_adv_prior * prior.mean()    
    
########################## 
##### train LPD to optimize the variational loss with the initial regularizer. This step initializes the LPD generator. 
message = 'initializing the generator: lambda = {:.2e}...\n'.format(lambda_adv_prior)
log_file.write(message)
print(message)  

G_optimizer = torch.optim.Adam(lpd_generator.parameters(), lr=1e-4, betas=(0.5,0.99))
num_epochs_lpd_init = 5 #just make a few passes over the data

for epoch in range(num_epochs_lpd_init):
    for idx, batch in enumerate(train_dataloader):
        phantom = batch["phantom"].to(device).requires_grad_(False)  #true images
        fbp = batch["fbp"].to(device).requires_grad_(False)  #FBP
        sinogram = batch["sinogram"].to(device).requires_grad_(False)  #sinogram
        
        ####### clip the FBP image in [0, 1] #########
        fbp_image = mayo_utils.cut_image(fbp.cpu().detach().numpy().squeeze(), vmin=0.0, vmax=1.0)
        x_init = torch.from_numpy(fbp_image).view(fbp.size()).to(device).requires_grad_(False)
        fbp = torch.from_numpy(fbp_image).view(fbp.size()).to(device)
        
        #### compute recon and variational loss
        recon = lpd_generator(sinogram, x_init)
        var_loss = compute_variational_loss(adv_reg, fwd_op, recon, sinogram)
        
        #### update generator
        G_optimizer.zero_grad()
        var_loss.backward(retain_graph=True)
        G_optimizer.step()

        ############# print some log ################
        if(idx%num_minibatches==num_minibatches-1):
            phantom_image = phantom.cpu().detach().numpy().squeeze()
            data_range = np.max(phantom_image) - np.min(phantom_image)
            recon_image = recon.cpu().detach().numpy().squeeze()
            recon_image = mayo_utils.cut_image(recon_image, vmin=0.0, vmax=1.0)
            
            psnr_gan_ar = compare_psnr(phantom_image,recon_image,data_range=data_range)
            ssim_gan_ar = compare_ssim(phantom_image,recon_image,data_range=data_range)
            psnr_fbp = compare_psnr(phantom_image,fbp_image,data_range=data_range)
            ssim_fbp = compare_ssim(phantom_image,fbp_image,data_range=data_range)
            
            train_log = "epoch:[{}/{}] batch:[{}/{}] FBP: PSNR {:.4f}, SSIM {:.4f}\t GAN-AR: PSNR {:.4f}, SSIM {:.4f}".\
               format(epoch+1, num_epochs_lpd_init, idx+1, len(train_dataloader), psnr_fbp, ssim_fbp, psnr_gan_ar, ssim_gan_ar)
            
            print(train_log)
            log_file.write(train_log)

###### save the initial generator
torch.save(lpd_generator.state_dict(),  model_path + "initial_lpd_generator_mayo_lambda={:.2e}.pt".format(lambda_adv_prior)) 
########################## 
##### jointly train the generator and the regularizer in adversarial manner 
message = 'jointly training the generator and the regularizer adversarially: lambda = {:.2e}...\n'.format(lambda_adv_prior)
log_file.write(message)
print(message)  
########## slightly reduce lr for stability ###################
G_optimizer = torch.optim.Adam(lpd_generator.parameters(), lr=2e-5, betas=(0.5,0.99))
ar_optimizer = torch.optim.Adam(adv_reg.parameters(), lr=2e-5, betas=(0.5,0.99))
lambda_gp, n_epochs = 10.0, 25
num_G_updating = 2
torch.autograd.set_detect_anomaly(True) #for debug
for epoch in range(n_epochs):
    for idx, batch in enumerate(train_dataloader):
        phantom = batch["phantom"].to(device).requires_grad_(False) #true images
        fbp = batch["fbp"].to(device).requires_grad_(False) #FBP
        sinogram = batch["sinogram"].to(device).requires_grad_(False) #sinogram
        
        ####### clip the FBP image in [0, 1] #########
        fbp_image = mayo_utils.cut_image(fbp.cpu().detach().numpy().squeeze(), vmin=0.0, vmax=1.0)
        x_init = torch.from_numpy(fbp_image).view(fbp.size()).to(device).requires_grad_(False)
        fbp = torch.from_numpy(fbp_image).view(fbp.size()).to(device)
        
        #first, update the unrolled LPD network for a fixed regularizer
        for _ in range(num_G_updating):
            recon = lpd_generator(sinogram, x_init)
            var_loss = compute_variational_loss(adv_reg, fwd_op, recon, sinogram)
            G_optimizer.zero_grad()
            var_loss.backward(retain_graph=True)
            G_optimizer.step()
            
        ### now, update the regularizer network to tell apart true from recon
        phantom_out_plus_l2, phantom_out = adv_reg(phantom)
        recon_out_plus_l2, recon_out = adv_reg(recon)
        diff_loss = phantom_out.mean()  - recon_out.mean() 
        gp_loss = adversarial_reg_models.compute_gradient_penalty(adv_reg, phantom.data, recon.data)
        loss = diff_loss + lambda_gp * gp_loss
        #parameter update
        ar_optimizer.zero_grad()
        loss.backward(retain_graph=True)
        ar_optimizer.step()
        
        ############# print some log ################
        if(idx%num_minibatches==num_minibatches-1):
            phantom_image = phantom.cpu().detach().numpy().squeeze()
            data_range = np.max(phantom_image) - np.min(phantom_image)
            recon_image = recon.cpu().detach().numpy().squeeze()
            recon_image = mayo_utils.cut_image(recon_image, vmin=0.0, vmax=1.0)
            
            psnr_gan_ar = compare_psnr(phantom_image,recon_image,data_range=data_range)
            ssim_gan_ar = compare_ssim(phantom_image,recon_image,data_range=data_range)
            psnr_fbp = compare_psnr(phantom_image,fbp_image,data_range=data_range)
            ssim_fbp = compare_ssim(phantom_image,fbp_image,data_range=data_range)
            
            train_log = "epoch:[{}/{}] batch:[{}/{}] FBP: PSNR {:.4f}, SSIM {:.4f}\t GAN-AR: PSNR {:.4f}, SSIM {:.4f}".\
               format(epoch+1, n_epochs, idx+1, len(train_dataloader), psnr_fbp, ssim_fbp, psnr_gan_ar, ssim_gan_ar)
            
            print(train_log)
            log_file.write(train_log)

        ######### save trained models #############
        torch.save(lpd_generator.state_dict(),  model_path + "gan_ar_generator_mayo_lambda={:.2e}.pt".format(lambda_adv_prior)) 
        torch.save(adv_reg.state_dict(),  model_path + "gan_ar_regularizer_mayo_lambda={:.2e}.pt".format(lambda_adv_prior))    

#################################
log_file.close()  
