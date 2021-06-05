import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import os
import odl

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

import mayo_utils, torch_wrapper, adversarial_reg_models
from skimage.measure import compare_ssim, compare_psnr
############ dataloaders #######################
print('creating dataloaders...')
output_datapath = './mayo_data_arranged_patientwise/'
transform_to_tensor = [transforms.ToTensor()]

eval_dataloader = DataLoader(mayo_utils.mayo_dataset(output_datapath, transforms_=transform_to_tensor, aligned = True, mode = 'test'),\
                              batch_size = 1, shuffle = False)

print('number of images during eval = %d'%len(eval_dataloader))

############################################
##############specify geometry parameters#################
import simulate_projections_for_train_and_test
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
lambda_adv_prior=0.1
#pretrained = False
uar_generator = adversarial_reg_models.LPD(fwd_op, adjoint_op, niter=20, sigma=0.01, tau=0.01).to(device) #generator model
adv_reg = adversarial_reg_models.ConvNetClassifier(in_channels=1, n_filters=16, add_l2 = False).to(device)

#load the pre-trained models for lambda=0.1 for quick eval
uar_generator.load_state_dict(torch.load(model_path + "gan_ar_generator_mayo_lambda={:.2e}.pt".format(lambda_adv_prior),map_location=torch.device(device)))
adv_reg.load_state_dict(torch.load(model_path + "gan_ar_regularizer_mayo_lambda={:.2e}.pt".format(lambda_adv_prior),map_location=torch.device(device)))

#if pretrained: #load the pre-trained models for lambda=0.1 for quick eval, set pretrained = False while not using pretrained models
#    uar_generator.load_state_dict(torch.load(model_path + "gan_ar_generator_mayo.pt",map_location=torch.device(device)))
#    adv_reg.load_state_dict(torch.load(model_path + "gan_ar_regularizer_mayo.pt",map_location=torch.device(device)))
#else:
#    uar_generator.load_state_dict(torch.load(model_path + "gan_ar_generator_mayo_lambda={:.2e}.pt".format(lambda_adv_prior),map_location=torch.device(device)))
#    adv_reg.load_state_dict(torch.load(model_path + "gan_ar_regularizer_mayo_lambda={:.2e}.pt".format(lambda_adv_prior),map_location=torch.device(device)))

print('networks loaded successfully!')

##################################################################################    
####### variational optimizer for the learned prior ####################
sq_loss = torch.nn.MSELoss(reduction='mean') #data-fidelity loss

def var_optimizer(adv_reg, x_init, x_ground_truth, y_test, n_iter, lambda_var, lr=0.50): 
    x_out = x_init.clone().detach().requires_grad_(True).to(device) 
    x_optimizer = torch.optim.SGD([x_out], lr=lr)
    x_test_np = x_ground_truth.cpu().detach().numpy()
    data_range = np.max(x_test_np) - np.min(x_test_np)
    
    for iteration in np.arange(n_iter):
        x_optimizer.zero_grad()
        y_out = fwd_op(x_out)
        data_loss = torch.sqrt(sq_loss(y_test, y_out)) #RMSE works better than MSE
        
        ####### compute the regularization term ############
        prior_plus_l2, prior = adv_reg(x_out)
        regularizer = lambda_var*prior.mean()
        variational_loss = data_loss + regularizer
        variational_loss.backward(retain_graph=True)
        x_optimizer.step()

        x_np = x_out.cpu().detach().numpy().squeeze()
        psnr = compare_psnr(np.squeeze(x_test_np),x_np,data_range=data_range)
        ssim = compare_ssim(np.squeeze(x_test_np),x_np,data_range=data_range)
        
        if(iteration%20==0):
            recon_log = '[iter: {:d}/{:d}\t PSNR: {:.4f}, SSIM: {:.4f}, var_loss: {:.6f}, regularization: AR {:.6f}\n]'\
            .format(iteration, n_iter, psnr, ssim, variational_loss.item(), regularizer.item())
            print(recon_log)
            
    
    x_np = x_out.cpu().detach().numpy().squeeze()
    x_np = mayo_utils.cut_image(x_np, vmin=0.0, vmax=1.0)
    psnr = compare_psnr(np.squeeze(x_test_np),x_np,data_range=data_range)
    ssim = compare_ssim(np.squeeze(x_test_np),x_np,data_range=data_range)
    return x_np

##################################################################################  
################## run evaluation on test data ###################################
log_file_name = "unrolled_adv_reg_reconstruction_log_lambda={:.2e}.txt".format(lambda_adv_prior)
try:
    os.remove(log_file_name)
except OSError:
    pass

log_file = open(log_file_name, "w+")
log_file.write("################ reconstruction log for UAR\n ################")
  
recon_image_path = './UAR_recon_image/'   
os.makedirs(recon_image_path, exist_ok=True) 
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
psnr_all = [] #create an empty list to save the data
ssim_all = []
time_all = []

for idx, batch in enumerate(eval_dataloader):
    phantom = batch["phantom"].to(device) #true images
    
    #####simulate proj. data and FBP#########
    start.record()
    sinogram, fbp = simulate_projections_for_train_and_test.compute_projection(phantom, num_angles=num_angles, det_shape=det_shape, \
                       space_range=space_range, geom=geom, noise_std_dev=noise_std_dev)
    end.record()
    torch.cuda.synchronize()
    time_fbp = start.elapsed_time(end)
    phantom_image = phantom.cpu().detach().numpy().squeeze()
    data_range = np.max(phantom_image) - np.min(phantom_image)
    fbp_image = mayo_utils.cut_image(fbp.cpu().detach().numpy().squeeze(), vmin=0.0, vmax=1.0)
    fbp = torch.from_numpy(fbp_image).view(fbp.size()).to(device)
    psnr_fbp = compare_psnr(phantom_image,fbp_image,data_range=data_range)
    ssim_fbp = compare_ssim(phantom_image,fbp_image,data_range=data_range)
    
    ######### reconstruct with the UAR generator ##########
    start.record()
    recon_end2end = uar_generator(sinogram, fbp)
    end.record()
    torch.cuda.synchronize()
    time_end2end = start.elapsed_time(end)
    
    recon_end2end_image = recon_end2end.cpu().detach().numpy().squeeze()
    recon_end2end_image = mayo_utils.cut_image(recon_end2end_image, vmin=0.0, vmax=1.0)
    psnr_end2end = compare_psnr(phantom_image,recon_end2end_image,data_range=data_range)
    ssim_end2end = compare_ssim(phantom_image,recon_end2end_image,data_range=data_range)
    np.save(recon_image_path + 'uar_end2end_slice{:d}_lambda={:.2e}'.format(idx,lambda_adv_prior) + '.npy', recon_end2end_image)
    ######### reconstruct with the corresponding regularizer (refinement)#########
    message = "refining the generator estimate by gradient-descent on the variational loss...\n"
    print(message)
    log_file.write(message)
    ### run refinement ####
    n_iter, lambda_var, lr = 100, lambda_adv_prior, 0.50
    x_init = recon_end2end #init with the end2end solution
    start.record()
    recon_refinement_image = var_optimizer(adv_reg, x_init, phantom, sinogram, n_iter=n_iter, lambda_var=lambda_var, lr=lr)
    end.record()
    torch.cuda.synchronize()
    time_refinement = time_end2end + start.elapsed_time(end)   
    psnr_refinement = compare_psnr(phantom_image,recon_refinement_image,data_range=data_range)
    ssim_refinement = compare_ssim(phantom_image,recon_refinement_image,data_range=data_range)
    np.save(recon_image_path + 'uar_refinement_slice{:d}_lambda={:.2e}'.format(idx,lambda_adv_prior) + '.npy', recon_refinement_image)
    
    recon_log = 'test-image: [{:d}/{:d}], FBP: PSNR = {:.2f}, SSIM = {:.2f}\t\
        lambda = {:.2e}: UAR end-to-end: PSNR = {:.2f}, SSIM = {:.2f};\t UAR refined: PSNR = {:.2f}, SSIM = {:.2f}\n'.\
        format((idx+1), len(eval_dataloader), psnr_fbp, ssim_fbp, lambda_adv_prior, psnr_end2end, ssim_end2end, psnr_refinement, ssim_refinement)
    print(recon_log)
    log_file.write(recon_log)
    
    psnr_all.append((psnr_fbp, psnr_end2end, psnr_refinement))
    ssim_all.append((ssim_fbp, ssim_end2end, ssim_refinement))
    time_all.append((time_fbp, time_end2end, time_refinement))    


###overall stat####
psnr_all_numpy = np.array(psnr_all, dtype=np.float32)
ssim_all_numpy = np.array(ssim_all, dtype=np.float32)
time_all_numpy = np.array(time_all, dtype=np.float32)


#### close log-file #### 
recon_log = 'average performance:: FBP: PSNR {:.2f} +/- {:.2f}, SSIM {:.2f} +/- {:.2f}, time {:.1f} +/- {:.1f}\t \
            end-to-end: PSNR {:.2f} +/- {:.2f}, SSIM {:.2f} +/- {:.2f}, time {:.1f} +/- {:.1f}\t \
            refinement: PSNR {:.2f} +/- {:.2f}, SSIM {:.2f} +/- {:.2f}, time {:.1f} +/- {:.1f}\n'.\
            format(np.mean(psnr_all_numpy[:,0]), np.std(psnr_all_numpy[:,0]),np.mean(ssim_all_numpy[:,0]), np.std(ssim_all_numpy[:,0]),np.mean(time_all_numpy[:,0]), np.std(time_all_numpy[:,0]),\
                   np.mean(psnr_all_numpy[:,1]), np.std(psnr_all_numpy[:,1]),np.mean(ssim_all_numpy[:,1]), np.std(ssim_all_numpy[:,1]),np.mean(time_all_numpy[:,1]), np.std(time_all_numpy[:,1]),\
                   np.mean(psnr_all_numpy[:,2]), np.std(psnr_all_numpy[:,2]),np.mean(ssim_all_numpy[:,2]), np.std(ssim_all_numpy[:,2]),np.mean(time_all_numpy[:,2]), np.std(time_all_numpy[:,2]))   
log_file.write(recon_log)  
print(recon_log)  
log_file.close()     
