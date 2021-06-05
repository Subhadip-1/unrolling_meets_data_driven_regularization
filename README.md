# unrolling_meets_data_driven_regularization
Contains python scripts for adversarially learning an iteratively unrolled reconstruction together with a data-driven regularizer for inverse problems. 

* To run the codes, you need a Ubuntu 18.04/20.04 system with miniconda installed on it. The training and evaluation scripts were executed on an NVIDIA Quadro RTX 6000 GPU with 24 GB of memory. 

# Instructions to run the codes:
* First, create a root folder (with any name), inside which you have to keep all the python scripts and the data folders (see below). Since it is cumbersome to download the scripts one by one, we provide them here: https://drive.google.com/drive/folders/18PS3bYMv6qf4JVKlfuH6YpazAv1TkaJt?usp=sharing. Download this, unzip, and make it your root directory within which the following sub-folders must be stored.  
* The phantoms used in our CT experiments are available (as `.npy` files) here: https://drive.google.com/drive/folders/1SHN-yti3MgLmmW_l0agZRzMVtp0kx6dD?usp=sharing. Download the `.zip` file containing the phantoms, unzip, and put inside the same directory where the scripts are kept (created in the previous step).
* Download the folder containing some pre-trained models and unzip it: https://drive.google.com/drive/folders/176uhC1WB-ooImD9cWJMqW_KRubW_kw06?usp=sharing.
* Create a conda environment with the required dependencies by `conda env create -f env_uar.yml`, and then activate it by `conda activate env_tomo`.  
* Run `python simulate_projections_for_train_and_test.py` to simulate the projection data and the FBP reconstructions, to be used for training the UAR generator and regularizer. Alternatively, download the pre-simulated projection data and FBPs (along with the images arranged patient-wise) used in the experiment from here: https://drive.google.com/drive/folders/1gKytBtkTtGxBLRcNInx2OLty4Gie3pCX?usp=sharing.  
* Run `python unrolled_adv_reg_train.py` for training the proposed UAR approach. `lambda_adv_prior` is set to 0.1 in the script. Change it if you want to train for a different lambda. 
* To evalutate the UAR model(s) (both end-to-end and with refinement) on test data using pre-trained models, run `python unrolled_adv_reg_eval.py`. We have set `lambda_adv_prior=0.1` since it leads to the best performance. This script applies a pre-trained UAR model on all 128 test slices and computes the overall statistics.   
