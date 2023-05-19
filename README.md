![globecom](./assets/ieee-globecom@halfx.png)
![comsoc](./assets/ieee-comsoc-new@2x.png)
![ieee](./assets/ieee@2x.png)

IEEE Global Communications Conference  
4–8 December 2023  
Kuala Lumpur, Malaysia

# Code Repository Description
This code repository is to accompany the paper _Deep Learning for the Spectral Radius in 5G Wireless Networks_ submitted to GLOBECOM2023 in April 2023.

# Usage
Specify sampling rate and run data/synthesize.m to generate sampled channel gains as datasets.  
Specify sampling rate and run data/preprocess_matlab_data.py to generate ground truths, i.e., spectral radius, left eigenvectors, and right eigenvectors.  
Specify sampling rate and fading type and run ODENet.py to train and test ODE-Net models.  
Specify sampling rate and fading type and run RNN.py to train and test RNN models.
Specify sampling rate and fading type and run NN.py to train and test ICNN and DNN models.
