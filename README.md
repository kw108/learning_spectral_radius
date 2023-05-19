![globecom](./assets/ieee-globecom@halfx.png)
![comsoc](./assets/ieee-comsoc-new@2x.png)
![ieee](./assets/ieee@2x.png)

IEEE Global Communications Conference  
4–8 December 2023  
Kuala Lumpur, Malaysia

# Code Repository Description
This code repository is to accompany the paper _Deep Learning for the Spectral Radius in 5G Wireless Networks_ submitted to GLOBECOM2023 in April 2023.

# Usage
## Data and Ground Truths Generation
Specify sample_rate in data/synthesize.m and run it to generate sampled channel gains as datasets.  
Specify sampling_rate in data/preprocess_matlab_data.py and run it to generate ground truths, i.e., spectral radii, left eigenvectors, and right eigenvectors.  

## Baseline Models
Specify sampling_rate (5e3, 1e4, 5e4, 1e5), fading_type ('rayleigh', 'rician'), and network_type ('ICNN', 'DNN') in NN.py. Run it to train and test ICNN and DNN models.  

## Time Series Models
Specify sampling_rate (5e3, 1e4, 5e4, 1e5), fading_type ('rayleigh', 'rician'), and uniform (True or False) in RNN.py. Run it to train and test RNN models.  
Specify sampling_rate (5e3, 1e4, 5e4, 1e5), fading type ('rayleigh', 'rician'), and uniform (True or False) in ODENet.py. Run it to train and test ODE-Net models.  
