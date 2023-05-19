% code with the Matlab R2021b version
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% create Rayleigh fading channel
% In Rayleigh model, only Non Line of Sight(NLOS) components are simulated 
% between transmitter and receiver. It is assumed that there is no LOS path 
% between transmitter and receiver.

% choose sampleRate from 5e3, 1e4, 5e4, 1e5 to generate different datasets
sample_rate = 5e4;
  
rayleigh_gain = zeros(1e5, 64);

for chan_id = 1:64
rayleigh_chan = comm.RayleighChannel( ...
    'SampleRate',sample_rate, ...
    'PathDelays',0, ...
    'AveragePathGains',0, ...
    'MaximumDopplerShift',200, ...
    'ChannelFiltering',false, ...
    'NumSamples',1e5, ...
    'FadingTechnique','Sum of sinusoids');

% synthesize Rayleigh channel gain data (both real and imaginary part)
rayleigh_gain(:,chan_id) = rayleigh_chan();
end


% create Rician fading channel
% In rician model, both Line of Sight (LOS) and non Line of Sight(NLOS) 
% components are simulated between transmitter and receiver.

rician_gain = zeros(1e5, 64);

for chan_id = 1:64
rician_chan = comm.RicianChannel( ...
    'SampleRate',sample_rate, ...
    'NumSamples',1e5, ...
    'KFactor',2.8, ...
    'DirectPathDopplerShift',5.0, ...
    'DirectPathInitialPhase',0.5, ...
    'MaximumDopplerShift',50, ...
    'DopplerSpectrum',doppler('Bell', 8), ...
    'ChannelFiltering',false);

% synthesize Rician channel gain data (both real and imaginary part)
rician_gain(:,chan_id) = rician_chan();
end


% create Nakagami fading channel
% It was originally developed empirically based on measurements. The Matlab did
% not provide an Object for the Nakagami fading channel yet.

nakagami_dist = makedist('Nakagami','mu',1,'omega',1);

% synthesize Nakagami channel gain data (both real and imaginary part)
nakagami_gain = random(nakagami_dist,[1e5,64]) + 1i*random(nakagami_dist,[1e5,64]);


% create Weibull fading channel
% Weibull distribution represents another generalization of the Rayleigh 
% distribution. The Matlab did not provide an Object for the Weibull fading 
% channel yet.

weibull_dist = makedist('Weibull','A',1,'B',1);

% synthesize Weibull channel gain data (both real and imaginary part)
weibull_gain = random(weibull_dist,[1e5,64]) + 1i*random(weibull_dist,[1e5,64]);


% save channel gain data into .mat file for later reuse by Python
save('channel_gains.mat', ...
     'rayleigh_gain','rician_gain','nakagami_gain','weibull_gain');
load('channel_gains.mat')
