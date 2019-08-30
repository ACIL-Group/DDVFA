%% This is an example of DDVFA's usage.
%
% PROGRAM DESCRIPTION
% This program exemplifies the usage of the DDVFA code provided.
%
% REFERENCES
% [1] L. E. Brito da Silva, I. Elnabarawy, and D. C. Wunsch II, "Distributed 
% dual vigilance fuzzy adaptive resonance theory learns online, retrieves 
% arbitrarily-shaped clusters, and mitigates order dependence," Neural Networks. 
% To appear.
% [2] L. E. Brito da Silva, D. C. Wunsch II, "A study on exploiting VAT to
% mitigate ordering effects in Fuzzy ART," Proc. Int. Joint Conf. Neural 
% Netw. (IJCNN), 2018, pp. 2351-2358.
% [3] G. Carpenter, S. Grossberg, and D. Rosen, "Fuzzy ART: Fast 
% stable learning and categorization of analog patterns by an adaptive 
% resonance system," Neural Networks, vol. 4, no. 6, pp. 759-771, 1991.
% [4] J. C. Bezdek and R. J. Hathaway, "VAT: a tool for visual assessment
% of (cluster) tendency," Proc. Int. Joint Conf. Neural Netw. (IJCNN), 
% vol. 3, 2002, pp. 2225-2230.
% 
% Code written by Leonardo Enzo Brito da Silva
% Under the supervision of Dr. Donald C. Wunsch II
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Clean up
clear variables; close all; fclose all; echo off; clc;

%% Add path
addpath('classes', 'functions', 'data');

%% Reproducibility
seed = 0;
generator = 'twister';        
rng(seed, generator); 

%% Data
fprintf('=========================================================\n');
fprintf('Setting up.\n');
fprintf('=========================================================\n');

% Load data
fprintf('Loading data...\n');
load('ACIL.mat')
[nSamples, dim] = size(data);
fprintf('Done.\n');

% Linear Normalization
fprintf('Normalizing data...\n');
data = mapminmax(data', 0, 1);
data = data';
fprintf('Done.\n');

% Randomization
fprintf('Randomizing data...\n');
Prng = randperm(nSamples);
data_rng = data(Prng, :);
fprintf('Done.\n');

%% DDVFA + Merge ART (until convergence)
fprintf('=========================================================\n');
fprintf('DDVFA + Merge ART (until convergence).\n');
fprintf('=========================================================\n');
fprintf('\t\t\t\t >>>>> DDVFA <<<<< \n');

% DDVFA parameters
params_DDVFA_1 = struct();
params_DDVFA_1.rho_lb = 0.80;
params_DDVFA_1.rho_ub = 0.85;
params_DDVFA_1.alpha = 1e-3;
params_DDVFA_1.beta = 1;
params_DDVFA_1.gamma = 3;
params_DDVFA_1.gamma_ref = 1;
nEpochs = 1;
method = 'single';

% DDVFA
DDVFA_1 = DistDualVigFuzzyART(params_DDVFA_1);  
DDVFA_1.display = true;
DDVFA_1 = DDVFA_1.train(data_rng, nEpochs, method); 

fprintf('\t\t\t\t >>>>> Merge ART <<<<< \n');

% Merge ART parameters
params_MFA            = struct();
params_MFA.rho        = DDVFA_1.rho;
params_MFA.alpha      = DDVFA_1.alpha;
params_MFA.beta       = DDVFA_1.beta;
params_MFA.gamma      = DDVFA_1.gamma;
params_MFA.gamma_ref  = DDVFA_1.gamma_ref;
nEpochs_MFA = 1;

% Merge ART
t = 1;
fprintf('Merge ART iteration: %d \n', t);
MFA = MergeFuzzyART(params_MFA);
MFA.display = true;
MFA = MFA.train(DDVFA_1, nEpochs_MFA);
MFA_old = MFA;
while true
    t = t + 1;
    fprintf('Merge ART iteration: %d \n', t);
    MFA.F2 = {};
    MFA = MFA.train(MFA_old, nEpochs_MFA); 
    if MFA.nCategories == MFA_old.nCategories        
        MFA = MFA.compress();   
        break;
    end    
    MFA_old = MFA;
end

%% VAT + DDVFA
fprintf('=========================================================\n');
fprintf('VAT + DDVFA\n');
fprintf('=========================================================\n');
fprintf('\t\t\t\t >>>>> VAT <<<<< \n');

% Sorting using VAT
fprintf('Running VAT...\n');
M = pdist2(data_rng, data_rng);
[R, Pvat] = VAT(M);
data_vat = data_rng(Pvat, :);
fprintf('Done.\n');

fprintf('\t\t\t\t >>>>> DDVFA <<<<< \n');

% DDVFA parameters
params_DDVFA_2 = struct();
params_DDVFA_2.rho_lb = 0.80;
params_DDVFA_2.rho_ub = 0.85;
params_DDVFA_2.alpha = 1e-3;
params_DDVFA_2.beta = 1;
params_DDVFA_2.gamma = 3;
params_DDVFA_2.gamma_ref = 1;
nEpochs = 1;
method = 'single';

% DDVFA
DDVFA_2 = DistDualVigFuzzyART(params_DDVFA_2);  
DDVFA_2.display = true;
DDVFA_2 = DDVFA_2.train(data_vat, nEpochs, method); 

%% Plot Categories
fprintf('=========================================================\n');
fprintf('Plotting results\n');
fprintf('=========================================================\n');
fprintf('Plotting categories...\n');

LINEWIDTH = 2;

figure
set(gcf, 'color', 'w', 'units', 'normalized', 'outerposition', [0 0 1 1]) 
subplot(1,3,1)
draw_categories(DDVFA_1, data_rng, LINEWIDTH)
title('DDVFA (shuffled data)')
subplot(1,3,2)
draw_categories(MFA, data_rng, LINEWIDTH)
title('DDVFA + Merge ART (shuffled data)')
subplot(1,3,3)
draw_categories(DDVFA_2, data_vat, LINEWIDTH)
title('VAT + DDVFA (VAT pre-processed data)')

fprintf('Done.\n');