%% """ Gamma-normalized Fuzzy ART for Merge Fuzzy ART"""
% 
% PROGRAM DESCRIPTION
% This is a MATLAB implementation of the "Gamma-normalized Fuzzy ART"
% network for Merge ART.
%
% REFERENCES
% [1] L. E. Brito da Silva, I. Elnabarawy, and D. C. Wunsch II, "Distributed 
% dual vigilance fuzzy adaptive resonance theory learns online, retrieves 
% arbitrarily-shaped clusters, and mitigates order dependence," Neural Networks. 
% To appear.
% [2] G. Carpenter, S. Grossberg, and D. Rosen, "Fuzzy ART: Fast 
% stable learning and categorization of analog patterns by an adaptive 
% resonance system," Neural networks, vol. 4, no. 6, pp. 759–771, 1991.
% 
% Code written by Leonardo Enzo Brito da Silva
% Under the supervision of Dr. Donald C. Wunsch II
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Gamma-normalized Fuzzy ART Class for Merge ART
classdef gammaFuzzyARTmerge    
    properties (Access = public)	% default properties' values are set
        rho = 0.6;                  % vigilance parameter: [0,1]  
        alpha = 1e-3;               % choice parameter: alpha > 0 
        beta = 1;                   % learning parameter: (0,1] (beta=1: "fast learning")    
        gamma = 3;                  % "pseudo" kernel width: gamma >= 1
        gamma_ref = 1;              % "reference" gamma for normalization procedure: 0<= gamma_ref < gamma 
        W = [];                     % weight vectors 
        n = 0;                      % number of encoded samples per category ("instance counting")
        labels = [];                % cluster labels for each sample
        dim = [];                   % data set dimension  
        nCategories = 0;            % total number of categories
        Epoch = 0;                  % current epoch
        T = [];                     % category activation/choice function vector
        M = [];                     % category match function vector 
        thres = [];                 % vigilance threshold
        display = true;             % displays training progress on the command window (displays intermediate steps)
    end   
    properties (Access = private)
        W_old = [];                 % old weights used for stopping criterion    
    end
    methods        
        %% Assign property values from within the class constructor
        function obj = gammaFuzzyARTmerge(settings) 
            obj.rho         = settings.rho;
            obj.alpha       = settings.alpha;
            obj.beta        = settings.beta;
            obj.gamma       = settings.gamma;
            obj.gamma_ref	= settings.gamma_ref;
        end         
        %% Train
        function obj = train(obj, x, counter, maxEpochs)
            
            % Display progress on command window
            if obj.display
                fprintf('Starting Training...\n');                
                backspace = '';          
            end 
            
            % Data Information            
            [nSamples, ~] = size(x);
            obj.labels = zeros(nSamples, 1);
            
            % Initialization 
            if isempty(obj.W)             
                obj.W = x(1, :);   
                obj.n = counter(1);                
                obj.nCategories = 1;  
                sample_no1 = 2;
            else
                sample_no1 = 1;
            end             
            obj.W_old = obj.W;
            
            % Learning 
            obj.Epoch = 0;
            while(true)
                obj.Epoch = obj.Epoch + 1;
                for i=sample_no1:nSamples  % loop over samples
                    if or(isempty(obj.T), isempty(obj.M)) % Check for already computed activation/match values
                        obj = activation_match(obj, x(i, :));  % Compute Activation/Match Functions
                    end     
                    [~, index] = sort(obj.T, 'descend');  % Sort activation function values in descending order                    
                    mismatch_flag = true;  % mismatch flag 
                    for j=1:obj.nCategories  % loop over categories                       
                        bmu = index(j);  % Best Matching Unit 
                        if (obj.M(bmu) >= obj.thres) % Vigilance Check - Pass 
                            obj = learn(obj, x(i, :), counter(i), bmu);  % Learning
                            obj.labels(i) = bmu;  % update sample labels
                            mismatch_flag = false;  % mismatch flag 
                            break; 
                        end                               
                    end  
                    if mismatch_flag  % If there was no resonance at all then create new category
                        obj.nCategories = obj.nCategories + 1;          % increment number of categories
                        obj.W(obj.nCategories,:) = x(i, :);             % fast commit 
                        obj.n(obj.nCategories, 1) = counter(i);         % number of samples associated with new category
                        obj.labels(i) = obj.nCategories;                % update sample labels 
                    end 
                    obj.T = [];  % empty activation vector
                    obj.M = [];  % empty match vector
                    % Display progress on command window
                    if obj.display
                       progress = sprintf('\tEpoch: %d \n\tSample ID: %d \n\tCategories: %d \n', obj.Epoch, i, obj.nCategories);
                       fprintf([backspace, progress]);
                       backspace = repmat(sprintf('\b'), 1, length(progress)); 
                    end                     
                end  
                sample_no1 = 1; % Start loop from 1st sample from 2nd epoch and onwards
                % Stopping Conditions
                if stopping_conditions(obj, maxEpochs)
                    break;
                end 
                obj.W_old = obj.W;                                
            end  
            % Display progress on command window
            if obj.display
                fprintf('Done.\n');
            end
        end
        % Activation/Match Functions
        function obj = activation_match(obj, x)              
            obj.T = zeros(obj.nCategories, 1);     
            obj.M = zeros(obj.nCategories, 1); 
            for j=1:obj.nCategories 
                W_norm = norm(obj.W(j, :), 1);
                obj.T(j, 1) = (norm(min(x, obj.W(j, :)), 1)/(obj.alpha + W_norm))^obj.gamma;
                obj.M(j, 1) = (W_norm^obj.gamma_ref)*obj.T(j, 1);
            end
        end   
        % Learning
        function obj = learn(obj, x, counter, index)
            obj.W(index, :) = obj.beta*(min(x, obj.W(index, :))) + (1-obj.beta)*obj.W(index, :);    
            obj.n(index, 1) = obj.n(index) + counter;
        end      
        % Stopping Criteria
        function stop = stopping_conditions(obj, maxEpochs)
            stop = false; 
            if isequal(obj.W, obj.W_old)
                stop = true;                                         
            elseif obj.Epoch >= maxEpochs
                stop = true;
            end 
        end    
    end 
end