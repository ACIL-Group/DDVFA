%% """ Distributed Dual Vigilance Fuzzy ART (DDVFA)"""
% 
% PROGRAM DESCRIPTION
% This is a MATLAB implementation of the "Distributed Dual Vigilance Fuzzy ART" (DDVFA) network.
%
% REFERENCES
% [1] L. E. Brito da Silva, I. Elnabarawy, and D. C. Wunsch II, "Distributed 
% dual vigilance fuzzy adaptive resonance theory learns online, retrieves 
% arbitrarily-shaped clusters, and mitigates order dependence," Neural Networks. 
% To appear.
% [2] G. Carpenter, S. Grossberg, and D. Rosen, "Fuzzy ART: Fast 
% stable learning and categorization of analog patterns by an adaptive 
% resonance system," Neural Networks, vol. 4, no. 6, pp. 759–771, 1991.
% 
% Code written by Leonardo Enzo Brito da Silva
% Under the supervision of Dr. Donald C. Wunsch II
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Distributed Dual Vigilance Fuzzy ART Class
classdef DistDualVigFuzzyART    
    properties (Access = public)                        % default properties' values are set
        rho = 0.6;                                      % vigilance parameter: [0,1]  
        alpha = 1e-3;                                   % choice parameter: alpha > 0
        beta = 1;                                       % learning parameter: (0,1] (default: beta=1 "fast learning")  
        gamma = 3;                                      % "pseudo" kernel width: gamma >= 1
        gamma_ref = 1;                                  % "reference" gamma for normalization: 0<= gamma_ref < gamma
        thres = [];                                     % vigilance threshold
        F2 = {};                                        % Global ART's F2 nodes (each node is a Gamma-normalized Fuzzy ART (FA))
        labels = [];                                    % cluster labels for each sample
        nCategories = 0;                                % number of F2 nodes
        Epoch = 0;                                      % current epoch
        settings = struct();                            % parameter setting of local FA nodes in DDVFA's F2 layer 
        method = 'single';                              % similarity method (activation and match): 'single', 'average', 'complete', 'median', 'weighted', or 'centroid'
        nSamples = [];                                  % number of data set samples
        dim = [];                                       % data set dimension 
        flags = struct();                               % flags
        display = true;                                 % displays training progress on the command window (displays intermediate steps)
    end   
    properties (Access = private)
        sample = [];                                    % current sample presented to DDVFA
        W = [];                                         % all F2 nodes' weight vectors
        W_old = [];                                     % old all F2 nodes' weight vectors (used for stopping criterion) 
    end
    methods        
        %% Assign property values from within the class constructor
        function obj = DistDualVigFuzzyART(settings) 
            % Global Fuzzy ART's settings
            obj.rho         = settings.rho_lb;  % lower bound vigilance parameter           
            obj.alpha       = settings.alpha;
            obj.beta        = settings.beta;
            obj.gamma       = settings.gamma;
            obj.gamma_ref	= settings.gamma_ref;
            % Local Fuzzy ARTs' settings
            obj.settings.rho        = settings.rho_ub; % upper bound vigilance parameter
            obj.settings.alpha      = settings.alpha;
            obj.settings.beta       = settings.beta;   
            obj.settings.gamma      = settings.gamma;
            obj.settings.gamma_ref	= settings.gamma_ref;
            % Flags
            obj.flags = struct('complement_coding', false, ...                     
                               'max_epoch', false, ...
                               'no_weight_change', false);            
        end         
        %% Train
        function obj = train(obj, data, maxEpochs, method) 
            % Display progress on command window
            if obj.display
                fprintf('Starting Training...\n');                
                backspace = '';          
            end 
            
            % Data Information            
            [obj.nSamples, obj.dim] = size(data);
            obj.labels = zeros(obj.nSamples, 1);
            
            % Train Settings
            obj.method = method;
            
            % Similarity type
            switch obj.method	% set function handle (simfcn)
                case 'single'	% single linkage
                    simfcn = @(obj, index, S) fcnmax(obj, index, S);
                case 'average'	% average linkage
                    simfcn = @(obj, index, S) fcnmean(obj, index, S);
                case 'complete'	% complete linkage
                    simfcn = @(obj, index, S) fcnmin(obj, index, S);
                case 'median'	% median linkage
                    simfcn = @(obj, index, S) fcnmedian(obj, index, S);
                case 'weighted'	% weighted linkage
                    simfcn = @(obj, index, S) fcnweighted(obj, index, S);
                case 'centroid'	% centroid linkage
                    simfcn = @(obj, index, S) fcncentroid(obj, index, S);
                otherwise                    
                    error('Error. \nThe method argument provided was: %s. \nMethod must be one of the following strings: ''single'', ''average'', ''complete'', ''median'', ''weighted'' or ''centroid''.', method)
            end
                        
            % Normalization and Complement coding
            x = DistDualVigFuzzyART.complement_coder(data);

            % Initialization             
            if isempty(obj.F2)
                % Global Fuzzy ART
                obj.nCategories = 1;
                obj.labels(1) = 1;
                sample_no1 = 2;
                % local Fuzzy ART
                obj.F2{obj.nCategories} = gammaFuzzyART(obj.settings);
                obj.F2{obj.nCategories}.W = x(1, :);   
                obj.F2{obj.nCategories}.n = 1;                
                obj.F2{obj.nCategories}.nCategories = 1; 
                obj.F2{obj.nCategories}.dim = obj.dim;
                obj.F2{obj.nCategories}.thres = obj.F2{obj.nCategories}.rho*(obj.dim^obj.gamma_ref); 
                obj.F2{obj.nCategories}.display = false;
            else
                sample_no1 = 1;
            end  
            obj.W_old = x(1, :);
            
            % Learning
            obj.thres = obj.rho*(obj.dim^obj.gamma_ref);
            obj.Epoch = 0;
            stop = false;
            while(~stop)
                obj.Epoch = obj.Epoch + 1;
                for i=sample_no1:obj.nSamples %loop over samples
                    obj.sample = x(i, :);
                    T = zeros(obj.nCategories, 1);  %T_class 
                    for j=1:obj.nCategories 
                        obj.F2{j} = obj.F2{j}.activation_match(x(i,:));  % Compute Activation/Match Functions   
                        T(1,j) = simfcn(obj, j, 'T');
                    end                       
                    [~, index] = sort(T, 'descend');  % Sort activation function values in descending order                    
                    mismatch_flag = true;  % Mismatch Flag
                    for j=1:obj.nCategories % loop over categories                         
                        bmu = index(j);  % Best Matching Unit                        
                        % Compute Match Function                           
                        M = simfcn(obj, bmu, 'M');
                        if (M >= obj.thres) % Vigilance Check - Pass   
                            nEpochs = 1;                            
                            obj.F2{bmu} = obj.F2{bmu}.train(x(i, :), nEpochs, 1); % Learning                            
                            obj.labels(i) = bmu;  % update sample labels 
                            mismatch_flag = false;  % mismatch flag 
                            break;
                        end                               
                    end   
                    if mismatch_flag  % If there was no resonance at all then create new category
                        % Global Fuzzy ART
                        obj.nCategories = obj.nCategories + 1;  % increment number of F2 nodes
                        obj.labels(i) = obj.nCategories;        % update sample labels 
                        % local Fuzzy ART
                        obj.F2{obj.nCategories} = gammaFuzzyART(obj.settings);
                        obj.F2{obj.nCategories}.W = x(i, :);   
                        obj.F2{obj.nCategories}.n = 1;                
                        obj.F2{obj.nCategories}.nCategories = 1; 
                        obj.F2{obj.nCategories}.dim = obj.dim;
                        obj.F2{obj.nCategories}.thres = obj.F2{obj.nCategories}.rho*(obj.dim^obj.gamma_ref); 
                        obj.F2{obj.nCategories}.display = false;
                    end                       
                    % Display progress on command window
                    if obj.display
                       s = 0;
                       for kx=1:obj.nCategories
                           s = s + obj.F2{kx}.nCategories;
                       end
                       progress = sprintf('\tEpoch: %d \n\tSample ID: %d \n\tNo. Clusters (FA nodes): %d \n\tTotal No. Categories: %d \n', obj.Epoch, i, obj.nCategories, s);
                       fprintf([backspace, progress]);
                       backspace = repmat(sprintf('\b'), 1, length(progress)); 
                    end  
                end     
                sample_no1 = 1; % Start loop from 1st sample from 2nd epoch and onwards 
                % Stopping Criteria                
                obj.W = [];
                for kx=1:obj.nCategories
                    obj.W = [obj.W ; obj.F2{kx}.W];
                end                
                if isequal(obj.W, obj.W_old)
                    obj.flags.no_weight_change = true;
                    stop = true;                
                elseif obj.Epoch >= maxEpochs  
                    obj.flags.max_epoch = true;
                    stop = true;
                end                                
                % Update old values
                obj.W_old = obj.W;                
            end  
            % Display progress on command window
            if obj.display
                fprintf('Done.\n');
            end
        end  
             
        % Similarity Methods 
        function value = fcnmax(obj, index, S) 
            value = max(obj.F2{index}.(S));                        
        end 
        function value = fcnmean(obj, index, S)
            value = mean(obj.F2{index}.(S));           
        end  
        function value = fcnmin(obj, index, S)
            value = min(obj.F2{index}.(S)); 
        end  
        function value = fcnmedian(obj, index, S)
            value = median(obj.F2{index}.(S));
        end  
        function value = fcnweighted(obj, index, S)
            value = obj.F2{index}.(S)' * (obj.F2{index}.n / sum(obj.F2{index}.n));           
        end 
        function value = fcncentroid(obj, index, S)
            Wc = min(obj.F2{index}.W, [], 1);     
            T = (norm(min(obj.sample, Wc), 1)/(obj.alpha + norm(Wc, 1)))^obj.gamma;  
            switch S
                case 'T'
                    value = T;   
                case 'M'
                    value = (norm(Wc, 1)^obj.gamma_ref)*T;  
            end
        end 
    end    
    methods(Static)  
        % Linear Normalization and Complement Coding
        function x = complement_coder(data)
            x = mapminmax(data', 0, 1);
            x = x';
            x = [x 1-x];
        end         
    end
end