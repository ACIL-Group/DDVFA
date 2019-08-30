%% """ Merge Fuzzy ART """
% 
% PROGRAM DESCRIPTION
% This is a MATLAB implementation of the "Merge ART" network.
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

% Merge Fuzzy ART Class
classdef MergeFuzzyART   
    properties (Access = public)        % default properties' values are set
        rho = 0.6;                      % vigilance parameter: [0,1]  
        alpha = 1e-3;                   % choice parameter:  alpha > 0
        beta = 1;                       % learning parameter: (0,1] (default: beta=1 "fast learning")
        gamma = 3;                      % "pseudo" kernel width: gamma >= 1
        gamma_ref = 1;                  % "reference" gamma for normalization procedure: 0<= gamma_ref < gamma 
        F2 = {};                        % F2 nodes (each node is a Gamma-normalized Fuzzy ART)
        labels = [];                    % cluster labels for each sample
        nCategories = 0;                % number of categories
        dim = [];                       % original dimension of data set  
        Epoch = 0;                      % current epoch   
        method = [];                    % similarity method (activation and match): inherited from DDVFA input
        simfcn;                         % function handle for similarity measure
        settings = struct();            % parameter setting of FA nodes in DDVFA's F2 layer
        display = true;                 % displays training progress on the command window (displays intermediate steps)
    end   
    properties (Access = private)
        sample = [];                    % current sample presented to Merge FA
        T = [];                         % activation/choice function vector
        M = [];                         % match function vector
        Tmatrices = {};                 % activation/choice function matrices   
    end
    methods        
        %% Assign property values from within the class constructor
        function obj = MergeFuzzyART(settings) 
            obj.rho         = settings.rho;            
            obj.alpha       = settings.alpha;
            obj.beta        = settings.beta;
            obj.gamma       = settings.gamma;
            obj.gamma_ref	= settings.gamma_ref;
        end         
        %% Train
        function obj = train(obj, DDVFA, maxEpochs) 
            
            % Display progress on command window
            if obj.display
                fprintf('Starting Merging...\n');                
                backspace = '';          
            end 
            
            % Data Information          
            nSamples = DDVFA.nCategories;
            ARTs = DDVFA.F2;
            obj.labels = zeros(length(DDVFA.labels), 1); 
            obj.method = DDVFA.method;
            obj.settings = DDVFA.settings;
            obj.dim = DDVFA.dim;
            
            % Similarity type
            switch obj.method  % set function handle simfcn
                case 'single'  % single linkage
                    obj.simfcn = @(SimMat) MergeFuzzyART.fcnmax(SimMat);
                    activation = @(obj, ARTs) activation1(obj, ARTs);
                    match = @(obj, ARTs) match1(obj, ARTs);
                case 'average'  % average linkage
                    obj.simfcn = @(SimMat) MergeFuzzyART.fcnmean(SimMat);
                    activation = @(obj, ARTs) activation1(obj, ARTs);
                    match = @(obj, ARTs) match1(obj, ARTs);
                case 'complete'  % complete linkage
                    obj.simfcn = @(SimMat) MergeFuzzyART.fcnmin(SimMat);
                    activation = @(obj, ARTs) activation1(obj, ARTs);
                    match = @(obj, ARTs) match1(obj, ARTs);
                case 'median'  % median linkage
                    obj.simfcn = @(SimMat) MergeFuzzyART.fcnmedian(SimMat);
                    activation = @(obj, ARTs) activation1(obj, ARTs);
                    match = @(obj, ARTs) match1(obj, ARTs);
                case 'weighted'  % weighted linkage
                    obj.simfcn = @(SimMat) MergeFuzzyART.fcnweighted(SimMat);
                    activation = @(obj, ARTs) activation2(obj, ARTs);
                    match = @(obj, ARTs) match1(obj, ARTs);
                case 'centroid'  % centroid linkage
                    obj.simfcn = [];
                    activation = @(obj, ARTs) activation3(obj, ARTs);
                    match = @(obj, ARTs) match3(obj, ARTs);
            end
            
            % Initialization             
            if isempty(obj.F2)
                obj.nCategories = 1;
                obj.F2{obj.nCategories} = ARTs{1};                 
                obj.labels(DDVFA.labels==1) = 1;                  
                sample_no1 = 2;
            else
                sample_no1 = 1;
            end   
            
            % Learning
            obj.Epoch = 0;
            while(true)
                obj.Epoch = obj.Epoch + 1;
                for i=sample_no1:nSamples %loop over samples 
                    obj.sample = ARTs{i};                    
                    if isempty(obj.T) % Check for already computed activation values
                        obj = activation(obj, ARTs{i});  % Compute Activation Function  
                    end                       
                    [~, index] = sort(obj.T, 'descend'); % Sort activation function values in descending order                    
                    mismatch_flag = true;  % Mismatch Flag 
                    for j=1:obj.nCategories % loop over categories                         
                        bmu = index(j);  % Best Matching Unit   
                        if isempty(obj.M)  % Check for already computed match values
                            obj = match(obj, ARTs{i});  % Compute Match Function    
                        end
                        if (obj.M(bmu) >= obj.rho) % Vigilance Check - Pass                             
                            obj = learn(obj, ARTs{i}, bmu);  % Combine/link
                            obj.labels(DDVFA.labels==i) = bmu;  % update sample labels 
                            mismatch_flag = false;  % mismatch flag 
                            break;
                        end                               
                    end                       
                    if mismatch_flag  % If there was no resonance at all then create new category
                        obj.nCategories = obj.nCategories + 1;          % increment number of F2 nodes
                        obj.F2{obj.nCategories} = ARTs{i};              % fast commit                
                        obj.labels(DDVFA.labels==i) = obj.nCategories;  % update sample labels  
                    end 
                    obj.T = [];  % empty activation vector
                    obj.M = [];  % empty match vector
                    obj.Tmatrices = {};  % empty activation matrix
                    % Display progress on command window
                    if obj.display
                       progress = sprintf('\tEpoch: %d \n\tSample ID: %d \n\tNo. Categories (FA nodes): %d \n', obj.Epoch, i, obj.nCategories);
                       fprintf([backspace, progress]);
                       backspace = repmat(sprintf('\b'), 1, length(progress)); 
                    end                     
                end     
                sample_no1 = 1; % Start loop from 1st sample from 2nd epoch and onwards
                % Stopping Criteria                               
                if obj.Epoch >= maxEpochs  
                    break;
                end 
            end
            % Display progress on command window
            if obj.display
                fprintf('Done.\n');
            end
        end          
        % Pairwise Activation Matrix
        function Tmatrix = activation_matrix(obj, index, ART)  
            Tmatrix = zeros(obj.F2{index}.nCategories, ART.nCategories);
            for i=1:obj.F2{index}.nCategories
                W = obj.F2{index}.W(i, :);
                W_norm = norm(W, 1);
                for j=1:ART.nCategories                        
                    x = ART.W(j, :);
                    tij = (norm(min(x, W), 1)/(obj.alpha + W_norm))^obj.gamma;   
                    Tmatrix(i, j) = tij;
                end
            end 
        end 
        % Activation1 Function (single, average, complete and median)
        function obj = activation1(obj, ART)            
            obj.T = zeros(obj.nCategories, 1);  % T_class  
            obj.Tmatrices = cell(obj.nCategories, 1);
            for k=1:obj.nCategories 
                obj.Tmatrices{k, 1} = activation_matrix(obj, k, ART);
                obj.T(k, 1) = obj.simfcn(obj.Tmatrices{k, 1});
            end     
        end  
        % Activation 2 Function (weighted)
        function obj = activation2(obj, ART)            
            obj.T = zeros(obj.nCategories, 1);  % T_class  
            obj.Tmatrices = cell(obj.nCategories, 1);
            for k=1:obj.nCategories 
                obj.Tmatrices{k, 1} = activation_matrix(obj, k, ART);   
                probW = obj.F2{k}.n/(sum(obj.F2{k}.n));
                probx = ART.n/(sum(ART.n));   
                obj.Tmatrices{k, 1} = diag(probW)*obj.Tmatrices{k, 1}*diag(probx);
                obj.T(k, 1) = obj.simfcn(obj.Tmatrices{k, 1});
            end     
        end  
        % Activation 3 Function (centroid)
        function obj = activation3(obj, ART)            
            obj.T = zeros(obj.nCategories, 1);  % T_class 
            for k=1:obj.nCategories                 
                W = min(obj.F2{k}.W, [], 1);
                x = min(ART.W, [], 1); 
                obj.T(k, 1) = (norm(min(x, W), 1)/(obj.alpha + norm(W, 1)))^obj.gamma;                
            end              
        end                        
        % Pairwise Match Matrix
        function Mmatrix = match_matrix(obj, index, ART)
            Tmatrix = obj.Tmatrices{index, 1};
            Mmatrix = zeros(size(Tmatrix));
            for i=1:obj.F2{index}.nCategories
                W = obj.F2{index}.W(i, :);
                W_norm = norm(W, 1);
                for j=1:ART.nCategories 
                    x = ART.W(j, :);      
                    tij = Tmatrix(i, j);
                    mij = ((W_norm/norm(x, 1))^obj.gamma_ref)*tij;
                    Mmatrix(i, j) = mij;
                end
            end        
        end         
        % Match 1 Function (single, average, complete, median and weighted)
        function obj = match1(obj, ART)
            obj.M = zeros(obj.nCategories, 1);  % M_class            
            for k=1:obj.nCategories 
                Mmatrix = match_matrix(obj, k, ART);    
                obj.M(k, 1) = obj.simfcn(Mmatrix);
            end                       
        end  
        % Match 3 Function (centroid)
        function obj = match3(obj, ART)            
            obj.M = zeros(obj.nCategories, 1);  % M_class            
            for k=1:obj.nCategories 
                W = min(obj.F2{k}.W, [], 1);
                x = min(ART.W, [], 1); 
                obj.M(k, 1) = ((norm(W, 1)/norm(x, 1)).^obj.gamma_ref)*obj.T(k, 1);
            end                       
        end        
        % Learning
        function obj = learn(obj, ART, index)  
            obj.F2{index}.W = [obj.F2{index}.W; ART.W];
            obj.F2{index}.n = [obj.F2{index}.n; ART.n];
            obj.F2{index}.nCategories = obj.F2{index}.nCategories + ART.nCategories;  
        end          
        % Compression
        function obj = compress(obj)  
            for index=1:obj.nCategories
                FA = gammaFuzzyARTmerge(obj.settings);
                FA.display = false;
                FA.dim = obj.dim;
                FA.thres = FA.rho*(obj.dim^obj.gamma_ref);                
                nEpochs = 1;                         
                data = obj.F2{index}.W;
                instance_countings = obj.F2{index}.n;
                FA = FA.train(data, instance_countings, nEpochs);
                
                % local Fuzzy ART
                gFA = gammaFuzzyART(obj.settings);
                gFA.display = false;
                gFA.W = FA.W;   
                gFA.n = FA.n;                
                gFA.nCategories = FA.nCategories; 
                gFA.dim = obj.dim;
                gFA.thres = FA.rho*(obj.dim^obj.gamma_ref);
                gFA.Epoch = 1;
                
                % Update Merge FA's F2 node "index"
                obj.F2{index} = gFA; 
            end   
        end        
    end    
    methods(Static)
        % Similarity Methods 
        function value = fcnmax(SimMat) 
            value = max(SimMat(:));            
        end 
        function value = fcnmean(SimMat)
            value = mean(SimMat(:));            
        end  
        function value = fcnmin(SimMat)
            value = min(SimMat(:));            
        end  
        function value = fcnmedian(SimMat)
            value = median(SimMat(:));            
        end  
        function value = fcnweighted(SimMat)                      
            value = sum(SimMat(:));           
        end 
    end
end