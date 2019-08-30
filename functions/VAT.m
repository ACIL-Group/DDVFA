%% """ VAT algorithm """
% 
% PROGRAM DESCRIPTION
% This is a MATLAB implementation of Visual Assessment of cluster Tendency (VAT).
%
% INPUT
% D: NxN dissimilarity matrix
%
% REFERENCES
% [1] J. C. Bezdek and R. J. Hathaway, "VAT: a tool for visual assessment
% of (cluster) tendency," Proc. Int. Joint Conf. Neural Netw. (IJCNN), 
% vol. 3, 2002, pp. 2225-2230.
%
% Code written by Leonardo Enzo Brito da Silva
% Under the supervision of Dr. Donald C. Wunsch II
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% VAT reordering algorithm 
function [Dstar, P, N] = VAT(D)

    % Setup
    N = size(D, 1);
    K = 1:N;
    P = zeros(1, N);

    % Starting point
    [~, ind] = max(D(:));
    [i, ~] = ind2sub([N N], ind);    
    P(1) = i;
    I(1) = i;
    J = K;
    J(J==i) = [];
    
    % Main VAT Loop
    for r=2:N
        Dtemp = D(I, J);
        [~, ind] = min(Dtemp(:));
        [~, j_temp] = ind2sub(size(Dtemp), ind);
        j = J(j_temp);        
        P(r) = j;
        I = [I; j];
        J(J==j) = [];        
    end
    
    % Output
    Dstar = D(P, P);
    
end