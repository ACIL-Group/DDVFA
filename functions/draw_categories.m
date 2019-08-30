%% """ Visualization of DDVFA/MFA hyperboxes """
% 
% PROGRAM DESCRIPTION
% This program plots DDVFA/MFA categories (hyperboxes) and data samples 
% according to the partition obtained.
%
% INPUTS
% ART: trained DDVFA or MFA class object
% data: data set matrix (rows: samples, columns: features)
% lw: line width of hyperboxes
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

% Plot Categories
function draw_categories(ART, data, lw)
    clrs = rand(ART.nCategories, 3);
    gscatter(data(:,1), data(:,2), ART.labels, clrs, '.', 10, 'off')
    for i=1:ART.nCategories
        for j=1:ART.F2{i}.nCategories        
            x = ART.F2{i}.W(j, 1);
            y = ART.F2{i}.W(j, 2);
            w = 1 - ART.F2{i}.W(j, 3) - ART.F2{i}.W(j, 1);
            h = 1 - ART.F2{i}.W(j, 4) - ART.F2{i}.W(j, 2);        
            if and((w>0), (h>0))
                pos = [x y w h]; 
                r = rectangle('Position', pos);
                r.FaceColor = 'none';
                r.EdgeColor = clrs(i,:);
                r.LineWidth = lw;
                r.LineStyle = '-';
                r.Curvature = [0 0]; 
            else
                X = [ART.F2{i}.W(j, 1) 1 - ART.F2{i}.W(j, 3)];
                Y = [ART.F2{i}.W(j, 2) 1 - ART.F2{i}.W(j, 4)];
                l = line(X, Y);
                l.Color = clrs(i,:);
                l.LineStyle = '-';
                l.LineWidth = lw;
                l.Marker = 'none';
            end
        end
    end
    axis square
    box on   
end