function [OPS, TPS, IPS, APS] = ...
    map2SubjScale(qTarget, qInterf, qArtif, qGlobal)
% Non-linear mapping to subjective scale
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Version 1.0
% Copyright 2010 Valentin Emiya (INRIA).
% This software is distributed under the terms of the GNU Public License
% version 3 (http://www.gnu.org/licenses/gpl.txt).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

q = [qGlobal,qTarget, qInterf, qArtif];

taskQ = NaN(4,1);
for nTask = 1:4
    load(sprintf('paramTask%d.mat',nTask));
    % Check number of inputs
    switch size(W,2)
        case 1
            x = q(nTask);
        case 3
            x = q(2:4);
        case 4
            x = q;
        otherwise
            error('map2SubjScale:wrongParameters','Wrong parameters');
    end
    % Non-linear mapping
    taskQ(nTask) = myMapping(x(:),W,b,v);
end

% Threshold to output values in [0 100]
taskQ = min(100,max(0,taskQ));

OPS = taskQ(1);
TPS = taskQ(2);
IPS = taskQ(3);
APS = taskQ(4);
return