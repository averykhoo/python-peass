function [y,logSig,diffLogSig] = myMapping(x,W,b,v)
% Non linear map from the feature scale to the subjective scale using
% sigmoids.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Version 1.0
% Copyright 2010 Valentin Emiya (INRIA).
% This software is distributed under the terms of the GNU Public License
% version 3 (http://www.gnu.org/licenses/gpl.txt).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if nargout>2
    [logSig diffLogSig] = myLogSig(W*x+b*ones(1,size(x,2)));
else
    logSig = myLogSig(W*x+b*ones(1,size(x,2)));
end

y = v.'*logSig;

return
