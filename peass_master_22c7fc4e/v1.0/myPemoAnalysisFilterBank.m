function [gfb_out_dec, analyzer,M] = myPemoAnalysisFilterBank(x,fs,M)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Version 1.0
% Copyright 2010 Valentin Emiya (INRIA).
% This software is distributed under the terms of the GNU Public License
% version 3 (http://www.gnu.org/licenses/gpl.txt).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if ~isdeployed
    addpath(genpath('gammatone/'));
end

if nargin < 2,
    error('Not enough input arguments.');
end

% ----------------------------------------------------------------------
% settings
% ----------------------------------------------------------------------
MinCF = 20;				% (desired) minimum center frequency of the Gammatone filterbank in Hz
MaxCF = fs/2; % (desired) maximum center frequency of the Gammatone filterbank in Hz
base_freq = 1000;			% one of the gammatone filters will have this center frequency
filters_per_ERB = 3.0;		% density of gammatone filterbank

% upsampling of input signal to avoid aliasing in upper gammatone filters
fsOrig = fs;
if fs/2 < 1.5*MaxCF,
    x = resample(x, round(1.5*fs), fs);
    fs = round(1.5*fs);
end

% ----------------------------------------------------------------------
% actual signal processing
% ----------------------------------------------------------------------

% gammatone filterbank
analyzer = Gfb_Analyzer_new(fs, MinCF, base_freq, MaxCF, filters_per_ERB);
analyzer.fsOrig = fsOrig;
analyzer.fast = true;
[gfb_out, analyzer] = Gfb_Analyzer_process(analyzer, x(:).');
Nb = size(gfb_out,1);
if nargin<3 || isempty(M)
    M = exp(-2*1i*pi/fs*analyzer.center_frequencies_hz(:)*(0:size(gfb_out,2)-1));
end
gfb_out = gfb_out.*M;

% decimate
bw = erbBW(analyzer.center_frequencies_hz);
alpha_dec = 2;
gfb_out_dec = cell(Nb,1);
Ndec = floor(fs./bw/alpha_dec);
for k=1:Nb
    gfb_out_dec{k} = resample(gfb_out(k,:),1,Ndec(k));
end

if nargout>1
    analyzer.Ndec = Ndec;
    analyzer.fs = fs;
    analyzer.bw = bw;
end
return;


