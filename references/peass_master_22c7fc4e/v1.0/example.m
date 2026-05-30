function example
% Run this file to see an example
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Version 1.0
% Copyright 2010 Valentin Emiya (INRIA).
% This software is distributed under the terms of the GNU Public License
% version 3 (http://www.gnu.org/licenses/gpl.txt).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%
% Set inputs
%%%%%%%%%%%%
originalFiles = {...
    'example/targetSrc.wav';...
    'example/interfSrc1.wav';...
    'example/interfSrc2.wav'};
estimateFile = 'example/targetEstimate.wav';

%%%%%%%%%%%%%
% Set options
%%%%%%%%%%%%%
options.destDir = 'example/';
options.segmentationFactor = 1; % increase this integer if you experienced "out of memory" problems

%%%%%%%%%%%%%%%%%%%%
% Call main function
%%%%%%%%%%%%%%%%%%%%
res = PEASS_ObjectiveMeasure(originalFiles,estimateFile,options);

%%%%%%%%%%%%%%%%%
% Display results
%%%%%%%%%%%%%%%%%

fprintf('************************\n');
fprintf('* INTERMEDIATE RESULTS *\n');
fprintf('************************\n');

fprintf('The decomposition has been generated and stored in:\n');
cellfun(@(s)fprintf(' - %s\n',s),res.decompositionFilenames);

fprintf('The ISR, SIR, SAR and SDR criteria computed with the new decomposition are:\n');
fprintf(' - SDR = %gdB\n - ISR = %gdB\n - SIR = %gdB\n - SAR = %gdB\n',...
    res.SDR,res.ISR,res.SIR,res.SAR);

fprintf('The audio quality (PEMO-Q) criteria computed with the new decomposition are:\n');
fprintf(' - qGlobal = %gdB\n - qTarget = %gdB\n - qInterf = %gdB\n - qArtif = %gdB\n',...
    res.qGlobal,res.qTarget,res.qInterf,res.qArtif);

fprintf('*************************\n');
fprintf('****  FINAL RESULTS  ****\n');
fprintf('*************************\n');
fprintf(' - Overall Perceptual Score: OPS = %g/100\n',res.OPS)
fprintf(' - Target-related Perceptual Score: TPS = %g/100\n',res.TPS)
fprintf(' - Interference-related Perceptual Score: IPS = %g/100\n',res.IPS)
fprintf(' - Artifact-related Perceptual Score: APS = %g/100\n',res.APS);

return
