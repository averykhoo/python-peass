function [qTarget, qInterf, qArtif, qGlobal, qTargetPSMt, qInterfPSMt, qArtifPSMt, qGlobalPSMt] = ...
    audioQualityFeatures(decompositionFilenames,options)
% Computes the qTarget, qInterf, qArtif, qGlobal features from the output of the new decomposition
% method:
%  - qTarget is related to the saillance of the distortion of the target
%  source in the source estimate
%  - qInterf is related to the saillance of the interference distortion
%  component in the source estimate
%  - qArtif is related to the saillance of the artifact distortion
%  component in the source estimate
%  - qGlobal is related to audio quality of the whole estimate compared to
%  the reference
%
% decompositionFilenames is a cell array with the file names of (the order
% below must be the same):
% - the true source
% - the the target distortion
% - the interference distortion component
% - the artifact and noise component
%
% options.pemoQExe is the path and program name of PEMO-Q (default:
% 'C:/Program Files/PEMO-Q v1.2/audioqual.exe' if exists,
% './PEMO-Q v1.1.2 demo/audioqual_demo.exe' otherwise)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Version 1.0
% Copyright 2010 Valentin Emiya (INRIA).
% This software is distributed under the terms of the GNU Public License
% version 3 (http://www.gnu.org/licenses/gpl.txt).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% default options
defaultOptions.pemoQExe = 'C:/Program Files/PEMO-Q v1.2/audioqual.exe';
if isempty(dir(defaultOptions.pemoQExe))
    defaultOptions.pemoQExe = './PEMO-Q v1.1.2 demo/audioqual_demo.exe';
end

if nargin<2
    options = defaultOptions;
else
    names = fieldnames(defaultOptions);
    for k=1:length(names)
        if ~isfield(options,names{k}) || isempty(options.(names{k}))
            options.(names{k}) = defaultOptions.(names{k});
        end
    end
end
options.pemoQExe = ['"' options.pemoQExe '"'];
if isunix
    options.pemoQExe = ['wine ' options.pemoQExe];
end

% load signals
[sTrue, fs] = wavread(decompositionFilenames{1});
eTarget = wavread(decompositionFilenames{2});
eInterf = wavread(decompositionFilenames{3});
eArtif = wavread(decompositionFilenames{4});

% store signals to compare in temporary files
tmpFilename = tempname;
testAll = [tmpFilename '_testAll.wav'];
refT = [tmpFilename '_target.wav'];
refI = [tmpFilename '_interf.wav'];
refA = [tmpFilename '_artif.wav'];
refG = [tmpFilename '_global.wav'];
wavwrite(sTrue+eTarget+eInterf+eArtif,fs,testAll);
wavwrite(sTrue+eInterf+eArtif,fs,refT);
wavwrite(sTrue+eTarget+eArtif,fs,refI);
wavwrite(sTrue+eTarget+eInterf,fs,refA);
wavwrite(sTrue,fs,refG);

% Apply audio quality measure
[qTarget qTargetPSMt] = aux_pemoq(refT,testAll,options);
[qInterf qInterfPSMt] = aux_pemoq(refI,testAll,options);
[qArtif qArtifPSMt] = aux_pemoq(refA,testAll,options);
[qGlobal qGlobalPSMt] = aux_pemoq(refG,testAll,options);

% delete temporary files
delete(testAll);
delete(refT);
delete(refI);
delete(refA);
delete(refG);

return

function [pemoPSM, pemoPSMt] = aux_pemoq(refFile,testFile,options)
if ~isempty(findstr(options.pemoQExe, 'demo'))
    fprintf('To unlock PEMO-Q demo, please enter the PIN shown in the new window\n');
end
[status pemo] = system(sprintf('%s %s %s [] [] 0 0 0', options.pemoQExe, refFile, testFile));
pemo = regexp(pemo, 'PSM.? = \d*.\d*', 'match');
pemoPSM = str2double(cell2mat(regexp(pemo{1},'\d+.?\d*', 'match')));
pemoPSMt = str2double(cell2mat(regexp(pemo{2},'\d+.?\d*', 'match')));

return
