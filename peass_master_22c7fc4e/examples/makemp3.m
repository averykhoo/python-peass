function makemp3(examples)

if nargin<1
    examples = {'exp03_test6','exp02_test6','exp07_test5','exp10_test5',...
        'exp05_anchorInterf','exp08_anchorArtif'};
    %     examples = {'exp08_test8','exp07_test5','exp10_test5',...
    %         'exp07_anchorDistTarget','exp05_anchorInterf','exp08_anchorArtif'};
    examples = {'exp06_test6','exp04_test7','exp08_test6','exp01_test7',...
        'exp08_anchorDistTarget','exp05_anchorInterf','exp06_anchorArtif'};
examples = {'exp01_test7','exp02_test6','exp04_test7','exp08_test6',...
    'exp06_anchorDistTarget','exp05_anchorInterf'};
end

mp3Dir = 'mp3/';
if isempty(dir(mp3Dir))
    mkdir(mp3Dir)
end
if 0
    wavDir = '../2009-12-09_seminaireMetiss/';
    
    
    % wavfiles = dir([wavDir '*.wav']);
    % for kf = 1:length(wavfiles)
    wav2mp3(wavDir,mp3Dir);
    % end
else
    tmpDir = [tempdir 'myWav/'];
    if isempty(dir(tmpDir))
        mkdir(tmpDir);
    end
    
    sourceDir = {[getRootDirData 'TSIA/TSIA_1proj_500_40/'],...
        [getRootDirData 'TSIA/TSIA_old/']};
    sourceDirStr = {'','_old'};
    
    if ispc
        origDir = 'Z:/MATLAB/2009-10-01_ObjMeasuresForSS/Procope_listening_test_results/';
    else
        origDir = '~/MATLAB/2009-10-01_ObjMeasuresForSS/Procope_listening_test_results/';
    end
    
    for k=1:length(examples)
        for kDir = 1:length(sourceDir)
            [T fs] = wavread([sourceDir{kDir} examples{k} '_true.wav']);
            S = wavread([sourceDir{kDir} examples{k} '_eTarget.wav']);
            I = wavread([sourceDir{kDir} examples{k} '_eInterf.wav']);
            A = wavread([sourceDir{kDir} examples{k} '_eArtif.wav']);
            
            wavwrite(T+S,fs,[tmpDir num2str(k) '-' examples{k} sourceDirStr{kDir} '_trsp.wav'])
            wavwrite(S,fs,[tmpDir num2str(k) '-' examples{k} sourceDirStr{kDir} '_eTarget.wav'])
            wavwrite(I,fs,  [tmpDir num2str(k) '-' examples{k} sourceDirStr{kDir} '_eInterf.wav'])
            wavwrite(A,fs,  [tmpDir num2str(k) '-' examples{k} sourceDirStr{kDir} '_eArtif.wav'])
            
            copyfile([origDir examples{k}(1:5) '_target.wav'],...
                [tmpDir num2str(k) '-' examples{k}(1:5) '_target.wav'],'f');
            copyfile([origDir examples{k}(1:5) '_anchorInterf.wav'],...
                [tmpDir num2str(k) '-' examples{k}(1:5) '_anchorInterf.wav'],'f');
            copyfile([origDir examples{k} '.wav'],...
                [tmpDir num2str(k) '-' examples{k} '.wav'],'f');
        end
    end
    wav2mp3(tmpDir,mp3Dir);
    rmdir(tmpDir,'s');
end
return
