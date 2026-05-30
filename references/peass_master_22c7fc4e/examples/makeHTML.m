function makeHTML

% if nargin<1
%     TSIADir = 'TSIA/TSIA_1proj/';
%     TSIADir = 'TSIA/TSIA_2proj/';
% end
% if nargin<2
%     origDir =...
%         './Procope_listening_test_results/';
% end
% if nargin<3
mp3Dir = 'mp3/';
% end
% addpath('../toolbox/emiya');

% if 0
%     wav2mp3(TSIADir,[TSIADir mp3Dir]);
% end

files = dir([mp3Dir '*old_trsp.mp3']);

NEx = length(files);
% testNames = {'target','anchorDistTarget','anchorInterf','anchorArtif',...
%     'test5','test6','test7','test8'};
componentNames = {'Distorted target (s<sub>j</sub>+e<sup>Target</sup>)',...
    'Target distortion (e<sub>j</sub><sup>Target</sup>)',...
    'Interference (e<sup>Interf</sup>)',...
    'Artifacts (e<sup>Artif</sup>)'};
suff = {'trsp','eTarget','eInterf','eArtif'};

% files = files([3,5,6,1,2,4,7:NEx]);
% files = files([2,1,4,6,3,5,7:NEx]);
% files = files([5,2,7,1,6,3,4]);

comments = {...
    '';
    '';
    '';
    '';%'In this example, the estimate is the mixture itself. Note that with the new decomposition, <ul> <li> the interference component is very low whereas the state-of-the-art decomposition seems to create an interference component that is not present in &#349;<sub>j</sub></li> <li>the sources are better removed from e<sup>Artif</sup> than with the state-of-the-art method </li> </ul>';
    '';
    'When the estimate is the mixture... Listen to the estimate of the target distortion.';
    ''};
% comments = {...
%     'In this example of singing voice, note that with the new decomposition, <ul> <li> the filtering distortions in the estimate &#349;<sub>j</sub> are present in the distorted target and interference signals whereas these components sounds very clean when using the state-of-the-art decomposition</li> <li>the sources are better removed from e<sup>Artif</sup> than with the state-of-the-art method </li> </ul>';
%     'In this speech example, note that with the new decomposition, <ul> <li> the interference component is very low whereas the state-of-the-art decomposition seems to create an interference component that is not present in &#349;<sub>j</sub></li> <li>the sources are better removed from e<sup>Artif</sup> than with the state-of-the-art method </li> </ul>';
%     '';
%     '';%'In this example, the estimate is the mixture itself. Note that with the new decomposition, <ul> <li> the interference component is very low whereas the state-of-the-art decomposition seems to create an interference component that is not present in &#349;<sub>j</sub></li> <li>the sources are better removed from e<sup>Artif</sup> than with the state-of-the-art method </li> </ul>';
%     '';
%     ''};

for nEx = 1:NEx
    fid = fopen(['Ex' num2str(nEx,'%02d') '.html'],'wt');
    fprintf(fid,'<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">\n');
    fprintf(fid,'<html><head></head>\n');
    fprintf(fid,'<body>\n');
    fprintf(fid,'<h1>Distortion components in audio source separation<br> </h1>\n');
    
    fprintf(fid,'In the source separation process, the estimate &#349;<sub>j </sub>of a target source s<sub>j </sub>is obtained from the mixture x=s<sub>1</sub>+...+s<sub>J </sub>of several sources.<br/> The distortion is modeled as the sum of three distortion components:<br/>');
    fprintf(fid,'<div style="text-align: center;">&#349;<sub>j</sub>-s<sub>j</sub>=e<sup>Target</sup>+e<sup>Interf</sup>+e<sup>Artif</sup><br/> </div>\n');
    fprintf(fid,'<ul>  <li>e<sup>Target</sup> is the target distortion</li>  <li>e<sup>Interf</sup> is the interference distortion component<br>  </li>  <li>e<sup>Artif</sup> is the artifact distortion component<br>  </li> </ul>\n');
    fprintf(fid,'We propose a new decomposition method to estimate e<sup>Target</sup>, e<sup>Interf</sup> and e<sup>Artif</sup>. Here are some sound examples to illustrate the resulting decomposition and compare it to the state-of-the-art decomposition method (E. Vincent, H. Sawada, P. Bofill, S. Makino, and J.P. Rosca, <span style="font-style: italic;">First Stereo Audio Source Separation Evaluation Campaign: Data, Algorithms and Results</span>, Int. Conf. on Independent Component Analysis and Signal Separation, 2007).<br/>\n');
    fprintf(fid,'<h2>Example %d<br> </h2>\n',nEx);
    fprintf(fid,'%s\n',comments{nEx});
    fprintf(fid,'<div style="text-align: center;">\n');
    s = sprintf('%s%s',mp3Dir,files(nEx).name(1:end-9-4));
    
    fprintf(fid,'Clean target (s<sub>j</sub>): ');
    aux_insertData(fid,[s(1:length(mp3Dir)+7) '_target.mp3'])
    fprintf(fid,'<br/>\n');
    
    fprintf(fid,'Mix (x): ');
    aux_insertData(fid,[s(1:length(mp3Dir)+7) '_anchorInterf.mp3'])
    fprintf(fid,'<br/>\n');
    
    fprintf(fid,'Estimate (&#349;<sub>j</sub>): ');
    aux_insertData(fid,[s '.mp3'])
    fprintf(fid,'<br/>\n');
    fprintf(fid,'<table style="margin-left: auto; margin-right: auto; vertical-align: middle; text-align: center;"><tbody>\n');
    fprintf(fid,'<tr style="background-color: rgb(187, 187, 187);">\n');
    fprintf(fid,' <td>Components</td>\n');
    fprintf(fid,' <td>Proposed decomposition</td>\n');
    fprintf(fid,' <td>State-of-the-art decomposition (SASSEC/BSS eval)</td>\n');
    fprintf(fid,'</tr>\n');
    
    for kComp = 1:length(componentNames)
        if mod(kComp,2)==0
            fprintf(fid,'<tr style="background-color: rgb(187, 187, 187);">\n');
        else
            fprintf(fid,'<tr>\n');
        end
        fprintf(fid,' <td>%s</td>\n',componentNames{kComp});
        
        fprintf(fid,'<td>');
        aux_insertData(fid,[s '_' suff{kComp} '.mp3'])
        fprintf(fid,'</td>\n');
        
        fprintf(fid,'<td>');
        aux_insertData(fid,[s '_old' '_' suff{kComp} '.mp3'])
        fprintf(fid,'</td>\n');
        
        fprintf(fid,'</tr>\n');
    end
    %     for kTest = 1:length(testNames)
    %         if mod(kTest,2)==0
    %             fprintf(fid,'<tr style="background-color: rgb(187, 187, 187);">\n');
    %         else
    %             fprintf(fid,'<tr>\n');
    %         end
    %         fprintf(fid,' <td>%s<br>',testNames{kTest});
    %         link = [s testNames{kTest} '.mp3'];
    %         aux_insertData(fid,componentNames{kComp},link)
    %         fprintf(fid,'</td>\n');
    %         for kComp = 1:length(componentNames)
    %             fprintf(fid,' <td>');
    %             link = [s testNames{kTest} '_' componentNames{kComp} '.mp3'];
    %             aux_insertData(fid,componentNames{kComp},link)
    %             fprintf(fid,' </td>\n');
    %         end
    %         fprintf(fid,'</tr>\n');
    %     end
    fprintf(fid,'</tbody></table>\n');
    fprintf(fid,'</div>\n');
    
    fprintf(fid,'<div style="text-align: left;">\nGo to: ');
    for nExOther = 1:NEx
        if nExOther~=nEx
            fprintf(fid,'<a href="Ex%02d.html" target="_self">Example %d</a> \n',nExOther,nExOther);
        end
    end
    fprintf(fid,'</div>\n');
    
    fprintf(fid,'<h2>Scatter plots<br> </h2>\n');
    fprintf(fid,'The proposed decomposition vs state-of-the-art decomposition are compared on a large number of test sounds, using several criteria:\n');
    fprintf(fid,'<ul>  <li>ISR, SIR and SAR;</li> <li> the proposed criteria q<sup>Target</sup>, q<sup>Interf</sup> and q<sup>Artif</sup> (audio quality measures of target preservation, interference suppression and absence of artifacts).</li></ul>\n');

    fprintf(fid,'<div style="text-align: center;">\n');
    fprintf(fid,'<table style="width: 80%%; text-align: center;" border="0" cellpadding="2" cellspacing="2"><tbody><tr>\n');
    fprintf(fid,'<td>\n');
    fprintf(fid,'<img alt="Scatter energy ratios" src="scatterER2-CO.jpg.jpg" width="800px"/>\n');
    fprintf(fid,'</td><td colspan="1" rowspan="2">\n');
    fprintf(fid,'<img alt="Scatter energy ratios" src="scatterLeg2-CO.jpg.jpg" width="250px"/>\n');
    fprintf(fid,'</td></tr><tr>\n');
    fprintf(fid,'<td>\n');
    fprintf(fid,'<img alt="Scatter audio q" src="scatterPQ2-CO.jpg.jpg" width="800px"/>\n');
    fprintf(fid,'</td></tr></tbody></table></div>\n');
    
    fprintf(fid,'<div style="text-align: left;">\nGo to: ');
    for nExOther = 1:NEx
        if nExOther~=nEx
            fprintf(fid,'<a href="Ex%02d.html" target="_self">Example %d</a> \n',nExOther,nExOther);
        end
    end
    fprintf(fid,'</div>\n');

    fprintf(fid,'</body>\n');
    fprintf(fid,'</html>\n');
    fclose(fid);
end
return

% function aux_insertData(fid,title,link)
% fprintf(fid,'<a href="%s">%s</a>\n',link,title);

function aux_insertData(fid,link)
fprintf(fid,...
    '<object type="application/x-shockwave-flash" data="dewplayer-mini.swf?mp3=%s" width="160" height="20"><param name="wmode" value="transparent"><param name="movie" value="../dewplayer-mini.swf?mp3=%s"></object>',...
    link,link);