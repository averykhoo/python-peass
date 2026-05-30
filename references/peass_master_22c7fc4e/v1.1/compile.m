function compile
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Version history
%  - Version 1.1, September 2011: added compilation for haircell and
%  adaptation MEX files
%  - Version 1.0, June 2010: first release
% Copyright 2010-2011 Valentin Emiya and Emmanuel Vincent (INRIA).
% This software is distributed under the terms of the GNU Public License
% version 3 (http://www.gnu.org/licenses/gpl.txt).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Remove syntax errors in adapt.c
fid=fopen('adapt.c','r');
txt=fscanf(fid,'%c');
fclose(fid);
pos=strfind(txt,'// ');
while ~isempty(pos),
    b=pos(1);
    e=strfind(txt(b+1:end),char(10));
    txt=[txt(1:b-1) txt(b+e(1):end)];
    pos=strfind(txt,'// ');
end
fid=fopen('adapt.c','w');
fprintf(fid,'%c',txt);
fclose(fid);

% Compile
mex toeplitzC.c
mex haircell.c
mex adapt.c
cd gammatone
mex Gfb_Analyzer_fprocess.c
cd ..

return