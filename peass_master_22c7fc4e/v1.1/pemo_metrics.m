function [PSM, PSMt, ODG] = pemo_metrics(mref,mtest,fs)

% PEMO_METRICS Example implementation of the PEMO-Q objective quality
% assessment metrics
% 
% Disclaimer: this implementation differs from the commercial version at
% http://www.hoertech.de/web_en/produkte/pemo-q.shtml
% When accuracy is a crucial concern, the use of the commercial version is
% recommended instead.
%
% [PSM, PSMt, ODG] = pemo_metrics(mref,mtest,fs)
%
% Inputs:
% mref: nband x nsampl x nmod internal representation of the reference
% signal
% mtest: nband x nsampl x nmod internal representation of the test signal
% fs: sampling frequency of the internal representation
%
% Outputs:
% PSM: PSM metric
% PSMt: PSMt metric
% ODG: PSMt metric expressed as a subjective difference grade
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright 2011 Emmanuel Vincent (INRIA).
% This software is distributed under the terms of the GNU Public License
% version 3 (http://www.gnu.org/licenses/gpl.txt).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Errors and warnings %%%
if nargin<2, error('Not enough input arguments.'); end
[nband,nsampl,nmod]=size(mref);
[nband2,nsampl2,nmod2]=size(mtest);
if (nband2~=nband) , error('The reference and the test representations must have the same number of frequency subbands.'); end
if (nsampl2~=nsampl), error('The reference and the test representations must have the same duration.'); end
if (nmod2~=nmod), error('The reference and the test representations must have the same number of modulation subbands.'); end

%%% Assimilation and cross-correlation %%%
% Assimilation
assim=(abs(mtest)<abs(mref));
mtest(assim)=.5*(mref(assim)+mtest(assim));
% PSM
PSM=zeros(nmod,1);
NMS=zeros(nmod,1);
for m=1:nmod,
    lref=mref(:,:,m);
    lref=lref(:)-mean(lref(:));
    ltest=mtest(:,:,m);
    NMS(m)=sum(ltest(:).*ltest(:));
    ltest=ltest(:)-mean(ltest(:));
    PSM(m)=sum(lref.*ltest)./sqrt(sum(lref.*lref)*sum(ltest.*ltest));
end
PSM=sum(PSM.*NMS)/sum(NMS);
% Frame-based PSM
flen=.01*fs;
nfram=floor(nsampl/flen);
nsampl=nfram*flen;
mref=mref(:,1:nsampl,:);
mtest=mtest(:,1:nsampl,:);
PSMt=zeros(nfram,1);
lPSM=zeros(nmod,1);
lNMS=zeros(nmod,1);
for t=1:nfram,
    for m=1:nmod,
        lref=mref(:,(t-1)*flen+1:t*flen,m);
        lref=lref(:)-mean(lref(:));
        ltest=mtest(:,(t-1)*flen+1:t*flen,m);
        lNMS(m)=sum(ltest(:).*ltest(:));
        ltest=ltest(:)-mean(ltest(:));
        lPSM(m)=sum(lref.*ltest)./sqrt(sum(lref.*lref)*sum(ltest.*ltest));
    end
    PSMt(t)=sum(lPSM.*lNMS)/sum(lNMS);
end
% Lowpass-filtered moving RMS value
RMS=zeros(nfram,1);
for t=1:nfram,
    ltest=mtest(:,max(1,(t-9)*flen+1):t*flen,:);
    RMS(t)=sqrt(sum(ltest(:).*ltest(:))/size(ltest,2));
end
RMS=filter(ones(10,1),1,RMS);
% Weighted percentile
[PSMt,ind]=sort(PSMt);
RMS=RMS(ind);
RMS=cumsum(RMS);
ind=find(RMS>=0.05*RMS(end));
PSMt=PSMt(ind(1));
% SDG mapping
ODG=double(PSMt<0.864)*max(-4,-0.22/(PSMt-0.98)-4.13)+double(PSMt>=0.864)*16.4*(PSMt-1);

return;