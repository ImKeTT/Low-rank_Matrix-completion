clc
clear all
warning off
sr=0.5;
for pp=1:1
% make data
m=500;
X=[];
n=500;
r=50;
w=[1:0.1:(r+10)/10].^1;
A0=randn(m,r).*repmat(w(1:r),m,1);
B0=randn(r,n);
X=A0*B0;
[nr,nc]=size(X);
M=ones(nr,nc);
idM=randperm(nr*nc,round(nr*nc*(1-sr)));
M(idM)=0;
%
X0=X;
sdX=std(X0(:));
E0=0.1*sdX*randn(size(X));% 0.1 0.2
X=X+E0;
SNR=norm(X-X0,'fro')/norm(X0,'fro')
X=X.*M;
%% Nuclear norm
tic;[Xr{1}]=MC_Nuclear_IALM(X,M);t(1)=toc;
%% FGSR-1/2
options.d=ceil(1.5*r);
options.regul_B='L2';
options.tol=1e-4;
options.lambda=0.003;
[Xr{2}]=MC_FGSR_PALM(X,M,options);
%% FGSR-p
options=[];
options.p=1/2^3;
options.d=ceil(r*1.5);
options.lambda=0.0003;% 1/(n^(0.5/(options.p/2)))
options.regul_B='L2';
[Xr{3},A,Z]=MC_FGSRp_PALM(X,M,options);
%%
for i=1:length(Xr)
re_error_M(pp,i)=norm((X0-Xr{i}).*(1-M),'fro')/norm(X0.*(1-M),'fro');
end


end


