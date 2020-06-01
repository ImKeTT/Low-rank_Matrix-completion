clc
clear all
warning off
sr=0.5;% sampling rate
for pp=1:1
% make data
m=500;
X=[];
n=500;
r=50;
%
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
%% Max Norm
d=ceil(1.5*r);
stepsize=0.0007;
[U,S,V]=svd(X0);A1=U*S^0.5;B1=S^0.5*V';
ballsize=max([sum(A1.^2,2).^0.5;sum(B1'.^2,2).^0.5]);
[Xr{1},A,Z]=MC_MAX_pgm(X,M,d,ballsize*0.9,stepsize,1000);
%% F-nuclear norm
options.d=ceil(1.5*r);
options.maxiter=1000;
options.lambda=0.1;% 0.1 0.5
[Xr{2}]=MC_FNuclear_PALM(X,M,options);
%% FGSR-2/3
options.d=ceil(1.5*r);
options.regul_B='L2';
options.tol=1e-4;
options.lambda=0.003;
[Xr{3}]=MC_FGSR_PALM(X,M,options);
%% FGSR-1/2
options.d=ceil(1.5*r);
options.regul_B='L21';
options.tol=1e-4;
options.lambda=0.001;
[Xr{4}]=MC_FGSR_PALM(X,M,options);
%%
for i=1:length(Xr)
re_error_M(pp,i)=norm((X0-Xr{i}).*(1-M),'fro')/norm(X0.*(1-M),'fro');
end


end


