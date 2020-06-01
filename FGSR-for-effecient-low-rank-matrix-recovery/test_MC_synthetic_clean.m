clc
clear all
warning off
sr=0.3;% sampling rate
for pp=1:1
% make data
m=500;
n=500;
r=50;
d=round(1.5*r);% initial rank
w=[1:0.1:(r+10)/10].^1;
A0=randn(m,r).*repmat(w(1:r),m,1);
B0=randn(r,n);
X=A0*B0;
[nr,nc]=size(X);
M=ones(nr,nc);
idM=randperm(nr*nc,round(nr*nc*(1-sr)));
M(idM)=0;
X0=X;
X=X.*M;
%% Nuclear norm
[Xr{1}]=MC_Nuclear_IALM(X,M);
%% Max Norm
stepsize=0.0007;
ballsize=max([sum(A0.^2,2).^0.5;sum(B0'.^2,2).^0.5]);
[Xr{2}]=MC_MAX_pgm(X,M,d,ballsize,stepsize,1000);
%% MF
[Xr{3}]=MC_FNuclear_ADMM(X,M,d,1,0.1,1000);
%% FGSR-2/3
options.d=d;
options.u=0.001;
options.regul_B='L2';
[Xr{4}]=MC_FGSR_ADMM(X,M,options);
%% FGSR-1/2
options.d=d;
options.u=0.0001;
options.regul_B='L21';
[Xr{5}]=MC_FGSR_ADMM(X,M,options);
%%
for i=1:length(Xr)
re_error_M(pp,i)=norm((X0-Xr{i}).*(1-M),'fro')/norm(X0.*(1-M),'fro');
end


end



