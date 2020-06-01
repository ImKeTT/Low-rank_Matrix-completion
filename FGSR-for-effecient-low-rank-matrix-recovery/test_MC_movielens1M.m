clc
clear all
%%
for j=1:10
load('movielens1M.mat');%%% 6000 users on 4000 movies
M_org=double(X~=0);
maxdim=sum(M_org(:))/prod(size(M_org))*min(size(M_org))
sm=sum(M_org');
idsm=find(sm>=5);
M_org=M_org(idsm,:);
X=X(idsm,:);
X(M_org==0)=mean(X(M_org==1));
%
[nr,nc]=size(X);
missrate=0.5;
M_t=ones(nr,nc);
% random mask
for i=1:nc
    idx=find(M_org(:,i)==1);
    lidx=length(idx);
    temp=randperm(lidx,ceil(lidx*missrate));
    temp=idx(temp);
    M_t(temp,i)=0;
end
%
M=M_t.*M_org;
Xm=X.*M;
vd=[20];
for pp=1:length(vd)
    d=vd(pp);% initial rank
%% F-nuclear norm
options.d=d;
options.maxiter=1000;
options.lambda=0.1;% 0.1 0.5
[Xr{1}]=MC_FNuclear_PALM(Xm,M,options);
%% max norm
[U,S,V]=svd(X,'econ');A1=U(:,1:d)*S(1:d,1:d)^0.5;B1=S(1:d,1:d)^0.5*V(:,1:d)';
ballsize=2.5;%max([sum(A1.^2,2).^0.5;sum(B1'.^2,2).^0.5]);%2.5
stepsize=0.0001;
[Xr{2}]=MC_MAX_pgm(Xm,M,d,ballsize,stepsize,1000);
%% FGSR 1/3
options.d=d;
options.regul_B='L2';options.maxiter=1000;
options.lambda=0.015;% 0.015
[Xr{3}]=MC_FGSR_PALM(Xm,M,options); 
%% FGSR 1/2
options.d=d;
options.regul_B='L21';options.maxiter=1000;
options.lambda=0.007;% 0.007
[Xr{4}]=MC_FGSR_PALM(Xm,M,options);
%%
for i=1:length(Xr)
    if ~isempty(Xr{i})
    MM=(~M).*M_org;
    E=(Xr{i}-X).*MM;
    NMAE(pp,i,j)=sum(abs(E(:)))/sum(MM(:))/4;
    RMSE(pp,i,j)=norm(E,'fro')/norm(X.*MM,'fro');
    %RMSE_A(pp,i)=(sum(abs(E(:).^2))/sum(MM(:)))^0.5;
    end
end
%
end
end




