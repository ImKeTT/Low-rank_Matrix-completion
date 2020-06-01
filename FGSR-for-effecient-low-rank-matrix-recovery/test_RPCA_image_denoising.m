clc
clear all
X=imread('im1.png');
% X=imread('im2.jpg');
% X=imresize(X,0.5);% im1 0.5
X=double(X)/255;
% [U,S,V]=svd(X);
rt=50;
for i=1:3
    [U,S,V]=svd(X(:,:,i));
    Xt(:,:,i)=U(:,1:rt)*S(1:rt,1:rt)*V(:,1:rt)';
end
X=Xt;
imshow(X)
Xn=imnoise(X,'salt & pepper',0.4); 
imshow(Xn)
%%
r=2*rt;
%% Nuclear norm
par=[0.02 0.03 0.04];
for i=1:length(par)
    for j=1:3
%     tic;[Xr{1,i}(:,:,j)]=inexact_alm_rpca(Xn(:,:,j),par(i),1e-8,1000);t_temp_n(i)=toc;
    tic;[Xr{1,i}(:,:,j),E_r] = RobustPCA(Xn(:,:,j), par(i), par(i)*10, 1e-5, 500);t_temp_n(i)=toc;
    end
    e{1,i}=norm(Xr{1,i}(:)-X(:))/norm(X(:));
end
%% F-Nuclear norm
par=[0.02 0.03 0.04];
opt.d=r;
opt.u=1;
for i=1:length(par)
    for j=1:3
        [Xr{2,i}(:,:,j)]=RPCA_FNuclear_ADMM(Xn(:,:,j),par(i),opt);
    end
    e{2,i}=norm(Xr{2,i}(:)-X(:))/norm(X(:));
end
%% FGSR 2/3
par=[0.02 0.03 0.04];
opt.regul_B='L2';
opt.d=r;
opt.u=1;opt.maxiter=1500;
for i=1:length(par)
    for j=1:3
        [Xr{3,i}(:,:,j),E{3,i},output]=RPCA_FGSR_ADMM(Xn(:,:,j),par(i),opt);
    end
    e{3,i}=norm(Xr{3,i}(:)-X(:))/norm(X(:));
end
%%%% FGSR 1/2
% par=[0.01 0.02 0.03];
% opt.regul_B='L21';
% opt.d=r;
% opt.u=0.1;opt.maxiter=1500;
% for i=1:length(par)
%     for j=1:3
%         [Xr{4,i}(:,:,j),E{4,i},output]=RPCA_FGSR_ADMM(Xn(:,:,j),par(i),opt);
%     end
%     e{4,i}=norm(Xr{4,i}(:)-X(:))/norm(X(:));
% end

%%
figure;
NC=5;
dp=[-0.05 0 0.035 0];
h=subplot(2,NC,1);imshow(X);ht=title({'Clean image';'ground truth'});set(ht,'FontWeight','light')
pos=get(h,'position');pos=pos+dp;set(h,'position',pos);

h=subplot(2,NC,2);imshow(Xn);ht=title({'Corrupted image';'40% salt&pepper noise'}');set(ht,'FontWeight','light')
pos=get(h,'position');pos=pos+dp;set(h,'position',pos);

h=subplot(2,NC,3);imshow(Xr{1});ht=title({'RPCA(nuclear norm)';'relative recovery error=0.033'});set(ht,'FontWeight','light')
pos=get(h,'position');pos=pos+dp;set(h,'position',pos);

h=subplot(2,NC,4);imshow(Xr{2});ht=title({'RPCA(F-nuclear norm)';'relative recovery error=0.022'});set(ht,'FontWeight','light')
pos=get(h,'position');pos=pos+dp;set(h,'position',pos);

h=subplot(2,NC,5);imshow(Xr{3});title({'RPCA(FGSR-2/3)';'relative recovery error=0.005'});
pos=get(h,'position');pos=pos+dp;set(h,'position',pos);



