clc
clear all
warning off
m=500;
n=500;
r=50;
noise_mag=1.0;
noise_density=0.4;
for p=1:1
    w=[1:0.1:(r+10)/10].^1;
    X=randn(m,r)*diag(w(1:r))*randn(r,n);
    sdX=std(X(:));
    E=noise_mag*sdX*randn(size(X));
    NSR=norm(X,'fro')/norm(E,'fro');
    for i=1:n
    id0=randperm(m,ceil(m*(1-noise_density)));
    E(id0,i)=0;
    end
    Xn=X+E;
    k=0;
    %% RPCA IALM or ADMM
    k=k+1;
    par=1/(n^(0.5/(1)));
    %par=[0.01:0.01:0.1];
    for i=1:length(par)
        %tic;[X_rpca{i},E_t]=inexact_alm_rpca(Xn,par(i),1e-8,1000);t_temp_n(i)=toc;
        tic;[X_rpca{i},E_r] = RobustPCA(Xn, par(i), par(i)*0.1, 1e-5, 500);t_temp_n(i)=toc;
        e_temp(i)=norm(X-X_rpca{i},'fro')/norm(X,'fro');
    end
    [~,idx]=min(e_temp);
    Xr{k}=X_rpca{idx};
    t(k)=t_temp_n(idx);
    %% Factor nuclear norm
    k=k+1;
    par=1/(n^(0.5/(1)));
    %par=[0.01:0.01:0.1];
    opt.d=ceil(1.5*r);
    for i=1:length(par)
        opt.u=par(i)*0.1;
        [X_rpca_fn{i},E,output]=RPCA_FNuclear_ADMM(Xn,par(i),opt);
        e_temp_fn(i)=norm(X-X_rpca_fn{i},'fro')/norm(X,'fro');
    end
    [~,idx_fn]=min(e_temp_fn);
    Xr{k}=X_rpca_fn{idx_fn};
    %% FGSR-2/3
    k=k+1;
    par=1/(n^(0.5/(2/3)));
    %par=[0.002:0.001:0.01];
    opt.regul_B='L2';
    opt.d=ceil(1.5*r);
    for i=1:length(par)
        opt.u=par(i)*0.1;
        [X_rpca_fgsr_23{i},E,output]=RPCA_FGSR_ADMM(Xn,par(i),opt);
        e_temp_fgsr_23(i)=norm(X-X_rpca_fgsr_23{i},'fro')/norm(X,'fro');
    end
    [~,idx_sp]=min(e_temp_fgsr_23);
    Xr{k}=X_rpca_fgsr_23{idx_sp};
    %% FGSR-1/2
    k=k+1;
    par=1/(n^(0.5/(1/2)));
    %par=[0.001:0.001:0.01];
    opt.regul_B='L21';
    opt.d=ceil(1.5*r);
    for i=1:length(par)
        opt.u=par(i)*0.1;
        [X_rpca_fgsr_12{i},E,output]=RPCA_FGSR_ADMM(Xn,par(i),opt);
        e_temp_fgsr_12(i)=norm(X-X_rpca_fgsr_12{i},'fro')/norm(X,'fro');
    end
    [~,idx_sp]=min(e_temp_fgsr_12);
    Xr{k}=X_rpca_fgsr_12{idx_sp};
    %% evaluation metric
    for k=1:length(Xr)
        re_error(p,k)=norm(X-Xr{k},'fro')/norm(X,'fro');
    end
end
re_mean=mean(re_error)

