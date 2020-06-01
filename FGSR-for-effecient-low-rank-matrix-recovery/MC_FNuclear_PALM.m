function [Xr,A,B,J]=MC_FNuclear_PALM(X,M,options)
[m,n]=size(X);
%
if isfield(options,'d')==0
    d=ceil(mean(M(:))*min(m,n)/2);
    disp(['The estimated initial rank is ' num2str(d)])
else
    d=options.d;
end
if isfield(options,'lambda')==0
    lambda=0.01;
else
    lambda=options.lambda;
end
if isfield(options,'maxiter')==0
    maxiter=500;
else
    maxiter=options.maxiter;
end
if min(m,n)>5000
    disp('Random initialization ... ')
    A=randn(m,d);
    B=randn(d,n);
else
    disp('SVD initialization ... ')
    [U,S,V]=svd(X,'econ');%m*n/sum(M(:))
    A=U(:,1:d)*(S(1:d,1:d).^(1/2));
    B=(S(1:d,1:d).^(1/2))*V(:,1:d)';
end
%
if isfield(options,'tol')==0
    tol=1e-4;
else
    tol=options.tol;
end
%
J=[];
iter=0;
rho=0.5;
while iter<maxiter
    iter=iter+1;
    % A_new
    tau=lambda*rho*norm(B*B');
    A_new=(lambda*(M.*(X-A*B))*B'+tau*A)/(1+tau); 
    % B_new
    tau=lambda*rho*norm(A_new'*A_new);
    AB=A_new*B;
    B_new=(lambda*A_new'*(M.*(X-AB))+tau*B)/(1+tau);
    %
    et=[norm(B_new-B,'fro')/norm(B,'fro') norm(A_new-A,'fro')/norm(A,'fro')];
    stopC=max(et);
    %
    isstopC=stopC<tol;
    if mod(iter,100)==0||isstopC||iter<=10
        disp(['iteration=' num2str(iter) '/' num2str(maxiter) ':  stopC=' num2str(stopC)])
    end
    if isstopC
        disp('converged')
        break;
    end
    B=B_new;
    A=A_new;
end
Xr=(A_new*B_new).*(1-M)+X.*M;
end
 