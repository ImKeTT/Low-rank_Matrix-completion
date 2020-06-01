function [X,A,B]=MC_FGSRp_PALM(X0,M,options)
% This is the FGSR (anly p<1) based LRMC (noisy) code in the following paper:
% Factor Group-Sparse Regularization for Efficient Low-Rank Matrix
% Recovery. Jicong Fan, Lijun Ding, Yudong Chen, Madeleine Udell. NeurIPS
% 2019.
% PALM+iterative reweighted minimization
% Written by Jicong Fan, 09/2019. E-mail: jf577@cornell.edu
[m,n]=size(X0);
X=X0;
%
p=options.p;
if isfield(options,'d')==0
    d=ceil(mean(M(:))*min(m,n)/2);
    disp(['The estimated initial rank is ' num2str(d)])
else
    d=options.d;
end
if isfield(options,'alpha')==0
    alpha=1;
else
    alpha=options.alpha;
end
if isfield(options,'lambda')==0
    lambda=0.01;
else
    lambda=options.lambda;
end
if isfield(options,'regul_B')==0
    regul_B='L2';
else
    regul_B=options.regul_B;
end
if isfield(options,'maxiter')==0
    maxiter=1000;
else
    maxiter=options.maxiter;
end
disp(['alpha=' num2str(alpha)])
disp(['regul_B=' regul_B])
if min(m,n)>5000
    disp('Random initialization ... ')
    A=randn(m,d);
    B=randn(d,n);
else
    disp('SVD initialization ... ')
    [U,S,V]=svd(X,'econ');%m*n/sum(M(:))
    switch regul_B
        case 'L2'
            A=alpha^(1/3)*U(:,1:d)*(S(1:d,1:d).^(2/3));
            B=alpha^(-1/3)*(S(1:d,1:d).^(1/3))*V(:,1:d)';
        case 'L21'
            A=U(:,1:d)*(S(1:d,1:d).^(1/2));
            B=(S(1:d,1:d).^(1/2))*V(:,1:d)';
    end
end


if isfield(options,'tol')==0
    tol=1e-4;
else
    tol=options.tol;
end
if isfield(options,'zeta')==0
    zeta=1e-2;
else
    zeta=options.zeta;
end
%
iter=0;
while iter<maxiter
    iter=iter+1;
    W=diag((sum(A.^2).^0.5+zeta).^(p-2));
    A_new=lambda*X*B'*inv(lambda*B*B'+W);
    % B_new
    switch regul_B
        case 'L2'
            B_new=inv(lambda*A_new'*A_new+alpha*eye(d))*(lambda*A_new'*(X));
%             % 1/2
        case 'L21'
            tau=1.01*lambda*normest(A_new)^2;
            temp=B-lambda*(-A_new')*((X-A_new*B))/tau;
            B_new=solve_l1l2(temp',alpha/tau)';
    end
%     X_new
    AB=A_new*B_new;
    X_new=AB;
    X_new=X_new.*(1-M)+X0.*M;

    et=[norm(B_new-B,'fro')/norm(B,'fro') norm(A_new-A,'fro')/norm(A,'fro') norm(X_new-X,'fro')/norm(X,'fro')];
    stopC=max(et);
    %
    isstopC=et(3)<tol/10;
    if mod(iter,100)==0||isstopC||iter<=5
        disp(['iteration=' num2str(iter) '/' num2str(maxiter) '  stopC=' num2str(stopC) ...
            '  e_X=' num2str(et(3)) ' d=' num2str(d) ])
    end
    if isstopC
        disp('converged')
        break;
    end
    B=B_new;
    A=A_new;
    X=X_new;
    % extract nnz
    id1=find(sum(A.^2)>1e-5);
    id2=find(sum((B').^2)>1e-5);
    id=intersect(id1,id2);
    A=A(:,id);
    B=B(id,:);
    d=length(id);

end
end
 