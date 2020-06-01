function [Xr,A,B,J]=MC_FGSR_PALM(X,M,options)
% This is the FGSR based LRMC (noisy) code in the following paper:
% Factor Group-Sparse Regularization for Efficient Low-Rank Matrix
% Recovery. Jicong Fan, Lijun Ding, Yudong Chen, Madeleine Udell. NeurIPS
% 2019.
% Written by Jicong Fan, 09/2019. E-mail: jf577@cornell.edu
[m,n]=size(X);
%
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
    switch regul_B
        case 'L2'
            A=alpha^(1/3)*U(:,1:d)*(S(1:d,1:d).^(2/3));
            B=alpha^(-1/3)*(S(1:d,1:d).^(1/3))*V(:,1:d)';
        case 'L21'
            A=U(:,1:d)*(S(1:d,1:d).^(1/2));
            B=(S(1:d,1:d).^(1/2))*V(:,1:d)';
    end
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
    temp=A-lambda*((M.*(X-A*B))*(-B'))/tau;
    A_new=solve_l1l2(temp,1/tau);   
    % B_new
    tau=lambda*rho*norm(A_new'*A_new);
    AB=A_new*B;
    switch regul_B
        case 'L2'
           B_new=(lambda*A_new'*(M.*(X-AB))+tau*B)/(alpha+tau);
           vB=B-B_new;
           %J(iter)=sum(sum(A_new.^2).^0.5)+0.5*alpha*sum(B_new(:).^2)+lambda*sum((M(:).*(X(:)-AB(:))).^2);
        case 'L21'
           temp=B-lambda*((-A_new')*(M.*(X-AB)))/tau;
           B_new=solve_l1l2(temp',alpha/tau)';
           %J(iter)=sum(sum(A_new.^2).^0.5)+sum(sum(B_new'.^2).^0.5)+lambda*sum((M(:).*(X(:)-AB(:))).^2);
    end
    J(iter)=0;
    %
    et=[norm(B_new-B,'fro')/norm(B,'fro') norm(A_new-A,'fro')/norm(A,'fro')];
    stopC=max(et);
    %
    isstopC=stopC<tol;
    if mod(iter,100)==0||isstopC||iter<=10
        disp(['iteration=' num2str(iter) '/' num2str(maxiter) ': J=' num2str(J(iter))...
            ', stopC=' num2str(stopC) ', d=' num2str(d)])
    end
    if isstopC
        disp('converged')
        break;
    end
    B=B_new;
    A=A_new;
    % extract nnz
    id1=find(sum(A.^2)>1e-5);
    id2=find(sum((B').^2)>1e-5);
    id=intersect(id1,id2);
    A=A(:,id);
    B=B(id,:);
    d=length(id);
end
Xr=(A_new*B_new).*(1-M)+X.*M;
end
%%
function [E] = solve_l1l2(W,lambda)
n = size(W,2);
E = W;
for i=1:n
    E(:,i) = solve_l2(W(:,i),lambda);
end
end

function [x] = solve_l2(w,lambda)
% min lambda |x|_2 + |x-w|_2^2
nw = norm(w);
if nw>lambda
    x = (nw-lambda)*w/nw;
else
    x = zeros(length(w),1);
end
end
 