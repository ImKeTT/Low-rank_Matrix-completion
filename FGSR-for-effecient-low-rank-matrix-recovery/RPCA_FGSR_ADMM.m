function [Xr,E,output]=RPCA_FGSR_ADMM(X,lambda,options)
% This is the FGSR based RPCA code in the following paper:
% Factor Group-Sparse Regularization for Efficient Low-Rank Matrix
% Recovery. Jicong Fan, Lijun Ding, Yudong Chen, Madeleine Udell. NeurIPS
% 2019.
% Written by Jicong Fan, 09/2019. E-mail: jf577@cornell.edu
[m,n]=size(X);
if isfield(options,'d')==0
    d=round(min(m,n)*0.25);
else
    d=options.d;
end
if isfield(options,'u')==0 % Lagrange parameter
    u=lambda;
else
    u=options.u;
end
if isfield(options,'maxiter')==0
    maxiter=1000;
else
    maxiter=options.maxiter;
end
if isfield(options,'tol')==0
    tol=1e-4;
else
    tol=options.tol;
end
if isfield(options,'regul_B')==0
    regul_B='L2';
else
    regul_B=options.regul_B;
end
if isfield(options,'alpha')==0
    alpha=1;
else
    alpha=options.alpha;
end
% spectral initialization
[U,S,V]=svd(X,'econ');% or parial SVD
switch regul_B
    case 'L2'
        A=alpha^(1/3)*U(:,1:d)*(S(1:d,1:d).^(2/3));
        B=alpha^(-1/3)*(S(1:d,1:d).^(1/3))*V(:,1:d)';
    case 'L21'
        A=U(:,1:d)*(S(1:d,1:d).^(1/2));
        B=(S(1:d,1:d).^(1/2))*V(:,1:d)';
    case 'L1'
        A=U(:,1:d)*S(1:d,1:d);
        B=V(:,1:d)';
end
E=zeros(m,n);
Q=zeros(m,n);
iter=0;
while iter<maxiter
    iter=iter+1;
    % B_new
    switch regul_B
        case 'L2'
            B_new=inv(u*A'*A+alpha*eye(d))*(u*A'*(X-E+Q/u));
        case 'L21'
            tau=1.01*u*normest(A)^2;
            temp=B-u*(-A')*((X-A*B-E+Q/u))/tau;
            B_new=solve_l1l2(temp',alpha/tau)';
        case 'L1'
            tau=1.01*u*normest(A)^2;
            temp=B-u*(-A')*((X-A*B-E+Q/u))/tau; 
            B_new=max(0,temp-alpha/tau)+min(0,temp+alpha/tau);
    end
    % A_new
    tau=1.01*u*normest(B_new)^2;
    temp=A-u*((X-A*B_new-E+Q/u)*(-B_new'))/tau;
    A_new=solve_l1l2(temp,1/tau);
    % E_new
    AB=A_new*B_new;
    temp=X-AB+Q/u;
    E_new = sign(temp) .* max(abs(temp) - lambda/u, 0);
    %
    Q=Q+u*(X-AB-E_new);
    % objective function
    switch regul_B
        case 'L2'
            J(iter)=sum(sum(A_new.^2).^0.5)+alpha*sum(B_new(:).^2)+sum(Q(:).*(X(:)-AB(:)-E_new(:)))+u*sum((X(:)-AB(:)-E_new(:)).^2);
        case 'L21'
            J(iter)=sum(sum(A_new.^2).^0.5)+alpha*sum(sum(B_new'.^2).^0.5)+sum(Q(:).*(X(:)-AB(:)-E_new(:)))+u*sum((X(:)-AB(:)-E_new(:)).^2);
        case 'L1'
            J(iter)=sum(sum(A_new.^2).^0.5)+alpha*sum(abs(B_new(:)))+sum(Q(:).*(X(:)-AB(:)-E_new(:)))+u*sum((X(:)-AB(:)-E_new(:)).^2);
    end
    %
    et=[norm(B_new-B,'fro')/norm(B,'fro') norm(A_new-A,'fro')/norm(A,'fro')...
        norm(E_new-E,'fro')/norm(E,'fro')];
    stopC=max(et);
    %
    isstopC=stopC<tol||et(3)<tol/10;
    if mod(iter,100)==0||isstopC||iter<=10
        disp(['iteration=' num2str(iter) '/' num2str(maxiter) ': J=' num2str(J(iter)) ', d=' num2str(d)...
            ', stopC=' num2str(stopC) ...
            ', e_E=' num2str(et(3))])
    end
    if isstopC
        disp('converged')
        break;
    end
    A=A_new;
    B=B_new;
    E=E_new;
    % extract nonzero columns
    id=find(sum(A.^2)>1e-5);
    A=A(:,id);
    B=B(id,:);
    d=length(id);
end
%   
Xr=X-E;
output.A=A;
output.B=B; 
output.J=J;
output.d=d;
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
 