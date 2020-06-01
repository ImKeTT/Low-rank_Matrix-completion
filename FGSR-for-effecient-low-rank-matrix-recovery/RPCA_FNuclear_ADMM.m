function [Xr,E,output]=RPCA_FNuclear_ADMM(X,lambda,options)
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
% spectral initialization
[U,S,V]=svd(X,'econ');% or parial SVD
A=U(:,1:d)*(S(1:d,1:d).^(1/2));
B=(S(1:d,1:d).^(1/2))*V(:,1:d)';
E=zeros(m,n);
Q=zeros(m,n);
iter=0;
while iter<maxiter
    iter=iter+1;
    % B_new
    B_new=inv(u*A'*A+eye(d))*(u*A'*(X-E+Q/u));
    % A_new
    A_new=(u*(X-E+Q/u)*B_new')*inv(u*B_new*B_new'+eye(d));
    % E_new
    AB=A_new*B_new;
    temp=X-AB+Q/u;
    E_new = sign(temp) .* max(abs(temp) - lambda/u, 0);
    %
    Q=Q+u*(X-AB-E_new);
    % objective function
    J(iter)=sum(A_new(:).^2)+sum(B_new(:).^2)+sum(Q(:).*(X(:)-AB(:)-E_new(:)))+u*sum((X(:)-AB(:)-E_new(:)).^2);
    %
    et=[norm(B_new-B,'fro')/norm(B,'fro') norm(A_new-A,'fro')/norm(A,'fro')...
        norm(E_new-E,'fro')/norm(E,'fro')];
    stopC=max(et);
    %
    isstopC=stopC<tol;
    if mod(iter,100)==0||isstopC||iter<=10
        disp(['iteration=' num2str(iter) '/' num2str(maxiter) ': J=' num2str(J(iter)) ', stopC=' num2str(stopC) ...
            ', e_E=' num2str(et(3))])
    end
    if isstopC
        disp('converged')
        break;
    end
    A=A_new;
    B=B_new;
    E=E_new;
end
%   
Xr=X-E;
output.A=A;
output.B=B; 
output.J=J;
end
 