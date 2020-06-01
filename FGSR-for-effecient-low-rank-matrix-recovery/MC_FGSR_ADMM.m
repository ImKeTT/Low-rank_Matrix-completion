function [X,A,B]=MC_FGSR_ADMM(X0,M,options)
% This is the FGSR based LRMC (noiseless) code in the following paper:
% Factor Group-Sparse Regularization for Efficient Low-Rank Matrix
% Recovery. Jicong Fan, Lijun Ding, Yudong Chen, Madeleine Udell. NeurIPS
% 2019.
% Written by Jicong Fan, 09/2019. E-mail: jf577@cornell.edu
[m,n]=size(X0);
X=X0;
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
if isfield(options,'u')==0
    u=0.001;
else
    u=options.u;
end
if isfield(options,'regul_B')==0
    regul_B='L2';
else
    regul_B=options.regul_B;
end
if isfield(options,'maxIter')==0
    maxIter=1000;
else
    maxIter=options.maxIter;
end
disp(['alpha=' num2str(alpha)])
disp(['regul_B=' regul_B])
if min(m,n)<5000
    [U,S,V]=svd(X0*m*n/sum(M(:)),'econ');
    switch regul_B
        case 'L2'
            A=alpha^(1/3)*U(:,1:d)*(S(1:d,1:d).^(2/3));
            B=alpha^(-1/3)*(S(1:d,1:d).^(1/3))*V(:,1:d)';
        case 'L21'
            A=U(:,1:d)*(S(1:d,1:d).^(1/2));
            B=(S(1:d,1:d).^(1/2))*V(:,1:d)';
    end
else
    A=randn(m,d);
    B=randn(d,m);
end    
if isfield(options,'tol')==0
    tol=1e-4;
else
    tol=options.tol;
end
%
iter=0;
Q=zeros(m,n);
%
while iter<maxIter
    iter=iter+1;
    % A_new
    tau=1*u*normest(B*B');
    temp=A-u*((X-A*B+Q/u)*(-B'))/tau;
    A_new=solve_l1l2(temp,1/tau);
    % B_new
    switch regul_B
        case 'L2'
            B_new=inv(u*A_new'*A_new+alpha*eye(d))*(u*A_new'*(X+Q/u));
        case 'L21'
            tau=3*u*normest(A_new)^2;
            temp=B-u*(-A_new')*((X-A_new*B+Q/u))/tau;
            B_new=solve_l1l2(temp',alpha/tau)';
        case 'L1'
            tau=1.5*u*normest(A_new)^2;
            temp=B-u*(-A_new')*((X-A_new*B+Q/u))/tau;
            B_new=max(0,temp-alpha/tau)+min(0,temp+alpha/tau);
    end
    % X_new
    AB=A_new*B_new;
    X_new=AB-Q/u;
    X_new=X_new.*(1-M)+X0.*M;
    % 
    Q=Q+u*(X_new-AB);    %
    et=[norm(A_new-A,'fro')/norm(A,'fro') norm(B_new-B,'fro')/norm(B,'fro')  norm(X_new-X,'fro')/norm(X,'fro')];
    stopC=max(et);
    %
    isstopC=stopC<tol||et(3)<tol/10;
    if mod(iter,50)==0||isstopC||iter<=5
        disp(['iteration=' num2str(iter) '/' num2str(maxIter) '  stopC=' num2str(stopC) ...
            '  e_A=' num2str(et(1)) '  e_B=' num2str(et(2)) '  e_X=' num2str(et(3)) ' d=' num2str(d)])
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
    %     
end
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
 