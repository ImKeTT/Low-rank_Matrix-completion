function [X,A,Z]=MC_F2(X0,M,d,alpha,u,maxIter)
[m,n]=size(X0);
X=X0;
[U,S,V]=svd(X0*m*m/sum(M(:)),'econ');
A=U(:,1:d)*(S(1:d,1:d).^0.5);
% A=randn(m,d);
Z=zeros(d,n);
u=1;
%
e=1e-6;
%
iter=0;
Q=zeros(m,n);
%
while iter<maxIter
    iter=iter+1;
%     u=u*1.001;
    % Z_new
    Z_new=inv(u/2*A'*A+alpha*eye(d))*(u/2*A'*(X+Q/u));
    % A_new
    A_new=u/2*(X+Q/u)*Z_new'*inv(u/2*Z_new*Z_new'+eye(d));
    % X_new
    X_new=A_new*Z_new-Q/u;
    X_new=X_new.*(1-M)+X0.*M;
    %
    Q=Q+u*(X_new-A_new*Z_new);
    %
    et=[norm(Z_new-Z,'fro')/norm(Z,'fro') norm(A_new-A,'fro')/norm(A,'fro') norm(X_new-X,'fro')/norm(X,'fro')];
    %
    isstopC=et(3)<e;
    if mod(iter,100)==0||isstopC
        disp(['iteration=' num2str(iter) '/' num2str(maxIter) '  stopC=' num2str(et(3)) ...
            '  e_X=' num2str(et(3))])
    end
    if isstopC
        disp('converged')
        break;
    end
%     u=u*1.01;
    Z=Z_new;
    A=A_new;
    X=X_new;
        %     
end
 