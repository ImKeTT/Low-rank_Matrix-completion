function [X,A,B]=MC_MAX_pgm(X0,M,d,alpha,t,maxIter)
% min 0.5|| P(X-)
[m,n]=size(X0);
X=X0;
%[U,S,V]=svd(X0*m*n/sum(M(:)),'econ');
[U,S,V]=svd(X0,'econ');
A=U(:,1:d)*(S(1:d,1:d).^0.5);
B=(S(1:d,1:d).^0.5)*V(:,1:d)';
%
e=1e-5;
%
iter=0;
%
while iter<maxIter
    iter=iter+1;
    %
    gA=(M.*(X-A*B))*(-B');
    A_new=A-t*gA;
    A_new=maxProj(A_new,alpha);
    %
%     gB=(-A')*(M.*(X-A*B));
    gB=(-A_new')*(M.*(X-A_new*B));
    B_new=B-t*gB;
    B_new=maxProj(B_new',alpha)';
    %
    et=max([norm(A_new-A,'fro')/norm(A,'fro') norm(B_new-B,'fro')/norm(B,'fro')]);
    %
    isstopC=et<e;
    if mod(iter,100)==0||isstopC
        disp(['iteration=' num2str(iter) '/' num2str(maxIter) '  stopC=' num2str(et)])
    end
    if isstopC
        disp('converged')
        break;
    end
%
    A=A_new;
    B=B_new;
        %     
end
X=M.*X+(1-M).*(A*B);
end
%%
function X=maxProj(X,alpha)
mX=sum(X.^2,2).^0.5;
id=find(mX>alpha);
X(id,:)=X(id,:)./repmat(mX(id)/alpha,1,size(X,2));
end


 