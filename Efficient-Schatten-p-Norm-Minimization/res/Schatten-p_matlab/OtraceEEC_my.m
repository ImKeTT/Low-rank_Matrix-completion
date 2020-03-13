%  min_X  ||X||_Sp^p   s.t.   X.*A = Y.*A
%  matrix completion problem
function [X, obj, st]=OtraceEEC_my(Y, A, p,X0,st)
% Y: m*n input data matrix, Y.*A is the observed elements in the data matrix
% A: m*n mask matrix, A(i,j)=1 if Y(i,j) is observed, otherwise A(i,j)=0.(H denoted in the paper)
% p: the p value of the Schatten p-Norm
% X0: m*n initialization of recovered data matrix (optional)
% st: regularization to avoid singularity of matrix
% X: recovered data matrix
% obj: objective values during iterations

% Ref:
% Feiping Nie, Heng Huang, Chris Ding. 
% Efficient Schatten-p Norm Minimization for Low-Rank Matrix Recovery. 
% The 26th AAAI Conference on Artificial Intelligence (AAAI), Toronto, Ontario, Canada, 2012.


[m, n] = size(Y);

Y = Y.*A;
%nargin ?????????????
if nargin > 3
    X = X0;
else
    X = Y;
end;

if nargin < 5
    temp = Y'*Y;
    st = 0.002*max(abs(diag(temp))); % sometimes a larger st will achieve better result
end;

%X' = X.T
D = ((X'*X+st*eye(n))^(1-p/2));

for iter = 1:50
    Lambda = zeros(m,n);
    
    for i = 1:m
        Ai = A(i,:)';
        idx=find(Ai==1);
        DA1 = D(idx,idx);
        disp(size(DA1))
        Lambda(i,idx) = (DA1)\(Y(i,idx)');% = (Y(i,idx)')/(DA1)
    end;

    X = (Lambda.*A)*D;   
    D = ((X'*X+st*eye(n))^(1-p/2));
    
    if ~isreal(D)
        disp('not real');
    end;
    
    obj(iter) = trace(real((X'*X+st*eye(n))^(p/2)));
end;




