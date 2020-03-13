%  min_X  ||X.*A = Y.*A||_F^2 + r*||X||_Sp^p
%  matrix completion problem
function [X, obj, st]=OtraceEIC_my(Y, A, p, r, X0, st)
% Y: m*n input data matrix, Y.*A is the observed elements in the data matrix
% A: m*n mask matrix, A(i,j)=1 if Y(i,j) is observed, otherwise A(i,j)=0
% p: the p value of the Schatten p-Norm
% r: parameter
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

if nargin > 4
    X = X0;
else
    X = Y;
end;

if nargin < 6
    temp = Y'*Y;
    st = 0.002*max(abs(diag(temp)));  % sometimes a larger st will achieve better result
end;


%D = p/2*(X'*X+st*eye(n))^(p/2-1);
[U, S, V] = svd(X'*X); s = zeros(n,1); t = diag(S); s(1:length(t)) = t;
D = p/2*U*diag((s+st).^(p/2-1))*V';

for iter = 1:50 
    for i = 1:m
        Ai = A(i,:);
        Yi = Y(i,:);
        AY = Ai.*Yi;
        Aid = diag(Ai);
        X(i,:) = (Aid+r*D)\(AY');
        if (i == 1)
            %disp((Aid+r*D)\(AY'))
        end
    end;
    %D = p/2*(X'*X+st*eye(n))^(p/2-1);
    [U, S, V] = svd(X'*X); s = zeros(n,1); t = diag(S); s(1:length(t)) = t;
    D = p/2*U*diag((s+st).^(p/2-1))*V';
    
    if ~isreal(D)
        disp('not real');
    end;
    
    obj(iter) = trace((X.*A-Y)'*(X.*A-Y))+ r*sum((s+st).^(p/2));
end;






