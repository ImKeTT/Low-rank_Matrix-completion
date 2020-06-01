% This folder contains the main codes utilized in the following paper:
% Factor Group-Sparse Regularization for Efficient Low-Rank Matrix
% Recovery. Jicong Fan, Lijun Ding, Yudong Chen, Madeleine Udell. NeurIPS
% 2019.
% Written by Jicong Fan, 09/2019. E-mail: jf577@cornell.edu

MC_FGSR_ADMM noiseless matrix completion FGSR-2/3 or FGSR-1/2
MC_FGSR_PALM noisy matrix completion FGSR-2/3 or FGSR-1/2
MC_FGSRp_PALM noisy matrix completion FGSR-p arbitrary small p

%
RPCA_FGSR_ADMM RPCA based on FGSR solved by ADMM
% other algorithms
MC_Nulcear_IALM Nuclear norm for matrix completion
MC_FNuclear_ADMM F-nuclear norm for noiseless matrix completion
MC_FNuclear_PALM F-nuclear norm for noisy matrix completion
MC_MAX_pgm Max norm with projected gradient method
%
inexact_alm_rpca RPCA based on nuclear norm solved by inexact ALM
RobustPCA RPCA based on nuclear norm solved by ADMM
RPCA_FNuclear_ADMM F-nuclear norm for robust PCA
%
test_XXX_XXX examples