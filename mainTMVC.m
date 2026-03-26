function mainALTL
clc;clear
addpath('ClusteringMeasure')
addpath('./data')
%% compared dataset

% load('AwA_fea.mat');
% load('UCI_3view.mat');
% load('WebKB.mat');
% load('Caltech101-20.mat');
% load('100Leaves.mat');
% load('3sourcesbr.mat'); 
% load('Digit10k.mat');
% load('Handwritten_numerals.mat'); 
% load('NUSWIDEOBJ.mat');
% load('cifar10.mat');
% load('VGGFace2_50.mat'); 
% load('mnist10k.mat'); 
% load('Digit10k.mat');
% load('Hdigit.mat');
% load('NUS.mat');
% load('MNIST.mat');

% load('CCV.mat');
% load('HW.mat');
% load('Scene15.mat');
% load('MSRC.mat');


% load('ORL.mat');


% load('Hdigit.mat');
% load('CCV.mat');
% load('Scene15.mat');
% load('Animal.mat');
% load('VGGFace2_50.mat'); 
% load('MNIST.mat');
% load('cifar10.mat');

% load('WebKB.mat');
% load('BDGP.mat');
% load('HW.mat');
% load('mnist4.mat');
load('./data/mfeat.mat');
gnd = Y;
%X = fea;
for i = 1:5
    %X{i} = X{i}';
end

nSmp = size(X{1},1);           % sample number
nV = length(X);                % view number
nC = length(unique(gnd)); 
clear fea;

lam1 = 2;                    % Parameters 2-3: alpha (tunable)
lam3 = 1;
d =20;

q =90;           %  number of anchors (tunable)
lam2 = 1e-4;                    
p = 0.7;



eta = 5;

%% Construct the initial similarity matrix
opt.ReducedDim = 100;
for pv = 1:nV
    pW = PCA(X{pv}',opt);
    X{pv} = X{pv} * pW;
end
K=[];
Z = cell(1,nV );
rand('twister',5489);
for iv = 1:nV 
    X{iv} = double(X{iv});
  X{iv} = NormalizeFea(X{iv});     
    [~, A{iv}] = litekmeans(X{iv},q,'MaxIter', 100,'Replicates',10);
  
    K(:,:,iv) = ComputingZ(X{iv},A{iv},nSmp,q); 
    Z{iv} = K(:,:,iv);
end  

%% My idea: Anchor-Guided Low-Rank Tensor Learning for Multi-view Subspace Clustering (ALTL)
tic;
O = TMVC(X',A,Z,gnd,q,lam1,lam2,lam3,d,p,eta);

%% metrics and stds
[ACC, NMI,Pur,ARI,FSC] = AlgorithmMeasure(O,gnd);
disp(['Clustering in the TMVC subspace. ACC and STD: ',num2str(ACC(1,1)*100),'±',num2str(ACC(2,1)*100)]);
disp(['Clustering in the TMVC subspace. NMI and STD: ',num2str(NMI(1,1)*100),'±',num2str(NMI(2,1)*100)]);
disp(['Clustering in the TMVC subspace. PUR and STD: ',num2str(Pur(1,1)*100),'±',num2str(Pur(2,1)*100)]);
disp(['Clustering in the TMVC subspace. ARI and STD: ',num2str(ARI(1,1)*100),'±',num2str(ARI(2,1)*100)]);
disp(['Clustering in the TMVC subspace. FSC and STD: ',num2str(FSC(1,1)*100),'±',num2str(FSC(2,1)*100)]);
toc;

end