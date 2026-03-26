function [ACC, NMI,Pur,ARI,FSC] = AlgorithmMeasure(Y,gnd)

rand('twister',5678);
nClass = length(unique(gnd));
n = 10;
ACC = zeros(2, 1);
NMI = zeros(2, 1);
Pur = zeros(2, 1);
ARI = zeros(2, 1);
FSC = zeros(2, 1);
tempACC = zeros(n,1);
tempNMI = zeros(n,1);
tempPur = zeros(n,1);
tempARI = zeros(n,1);
tempFSC = zeros(n,1);

for i = 1:n
%      rand('twister',5678);
%     label = kmeans(Y,nClass,'emptyaction','singleton');
%     label = litekmeans(Y,nClass,'Replicates',10);
   
    label = litekmeans(Y,nClass,'MaxIter',100,'Replicates',10);
    idx = bestMap(gnd,label);
    tempACC(i) = length(find(gnd == idx))/length(gnd);
    [~,iNMI,~] = compute_nmi(label, gnd);
    tempNMI(i) = iNMI;
%     tempNMI(i) = MutualInfo(gnd,idx);
    tempPur(i) = Purity(gnd,idx);
    tempARI(i) = rand_index(gnd,idx);
    [iFSC, ~,~]=compute_f(label, gnd);
    tempFSC(i) = iFSC;

%     tempACC(i) = clusterAccMea(gnd,idx);
%     tempNMI(i) = nmi(gnd,idx);
%     tempPur(i) = Purity(gnd,idx);
%     tempARI(i) = rand_index(gnd,idx);

%     label = litekmeans(Y,nClass,'Replicates',15);
%     res = bestMap(gnd,label);
%     tempACC(i) = sum((gnd - res)==0)/length(gnd);
%     tempNMI(i) = MutualInfo(gnd,label);
end
               
ACC(:,1) = [mean(tempACC),std(tempACC)];   % mean and std
NMI(:,1) = [mean(tempNMI),std(tempNMI)];   % mean and std
Pur(:,1) = [mean(tempPur),std(tempPur)];   % mean and std
ARI(:,1) = [mean(tempARI),std(tempARI)];   % mean and std
FSC(:,1) = [mean(tempFSC),std(tempFSC)];   % mean and std;

end