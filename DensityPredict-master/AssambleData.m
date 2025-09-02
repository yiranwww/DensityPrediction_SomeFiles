% Manually load data files of 1st part

load('inputsC.mat')
load('JBoutputC.mat')
load('MSISoutputC.mat')

%% save data variables of 1st part
C = Champ;

S = SolJB;
t = timepos;
EJ = EstiJB;
EM = EstiMSIS;

G = Grace;

%% Manually load data files of 2nd part

% load('inputsC.mat')
% load('JBoutputC.mat')
% load('MSISoutputC.mat')

%% Assamble two data parts
for i =1:16
    
    Champ.data{i} = cat(2,C.data{i},Champ.data{i});
    
end

SolJB = cat(2,S,SolJB);
timepos = cat(2,t,timepos);
EstiJB = cat(2,EJ,EstiJB);
EstiMSIS = cat(2,EM,EstiMSIS);

for i =1:16
    
    Grace.data{i} = cat(2,G.data{i},Grace.data{i});
    
end

%% Save variables into new files
save('inputsC.mat','Champ','SolJB','timepos');
save('JBoutputC.mat','EstiJB');
save('MSISoutputC.mat','EstiMSIS');

save('inputsA.mat','Grace','SolJB','timepos');
save('JBoutputA.mat','EstiJB');
save('MSISoutputA.mat','EstiMSIS');