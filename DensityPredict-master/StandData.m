%% Standardization each varaibles by itself
% Standardization the input in one step.
clear; clc; 
% load the training data
loadPath = '';
savePath = '';
X = load(['GPinput.mat']);
X = X.xdata;
Y = load(['GPoutputC.mat']);
Y = Y.ydata;


XT = load(['GPinputC_test.mat']);
YT = load(['GPoutputC_test.mat']);
XT = XT.xdata;
YT = (YT.ydata);


[inf_x,inf_y] = find(X==-Inf);
X(inf_x, :) = [];
Y(inf_x, :) = [];
% Normalize the data by the variables
% basic parameter
a=0; % lower
b=1; % higher
period = 3; % Length of DST
period_symh = 15;
DelaysJB = 16;      % delays of JB2008 estimation
DelaysMSIS = 16;  % delays of  MSIS estimation
% Input of training 
X_Latitude = X(:, 1); X_Longtitude = X(:, 2);
X_Height = X(:, 3);
X_EstiJB = X(:,4); X_EstiSIS = X(:, 5);
X_F107 = X(:, 6); X_F107A = X(:,7); X_F30 = X(:, 8);
X_DST = X(:, 9:9+period); 
X_AP = X(:, 9+period+1);
X_symh = X(:, 9 + period + 2: 9 + period +2+ period_symh);
X_MSIS = X(:, end-DelaysMSIS+1: end);
X_JB = X(:, end-DelaysMSIS-DelaysJB+1: end-DelaysJB);
% Input of test
Test_Nor__Latitude= XT(:, 1);
Test_Nor__Longtitude = XT(:, 2);
Test_Nor__Height = XT(:, 3);
Test_Nor__EstiJB = XT(:,4); Test_Nor__EstiSIS = XT(:, 5);
Test_Nor__F107 = XT(:, 6); Test_Nor__F107A = XT(:,7); Test_Nor__F30 = XT(:, 8);
Test_Nor__DST = XT(:, 9:9+period); Test_Nor__AP = XT(:, 9+period+1);
Test_Nor__symh = XT(:, 9 + period + 2: 9 + period +2+ period_symh);
Test_Nor__MSIS = XT(:, end-DelaysMSIS+1: end);
Test_Nor__JB = XT(:, end-DelaysMSIS-DelaysJB+1: end-DelaysJB);
%% Normalization parameters
Latimean = mean(X_Latitude);
Latistd = std(X_Latitude);
Nor_Latitude = (X_Latitude - Latimean) / Latistd;
Test_Nor_Latitude = (Test_Nor__Latitude - Latimean) / Latistd;

Longmean = mean(X_Longtitude);
Longstd = std(X_Longtitude);
Nor_Longtitude = (X_Longtitude - Longmean) / Longstd;
Test_Nor_Longtitude = (Test_Nor__Longtitude - Longmean) / Longstd;

Heightmean = mean(X_Height);
Heightstd = std(X_Height);
Nor_Height = (X_Height - Heightmean) / Heightstd;
Test_Nor_Height = (Test_Nor__Height - Heightmean) / Heightstd;



EstiJBmean = mean(X_EstiJB);
EstiJBstd = std(X_EstiJB);
Nor_EstiJB = (X_EstiJB - EstiJBmean) /EstiJBstd;
Test_Nor_EstiJB =  (Test_Nor__EstiJB - EstiJBmean) /EstiJBstd; 

EstiSISmean = mean(X_EstiSIS);
EstiSISstd = std(X_EstiSIS);
Nor_EstiSIS = (X_EstiSIS - EstiSISmean) / EstiSISstd;
Test_Nor_EstiSIS = (Test_Nor__EstiSIS - EstiSISmean) / EstiSISstd;

F107mean = mean(X_F107);
F107std = std(X_F107);
Nor_F107 = (X_F107 - F107mean) / F107std;
Test_Nor_F107 = (Test_Nor__F107 - F107mean) / F107std;

F107Amean = mean(X_F107A);
F107Astd = std(X_F107A);
Nor_F107A = (X_F107A - F107Amean) ./ F107Astd;
Test_Nor_F107A = (Test_Nor__F107A - F107Amean) ./ F107Astd;

F30mean = mean(X_F30);
F30std = std(X_F30);
Nor_F30 = (X_F30 - F30mean) / F30std;
Test_Nor_F30 = (Test_Nor__F30 - F30mean) / F30std;

DSTmean = mean(X_DST);
DSTstd = std(X_DST);
Nor_DST = (X_DST - DSTmean) ./ DSTstd;
Test_Nor_DST = (Test_Nor__DST - DSTmean) ./ DSTstd;

APmean = mean(X_AP);
APstd = std(X_AP);
Nor_AP = (X_AP - APmean) / APstd;
Test_Nor_AP = (Test_Nor__AP - APmean) / APstd;

symhmean = mean(X_symh);
symhstd = std(X_symh);
Nor_symh = (X_symh - symhmean) ./ symhstd;
Test_Nor_symh = (Test_Nor__symh - symhmean) ./ symhstd;


MSISmean = mean(X_MSIS);
MSISstd = std(X_MSIS);
Nor_MSIS = (X_MSIS - MSISmean) ./ MSISstd;
Test_Nor_MSIS = (Test_Nor__MSIS - MSISmean) ./ MSISstd;

JBmean = mean(X_JB);
JBstd = std(X_JB);
Nor_JB = (X_JB - JBmean) ./ JBstd;
Test_Nor_JB = (Test_Nor__JB - JBmean) ./ JBstd;



Nor_X = [Nor_Latitude Nor_Longtitude Nor_Height Nor_EstiJB Nor_EstiSIS Nor_F107 Nor_F107A...
    Nor_F30 Nor_DST Nor_AP Nor_symh Nor_MSIS Nor_JB];
Nor_XT = [Test_Nor_Latitude Test_Nor_Longtitude Test_Nor_Height Test_Nor_EstiJB Test_Nor_EstiSIS Test_Nor_F107 Test_Nor_F107A...
    Test_Nor_F30 Test_Nor_DST Test_Nor_AP Test_Nor_symh Test_Nor_MSIS Test_Nor_JB];






Ymean = mean(Y);
Ystd = std(Y);
Nor_Y = ((Y- Ymean) / Ystd);
Nor_YT = ((YT - Ymean) / Ystd);


save(['GPinput_Nor.mat'],'Nor_X');
save(['GPoutput_Nor.mat'],'Nor_Y');
save(['GPinput_Nor_test.mat'],'Nor_XT');
save(['GPoutput_Nor_test.mat'],'Nor_YT');



                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     