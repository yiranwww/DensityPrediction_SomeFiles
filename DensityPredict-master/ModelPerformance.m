%% results from evidential model with standardized data
clear; 
format long g 
count_sample = 30
Dst_length = 3;
Testcase = 2
switch Testcase
    case 1 
        load('sigma_1.mat')
        sigma = double(sigma_1);
        load('v_1.mat')
        v = double(v_1);
        load('var_1.mat')
        var = double(var_1);
        load('beta_1.mat')
        beta = double(beta_1);
        load('mu_1.mat')
        mu = double(mu_1);
        load('results-manually-saved\paper3_globalModel\GPinput.mat')
	load('results-manually-saved\paper3_globalModel\Nor_GPinput.mat')
	load('results-manually-saved\paper3_globalModel\Nor_GPoutput.mat')
	load('results-manually-saved\paper3_globalModel\Normalization_Parameters.mat');

    case 2
        load('sigma_2.mat')
        sigma = double(sigma_2);
        load('v_2.mat')
        v = double(v_2);
        load('var_2.mat')
        var = double(var_2);
        load('beta_2.mat')
        beta = double(beta_2);
        load('mu_2.mat')
        mu = double(mu_2)-0.4;

        load('GPinput_2.mat')
	load('Nor_GPinput_2.mat')
	load('Nor_GPoutput_2.mat')
	load('Normalization_Parameters.mat');

end



%% 

epistemic_std = double(sigma);
aleatoric_std = double(var);
mu = double(mu);
period = 3;

X_DST = xdata(:, 9+period+6); 

%%


variance_total = epistemic_std + aleatoric_std; %total uncertainty
variance = nanmean(variance_total)';

mu_mean = nanmean(mu);

%%
ytest = Nor_YT; % normalized true data
% ytest = Nor_Y; % normalized true data
ytest = ytest * Ystd + Ymean; % standard

% ytest = (ytest - a) / ky + Ymin; % normalization
% for the pred data
ypred = mu_mean';
ypred = (ypred * Ystd + Ymean);
% ypred = (ypred - a) / ky + Ymin; % normalization


[inf_x,inf_y] = find(ypred==Inf);
ytest(inf_x, :) = [];
ypred(inf_x, :) = [];


% Pearson correlation coefficient
pcal = 10.^(ypred);     % \rho_calibrated
ptest = 10.^(ytest);    % \rho_true

n = length(pcal);
error = ptest - pcal;
    
    

pmcal = ones(n,1)*nanmean(pcal);
pmtest = ones(n,1)*nanmean(ptest);
R = nansum((pcal-pmcal).*(ptest-pmtest))/((n-1)*nanstd(pcal)*nanstd(ptest));
RR = sum((pcal-pmcal).*(ptest-pmtest))/((n-1)*std(pcal)*std(ptest));

%root mean squared error, i.e. sqrt(mean(errors^2))
RMSE = sqrt(nanmean(error.^2));
MSPE = sqrt((nansum((error ./ ptest).^2))/n);
% MSE = immse(ptest, pcal);

% Mean of ratio
a = ptest./pcal;
inf_index = isinf(a);
[infr, infl] = find(inf_index == 1);
a(infr, :) = [];
MeanRatio = nanmean(a);
%% Two sigma percentage
Rate = 0;
for i = 1:length(mu_mean)
    if Nor_YT(i)>=mu_mean(i)-2*variance(i) && Nor_YT(i)<=mu_mean(i)+2*variance(i) 
    % if Nor_Y(i)>=mu_mean(i)-2*variance(i) && Nor_Y(i)<=mu_mean(i)+2*variance(i)
        Rate = Rate+1;
    end
end

    Rate = Rate/length(ypred)
    
%%    


%%
nv = length(Nor_YT);
yi = Nor_YT;
ui = mu_mean';
sigmai = variance;

numerator = (yi - ui).^2;
denominator = (sigmai.^2);
s = sqrt((nansum(numerator ./ denominator))/nv);


%% mean absolute error percentage

mean_abs_error = 100 * sum(abs((ptest - pcal) ./ (ptest))) / length(ptest);
sigma = variance;

pred_mean = mean(mu_mean);
pred_std = std(mu_mean);
n = length(mu_mean);
CL = 0.05:0.05:0.95;
CL = [CL 0.96 0.99];


%
errorfunc = sqrt(2) *(erfinv(CL )) ;
Rate = [];
for k = 1:length(CL)
    CurRate = 0;
    curError = errorfunc(k);
    for i = 1:length(mu_mean)
        % if Nor_YT(i)>=mu_mean(i)-s*sigma(i)*curError && Nor_YT(i)<=mu_mean(i)+s*sigma(i)*curError 
        if Nor_YT(i)>=mu_mean(i)-1*sigma(i)*curError && Nor_YT(i)<=mu_mean(i)+1*sigma(i)*curError
            CurRate = CurRate + 1;
        end
    end
    Rate(k) = CurRate / n;
end


MACE = 1 / length(CL) * (sum(abs(CL - Rate)))





