%% results from evidential model with standardized data
clear; 
format long g 
ModelNum = 5
Dst_length = 3;
Testcase = 1
switch Testcase
    case 1 
        load('results-manually-saved\TrainOnCA_New\sigma_1.mat')
        sigma = double(sigma_1);
        load('results-manually-saved\TrainOnCA_New\v_1.mat')
        v = double(v_1);
        load('results-manually-saved\TrainOnCA_New\var_1.mat')
        var = double(var_1);
        load('results-manually-saved\TrainOnCA_New\beta_1.mat')
        beta = double(beta_1);
        load('results-manually-saved\TrainOnCA_New\mu_1.mat')
        mu = double(mu_1);
        load('results-manually-saved\paper3_globalModel\GPinputC_03_1027_1103.mat')
        load('results-manually-saved\paper3_globalModel\Nor_GPinputC_03_1027_1103_CA.mat')
        load('results-manually-saved\paper3_globalModel\Nor_GPoutputC_03_1027_1103_CA.mat')
        load('results-manually-saved\TrainOnCA_New\Normalization_Parameters_CA.mat');

    case 2
        load('results-manually-saved\TrainOnCA_New\sigma_2.mat')
        sigma = double(sigma_2);
        load('results-manually-saved\TrainOnCA_New\v_2.mat')
        v = double(v_2);
        load('results-manually-saved\TrainOnCA_New\var_2.mat')
        var = double(var_2);
        load('results-manually-saved\TrainOnCA_New\beta_2.mat')
        beta = double(beta_2);
        load('results-manually-saved\TrainOnCA_New\mu_2.mat')
        mu = double(mu_2);

        load('results-manually-saved\paper3_globalModel\GPinputA_03_1027_1103.mat')
        load('results-manually-saved\paper3_globalModel\Nor_GPinputA_03_1027_1103_CA.mat')
        load('results-manually-saved\paper3_globalModel\Nor_GPoutputA_03_1027_1103_CA.mat')
        load('results-manually-saved\TrainOnCA_New\Normalization_Parameters_CA.mat');

end


load('results-manually-saved\TrainOnCA_New\delete_index.mat')

%% 

epistemic_std = double(sigma);
aleatoric_std = double(var);
mu = double(mu);
epistemic_std(:, 1:35) = [];
aleatoric_std(:, 1:35) = [];
mu(:, 1:35) = [];
Nor_YT(1:35, :) = [];
Nor_XT(1:35, :) = [];
period = 3;
xdata(badx1, :)=[];
xdata(badx2, :) = [];
X_DST = xdata(:, 9+period+6); 
X_DST(1:35,:) = [];
%%
% [nanindex, nanindex_y] = find(isnan(mu));
% if isempty(nanindex)
%     mu_nonan = mu;
%     
% else
%     
%     mu_nonan = mu(~isnan(mu));
% end

% mu_nonan = rmmissing(mu, 1);
variance_total = epistemic_std + aleatoric_std; %total uncertainty
variance = nanmean(variance_total)';

mu_mean = nanmean(mu);
% [a, n] = size(mu_nonan);
% a = min(a,n);
% %%
% if a == 1
%     mu_mean = mu_nonan;
% else
%     mu_mean = nanmean(mu_nonan);
% % end

%% 
% mu_mean = movmean(mu_mean, [0,90]);
% Nor_YT = movmean(Nor_YT, [0,90]);
% variance = movmean(variance, [0, 90]);
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
    
%     relative_error = error ./ ptest;
%     mean_re = mean(relative_error);
%     std_re = std(relative_error);5
    
    

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
%     for i = 1:length(mu)
%         if Nor_YT(i)>=mu(i)-2*variance(i) && Nor_YT(i)<=mu(i)+2*variance(i) 
%             Rate = Rate+1;
%         end
%     end
    Rate = Rate/length(ypred)
    
%%    
% %% calculate the predicted error
% [cal_time, n_length] = size(mu_nonan);
% for i = 1:cal_time
% % for i = 1
%   cur_pred = mu_nonan(i, :)';
%   pred_error(i, :) = Nor_YT - cur_pred;
% % pred_error(i, :) = Nor_Y - cur_pred;
% %       cur_alea = aleatoric_std(i, :);
% %       cur_epis = epustemic_std(i, :);
% 
% end
% 
% mean_error = nanmean(pred_error);

%%
nv = length(Nor_YT);
yi = Nor_YT;
ui = mu_mean';
sigmai = variance;

numerator = (yi - ui).^2;
denominator = (sigmai.^2);
s = sqrt((nansum(numerator ./ denominator))/nv);


%% mean absolute error percentage
% mean_abs_error = 100 * sum(abs((Nor_YT - mu_mean') ./ (Nor_YT))) / length(Nor_YT);
mean_abs_error = 100 * sum(abs((ptest - pcal) ./ (ptest))) / length(ptest);
%% Calibration Curve Calculation
% variance = var;
% mu_mean = mu;
%  variance = aleatoric_std + epistemic_std;
sigma = variance;


% nan_mu_index = find(isnan(mu));
% mu(nan_mu_index) = [];
% Nor_YT(nan_mu_index) = [];
% sigma(nan_mu_index) = [];
pred_mean = mean(mu_mean);
pred_std = std(mu_mean);
n = length(mu_mean);
CL = 0.05:0.05:0.95;
CL = [CL 0.96 0.99];


%
errorfunc = sqrt(2) *(erfinv(CL )) ;
Rate_all = [];
for k = 1:length(CL)
    CurRate = 0;
    curError = errorfunc(k);
    for i = 1:length(mu_mean)
        % if Nor_YT(i)>=mu_mean(i)-s*sigma(i)*curError && Nor_YT(i)<=mu_mean(i)+s*sigma(i)*curError 
        if Nor_YT(i)>=mu_mean(i)-1*sigma(i)*curError && Nor_YT(i)<=mu_mean(i)+1*sigma(i)*curError
            CurRate = CurRate + 1;
        end
    end
    Rate_all(k) = CurRate / n;
end

%
figure(1)
plot(CL, CL, '-o', 'MarkerSize', 10,'LineWidth',1.5);
hold on
plot(CL, Rate_all, '-*', 'MarkerSize', 10,'LineWidth',1.5);
legend('perfectly Cali system', 'Predicted Results',  'FontSize', 30);
xlabel('Confidence Level', 'FontSize',18, 'FontWeight', 'bold');
ylabel('Coverage Rate','FontSize',18, 'FontWeight', 'bold');
set(gca,'FontSize',18, 'FontWeight', 'bold');
%  title('Calivration Curve for Evid Model','FontSize',18+5, 'FontWeight', 'bold');

MACE = 1 / length(CL) * (sum(abs(CL - Rate_all)))

    %% Save data


switch Testcase
    case 1
       test_path = ['results-manually-saved\TrainOnCA_New\Test_C_03_1027_1103_CA_Result',...
        num2str(ModelNum), '.mat'];
        % save(train_path, 'MeanRatio_train', 'R_train', 'Rate_train', 'RMSE_train');
        save(test_path, 'MeanRatio', 'R', 'Rate', 'RMSE','MACE', 'Rate_all');

        all_path = ['results-manually-saved\TrainOnCA_New\Test_C_03_1027_1103_CA_All_Result_',...
            num2str(ModelNum), '.mat'];
        save(all_path); 
    case 2 
        test_path = ['results-manually-saved\TrainOnCA_New\Test_A_03_1027_1103_CA_Result',...
        num2str(ModelNum), '.mat'];
        % save(train_path, 'MeanRatio_train', 'R_train', 'Rate_train', 'RMSE_train');
        save(test_path, 'MeanRatio', 'R', 'Rate', 'RMSE','MACE', 'Rate_all');

        all_path = ['results-manually-saved\TrainOnCA_New\Test_A_03_1027_1103_CA_All_Result_',...
            num2str(ModelNum), '.mat'];
        save(all_path); 
end

%%
figure(15)

plot(ypred);
hold on
plot(ytest)
legend('pred', 'truth')
mean(ytest) - mean(ypred)


%%
rmse_pred = rmse(Nor_YT, mu_mean');
rmse_jb = rmse(Nor_YT, Nor_XT(:, 10));
rmse_nrl = rmse(Nor_YT, Nor_XT(:, 11));