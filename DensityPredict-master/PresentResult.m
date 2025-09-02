%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  This function is used for ploting all variables of the data 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath packages/GPz-Tianyu/GPz/ % path to GPz


sat = 1;
dataPath = ' ';     % in form of 'folder\'
startdate = '07/01/2007';       % the time point relateing to a date is the begining of the day
enddate = '08/01/2007';


% Specify the date/time of period evolved
x1 = datenum(startdate);
x2 = datenum(enddate);

% specify the figures for ploting. "true" if want to plot
Residual = true;
SigmaRegion = true;
Relative = false;
EmpiricalError = false;
F107lag = false;
F107Alag = false;
F30 = false;
Dst = false;
Ap = false;
TrueDensity = false;


switch sat
    case 1
        load(['results-manually-saved/' dataPath 'GPinputC.mat']);       % 'xdata'
        load(['results-manually-saved/' dataPath 'GPoutputC.mat']);      % 'ydata'
    case 2
        load(['results-manually-saved/' dataPath 'GPinputA.mat']);       % 'xdata'
        load(['results-manually-saved/' dataPath 'GPoutputA.mat']);      % 'ydata'
end
X = xdata;
Y = ydata;
    

if Residual || SigmaRegion || Relative
    testing = true(length(Y),1);
    [ypred,sigma] = predict(X,model,'Psi',[],'selection',testing);
    sigma = sqrt(sigma);
    ysd = sigma;
    clear yint
    yint(:,1) = ypred+2*ysd;
    yint(:,2) = ypred-2*ysd;
%     ytest = Y;
end

%% create samples

x= linspace(x1,x2,length(Y));
% x = 1:length(Y); 

%% residual and relative errors of testing result

if Residual
    figure (11)
%     plot(dates,10.^ydata-10.^ypred2,'bx',x,[10.^(2*ysd+ypred)-10.^ypred,10.^(-2*ysd+ypred)-10.^ypred],'r-','LineWidth',0.1);
    plot(x,10.^Y-10.^ypred,'bx',x,[10.^(2*ysd+ypred)-10.^ypred,10.^(-2*ysd+ypred)-10.^ypred],'r-','LineWidth',0.01);      % in form of true value
    % plot(x,ytest-ypred,'.',x,[2*ysd,-2*ysd],'-');   % in form of logarithm
    grid on; axis tight;
    legend('Residual error of density prediction','Boundary of 2\sigma','fontsize',12)
    xlabel('date','fontsize',14)
    ylabel('Residual error (kg/m3)','fontsize',14)
    datetick('x',2,'keepticks');
    xlim([x(1) x(end)])
    % ylim([-2e-12 2e-12])
    

%     p0 = plot(x, 10.^Y, 'bx','linewidth',linewidth);
%     hold on; grid on; axis tight;
%     % sigma boundary
%     for k = 1:2
%         p2 = fill([x, flip(x)], [10.^(ypred-k*ysd); flip(10.^(ypred+k*ysd))], 'r',...
%             'EdgeColor','none','FaceColor',[1,0.2,0.2],'FaceAlpha',0.25);
%     end
%     % means
%     p1 = plot(x, 10.^ypred, 'r-','linewidth',linewidth);
%     title('Two-sigma region','FontSize',fontsize);
%     legend('residual error of density prediction','boundary of 2\sigma','fontsize',12)
%     xlabel('time sequence','fontsize',14)
%     ylabel('Residual error (kg/m3)','fontsize',14)
%     xlim([x(1) x(end)])
    
end

if SigmaRegion
    figure (12)
%     plot(dates,10.^ydata,'b.','LineWidth',0.25);
    plot(x,10.^Y,'b.','LineWidth',0.25);       % draw dots of true density
    hold on; grid on; axis tight;
    yForFill=[10.^yint(:,2)',fliplr(10.^yint(:,1)')];
    xForFill=[x,fliplr(x)];
%     fill(xForFill,yForFill,'c','FaceAlpha',0.25,'EdgeAlpha',1,'EdgeColor','c'); 
    % means
%     p1 = plot(x, 10.^ypred, 'r-','linewidth',0.5);
    p1 = fill([x, flip(x)], [10.^(ypred-2*ysd); flip(10.^(ypred+2*ysd))], 'r',...
        'EdgeColor','none','FaceColor',[1,0,0],'FaceAlpha',1);       % purple: [0.6,0,0.8]
    p2 = fill([x, flip(x)], [10.^(ypred-2*ysd); flip(10.^(ypred+2*ysd))], 'r',...
        'EdgeColor','none','FaceColor',[1,0,0],'FaceAlpha',1);

    legend('True density','Predicted 2\sigma area','fontsize',12)
    xlabel('date','fontsize',14)
    ylabel('Atmospheric density (kg/m3)','fontsize',14)
    datetick('x',2,'keepticks');
    xlim([x(1) x(end)])
    % ylim([0.5e-12 7.5e-12])
end

% seperate true densities into two sets, inside or outside the two-sigma area
in = []; out =[];
Rate = 0;
for i = 1:length(Y)
    if Y(i)>=ypred(i)-2*sigma(i) && Y(i)<=ypred(i)+2*sigma(i) 
        Rate = Rate+1;
        in = cat(1,in,[x(i),Y(i),ypred(i)]);
    else
        out = cat(1,out,[x(i),Y(i),ypred(i)]);
    end
end
Rate = Rate/length(Y);


if Relative
    figure (13)
    semilogy(x,abs(100.*(10.^Y-10.^ypred)./(10.^Y)),'x');
    legend('Relative error of prediction','fontsize',12)
    xlabel('time sequence','fontsize',14)
    ylabel('Relative error (%)','fontsize',14)
    grid on
    xlim([x(1) x(end)])
    % ylim([-100 100])

    figure (14)
    histogram(abs(100.*(10.^Y-10.^ypred)./(10.^Y)));
    legend('Prediction relative error distribution','fontsize',12)
    xlabel('Relative error (%)','fontsize',14)
    ylabel('Number of data','fontsize',14)
    % xlim([-5 120])
end


%% Estimation error of model JB and NRL

if EmpiricalError
    figure (15)
    semilogy(i,abs(100.*(10.^Y-10.^X(:,1))./(10.^Y)),'x');
    legend('Relative error of JB model estimation')
    xlabel('time sequence')
    ylabel('Relative error from JB (%)')
    grid on


    figure (16)
    semilogy(i,abs(100.*(10.^Y-10.^X(:,2))./(10.^Y)),'x');
    legend('Relative error of NRLMSISE-00 estimation')
    xlabel('time sequence')
    ylabel('Relative error from MSIS (%)')
    grid on
end

%% Solar and geomagnetic plots of specified data

if F107lag
    figure (17)
    plot(x,X(:,3));
    legend('Daily F10.7 with 1-day lag','fontsize',12);
    xlabel('date','fontsize',14);
    ylabel('F10.7(sfu)','fontsize',14);
    datetick('x',2);
    xlim([x(1) x(end)]);
%     ylim([100 190]);
end

if F107Alag
    figure (18)
    plot(x,X(:,4));
    legend('Averaged F10.7 with 1-day lag','fontsize',12);
    xlabel('date','fontsize',14);
    ylabel('F10.7A(sfu)','fontsize',14);
    datetick('x',2);
    xlim([x(1) x(end)]);
%     ylim([129 150]);
end

if F30
    figure (19)
    plot(x,X(:,5));
    legend('Daily F30','fontsize',12);
    xlabel('date','fontsize',14);
    ylabel('F30(sfu)','fontsize',14);
    datetick('x',2);
    xlim([x(1) x(end)]);
%     ylim([70 120]);
end

if Dst
    figure (20)
    plot(x,X(:,6));
    legend('Hourly Dst','fontsize',12);
    xlabel('date','fontsize',14);
    ylabel('Dst(nT)','fontsize',14);
    datetick('x',2);
    xlim([x(1) x(end)]);
%     ylim([-80 30]);
end

if Ap
    figure (21)
    plot(x,X(:,7));
    legend('3-hour averaged Ap','fontsize',12);
    xlabel('date','fontsize',14);
    ylabel('Ap(2nT)','fontsize',14);       %单位名不明
    datetick('x',2);
    xlim([x(1) x(end)]);
end


%% true density
if TrueDensity
    figure (22)
    plot(10.^(Y));
    legend('True density','fontsize',12)
    xlabel('time sequence','fontsize',14)
    ylabel('\rho_{true} (kg/m^3)','fontsize',14)       
    xlim([0 length(Y)])
end


