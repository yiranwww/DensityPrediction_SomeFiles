
clear
format long g

%% Set up initial settings

sat = 1     % switch 1/2 denote to CHAMP/GRACE-A data or others as needed loaded
DelaysJB = 16;      % delays of JB2008 estimation
DelaysMSIS = 16;        % delays of NRL estimation
DelayTime = 300;        % The time between each delay 
testnum = 1;        % number of points used for testing in total data
Logarithm = true;      % false/true means no/yes use the logarithm to transfer values 
selection = false;      % false/true denote to close/open the selection of input according to length-scale
loadPath = '';     
savePath = '';          

% loading estimation data (copy the files previously to the doc)
switch sat
    case 1 
        load(['JBoutputC.mat']);      
        load(['MSISoutputC.mat']);     
        load(['inputsC.mat']);         
        Satellite = CHAMP;
    case 2
        load([ 'JBoutputA.mat']);       % 'EstiJB'
        load([ 'MSISoutputA.mat']);     % 'EstiMSIS'
        load([ 'inputsA.mat']);         % 'Grace','SolJB','timepos'
        Satellite = GRACE;
end
for n = 2:length(Satellite.data{1})     % find the data relsolution
    resolution = Satellite.data{1}(n)-Satellite.data{1}(n-1);
    if Satellite.data{1}(n) ~= -1 & Satellite.data{1}(n-1) ~= -1
        break;
    end
end
DelayNum = DelayTime/resolution;           % the total time between each delay of density sample 

%%

% READ GEOMAGNETIC  DST VALUE
fid = fopen('data/DSTFILE.txt','r');
DSTdata = textscan(fid,'DST%2f%2f*%2f%*6s%*3f%4f%4f%4f%4f%4f%4f%4f%4f%4f%4f%4f%4f%4f%4f%4f%4f%4f%4f%4f%4f%4f%4f%4f%4f%4f');   
% read geomagnetic Kp and Ap data
fid = fopen('data/sw19571001.txt','r');
swdata = fscanf(fid,'%d %d %d %d %d %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %d %f %f %f %f %f %f %f',[33 inf]);
swdata = swdata([1:3 6:23],:);      
fclose(fid);

% READ Solar  F30 VALUE
fid = fopen('data/radio_flux_adjusted.txt','r');
SOLdata = fscanf(fid,'%d %d %d %d %f %f %f %d %f %f %f %d %f %f %f %d %f %f %f %d %f %f %f %d',[24 inf]);
SOLdata = SOLdata([1:3 5 9 13 17 21],:);        % only YMD and f30, f15, f107, f8, f3.2 are loaded
fclose(fid);


% % READ 2007 SYMH data
% load('data\symh\SYMH.mat');
% Read SYMH 03 - 07
load('data\symh\symh_00_07_min.mat');


% mark the missing point in all data, and pick the data involved in the simulation period
for n = 1:length(Satellite.data{1})
    if Satellite.data{1}(n) == -1
        DST(n) = 0.5;        % the geomagnetic indice of missing point is marked as 0.5
        Kp(n) = 0.5;
        Kp(n) = 0.5;
        F30(n) = 0;         % the solar indice of missing point is marked as 0
        continue;
    else
        i = find(timepos(1,n)==DSTdata{1}(:)+2000 & timepos(2,n)==DSTdata{2}(:) & timepos(3,n)==DSTdata{3}(:));
        ii = timepos(4,n)+4;
%         DST(n, :) = DSTdata{ii}(i); % No time delay
        DST(n, :) = DSTdata{ii}(i-3:i);
        s = find(timepos(1,n)==SYMH(:,1) & timepos(2,n)==SYMH(:, 2) & timepos(3,n)==SYMH(:, 3) & timepos(4,n)==SYMH(:, 4) & timepos(5, n) == SYMH(:, 5));
%         ss = 5; % every hr
        ss = 7;  % every min
        symh(n, :) = SYMH(s-15:s, ss);
        j = find(timepos(1,n)==swdata(1,:) & timepos(2,n)==swdata(2,:) & timepos(3,n)==swdata(3,:));
        jj = ceil((timepos(4,n)+1)/3)+3;
        Kp(n) = swdata(jj,j);     
        Ap(n) = swdata(jj+9,j);
        k = find(timepos(1,n)==SOLdata(1,:) & timepos(2,n)==SOLdata(2,:) & timepos(3,n)==SOLdata(3,:));
        F30(n) = SOLdata(4,k);
        latitude(n) = timepos(7,n);
        longtitude(n) = timepos(8,n);
        height(n) = timepos(9,n);
        localSolarTime(n) = timepos(10, n);
        local_time = datetime([timepos(1,n), timepos(2, n), timepos(3, n), timepos(4, n), timepos(5, n), timepos(6, n)]);
        satellite_doy(n)=day(local_time, 'dayofyear');
    
    
        t1(n) = sin(2*pi*satellite_doy(n)/365.25);
        t2(n) = cos(2*pi*satellite_doy(n)/365.25);
        t3(n) = sin(2*pi*timepos(4, n) / 24);
        t4(n) = cos(2*pi*timepos(4, n) / 24);
        t5(n) = sin(2*pi*localSolarTime(n) /24);
        t6(n) = cos(2*pi*localSolarTime(n) /24);
    end
end
DST = double(DST);
Kp = double(Kp);     
Ap = double(Ap);

F107lag = SolJB(1,:);
F107Alag = SolJB(2,:);
latitude = double(latitude);
longtitude = double(longtitude);
height = double(height);

symh = symh';
symh = double(symh);



% Assembling the input 'data' with the form of  [EstiJB(3,:); EstiMSIS; F107lag; F107Alag; F30; DST; Ap; EstiJB_delays; EstiMSIS_delays]
ydata = Satellite.data{16}(:);        % shall be in form of n-by-1 vector
xdata = [t1; t2; t3; t4; t5; t6; latitude; longtitude; height; logJB; logMSIS; F107lag; F107Alag; F30; DST; Ap; symh];
DefaultInputNum = size(xdata,1);
logMSIS = log10(EstiMSIS);
logJB = log10(EstiJB(3,:));
for i = 1:DelaysJB        % Delay of JB
    zeroJB = cat(2,zeros(1,i*DelayNum),logJB);
    zeroJB = zeroJB(1:end-i*DelayNum);
    xdata = [xdata; zeroJB];
end
for i = 1:DelaysMSIS        % Delay of MSIS
    zeroMSIS = cat(2,zeros(1,i*DelayNum),logMSIS);
    zeroMSIS = zeroMSIS(1:end-i*DelayNum);
    xdata = [xdata; zeroMSIS];
end
xdata = xdata';
for n = 1:length(Satellite.data{1})     % mark the point missing while it has element from missing data point
    if DelaysJB>0 || DelaysMSIS>0       
        if ~all(xdata(n,[1:2,8:7+DelaysJB+DelaysMSIS]))
            ydata(n) = 0;
        end
    else
        if ~all(xdata(n,1:2))
            ydata(n) = 0;
        end
    end
    if abs(sum(xdata(n,:)))==inf        % mark, if the delayed estimations have zeros
            ydata(n) = 0;
    end
    if ydata(n)<0       % mark, if the target density is negative
            ydata(n) = 0;
    end
end

ydata(isnan(ydata)) = 0;    
xdata = xdata(find(ydata),:); 
ydata = ydata(find(ydata));
% transfer densities in the 1st and 2nd input with logarithm, or transfer the delays of densities back
if Logarithm
    xdata(:,1) = log10(xdata(:,1));     
    xdata(:,2) = log10(xdata(:,2));
    ydata = log10(ydata);
else
    for i = 1:DelaysJB
        xdata(:,i+DefaultInputNum) = 10.^xdata(:,i+DefaultInputNum);
    end
    for i = 1:DelaysMSIS
        xdata(:,i+DefaultInputNum+DelaysJB) = 10.^xdata(:,i+DefaultInputNum+DelaysJB);
    end
end

% select inputs according to length-scale 
 

% Seperate some data for testing
i = length(ydata)-testnum;
if i<1
    fprintf('the total valid points are less than the number of testing as required\n');
end
xtrain = xdata(1:i,:);
ytrain = ydata(1:i);
xtest = xdata(i+1:end,:);
ytest = ydata(i+1:end);

% save all input and target variable data for GPz simulation
switch sat
    case 1
        save(['GPinputC.mat'],'xdata');
        save(['GPoutputC.mat'],'ydata');
    case 2 
        save(['GPinputA.mat'],'xdata');
        save(['GPoutputA.mat'],'ydata');
end

