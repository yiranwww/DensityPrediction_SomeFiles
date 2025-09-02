figure;

%% 设置多条曲线时，自动颜色顺序和图标顺序，必须放最前
set(gca,'NextPlot','replacechildren');
% set(gca,'LineStyleOrder','-^|-*|-o');
% set(gca,'ColorOrder',[0,0,0; 1,0,0]);

%% 画图的内容
temp = 0:0.1:5;
plot(temp,sin(temp),temp,cos(temp),temp,cos(sin(temp)));
%%% 标注
xlabel('xasdfasd1','FontSize',16,'FontName','TimeNewRoman');
ylabel('速度v/s','FontSize',16,'FontName','TimeNewRoman','Rotation',0,'HorizontalAlignment','Right');
title('速度$v/s$','FontSize',33,'FontName','timesnewroman','Interpreter','Latex');

%% 设置具体内容
%%% 坐标上的数字大小
set(gca,'FontSize',14,...
        'FontName','Times New Roman');
%%% label，title字号，字体
set(get(gca,'xLabel'),'FontSize',16,...
                      'FontName','Times New Roman');
set(get(gca,'yLabel'),'FontSize',16,...
                      'FontName','Times New Roman',...
                      'Rotation',0,...
                      'HorizontalAlignment','Right');
set(get(gca,'title'),'FontName','Times New Roman',...
                     'FontSize',33);
%%% 线的宽度也可以后期调节
set(get(gca,'Children'),'LineWidth',5);