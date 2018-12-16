function [bestp,bestc,bestg,pArray] = GridSVMcg(train_label,train_data,cmin,cmax,gmin,gmax,v,cstep,gstep,accstep)
% [bestacc,bestc,bestg,cg] = SVMcgPP(train_label,train,cmin,cmax,gmin,gmax,v,cstep,gstep,accstep)
%
% train_label:训练 集标签.要求与libsvm工具箱中要求一致.
% train:训练集.要求与libsvm工具箱中要求一致.
% cmin:惩罚参数c的变化范围的最小值(取以2为底的对数后),即 c_min = 2^(cmin).默认为 -5
% cmax:惩罚参数c的变化范围的最大值(取以2为底的对数后),即 c_max = 2^(cmax).默认为 5
% gmin:参数g的变化范围的最小值(取以2为底的对数后),即 g_min = 2^(gmin).默认为 -5
% gmax:参数g的变化范围的最小值(取以2为底的对数后),即 g_min = 2^(gmax).默认为 5
%
% v:cross validation的参数,即给测试集分为几部分进行cross validation.默认为 3
% cstep:参数c步进的大小.默认为 1
% gstep:参数g步进的大小.默认为 1
% accstep:最后显示准确率图时的步进大小. 默认为 1.5
 
%% 对缺省参数进行初始化
if nargin < 3
    accstep = 1.5;
    v = 3;
    cstep = 1;
    gstep = 1;
    gmax = 5;
    gmin = -5;
    cmax = 5;
    cmin = -5;
elseif nargin < 4
    accstep = 1.5;
    v = 3;
    cstep = 1;
    gstep = 1;
    gmax = 5;
    gmin = -5;
    cmax = 5;   
elseif nargin < 5
    accstep = 1.5;
    v = 3;
    cstep = 1;
    gstep = 1;
    gmax = 5;
    gmin = -5;
elseif nargin < 6
    accstep = 1.5;
    v = 3;
    cstep = 1;
    gstep = 1;
    gmax = 5;
elseif nargin < 7
    accstep = 1.5;
    v = 3;
    cstep = 1;
    gstep = 1;
elseif nargin < 8
    accstep = 1.5;
    cstep = 1;
    gstep = 1;
elseif nargin < 10
    accstep = 1.5;
end

%% 获取c和g的组合
cArray = cmin:cstep:cmax;
gArray = gmin:gstep:gmax;
m = length(cArray);    %C数组的长度
n = length(gArray);    %G数组的长度
A = zeros(m * n, 2);   %c和g的组合
pArray =[];
k = 1;
for i = 1:m
    for j = 1:n
        A(k, 1) = cArray(i);
        A(k, 2) = gArray(j);
        k = k + 1;
    end
end

%% 对训练集进行交叉验证，从中选择交叉验证概率最高，c最小的c和g组合
bestc = 0;
bestg = 0;
bestp = 0;
b = 2;  %指数的底
MyPar = parpool;    %打开并行处理池
parfor i = 1:m * n
    cmd = ['-v ',num2str(v),' -c ',num2str( b ^ A(i, 1) ),' -g ',num2str( b ^ A(i, 2)), ' -t 2', ' -w1 12', ' -w2 1', ' -w3 2', ' -h 0'];   %高斯核函数   
    pArray(i) = svmtrain(train_label, train_data, cmd); %交叉验证概率计算
end
delete(MyPar)   %计算完成后关闭并行处理池
% parfor i = 1:m
%     for j = 1:n
%         cmd = ['-v ',num2str(v),' -c ',num2str( basenum^X(i,j) ),' -g ',num2str( basenum^Y(i,j) )];
%         cg(i,j) = svmtrain(train_label, train, cmd);
%     end
% end
%         if cg(i,j) > bestacc
%             bestacc = cg(i,j);
%             bestc = basenum^X(i,j);
%             bestg = basenum^Y(i,j);
%         end
%         if ( cg(i,j) == bestacc && bestc > basenum^X(i,j) )
%             bestacc = cg(i,j);
%             bestc = basenum^X(i,j);
%             bestg = basenum^Y(i,j);
%         end
[bestp, I] = max(pArray);
bestc = b ^ A(I, 1);
bestg = b ^ A(I, 2);
[X, Y] = meshgrid(cmin:cstep:cmax,gmin:gstep:gmax);  %获取网格坐标
pMatrix = reshape(pArray, [n, m]);    %重新排列交叉验证概率数组，获得矩阵形式

%% 绘制不同c和g的等高线图
[C, h] = contour(X, Y, pMatrix, 50:accstep:100);    %绘制交叉验证概率矩阵中60%~100%的等高线图
clabel(C, h, 'FontSize', 10, 'Color', 'r');
xlabel('log2c', 'FontSize', 10);
ylabel('log2g', 'FontSize', 10);
grid on;
