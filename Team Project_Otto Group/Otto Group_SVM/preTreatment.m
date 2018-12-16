%数据预处理
[~, ~, rawFile] = xlsread('D:\Software\MATLAB\Project\SVM\libsvm-master\matlab\train.csv');
rawData = cell2mat(rawFile(2:end,2:(end-1)));   %原始数据集
rawTargets = char(rawFile(2:end,end));  %原始数据集的类别
[m, n] = size(rawData);    %m为样本数，n为特征数
data = zeros(m, n); %有序数据集
targets = [];   %有序数据集的类别
c = 9;  %类别总数9
for i = 1:c
    str = strcat('Class_', num2str(i));
    IndexArray{i} = findStrInArray(rawTargets, str);    %将相同类别的数据放在一个元胞中
end
k = 0;
for i = 1:c
    for j = 1:length(IndexArray{i})
        k = k + 1;
        data(k,:) = rawData(IndexArray{i}(j),:);    %对原始数据集进行排序
    end
    targets = [targets;i * ones(length(IndexArray{i}),1)];  %对每个类别贴标签
end
[stdData, PS] = mapstd(data',0,1);  %将样本归一化——均值为零，方差为一
stdData = stdData'; %归一化样本