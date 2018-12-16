%����Ԥ����
[~, ~, rawFile] = xlsread('D:\Software\MATLAB\Project\SVM\libsvm-master\matlab\train.csv');
rawData = cell2mat(rawFile(2:end,2:(end-1)));   %ԭʼ���ݼ�
rawTargets = char(rawFile(2:end,end));  %ԭʼ���ݼ������
[m, n] = size(rawData);    %mΪ��������nΪ������
data = zeros(m, n); %�������ݼ�
targets = [];   %�������ݼ������
c = 9;  %�������9
for i = 1:c
    str = strcat('Class_', num2str(i));
    IndexArray{i} = findStrInArray(rawTargets, str);    %����ͬ�������ݷ���һ��Ԫ����
end
k = 0;
for i = 1:c
    for j = 1:length(IndexArray{i})
        k = k + 1;
        data(k,:) = rawData(IndexArray{i}(j),:);    %��ԭʼ���ݼ���������
    end
    targets = [targets;i * ones(length(IndexArray{i}),1)];  %��ÿ���������ǩ
end
[stdData, PS] = mapstd(data',0,1);  %��������һ��������ֵΪ�㣬����Ϊһ
stdData = stdData'; %��һ������