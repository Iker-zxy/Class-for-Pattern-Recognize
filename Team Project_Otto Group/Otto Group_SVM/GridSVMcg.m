function [bestp,bestc,bestg,pArray] = GridSVMcg(train_label,train_data,cmin,cmax,gmin,gmax,v,cstep,gstep,accstep)
% [bestacc,bestc,bestg,cg] = SVMcgPP(train_label,train,cmin,cmax,gmin,gmax,v,cstep,gstep,accstep)
%
% train_label:ѵ�� ����ǩ.Ҫ����libsvm��������Ҫ��һ��.
% train:ѵ����.Ҫ����libsvm��������Ҫ��һ��.
% cmin:�ͷ�����c�ı仯��Χ����Сֵ(ȡ��2Ϊ�׵Ķ�����),�� c_min = 2^(cmin).Ĭ��Ϊ -5
% cmax:�ͷ�����c�ı仯��Χ�����ֵ(ȡ��2Ϊ�׵Ķ�����),�� c_max = 2^(cmax).Ĭ��Ϊ 5
% gmin:����g�ı仯��Χ����Сֵ(ȡ��2Ϊ�׵Ķ�����),�� g_min = 2^(gmin).Ĭ��Ϊ -5
% gmax:����g�ı仯��Χ����Сֵ(ȡ��2Ϊ�׵Ķ�����),�� g_min = 2^(gmax).Ĭ��Ϊ 5
%
% v:cross validation�Ĳ���,�������Լ���Ϊ�����ֽ���cross validation.Ĭ��Ϊ 3
% cstep:����c�����Ĵ�С.Ĭ��Ϊ 1
% gstep:����g�����Ĵ�С.Ĭ��Ϊ 1
% accstep:�����ʾ׼ȷ��ͼʱ�Ĳ�����С. Ĭ��Ϊ 1.5
 
%% ��ȱʡ�������г�ʼ��
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

%% ��ȡc��g�����
cArray = cmin:cstep:cmax;
gArray = gmin:gstep:gmax;
m = length(cArray);    %C����ĳ���
n = length(gArray);    %G����ĳ���
A = zeros(m * n, 2);   %c��g�����
pArray =[];
k = 1;
for i = 1:m
    for j = 1:n
        A(k, 1) = cArray(i);
        A(k, 2) = gArray(j);
        k = k + 1;
    end
end

%% ��ѵ�������н�����֤������ѡ�񽻲���֤������ߣ�c��С��c��g���
bestc = 0;
bestg = 0;
bestp = 0;
b = 2;  %ָ���ĵ�
MyPar = parpool;    %�򿪲��д����
parfor i = 1:m * n
    cmd = ['-v ',num2str(v),' -c ',num2str( b ^ A(i, 1) ),' -g ',num2str( b ^ A(i, 2)), ' -t 2', ' -w1 12', ' -w2 1', ' -w3 2', ' -h 0'];   %��˹�˺���   
    pArray(i) = svmtrain(train_label, train_data, cmd); %������֤���ʼ���
end
delete(MyPar)   %������ɺ�رղ��д����
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
[X, Y] = meshgrid(cmin:cstep:cmax,gmin:gstep:gmax);  %��ȡ��������
pMatrix = reshape(pArray, [n, m]);    %�������н�����֤�������飬��þ�����ʽ

%% ���Ʋ�ͬc��g�ĵȸ���ͼ
[C, h] = contour(X, Y, pMatrix, 50:accstep:100);    %���ƽ�����֤���ʾ�����60%~100%�ĵȸ���ͼ
clabel(C, h, 'FontSize', 10, 'Color', 'r');
xlabel('log2c', 'FontSize', 10);
ylabel('log2g', 'FontSize', 10);
grid on;
