%Ԥ����Լ�
[bestp,bestc,bestg,cg] = GridSVMcg(trainTargets, trainData, 4, 12, -4, 4, 3, 1, 1);   %������ŵ�c��g
%[bestCVaccuarcy,bestc,bestg,pso_option] = psoSVMcg(trainTargets, trainData);    %����Ⱥ�㷨Ѱ�����ŵ�c��g
cmd = ['-c ',num2str(bestc),' -g ',num2str(bestg), ' -w1 12', ' -w2 1', ' -w3 2'];
model = svmtrain(trainTargets, trainData, cmd);     %�������ѵ��ģ��
[predict_label, accuracy, dec_values] = svmpredict(testTargets, testData, model);   %���Լ�Ԥ��