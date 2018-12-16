%预测测试集
[bestp,bestc,bestg,cg] = GridSVMcg(trainTargets, trainData, 4, 12, -4, 4, 3, 1, 1);   %获得最优的c和g
%[bestCVaccuarcy,bestc,bestg,pso_option] = psoSVMcg(trainTargets, trainData);    %粒子群算法寻找最优的c和g
cmd = ['-c ',num2str(bestc),' -g ',num2str(bestg), ' -w1 12', ' -w2 1', ' -w3 2'];
model = svmtrain(trainTargets, trainData, cmd);     %获得最优训练模型
[predict_label, accuracy, dec_values] = svmpredict(testTargets, testData, model);   %测试集预测