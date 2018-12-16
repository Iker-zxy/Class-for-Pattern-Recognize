# -*-Otto Group 商品识别_随机森林-*-
import time
import preprocess   #数据预处理包
from sklearn.ensemble import RandomForestClassifier

# ------ 加载数据集 ------ #
def loaddata():
	print ("loading data...")
	#加载测试数据集 train.csv, 将其分为训练数据和校验数据
	data,label = preprocess.loadTrainSet()
	val_data = data[0:6000]       #校验数据选择前6000个
	val_label = label[0:6000]     #前6000个校验数据的类别
	train_data = data[6000:]      #训练数据
	train_label = label[6000:]    #训练数据的类别
	#加载测试数据集 test.csv
	test_data = preprocess.loadTestSet()
	return train_data,train_label,val_data,val_label,test_data

# ------ 构建随机森林模型并完成分类 ------ #
def rf(train_data,train_label,val_data,val_label,test_data,name="RandomForest_submission.csv"):
	print ("Start training Random forest...")
	rfClf = RandomForestClassifier(n_jobs=4, n_estimators=1000, max_features=20, min_samples_split=3,
                                    bootstrap=False, verbose=3, random_state=23)   #建立分类型决策树
	rfClf.fit(train_data,train_label)  #训练模型
	val_pred_label = rfClf.predict_proba(val_data)  #判定结果
	logloss = preprocess.evaluation(val_label,val_pred_label)  #根据官方评估公式计算
	print ("logloss of validation set:",logloss)
	print ("Start classify test set...")
	test_label = rfClf.predict_proba(test_data)
	preprocess.saveResult(test_label,filename = name)

# ------ 主函数 ------ #
if __name__ == "__main__":
	t1 = time.time()
	train_data,train_label,val_data,val_label,test_data = loaddata()
	rf(train_data,train_label,val_data,val_label,test_data) 
	t2 = time.time()
	print ("Done! It cost",t2-t1,"s")
