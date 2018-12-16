# -*-Otto Group 商品识别_随机森林（数据预处理）-*-
# -*-加载训练数据集函数loadTrainSet()-*-
# -*-加载测试数据集函数loadTestSet()-*-
# -*-评估函数evaluation(label,pred_label)-*-
# ------ 答案存入csv文件saveResult(testlabel,filename = "submission.csv") ------ #
import csv
import random
import numpy  as np

# ------ 加载训练数据集 ------ #
def loadTrainSet():
	traindata = []
	trainlabel = []
	table = {"Class_1":1,"Class_2":2,"Class_3":3,"Class_4":4,"Class_5":5,
	"Class_6":6,"Class_7":7,"Class_8":8,"Class_9":9}
	with open("train.csv") as f:
		rows = csv.reader(f)
		next(rows)
		for row in rows:                #row为train.csv文件[序号+每个商品的93个feat+label](95)
			l = []        
			for i in range(1,94):
				l.append(int(row[i]))   #l中存入第n个商品的93个feat
			traindata.append(l)         #存入train所有商品的feat
			trainlabel.append(table.get(row[-1]))   #存入train所有商品的label(class num改为num)
	f.close()

	traindata = np.array(traindata,dtype="float")
	trainlabel = np.array(trainlabel,dtype="int")
	#数据标准化(零均值，归一化)
	mean = traindata.mean(axis=0)       #求均值
	std = traindata.std(axis=0)         #求标准差
	traindata = (traindata - mean)/std
	x_range = range(len(trainlabel))
	#打乱数据顺序
	randomIndex = [i for i in x_range]   #长度为49502
	random.shuffle(randomIndex)          #0-49501顺序打乱
	traindata = traindata[randomIndex]
	trainlabel = trainlabel[randomIndex]
	return traindata,trainlabel

# ------ 加载测试数据集 ------ #
def loadTestSet():
	testdata = []
	with open("test.csv") as f:
		rows = csv.reader(f)
		next(rows)
		for row in rows:
			l = []
			for i in range(1,94):
				l.append(int(row[i]))
			testdata.append(l)
	f.close()
	testdata = np.array(testdata,dtype="float")
	#数据标准化(零均值，归一化)
	mean = testdata.mean(axis=0)
	std = testdata.std(axis=0)
	testdata = (testdata - mean)/std
	return testdata


# ------ 评估函数 ------ #
def evaluation(label,pred_label):
	num = len(label)
	logloss = 0.0
	for i in range(num):
		p = max(min(pred_label[i][label[i]-1],1-10**(-15)),10**(-15))
		logloss += np.log(p)
	logloss = -1*logloss/num
	return logloss


# ------ 答案存入csv文件 ------ #
def saveResult(testlabel,filename = "submission.csv"):
	with open(filename,'w',newline ='') as myFile:
		myWriter=csv.writer(myFile)
		byte_temp=['id','Class_1','Class_2','Class_3','Class_4','Class_5','Class_6','Class_7','Class_8','Class_9']
		myWriter.writerow(byte_temp)
		id_num = 1
		for eachlabel in testlabel:
			l = []
			l.append(id_num)
			l.extend(eachlabel)
			myWriter.writerow(l)
			id_num += 1