import numpy as np
from preprocessing import load_mnist

## 神经网络程序中需要用到的函数
def sigmoid(x):
    s = 1/(1+np.exp(-x))
    return s

def softmax(x):
    e_x = np.exp(x)
    return e_x / np.sum(e_x, axis=0)

def cross_entropy_error(y, t):
    delta = 1e-7
    out = 0 - np.sum(t * np.log(y + delta))
    return out/(y.shape[1])

def predict(W1,b1,W2,b2,x):
    z1 = np.dot(W1,x) + b1 
    h = sigmoid(z1)               
    z2 = np.dot(W2,h) + b2     
    y = softmax(z2)              
    return y

def accuracy(W1,b1,W2,b2,x,t):
    y = predict(W1,b1,W2,b2,x)
    y = np.argmax(y, axis=0)
    if t.ndim != 1 : t = np.argmax(t, axis=0)
    accuracy = np.sum(y == t) / float(x.shape[1])

    return accuracy

## 读入数据
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
x_train = np.transpose(x_train)  # (784,60000)
t_train = np.transpose(t_train)
x_test = np.transpose(x_test)
t_test = np.transpose(t_test)
## 构建网络
input_size = 784 
hidden_size = 50
output_size = 10


## hyperparament
iters_num = 30000    # 迭代次数
train_size = x_train.shape[1]  # 60000
batch_size = 100   
learning_rate = 0.5   # 学习率

train_loss_list = []
train_acc_list = []   # 训练准确率
test_acc_list = []   # 测试准确率

iter_per_epoch = max(train_size / batch_size, 1)  # 600

# 初始化权值和阈值
weight_init_std = 0.01
W1 = weight_init_std * np.random.randn(hidden_size, input_size)   # (50,784)
b1 = np.zeros([hidden_size,1])  # (50,1)
W2 = weight_init_std * np.random.randn(output_size, hidden_size)  # (10,50)
b2 = np.zeros([output_size,1])  # (10,1)


for i in range(iters_num):
    # 获取mini-batch
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[:,batch_mask] # x_batch (784,100)
    t_batch = t_train[:,batch_mask] # t_batch (10,100)

    # 信息正向传播
    z1 = np.dot(W1,x_batch) + b1 # (50,100) =(50,784)*(784,100)+(50,100)
    h = sigmoid(z1)               # h (50,100)
    z2 = np.dot(W2, h) + b2     # (10,100)=(10,50)*(50,100)+(10,100)
    y = softmax(z2)               # y (10,100)

    # 计算损失函数
    loss = cross_entropy_error(y, t_batch)

    # 记录学习过程
    train_loss_list.append(loss)

    # 计算梯度
    delta2 = t_batch -y    # (10,100)
    dW2 = np.dot(delta2,np.transpose(h))/batch_size      # (10,50)=(10,100)*(100,50)
    db2 = np.dot(delta2,np.ones([batch_size,1]))/batch_size  # (10,1)=(10,100)*(100,1)

    delta1 = np.dot(np.transpose(W2), delta2)*h*(1-h)   # (50,100) = (50,10)*(10,100).*(50,100).*(50,100)
    dW1 = np.dot(delta1,np.transpose(x_batch))/batch_size # (50,784)=(50,100)*(100,784)
    db1 = np.dot(delta1,np.ones([batch_size,1])) /batch_size    # (50,1)=(50,100)*(100,1)

    # 参数更新
    W2 += learning_rate * dW2
    b2 += learning_rate * db2
    W1 += learning_rate * dW1
    b1 += learning_rate * db1

    # 计算每个epoch的识别精度
    if i % iter_per_epoch == 0:
        # 训练数据和测试数据准确率
        train_acc = accuracy(W1,b1,W2,b2,x_train,t_train)
        test_acc = accuracy(W1,b1,W2,b2,x_test,t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(loss,train_acc,test_acc)

## 运行结果
# 2.3040706091343304 0.09863333333333334 0.0958
# 0.28961521512041527 0.9113166666666667 0.9129
# 0.1921242844976112 0.93075 0.9316
# 0.1891586816023407 0.9434833333333333 0.9398
# 0.1003634224735616 0.9517666666666666 0.9487
# 0.051866777924511186 0.9564333333333334 0.9527
# 0.07172360170632575 0.9615166666666667 0.9542
# 0.21441582352185648 0.9646666666666667 0.959
# 0.11572362139139296 0.9671833333333333 0.9599
# 0.09095220498152656 0.9697 0.9617
# 0.1382166812586494 0.97185 0.9634
# 0.07507840592954794 0.9726333333333333 0.9646
# 0.10607956540926464 0.9748333333333333 0.9664
# 0.05210701799513539 0.97655 0.9666
# 0.04626495754931174 0.9777333333333333 0.9674
# 0.04084230224776894 0.97965 0.9688
# 0.11171917168801664 0.9792666666666666 0.9675
# 0.05612538178513386 0.9812833333333333 0.9683
# 0.1186567790168904 0.9818666666666667 0.9687
# 0.03859125922106604 0.9818333333333333 0.969
# 0.10879301959397658 0.9832833333333333 0.9696
# 0.03912446390823776 0.9843 0.9712
# 0.06631756208785464 0.9844333333333334 0.9702
# 0.15351636655701625 0.9852333333333333 0.9696
# 0.01858272746404142 0.9855333333333334 0.9708
# 0.04822175124097575 0.9864666666666667 0.9695
# 0.015888655339938502 0.9861833333333333 0.9697
# 0.044213287512438804 0.9878833333333333 0.9715
# 0.027295864308971036 0.9875333333333334 0.9702
# 0.06479083838640745 0.9885833333333334 0.9714
# 0.02051919464081634 0.98905 0.9725
# 0.015986254236823424 0.9895666666666667 0.9713
# 0.04210142923978525 0.9894166666666667 0.972
# 0.07916599674200157 0.9894833333333334 0.9705
# 0.06896683234012752 0.9906333333333334 0.9721
# 0.03632265913718414 0.99135 0.9718
# 0.024415118272971572 0.991 0.9719
# 0.09340213127419775 0.9920333333333333 0.9729
# 0.0682863077335279 0.9918666666666667 0.9712
# 0.00683263383167319 0.9921 0.9721
# 0.02935252647984703 0.9930666666666667 0.9726
# 0.02453386907449574 0.9929833333333333 0.972
# 0.02840109456637797 0.9928166666666667 0.9728
# 0.018930343173986026 0.9935166666666667 0.9739
# 0.024157499229930997 0.9937 0.973
# 0.020584132942723876 0.994 0.973
# 0.02068580399430185 0.9944 0.9731
# 0.054301667951511874 0.9945 0.9716
# 0.015249718101183318 0.9948333333333333 0.9744
# 0.08426353923441521 0.9946833333333334 0.9738