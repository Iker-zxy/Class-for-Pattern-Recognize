import numpy as np
from PIL import Image

def rebuild_img(U, Sigma, VT, p): #p表示奇异值的百分比
    a = np.zeros((D,N))
    Sum = (int)(sum(Sigma))
    curSum = 0
    num = 0
    while curSum <= Sum * p:
        curSum += Sigma[num] 
        num += 1
    print("信息保留率为"+ str(p * 100)+ "%时需要奇异值个数",num)
    U_num = U[:,0:num] # 保留的U矩阵
    VT_num = VT[0:num,:] # 保留的VT矩阵
    Sigma_num = Sigma[0:num]
    S =np.zeros((num,num))
    S = np.diag(Sigma_num)# 保留的sigma矩阵
    a = U_num.dot(S).dot(VT_num)
    a[a < 0] = 0
    a[a > 255] = 255
    #按照最近距离取整数，并设置参数类型为uint8
    return np.rint(a).astype("uint8")

img = Image.open('SVD_test.jpg', 'r')
A = np.array(img)
D = A.shape[0]
N = A.shape[1]
for p in (np.arange(1,10,1)/10): # 设置奇异值占比（10%,20%...90%） arange最好用于整数
    # 对每一个通道分别进行处理
    U, Sigma, VT = np.linalg.svd(A[:, :, 0])
    R = rebuild_img(U, Sigma, VT, p)

    U, Sigma, VT = np.linalg.svd(A[:, :, 1])
    G = rebuild_img(U, Sigma, VT, p)

    U, Sigma, VT = np.linalg.svd(A[:, :, 2])
    B = rebuild_img(U, Sigma, VT, p)
    # 把三个通道合起来，axis=2表示把通道数放在最后
    I = np.stack((R, G, B), 2)
    #保存图片在指定文件夹下
    Image.fromarray(I).save("D:\\onedrive\\13.Code_for_public\\" + str(p * 100) + "%.jpg")
    print(str(p * 100) + "%.jpg"+" has finished")