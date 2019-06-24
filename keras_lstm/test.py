import numpy as np
import tensorflow as tf

a = [1,2,3]

b = [i+4 for i in a]
print(b)
a = np.array([[1,2],[2,3],[3,4]])

b = a[[1,2]]
print(b)

a = np.array(range(24*2)).reshape([4,3,4])
mean_a = np.mean(a,axis=0)
std_a = np.std(a,axis=0)
max_a = np.max(a,axis=0)
print(a,a.shape)
print("*"*60)
print(max_a)
print(max_a.shape)


# print("&&&"*30)
# print(a-mean_a)
# print((a-mean_a).shape)
# print("*^"*30)
# print("std:",std_a)
# print("std:",std_a.shape)
# print("biaozhunhua:",(a-mean_a)/std_a)




# fw.close()

