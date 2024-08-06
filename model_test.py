import numpy as np

# 加载数据集
data = np.load('x_qua_test.npy')

# 计算最大最小值
min_val = np.min(data)
max_val = np.max(data)

print("Min value:", min_val)
print("Max value:", max_val)
