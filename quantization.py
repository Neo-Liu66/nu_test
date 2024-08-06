import numpy as np

# 加载数据集
data = np.load('x_qua_test.npy')

# 计算最大最小值
min_val = np.min(data)
max_val = np.max(data)
print(data)
print("Min value:", min_val)
print("Max value:", max_val)

# 缩放数据到int8范围
scaled_data = (data - min_val) / (max_val - min_val) * 255 - 128
scaled_data = np.round(scaled_data).astype(np.int8)

# 将缩放后的数据量化回正常范围
quantized_data = (scaled_data + 128) / 255 * (max_val - min_val) + min_val

print("First 5 samples of quantized data:", data[:5])
print("First 5 samples of scaled data:", scaled_data[:5])
print("First 5 samples of quantized data:", quantized_data[:5])



