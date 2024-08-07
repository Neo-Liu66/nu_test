import os as os
import pathlib as pathlib
import numpy as np
# from pycoral.utils.edgetpu import make_interpreter
# from pycoral.utils import edgetpu
import tflite_runtime.interpreter as tflite
import time
import platform

script_dir = pathlib.Path(__file__).parent.absolute()
model_file = os.path.join(script_dir, 'int8_cpu_2D.tflite')
print(f"测试点-模型路径：{model_file}")
# model_file2 = os.path.join(script_dir, '2d_tpu.tflite')
# print(f"测试点-模型路径：{model_file2}")
data_file = os.path.join(script_dir, 'x_test_noisy1.npy')
print(f"测试点-数据路径：{data_file}")
data_file = np.load(data_file)

EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]

def make_interpreter(model_file):
  model_file, *device = model_file.split('@')
  return tflite.Interpreter(
      model_path=model_file,
      experimental_delegates=[
          tflite.load_delegate(EDGETPU_SHARED_LIB,
                               {'device': device[0]} if device else {})
      ])

#######################################################################
# model_file='tpu_part.tflite'
# 2d_cpu = np.load("x_test_noisy1.npy")
########################################################################

# %% 2. Run tensorflow lite models
def runTFLite(input_data):
    print('进入运行函数')
    #interpreter = tflite.Interpreter(model_path=model_file)
    interpreter = make_interpreter(model_file)
    # interpreter = make_interpreter(model_file2)
    #interpreter = tflite.Interpreter(model_file)
    print('模型导入成功')
    interpreter.allocate_tensors()
    print('张量分配成功')

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print('获取输入输出信息成功')
    # Prepare the test dataset (replace with your test data)

    #test_data = input_data.astype(np.int8)
    min_val = np.min(input_data)
    max_val = np.max(input_data)
    scaled_data = (input_data - min_val) / (max_val - min_val) * 255 - 128
    test_data = np.round(scaled_data).astype(np.int8)

    # Run inference on each test sample
    results = []
    start_time = time.time()
    for sample in test_data:
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], sample.reshape((1,1, 800,1)))
        # Run inference
        interpreter.invoke()
        # Get the output
        output_data = interpreter.get_tensor(output_details[0]['index'])
        results.append(output_data)
    end_time = time.time()

    total_time = end_time - start_time
    # Convert the results to a NumPy array
    results = np.array(results)
    print(results.shape)
    results = np.squeeze(results, axis=(1, 2, 4))
    results = (results + 128) / 255 * (max_val - min_val) + min_val
    return results, total_time


def main():
    decoded_layer, total_time = runTFLite(data_file)
    print(f'The time required to execute inference is:{total_time}')
    np.save('output_result.npy', decoded_layer)
    print("First 5 samples of quantized data:", decoded_layer[:5])

if __name__ == "__main__":
    main()

