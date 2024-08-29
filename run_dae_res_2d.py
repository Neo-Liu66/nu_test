import os as os
import pathlib as pathlib
import numpy as np
import time
from tflite_runtime.interpreter import Interpreter

script_dir = pathlib.Path(__file__).parent.absolute()
model_file = os.path.join(script_dir, 'DAE_Res_2D.tflite')
print(f"测试点-模型路径：{model_file}")
# model_file2 = os.path.join(script_dir, '2d_tpu.tflite')
# print(f"测试点-模型路径：{model_file2}")
data_file = os.path.join(script_dir, 'x_test_noisy1.npy')
print(f"测试点-数据路径：{data_file}")
input_data = np.load(data_file)
print(f"测试点-打印数据：{input_data}")

# %% 2. Run tensorflow lite models
def runTFLite(input_data):
    print('进入运行函数')
    #interpreter = tflite.Interpreter(model_path=model_file)
    interpreter = Interpreter(model_path=model_file)
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
    data_file = input_data.astype(np.float32)

    # Run inference on each test sample
    results = []
    start_time = time.time()
    for sample in data_file:
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
    results = np.squeeze(results,axis=(1, 3))
    return results, total_time


def main():
    decoded_layer, total_time = runTFLite(input_data)
    print(f'DAE_Res_2D on CPU Inference time is:{total_time}')
    np.save('DAE_Res_2D_result.npy', decoded_layer)

if __name__ == "__main__":
    main()

