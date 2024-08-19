import os as os
import pathlib as pathlib
import numpy as np
import time
from tflite_runtime.interpreter import Interpreter


script_dir = pathlib.Path(__file__).parent.absolute()
model_file = os.path.join(script_dir, 'float32_cpu.tflite')
data_file = os.path.join(script_dir, 'x_test_noisy1.npy')
data_file = np.load(data_file)


# %% 2. Run tensorflow lite models
def runTFLite(input_data):
    interpreter = Interpreter(model_path=model_file)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    data_file = input_data.astype(np.float32)


    # Run inference on each test sample
    results = []
    start_time = time.time()
    elapsed_time = 0  # 初始化已用时间

    while elapsed_time < 60:  # 只要已用时间小于300秒
        for sample in data_file:
            # 设置输入张量
            interpreter.set_tensor(input_details[0]['index'], sample.reshape((1, 1, 800, 1)))
            # 运行推理
            interpreter.invoke()
            # 获取输出
            output_data = interpreter.get_tensor(output_details[0]['index'])
            results.append(output_data)

        # 更新已用时间
        elapsed_time = time.time() - start_time
    print(f'cpu_float32 推理总时长为{elapsed_time}')
    # Convert the results to a NumPy array
    results = np.array(results)
    print(results.shape)
    results = np.squeeze(results, axis=(1, 2, 4))
    return results, elapsed_time


def main():
    decoded_layer, total_time = runTFLite(data_file)
    print(f'float32 on CPU Inference time is:{total_time}')
    np.save('float32_CPU_result.npy', decoded_layer)

if __name__ == "__main__":
    main()

