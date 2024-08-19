import os as os
import pathlib as pathlib
import numpy as np
import time
from tflite_runtime.interpreter import Interpreter


script_dir = pathlib.Path(__file__).parent.absolute()
model_file = os.path.join(script_dir, 'int8_cpu.tflite')
data_file = os.path.join(script_dir, 'x_test_noisy1.npy')
data_file = np.load(data_file)


# %% 2. Run tensorflow lite models
def runTFLite(input_data):
    interpreter = Interpreter(model_path=model_file)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    #test_data = input_data.astype(np.float32)
    min_val = np.min(input_data)
    max_val = np.max(input_data)
    scaled_data = (input_data - min_val) / (max_val - min_val) * 255 - 128
    test_data = np.round(scaled_data).astype(np.int8)

    # Run inference on each test sample
    results = []
    start_time = time.time()
    elapsed_time = 0  # 初始化已用时间

    for _ in range(5):
        start_time = time.time()
        for sample in test_data:

            # 设置输入张量
            interpreter.set_tensor(input_details[0]['index'], sample.reshape((1, 1, 800, 1)))
            # 运行推理
            interpreter.invoke()
            # 获取输出
            output_data = interpreter.get_tensor(output_details[0]['index'])
            results.append(output_data)

        # 更新已用时间
        end_time = time.time()
        # 更新已用时间
        used_time = end_time - start_time
        print(f"int8_CPU,用时{used_time}")
    results = np.array(results)
    print(results.shape)
    results = np.squeeze(results, axis=(1, 2, 4))
    results = (results + 128) / 255 * (max_val - min_val) + min_val
    return results, used_time


def main():
    decoded_layer, total_time = runTFLite(data_file)
    np.save('int8_CPU_result.npy', decoded_layer)



if __name__ == "__main__":
    main()

