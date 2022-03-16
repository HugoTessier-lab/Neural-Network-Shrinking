import onnxruntime
import numpy as np
from time import time
import argparse

if __name__ == '__main__':
    t = time()
    log = ''
    parser = argparse.ArgumentParser(description='Neural Network Shrinking')


    def tuple_type(strings):
        strings = strings.replace("(", "").replace(")", "")
        mapped_int = map(int, strings.split(","))
        return tuple(mapped_int)


    parser.add_argument('--input_shape', type=tuple_type, default=(1, 3, 100, 100))
    parser.add_argument('--pruning_rate', type=float, default=0.)
    args = parser.parse_args()

    log += f'Pruning rate : {args.pruning_rate}, Input shape : {args.input_shape}, '
    name = f'./onnx_networks/rate_{args.pruning_rate}_dims_{args.input_shape}.onnx'
    ort_session = onnxruntime.InferenceSession(name, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider',
                                                                'CPUExecutionProvider'])
    ort_inputs = {ort_session.get_inputs()[0].name: np.ones(args.input_shape, dtype=np.float32)}
    log += f'Overhead time : {time() - t}, '

    t = time()
    for _ in range(1000):  # Run
        ort_session.run(None, ort_inputs)
    t = time() - t
    log += f'Elapsed time for 1000 runs {t}\n'
    with open('./results.txt', 'a') as f:
        f.write(log)
