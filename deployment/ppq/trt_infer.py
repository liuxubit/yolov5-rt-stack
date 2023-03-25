import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import argparse
from PIL import Image

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem
    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)
    def __repr__(self):
        return self.__str__()

def alloc_buf_N(engine, data):

    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    data_type = []

    for binding in engine:
        # if engine.binding_is_input(binding):
        if engine.get_tensor_mode(binding):
            size = data.shape[0] * data.shape[1] * data.shape[2] * data.shape[3]
            # dtype = trt.nptype(engine.get_binding_dtype(binding))
            dtype = trt.nptype(engine.get_tensor_dtype(binding))
            data_type.append(dtype)
            
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            bindings.append(int(device_mem))
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            host_mem = cuda.pagelocked_empty(size, data_type[0])
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            outputs.append(HostDeviceMem(host_mem, device_mem))

    return inputs, outputs, bindings, stream

def do_inference_v2(context, inputs, bindings, outputs, stream, data):

    for inp in inputs:
        cuda.memcpy_htod_async(inp.device, inp.host, stream)
    context.set_binding_shape(0, data.shape)
    # context.set_input_shape(0, data.shape)

    context.execute_async(batch_size=1, bindings=bindings, stream_handle=stream.handle)

    for out in outputs:
        print(f"\n123*********************")
        cuda.memcpy_dtoh_async(out.host, out.device, stream)
    
    stream.synchronize()

    return [out.host for out in outputs]

trt_logger = trt.Logger(trt.Logger.INFO)

def load_engine(engine_path):
    TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
    with open(engine_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

if __name__ == "__main__":

    parser = argparse.ArgumentParser("2trt tool.", add_help=True)

    parser.add_argument(
        "--engine_path",
        type=str,
        default="./model/quantized_yolov5.engine",
        help="The path of quantized engine",
    )

    parser.add_argument(
        "--input_path",
        type=str,
        default="./distilled_data/1.jpg",
        help="The path of input_data",
    )

    args = parser.parse_args()
    src_data = cv2.imread(args.input_path)
    print(f"args.input_path = {args.input_path}")
    print(f"src_data.shape = {src_data.shape}")
    batch = 1
    height, width, channel = src_data.shape
    print(f"height = {height}\twidth = {width}\tchannel = {channel}")
    inputs = np.array(src_data).astype(np.float32)
    data = np.reshape(inputs, (batch, channel, height, width))
    engine = load_engine(args.engine_path)
    context = engine.create_execution_context()
    inputs_alloc_buf, outputs_alloc_buf, binding_alloc_buf, stream_alloc_buff = alloc_buf_N(engine, data)
    inputs_alloc_buf[0].host = np.ascontiguousarray(inputs)

    trt_feature = do_inference_v2(context, inputs_alloc_buf, binding_alloc_buf, outputs_alloc_buf, stream_alloc_buff, data)
    trt_feature = np.asarray(trt_feature)
    print(trt_feature)
