import os
import json
import argparse
import tensorrt as trt

TRT_LOGGER = trt.Logger()

EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

def GiB(val):
    return val * 1 << 30

def json_load(filename):
    with open(filename) as json_file:
        data = json.load(json_file)
    return data

def set_dynamic_range(network, json_file):
    
    quant_param_json = json_load(json_file)
    act_quant = quant_param_json["act_quant_info"]

    for i in range(network.num_inputs):
        input_tensor = network.get_input(i)
        if act_quant.__contains__(input_tensor.name):
            print(f"input_tensor.name = {input_tensor.name}")
            value = act_quant[input_tensor.name]
            tensor_max = abs(value)
            tensor_min = - abs(value)
            input_tensor.dynamic_range = (tensor_min, tensor_max)
    
    for i in range(network.num_layers):
        layer = network.get_layer(i)

        for output_index in range(layer.num_outputs):
                tensor = layer.get_output(output_index)
                if act_quant.__contains__(tensor.name):
                    value = act_quant[tensor.name]
                    tensor_max = abs(value)
                    tensor_min = - abs(value)
                    print(f"tensor_min = {tensor_min}\ttensor_max = {tensor_max}")
                    print(f"tensor.name = {tensor.name}")
                    tensor.dynamic_range = (tensor_min, tensor_max)
                else:
                    print(f"tensor.name = {tensor.name}")

def build_engine(onnx_file, json_file, engine_file):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(EXPLICIT_BATCH)

    config = builder.create_builder_config()

    parser = trt.OnnxParser(network, TRT_LOGGER)
    config.max_workspace_size = GiB(1)

    if not os.path.exists(onnx_file):
        quit(f"onnx_file {onnx_file} not found")
    
    with open(onnx_file, "rb")as model:
        if not parser.parse(model.read()):
            print(f"Failed to parse model")
            for error in range(parser.num_errors):
                print(f"parser.error = {parser.get_error(error)}")
            return None
    
    config.set_flag(trt.BuilderFlag.INT8)

    set_dynamic_range(network, json_file)

    engine = builder.build_engine(network, config)

    with open(engine_file, "wb") as f:
        f.write(engine.serialize())

if __name__ == "__main__":

    parser = argparse.ArgumentParser("2trt tool.", add_help=True)

    parser.add_argument(
        "--onnx_model_path",
        type=str,
        default="./model/quantized_float_yolov5.onnx",
        help="The path of quantized float yolov5",
    )
    
    parser.add_argument(
        "--onnx_json_path",
        type=str,
        default="./model/quantized_yolov5.json",
        help="The path of quantized yolov5 json file",
    )

    parser.add_argument(
        "--quantized_res_path",
        type=str,
        default="./model/quantized_yolov5.engine",
        help="quantized outputs",
    )

    args = parser.parse_args()
    print(f"Command Line Args: {args}")

    build_engine(args.onnx_model_path, args.onnx_json_path, args.quantized_res_path)



            









