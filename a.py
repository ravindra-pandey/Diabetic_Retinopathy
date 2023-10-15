import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
trt_model = trt.create_inference_graph(
    input_graph_def=frozen_graph,
    outputs=output_names,
    max_batch_size=1,
    max_workspace_size_bytes=(1 << 30),
    precision_mode="FP16")
