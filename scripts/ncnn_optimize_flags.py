import sys
import os

# Comprehensive list of standard ncnn layer types
# These names match the first column in an ncnn .param file
ALL_NCNN_LAYERS = [
    "AbsVal", "ArgMax", "BatchNorm", "Bias", "BinaryOp", "BNLL", "Cast", "Clip",
    "Concat", "Convolution", "Convolution1D", "ConvolutionDepthWise", "Crop",
    "Deconvolution", "DeconvolutionDepthWise", "DetectionOutput", "Dropout",
    "Eltwise", "ELU", "Embed", "Exp", "ExpandDims", "Flatten", "InnerProduct",
    "Input", "Interp", "Log", "LRN", "LSTM", "MemoryData", "MVN", "Normalize",
    "Packing", "Padding", "Permute", "Pooling", "Power", "PReLU", "PriorBox",
    "Proposal", "PSROIPooling", "Quantize", "ReLU", "Reorg", "Requantize",
    "Reshape", "RNN", "ROIAlign", "ROIPooling", "Scale", "Selu", "ShuffleChannel",
    "Sigmoid", "Slice", "Softmax", "Split", "SPP", "Squeeze", "Tanh", "Threshold",
    "Tile", "UnaryOp", "YoloDetectionOutput", "Yolov3DetectionOutput", "Mish",
    "GELU", "Softplus", "Swish", "HardSigmoid", "HardSwish", "PixelShuffle", "GLU"
]

def get_used_layers(param_files):
    used_layers = set()
    
    for file_path in param_files:
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found. Skipping.", file=sys.stderr)
            continue
            
        with open(file_path, 'r') as f:
            lines = f.readlines()
            # .param files: line 0 is magic, line 1 is counts. Layers start at line 2.
            for line in lines[2:]:
                parts = line.split()
                if not parts:
                    continue
                
                layer_type = parts[0]
                used_layers.add(layer_type)
                
    return used_layers

def generate_cmake_flags(used_layers):
    # Standardize used layers to lowercase for comparison
    used_lower = {l.lower() for l in used_layers}
    
    flags = []
    always_on_layers = ["convolution", "innerproduct", "padding", "flatten", "relu", "clip", "input",
                         "split", "noop", "packing", "cast", "reshape", "binaryop", "unaryop", "scale" ]
    for layer in sorted(ALL_NCNN_LAYERS):
        # ncnn CMake flags are always lowercase
        layer_flag_name = layer.lower()
        
        status = "ON" if (layer_flag_name in used_lower) or (layer_flag_name in always_on_layers) else "OFF"
        flags.append(f"-DWITH_LAYER_{layer_flag_name}={status}")
        
    return flags

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ncnn_optimize_flags.py <model1.param> <model2.param> ...")
        sys.exit(1)

    param_paths = sys.argv[1:]
    used = get_used_layers(param_paths)
    cmake_flags = generate_cmake_flags(used)

    print("\n# Copy and paste these flags into your ncnn CMake build command:\n")
    print(" ".join(cmake_flags))