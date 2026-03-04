import sys
import os

# Comprehensive list of standard ncnn layer types
# These names match the first column in an ncnn .param file
ALL_NCNN_LAYERS = [
 "absval", "argmax", "batchnorm", "bias", "binaryop", "bnll", "cast", "celu", "clip", "concat", "convolution", "convolution1d", "convolution3d", "convolutiondepthwise", "convolutiondepthwise1d", "convolutiondepthwise3d", "copyto", "crop", "cumulativesum", "deconvolution", "deconvolution1d", "deconvolution3d", "deconvolutiondepthwise", "deconvolutiondepthwise1d", "deconvolutiondepthwise3d", "deepcopy", "deformableconv2d", "dequantize", "detectionoutput", "diag", "dropout", "einsum", "eltwise", "elu", "embed", "erf", "exp", "expanddims", "flatten", "flip", "fold", "gelu", "gemm", "glu", "gridsample", "groupnorm", "gru", "hardsigmoid", "hardswish", "innerproduct", "input", "instancenorm", "interp", "inversespectrogram", "layernorm", "log", "lrn", "lstm", "matmul", "memorydata", "mish", "multiheadattention", "mvn", "noop", "normalize", "packing", "padding", "permute", "pixelshuffle", "pooling", "pooling1d", "pooling3d", "power", "prelu", "priorbox", "proposal", "psroipooling", "quantize", "reduction", "relu", "reorg", "requantize", "reshape", "rmsnorm", "rnn", "roialign", "roipooling", "rotaryembed", "scale", "sdpa", "selu", "shrink", "shufflechannel", "sigmoid", "slice", "softmax", "softplus", "spectrogram", "split", "spp", "squeeze", "statisticspooling", "swish", "tanh", "threshold", "tile", "unaryop", "unfold", "yolodetectionoutput", "yolov3detectionoutput"
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