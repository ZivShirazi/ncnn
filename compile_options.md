### Recommended compile flags

For the example OCR, the recommended compile flags are:

```bash
 cmake -S . -B build-size -A x64 -DCMAKE_BUILD_TYPE=Release -DNCNN_F16C=ON -DNCNN_INT8=ON -DNCNN_BF16=OFF -DNCNN_SIMPLEVK=OFF -DNCNN_SIMPLESTL=OFF -DNCNN_SHARED_LIB=OFF -DNCNN_BUILD_WITH_STATIC_CRT=ON -DNCNN_ENABLE_LTO=OFF -DNCNN_BUILD_EXAMPLES=ON -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_BENCHMARK=OFF -DNCNN_BUILD_TESTS=OFF -DNCNN_VULKAN=OFF -DNCNN_OPENMP=OFF -DNCNN_THREADS=OFF -DNCNN_RUNTIME_CPU=ON -DNCNN_C_API=OFF -DNCNN_PLATFORM_API=OFF-DNCNN_PIXEL=ON -DNCNN_PIXEL_ROTATE=ON -DNCNN_PIXEL_AFFINE=ON -DNCNN_PIXEL_DRAWING=OFF -DNCNN_AVX=ON -DNCNN_AVX2=OFF -DNCNN_FMA=OFF -DNCNN_XOP=OFF -DNCNN_AVX512=OFF -DNCNN_DISABLE_RTTI=ON -DNCNN_STRING=OFF -DNCNN_STDIO=OFF -DWITH_LAYER_absval=OFF -DWITH_LAYER_argmax=OFF -DWITH_LAYER_batchnorm=OFF -DWITH_LAYER_bias=OFF -DWITH_LAYER_binaryop=ON -DWITH_LAYER_bnll=OFF -DWITH_LAYER_cast=ON -DWITH_LAYER_celu=OFF -DWITH_LAYER_clip=ON -DWITH_LAYER_concat=ON -DWITH_LAYER_convolution=ON -DWITH_LAYER_convolution1d=OFF -DWITH_LAYER_convolution3d=OFF -DWITH_LAYER_convolutiondepthwise=ON -DWITH_LAYER_convolutiondepthwise1d=OFF -DWITH_LAYER_convolutiondepthwise3d=OFF -DWITH_LAYER_copyto=OFF -DWITH_LAYER_crop=OFF -DWITH_LAYER_cumulativesum=OFF -DWITH_LAYER_deconvolution=ON -DWITH_LAYER_deconvolution1d=OFF -DWITH_LAYER_deconvolution3d=OFF -DWITH_LAYER_deconvolutiondepthwise=OFF -DWITH_LAYER_deconvolutiondepthwise1d=OFF -DWITH_LAYER_deconvolutiondepthwise3d=OFF -DWITH_LAYER_deepcopy=OFF -DWITH_LAYER_deformableconv2d=OFF -DWITH_LAYER_dequantize=OFF -DWITH_LAYER_detectionoutput=OFF -DWITH_LAYER_diag=OFF -DWITH_LAYER_dropout=OFF -DWITH_LAYER_einsum=OFF -DWITH_LAYER_eltwise=OFF -DWITH_LAYER_elu=OFF -DWITH_LAYER_embed=OFF -DWITH_LAYER_erf=OFF -DWITH_LAYER_exp=OFF -DWITH_LAYER_expanddims=OFF -DWITH_LAYER_flatten=ON -DWITH_LAYER_flip=OFF -DWITH_LAYER_fold=OFF -DWITH_LAYER_gelu=OFF -DWITH_LAYER_gemm=ON -DWITH_LAYER_glu=OFF -DWITH_LAYER_gridsample=OFF -DWITH_LAYER_groupnorm=OFF -DWITH_LAYER_gru=OFF -DWITH_LAYER_hardsigmoid=ON -DWITH_LAYER_hardswish=ON -DWITH_LAYER_innerproduct=ON -DWITH_LAYER_input=ON -DWITH_LAYER_instancenorm=OFF -DWITH_LAYER_interp=ON -DWITH_LAYER_inversespectrogram=OFF -DWITH_LAYER_layernorm=ON -DWITH_LAYER_log=OFF -DWITH_LAYER_lrn=OFF -DWITH_LAYER_lstm=OFF -DWITH_LAYER_matmul=OFF -DWITH_LAYER_memorydata=OFF -DWITH_LAYER_mish=OFF -DWITH_LAYER_multiheadattention=ON -DWITH_LAYER_mvn=OFF -DWITH_LAYER_noop=ON -DWITH_LAYER_normalize=OFF -DWITH_LAYER_packing=ON -DWITH_LAYER_padding=ON -DWITH_LAYER_permute=ON -DWITH_LAYER_pixelshuffle=OFF -DWITH_LAYER_pooling=ON -DWITH_LAYER_pooling1d=OFF -DWITH_LAYER_pooling3d=OFF -DWITH_LAYER_power=OFF -DWITH_LAYER_prelu=OFF -DWITH_LAYER_priorbox=OFF -DWITH_LAYER_proposal=OFF -DWITH_LAYER_psroipooling=OFF -DWITH_LAYER_quantize=OFF -DWITH_LAYER_reduction=OFF -DWITH_LAYER_relu=ON -DWITH_LAYER_reorg=OFF -DWITH_LAYER_requantize=OFF -DWITH_LAYER_reshape=ON -DWITH_LAYER_rmsnorm=OFF -DWITH_LAYER_rnn=OFF -DWITH_LAYER_roialign=OFF -DWITH_LAYER_roipooling=OFF -DWITH_LAYER_rotaryembed=OFF -DWITH_LAYER_scale=ON -DWITH_LAYER_sdpa=OFF -DWITH_LAYER_selu=OFF -DWITH_LAYER_shrink=OFF -DWITH_LAYER_shufflechannel=OFF -DWITH_LAYER_sigmoid=ON -DWITH_LAYER_slice=OFF -DWITH_LAYER_softmax=ON -DWITH_LAYER_softplus=OFF -DWITH_LAYER_spectrogram=OFF -DWITH_LAYER_split=ON -DWITH_LAYER_spp=OFF -DWITH_LAYER_squeeze=ON -DWITH_LAYER_statisticspooling=OFF -DWITH_LAYER_swish=ON -DWITH_LAYER_tanh=OFF -DWITH_LAYER_threshold=OFF -DWITH_LAYER_tile=OFF -DWITH_LAYER_unaryop=ON -DWITH_LAYER_unfold=OFF -DWITH_LAYER_yolodetectionoutput=OFF -DWITH_LAYER_yolov3detectionoutput=OFF 
```

Build with:
```bash
cmake --build build-size --config MinSizeRel --target ppocrv5_nocv
```

- Which result in a size of 12 MB (10MB is the model which is a compiledd buffer, and 2MB is the inference engine and image processing).  
    - For Release, the size is 12.5MB.

- Without static CRT, the size is 11.7MB for MinSizeRel, and 12.2MB for Release.  

- If we only use int8 quantized models, compiling with `-DNCNN_F16C=OFF` can also decrease binary size by 0.1MB.

#### .exe vs .lib

The sizes documented above are for an `.exe` binary.  
A compilation for `.lib` will be larger, but when we will link it with another exe\dll that will use it, the size added should be equal or less then the `.exe` size (we might need to add `/GL` flag to the `.lib` + `.dll`, and `/LTCG` flag to the `.dll`)

### Flags meanings

- `NCNN_PIXEL`: convert and resize from/to image pixel. Function to convert image format, resize, use ncnn::Mat.  
For example: from_pixels_roi_resize, yuv420sp2rgb, resize_bilinear_c1, 
 
- `NCNN_PIXEL_ROTATE`: rotate image pixel orientation. For example: kanna_rotate_c1

- `NCNN_PIXEL_AFFINE`: warp affine image pixel. For example: get_affine_transform, warpaffine_bilinear_c1. Use to caculate and apply rotation and crops.

- `NCNN_PIXEL_DRAWING`: draw basic figure and text. For example: draw_circle_c1, draw_rectangle_c1.

- `NCNN_THREADS`: build with threads. If on, uses mutexes, condition variables, thread local storage. We can't use it!

- `NCNN_C_API`: build with C API. If ON, exports C functions in c_api.h file.

- `NCNN_PLATFORM_API`: build with platform api candy. Not really sure what that means, but in the code it is only used for android includes, so probably irrelevant for us

- `NCNN_VULKAN`: vulkan compute support. Enable GPU acceleration. BUT - increase size by about 3 MB.

- `NCNN_OPENMP`: openmp support. OpenMP is an API that supports shared-memory multiprocessing programming in C, C++.  
Disabling this option means that the inference is executed on a single thread.    
If enabled, adds 0.1MB the binary.
if `NCNN_THREADS` is OFF, then using OpenMP can be unsafe, since no thread synchronization is implemented which might lead to race conditions.  
Because we can't use `NCNN_THREADS`, then we also shouldn't use `NCNN_OPENMP`.

- `NCNN_RUNTIME_CPU`: runtime dispatch cpu routines. Meaning, the implementation that matches the CPU (like AVX, FMA…) is detmined in runtime, based on the currebt CPU.  
If DCNN_RUNTIME_CPU is off, an implementation from layer_registry_arch is used (for example, from src\layer\x86\bias_x86.cpp).  
If layer_registry_arch does not have implementation for this layer, generic implementation is taken from layer_registry (for example, from src\layer\bias.cpp).
For example, if DCNN_RUNTIME_CPU and NCNN_AVX are ON, then a new implementation is generated in build\src\layer\x86\bias_x86_avx.cpp using the ncnn_add_arch_opt_layer cmake macro(it is probably the same impl as in bias_x86.cpp, just with another name and flags that enable use of the avx implementation) and is used by layer_registry_avx. When creating the layer in runtime, the function first look for an impl in layer_registry_avx, then in layer_registry_x86, and then in the generic layer_registry.
    - `NCNN_AVX`, `NCNN_AVX2` and `NCNN_AVX512` are Intel/AMD CPU instruction set extensions (SIMD) designed to speed up parallelizable tasks.   AVX512 is better than AVX2, which is better than AVX. 
    A processor that supports AVX2 is backward compatible with AVX, and a processor with AVX-512 is also backward compatible with AVX2 and AVX.  
    So supporting AVX is probably the best place in the middle between size and speed optimizations.
    - `NCNN_FMA`: FMA is and instruction set that combines multiplication and addition into one step (a\\b + c).  
    A CPU might support AVX and FMA.
    - `NCNN_XOP`: XOP is another instruction set extension. It is now deprecated, and was only used by AMD CPUs(and not Intel) 
    - There are many more optimization flags for instruction set extensions. For example, `NCNN_AVXVNNI`, `NCNN_VSX_SSE2` and many more.  
    If we won't define an optimization flag to be OFF, then it would be set to whether the compiler support this instruction set extension, and whether we enabled or diabled the "father" option of this flags(for example, if `NCNN_AVX2` is `OFF`, `NCNN_AVXVNNI` will always be `OFF`. If `NCNN_AVX2` is `ON`, and we havn't set `NCNN_AVXVNNI`, then `NCNN_AVXVNNI` will be `ON` if the compiler support it)
    - If NCNN_RUNTIME_CPU is OFF but NVNN_AVX is ON, then the Cmake adds `/arch:AVX` to compile options, and then the x86 impl is always the x86_avx impl (because `__AVX__` is defined now). This means that if we know the instruction set on our target, we don't have to use NCNN_RUNTIME_CPU. 

- `NCNN_INT8`: Support int8 inference. Adds about 0.25MB to binary.
- `NCNN_BF16`: Support B-float16 inference. Shouldn't add much to the binary, but not sure we will need it.

### Use .ncnn model as buffer

To convert onnx to .param and .bin files, use pnnx tool (pip install\binary distribution). See docs\how-to-use-and-FAQ\use-ncnn-with-pytorch-or-onnx.md

After converting to ncnn files, convert to h files:
ncnn2mem.exe PP_OCRv5_mobile_det.ncnn.param PP_OCRv5_mobile_det.ncnn.bin PP_OCRv5_mobile_det.id.h PP_OCRv5_mobile_det.mem.h

The ncnn2mem tool can be download from ncnn artifacts

See docs\how-to-use-and-FAQ\use-ncnn-with-alexnet.md for how to use the .h files.

### Disable unused operators

Use the ncnn_optimize_flags.py script to disable unneeded flags. 

## int8 quantization

See `docs\how-to-use-and-FAQ\quantized-int8-inference.md`. Both static and dynamic quantization is possible.