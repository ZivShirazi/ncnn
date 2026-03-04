- `NCNN_PIXEL`: convert and resize from/to image pixel. Function to convert image format, resize, use ncnn::Mat.  
For example: from_pixels_roi_resize, yuv420sp2rgb, resize_bilinear_c1, 
 
- `NCNN_PIXEL_ROTATE`: rotate image pixel orientation. For example: kanna_rotate_c1

- `NCNN_PIXEL_AFFINE`: warp affine image pixel. For example: get_affine_transform, warpaffine_bilinear_c1. Use to caculate and apply rotation and crops.

- `NCNN_PIXEL_DRAWING`: draw basic figure and text. For example: draw_circle_c1, draw_rectangle_c1.

- `NCNN_THREADS`: build with threads. If on, uses mutexes, condition variables, thread local storage.

- `NCNN_C_API`: build with C API. If ON, exports C functions in c_api.h file.

- `NCNN_PLATFORM_API`: build with platform api candy. Not really sure what that means, but in the code it is only used for android includes, so probably irrelevant for us

- `NCNN_VULKAN`: vulkan compute support. Enable GPU acceleration. BUT - increase size by about 3 MB.

- `NCNN_OPENMP`: openmp support. OpenMP is an API that supports shared-memory multiprocessing programming in C, C++.  
Disabling this option means that the inference is executed on a single thread.  
Still not sure why the NCNN_THREADS option exists if it doesn't seems to be related to using OpenMP for multithreading (when using #pragma omp parallel num_thread)


# 
# 
#   
# 
# If NCNN_RUNTIME_CPU is OFF but NVNN_AVX is ON, then the Cmake adds  /arch:AVX to compile options, and then the x86 impl is always the x86_avx impl (because __AVX__ is defined now). This means that if we know the instruction set on our target, we don't have to use NCNN_RUNTIME_CPU. But I believe it is a small diference
- `NCNN_RUNTIME_CPU`: runtime dispatch cpu routines. Meaning, the implementation that matches the CPU (like AVX, FMA…) is detmined in runtime, based on the currebt CPU.  
If DCNN_RUNTIME_CPU is off, an implementation from layer_registry_arch is used (for example, from src\layer\x86\bias_x86.cpp).  
If layer_registry_arch does not have implementation for this layer, generic implementation is taken from layer_registry (for example, from src\layer\bias.cpp).
For example, if DCNN_RUNTIME_CPU and NCNN_AVX are ON, then a new implementation is generated in build\src\layer\x86\bias_x86_avx.cpp using the ncnn_add_arch_opt_layer cmake macro(it is probably the same impl as in bias_x86.cpp, just with another name and flags that enable use of the avx implementation) and is used by layer_registry_avx. When creating the layer in runtime, the function first look for an impl in layer_registry_avx, then in layer_registry_x86, and then in the generic layer_registry.
    - `NCNN_AVX`, `NCNN_AVX2` and `NCNN_AVX512` are Intel/AMD CPU instruction set extensions (SIMD) designed to speed up parallelizable tasks.   AVX512 is better than AVX2, which is better than AVX. 
    A processor that supports AVX2 is backward compatible with AVX, and a processor with AVX-512 is also backward compatible with AVX2 and AVX.  
    So supporting AVX is probably the best place in the middle between size and speed optimizations.
    - There are many more optimization flags. For example, `NCNN_AVXVNNI`, `NCNN_VSX_SSE2` and many more.  
    If we won't define an optimization flag to be OFF, then it would be set to whether the compiler support this instruction set extension, and whether we enabled or diabled the "father" option of this flags(for example, if `NCNN_AVX2` is `OFF`, `NCNN_AVXVNNI` will always be `OFF`. If `NCNN_AVX2` is `ON`, and we havn't set `NCNN_AVXVNNI`, then `NCNN_AVXVNNI` will be `ON` if the compiler support it)

&nbsp;-D=OFF 

NCNN_AVX2=OFF

-DNCNN_AVX512=OFF

\# FMA is and instruction set that combines multiplication and addition into one step (a\\b + c).

\# A CPU might support AVX and FMA

&nbsp;-DNCNN\_FMA=OFF

\# XOP is another instruction set extension that is now deprecated, was only used by AMD cpus(and not Intel) 

&nbsp;-DNCNN\_XOP=OFF 



\# Support int8 inference.

\# Adds about 0.25MB to binary

-DNCNN\_INT8=OFF

\# Support B-float16 inference.

\# Shouldn't add much to the binary, but not sure we will need it

&nbsp;-DNCNN\_BF16=OFF
```


For a main compilation with ncnn::Net only,

This is 1.3MB build

Without flags, it is 1.6MB (same for without flags + ENABLE\_LTO=FALSE)



For the full ppocrv5\_nocv.cpp example (without the models), 

This is 1.6MB

for without flags + ENABLE\_LTO=FALSE it is 2MB (if -DNCNN\_BUILD\_WITH\_STATIC\_CRT=OFF, it is 1.7MB. Which means that if we link with a library that is already statically linked with CRT, then we will add between 1.7 to 2 MB to this library) 



Support in int8 adds 0.25MB





------------------------



&nbsp;cmake -S . -B build-size -A x64 -DCMAKE\_BUILD\_TYPE=Release -DNCNN\_SHARED\_LIB=OFF -DNCNN\_BUILD\_WITH\_STATIC\_CRT=ON -DNCNN\_BUILD\_EXAMPLES=ON -DNCNN\_BUILD\_TOOLS=OFF -DNCNN\_BUILD\_BENCHMARK=OFF -DNCNN\_BUILD\_TESTS=OFF -DNCNN\_VULKAN=OFF -DNCNN\_OPENMP=OFF -DNCNN\_THREADS=OFF -DNCNN\_RUNTIME\_CPU=ON -DNCNN\_INT8=ON -DNCNN\_BF16=OFF -DNCNN\_C\_API=OFF -DNCNN\_PLATFORM\_API=OFF -DNCNN\_PIXEL=ON -DNCNN\_PIXEL\_ROTATE=ON -DNCNN\_PIXEL\_AFFINE=ON -DNCNN\_PIXEL\_DRAWING=OFF -DNCNN\_AVX=ON -DNCNN\_AVX2=OFF -DNCNN\_FMA=OFF -DNCNN\_XOP=OFF -DNCNN\_AVX512=OFF



The size is 3.34MB for Release

2.71MB for MinSizeRel



If we know for sure that all CPUs we will run on support AVX, we can disable NCNN\_RUNTIME\_CPU, and then the MinSizeRel binary is 2.2MB



------------------------------------

To convert onnx to .param and .bin files, use pnnx tool (pip install\binary distribution). See docs\how-to-use-and-FAQ\use-ncnn-with-pytorch-or-onnx.md

After converting to ncnn files, convert to h files:
ncnn2mem.exe PP_OCRv5_mobile_det.ncnn.param PP_OCRv5_mobile_det.ncnn.bin PP_OCRv5_mobile_det.id.h PP_OCRv5_mobile_det.mem.h

The ncnn2mem tool can be download from ncnn artifacts

See docs\how-to-use-and-FAQ\use-ncnn-with-alexnet.md for how to use the .h files.

----------------

Use the ncnn_optimize_flags.py script to disable unneeded flags. For the example OCR, we will get:

```bash
 cmake -S . -B build-size -A x64 -DCMAKE_BUILD_TYPE=Release -DNCNN_F16C=ON -DNCNN_INT8=ON -DNCNN_BF16=OFF -DNCNN_SIMPLEVK=OFF -DNCNN_SIMPLESTL=OFF -DNCNN_SHARED_LIB=OFF -DNCNN_BUILD_WITH_STATIC_CRT=ON -DNCNN_ENABLE_LTO=OFF -DNCNN_BUILD_EXAMPLES=ON -DNCNN_BUILD_TOOLS=OFF -DNCNN_BUILD_BENCHMARK=OFF -DNCNN_BUILD_TESTS=OFF -DNCNN_VULKAN=OFF -DNCNN_OPENMP=OFF -DNCNN_THREADS=OFF -DNCNN_RUNTIME_CPU=ON -DNCNN_C_API=OFF -DNCNN_PLATFORM_API=OFF-DNCNN_PIXEL=ON -DNCNN_PIXEL_ROTATE=ON -DNCNN_PIXEL_AFFINE=ON -DNCNN_PIXEL_DRAWING=OFF -DNCNN_AVX=ON -DNCNN_AVX2=OFF -DNCNN_FMA=OFF -DNCNN_XOP=OFF -DNCNN_AVX512=OFF -DNCNN_DISABLE_RTTI=ON -DNCNN_STRING=OFF -DNCNN_STDIO=OFF -DWITH_LAYER_absval=OFF -DWITH_LAYER_argmax=OFF -DWITH_LAYER_batchnorm=OFF -DWITH_LAYER_bias=OFF -DWITH_LAYER_binaryop=ON -DWITH_LAYER_bnll=OFF -DWITH_LAYER_cast=ON -DWITH_LAYER_celu=OFF -DWITH_LAYER_clip=ON -DWITH_LAYER_concat=ON -DWITH_LAYER_convolution=ON -DWITH_LAYER_convolution1d=OFF -DWITH_LAYER_convolution3d=OFF -DWITH_LAYER_convolutiondepthwise=ON -DWITH_LAYER_convolutiondepthwise1d=OFF -DWITH_LAYER_convolutiondepthwise3d=OFF -DWITH_LAYER_copyto=OFF -DWITH_LAYER_crop=OFF -DWITH_LAYER_cumulativesum=OFF -DWITH_LAYER_deconvolution=ON -DWITH_LAYER_deconvolution1d=OFF -DWITH_LAYER_deconvolution3d=OFF -DWITH_LAYER_deconvolutiondepthwise=OFF -DWITH_LAYER_deconvolutiondepthwise1d=OFF -DWITH_LAYER_deconvolutiondepthwise3d=OFF -DWITH_LAYER_deepcopy=OFF -DWITH_LAYER_deformableconv2d=OFF -DWITH_LAYER_dequantize=OFF -DWITH_LAYER_detectionoutput=OFF -DWITH_LAYER_diag=OFF -DWITH_LAYER_dropout=OFF -DWITH_LAYER_einsum=OFF -DWITH_LAYER_eltwise=OFF -DWITH_LAYER_elu=OFF -DWITH_LAYER_embed=OFF -DWITH_LAYER_erf=OFF -DWITH_LAYER_exp=OFF -DWITH_LAYER_expanddims=OFF -DWITH_LAYER_flatten=ON -DWITH_LAYER_flip=OFF -DWITH_LAYER_fold=OFF -DWITH_LAYER_gelu=OFF -DWITH_LAYER_gemm=ON -DWITH_LAYER_glu=OFF -DWITH_LAYER_gridsample=OFF -DWITH_LAYER_groupnorm=OFF -DWITH_LAYER_gru=OFF -DWITH_LAYER_hardsigmoid=ON -DWITH_LAYER_hardswish=ON -DWITH_LAYER_innerproduct=ON -DWITH_LAYER_input=ON -DWITH_LAYER_instancenorm=OFF -DWITH_LAYER_interp=ON -DWITH_LAYER_inversespectrogram=OFF -DWITH_LAYER_layernorm=ON -DWITH_LAYER_log=OFF -DWITH_LAYER_lrn=OFF -DWITH_LAYER_lstm=OFF -DWITH_LAYER_matmul=OFF -DWITH_LAYER_memorydata=OFF -DWITH_LAYER_mish=OFF -DWITH_LAYER_multiheadattention=ON -DWITH_LAYER_mvn=OFF -DWITH_LAYER_noop=ON -DWITH_LAYER_normalize=OFF -DWITH_LAYER_packing=ON -DWITH_LAYER_padding=ON -DWITH_LAYER_permute=ON -DWITH_LAYER_pixelshuffle=OFF -DWITH_LAYER_pooling=ON -DWITH_LAYER_pooling1d=OFF -DWITH_LAYER_pooling3d=OFF -DWITH_LAYER_power=OFF -DWITH_LAYER_prelu=OFF -DWITH_LAYER_priorbox=OFF -DWITH_LAYER_proposal=OFF -DWITH_LAYER_psroipooling=OFF -DWITH_LAYER_quantize=OFF -DWITH_LAYER_reduction=OFF -DWITH_LAYER_relu=ON -DWITH_LAYER_reorg=OFF -DWITH_LAYER_requantize=OFF -DWITH_LAYER_reshape=ON -DWITH_LAYER_rmsnorm=OFF -DWITH_LAYER_rnn=OFF -DWITH_LAYER_roialign=OFF -DWITH_LAYER_roipooling=OFF -DWITH_LAYER_rotaryembed=OFF -DWITH_LAYER_scale=ON -DWITH_LAYER_sdpa=OFF -DWITH_LAYER_selu=OFF -DWITH_LAYER_shrink=OFF -DWITH_LAYER_shufflechannel=OFF -DWITH_LAYER_sigmoid=ON -DWITH_LAYER_slice=OFF -DWITH_LAYER_softmax=ON -DWITH_LAYER_softplus=OFF -DWITH_LAYER_spectrogram=OFF -DWITH_LAYER_split=ON -DWITH_LAYER_spp=OFF -DWITH_LAYER_squeeze=ON -DWITH_LAYER_statisticspooling=OFF -DWITH_LAYER_swish=ON -DWITH_LAYER_tanh=OFF -DWITH_LAYER_threshold=OFF -DWITH_LAYER_tile=OFF -DWITH_LAYER_unaryop=ON -DWITH_LAYER_unfold=OFF -DWITH_LAYER_yolodetectionoutput=OFF -DWITH_LAYER_yolov3detectionoutput=OFF 
```
Build with:
```bash
cmake --build build-size --config MinSizeRel --target ppocrv5_nocv
```

which result in a size of 12 MB (10MB is the model, 2 is the inference engine and image processing)
For Release, the size is 12.5MB,
For without static CRT, the size is 11.7MB for MinSizeRel, and 12.2MB for Release.
If we only use int8 quantized models, compiling with `-DNCNN_F16C=OFF` can also decrease binary size by 0.1MB.
