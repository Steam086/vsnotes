vllm尝试使用了许多使用NCCL的方式，但是都没有很好的解决问题，最后使用Python脚本写了一个wrapper直接绑定NCCL的动态库。

vllm中处理allreduce的优先级是：
- CustomAllreduce
- Pynccl
- torch.distributed

torch.distributed在最后被考虑，因为：`torch.distributed.all_reduce` 包含许多其他潜在的 CUDA API，这些 API 在捕获 CUDA 图时是不允许的。 



## pynccl_wrapper.py中的解释

这个文件是一个 **纯 Python 封装** 的 NCCL 库。  
其主要目的是将 NCCL 与 **CUDA 图（CUDA graph）** 结合使用。

在编写此脚本之前，我们尝试了以下方法：

1. **尝试使用 `cupy`**：  
    虽然 `cupy` 能够正确调用 NCCL，但它自身在初始化 NCCL 通信器时经常会出现卡住的问题。
    
2. **尝试使用 `torch.distributed`**：  
    但是 `torch.distributed.all_reduce` 包含许多其他潜在的 CUDA API，这些 API 在捕获 CUDA 图时是不允许的。  
    详细信息请参考链接：  
    [https://discuss.pytorch.org/t/pytorch-cudagraph-with-nccl-operation-failed/](https://discuss.pytorch.org/t/pytorch-cudagraph-with-nccl-operation-failed/)
    

---

### 另一个被否决的方案

我们也考虑过为 NCCL 编写一个 **C/C++ 绑定**，这种方案通常是可行的，  
但我们经常会遇到与 NCCL 版本相关的问题，并且需要在不同版本的 NCCL 之间切换。  
详细信息请参考链接：  
[https://github.com/NVIDIA/nccl/issues/1234](https://github.com/NVIDIA/nccl/issues/1234)

C/C++ 绑定不够灵活，因为每次切换 NCCL 版本时都需要重新编译代码。  
相比之下，这个当前实现的 **纯 Python 封装** 更加灵活。  
我们可以通过更改环境变量 `VLLM_NCCL_SO_PATH`，或者代码中的 `so_file` 变量，轻松地在不同版本的 NCCL 之间进行切换。