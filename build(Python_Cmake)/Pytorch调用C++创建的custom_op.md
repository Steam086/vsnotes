
参考链接：

【1】  [PyTorch Custom Operators](https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html)
【2】 [Custom C++ and CUDA Operators](https://pytorch.org/tutorials/advanced/cpp_custom_ops.html#cpp-custom-ops-tutorial)
【3】  [The Custom Operators Manual](https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit?tab=t.0#heading=h.uaq109nj98ly)

*如果编写的函数可以用Pytorch的内置算子组合完成，就不要自定义算子了，直接编写Python函数*。
## 为什么要创建自定义算子？

出于以下两个原因，你可能希望创建自定义算子：
1. 你有一些自定义的 CPU、CUDA 或其他后端内核，并且想要将它们与 PyTorch 集成。
2. 你有一些代码，希望 PyTorch 将其视作不透明的可调用对象（当作黑箱）。

例如，你可能想要调用一些低级别的第三方库，如 LAPACK（线性代数库）或 CUBLAS（NVIDIA 的 CUDA 基本线性代数子程序库），或者你可能已经在.cu 文件中编写了大量 CUDA 内核。

你的自定义算子内核应**尽可能少地**包含 PyTorch 内置算子 。在 C++ 自定义算子中包含 PyTorch 内置算子会使其对诸如 torch.compile 之类的 PyTorch 子系统不可见，这样会隐藏优化机会。




### torch 如何调用注册好的函数

详见：[[Cmake与Python Module]]


