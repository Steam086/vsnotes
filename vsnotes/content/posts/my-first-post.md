---
date: '2025-02-11T15:03:59+08:00'
draft: true
title: 'My First Post'
---


# 相关代码
vllm
- _custom_ops.py
- distributed
    - parallel_state.py
    - device_communicators
        - custom_all_reduce.py
csrc
- custom_all_reduce.cuh
- custom_all_reduce.cu

关键细节可以直接跳转到：[Reduce Scatter](#ReduceScatter)
# 符号解释

| name              | 说明                                                                                                                                            |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| world_size（ngpus） | node中的GPU数量                                                                                                                                   |
| rank_data         | 一个分配在GPU上的指针池，其中C++中对RankData的定义是 ``struct{void * ptrs[8]}``这里取8是因为CustomAllreduce操作支持的最大GPU数量是8，这几个指针分别指向同一node上的多个GPU上的即将用于allreduce操作的输入变量 |
| "register"        | 下面函数的标识符中有register存在，register应该表示的是： 将内存中的RankData拷贝到GPU显存的rank_data池中                                                                        |
| handle和ptr        | handle是通过CUDA进程间通信（IPC）函数获取的返回值，可以传递给其他进程并在其他进程通过OpenIpcHandle打开以获取ptr指针                                                                      |

# API overview

- [create_shared_buffer](#create_shared_buffer)/free_shared_buffer ：	创建、释放共享内存（GPU上的内存）
- [capture](#capture): @contextmanager	The main responsibility of this context manager is the `register_graph_buffers` call at the end of the context.
- [register_graph_buffers](#register_graph_buffers)
- should_custom_ar	：判断是否应该使用custom_all_reduce
- [all_reduce](#all_reduce) :custom_all_reduce调用的函数，调用了cuda定义的函数
- custom_all_reduce	对外调用的接口
- [is_weak_contiguous](#is_weak_contiguous)(Tensor): 判断Tensor是否“弱连续”

# Details
## **is_weak_contiguous**

判断一个Tensor是否为“弱连续”，弱连续的条件比`torch.Tensor.is_contiguous`宽松一些，代码如下。
- definition
```
def is_weak_contiguous(inp: torch.Tensor):

return inp.is_contiguous() or (inp.storage().nbytes() -

inp.storage_offset() * inp.element_size()

== inp.numel() * inp.element_size())
```

- 如果输入张量在内存中分布“没有间隔”，而且处于整个已分配空间的最后一部分，那么`is_weak_contiguous`返回值为True，如下图所示：此时is_weak_contiguous返回值为False
![[Pasted image 20241218185931.png]]
- 如下图所示，此时`is_weak_contiguous`返回True
![[Pasted image 20241218190221.png]]
例如下面的程序输出为False，但是is_weak_contiguous输出为True，
```
import torch

# example 1 矩阵转置操作会导致is_contiguous返回False，
# 但是is_weak_contiguous返回值仍然为True
x = torch.randn(2, 2)
y = x.transpose(0, 1)
print(y.is_contiguous()) # False
print(is_weak_contiguous(y)) # True

# example 2 处于整个Tensor的后半部分，且内存连续的tensor
a = torch.randn(8)
b = a[8:16].reshape(4, 2).transpose(0, 1)
c = a[0:8].reshape(4, 2).transpose(0, 1)
print(is_contiguous(b)) # True
print(is_contiguous(c)) # False


```
这个函数会在is_contiguous()返回值为False时继续判断，如果输入的Tensor在分配的内存中处于最后一段而且在内存中连续，则返回True

 此函数的作用：
    进行进程间通信时需要使用$getMemHandle$，获取到的Handle是指向预分配内存起始地址的Handle，需要有一个offset表示输入Tensor相对这个起始地址的偏移量 

## **create_shared_buffer**

| 返回字段 |    类型     | 说明                     |
| ---- | :-------: | ---------------------- |
| ptrs | List[int] | 一个指针数组，元素个数为world_size |

创建共享内存并返回指向共享内存的指针$ptrs$,其中调用了CUDA内存分配函数，并使用OpenIpcHandle打开了其他同一node上其他设备的共享内存handle。

## **capture**:
这个函数是一个 `@contextmanager`，主要目的是在graph_capture最后调用 `register_graph_buffers`，将所有allreduce用到的输入地址注册到rank_data中。

解释：这个函数仅用于CUDA graph模式中，在CUDA graph 模式中，所有的操作不会立即被执行，CUDA会根据操作预先构建计算图，并一次性提交到GPU中执行，其中allreduce操作进行进程间通信需要将input注册到 `rank_data`中，这个注册的操作不会每次调用allreduce都执行一次，会在调用allreduce时将需要注册的ptr存入一个待注册数组（`graph_unreg_buffers_`）中，等到调用 `register_graph_buffers`时再将这些未被注册的ptr 进行 1. allgather获取其他进程中的handles。 2. 将这些获取到的handles打开并注册到 `rank_data`中

## **register_graph_buffers**

总是在capture上下文的最后调用，将capture上下文中执行的Allreduce代码中需要的input全部通过open_ipc_handle进行注册
```
@contextmanager
def capture(self):
    try:
        self._IS_CAPTURING = True
        yield
    finally:
        self._IS_CAPTURING = False
        if not self.disabled:
            self.register_graph_buffers()
```
## **all_reduce**
先进行一个条件的判断（是否处于CUDA graph 模式）如果不处于CUDA graph 模式，直接将input拷贝到预先分配的GPU buffer中，如果处于CUDA graph模式，直接input放入 `graph_unreg_buffers_`并进行allreduce操作。前面解释了这样做的原因

在C++函数内部有更细节的处理：

如果满足一些特定条件（full_nvlink_且输入Tensor比较大，在world_size<=4时的阈值为512KB，world_size<=8时的阈值为256KB），将调用 `cross_device_reduce_2stage`（CUDA核函数），否则调用 `cross_device_reduce_1stage`

`cross_device_reduce_2stage`详细解释：

- **stage 1: reduce scatter**
  首先，节点中的所有GPU只负责一部分的reduce，比如对于一个GPU的rank=rank，它负责处理 `input[start:end]` ，其中

  ```apache
  part = size / ngpus; 
  start = rank * part ; 
  end = rank == world_size - 1 ? size : start + part; 
  ```

  将这一部分reduce之后的结果放入一个预先分配的shared_memory中
- **stage 2: allgather.**
  每个GPU读取shared_memory 中的数据，并将这些数据copy到result（最终的返回结果)中。

重要代码简化版（部分同步代码省略）：

- 第一阶段

```apache
for (int idx = start + tid; idx < end; idx += stride) {
    // 将reduce结果存入保存中间结果的共享内存
    tmp_shared_buf[rank][idx] = packed_reduce(ptrs,idx);//
}
```

- 第2阶段：

```apache
// allgather操作
for (int idx = tid; idx < largest_part; idx += stride) {
    for (int i = 0; i < ngpus; i++) {
        int gather_from_rank = ((rank + i) % ngpus);
        if (gather_from_rank == ngpus - 1 || idx < part) {
            int dst_idx = gather_from_rank * part + idx;
            result[dst_idx] = tmp_shared_buf[i][idx];
        }
    }
}
```

关于第二阶段的同步操作，非常重要：
visibility across devices is only guaranteed between threads that have the same tid.
# ReduceScatter

![](https://cdn.nlark.com/yuque/0/2024/png/32583568/1734110259288-a5efb5c4-9ab3-486c-9797-80a81cfa3363.png?x-oss-process=image%2Fformat%2Cwebp)
四个设备的`reduce scatte`示意图

第一阶段： `reduce`
- 如上图所示，一个待reduce的Tensor，假设有4个设备rank1 2 3 4，在第一阶段时，每个GPU负责一个部分的reduce，如GPU1负责A区域的reduce，GPU2负责B区域以此类推。
- reduce结束后，每个GPU上的都得到了最终reduce结果的一部分（保存在临时缓冲区），将它们allgather之后就完成了全部的reduce

第二阶段： `allgather`
- 这里顺序有一定的讲究，比如rank2在allgaher操作时，读取的顺序依次是B C D A，因为rank2中已经有B的reduce结果了，rank3的顺序是 C D A B，

两个阶段中的线程（GPU线程）同步操作：

1. 最简单的同步操作：
给所有线程设置一个`barrier`，等所有GPU上的所有线程都`reduce`结束后再进行第二阶段的`allgather`操作，但是显然这样的同步操作拖累了性能，造成了不必要的等待。
2. vllm的同步操作实现：
假设一个节点内有4个GPU，需要`allreduce`的`input`张量大小为403，即
```
world_size = 4
tensor.size() = 403
```
- 那么，除了最后一个GPU，其余GPU负责的区域大小都是 403 / 4 = 100，最后一个GPU负责的区域大小是103
- 对一个进程内部，假设有10个线程，那么tid（GPU线程id）为1的就负责处理【1，11，21，...，91，101（如果有）】

在一个进程中，tid=1 的线程，在`allgather`阶段，只需要其他进程中`tid`相同的线程`reduce`的结果，所以一个线程 tid=1 的线程只需要等待其他 $tid=1,rank\in [0,ngpus]$ 线程reduce结束后就可以进行`allgather`操作。
也就是说，GPU1中的线程1要等待所有GPU中的tid为1的线程`reduce`操作结束之后才能进行第二阶段的`allgather`操作。

但是在源码中，一个allreduce操作使用的GPU中的线程数量最高达到 $36 \times 512$  个，如果为这些线程全部设置同步操作，GPU之间的开销未免有些大（$36 \times 512$ 个线程全部都要与其他GPU的线程进行通信）。所以，源码中的实现采取的操作是：让一个block中的前`ngpus`个线程去与其他GPU通信。

**省流版：**
简单来说，一个GPU中有多个线程 （36 * 512个），这些线程被分成多个block(36个)，每个block有多个thread（512个），其中线程同步只能发生在block内
所以上面的步骤可以说成是每个block派 `ngpus` 个线程与其他GPU进行同步操作，然后block内的线程再进行同步。

# QA
**Question 1：**
（这里我对源代码比较有疑问的一个点是：既然第二阶段`allgather`只有`read`操作，为什么还要将第一阶段的结果保存到临时缓冲区中而不是直接保存到最终的结果`result`中？ 直接保存到`result`中感觉不会影响第二阶段的`allgather`）

**已解决：** 
这里进行运算操作时，只有input张量和临时缓冲区是被注册到shared_buffer中的，其他进程访问不到`result`张量。如果每次进行`allreduce`都注册一个`result`变量作为共享内存，会增加进程间通信开销（在进程间传递shared_memory句柄）。

**Question 2：**
CustomAllreduce的代码执行之前会有判断当前的张量并行数是否是`[2, 4, 6, 8]`，只有这几种数量的GPU且支持nvlink而且支持p2p通信才能进行CustomAllreduce，但是阅读了所有相关代码后发现代码逻辑中并没有出现与这些数相关的逻辑，只支持`[2, 4, 6, 8]`是否和nvlink拓扑结构有关？

**未解决**



