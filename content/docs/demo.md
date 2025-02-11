---
date: '2025-02-11T16:36:38+08:00'
draft: true
title: 'Demo'
math: true
---
test
# 第一个Doc
12344444223123
公式展示：
$\in \sum_{10}^{100}$
$$\in K_1^2$$
Python代码展示：
```Python {filename="hello.py",linenos=table,linenostart=42} 
import numpy as np

x = np.array([100])
```

## title
- 展示0
- 展示1

表格展示：
| name              | 说明                                                                                                                                            |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| world_size（ngpus） | node中的GPU数量                                                                                                                                   |
| rank_data         | 一个分配在GPU上的指针池，其中C++中对RankData的定义是 ``struct{void * ptrs[8]}``这里取8是因为CustomAllreduce操作支持的最大GPU数量是8，这几个指针分别指向同一node上的多个GPU上的即将用于allreduce操作的输入变量 |
| "register"        | 下面函数的标识符中有register存在，register应该表示的是： 将内存中的RankData拷贝到GPU显存的rank_data池中                                                                        |
| handle和ptr        | handle是通过CUDA进程间通信（IPC）函数获取的返回值，可以传递给其他进程并在其他进程通过OpenIpcHandle打开以获取ptr指针                                                                      |
