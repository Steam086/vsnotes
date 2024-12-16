description：
如何使用Pytorch调用C++中实现的自定义op
### command line
```
将cmake中定义的target编译
cmake --build <dir> [options]
将编译之后的结果安装（copy）到指定的位置
cmake --install <dir> [options] 
```
正常情况下可以直接使用cmake构建然后在Python中使用


一般情况下可以使用cmake和setuptools结合
先用cmake构建，然后再将构建好的动态库移入模块中供调用

### 直接在python中import 未注册的 .so文件

from tsmy import _C ImportError: dynamic module does not define module export function (PyInit__C)


#### 1. 方法一：直接不做额外处理，直接load_library

torch.ops.load_library('/path/to/your/library.so')

#### 2. 方法二： 定义export function，使用import语句

from . import _C


两种方法结束之后都可以使用torch.ops.namespace.xxx()调用动态库中的函数了

e.g. 

有如下的文件结构
- csrc
	- \_C.abi3.so   # 构建好的动态库
- main.py

main.py 
```
import torch
from csrc import _C

torch.ops._C.your_function()

```

```
import torch

torch.ops.load_library('./csrc/_C.abi3.so')
torch.ops._C.your_function()
```

#### vllm源码中使用使用了第二种方法

主要是通过torch_bindings.cpp 和 core/registration.h实现的

在torch_bindings.cpp 中将自定义op绑定到方法中
在registration.h 中将模块名注册，这样就可以通过import将动态库导入

例如在vllm/__custom_ops.py中,使用了
```
import vllm._C
import vllm._moe_C
```
将模块导入，以便后面调用
torch.ops._C.xxx()