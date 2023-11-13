
#### 安装本科毕业设计的虚拟环境

> 
> 
> ```sh
> # 列出所有的环境
> conda env list
> 
> # 创建名为“tf2”的虚拟环境，并指定 Python 的版本
> conda create -n tf2 python=3.6
> 
> # 查看是否创建成功
> conda env list
> 
> # 进入名为“tf2”的虚拟环境
> conda activate tf2
> ```
> 
> > 
> > 虚拟环境内的操作
> > 
> > ```sh
> > # 列出当前环境下的所有库
> > conda list
> > 
> > # 安装tensorflow2.0.0 CPU版 正式版本
> > conda install tensorflow==2.0.0  #conda install tensorflow-gpu==2.0.0是GPU版正式版本
> > 
> > # 列出当前环境下的所有库
> > conda list
> > 
> > # 验证tensorflow是否安装成功
> > python
> > import tensorflow as tf
> > print(tf.__version__)  # 输出 2.0.0
> > print(tf.test.is_gpu_available()) # 输出False，因为我们安装的是CPU版本，不支持GPU
> > exit()
> >  
> >  
> >  
> > # 退出虚拟环境
> > conda deactivate
> > ```
> > 
> 
> 
> 
>
> 
>
> 
>
> 
>
> 

































