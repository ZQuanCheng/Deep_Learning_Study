
### 深度学习的相关库、基本常识

> 
> <div align=center>
> <img src="./images/DL.png"  style="zoom:100%"/>
> </div> 
> 
> ① NumPy 包为 Python 加上了关键的数组变量类型，弥补了 Python 的不足；
> ② Pandas 包在 NumPy 数组的基础上添加了与 Excel 类似的行列标签；
> ③ Matplotlib 库借鉴 Matlab，帮 Python 具备了绘图能力，使其如虎添翼；
> ④ Scikit-learn 库是机器学习库，内含分类、回归、聚类、降维等多种算法；
> ⑤ TensorFlow 库是 Google 公司开发的深度学习框架，于 2015 年问世；
> ⑥ PyTorch 库是 Facebook 公司开发的深度学习框架，于 2017 年问世。
> 
> 
> ⚫ 人工智能是一个很大的概念，其中一个最重要的分支就是机器学习；
> ⚫ 机器学习的算法多种多样，其中最核心的就是神经网络；
> ⚫ 神经网络的隐藏层若足够深，就被称为深层神经网络，也即深度学习；
> ⚫ 深度学习包含深度神经网络、卷积神经网络、循环神经网络等。
> 
> 

### 虚拟环境

> 
> PyTorch 为 1.12.0 
>
> Python 为 3.9
>
> NumPy 为 1.21 版本
> 
> Pandas 为 1.2.4 版本
> 
> Matplotlib 为 3.5.1 版本；
>
> 
> > ```sh
> > # 列出所有的环境
> > conda env list
> > 
> > # 进入名为“DL”的虚拟环境
> > conda activate DL
> > ```
> > 
> > 
> > 虚拟环境内的操作 (其实之前已经安装好了)
> > 
> > ```sh
> > # 列出当前环境下的所有库
> > conda list
> > 
> > # 安装 numpy 1.21.5版 
> > conda install numpy==1.21.5
> > # 查看版本
> > pip show numpy 
> > 
> > # 安装 Pandas 1.2.4 版本
> > conda install pandas==1.2.4
> > # 查看版本
> > pip show pandas
> > 
> > # 安装 Matplotlib 3.5.1 版本；
> > conda install matplotlib==3.5.1
> > # 查看版本
> > pip show matplotlib 
> > 
> > # 退出虚拟环境
> > conda deactivate
> > ```
> > 
> 
> 
> 




### 一、张量

#### 1.1 数组与张量

> 
> <font color="yellow"> `NumPy` 和 `PyTorch` 的基础语法几乎一致</font> ，具体表现为：
> 
> > * `np` 对应 `torch`；
> > * 数组 `array` 对应张量 `tensor`；
> > * `NumPy` 的 `n` 维数组对应着 `PyTorch` 的 `n` 阶张量。
> 
> <font color="yellow">数组与张量之间可以相互转换：</font>
> 
> > * 数组 `arr` 转为张量 `ts`：`ts = torch.tensor(arr)`
> > * 张量 `ts` 转为数组 `arr`：`arr = np.array(ts)`
> 
> 


#### 1.2 从数组到张量

> 
> `PyTorch` 只是少量修改了 `NumPy` 的函数或方法
> 
> <div align=center>
> <img src="./images/Tensor_1.png"  style="zoom:100%"/>
> </div> 
> 
> 



#### 1.3 用 GPU 存储张量

> Jupyter 上运行如下代码
> 
> <div align=center>
> <img src="./images/Tensor_2.png"  style="zoom:100%"/>
> </div> 
>  
> Pycharm 上运行如下代码
> 
> <div align=center>
> <img src="./images/Tensor_3.png"  style="zoom:100%"/>
> </div> 
>  
> <font color="yellow">以上操作可以把数据集搬到 `GPU` 上，但是神经网络模型也要搬到 `GPU` 上才可正常运行，使用下面的代码即可。</font>
> 
> <div align=center>
> <img src="./images/Tensor_4.png"  style="zoom:100%"/>
> </div> 
>  
> 想要查看显卡是否在运作时，在 `cmd` 中输入：`nvidia-smi`，如图所示。
> 
> <div align=center>
> <img src="./images/Tensor_5.png"  style="zoom:100%"/>
> </div> 
> 
> 
> 
> 
> 




### 二、DNN 的原理

>
> 神经网络可以分为这么几步：
> 
> 1. 划分数据集
> 2. 训练网络
> 3. 测试网络
> 4. 使用网
> 

#### 2.1 划分数据集

> 
> 
> 
> 
> 
> 
> 
> 


















