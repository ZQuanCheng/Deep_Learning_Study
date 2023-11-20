
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







### 一、CNN 的原理

#### 1.1 从 DNN 到 CNN

> 
> <font color="pink"> 1. 卷积层与汇聚 </font>
> 
> > * 深度神经网络`DNN`中，相邻层的所有神经元之间都有连接，这叫全连接；
> > 卷积神经网络 `CNN` 中，新增了`卷积层（Convolution）`与`汇聚（Pooling）`。
> > 
> > * `DNN` 的全连接层对应 `CNN` 的卷积层，汇聚是与激活函数类似的附件；
> > <font color="gree">单个卷积层的结构是：卷积层-激活函数-(汇聚)，其中汇聚可省略。</font>
> > 
> > 
> > 
> 
> 
> <font color="pink"> 2. CNN：专攻多维数据 </font>
> 
> > 在深度神经网络 `DNN` 课程的最后一章，使用 `DNN` 进行了手写数字的识别。但是，图像至少就有二维，向全连接层输入时，需要多维数据拉平为 `1` 维数据，这样一来，图像的形状就被忽视了，很多特征是隐藏在空间属性里的，如图 1-1。
> > 
> > <div align=center>
> > <img src="./images/CNN_1.png"  style="zoom:100%"/>
> > </div> 
> >  
> > 而卷积层可以保持输入数据的维数不变，当输入数据是二维图像时，卷积层会以多维数据的形式接收输入数据，并同样以多维数据的形式输出至下一层，如图 1-2 所示。
> > 
> > <div align=center>
> > <img src="./images/CNN_2.png"  style="zoom:100%"/>
> > </div> 
> >   
> > 
> 
> 
> 



#### 1.2 卷积层

> 
> `CNN` 中的`卷积层`与 `DNN` 中的`全连接层`是平级关系，全连接层中的权重与偏置即 $y = \omega_1 x_1 + \omega_2 x_2 + \omega_3 x_3 + b$ 中的 $\omega$ 与 $b$，卷积层中的权重与偏置变得稍微复杂。
> 
> <font color="pink"> 1. 内部参数：权重（卷积核）</font>
> 
> > 当输入数据进入卷积层后，输入数据会与卷积核进行卷积运算，如图 1-3。
> > 
> > <div align=center>
> > <img src="./images/CNN_3.png"  style="zoom:100%"/>
> > </div> 
> >    
> > 图 1-3 中，输入大小是`(4, 4)`，卷积核大小是`(3, 3)`，输出大小是`(2, 2)`。卷积运算的原理是逐元素乘积后再相加，如图 1-4 所示。
> > 
> > <div align=center>
> > <img src="./images/CNN_4.png"  style="zoom:100%"/>
> > </div> 
> > 
> > 
> 
> 
> <font color="pink"> 2. 内部参数：偏置 </font>
> 
> > 在卷积运算的过程中也存在偏置，如图 1-5 所示。
> > 
> > <div align=center>
> > <img src="./images/CNN_5.png"  style="zoom:100%"/>
> > </div> 
> > 
> 
> 
> <font color="pink"> 3. 外部参数：填充 </font>
> 
> > 为了防止经过多个卷积层后图像越卷越小，可以在进行卷积层的处理之前，向输入数据的周围填入固定的数据（比如 `0`），这称为`填充（padding）`。
> > 
> > <div align=center>
> > <img src="./images/CNN_6.png"  style="zoom:100%"/>
> > </div> 
> > 
> > 图 1-6 中，对大小为`(4, 4)`的输入数据应用了幅度为 `1` 的填充，填充值为 `0`。
> > 
> 
> 
> <font color="pink"> 4. 外部参数：步幅 </font>
> 
> > 使用卷积核的位置间隔被称为`步幅（stride）`，之前的例子中步幅都是 `1`，如果将步幅设为 `2`，则如图 1-7 所示，此时使用卷积核的窗口的间隔变为 `2`。
> > 
> > <div align=center>
> > <img src="./images/CNN_7.png"  style="zoom:100%"/>
> > </div> 
> > 
> 
> <font color="gree"> 综上，增大填充后，输出尺寸会变大；而增大步幅后，输出尺寸会变小。 </font>
>
> 
> <font color="pink"> 5. 输入与输出尺寸的关系 </font>
> 
> > 假设输入尺寸为`(H, W)`，卷积核的尺寸为`(FH, FW)`，填充为 `P`，步幅为 `S`。则输出尺寸`(OH, OW)`的计算公式为
> > 
> > <div align=center>
> > <img src="./images/CNN_8.png"  style="zoom:100%"/>
> > </div> 
> > 
> 
> 
> 
> 



#### 1.3 多通道

> 
> 在上一小节讲的卷积层，仅仅针对二维的输入与输出数据（一般是灰度图像），可称之为`单通道`。
> 
> <font color="gree"> 但是，彩色图像除了`高（Height）`、`长（Width）`两个维度之外，还有第三个维度：`通道（channel）`。 </font>
> 
> 例如，以 `RGB` 三原色为基础的彩色图像，其通道方向就有红、黄、蓝三部分，可视为 3 个单通道二维图像的混合叠加。
> 
> <font color="gree">一般的，当输入数据是二维时，权重被称为`卷积核（Kernel）`；当输入数据是三维或更高时，权重被称为`滤波器（Filter）`。 </font>
> 
> 
> <font color="pink"> 1. 多通道输入 </font>
> 
> > 
> > 
> > 
> > 
> > 
> > 
> > 
> > 
> > 
> 
> 
> 
> 
> 
> 
> 









