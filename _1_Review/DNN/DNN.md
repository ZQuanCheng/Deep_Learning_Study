
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
> <font color="yellow">数据集里每个样本必须包含输入与输出，将数据集按一定的比例划分为训练集与测试集，分别用于训练网络与测试网络</font>
> 
> <div align=center>
> <img src="./images/DNN_1.png"  style="zoom:100%"/>
> </div> 
> 
> 考虑到数据集的输入特征与输出特征都是 `3` 列，因此神经网络的输入层与输出层也必须都是 `3` 个神经元，隐藏层可以自行设计，如图所示
> 
> <div align=center>
> <img src="./images/DNN_2.png"  style="zoom:100%"/>
> </div> 
> 
> 考虑到 `Python 列表`、`NumPy 数组`以及 `PyTorch 张量`都是从`索引[0]`开始，再加之输入层没有内部参数（权重 ω 与偏置 b），所以习惯将`输入层`称之为`第 0 层`。
>
> 所以上图中的神经网络只有4层，不包含输入层
> 
> 


#### 2.2 训练网络

> `训练 = 拟合复杂函数`
> 
> 神经网络的训练过程，就是经过很多次前向传播与反向传播的轮回，最终不断调整其`内部参数`（权重 ω 与偏置 b），以拟合任意复杂函数的过程。内部参数一开始是随机的（如 Xavier 初始值、He 初始值），最终会不断优化到最佳。
>
> 还有一些训练网络前就要设好的`外部参数`：网络的层数、每个隐藏层的节点数、每个节点的激活函数类型、学习率、轮回次数、每次轮回的样本数等等。
> 
> <font color="gree">业界习惯把内部参数称为参数，外部参数称为超参数。</font>
> 
> <font color="pink">（1）前向传播 </font>
>
> > 将单个样本的 3 个输入特征送入神经网络的输入层后，神经网络会逐层计算到输出层，最终得到神经网络预测的 3 个输出特征。
> > 
> > 计算过程中所使用的参数就是内部参数，所有的隐藏层与输出层的神经元都有内部参数，以第 1 层的第 1 个神经元，如图所示。
> > 
> > <div align=center>
> > <img src="./images/DNN_3.png"  style="zoom:100%"/>
> > </div> 
> > 
> > 该神经元节点的计算过程为 $y = \omega_1 x_1 + \omega_2 x_2 + \omega_3 x_3 + b$ 。
> > 
> > 你可以理解为，每一根线就是一个权重 $\omega$，每一个神经元节点也都有它自己的偏置 $b$。
> > 
> > 当然，每个神经元节点在计算完后，由于这个方程是线性的，因此必须在外面套一个非线性的函数：$y = \sigma \left(  \omega_1 x_1 + \omega_2 x_2 + \omega_3 x_3 + b  \right)$
> > 
> > $\sigma$ 被称为激活函数。
> > 
> > 如果你不套非线性函数，那么即使 10 层的网络，也可以用 1 层就拟合出同样的方程，如下图所示
> > 
> > <div align=center>
> > <img src="./images/DNN_4.png"  style="zoom:100%"/>
> > </div> 
> > 
> > 
>
> 
> 
> <font color="pink">（2）反向传播 </font>
>
> > 经过前向传播，网络会根据当前的内部参数计算出输出特征的预测值。但是这个预测值与真实值直接肯定有差距，因此需要一个损失函数来计算这个差距。例如，求预测值与真实值之间差的绝对值，就是一个典型的损失函数。
> >  
> > <font color="yellow">损失函数计算好后，逐层退回求梯度，这个过程很复杂，原理不必掌握，大致意思就是，看每一个内部参数是变大还是变小，才会使得损失函数变小。这样就达到了优化内部参数的目的。</font>
> > 
> > <font color="gree">在这个过程中，有一个`外部参数`叫`学习率`。学习率越大，内部参数的优化越快，但过大的学习率可能会使损失函数越过最低点，并在谷底反复横跳。因此，在网络的训练开始之前，选择一个合适的学习率很重要。</font>
> > 
> > 
> 
> <font color="pink">（3）batch_size </font>
>
> > 前向传播与反向传播一次时，有三种情况：
> > 
> > > * `批量梯度下降`（Batch Gradient Descent，`BGD`），把所有样本一次性输入进网络，这种方式计算量开销很大，速度也很慢。
> > > 
> > > * `随机梯度下降`（Stochastic Gradient Descent，`SGD`），每次只把一个样本输入进网络，每计算一个样本就更新参数。这种方式虽然速度比较快，但是收敛性能差，可能会在最优点附近震荡，两次参数的更新也有可能抵消。
> > > 
> > > * `小批量梯度下降`（Mini-Batch Gradient Decent，`MBGD`）是为了中和上面二者而生，这种办法把样本划分为若干个批，按批来更新参数。
> > 
> > 所以，`batch_size` 即一批中的样本数，也是一次喂进网络的样本数。此外，由于 `Batch Normalization` 层（用于将每次产生的小批量样本进行标准化）的存在，`batch_size` 一般设置为 `2` 的幂次方，并且不能为 `1`。
> > 
> > <font color="yellow">注：PyTorch 实现时只支持批量`BGD`与小批量`MBGD`，不支持单个样本的输入方式。`PyTorch` 里的 `torch.optim.SGD` 只表示梯度下降
> > 
> > <font color="gree">批量与小批量见第四、五章。</font>
> > 
> > 
> 
> <font color="pink">（4）epochs </font>
>
> > `1` 个 `epoch` 就是指全部样本进行 `1` 次前向传播与反向传播。
>
> 
> <font color="pink">（5）假设有 `10240` 个训练样本，`batch_size` 是 `1024`，`epochs` 是 `5`。那么： </font>
> 
> > * 全部样本将进行 `5` 次前向传播与反向传播；
> > * `1` 个 `epoch`，将发生 `10` 次（ `10240 ÷ 1024` ）前向传播与反向传播；
> > * 一共发生 `50` 次（`10 × 5`）前向传播和反向传播
> 
> 
> 
> 
> 




#### 2.3 测试网络

> 
> 为了防止训练的网络过拟合，因此需要拿出少量的样本进行测试。过拟合的意思是：网络优化好的内部参数只能对训练样本有效，换成其它就寄。以线性回归为例，过拟合如图 2-3（b）所示。
> > 
> > <div align=center>
> > <img src="./images/DNN_5.png"  style="zoom:100%"/>
> > </div> 
> > 
> 
> 当网络训练好后，拿出测试集的输入，进行 1 次前向传播后，将预测的输出与测试集的真实输出进行对比，查看准确率。
> 
> 



#### 2.4 使用网络

> 
> 真正使用网络进行预测时，样本只知输入，不知输出。直接将样本的输入进行 `1` 次前向传播，即可得到预测的输出
> 
> 




### 三、DNN 的实现


> 
> <font color="pink">`torch.nn` 提供了搭建网络所需的所有组件，`nn` 即 `Neural Network` 神经网络。因此，可以单独给 `torch.nn` 一个别名，即 `import torch.nn as nn`。</font>
> > 
> > <div align=center>
> > <img src="./images/PyTorch_DNN_1.png"  style="zoom:100%"/>
> > </div> 
> > 
> > <font color="yellow">含有`inline`的行属于`Jupyter`独有,使用`Pycharm`请删除`inline`行</font>
> > 
> > <div align=center>
> > <img src="./images/PyTorch_DNN_2.png"  style="zoom:100%"/>
> > </div> 
> > 
> > 
> > `Jupyter Notebook` 代码如下：
> > 
> > ```python
> > import torch
> > import torch.nn as nn
> > import matplotlib.pyplot as plt
> > %matplotlib inline 
> > 
> > # 展示高清图
> > from matplotlib_inline import backend_inline
> > backend_inline.set_matplotlib_formats('svg') # 使用svg格式显示绘图
> > # 指定matplotlib输出svg图表以获得更清晰的图像
> > ```
> > 
> > 
> > `Pycharm` 代码如下：
> > 
> > ```python
> > import torch
> > import torch.nn as nn
> > import matplotlib.pyplot as plt
> > # %matplotlib inline # Jupyter才需要
> > 
> > # 展示高清图
> > from matplotlib_inline import backend_inline
> > 
> > if __name__ == '__main__':
> >     # 展示高清图
> >     # 之前已导入库from matplotlib_inline import backend_inline
> >     backend_inline.set_matplotlib_formats('svg') # 使用svg格式显示绘图
> >     # 指定matplotlib输出svg图表以获得更清晰的图像
> > ```
> > 
> > <font color="gree">上面的代码语句是固定的，记住即可</font>
>
> > 注：
> > 
> > 使用`Jupyter Notebook`，需要先运行 `Jupyter Notebook` 的魔术命令`%matplotlib inline`。这条命令的作用是将 `Matplotlib` 绘制的图形嵌入在当前页面中。而在桌面环境中绘图时，不需要添加此命令，而是在全部绘图代码之后追加`plt.show()`
> > 
> 




#### 3.1 制作数据集

> 
> <font color="pink">在训练之前，要准备好训练集的样本。</font>
> 
> > 这里生成 `10000` 个样本，设定 `3` 个输入特征与 `3` 个输出特征，其中
> > 
> > * 每个输入特征相互独立，均服从均匀分布；
> > * 当 `(X1+X2+X3)< 1` 时，`Y1` 为 `1`，否则 `Y1` 为 `0`；
> > * 当 `1<(X1+X2+X3)<2` 时，`Y2` 为 `1`，否则 `Y2` 为 `0`；
> > * 当 `(X1+X2+X3)>2` 时，`Y3` 为 `1`，否则 `Y3` 为 `0`；
> > * `.float()`将布尔型张量`bool tensor`转化为浮点型张量`float tensor`。
> > 
> > 
> > 
> > `Jupyter Notebook` 代码如下：
> > 
> > ```python
> > # 生成数据集
> > X1 = torch.rand(10000,1) # 输入特征 1
> > X2 = torch.rand(10000,1) # 输入特征 2
> > X3 = torch.rand(10000,1) # 输入特征 3
> > Y1 = ( (X1+X2+X3)<1 ).float() # 输出特征 1
> > Y2 = ( (1<(X1+X2+X3)) & ((X1+X2+X3)<2) ).float() # 输出特征 2
> > Y3 = ( (X1+X2+X3)>2 ).float() # 输出特征 3
> > Data = torch.cat([X1,X2,X3,Y1,Y2,Y3],axis=1) # 整合数据集
> > Data = Data.to('cuda:0') # 把数据集搬到 GPU 上
> > Data.shape
> > ```
> > 
> > 
> > `Pycharm` 代码如下：
> > 
> > ```python
> > 
> > if __name__ == '__main__':
> > 
> >     ...
> > 
> >     # 生成数据集
> >     X1 = torch.rand(10000, 1)  # 输入特征 1
> >     X2 = torch.rand(10000, 1)  # 输入特征 2
> >     X3 = torch.rand(10000, 1)  # 输入特征 3
> >     print(X1.shape, X2.shape, X3.shape)  # torch.Size([10000, 1]) torch.Size([10000, 1]) torch.Size([10000, 1])
> >     
> >     Y1 = ((X1+X2+X3) < 1).float()  # 输出特征 1
> >     Y2 = ((1 < (X1+X2+X3)) & ((X1+X2+X3) < 2)).float()  # 输出特征 2
> >     Y3 = ((X1+X2+X3) > 2).float()  # 输出特征 3
> >     print(Y1.shape, Y2.shape, Y3.shape)  # torch.Size([10000, 1]) torch.Size([10000, 1]) torch.Size([10000, 1])
> >     
> >     Data = torch.cat([X1, X2, X3, Y1, Y2, Y3], axis=1)  # 整合数据集; cat就是numpy中的concatenate
> >     print(Data.type())  # torch.FloatTensor
> >     
> >     Data = Data.to('cuda:0')  # 把数据集搬到 GPU 上
> >     print(Data.type())  # torch.cuda.FloatTensor
> >     print(Data.shape)  # torch.Size([10000, 6])
> > ```
> > 
> 
> <font color="pink">然后划分训练集与测试集</font>
> 
> > <font color="yellow">以下代码属于通用型代码，便于我们手动分割训练集与测试集</font>
> > 
> > 
> > `Jupyter Notebook` 代码如下：
> > 
> > ```python
> > # 划分训练集与测试集
> > train_size = int(len(Data) * 0.7) # 训练集的样本数量
> > test_size = len(Data) - train_size # 测试集的样本数量
> > Data = Data[torch.randperm( Data.size(0)) , : ] # 打乱样本的顺序
> > train_Data = Data[ : train_size , : ] # 训练集样本
> > test_Data = Data[ train_size : , : ] # 测试集样本
> > train_Data.shape, test_Data.shape
> > ```
> > 
> > 
> > `Pycharm` 代码如下：
> > 
> > ```python
> > 
> > if __name__ == '__main__':
> > 
> >     ...
> > 
> >     # 划分训练集与测试集
> >     train_size = int(len(Data) * 0.7)  # 训练集的样本数量
> >     test_size = len(Data) - train_size  # 测试集的样本数量
> >     Data = Data[torch.randperm(Data.size(0)), :]  # 打乱样本的顺序
> >     train_Data = Data[:train_size, :]  # 训练集样本
> >     test_Data = Data[train_size:, :]  # 测试集样本
> >     print(train_Data.shape, test_Data.shape)  # torch.Size([7000, 6]) torch.Size([3000, 6])
> > ```
> > 
> 
> 
> 
> 
> <font color="pink">注：`One-Hot` 编码</font>
> 
> > 事实上，数据的 `3` 个输出特征组合起来是一个 `One-Hot 编码`（独热编码）。
> > 
> > 举例说明
> > 
> > 假如有三种颜色特征：红、黄、蓝。 在利用机器学习的算法时一般需要进行向量化或者数字化。那么你可能想令 `红=1，黄=2，蓝=3`. 那么这样其实实现了标签编码，即给不同类别以标签。然而这意味着机器可能会学习到 “红 < 黄 < 蓝”，但这并不是我们的让机器学习的本意，只是想让机器区分它们，并无大小比较之意。所以这时标签编码是不够的，需要进一步转换。因为有三种颜色状态，所以就有`3 bit`。即 `红色：1 0 0 ，黄色: 0 1 0，蓝色：0 0 1` 。如此一来每两个向量之间的距离都是 $\sqrt{2}$，在向量空间距离都相等，所以这样不会出现偏序性，基本不会影响基于向量空间度量算法的效果
> > 
> > `One-Hot` 编码是一种在机器学习和数据处理中常用的技术，特别是在处理分类数据时。这种方法将分类变量转换为一种形式，使得可以用二进制来表示每一个类别。
> > 
> > `One-Hot` 编码的基本原理和步骤如下：
> > 
> > 1. **确定类别数量**：首先，确定你的数据中有多少个不同的类别。例如，如果你有一个包含“苹果”，“香蕉”，和“橘子”的水果类别数据，那么你有三个类别。
> > 
> > 2. **创建二进制特征**：对于每一个类别，创建一个新的二进制特征（列）。每个特征对应一个类别。在我们的例子中，你会创建三个新特征：一个代表“苹果”，一个代表“香蕉”，一个代表“橘子”。
> > 
> > 3. **编码**：对于数据集中的每个实例，根据它属于哪个类别，在相应的特征列上标记为1，其余列标记为0。例如，如果一个数据实例是“香蕉”，那么“香蕉”列将被标记为1，而“苹果”和“橘子”列将被标记为0。
> > 
> > 4. **结果**：最终，每个实例都将通过一系列的0和1来表示，这些0和1标识了它属于哪个类别。
> > 
> > One-Hot 编码的主要优点是它可以使模型更容易处理和理解分类数据。然而，当类别数量非常多时，这种编码方法可能导致数据维度的显著增加，这种情况下可能需要考虑其他编码技术，如嵌入式编码。
> > 
> > 
> 
> 
> 
> <font color="pink">可以直接看数据</font>
> 
> > <div align=center>
> > <img src="./images/PyTorch_DNN_3.png"  style="zoom:100%"/>
> > </div> 
> 
> 
> 





#### 3.2 搭建神经网络

> 
> <font color="pink">搭建神经网络时，以 `nn.Module` 作为父类，我们自己的神经网络可直接继承父类的方法与属性，`nn.Module` 中包含网络各个层的定义。</font>
> 
> 在定义的神经网络子类中，通常包含 `__init__` 特殊方法与 `forward` 方法。
> 
> > `__init__` 特殊方法用于构造自己的神经网络结构，
> > 
> > `forward` 方法用于将输入数据进行前向传播。由于张量可以自动计算梯度，所以不需要出现反向传播方法。
> > 
> > 
> > 
> > `Jupyter Notebook` 代码块（2个）如下：
> > 
> > ```python
> > # 代码块1
> > class DNN(nn.Module):
> >     def __init__(self):
> >         ''' 搭建神经网络各层 '''
> >         super(DNN,self).__init__()
> >         self.net = nn.Sequential( # 按顺序搭建各层
> >             nn.Linear(3, 5), nn.ReLU(), # 第 1 层：全连接层
> >             nn.Linear(5, 5), nn.ReLU(), # 第 2 层：全连接层
> >             nn.Linear(5, 5), nn.ReLU(), # 第 3 层：全连接层
> >             nn.Linear(5, 3) # 第 4 层：全连接层
> >         )
> >     def forward(self, x):
> >         ''' 前向传播 '''
> >         y = self.net(x) # x 即输入数据
> >         return y # y 即输出数据
> > ```
> > ```python
> > # 代码块2
> > model = DNN().to('cuda:0') # 创建子类的实例，并搬到 GPU 上
> > model # 查看该实例的各层
> > ```
> > 
> > 
> > `Pycharm` 代码如下：
> > 
> > ```python
> > class DNN(nn.Module):
> >     def __init__(self):
> >         ''' 搭建神经网络各层 '''
> >         super(DNN, self).__init__()
> >         self.net = nn.Sequential(    # 按顺序搭建各层
> >             nn.Linear(3, 5), nn.ReLU(),  # 第 1 层：全连接层
> >             nn.Linear(5, 5), nn.ReLU(),  # 第 2 层：全连接层
> >             nn.Linear(5, 5), nn.ReLU(),  # 第 3 层：全连接层
> >             nn.Linear(5, 3)              # 第 4 层：全连接层
> >         )
> > 
> >     def forward(self, x):
> >         ''' 前向传播 '''
> >         y = self.net(x)  # x 即输入数据, 这里的net和__init__()中的net要一致，自己起名
> >         return y         # y 即输出数据
> > 
> > 
> > 
> > if __name__ == '__main__':
> > 
> >     ...
> > 
> >     model = DNN().to('cuda:0')  # 创建子类的实例，并搬到 GPU 上
> >     print(model)                # 查看该实例的各层
> > ```
> > 
> > 
> > <div align=center>
> > <img src="./images/PyTorch_DNN_4.png"  style="zoom:100%"/>
> > </div> 
> > 
> > 在上面的 `nn.Sequential()` 函数中，每一个隐藏层后都使用了 `RuLU` 激活函数，各层的神经元节点个数分别是：`3、5、5、5、3`
> > 
> > 注意，输入层有 `3` 个神经元、输出层有 `3` 个神经元，这不是巧合，是有意而为之。输入层的神经元数量必须与每个样本的输入特征数量一致，输出层的神经元数量必须与每个样本的输出特征数量一致
> > 
>
> 
> <font color="pink">注：各种库函数的意义</font>
> 
> >  `super(Net, self).__init__()`
> > 
> > > 参考： https://www.cnblogs.com/lusiqi/p/17202800.html
> > > 
> > > `super()` 函数是用来调用父类（超类）的一个方法。这里的 `Net` 是当前类的名称，`self` 是对当前实例的引用。
> > > 
> > > 子类 `class Net` 继承父类 `class nn.Module`, `super(Net,self).__init__()` 就是对继承自父类 `nn.Module` 的属性进行初始化。并且是用 `nn.Module` 的初始化方法来初始化继承的属性。
> > > 
> > > 也就是：用父类的方法初始化子类的属性。
> > > 
> > > 为啥要用父类的方法去初始化属性呢？原因很简单：因为父类的方法已经写好了，我们只需要调用就可以了。
> > > 
> > > `python` 中创建类后，通常会创建一个 `__init__()` 方法，这个方法会在创建类的实例的时候自动执行。类似于C++中的构造函数。
> > 
> > `nn.Sequential()`
> > 
> > > 
> > > `nn.Sequential()` 用于将多个模块按顺序封装成一个单独的模块。
> > > 
> > > 理解 `nn.Sequential()` 的关键点如下：
> > > 
> > > 1. **简化模型定义**：通常，当你使用 `PyTorch` 创建一个神经网络时，你需要定义一个 `forward` 方法，该方法明确指定了数据在网络中的传播路径。使用 `nn.Sequential()`，你可以避免编写自己的 `forward` 方法，因为 `nn.Sequential` 会自动按照添加模块的顺序来处理数据流。
> > > 
> > > 2. **按顺序排列层和函数**：在 `nn.Sequential()` 容器中，你可以按顺序添加网络层、激活函数等组件。当输入数据传递给一个 `nn.Sequential` 模型时，数据会依次经过这些层和函数。
> > > 
> > > 3. **易于修改和扩展**：如果你的网络结构是一系列顺序操作，使用 `nn.Sequential()` 会使得修改层和其他元素变得更加简单，因为你只需要添加或删除 `nn.Sequential()` 中的元素即可。
> > > 
> > > 
> > 
> > 
> > `nn.Linear()`
> > 
> > > `nn.Linear(3, 5)` 实际上是 `nn.Linear(in_features=3, out_features=5)`
> > > 
> > > 它用于创建一个线性变换层（也称为全连接层或稠密层）, 将数据从 3 维映射到 5 维。
> > > 
> > > 理解 `nn.Linear` 的关键点如下：
> > > 
> > > 
> > > 
> > > 
> 
> 
> 



#### 3.3 网络的内部参数

> 
> <font color="pink">神经网络的内部参数是权重与偏置，内部参数在神经网络训练之前会被赋予随机数，随着训练的进行，内部参数会逐渐迭代至最佳值。</font>
>  
> 现对参数进行查看
> > 
> > `Jupyter Notebook` 代码块如下：
> > 
> > ```python
> > # 查看内部参数（非必要）
> > for name, param in model.named_parameters():
> >     print(f"参数:{name}\n 形状:{param.shape}\n 数值:{param}\n")
> > ```
> > 
> > 
> > 
> > `Pycharm` 代码如下：
> > 
> > ```python
> > 
> >     ...
> > 
> > if __name__ == '__main__':
> > 
> >     ...
> > 
> >     # 查看内部参数（非必要）
> >     for name, param in model.named_parameters():
> >         print(f"参数:{name}\n 形状:{param.shape}\n 数值:{param}\n")
> > ```
> > 
> > 
> > 代码一共给了我们 `8` 个参数，其中参数与形状的结果如表 3-1 所示，考虑到其数值初始状态时是随机的（如 Xavier 初始值、He 初始值），此处不讨论。
> > 
> > <div align=center>
> > <img src="./images/PyTorch_DNN_5.png"  style="zoom:100%"/>
> > </div> 
> > 
> > `Jupyter` 的 `Out [7]` 如下
> > 
> > ```c
> > DNN(
> >   (net): Sequential(
> >     (0): Linear(in_features=3, out_features=5, bias=True)
> >     (1): ReLU()
> >     (2): Linear(in_features=5, out_features=5, bias=True)
> >     (3): ReLU()
> >     (4): Linear(in_features=5, out_features=5, bias=True)
> >     (5): ReLU()
> >     (6): Linear(in_features=5, out_features=3, bias=True)
> >   )
> > )
> > ```
> > 
> > <font color="gree">可见，具有权重与偏置的地方只有 `net.0、net.2、net.4、net.6`，结合 `Out [7]` 的结果，可知这几个地方其实就是所有的隐藏层与输出层，这符合理论。</font>
> > 
> > * 首先，`net.0.weight` 的权重形状为`[5, 3]`，`5` 表示它自己的节点数是 `5`，`3` 表
示与之连接的前一层的节点数为 `3`。
> > * 其次，由于 `In [3]` 里进行了 `model = DNN().to('cuda:0')` 操作，因此所有的内部参数都自带 `device='cuda:0'`。
> > * 最后，注意到 `requires_grad=True`，说明所有需要进行反向传播的内部参数（即权重与偏置）都打开了张量自带的梯度计算功能。
> > 
> > <div align=center>
> > <img src="./images/DNN_2.png"  style="zoom:100%"/>
> > </div> 
> > 
> 
> 
> 



#### 3.4 网络的外部参数（调参关注点）

> 
> <font color="pink">外部参数即超参数，这是调参师们关注的重点。</font>
> 
> <font color="gree">搭建网络时的超参数有：</font>
> 
> > * 网络的层数
> > * 各隐藏层节点数
> > * 各节点激活函数
> > * 内部参数的初始值
> 
> <font color="gree">训练网络的超参数有：</font>
> 
> > * 损失函数
> > * 学习率
> > * 优化算法
> > * batch_size
> > * epochs
> 
> 
> 
> <font color="pink">（1）激活函数</font>
> 
> > `PyTorch 1.12.0` 版本进入 https://pytorch.org/docs/1.12/nn.html 搜索 `Non-linear 
Activations`，即可查看 `torch` 内置的所有非线性激活函数（以及各种类型的层）
> > 
> > https://pytorch.org/docs/1.12/nn.html#non-linear-activations-weighted-sum-nonlinearity
> > 
> 
> <font color="pink">（2）损失函数</font>
> 
> > 进入 https://pytorch.org/docs/1.12/nn.html 搜索 `Loss Functions`，即可查看 `torch`内置的所有损失函数。
> > 
> > https://pytorch.org/docs/1.12/nn.html#loss-functions
> > 
> > ```python
> > # 损失函数的选择
> > loss_fn = nn.MSELoss()
> > ```
>
> 
> <font color="pink">（3）学习率与优化算法</font>
> 
> > 进入 https://pytorch.org/docs/1.12/optim.html ，可查看 `torch` 的所有优化算法。
> > 
> > 
> > ```python
> > # 优化算法的选择
> > learning_rate = 0.01 # 设置学习率
> > optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) # 除了SGD还有Adam
> > ```
> > 
> > 注：`PyTorch` 实现时只支持 `BGD` 或 `MBGD`，不支持单个样本的输入方式。
> > 
> > <font color="yellow">这里的 `torch.optim.SGD` 只表示梯度下降（并不是`随机梯度下降，Stochastic Gradient Descent，SGD`），具体的批量与小批量见第四、五章。</font>
> > 
> > 
> 
> 
> 


#### 3.5 训练网络

>
> 
> <font color="pink"> 开始训练网络 </font>
> 
> > <font color="gree">代码如下</font>
> > 
> > `Jupyter Notebook` 代码块如下：
> > 
> > ```python
> > # 训练网络
> > epochs = 1000
> > losses = []  # 记录损失函数变化的列表
> > 
> > # 给训练集划分输入与输出
> > X = train_Data[:, :3]  # 前 3 列为输入特征
> > Y = train_Data[:, -3:]  # 后 3 列为输出特征
> > 
> > # 固定的套路，进行训练
> > for epoch in range(epochs):
> >     Pred = model(X)  # 一次前向传播（批量，BGD，把所有样本一次性输入进网络）
> >     loss = loss_fn(Pred, Y)  # 计算损失函数
> >     losses.append(loss.item())  # 记录损失函数的变化
> >     optimizer.zero_grad()  # 清理上一轮滞留的梯度
> >     loss.backward()  # 一次反向传播
> >     optimizer.step()  # 优化内部参数
> > 
> > # 图像展示loss的变化
> > Fig = plt.figure()
> > plt.plot(range(epochs), losses)
> > plt.ylabel('loss'), plt.xlabel('epoch')
> > plt.show()
> > ```
> > 
> > 
> > `Pycharm` 代码如下：
> > 
> > ```python
> > 
> >     ...
> > 
> > if __name__ == '__main__':
> > 
> >     ...
> > 
> >    # 训练网络
> >    epochs = 1000
> >    losses = []  # 记录损失函数变化的列表
> >    
> >    # 给训练集划分输入与输出
> >    X = train_Data[:, :3]  # 前 3 列为输入特征
> >    Y = train_Data[:, -3:]  # 后 3 列为输出特征
> >    
> >    # 固定的套路，进行训练
> >    for epoch in range(epochs):
> >        Pred = model(X)  # 一次前向传播（批量，BGD，把所有样本一次性输入进网络）
> >        loss = loss_fn(Pred, Y)  # 计算损失函数
> >        losses.append(loss.item())  # 记录损失函数的变化
> >        optimizer.zero_grad()  # 清理上一轮滞留的梯度
> >        loss.backward()  # 一次反向传播
> >        optimizer.step()  # 优化内部参数
> > 
> >    # 图像展示loss的变化
> >    Fig = plt.figure()
> >    plt.plot(range(epochs), losses)
> >    plt.ylabel('loss'), plt.xlabel('epoch')
> >    plt.show()
> > ```
> 
> > 
> > <div align=center>
> > <img src="./images/PyTorch_DNN_6.png"  style="zoom:100%"/>
> > </div> 
> > 
>
> 
> <font color="gree"> 注： </font>
> 
> > `.item()`方法可将 `PyTorch` 张量退化为普通元素。作用：取出单元素张量的元素值并返回该值，保持原元素类型不变。,即：原张量元素为整形，则返回整形，原张量元素为浮点型则返回浮点型，etc.
> > 
> > `losses.append(loss.item())`中，`.append()`是指在列表 `losses` 后再附加 `1` 个元素
> 
> > 
> 



#### 3.6 测试网络

>
> <font color="pink"> 测试时，只需让测试集进行 1 次前向传播即可</font>
>
> > 这个过程不需要计算梯度，因此可以在该局部关闭梯度，该操作使用 `with torch.no_grad():` 命令。 
> > 
> > 
> > 考虑到输出特征是`one-hot`独热编码，而`预测的结果数据`一般都是`接近 0 或 1 的小数`，为了能让预测数据与真实数据之间进行比较，因此要对预测数据进行规整, 使得预测数据也呈现出`one-hot`格式。
> > 
> > 例如，使用 `Pred[:,torch.argmax(Pred, axis=1)] = 1` 命令将每行最大的数置 `1`，接着再使用
`Pred[Pred!=1] = 0` 将不是 `1` 的数字置 `0`，这就使预测数据与真实数据的格式相同。
> > 
> > ```python
> > 前向传播输出结果：[0.9573, 0.5782, 0.134, 0.7368, 0.9034]
> > 进行规整后的输出：[1, 0, 0, 0, 0]
> > ```
> > 
> > <font color="gree">代码如下</font>
> > 
> > `Jupyter Notebook` 代码块如下：
> > 
> > ```python
> > # 测试网络
> > # 给测试集划分输入与输出
> > X = test_Data[:, :3]  # 前 3 列为输入特征
> > Y = test_Data[:, -3:]  # 后 3 列为输出特征
> > with torch.no_grad():  # 该局部关闭梯度计算功能, 下面几行代码将不再计算梯度，节约内存，运算速度更快
> >     Pred = model(X)  # 一次前向传播（批量）
> >     Pred[:, torch.argmax(Pred, axis=1)] = 1
> >     Pred[Pred != 1] = 0
> >     correct = torch.sum((Pred == Y).all(1))  # 预测正确的样本
> >     total = Y.size(0)  # 全部的样本数量
> >     print(f'测试集精准度: {100*correct/total} %')
> > ```
> > 
> > 
> > `Pycharm` 代码如下：
> > 
> > ```python
> > 
> >     ...
> > 
> > if __name__ == '__main__':
> > 
> >     ...
> > 
> >     # 测试网络
> >     # 给测试集划分输入与输出
> >     X = test_Data[:, :3]  # 前 3 列为输入特征
> >     Y = test_Data[:, -3:]  # 后 3 列为输出特征
> >     with torch.no_grad():  # 该局部关闭梯度计算功能, 下面几行代码将不再计算梯度，节约内存，运算速度更快
> >         Pred = model(X)  # 一次前向传播（批量）
> >         Pred[:, torch.argmax(Pred, axis=1)] = 1
> >         Pred[Pred != 1] = 0
> >         correct = torch.sum((Pred == Y).all(1))  # 预测正确的样本
> >         total = Y.size(0)  # 全部的样本数量
> >         print(f'测试集精准度: {100*correct/total} %')
> > ```
>
> 
> > 
> > <div align=center>
> > <img src="./images/PyTorch_DNN_7.png"  style="zoom:100%"/>
> > </div> 
> > 
> 
>
> <font color="gree"> 注： </font>
> 
> > 在计算 `correct` 时需要动点脑筋。
> > 
> > 首先，`(Pred == Y)`计算预测的输出与真实的输出的各个元素是否相等，返回一个 `3000` 行、`3` 列的布尔型张量。
> > 
> > 其次，`(Pred == Y).all(1)`检验该布尔型张量每一行的 `3` 个数据是否都是 `True`，对于全是 `True` 的样本行，结果就是 `True`，否则是 `False`。`all(1)`中的 `1` 表示按“行”扫描，最终返回一个形状为 `3000` 的`一阶张量`。
> > 
> > 最后，`torch.sum( (Pred == Y).all(1) )`的意思就是看这 `3000` 个向量相加，`True`会被当作 `1`，False 会被当作 `0`，这样相加刚好就是预测正确的样本数
> > 
> 
> 
> 


#### 3.7 保存与导入网络

>
> <font color="pink"> 现在我们要考虑一件大事，那就是有时候训练一个大网络需要几天，那么必
须要把整个网络连同里面的优化好的内部参数给保存下来。 </font>
>
> 现以本章前面的代码为例，当网络训练好后，将网络以文件的形式保存下来，并通过文件导入给另一个新网络，让新网络去跑测试集，看看测试集的准确率是否也是 `67%`。
> 
> 
> <font color="pink">（1）保存网络 </font>
> 
> > 通过“`torch.save(模型名, '文件名.pth')`”命令，可将该模型完整的保存至 `Jupyter` 的工作路径下。
> > 
> > `Jupyter Notebook` 代码如下：
> > 
> > ```python 
> > # 保存网络
> > torch.save(model, 'model.pth')
> > ```
> > 
> > <div align=center>
> > <img src="./images/PyTorch_DNN_8.png"  style="zoom:100%"/>
> > </div> 
> > 
> > <font color="gree">那么 `Pycharm` 呢？</font>
> > 
> > > <font color="yellow">其实代码是一样的，只是会遇到问题</font>
> > > 
> > > 参考博客：https://blog.csdn.net/qq_37209590/article/details/122699573
> > > 
> > > `main.py` 文件如下
> > > 
> > > ```python
> > > import torch
> > > import torch.nn as nn
> > > import matplotlib.pyplot as plt
> > > from matplotlib_inline import backend_inline # 展示高清图
> > > 
> > > class DNN(nn.Module):
> > >     def __init__(self):
> > >         ''' 搭建神经网络各层 '''
> > >         super(DNN, self).__init__()
> > >         self.net = nn.Sequential(    # 按顺序搭建各层
> > >             nn.Linear(3, 5), nn.ReLU(),  # 第 1 层：全连接层
> > >             nn.Linear(5, 5), nn.ReLU(),  # 第 2 层：全连接层
> > >             nn.Linear(5, 5), nn.ReLU(),  # 第 3 层：全连接层
> > >             nn.Linear(5, 3)              # 第 4 层：全连接层
> > >         )
> > > 
> > >     def forward(self, x):
> > >         ''' 前向传播 '''
> > >         y = self.net(x)  # x 即输入数据
> > >         return y         # y 即输出数据
> > > 
> > > 
> > > 
> > > if __name__ == '__main__':
> > >     # 展示高清图
> > >     # 之前已导入库from matplotlib_inline import backend_inline
> > >     backend_inline.set_matplotlib_formats('svg')
> > > 
> > >     # 生成数据集
> > >     X1 = torch.rand(10000, 1)  # 输入特征 1
> > >     X2 = torch.rand(10000, 1)  # 输入特征 2
> > >     X3 = torch.rand(10000, 1)  # 输入特征 3
> > >     print(X1.shape, X2.shape, X3.shape)  # torch.Size([10000, 1]) torch.Size([10000, 1]) torch.Size([10000, 1])
> > > 
> > >     Y1 = ((X1 + X2 + X3) < 1).float()  # 输出特征 1
> > >     Y2 = ((1 < (X1 + X2 + X3)) & ((X1 + X2 + X3) < 2)).float()  # 输出特征 2
> > >     Y3 = ((X1 + X2 + X3) > 2).float()  # 输出特征 3
> > >     print(Y1.shape, Y2.shape, Y3.shape)  # torch.Size([10000, 1]) torch.Size([10000, 1]) torch.Size([10000, 1])
> > > 
> > >     Data = torch.cat([X1, X2, X3, Y1, Y2, Y3], axis=1)  # 整合数据集; cat就是numpy中的concatenate
> > >     print(Data.type())  # torch.FloatTensor
> > > 
> > >     Data = Data.to('cuda:0')  # 把数据集搬到 GPU 上
> > >     print(Data.type())  # torch.cuda.FloatTensor
> > >     print(Data.shape)  # torch.Size([10000, 6])
> > > 
> > >     # 划分训练集与测试集
> > >     train_size = int(len(Data) * 0.7)  # 训练集的样本数量
> > >     test_size = len(Data) - train_size  # 测试集的样本数量
> > >     Data = Data[torch.randperm(Data.size(0)), :]  # 打乱样本的顺序
> > >     train_Data = Data[:train_size, :]  # 训练集样本
> > >     test_Data = Data[train_size:, :]  # 测试集样本
> > >     print(train_Data.shape, test_Data.shape)  # torch.Size([7000, 6]) torch.Size([3000, 6])
> > > 
> > >     # print(Data)
> > > 
> > >     model = DNN().to('cuda:0')  # 创建子类的实例，并搬到 GPU 上
> > >     print(model)                # 查看该实例的各层
> > > 
> > >     # 查看内部参数（非必要）
> > >     for name, param in model.named_parameters():
> > >         print(f"参数:{name}\n 形状:{param.shape}\n 数值:{param}\n")
> > > 
> > >     # 损失函数的选择
> > >     loss_fn = nn.MSELoss()
> > > 
> > >     # 优化算法的选择
> > >     learning_rate = 0.01  # 设置学习率
> > >     optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
> > > 
> > >     # 训练网络
> > >     epochs = 1000
> > >     losses = []  # 记录损失函数变化的列表
> > >     # 给训练集划分输入与输出
> > >     X = train_Data[:, :3]  # 前 3 列为输入特征
> > >     Y = train_Data[:, -3:]  # 后 3 列为输出特征
> > >     for epoch in range(epochs):
> > >         Pred = model(X)  # 一次前向传播（批量，BGD，把所有样本一次性输入进网络）
> > >         loss = loss_fn(Pred, Y)  # 计算损失函数
> > >         losses.append(loss.item())  # 记录损失函数的变化
> > >         optimizer.zero_grad()  # 清理上一轮滞留的梯度
> > >         loss.backward()  # 一次反向传播
> > >         optimizer.step()  # 优化内部参数
> > > 
> > >     Fig = plt.figure()
> > >     plt.plot(range(epochs), losses)
> > >     plt.ylabel('loss'), plt.xlabel('epoch')
> > >     plt.show()
> > > 
> > > 
> > >     # 测试网络
> > >     # 给测试集划分输入与输出
> > >     X = test_Data[:, :3]  # 前 3 列为输入特征
> > >     Y = test_Data[:, -3:]  # 后 3 列为输出特征
> > >     with torch.no_grad():  # 该局部关闭梯度计算功能, 下面几行代码将不再计算梯度，节约内存，运算速度更快
> > >         Pred = model(X)  # 一次前向传播（批量）
> > >         Pred[:, torch.argmax(Pred, axis=1)] = 1
> > >         Pred[Pred != 1] = 0
> > >         correct = torch.sum((Pred == Y).all(1))  # 预测正确的样本
> > >         total = Y.size(0)  # 全部的样本数量
> > >         print(f'测试集精准度: {100*correct/total} %')
> > > 
> > >     # 保存网络
> > >     torch.save(model, 'model.pth')
> > > ```
> > > 
> > > 报错：`_pickle.PicklingError: Can't pickle <class '__main__.DNN'>: attribute lookup DNN on __main__ failed`
> > > 
> > > 报错信息：意思是在`main`里面找不到`DNN`
> > > 
> > > 问题定位：因为我是在同一个文件`main.py`里定义的网络结构(`class DNN`)，在后面的`if __name__ == "__main__"`里面使用，故出现上述报错。
> > > 
> > > 解决办法：将`class DNN`部分代码剪切粘贴到一个单独的文件`DNN_Model.py`中，在`main.py`中`import`即可解决问题。
> > > 
> > > 
> > > `DNN_Model.py` 文件如下
> > > 
> > > ```python
> > > import torch.nn as nn
> > > 
> > > class DNN(nn.Module):
> > >     def __init__(self):
> > >         ''' 搭建神经网络各层 '''
> > >         super(DNN, self).__init__()
> > >         self.net = nn.Sequential(    # 按顺序搭建各层
> > >             nn.Linear(3, 5), nn.ReLU(),  # 第 1 层：全连接层
> > >             nn.Linear(5, 5), nn.ReLU(),  # 第 2 层：全连接层
> > >             nn.Linear(5, 5), nn.ReLU(),  # 第 3 层：全连接层
> > >             nn.Linear(5, 3)              # 第 4 层：全连接层
> > >         )
> > > 
> > >     def forward(self, x):
> > >         ''' 前向传播 '''
> > >         y = self.net(x)  # x 即输入数据
> > >         return y         # y 即输出数据
> > > ```
> > > 
> > > `main.py` 文件如下
> > > 
> > > ```python
> > > import torch
> > > import torch.nn as nn
> > > import matplotlib.pyplot as plt
> > > from matplotlib_inline import backend_inline # 展示高清图
> > > 
> > > from DNN_Model import DNN
> > > 
> > > if __name__ == '__main__':
> > >     # 展示高清图
> > >     # 之前已导入库from matplotlib_inline import backend_inline
> > >     backend_inline.set_matplotlib_formats('svg')
> > > 
> > >     # 生成数据集
> > >     X1 = torch.rand(10000, 1)  # 输入特征 1
> > >     X2 = torch.rand(10000, 1)  # 输入特征 2
> > >     X3 = torch.rand(10000, 1)  # 输入特征 3
> > >     print(X1.shape, X2.shape, X3.shape)  # torch.Size([10000, 1]) torch.Size([10000, 1]) torch.Size([10000, 1])
> > > 
> > >     Y1 = ((X1 + X2 + X3) < 1).float()  # 输出特征 1
> > >     Y2 = ((1 < (X1 + X2 + X3)) & ((X1 + X2 + X3) < 2)).float()  # 输出特征 2
> > >     Y3 = ((X1 + X2 + X3) > 2).float()  # 输出特征 3
> > >     print(Y1.shape, Y2.shape, Y3.shape)  # torch.Size([10000, 1]) torch.Size([10000, 1]) torch.Size([10000, 1])
> > > 
> > >     Data = torch.cat([X1, X2, X3, Y1, Y2, Y3], axis=1)  # 整合数据集; cat就是numpy中的concatenate
> > >     print(Data.type())  # torch.FloatTensor
> > > 
> > >     Data = Data.to('cuda:0')  # 把数据集搬到 GPU 上
> > >     print(Data.type())  # torch.cuda.FloatTensor
> > >     print(Data.shape)  # torch.Size([10000, 6])
> > > 
> > >     # 划分训练集与测试集
> > >     train_size = int(len(Data) * 0.7)  # 训练集的样本数量
> > >     test_size = len(Data) - train_size  # 测试集的样本数量
> > >     Data = Data[torch.randperm(Data.size(0)), :]  # 打乱样本的顺序
> > >     train_Data = Data[:train_size, :]  # 训练集样本
> > >     test_Data = Data[train_size:, :]  # 测试集样本
> > >     print(train_Data.shape, test_Data.shape)  # torch.Size([7000, 6]) torch.Size([3000, 6])
> > > 
> > >     # print(Data)
> > > 
> > >     model = DNN().to('cuda:0')  # 创建子类的实例，并搬到 GPU 上
> > >     print(model)                # 查看该实例的各层
> > > 
> > >     # 查看内部参数（非必要）
> > >     for name, param in model.named_parameters():
> > >         print(f"参数:{name}\n 形状:{param.shape}\n 数值:{param}\n")
> > > 
> > >     # 损失函数的选择
> > >     loss_fn = nn.MSELoss()
> > > 
> > >     # 优化算法的选择
> > >     learning_rate = 0.01  # 设置学习率
> > >     optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
> > > 
> > >     # 训练网络
> > >     epochs = 1000
> > >     losses = []  # 记录损失函数变化的列表
> > >     # 给训练集划分输入与输出
> > >     X = train_Data[:, :3]  # 前 3 列为输入特征
> > >     Y = train_Data[:, -3:]  # 后 3 列为输出特征
> > >     for epoch in range(epochs):
> > >         Pred = model(X)  # 一次前向传播（批量，BGD，把所有样本一次性输入进网络）
> > >         loss = loss_fn(Pred, Y)  # 计算损失函数
> > >         losses.append(loss.item())  # 记录损失函数的变化
> > >         optimizer.zero_grad()  # 清理上一轮滞留的梯度
> > >         loss.backward()  # 一次反向传播
> > >         optimizer.step()  # 优化内部参数
> > > 
> > >     Fig = plt.figure()
> > >     plt.plot(range(epochs), losses)
> > >     plt.ylabel('loss'), plt.xlabel('epoch')
> > >     plt.show()
> > > 
> > > 
> > >     # 测试网络
> > >     # 给测试集划分输入与输出
> > >     X = test_Data[:, :3]  # 前 3 列为输入特征
> > >     Y = test_Data[:, -3:]  # 后 3 列为输出特征
> > >     with torch.no_grad():  # 该局部关闭梯度计算功能, 下面几行代码将不再计算梯度，节约内存，运算速度更快
> > >         Pred = model(X)  # 一次前向传播（批量）
> > >         Pred[:, torch.argmax(Pred, axis=1)] = 1
> > >         Pred[Pred != 1] = 0
> > >         correct = torch.sum((Pred == Y).all(1))  # 预测正确的样本
> > >         total = Y.size(0)  # 全部的样本数量
> > >         print(f'测试集精准度: {100*correct/total} %')
> > > 
> > >     # 保存网络
> > >     torch.save(model, 'model.pth')
> > > ```
> > > 
> > > 
> 
> 
> <font color="pink">（2）导入网络 </font>
> 
> > 通过“`新网络 = torch.load('文件名.pth ')`”命令，可将该模型完整的导入给新网络。
> > 
> > 代码如下：
> > 
> > ```python 
> > # 把模型赋给新网络
> > new_model = torch.load('model.pth')
> > ```
> 
> <font color="pink">（3）用新模型进行测试 </font>
> 
> > 
> > `Pycharm load_model.py` 文件如下
> > 
> > ```python
> > import torch
> > import torch.nn as nn
> > import matplotlib.pyplot as plt
> > from matplotlib_inline import backend_inline # 展示高清图
> > 
> > 
> > if __name__ == '__main__':
> >     # 展示高清图
> >     # 之前已导入库from matplotlib_inline import backend_inline
> >     backend_inline.set_matplotlib_formats('svg')
> > 
> >     # 生成数据集
> >     X1 = torch.rand(10000, 1)  # 输入特征 1
> >     X2 = torch.rand(10000, 1)  # 输入特征 2
> >     X3 = torch.rand(10000, 1)  # 输入特征 3
> >     print(X1.shape, X2.shape, X3.shape)  # torch.Size([10000, 1]) torch.Size([10000, 1]) torch.Size([10000, 1])
> > 
> >     Y1 = ((X1 + X2 + X3) < 1).float()  # 输出特征 1
> >     Y2 = ((1 < (X1 + X2 + X3)) & ((X1 + X2 + X3) < 2)).float()  # 输出特征 2
> >     Y3 = ((X1 + X2 + X3) > 2).float()  # 输出特征 3
> >     print(Y1.shape, Y2.shape, Y3.shape)  # torch.Size([10000, 1]) torch.Size([10000, 1]) torch.Size([10000, 1])
> > 
> >     Data = torch.cat([X1, X2, X3, Y1, Y2, Y3], axis=1)  # 整合数据集; cat就是numpy中的concatenate
> >     print(Data.type())  # torch.FloatTensor
> > 
> >     Data = Data.to('cuda:0')  # 把数据集搬到 GPU 上
> >     print(Data.type())  # torch.cuda.FloatTensor
> >     print(Data.shape)  # torch.Size([10000, 6])
> > 
> >     # 划分训练集与测试集
> >     train_size = int(len(Data) * 0.7)  # 训练集的样本数量
> >     test_size = len(Data) - train_size  # 测试集的样本数量
> >     Data = Data[torch.randperm(Data.size(0)), :]  # 打乱样本的顺序
> >     train_Data = Data[:train_size, :]  # 训练集样本
> >     test_Data = Data[train_size:, :]  # 测试集样本
> >     print(train_Data.shape, test_Data.shape)  # torch.Size([7000, 6]) torch.Size([3000, 6])
> > 
> >     # print(Data)
> > 
> > 
> >     # 把模型赋给新网络
> >     new_model = torch.load('model.pth')
> > 
> >     # 测试网络
> >     # 给测试集划分输入与输出
> >     X = test_Data[:, :3]  # 前 3 列为输入特征
> >     Y = test_Data[:, -3:]  # 后 3 列为输出特征
> >     with torch.no_grad():  # 该局部关闭梯度计算功能
> >         Pred = new_model(X)  # 用新模型进行一次前向传播
> >         Pred[:, torch.argmax(Pred, axis=1)] = 1
> >         Pred[Pred != 1] = 0
> >         correct = torch.sum((Pred == Y).all(1))  # 预测正确的样本
> >         total = Y.size(0)  # 全部的样本数量
> >         print(f'测试集精准度: {100*correct/total} %')
> > 
> > ```
> > 
> > <font color="yellow"> `Pycharm` 不够直观，我们用`Jupyter NoteBook`看 </font>
> > 
> > <div align=center>
> > <img src="./images/PyTorch_DNN_9.png"  style="zoom:100%"/>
> > </div> 
> > 
> 
> 







### 四、批量梯度下降


> <font color="yellow"> 前情提要 </font>
>
> > 前向传播与反向传播一次时，有三种情况：
> > 
> > > * `批量梯度下降`（Batch Gradient Descent，`BGD`），把所有样本一次性输入进网络，这种方式计算量开销很大，速度也很慢。
> > > 
> > > * `随机梯度下降`（Stochastic Gradient Descent，`SGD`），每次只把一个样本输入进网络，每计算一个样本就更新参数。这种方式虽然速度比较快，但是收敛性能差，可能会在最优点附近震荡，两次参数的更新也有可能抵消。
> > > 
> > > * `小批量梯度下降`（Mini-Batch Gradient Decent，`MBGD`）是为了中和上面二者而生，这种办法把样本划分为若干个批，按批来更新参数。
> > 
> > 所以，`batch_size` 即一批中的样本数，也是一次喂进网络的样本数。此外，由于 `Batch Normalization` 层（用于将每次产生的小批量样本进行标准化）的存在，`batch_size` 一般设置为 `2` 的幂次方，并且不能为 `1`。
> > 
> > <font color="yellow">注：PyTorch 实现时只支持批量`BGD`与小批量`MBGD`，不支持单个样本的输入方式。`PyTorch` 里的 `torch.optim.SGD` 只表示梯度下降
> > 
> > <font color="gree">批量与小批量见第四、五章。</font>
> > 
> > 
> 
> 
> <font color="pink">本小节将完整、快速地再展示一遍批量梯度下降（BGD）的全过程。</font>
>
> 新建 `test_ch4.ipynb`
> 
> > `Jupyter Notebook` 代码块（2个）如下：
> > 
> > ```python
> > import numpy as np
> > import pandas as pd
> > import torch
> > import torch.nn as nn
> > import matplotlib.pyplot as plt
> > %matplotlib inline
> > ```
> > 
> > ```python
> > # 展示高清图
> > from matplotlib_inline import backend_inline
> > backend_inline.set_matplotlib_formats('svg')
> > ```
> > 
> > 
> > 
> > `Pycharm` 代码如下：
> > 
> > ```python
> > import numpy as np
> > import pandas as pd
> > import torch
> > import torch.nn as nn
> > import matplotlib.pyplot as plt
> > from matplotlib_inline import backend_inline # 展示高清图
> > 
> > if __name__ == '__main__':
> >     # 展示高清图
> >     # 之前已导入库from matplotlib_inline import backend_inline
> >     backend_inline.set_matplotlib_formats('svg')
> > ```
> 
> 
> 







#### 4.1 制作数据集

>
> 这一次的数据集将从 `Excel` 中导入，需要 `Pandas` 库中的 `pd.read_csv()` 函数，这在《Pandas 标签库》讲义的第六章中有详细的介绍。
> 
> 由于当前`.xlsx`格式不适合处理，我们需要转换成`.csv`文件
> 
> > 
> > <div align=center>
> > <img src="./images/PyTorch_DNN_10.png"  style="zoom:100%"/>
> > <img src="./images/PyTorch_DNN_11.png"  style="zoom:100%"/>
> > <img src="./images/PyTorch_DNN_12.png"  style="zoom:100%"/> 
> > <img src="./images/PyTorch_DNN_13.png"  style="zoom:100%"/> 
> > <img src="./images/PyTorch_DNN_14.png"  style="zoom:100%"/> 
> > <img src="./images/PyTorch_DNN_15.png"  style="zoom:100%"/> 
> > </div> 
> > 
> 
> 
> > `Jupyter Notebook` 代码块（2个）如下：
> > 
> > ```python
> > # 准备数据集
> > df = pd.read_csv('Data.csv', index_col=0) # 导入数据
> > arr = df.values # Pandas 对象退化为 NumPy 数组
> > arr = arr.astype(np.float32) # 转为 float32 类型数组
> > ts = torch.tensor(arr) # 数组转为张量
> > ts = ts.to('cuda') # 把训练集搬到 cuda 上
> > ts.shape
> > ```
> > 
> > ```python
> > # 划分训练集与测试集
> > train_size = int(len(ts) * 0.7) # 训练集的样本数量
> > test_size = len(ts) - train_size # 测试集的样本数量
> > ts = ts[ torch.randperm( ts.size(0) ) , : ] # 打乱样本的顺序
> > train_Data = ts[ : train_size , : ] # 训练集样本
> > test_Data = ts[ train_size : , : ] # 测试集样本
> > train_Data.shape, test_Data.shape
> > ```
> > 
> > 
> > 
> > `Pycharm` 代码如下：
> > 
> > ```python
> > import numpy as np
> > import pandas as pd
> > import torch
> > import torch.nn as nn
> > import matplotlib.pyplot as plt
> > from matplotlib_inline import backend_inline # 展示高清图
> > 
> > if __name__ == '__main__':
> >     
> >     ...
> > 
> >     # 准备数据集
> >     df = pd.read_csv('Data.csv', index_col=0)  # 导入数据
> >     arr = df.values  # Pandas 对象退化为 NumPy 数组
> >     arr = arr.astype(np.float32)  # 转为 float32 类型数组
> >     ts = torch.tensor(arr)  # 数组转为张量
> >     ts = ts.to('cuda')  # 把训练集搬到 cuda 上
> >     print(ts.shape)  # torch.Size([759, 9])
> > 
> >     # 划分训练集与测试集
> >     train_size = int(len(ts) * 0.7)  # 训练集的样本数量
> >     test_size = len(ts) - train_size  # 测试集的样本数量
> >     ts = ts[torch.randperm(ts.size(0)), :]  # 打乱样本的顺序
> >     train_Data = ts[: train_size, :]  # 训练集样本
> >     test_Data = ts[train_size:, :]  # 测试集样本
> >     print(train_Data.shape, test_Data.shape)   # torch.Size([531, 9]) torch.Size([228, 9])
> > ```
> 
> 
> 
> 
> 





#### 4.2 搭建神经网络

> 
> 注意到前面的数据集，输入有 `8` 个特征，输出有 `1` 个特征，那么神经网络的输入层必须有 `8` 个神经元，输出层必须有 `1` 个神经元。
> 
> 隐藏层的层数、各隐藏层的节点数属于外部参数（超参数），可以自行设置。
> 
> > 
> > `Jupyter Notebook` 代码块（2个）如下：
> > 
> > ```python
> > # 代码块1
> > class DNN(nn.Module):
> >     def __init__(self):
> >         ''' 搭建神经网络各层 '''
> >         super(DNN,self).__init__()
> >         self.net = nn.Sequential( # 按顺序搭建各层
> >             nn.Linear(8, 32), nn.Sigmoid(), # 第 1 层：全连接层
> >             nn.Linear(32, 8), nn.Sigmoid(), # 第 2 层：全连接层
> >             nn.Linear(8, 4), nn.Sigmoid(), # 第 3 层：全连接层
> >             nn.Linear(4, 1) # 第 4 层：全连接层
> >         )
> >     def forward(self, x):
> >         ''' 前向传播 '''
> >         y = self.net(x) # x 即输入数据
> >         return y # y 即输出数据
> > ```
> > ```python
> > # 代码块2
> > model = DNN().to('cuda:0') # 创建子类的实例，并搬到 GPU 上
> > model # 查看该实例的各层
> > ```
> > 
> > 
> > `Pycharm` 代码如下：
> > 
> > ```python
> > class DNN(nn.Module):
> >     def __init__(self):
> >         ''' 搭建神经网络各层 '''
> >         super(DNN, self).__init__()
> >         self.net = nn.Sequential(    # 按顺序搭建各层
> >             nn.Linear(8, 32), nn.Sigmoid(), # 第 1 层：全连接层
> >             nn.Linear(32, 8), nn.Sigmoid(), # 第 2 层：全连接层
> >             nn.Linear(8, 4), nn.Sigmoid(), # 第 3 层：全连接层
> >             nn.Linear(4, 1) # 第 4 层：全连接层
> >         )
> > 
> >     def forward(self, x):
> >         ''' 前向传播 '''
> >         y = self.net(x)  # x 即输入数据, 这里的net和__init__()中的net要一致，自己起名
> >         return y         # y 即输出数据
> > 
> > 
> > 
> > if __name__ == '__main__':
> > 
> >     ...
> > 
> >     model = DNN().to('cuda:0')  # 创建子类的实例，并搬到 GPU 上
> >     print(model)                # 查看该实例的各层
> > ```
> > 
>
> 
> > 
> > `Jupyter` 的 `Out [6]` 如下
> > 
> > ```c
> > DNN(
> >   (net): Sequential(
> >     (0): Linear(in_features=8, out_features=32, bias=True)
> >     (1): Sigmoid()
> >     (2): Linear(in_features=32, out_features=8, bias=True)
> >     (3): Sigmoid()
> >     (4): Linear(in_features=8, out_features=4, bias=True)
> >     (5): Sigmoid()
> >     (6): Linear(in_features=4, out_features=1, bias=True)
> >   )
> > )
> > ```
> > 
> 
> 
>





#### 4.3 训练网络

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
















