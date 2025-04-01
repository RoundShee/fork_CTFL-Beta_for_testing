## 说明：
这个仓库是我fork的，现在进行代码解读。
## 分析 
### /radar/generate.m
可以看到是数据生成,且存储方式是连续存放。  
    连续关系1-1000是CW，1001:2000为LFM，2001:3000为BPSK, 3001:4000为FSK， 4001:5000为NLFM， 5001:6000为LFM/NLFM  
    每种具有两种不同信噪比分别存储radar/material/t1/或t2中  
    为什么论文中要说复制三份到三个RGB，因为数据存放成了PNG格式，方便查看。  
使用MSST_Y.m生成需要的时频图，但还提供了MSST_Y_new.m；其区别暂时先搁置；  
### /dataset.py  
常规定义数据集读取方式，不用改。  
数据送入是成对送入的,比如./data/TFIs30_10  
那最后的下采样模型的训练就是没写呗?  
### main.py
有完整的特征提取网络部分的训练方法,以及写好的流程,  
而下采样的训练没有写道main函数中  
### config.py
所有网络的参数定义,以及特征提取网络的检查点等设置  
### model.py
Model使用ResNet骨架作为特征提取网络  
DownStreamModel使用孪生网络作为特征向量的分类判决  
D写的很妙,且作为Model的损失函数  
#### D
D模型在计算传入的p,z的方向相似性，是个损失函数对应原文公式(13).且仅更新p所在的网络.  
#### Model
使用ResNet50取出最后一层FC作为backbone,定义投影头根据参数设置隐藏层大小及输出大小,从特征提取到投影输出  
其中还定义了prediction  
最为精彩的是此模型前向传播的定义,第一次感受到深度学习模型定义的灵活性.  
### /radar/generate.m
这里问题似乎很多,原论文使用的是硬件模拟的方式,我怀疑这里有问题.  
首先不满足采样定理,且载波频率定义复杂,最终也不对.  
其次是代码意图使用几组随机固定的SNR,但代码实际上几乎与定义的noise无关.  
要不自己写吧.这里放一下原文:
```raw paper - Experiments and results discussion
The proposed method is used to classify the measured signals. The
universal software radio peripherals(USRPs) are used to measure the
signals from 6 transmitters. The modulations are CW, LFM, NLFM,
2FSK, BPSK, and LFM/NLFM. For each signal source, 1200 signals are
received and processed. For the MLP, the predictor, and the downstream 
classifier, there is 1 input layer, 1 hidden layer and 1 output
layer. For the projector, there is 1 input layer, 2 hidden layer, and
1 output layer. There are BN layer and ReLU after each layer except
the output layer. For the contrastive learning and the downstream
classifier, the learning rates are 0.01 and 0.001, respectively.
```
自己重写了CW用python实现部分,发现问题,30长的窗10次迭代,输出200*400尺寸大小,但MATLAB用png存储却成了875*656  
其中NLFM部分,不太好复用我已有代码现进行合理的推导:  
```MATLAB
Sig = sin(2*pi*fc*U1(i-4000)*t - pi*U2(i-4000)*k0*sin(1.5*t));  
```
对上式中的相位进行求导有:  $2\pi f_c U_1 - 1.5\pi U_2 \cos(1.5t)$  
故在一个脉冲之间要明显的看到频率的起伏需要参数1.5修改为1e6的数量级  
### /radar/MSST_Y.m
我认为我需要给他改一改,写的真拉.这是在防止我看懂吗?  
这应该是个大工作量了, 暂时计划放弃了.时间不够  

额,我竟然找到了MSST_Y作者的文件,[MATLAB_Exchange](https://ww2.mathworks.cn/matlabcentral/fileexchange/68571-multisynchrosqueezing-transform)  
那这是谁的问题??  
不用多说,我还是知识浅薄了  
用DeepSeek生成一个吧  
### filter.py
这里面就是一个对TFIs的滤波后展示  
作用是什么?  我生成的图带进去是黑的......  
### 其余几个linear文件
似乎是换了最后的下采样的分类方法-对应原文中使用几个作为对比  
## 我的修改issue
### generate部分
目前我还是觉得我生成的图,有些SNR过低的感觉是不是没用?  
### 叨叨  
要复试了,但似乎很多看不完,也不想看,这不就干点明确的事吗?写写代码,  
现在干到哪了?需要加载模型,训练一个下采样试试,但是,首先,需要写类似于孪生网络学习那样的结构  
明确如下:
1. 建立支持集样本,训练集样本,这两部分都是有标签的;支持集样本需要平均吗?样本量多大?
2. 我怎么感觉之前的随机都是在埋坑;支持集样本应该也是特征取均值吧,原文没写.  
3. 现在发现一个巨大的问题,我一直怀疑的BPSK无法区分的问题,为什么BPSK出来两根线???  
论文里的BPSK信号有两根线,但根据通信原理,确实不应该出现两根线.  这篇论文也有问题?  
### 复试结束,返校.  
现在又看了看代码,发现这个config写的很多都没用上,比如说batch_size大小,check_point路径  
那么现在,应该在下采样部分充分利用config重写,增加判断条件等,之前的特征模型训练部分保持不变  
  
再看下采样的训练,似乎前面的理解是有问题的.最后的网络仅仅是全连接层的一个简单的训练,微调.  
所以现在仅仅是缺少不加噪的信号集合  
因此接下来就是生成不加噪的信号集合,评估集还要生成新的加噪数据吗? 

