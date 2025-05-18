# 基于对比时频学习网络改进的雷达信号分选  
本项目在[CTFL-Beta](https://github.com/Youngluc/CTFL-Beta)的基础上进行修改，  
作为本人本科毕设的一部分使用  
## 运行环境  
整个代码使用算力服务器远程SSH方式使用，但经测试本地3060移动端的硬件资源也可运行。  
执行如下代码在有conda的容器中建立基础的远程环境以及python虚拟环境  
```bash
apt update
apt install p7zip-full p7zip-rar
apt install byobu
byobu-enable
conda create --name your_name python=3.10.11
conda activate your_name
pip install -r requirements.txt
```
## 雷达脉冲数据  
分选脉冲信号完全基于脉内参数，故一切生成均基于单个脉冲内的调制情况  
重写生成方式，并修改为12种不同参数的雷达信号-默认使用MSST算法的默认参数  
生成时域信号至/data/raw12/  包含训练数据和测试数据
对时域信号处理为时频信号  
```bash
python ./data/generate.py
```
## 网络模型训练  
### 预训练模型-特征提取  
使用默认参数：如隐藏层大小、MSST算法的时频图数据等  
```bash
python ./main.py --pretrain
```
训练log输出至./shee_process/  
预训练模型保存至./checkpoint/  
查看log曲线:  
```bash
cd shee_process
python ./test_for_log.py
```
###  分类网络  
默认网络模型  
默认参数  
选择训练好的特征提取网络  
```bash
python main.py --train_down --checkpoints ./checkpoints/pretrain_model_name.pth
```
  
查看log曲线:  
```bash
cd shee_process
python ./test_for_log_151.py
```

## 模型测试  
注意修改模型路径  
```bash
python shee_confuse.py
```
运行完毕后输出混淆矩阵以及多项参数评估-均在png  
## 其它文件说明  
### ./data/image_diff.py
分析MSST或SPWVD输出的多张png的差异变化  
### ./data/copy_MSST.py
基于国内学者于刚老师在[MATLAB Exchange](https://ww2.mathworks.cn/matlabcentral/fileexchange/68571-multisynchrosqueezing-transform)上发布的文件进行迁移到python  
### ./data/two_curve.py
顾名思义，曲线画一块做对比  
# 额外说明  
诸多变量或者文件名被命名含有shee或者其他元素，无实际意义  
整个代码并不是完全依赖于带参运行，希望实现分默认功能仍需修改python文件  
