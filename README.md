# survey-computer-vision

## 计算机视觉综述论文分方向整理（持续更新）<br>

## 目录
* [1.目标检测](#1)<br>
* [2.图像分割](#2)<br>

<br><br>

<a name="1"/>

## 1.目标检测（object detection）

### 综述一

**标题：Deep Domain Adaptive Object Detection: a Survey**（深度域适应目标检测）<br>
作者：Wanyi Li, Peng Wang<br>
单位：中国科学院自动化研究所<br>
链接：https://arxiv.org/abs/2002.06797<br><br>


本文共梳理了**46篇**相关文献，由**中科院自动化所**学者发布。基于深度学习(DL)的目标检测已经取得了很大的进展，这些方法通常假设有大量的带标签的训练数据可用，并且训练和测试数据从相同的分布中提取。然而，这两个假设在实践中并不总是成立的。深域自适应目标检测(DDAOD)作为一种新的学习范式应运而生。本文综述了深域自适应目标检测方法的研究进展。<br>

* 深度域适应目标检测算法概述：[图1](https://tva1.sinaimg.cn/large/006C3FgEgy1gmxn18py91j31cu06stb4.jpg).[图2](https://tva1.sinaimg.cn/large/006C3FgEgy1gmxn1gz4ifj30ty10s142.jpg).[图3](https://tva1.sinaimg.cn/large/006C3FgEgy1gmxn2hhtbwj31220ewq8c.jpg)<br>

<br><br>

### 综述二

**标题：Foreground-Background Imbalance Problem in Deep Object Detectors: A Review**（深度目标检测器中前景-背景不平衡问题综述）<br>
**作者：** Joya Chen, Tong Xu<br>
**单位：** 中国科学技术大学<br>
**链接：** https://arxiv.org/abs/2006.09238<br><br>

本文研究了不平衡问题解决方案的最新进展。分析了包括一阶段和两阶段在内的各种深度检测器中不平衡问题的特征。将现有解决方案分为两类：抽样和非抽样方案，并在COCO上进行了实验对比。<br>


* [用于解决各种对象检测框架中的前景-背景不平衡问题的不同解决方案总结](https://tva1.sinaimg.cn/large/006C3FgEgy1gmxnwn8kq5j31740jwgsz.jpg)（即基于anchor-based one-stage, anchor-free onestage, two-stage的方法）。 这些解决方案包括小批量偏差采样，OHEM，IoU平衡采样，人为丢失，GHM-C，ISA，ResObj，免采样，AP丢失，DR丢失。文章在检测管道中可视化它们的使用范围。<br>
* [前景-背景不平衡问题的不同解决方案的比较。](https://tva1.sinaimg.cn/large/006C3FgEgy1gmxo99uwxij318e0cgn1c.jpg) 文章通过准确性（AP），相对准确性改进（∆AP），超参数的数量（参数）和效率（速度）进行了比较。<br>

<br><br>

### 综述三

**标题：A Review and Comparative Study on Probabilistic Object Detection in Autonomous Driving**（自动驾驶中的概率目标检测方法综述与比较研究）<br>
作者：Di Feng,Ali Harakeh,Steven Waslander,Klaus Dietmayer<br>
链接：https://arxiv.org/abs/2011.10671<br>


近年来，深度学习已成为实际的目标检测方法，并且提出了许多概率目标检测器。然而，关于深度目标检测的不确定性估计尚无总结，而且现有方法不仅建立在不同的网络上架构和不确定性估算方法，而且还可以使用各种评估指标对不同的数据集进行评估。结果，方法的比较仍然具有挑战性，最适合特定应用的模型选择也是如此。本文旨在通过对现有的用于自动驾驶应用的概率目标检测方法进行回顾和比较研究,来缓解这一问题。

* [城市驾驶场景中概率对象检测的概念图。](https://tva1.sinaimg.cn/large/006C3FgEgy1gmxoqqq4wzj30n00buqh5.jpg)使用分类概率对每个对象进行分类，并使用置信区间预测其边界框。 RGB相机图像来自BDD100k数据集。<br>
* [不确定性估计在自动驾驶中的应用及实例参考](https://tva1.sinaimg.cn/large/006C3FgEgy1gmxosf367bj31ci0j6tde.jpg)<br>
* [最先进的概率目标检测器](https://tva1.sinaimg.cn/large/006C3FgEgy1gmxovfglv7j31am0beahm.jpg)中通常包含的关键构建块的插图，包括基础网络，检测头和后处理阶段。架构图下方还列出了每个构件的可能变体。 2D图像上的输出检测结果显示为类别概率（橙色），边界框平均值（红色）和边界框角协方差矩阵的95％置信度等值线（绿色）。<br>
* [概率目标检测器概述](https://tva1.sinaimg.cn/large/006C3FgEgy1gmxotcn5fsj30vo0tywlv.jpg)<br>
* [通过测试BDD验证数据集上的检测器，不进行数据集偏移的评估。](https://tva1.sinaimg.cn/large/006C3FgEgy1gmxoulivezj319w0ckn07.jpg)将所有七个动态对象类别的结果取平均值。 对于NLL，较低的值表示较好的性能。<br>

<br><br>

### 综述四

**标题：An Overview Of 3D Object Detection**（三维目标检测技术综述）<br>
**作者**： Yilin Wang, Jiayi Ye<br>
**单位：** 阿尔伯塔大学<br>
**链接**：https://arxiv.org/abs/2010.15614<br><br>

本文共梳理37篇相关文献。由阿尔伯塔大学学者发布。点云3D对象检测最近受到了广泛关注，并成为3D计算机视觉社区中一个活跃的研究主题。 然而，由于点云的复杂性，在LiDAR（光检测和测距）中识别3D对象仍然是一个挑战。 行人，骑自行车的人或交通锥等物体通常用稀疏点表示，这使得仅使用点云进行检测就相当复杂。 在这个项目中，我们提出了一个使用RGB和点云数据来执行多类对象识别的框架。 我们使用现有的2D检测模型来定位RGB图像上的感兴趣区域（ROI），然后在点云中进行像素映射策略，最后将初始2D边界框提升到3D空间。 我们使用最近发布的nuScenes数据集（包含许多数据格式的大规模数据集）来训练和评估我们提出的体系结构。<br>

* [YOLO的3D点云中的对象检测示例](https://tva1.sinaimg.cn/large/006C3FgEgy1gmxoy9fvvkj30js0b2n4y.jpg)<br>

<br><br>

### 综述五

**标题：Camouflaged Object Detection and Tracking: A Survey**（伪装目标检测与跟踪研究综述）<br>
**作者**：Ajoy Mondal<br>
**单位：** IIIT Hyderabad<br>
**链接**：https://arxiv.org/abs/2012.13581<br><br>

运动目标的检测和跟踪应用于各个领域，包括监视，异常检测，车辆导航等。关于目标检测和跟踪的文献非常丰富，然而，由于其复杂性，对伪装目标检测与跟踪的研究目前取得的进展有限。本文从理论角度回顾了基于计算机视觉算法的现有伪装目标检测和跟踪技术。还讨论了该领域中一些值得探讨的问题及未来的研究方向。<br>

* [各种挑战的直观图示](https://tva1.sinaimg.cn/large/006C3FgEgy1gmxp2sguo8j30qk0os4qp.jpg)上图为（a）照明变化，（b）背景杂波，（c）部分遮挡，（d）完全遮挡，（e）物体比例改变，（f）物体方向改变，（g）伪装物体，（h ）姿势变化，以及（i）不规则形状的物体。<br>

<br><br>

<a name="2"/>

## 2.图像分割

<br>

### 综述一

**标题：A Survey on Deep Learning Methods for Semantic Image Segmentation in Real-Time**（深度学习实时语义图像分割方法综述）<br>
**作者**： Georgios Takos<br>
**链接**：https://arxiv.org/abs/2009.12942<br><br>


本文共梳理了9篇相关文献。语义图像分割是计算机视觉中增长最快的领域之一，具有多种应用程序。 在许多领域，例如机器人技术和自动驾驶汽车，语义图像分割至关重要，因为语义分割为基于像素级别的场景理解提供了采取动作所需的必要上下文。 此外，医学诊断和治疗的成功取决于对所考虑数据的极其准确的理解，并且语义图像分割是许多情况下的重要工具之一。 深度学习的最新发展提供了许多工具来有效地解决这一问题，并且提高了准确性。 这项工作对图像分割中的最新深度学习体系结构进行了全面分析，更重要的是，它提供了广泛的技术列表以实现快速推理和计算效率。<br>

* [完全卷积网络架构](https://tva1.sinaimg.cn/large/006C3FgEgy1gmxz04imiqj30sq0fsdhg.jpg)<br>
* [DeconvNet体系结构](https://tva1.sinaimg.cn/large/006C3FgEgy1gmxz0icrrdj30ss08ijsc.jpg)<br>
* [比例感知语义图像分割架构](https://tva1.sinaimg.cn/large/006C3FgEgy1gmxz375hbdj30r20ee40e.jpg)<br>
* [Cityscapes像素级语义标签任务最佳表现模型](https://tva1.sinaimg.cn/large/006C3FgEgy1gmxz3k6msoj30rw0nk421.jpg)<br>

<br><br>

### 综述二

**标题：Image Segmentation Using Deep Learning: A Survey**（使用深度学习进行图像分割：综述）<br>
作者：Shervin Minaee, Demetri Terzopoulos<br>
链接：https://arxiv.org/abs/2001.05566<br><br>


本文梳理了**172篇**相关文献。最近，由于深度学习模型在各种视觉应用中的成功，已经有大量旨在利用深度学习模型开发图像分割方法的工作。本文提供了对文献的全面回顾，涵盖了语义和实例级分割的众多开创性作品，包括全卷积像素标记网络，编码器-解码器体系结构，多尺度以及基于金字塔的方法，递归网络，视觉注意模型和对抗环境中的生成模型。本文研究了这些深度学习模型的相似性，优势和挑战，研究了使用最广泛的数据集，报告了性能，并讨论了该领域有希望的未来研究方向。<br>

* [DeepLabV3在样本图像上的分割结果](https://tva1.sinaimg.cn/large/006C3FgEgy1gmxz4lgz00j30i40dutbh.jpg)<br>
* [U-net模型。 蓝色框表示具有其指定形状的要素地图块](https://tva1.sinaimg.cn/large/006C3FgEgy1gmxz4z82puj30j40dw0t8.jpg)<br>
* [DeepLabv3 +模型](https://tva1.sinaimg.cn/large/006C3FgEgy1gmxz6eqzt8j30ky0bymy0.jpg)<br>
* [2014年至2020年基于DL的2D图像分割算法的时间轴。橙色，绿色和黄色块分别表示语义，实例和全景分割算法](https://tva1.sinaimg.cn/large/006C3FgEgy1gmxz6ol54kj30s80aqaaz.jpg)<br>
* [以mIoU和平均准确度（mAcc）表示，NYUD-v2和SUN-RGBD数据集上的分割模型的性能](https://tva1.sinaimg.cn/large/006C3FgEgy1gmxz71a0u3j30kk0f23zm.jpg)<br>

<br><br>

### 综述三

**标题：Unsupervised Domain Adaptation in Semantic Segmentation: a Review（语义分割中的无监督自适应研究进展）**<br>
作者：Marco Toldo,  Pietro Zanuttigh<br>
链接：https://arxiv.org/abs/2005.10876<br><br>

本文共梳理**120篇**相关文献。本文的目的是概述用于语义分割的深度网络的无监督域自适应（UDA）的最新进展。 这项任务引起了广泛的兴趣，因为语义分割模型需要大量的标记数据，而缺乏适合特定要求的数据是部署这些技术的主要限制。<br>

* [从分类（稀疏任务）到语义分割（密集任务）的一些样本图像上一些可能的视觉任务的概述](https://tva1.sinaimg.cn/large/006C3FgEgy1gmxzb9uhtbj30r20d0juw.jpg)<br>
* [可以在不同的空间执行域移位自适应：输入级别，功能级别和输出级别](https://tva1.sinaimg.cn/large/006C3FgEgy1gmxzbibzy1j30r80c80ue.jpg)<br>
* [最受欢迎的用于语义分割的UDA策略的维恩图。 每种方法都属于代表使用的自适应技术的集合](https://tva1.sinaimg.cn/large/006C3FgEgy1gmxzbpjh1bj30r80hsgod.jpg)<br>

<br><br>

### 综述四

**标题：A survey of loss functions for semantic segmentation**（语义分割损失函数综述）<br>
作者：Shruti Jadon<br>
链接：https://arxiv.org/abs/2006.14822<br><br>


本文共梳理了**23篇**相关文献。在本文中，我们总结了一些众所周知的损失函数，这些函数广泛用于图像分割，并列出了使用它们可以帮助快速，更好地收敛模型的情况。 此外，我们还引入了新的log-cosh骰子损失函数，并将其在NBFS头骨分割开源数据集上的性能与广泛使用的损失函数进行了比较。 我们还展示了某些损失函数在所有数据集上都能很好地发挥作用，并且在未知的数据分发方案中可以被视为很好的基准选择。<br>

* [语义分段损失函数的类型](https://tva1.sinaimg.cn/large/006C3FgEgy1gmxze0odc7j30r80gmabt.jpg)<br>
* [二元交叉熵损失函数图。 在这里，熵在Y轴上定义，事件的概率在X轴上](https://tva1.sinaimg.cn/large/006C3FgEgy1gmxzeayau4j30p80i20u5.jpg)<br>
* [语义分段损失功能的附表](https://tva1.sinaimg.cn/large/006C3FgEgy1gmxzel7n95j30s20gk76w.jpg)<br>

<br><br>

### 综述五
**标题：A Survey on Instance Segmentation: State of the art**（实例分割技术综述）<br>
作者：Abdul Mueed Hafiz, Ghulam Mohiuddin Bhat<br>
链接：https://arxiv.org/abs/2007.00047<br><br>

本文共梳理**143篇**相关文章，由克什米尔大学的学者发布。目标检测或定位是从粗略到精细的数字图像推断的增量步骤。它不仅提供图像对象的类别，而且还提供已分类图像对象的位置。该位置以边界框或质心的形式给出。语义分割可通过预测输入图像中每个像素的标签来进行精细推断。每个像素根据其所在的对象类别进行标记。为进一步发展，实例分割为属于同一类的对象的单独实例提供了不同的标签。因此，实例分割可以被定义为同时解决对象检测和语义分割问题的技术。在这份关于实例分割的调查论文中，讨论了实例分割的背景，问题，技术，演变，流行的数据集，相关技术以及最新范围。本文为那些希望在实例分割领域进行研究的人提供了宝贵的信息。<br>

* [对象识别的演变](https://tva1.sinaimg.cn/large/006C3FgEgy1gmxzfttve4j30rs0gygos.jpg)从粗略推断到细粒度推断：（a）图像分类（b）对象检测或定位（c）语义分割（d）实例分割<br>
* [实例分割中重要技术的时间表](https://tva1.sinaimg.cn/large/006C3FgEgy1gmxzgbzvqgj30ro0a0jrn.jpg)<br>
* [PANet框架](https://tva1.sinaimg.cn/large/006C3FgEgy1gmxzgn10rlj30s40bq755.jpg)（a）FPN主干（b）自下而上的路径扩充（c）自适应功能池（d）盒支（e）全连接融合<br>
* [Microsoft COCO数据集上值得注意的实例细分工作](https://tva1.sinaimg.cn/large/006C3FgEgy1gmxzgy7gtbj30ro0e8q49.jpg)<br>



