# survey-computer-vision（2020-2021）

## 计算机视觉综述论文分方向整理（持续更新）<br>

## 目录
* [1.目标检测](#1)<br>
* [2.图像分割](#2)<br>
* [3.医学影像](#3)<br>
* [4.目标跟踪](#4)<br>
* [5.人脸](#5)<br>

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



<br><br>

<a name="3"/>

## 3.医学影像


### 综述一

**标题**：A Comprehensive Review for Breast Histopathology Image Analysis Using  Classical and Deep Neural Networks（使用经典和深层神经网络进行的乳房组织病理学图像分析的全面综述）<br>
**作者**：Xiaomin Zhou,  Tao Jiang<br>
**链接**：https://arxiv.org/abs/2003.12255<br>


本文梳理了**184篇**相关文献，由**东北大学**学者发布。<br>

乳腺癌是女性中最常见和最致命的癌症之一。 由于组织病理学图像包含足够的表型信息，因此它们在乳腺癌的诊断和治疗中起着不可或缺的作用。 为了提高乳腺组织病理学图像分析（BHIA）的准确性和客观性，人工神经网络（ANN）方法被广泛用于乳腺组织病理学图像的分割和分类任务。 在这篇综述中，我们提出了基于人工神经网络的BHIA技术的全面概述。<br>

H＆E染色的图像类型不同，（a）是正常组织，（b）是良性异常，（c）是原位癌，（d）是浸润性癌。 这些图像来自[BACH数据集](https://tva1.sinaimg.cn/large/006C3FgEgy1gn3j0htpedj30p00mstjx.jpg)<br>


[AI知识系统中ANN技术的结构](https://tva1.sinaimg.cn/large/006C3FgEgy1gn3j0wr3hbj30sq0i475d.jpg)<br>

<br><br>

### 综述二

**标题**：Medical Image Registration Using Deep Neural Networks: A Comprehensive Review（使用深度神经网络的医学图像配准：全面综述）<br>
**作者**：Hamid Reza Boveiri, Ali Reza MehdiZadeh<br>
**链接**：https://arxiv.org/abs/2002.03401<br>


本文梳理了**117篇**相关文献。以图像为指导的干预措施正在挽救大量患者的生命，在这些患者中，图像配准问题确实应被视为最复杂的问题。另一方面，由于在当代的多核GPU上实现深度神经网络的可能性，使得机器学习领域取得了巨大的进步，这为许多医疗应用打开了一个有希望的挑战之门，本文对使用深度神经网络进行医学图像配准的最新文献进行了全面回顾，系统地涵盖了该领域的相关作品，包括关键概念，统计分析，关键技术，主要贡献，挑战和未来方向。<br>


[基于优化程序的常规图像配准技术的工作流程](https://tva1.sinaimg.cn/large/006C3FgEgy1gn3j34jfn7j30hm0t8dgw.jpg)<br>

<br><br>

### 综述三

**标题**：Towards Automatic Threat Detection: A Survey of Advances of Deep Learning within X-ray Security Imaging（迈向自动威胁检测：X射线安全成像中深度学习进展综述）<br>
**作者**：Samet Akcay, Toby Breckon<br>
**链接**：https://arxiv.org/abs/2001.01293<br>


本文梳理了**151篇**相关文献，由**英国杜伦大学**学者发布。X射线安全检查被广泛用于维护航空/运输安全，其重要性引起了对自动检查系统的特别关注。 本文旨在通过将领域分类为常规机器学习和当代深度学习应用程序来回顾计算机化X射线安全成像算法。将深度学习方法分为有监督，半监督和无监督学习，着重论述分类，检测，分割和异常检测任务，同时包含有完善的X射线数据集。<br>


[X射线安全成像中深度学习应用程序中使用的数据集](https://tva1.sinaimg.cn/large/006C3FgEgy1gn3j54j7zzj30s20esq4s.jpg)<br>

[输入的X射线图像和输出取决于深度学习任务](https://tva1.sinaimg.cn/large/006C3FgEgy1gn3j5avqflj30t007smz4.jpg)（a）通过ResNet-50进行分类，（b）使用YOLOv3进行检测，并通过Mask RCNN进行分割<br>

<br><br>

### 综述四

**标题**：Deep neural network models for computational histopathology: A survey（用于计算组织病理学的深度神经网络模型综述）<br>
**作者**：Chetan L. Srinidhi, Anne L. Martel<br>
**链接**：https://arxiv.org/abs/1912.12378<br>


本文梳理了**130篇**相关文献，由**多伦多大学**学者发布。本文对组织病理学图像分析中使用的最新深度学习方法进行了全面回顾，包括有监督，弱监督，无监督，迁移学习等领域，并总结了几个现有的开放数据集。<br>

[监督学习模型概述](https://tva1.sinaimg.cn/large/006C3FgEgy1gn3jsmlqfbj30rq0go40q.jpg)<br>

<br><br>

### 综述五

**标题**：A scoping review of transfer learning research on medical image analysis using ImageNet（利用ImageNet进行医学图像分析的迁移学习研究述评）<br>
**作者**：Mohammad Amin Morid, Guilherme Del Fiol<br>
**链接**：https://arxiv.org/abs/2004.13175<br>


本文共梳理**144篇**相关文献。在非医学ImageNet数据集上受过良好训练的卷积神经网络（CNN）运用转移学习（TL），近年来在医学图像分析方面显示出令人鼓舞的结果。 本文旨在进行范围界定审查，以识别这些研究并根据问题描述，输入，方法和结果总结其特征。<br>

每个解剖部位使用不同可视化方法的频率。[ 仅显示包含的研究中总体频率至少为5％的网站](https://tva1.sinaimg.cn/large/006C3FgEgy1gn3kav654mj30ru0gm74s.jpg)<br>

<br><br>

### 综述六

**标题**：Deep Learning Based Brain Tumor Segmentation: A Survey（基于深度学习的脑肿瘤分割研究综述）<br>
**作者**：Zhihua Liu, Huiyu Zhou<br>
**链接**：https://arxiv.org/abs/2007.09479<br>


本文共梳理129篇相关文献。脑肿瘤分割是医学图像分析中一个具有挑战性的问题。脑肿瘤分割的目标是使用正确定位的遮罩生成脑肿瘤区域的准确轮廓。许多基于深度学习的方法已应用于脑肿瘤分割，并获得了令人印象深刻的系统性能。本文的目的是对最近开发的基于深度学习的脑肿瘤分割技术进行全面的调查。本文中涉及的文献广泛涵盖了技术方面，例如不同方法的优缺点，预处理和后处理框架，数据集和评估指标。<br>

[不同种类的全卷积网络（FCN）之间的高级比较](https://tva1.sinaimg.cn/large/006C3FgEgy1gn3kblh85oj30s00d840b.jpg)<br>

<br><br>

### 综述七

**标题**：A Survey on Deep Learning for Neuroimaging-based Brain Disorder Analysis（基于神经成像的脑疾病分析深度学习研究综述）<br>
**作者**：Li Zhang, Daoqiang Zhang<br>
**链接**：https://arxiv.org/abs/2005.04573<br>


本文共梳理**131篇**相关文献。深度学习最近已用于分析神经影像，例如结构磁共振成像（MRI），功能性MRI和正电子发射断层扫描（PET），并且在传统的机器学习方面，在对脑部疾病的计算机辅助诊断中，其性能得到了显着改善。 本文概述了深度学习方法在基于神经影像的脑部疾病分析中的应用。 本文回顾了深度学习方法，以计算机辅助分析四种典型的脑部疾病，包括阿尔茨海默氏病，帕金森氏病，自闭症谱系障碍和精神分裂症。<br>

[生成对抗网络的体系结构](https://tva1.sinaimg.cn/large/006C3FgEgy1gn4l7g2xbvj30ou0bu74g.jpg)<br>

<br><br>

### 综述八

**标题**：A review: Deep learning for medical image segmentation using  multi-modality fusion（多模态融合用于医学图像分割的深度学习综述）<br>
**作者**：Tongxue Zhou,  Stéphane Canu<br>
**链接**：https://arxiv.org/abs/2004.10664<br>


本文共梳理**79篇**相关文献。多模态在医学成像中被广泛使用，因为它可以提供有关目标（肿瘤，器官或组织）的多信息。使用多模态的细分包括融合多信息以改善分割效果。近来，基于深度学习的方法在图像分类，分割，目标检测和跟踪任务中展现了最先进的性能。由于其对大量数据的自学习和泛化能力，深度学习最近也引起了人们对多模式医学图像分割的极大兴趣。本文概述了基于深度学习的多模式医学图像分割任务方法，提出了不同的深度学习网络架构，然后分析了它们的融合策略并比较了它们的结果，本文还讨论医学图像分割中的一些常见问题。<br>

深度学习网络体系结构摘要，[ILSVRC：ImageNet大规模视觉识别挑战](https://tva1.sinaimg.cn/large/006C3FgEgy1gn4l7yii8fj30si072dgt.jpg)<br>

[输入级融合的通用网络体系结构](https://tva1.sinaimg.cn/large/006C3FgEgy1gn4l84q6m3j30s805ugme.jpg)<br>

<br><br>

### 综述九

**标题**：Medical Instrument Detection in Ultrasound-Guided Interventions: A Review（超声引导治疗的医疗器械检测）<br>
**作者**：Hongxu Yang, Peter H. N. de With<br>
**链接**：https://arxiv.org/abs/2007.04807<br>


本文共梳理**94篇**相关文献。医疗器械检测对于计算机辅助干预至关重要，因为它将有助于外科医生更好地解释并有效地找到器械，从而获得更好的结果。 本文回顾了超声引导介入治疗中的医疗仪器检测方法。 本文对仪器检测方法进行了全面的回顾，其中包括传统的非数据驱动方法和数据驱动方法。本文还讨论了超声中医疗器械检测的主要临床应用，包括麻醉，活检，前列腺近距离放射治疗和心脏导管插入术，这些已在临床数据集上得到验证。 最后，我们选择了几本主要出版物来总结计算机辅助干预社区的关键问题和潜在的研究方向。<br>

基于CRF的针头检测的框图。提取并选择每个体素的特征向量，分别用于体素分类和完全连接的3D CRF。借助初始体素分类，[3D CRF可以处理所选特征和分割后的体积之间的上下文相关性](https://tva1.sinaimg.cn/large/006C3FgEgy1gn4l8hpe2gj30s009ata5.jpg)<br>

### 综述十
**标题**：A Survey on Incorporating Domain Knowledge into Deep Learning for Medical Image Analysis（域知识驱动的医学图像深度学习研究综述）<br>
**作者**：Xiaozheng Xie, Shaojie Tang<br>
**链接**：https://arxiv.org/abs/2004.12150<br>

本文共梳理**268篇**相关文献。尽管像CNN这样的深度学习模型在医学图像分析中取得了巨大的成功，但医学数据集的小规模仍然是该领域的主要瓶颈。为了解决这个问题，研究人员已经开始寻找超出当前可用医学数据集的外部信息。传统方法通常通过转移学习来利用自然图像中的信息。最近的工作利用医生的领域知识来创建类似于医生的培训方式，模仿其诊断模式或专注于他们特别关注的特征或领域的网络。本文总结了将医学领域知识整合到用于各种任务的深度学习模型中的最新进展，例如疾病诊断，病变，器官和异常检测，病变和器官分割。对于每个任务，本文系统地对已使用的不同种类的医学领域知识及其相应的集成方法进行分类。<br>

信息分类方法和疾病诊断方法；病变，器官和异常检测；[病变和器官分割](https://tva1.sinaimg.cn/large/006C3FgEgy1gn4l9f63fwj30sw0bkgmt.jpg)<br>

<br><br>

### 综述十一

**标题**：A Review on End-To-End Methods for Brain Tumor Segmentation and Overall Survival Prediction（脑肿瘤的端到端分割和总体生存预测方法综述）<br>
**作者**：Snehal Rajput, Mehul S Raval<br>
**链接**：https://arxiv.org/abs/2006.01632<br>


本文共梳理**39篇**相关文献。脑肿瘤分割旨在从健康的脑组织中区分出肿瘤组织。肿瘤组织包括坏死，肿瘤周围水肿和活动性肿瘤。相反，健康的脑组织包括白质，灰质和脑脊液。基于MRI的脑肿瘤分割研究正受到越来越多的关注。 1.它不像X射线或计算机断层扫描成像一样照射电离辐射。2.生成内部人体结构的详细图片。MRI扫描输入到基于深度学习的方法中，这些方法可用于自动脑肿瘤分割。来自分段的特征被馈送到预测患者的整体存活的分类器。本文的目的是对涵盖脑肿瘤分割和总体生存预测的最新技术进行全面概述。<br>

[BTS和OS预测的端到端方法的示意图](https://tva1.sinaimg.cn/large/006C3FgEgy1gn4l9w7ec4j30se0a4q3n.jpg)<br>

<br><br>

### 综述十二

**标题**：Machine learning for COVID-19 detection and prognostication using chest radiographs and CT scans: a systematic methodological review（利用胸片和CT扫描进行冠状病毒检测和预测的机器学习：一项系统方法学综述）<br>
**作者**： Michael Roberts, Carola-Bibiane Schönlieb (on behalf of the AIX-COVNET collaboration)<br>
**链接**：https://arxiv.org/abs/2008.06388<br>

本文共梳理**108篇**相关文献。机器学习方法为根据护理标准胸部X光片（CXR）和计算机断层扫描（CT）图像快速，准确地检测和预测COVID-19提供了广阔的前景。 本文通过OVID搜索EMBASE，通过PubMed检索MEDLINE，bioRxiv，medRxiv和arXiv，以查找从2020年1月1日至2020年10月3日上载的已发表论文和预印本，其中描述了用于诊断或预测COVID-来自CXR或CT图像的19。审查发现，由于方法论上的缺陷和/或潜在的偏见，没有一个模型可以用于临床。鉴于迫切需要验证的COVID-19模型，因此这是一个主要弱点。为了解决这个问题，本文提出了许多建议，如果遵循这些建议，将能很好地解决这些问题以及提供更高质量的模型开发。<br>

<br><br>

### 综述十三 

**标题**：A Review on Deep Learning Techniques for the Diagnosis of Novel Coronavirus (COVID-19)（新型冠状病毒(冠状病毒)诊断的深度学习技术综述）<br>
**作者**： Md. Milon Islam, Jia Zeng<br>
**链接**：https://arxiv.org/abs/2008.04815<br>


本文共梳理**148篇**相关文献。新型冠状病毒（COVID-19）爆发已在世界范围内引发了灾难性疾病，并且已成为过去一百年来最严重的疾病之一。深度学习技术已证明是临床医生用于自动诊断COVID-19的武器库中的强大工具。本文旨在概述基于深度学习技术的最新开发的系统，该系统使用诸如计算机断层扫描（CT）和X射线等不同的医学成像模式。这篇综述专门讨论了使用深度学习技术为COVID-19诊断开发的系统，并提供了对用于训练这些网络的知名数据集的见解。本文旨在为专家（医学或其他方面）和技术人员提供有关深度学习技术在这方面的使用方式的新见解，以及它们如何在对抗COVID-19爆发中进一步发挥作用。<br>

[基于深度学习的COVID-19诊断系统的一般流程](https://tva1.sinaimg.cn/large/006C3FgEgy1gn4laqdwzvj30qm0huwjq.jpg)<br>

<br><br>

### 综述十四

**标题**：A Comprehensive Review for MRF and CRF Approaches in Pathology Image Analysis（病理图像分析中MRF和CRF方法综述）<br>
**作者**： Chen Li, Hong Zhang<br>
**链接**：https://arxiv.org/abs/2009.13721<br>


本文共梳理**132篇**相关文献。病理图像分析是多种疾病临床诊断的重要程序。为了提高诊断的准确性和客观性，如今提出了越来越多的智能系统。在这些方法中，随机场模型在提高调查性能中起着不可或缺的作用。在这篇综述中介绍了基于马尔可夫随机场（MRF）和条件随机场（CRF）的病理图像分析的全面概述，这是两种流行的随机场模型。<br>


[使用MaRACel模型和ESD进行Gleason分级的自动腺体分割流程图](https://tva1.sinaimg.cn/large/006C3FgEgy1gn4lb3mcz9j30re0e0tc5.jpg)<br>

<br><br>

### 综述十五 

**标题**：Medical Image Segmentation Using Deep Learning: A Survey（基于深度学习的医学图像分割研究综述）<br>
**作者**： Tao Lei, Asoke K. Nandi<br>
**链接**：https://arxiv.org/abs/2009.13120<br>


本文共梳理**162篇**相关文献。本文提出了使用深度学习技术的医学图像分割的综合调查。本文做出了两个原创性贡献。首先，与传统的将医学图像分割的深度学习文献直接分为许多组并针对每个组详细介绍的传统综述相比，本文根据从粗糙到精细的多层次结构对当前流行的文献进行分类。重点关注有监督和弱监督的学习方法，不包括无监督的方法，因为它们已在许多旧调查中引入，并且目前还不流行。对于有监督的学习方法，本文从三个方面分析文献：骨干网的选择，网络块的设计以及损失功能的改进。对于弱监督学习方法，本文分别根据数据扩充，迁移学习和交互式分段来研究文献。<br>

[V-Net架构](https://tva1.sinaimg.cn/large/006C3FgEgy1gn4lbj5x8qj30nu0fcab1.jpg)<br>

<br><br>

### 综述十六

**标题**： A Survey on Deep Learning and Explainability for Automatic Image-based Medical Report Generation（基于图像的医学报告自动生成的深度学习和可解释性研究综述）<br>
**作者**： Pablo Messina, Daniel Capurro<br>
**链接**：https://arxiv.org/abs/2010.10563<br>

本文共梳理**159篇**相关文献。每年，医生都面临着患者对基于图像的诊断的日益增长的需求，这一问题可以通过最近的人工智能方法来解决。在这种情况下，本文调查了从医学图像自动生成报告的工作，重点是使用深度神经网络的方法，涉及以下方面：（1）数据集，（2）体系结构设计，（3）可解释性和（4） 评估指标。本文的调查确定了有趣的发展，但也存在挑战。 其中，当前对生成的报告的评估特别薄弱，因为它主要依赖于传统的自然语言处理（NLP）度量标准，该度量标准不能准确地反映医学上的正确性。<br>

<br><br>

### 综述十七

**标题**：High-level Prior-based Loss Functions for Medical Image Segmentation: A  Survey（基于高层先验损失函数的医学图像分割综述）<br>
**作者**： Rosana El Jurdia,  Fahed Abdallah<br>
**链接**：https://arxiv.org/abs/2011.08018<br>


如今，深度卷积神经网络（CNN）已经证明了在各种成像方式和任务之间进行监督医学图像分割的最新技术性能。尽管取得了早期的成功，但分割网络仍可能会产生解剖学异常的分割，在对象边界附近出现孔洞或不准确。为了减轻这种影响，最近的研究工作集中在合并空间信息或先验知识以加强解剖学上合理的分割。本文将重点放在损失函数级别的高优先级，根据先验的性质对文章进行分类：对象形状，大小，拓扑和区域间约束。本文重点介绍了当前方法的优势和局限性，讨论了与基于先验损失的设计和整合以及优化策略相关的挑战，并提出了未来的研究方向。<br>

可以从[拓扑先验（a），区域间先验（b）中受益的目标细分对象的示例](https://tva1.sinaimg.cn/large/006C3FgEgy1gn4lc7ah29j30sk0daadk.jpg)<br>


<br><br>

### 综述十八

**标题**：Deep Learning in Computer-Aided Diagnosis and Treatment of Tumors: A Survey（计算机辅助肿瘤诊疗中的深度学习研究综述）<br>
**作者**： Dan Zhao, Zhigang Fu<br>
**链接**：https://arxiv.org/abs/2011.00940<br>

本文共梳理**104篇**相关文献。肿瘤的计算机辅助诊断和治疗是近年来深度学习的热门话题，它构成了一系列医学任务，例如检测肿瘤标志物，肿瘤的轮廓，肿瘤的亚型和分期，预测治疗效果，以及药物开发。同时，在主流任务场景中产生了一些具有精确定位和出色性能的深度学习模型。因此，本文从任务导向的角度介绍深度学习方法，主要侧重于医疗任务的改进。然后总结了肿瘤诊断和治疗四个阶段的最新进展，分别是体外诊断（IVD），影像学诊断（ID），病理学诊断（PD）和治疗计划（TP）。根据每个阶段的特定数据类型和医疗任务，本文介绍了深度学习在肿瘤计算机辅助诊断和治疗中的应用并分析了其中的出色著作。<br>


[CNN的结构：](https://tva1.sinaimg.cn/large/006C3FgEgy1gn4lcmp1vvj30mm0dg3zr.jpg)<br>

<br><br>

### 综述十九

**标题**：Multiple Sclerosis Lesion Segmentation \-- A Survey of Supervised  CNN-Based Methods（多发性硬化病变分割\--基于有监督CNN的方法综述）<br>
**作者**：Huahong Zhang,Ipek Oguz<br>
**链接**：https://arxiv.org/abs/2012.08317<br>


本文共梳理**93篇**相关文献。由**范德堡大学学者**发布病变分割是对多发性硬化症患者的MRI扫描进行定量分析的一项核心任务。深度学习技术在各种医学图像分析应用程序中的最新成功使人们对该挑战性问题重新产生了兴趣，并引发了新算法开发的热潮。本文研究了基于监督的CNN的MS病变分割方法，将这些评论的作品分解为它们的算法组成部分，并分别进行讨论。<br>

<br><br>

### 综述二十

**标题**：3D Bounding Box Detection in Volumetric Medical Image Data: A Systematic  Literature Review（体医学图像数据中三维包围盒检测的系统文献综述）<br>
**作者**：Daria Kern,Andre Mastmeyer<br>
**链接**：https://arxiv.org/abs/2012.05745<br>


本文共梳理了**68篇**相关文文献。由**阿伦应用技术大学**学者发布本文讨论了体积医学图像数据中3D边界框检测的当前方法和趋势，并比较了2D和3D实现。 本文介绍了多种用于定位解剖结构的方法，结果表明，大多数研究最近都集中在深度学习方法上，例如卷积神经网络与具有手动特征工程的方法，例如随机回归森林。<br>

[BB墙（6个不透明方块）](https://tva1.sinaimg.cn/large/006C3FgEgy1gn4ldh6nj2j30lw0lg3za.jpg)<br>

[组装切片以形成长方体BB时的二维检测问题](https://tva1.sinaimg.cn/large/006C3FgEgy1gn4ldrmt5uj30mo0b2wfp.jpg)<br>

<br><br>

### 综述二十一

**标题**：Learning-Based Algorithms for Vessel Tracking: A Review（基于学习的血管跟踪算法综述）
**作者**：Dengqiang Jia,Xiahai Zhuang
**链接**：https://arxiv.org/abs/2012.08929

开发有效的血管跟踪算法对于基于影像的血管疾病诊断和治疗至关重要。血管跟踪旨在解决识别问题，例如关键（种子）点检测，中心线提取和血管分割。已经开发了广泛的图像处理技术来克服血管追踪的问题，血管追踪的问题主要归因于血管的复杂形态和血管造影的图像特征。本文介绍了有关血管跟踪方法的文献综述，重点是基于机器学习的方法。

[使用常规机器学习方法在监督训练下进行视网膜血管分割的示意图](https://tva1.sinaimg.cn/large/006C3FgEgy1gn4lefa6d5j30s20mggoe.jpg)


<br><br> 

<a name="4"/> 

## 目标跟踪

### 综述一

【1】 Correlation Filter for UAV-Based Aerial Tracking: A Review and Experimental Evaluation<br>

**标题**:相关过滤无人机空中跟踪技术综述与实验评估<br>
**作者**： Changhong Fu, Geng Lu<br>
**链接**：https://arxiv.org/abs/2010.06255<br>


本文共梳理**94篇**相关文献，由**同济大学学者**提出。配备有视觉跟踪方法的基于无人机（UAV）的遥感系统已被广泛用于航空，导航，农业，运输和公共安全等。如上所述，基于UAV的航空跟踪平台已经从研究阶段逐步发展到实际应用阶段，成为未来主要的航空遥感技术之一。但是，由于现实世界中充满挑战的情况，无人机的机械结构（特别是在强风条件下）的振动以及有限的计算资源，准确性，鲁棒性和高效率对于机载跟踪方法都是至关重要的。最近，基于区分相关滤波器（DCF）的跟踪器以其高计算效率和在单个CPU上具有吸引力的鲁棒性而引人注目，并在UAV视觉跟踪社区中蓬勃发展。本文首先概括了基于DCF的跟踪器的基本框架，在此基础上，根据其解决各种问题的创新，有序总结了20种基于DCF的最新跟踪器。此外，对各种流行的无人机跟踪基准进行了详尽和定量的实验，即UAV123，UAV123_10fps，UAV20L，UAVDT，DTB70和VisDrone2019-SOT。<br>

在无人机追踪基准\[UAVDT\]下，[基于DCF的追踪器和深度追踪器的性能比较](https://tva1.sinaimg.cn/large/006C3FgEgy1gn4o0bz0p6j30s20aitab.jpg)。 图例中带有\*的跟踪器是在GPU上运行的结果，该GPU利用GPU加速了卷积和池计算。 当跟踪速度在单个CPU上达到红色虚线（30FPS）时，就可以满足无人机实时跟踪的要求。<br>

[无人机平台上基于DCF的方法的一般跟踪结构](https://tva1.sinaimg.cn/large/006C3FgEgy1gn4o0n7901j30sk0gw42f.jpg)，可分为训练阶段，模型更新和检测阶段。<br>

[六个基准测试中的原始属性（在线下）和新属性之间的对应关系](https://tva1.sinaimg.cn/large/006C3FgEgy1gn4o0vwxu6j30s00d4t93.jpg)，以及每个基准测试对每个新属性的序列号贡献<br>

<br><br>

### 综述二

【2】 Multi-modal Visual Tracking: Review and Experimental Comparison<br>

**标题**：多模态视觉跟踪：综述与实验比较<br>
**作者**：Pengyu Zhang,Dong Wang,Huchuan Lu<br>
**链接**：https://arxiv.org/abs/2012.04176<br>

本文共梳理**127篇**相关文献。视觉对象跟踪作为计算机视觉中的一项基本任务，近年来引起了很多关注。为了将跟踪器扩展到更广泛的应用范围，研究人员引入了来自多种模式的信息来处理特定的场景，这是新兴方法和基准的有前途的研究前景。 为了全面回顾多模式跟踪，本文从不同方面总结了多模式跟踪算法，特别是在统一分类法中的可见深度（RGB-D）跟踪和可见热（RGB-T）跟踪，提供了有关基准和挑战的详细描述。 此外，本文进行了广泛的实验，以分析跟踪器在五个数据集上的有效性：PTB，VOT19-RGBD，GTOT，RGBT234和VOT19-RGBT。<br>

[早期融合（EF）和晚期融合（LF）的工作流程](https://tva1.sinaimg.cn/large/006C3FgEgy1gn4o2enfm6j30sg09eglx.jpg)。基于EF的方法进行特征融合并共同建模； 而基于LF的方法旨在分别为每个模态建模，然后组合其决策<br>

[OAPF框架](https://tva1.sinaimg.cn/large/006C3FgEgy1gn4o2kgl38j30s40bq76h.jpg)。应用了带有遮挡处理的粒子滤波方法，其中遮挡模型是针对模板模型构造的。 当目标被遮挡时，遮挡模型用于预测位置而无需更新模板模型<br>

[JMMAC的工作流程](https://tva1.sinaimg.cn/large/006C3FgEgy1gn4o2puy9jj30s60aeac6.jpg)。基于CF的跟踪器用于对外观提示进行建模，同时考虑了相机和目标运动，从而获得了可观的性能。

<br><br>  

<a name="5"/>

## 人脸识别

### 综述一
**The Elements of End-to-end Deep Face Recognition: A Survey of Recent Advances**<br>
**标题：** 端到端深度人脸识别原理：最新进展综述<br>
**作者：** Hang Du, Tao Mei<br>
**链接：** https://arxiv.org/abs/2009.13290<br>

本文共梳理371篇相关文献。随着深度卷积神经网络和大规模数据集的最新发展，深度人脸识别取得了显着进展，并已广泛应用于现实应用中。给定自然图像或视频帧作为输入，端到端深脸识别系统将输出脸部特征以进行识别。为此，整个系统通常由三个关键要素构成：面部检测，面部预处理和面部表示。这三个要素都由深度卷积神经网络实现。在本文中，由于蓬勃发展的深度学习技术极大地提高了它们的能力，因此我们对端到端深度人脸识别各个元素的最新进展进行了全面的调查，分别回顾了基于深度学习的每个元素的进展，涵盖了许多方面，例如最新的算法设计，评估指标，数据集，性能比较，存在的挑战以及有希望的未来研究方向。<br>

[端到端深度人脸识别系统的标准管道](https://tva1.sinaimg.cn/large/006C3FgEgy1gnbjnryhllj30u209smyy.jpg)<br>


[代表性人脸检测方法的发展](https://tva1.sinaimg.cn/large/006C3FgEgy1gnbjogybxsj30ua06qdg8.jpg)<br>

<br><br>

### 综述二

**Face Image Quality Assessment: A Literature Survey**<br>
**标题：** 人脸图像质量评价的文献综述<br>
**作者：** Torsten Schlett, Christoph Busch<br>
**链接：** https://arxiv.org/abs/2009.01103<br>


本文共梳理173篇相关文献。人脸分析和识别系统的性能取决于所采集人脸数据的质量，该质量受众多因素影响。因此，根据生物统计实用程序自动评估面部数据的质量对于过滤掉低质量的数据很有用。本文概述了在面部生物识别技术框架中的面部质量评估文献，重点是基于可见波长面部图像的面部识别，而不是例如深度或红外质量评估，观察到了基于深度学习的方法的趋势，包括最近方法之间的显着概念差异。<br>

[典型的FQA过程](https://tva1.sinaimg.cn/large/006C3FgEgy1gnbjr838rsj30j80660t2.jpg)：对人脸图像进行预处理，然后应用FQAA，从而输出标量质量得分，并据此做出决策。<br>

<br><br>

### 综述三

**The Use of AI for Thermal Emotion Recognition: A Review of Problems and  Limitations in Standard Design and Data**<br>
**标题:** 人工智能在热情感识别中的应用：标准设计和数据中的问题和限制综述<br>
**作者：** Catherine Ordun,  Sanjay Purushotham<br>
**链接：** https://arxiv.org/abs/2009.10589<br>

随着对Covid-19放映的热成像越来越关注，公共部门可能会相信有新的机会将热学用作计算机视觉和AI的方式。自九十年代末以来，一直在进行热生理学研究。这项研究位于医学，心理学，机器学习，光学和情感计算的交叉领域。本文回顾了用于面部表情识别的热成像与RGB成像的已知因素。热成像可能会在RGB上为计算机视觉提供一种半匿名的方式，这种方式已被面部识别中的误用所困扰。但是，要想将热图像用作任何以人为中心的AI任务的来源，过渡并不容易，并且要依赖于跨多个人口统计的高保真度数据源的可用性和全面的验证。本文使读者简要回顾了热FER中的机器学习以及为AI训练收集和开发热FER数据的局限性。<br>

[静止（向上）和疲劳（向下）面部的RGB，近红外和热图像。 ](https://tva1.sinaimg.cn/large/006C3FgEgy1gnbjzm2nq8j30b209u0uf.jpg)在热图像中，较暗的像素对应于较冷的像素，较浅的像素对应于较热的像素。<br>

[长波IR的波长范围为8 µm至15 µm](https://tva1.sinaimg.cn/large/006C3FgEgy1gnbk0ogsm8j30fm06qmx9.jpg)<br>

<br><br>

### 综述四

**A Survey On Anti-Spoofing Methods For Face Recognition with RGB Cameras of Generic Consumer Devices**<br>
**标题:** 通用消费类设备RGB摄像头人脸识别反欺骗方法综述<br>
**作者：** Zuheng Ming, Jean-Christophe Burie<br>
**链接：** https://arxiv.org/abs/2010.04145<br>

本文共梳理191篇相关文献。基于面部识别的生物识别系统的广泛部署已使面部表情攻击检测（面部反欺骗）成为一个越来越关键的问题。本文彻底研究了在过去的二十年中仅需要普通消费类设备的RGB摄像头的面部表情攻击检测（PAD）方法。本文介绍了现有人脸PAD方法的面向攻击场景的类型，并回顾了50多种最新的人脸PAD方法及其相关问题，描述了面部PAD领域的主要挑战，演变和当前趋势，并提供了对其未来研究的见识。<br>


[面部表情攻击的类型](https://tva1.sinaimg.cn/large/006C3FgEgy1gnbk3fi3tmj30ye0fut9k.jpg)<br>


[每个PAD方法子类型旨在检测的Presentation Attacks（PA）的类型](https://tva1.sinaimg.cn/large/006C3FgEgy1gnbk4ellmcj30s60cut9n.jpg)<br>


[基于局部rPPG相关的方法的框架](https://tva1.sinaimg.cn/large/006C3FgEgy1gnbk59cl36j30zq0eytak.jpg)<br>

<br><br>

### 综述五

**An Overview of Facial Micro-Expression Analysis: Data, Methodology and  Challenge**<br>
**标题：** 人脸微表情分析综述：数据、方法学与挑战<br>
**作者：** Hong-Xia Xie,Ling Lo,Hong-Han Shuai,Wen-Huang Cheng<br>
**链接：** https://arxiv.org/abs/2012.11307<br>

本文共梳理205篇相关文献。面部微表情表示在情感交流过程中出现的短暂而微妙的面部运动。与宏表达式相比，由于时间跨度短且粒度变化小，因此微表达式的分析更具挑战性。近年来，微表情识别（MER）引起了很多关注，因为它可以使广泛的应用受益，例如，警察审讯，临床诊断，抑郁症分析和业务谈判。本文提供了全新的概述，以讨论当今有关MER任务的新研究方向和挑战。本文从三个新方面回顾了MER方法：从宏观到微观的适应，基于关键顶点帧的识别以及基于面部动作单元的识别。此外，为了缓解有限的和有偏差的ME数据的问题，对合成数据的生成进行了调查，以丰富微表达数据的多样性。<br>

[此调查的组织结构是根据通用MER管道进行的](https://tva1.sinaimg.cn/large/006C3FgEgy1gnbk8fmtaij31000c0wge.jpg)<br>

[TSNNLF中用于MER的2D CNN和3D CNN的组合结构](https://tva1.sinaimg.cn/large/006C3FgEgy1gnbk9c1jtfj30ig08cgm4.jpg)<br>

<br><br>

### 综述六

**Survey on 3D face reconstruction from uncalibrated images**<br>
**标题:** 基于未标定图像的三维人脸重建技术综述<br>
**作者：** Araceli Morales, Federico M. Sukno<br>
**链接：** https://arxiv.org/abs/2011.05740<br>

本文共梳理203篇相关文献。近来，许多注意力集中在将3D数据合并到面部分析及其应用中。尽管提供了面部的更准确表示，但3D面部图像的获取要比2D图片更为复杂。在开发系统上投入了大量的精力后，这些系统可以从未经校准的2D图像重建3D人脸。然而，从3D到2D的面部重建问题是不适的，因此需要先验知识来限制解决方案空间。本文回顾了过去十年中的3D人脸重建方法，重点介绍了仅使用在不受控制的条件下捕获的2D图片的方法。本文基于用于添加先验知识的技术对提出的方法进行分类，考虑了三种主要策略，即统计模型拟合，光度法和深度学习，并分别对其进行了回顾。此外，鉴于统计3D面部模型作为先验知识的相关性，本文还解释了构建过程并提供了可公开获得的3D面部模型的完整列表。<br>


[可用的3DMM的特征](https://tva1.sinaimg.cn/large/006C3FgEgy1gnbkc0jyizj31300aajrq.jpg)<br>

<br><br>

### 综述七

**DeepFakes and Beyond: A Survey of Face Manipulation and Fake Detection**<br>
**标题：** DeepFakes：面部操纵和伪造检测综述<br>
**作者：** Ruben Tolosana, Javier Ortega-Garcia<br>
**链接：** https://arxiv.org/abs/2001.00179<br>


本文梳理了105篇相关文献。本文对操纵人脸的图像技术（包括DeepFake方法）以及检测此类技术的方法进行了全面综述。论述了四种类型的面部操作：全脸合成、面部身份交换（DeepFakes）、面部属性操作以及面部表情操作。<br>


[每个面部操作组的真实和伪造示例](https://tva1.sinaimg.cn/large/006C3FgEgy1gnbkjekyelj312c0uetiv.jpg)<br>

<br><br>

### 综述八

**Deep Learning Based Single Sample Per Person Face Recognition: A Survey**<br>
**标题：** 基于深度学习的单样本人脸识别研究综述<br>
**作者：** Delong Chen,  Zewen Li<br>
**链接：** https://arxiv.org/abs/2006.11395![006C3FgEgy1gnbkoa0ujxj313009sdg8](https://tva1.sinaimg.cn/large/006C3FgEgy1gnbkoa0ujxj313009sdg8.jpg)<br>

本文共梳理62篇相关文献。在某些实际情况下，训练集中的每个身份只有一个样本。这种情况称为每人单样本（SSPP），这对有效训练深度模型提出了巨大挑战。为了解决该问题并释放深度学习的全部潜力，近年来已经提出了许多基于深度学习的SSPP人脸识别方法。对于基于SSPP的传统方法的人脸识别方法已经进行了几项全面的调查，但是很少涉及基于新兴深度学习的方法。本文将重点放在这些深层方法上，将它们分类为虚拟样本方法和通用学习方法。在虚拟样本方法中，将生成虚拟人脸图像或虚拟人脸特征，以利于深度模型的训练。在通用学习方法中，使用了额外的多样本通用集。在常规学习方法部分的分析中，涉及传统方法和深度特征合并，损失函数改进和网络结构改进的工作。<br>


[研究领域之间的关系](https://tva1.sinaimg.cn/large/006C3FgEgy1gnbkpba6aaj30i204yq2v.jpg)<br>

[虚拟人脸图像和特征生成方法图](https://tva1.sinaimg.cn/large/006C3FgEgy1gnbkpx5ey2j30jc08q0st.jpg)<br>

<br><br>

### 综述九

**A survey of face recognition techniques under occlusion**<br>
**标题：** 遮挡下的人脸识别技术综述<br>
**作者：** Dan Zeng,  Luuk Spreeuwers<br>
**链接：** https://arxiv.org/abs/2006.11366<br>

本文共梳理193篇相关文献。遮挡下人脸识别能力有限是一个长期存在的问题，它对人脸识别系统甚至人类提出了独特的挑战。与其他挑战（例如姿势变化，不同表情等）相比，有关遮挡的问题未被研究覆盖。尽管如此，遮挡的脸部识别对于在现实应用中充分发挥脸部识别的潜力至关重要。本文将范围限制为遮挡人脸识别，首先探讨了什么是遮挡问题以及哪些固有的困难会出现，介绍了遮挡下的人脸检测，这是人脸识别的第一步。其次，文章介绍现有的面部识别方法如何解决遮挡问题并将其分为三类，即1）遮挡鲁棒特征提取方法，2）遮挡感知的面部识别方法和3）基于遮挡恢复的面部识别方法。<br>

[OFR中涉及不同的遮挡人脸识别测试场景](https://tva1.sinaimg.cn/large/006C3FgEgy1gnbkt91k7pj30ig0ecq4e.jpg)<br>

<br><br>

### 综述十

**Biometric Quality: Review and Application to Face Recognition with FaceQnet**<br>
**标题：** 生物特征质量：FaceQnet在人脸识别中的应用<br>
**作者：** Javier Hernandez-Ortega, Laurent Beslay<br>
**链接：** https://arxiv.org/abs/2006.03298<br>


本文共梳理53篇相关文献。“计算机系统的输出只能与输入的信息一样准确。”这个相当琐碎的陈述是生物识别的驱动概念之一的基础：生物识别质量。如今，质量已被广泛认为是导致自动生物识别系统性能好坏的首要因素。这些算法在系统的正确运行中起着举足轻重的作用，向用户提供反馈并作为宝贵的审计工具。尽管它们被一致接受，但在这些方法的开发中却缺少一些最常用和最广泛使用的生物特征。本文通过开发FaceQnet解决了对更好的面部质量度量的需求。 FaceQnet是一种新颖的开源人脸质量评估工具，受深度学习技术的启发和支持，该工具为人脸图像分配了标量质量度量，以预测其识别精度。 NIST已在工作中以及独立地对FaceQnet的两个版本进行了全面评估，显示了该方法的合理性及其相对于当前最新技术指标的竞争力。<br>

由NIST在2019年组织的FRVT-QA活动的结果的[简短摘要](https://tva1.sinaimg.cn/large/006C3FgEgy1gnbl794y2dj30ja0isacf.jpg)，用于评估面部质量指标。<br>


<br><br>

### 综述十一

**Threat of Adversarial Attacks on Face Recognition: A Comprehensive  Survey**<br>
**标题：** 对抗攻击对人脸识别的威胁：综述<br>
**作者：** Fatemeh Vakhshiteh,  Ahmad Nickabadi<br>
**链接：** https://arxiv.org/abs/2007.11709<br>


人脸识别（FR）系统已展示出出色的验证性能，表明适用于现实世界的应用程序，从社交媒体中的照片标记到自动边界控制（ABC）。但是，在具有基于深度学习的体系结构的高级FR系统中，仅提高识别效率是不够的，并且系统还应承受旨在针对其熟练程度而设计的潜在攻击。最近的研究表明，（较深的）FR系统对难以察觉或可感知但看起来自然的对抗输入图像表现出令人着迷的脆弱性，从而使模型无法正确预测输出。本文对针对FR系统的对抗性攻击进行了全面调查，并详细阐述了针对这些系统的新对策的能力。<br>


[旨在欺骗FR系统的对抗性攻击产生方法的大致分类](https://tva1.sinaimg.cn/large/006C3FgEgy1gnblak1zkej30zg0i6mxq.jpg)<br>

<br><br>

### 综述十二

**Cross-ethnicity Face Anti-spoofing Recognition Challenge: A Review**<br>
**标题：** 跨种族人脸反欺骗识别挑战：综述<br>
**作者：** Ajian Liu, Stan Z. Li<br>
**链接：** https://arxiv.org/abs/2004.10998![006C3FgEgy1gnblcxfjbvj31380e60ul](https://tva1.sinaimg.cn/large/006C3FgEgy1gnblcxfjbvj31380e60ul.jpg)<br>

本文共梳理47篇相关文献。最近，一个多族裔面部反欺骗数据集CASIA-SURF CeFA已经发布，目的是测量种族偏见。它是涵盖3种种族，3种形态，1,607个主题，2D加3D攻击类型的最大的最新跨种族面部反欺骗数据集，并且是最近发布的面部反欺骗数据集中第一个包含显式种族标签的数据集。本文围绕这一新颖资源组织了Chalearn面部反欺骗攻击检测挑战赛，该挑战赛由单模式（例如RGB）和多模式（例如RGB，深度，红外（IR））轨道组成，以促进旨在缓解这种情况的研究种族偏见。在开发阶段，这两个轨道都吸引了340个团队，最后有11个和8个团队分别在单模式和多模式面部反欺骗识别挑战中提交了其代码。所有结果均由组委会进行了验证并重新运行，并将结果用于最终排名。本文概述了这一挑战，包括其设计，评估协议和结果摘要。<br>


[多模式赛道的9个团队的ROC](https://tva1.sinaimg.cn/large/006C3FgEgy1gnble0rehqj312e0dgwgm.jpg)。 从左到右是协议4_1、4_2和4_3的ROC<br>

[用于人脸反欺骗的多任务网络体系结构](https://tva1.sinaimg.cn/large/006C3FgEgy1gnblf2omszj30ik06w3zo.jpg)<br>

<br><br>

### 综述十三

**The Creation and Detection of Deepfakes: A Survey**<br>
**标题：** 深度伪装的产生与检测：综述<br>
**作者：** Yisroel Mirsky, Wenke Lee<br>
**链接：** https://arxiv.org/abs/2004.11138<br>

本文共梳理193篇相关文献。生成式深度学习算法已经发展到难以分辨真假之间的区别的程度。 在2018年，人们发现将这种技术用于不道德和恶意的应用非常容易，例如错误信息的传播，政治领导人的冒充以及无辜者的诽谤。 从那时起，这些“深造假”就有了长足发展。在本文中，我们探讨了Deepfake的创建和检测，并提供了有关这些架构如何工作的深入视图。 这项调查的目的是使读者对（1）如何创建和检测深造假；（2）该领域的当前趋势和进步；（3）当前防御解决方案的缺点；以及（ 4）需要进一步研究和关注的领域。


[Deepfake信息信任表](https://tva1.sinaimg.cn/large/006C3FgEgy1gnblrer74jj30i60figmb.jpg)<br>


[对抗性机器学习与deepfake之间的区别](https://tva1.sinaimg.cn/large/006C3FgEgy1gnbltylia2j30es0igdg6.jpg)<br>



