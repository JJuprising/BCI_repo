# 项目说明

本项目包含了SSVEP算法模型以及自采集系统

* Classifier/ 包括了公开数据集分类(PublicDataset/)和基于MindBridge的自采集数据集分类(SelfDataset/)
* MindBridge/ 包括了控制系统代码(ssvep_controller/)和字符拼写器代码(ssvep_speller/)

# 要点

基于SSVEP的论文，一般可以分为两种

* 一是单纯的算法创新，也就是我们做的，只在公开数据集上跑算法超过SOTA水平
* 二是控制系统，通过自采集数据，训练模型，用于实时控制。这种一般就需要在系统上有所创新(协同控制可以算一个)
* 另外，更加容易中的，是将前两种结合，进行算法创新之后，不仅在公开数据集上达到最优，再应用到控制系统中

下面列举几篇参考，但不限于这几篇

* 对于算法创新，直接看PublicDataset/下的README中提到的论文，都是前几年的一些最优模型
* 对于控制系统

  * Mobile robot navigation with a self-paced brain–computer interface based on high-frequency SSVEP
  * Navigation assistance for a BCI-controlled humanoid robot
  * Brain–Computer Interface-Based Stochastic Navigation and Control of a Semiautonomous Mobile Robot in Indoor Environments
  * A Low-Cost EEG System-Based Hybrid Brain-Computer Interface for Humanoid Robot Navigation and Recognition
* 对于两种结合
  * Xie, Shanghong, et al. "Multi-degree-of-freedom unmanned aerial vehicle control combining a hybrid brain-computer interface and visual obstacle avoidance." *Engineering Applications of Artificial Intelligence* 133 (2024): 108294.
  * Ban, Nianming, et al. "Multifunctional robot based on multimodal brain-machine interface." *Biomedical Signal Processing and Control* 91 (2024): 106063.
  * Wu, Qingfu, et al. "A multiple command UAV control system based on a hybrid brain-computer interface." *2023 International Joint Conference on Neural Networks (IJCNN)*. IEEE, 2023.