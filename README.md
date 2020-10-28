# leapmotion_gestures_recognition
Gesture Recognition

​		LSTM适合处理和预测[时间序列](https://baike.baidu.com/item/时间序列)中间隔和延迟非常长的重要事件   记住需要长时间记忆的，忘记不重要的信息 适合需要“长期记忆”的任务。LSTM设计用于顺序数据，按顺序处理每个时间步，并学习存储和使用先前时间步中的相关信息。例如，如果您查看单个快照，则“停止”手势就像“波动”手势，但是如果您随着时间的推移查看它，则“波动”手势会增加左右移动，LSTM可以拾取并用来区分两个手势。与“普通”的全连接前馈模型相比，LSTM在顺序数据上的性能要好得多。

#### 1、Training Data Flow：

重要的是要了解在本项目中如何转换和使用leap motion设备中的数据。 看起来像这样：

raw data -> select variables of interest -> calculate some derived variables -> drop unwanted variables -> standardize variables -> split into examples

此过程大部分由文件`params/`夹中的文件控制。



一个很好的起点是Notebook ' exploration-many2one.ipynb'，它从加载数据到训练模型都可以逐步进行。 数据可以从（https://drive.google.com/drive/folders/1gx3efh6_GlQn0re9NhdmoCJ58lR3FDVp?usp=sharing）下载。

{补充：.ipynb” 文件是使用 Jupyter Notebook 来编写Python程序时的文件。Jupyter Notebook（此前被称为 IPython notebook）是一个交互式笔记本}

#### 2、raw data -> select variables of interest：

leap motion设备输出无关紧要的信息，其中大部分对于预测都是多余的。 文件“ VoI.txt”控制在使用负责从leap motion设备收集和处理数据的方法时，将选择要关        注的变量（VoI）。

在此阶段，还将检查帧频，并每第n帧获取一次接近目标帧频。 我使用的目标帧速率是5fps。 我最初使用25fps，但是较高的帧速率没有帮助。

#### 3、select variables of interest -> calculate some derived variables



我们可能需要更高级的变量，以使我们的模型更具参考价值。这样的示例包括使用指尖位置来计算其与手掌所在平面的距离，或它们与手掌的距离。与使用每个指尖相对于跳跃运动设备的x / y / z坐标相比，此类信息更具参考价值。此过程控制如下：

- `params/derived_features_one_handed.txt` 包含应用于数据的方法列表，这些数据将生成新的单手功能（例如同一只手的拇指和食指之间的距离）
- `params/derived_features_two_handed.txt` 包含将应用于数据的方法列表，这些数据将生成新的两个手部特征（例如，左右食指之间的距离）
- `src/features.py` 包含所有可以应用的可能方法

#### 4、calculate some derived variables -> drop unwanted variables

可能仅包括某些VoI，因为它们是计算派生变量所需的。`params/VoI_drop.txt`包含此时应删除的此类变量的列表。

#### 5、drop unwanted variables -> standardize variables

现在，我们只有将要输入到模型中进行训练或预测的变量。 它们只需要居中和标准化即可，因此它们具有单位方差和均值为零。 有字典，每个字典的标准差和均值分别在`params / stds_dict.json`和`params / means_dict.json`中。

第一次使用新的derived variables或新的VoI时，它们在相关词典中将没有均值和标准差。 上面推荐的notebook包含一个用于重新生成这些代码的代码块。

#### 6、standardize variables -> split into examples

最后一步是将数据分割为一定数量的帧。我一直在使用25至40帧之间的值。