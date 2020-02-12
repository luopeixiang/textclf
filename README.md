Table of Contents:

* [TextClf简介](#textclf简介)
   * [前言](#前言)
   * [系统设计思路](#系统设计思路)
   * [目录结构](#目录结构)
* [安装](#安装)
* [快速开始](#快速开始)
   * [预处理](#预处理)
   * [训练一个逻辑回归模型](#训练一个逻辑回归模型)
   * [加载训练完毕的模型进行测试分析](#加载训练完毕的模型进行测试分析)
   * [训练TextCNN模型](#训练textcnn模型)
* [TODO](#todo)
* [参考](#参考)


## TextClf简介

### 前言

TextClf 是一个面向文本分类场景的工具箱，它的目标是可以通过配置文件快速尝试多种分类算法模型、调整参数、搭建baseline，从而让使用者能有更多精力关注于数据本身的特点，做针对性改进优化。

TextClf有以下这些特性：

* 同时支持机器学习模型如逻辑回归与深度学习模型如TextCNN、Bert
* 支持多种优化方法，如`Adam` 、`AdamW` 、`Adamax`、`RMSprop`等等
* 支持多种学习率调整的方式，如`ReduceLROnPlateau` 、  `StepLR` 、 `MultiStepLR`
* 支持多种损失函数，如`CrossEntropyLoss`、`CrossEntropyLoss with label smoothing`、`FocalLoss`
* 可以通过和程序交互生成配置，再通过修改配置文件快速调整参数。
* 在训练深度学习模型时，支持使用对`embedding`层和`classifier`层分别使用不同的学习率进行训练
* 支持从断点（checkpoint）重新训练
* 具有清晰的代码结构，可以让你很方便的加入自己的模型，使用`textclf`，你可以不用去关注优化方法、数据加载等方面，可以把更多精力放在模型实现上。



与其他文本分类框架  [NeuralClassifier](https://git.code.oa.com/DeepText/NeuralClassifier) 的比较：

* `NeuralClassifier`不支持机器学习模型，也不支持Bert/Xlnet等深度的预训练模型。

* `TextClf`会比`NeuralClassifier`对新手更加友好，清晰的代码结构也会使得你能方便地对它进行拓展。

* 特别地，对于深度学习模型，`TextClf`将其看成两个部分，`Embedding`层和`Classifier`层。

  `Embedding`层可以是随机初始化的词向量，也可以是预训练好的静态词向量（`word2vec、glove、fasttext`），也可以是动态词向量如`Bert`、`Xlnet`等等。

  `Classifier`层可以是MLP，CNN，将来也会支持RCNN，RNN with attention等各种模型。

  通过将`embedding`层和`classifier`层分开，在配置深度学习模型时，我们可以选择对`embedding`层和`classifier`层进行排列组合，比如`Bert embedding + CNN`  ，`word2vec + RCNN` 等等。

  这样，通过比较少的代码实现，`textclf`就可以涵盖更多的模型组合的可能。

  

### 系统设计思路

TextClf将文本分类的流程看成**预处理、模型训练、模型测试**三个阶段。

预处理阶段做的事情主要是：

* 读入原始数据，进行分词，构建词典
* 分析标签分布等数据特点
* 保存成二进制的形式方便快速读入

数据经过预处理之后，我们就可以在上面训练各种模型、比较模型的效果。

模型训练阶段负责的是：

* 读入预处理过的数据
* 根据配置初始化模型、优化器等训练模型必需的因素
* 训练模型，根据需要最优模型

测试阶段的功能主要是：

* 加载训练阶段保存的模型进行测试
* 支持使用文件输入或者终端输入两种方式进行测试



为了方便地对预处理、模型训练、模型测试阶段进行控制，`TextClf`使用了`json`文件来对相关的参数（如预处理中指定原始文件的路径、模型训练阶段指定模型参数、优化器参数等等）进行配置。运行的时候，只要指定配置文件，`TextClf`就会根据文件中的参数完成预处理、训练或者测试等工作，详情可参见 [快速开始](#快速开始)  部分。



### 目录结构

`textclf`源代码目录下有六个子目录和两个文件，每项的作用如下所示：

```bash
├── config		# 包括预处理、模型训练、模型测试的各种参数及其默认设置
├── data		# 数据预处理、数据加载的代码
├── models		# 主要包括深度学习模型的实现
├── tester		# 负责加载模型进行测试
├── __init__.py # 模块的初始化文件
├── main.py		# textclf的接口文件，运行textclf会调用该文件中的main函数
├── trainer		# 负责模型的训练
└── utils		# 包含各种工具函数
```



## 安装

依赖环境：`python >=3.6`


使用ssh方式clone安装：
```bash
git clone git@git.code.oa.com:v_pxluo/textclf.git && cd textclf && pip3 install -e .
```

使用http方式clone安装：
```bash
git clone http://git.code.oa.com/v_pxluo/textclf.git && cd textclf && pip3 install -e .
```

上述命令首先将代码`clone`到本地， 然后切换到项目目录，使用`pip`安装`textclf`及其依赖。
之后就可以使用`textclf`了！



## 快速开始

下面我们看一下如何使用`textclf`训练模型进行文本分类。

在目录`examples/query_intent_toy_data` 下有以下文件：

```bash
 1000行 train.csv
   50行 test.csv
   50行 valid.csv
```

这些数据来自小程序意图分类项目，在这里用作演示。

文件的格式如下：

```bash
玩机工具加强版  homepage
防紫外线脸罩    goods
阿玛尼口红400   goods
爱拍    homepage
全能解析        homepage
婚庆摆设假花    goods
长毛衣  goods
湘南面霜        infomation
校园安全月      infomation
小腐基  infomation
```

文件每一行由两个字段组成，分别是句子和对应的label，句子和label之间使用`\t`字符隔开。



### 预处理

第一步是预处理。预处理将会完成读入原始数据，进行分词，构建词典，保存成二进制的形式方便快速读入等工作。要对预处理的参数进行控制，需要相应的配置文件，`textclf`中的`help-config`功能可以帮助我们快速生成配置，运行：

```bash
textclf help-config
```

输入`0`让系统为我们生成默认的`PreprocessConfig`，接着将它保存成`preprocess.json`文件：

```bash
(textclf) luo@luo-pc:~/projects$ textclf help-config
Config  有以下选择(Default: DLTrainerConfig): 
0. PreprocessConfig     预处理的设置
1. DLTrainerConfig      训练深度学习模型的设置
2. DLTesterConfig       测试深度学习模型的设置
3. MLTrainerConfig      训练机器学习模型的设置
4. MLTesterConfig       测试机器学习模型的设置
输入您选择的ID (q to quit, enter for default):0
Chooce value PreprocessConfig   预处理的设置
输入保存的文件名(Default: config.json): preprocess.json
已经将您的配置写入到 preprocess.json,你可以在该文件中查看、修改参数以便后续使用
Bye!
```

打开文件`preprocess.json`，可以看到以下内容：

```bash
{
    "__class__": "PreprocessConfig",
    "params": {
        "train_file": "train.csv",
        "valid_file": "valid.csv",
        "test_file": "test.csv",
        "datadir": "dataset",
        "tokenizer": "char",
        "nwords": -1,           
        "min_word_count": 1
    }
}
```

`params`中是我们可以进行设置的参数，这些字段的详细含义可以[查看文档](docs/preprocess.md)。这里我们只需要把`datadir`字段修改成`query_intent_toy_data`目录即可（最好使用绝对路径，若使用相对路径，要确保当前工作目录正确访问该路径。）

然后，就可以根据配置文件进行预处理了：

```bash
textclf --config-file preprocess.json preprocess
```

如无错误，输出如下：

```bash
Tokenize text from ./textclf/examples/query_intent_toy_data/train.csv...
1000it [00:00, 530655.87it/s]
Tokenize text from ./textclf/examples/query_intent_toy_data/valid.csv...
50it [00:00, 288863.91it/s]
Tokenize text from ./textclf/examples/query_intent_toy_data/test.csv...
50it [00:00, 289661.88it/s]
Label Prob:
+------------+-------------+-------------+------------+
|            |   train.csv |   valid.csv |   test.csv |
+============+=============+=============+============+
| homepage   |      0.2200 |      0.1400 |     0.2200 |
+------------+-------------+-------------+------------+
| goods      |      0.3030 |      0.4200 |     0.2400 |
+------------+-------------+-------------+------------+
| infomation |      0.4070 |      0.3600 |     0.4000 |
+------------+-------------+-------------+------------+
| video      |      0.0700 |      0.0800 |     0.1400 |
+------------+-------------+-------------+------------+
| Sum        |   1000.0000 |     50.0000 |    50.0000 |
+------------+-------------+-------------+------------+
Dictionary Size: 1508
Saving data to ./textclf.joblib...
```

预处理会打印每个数据集标签分布的信息。同时，处理过后的数据被保存到二进制文件`./textclf.joblib`中了。

预处理中的详细参数说明，请查看[文档](docs/preprocess.md)。



### 训练一个逻辑回归模型

同样的，我们先使用`textclf help-config`生成`train_lr.json`配置文件，输入`3` 选择训练机器学习模型的配置。根据提示分别选择`CountVectorizer`（文本向量化的方式）以及模型`LR`：

```bash
(textclf) luo@luo-pc:~/projects$ textclf help-config
Config  有以下选择(Default: DLTrainerConfig): 
0. PreprocessConfig     预处理的设置
1. DLTrainerConfig      训练深度学习模型的设置
2. DLTesterConfig       测试深度学习模型的设置
3. MLTrainerConfig      训练机器学习模型的设置
4. MLTesterConfig       测试机器学习模型的设置
输入您选择的ID (q to quit, enter for default):3
Chooce value MLTrainerConfig    训练机器学习模型的设置
正在设置vectorizer
vectorizer 有以下选择(Default: CountVectorizer): 
0. CountVectorizer
1. TfidfVectorizer
输入您选择的ID (q to quit, enter for default):0
Chooce value CountVectorizer
正在设置model
model 有以下选择(Default: LogisticRegression): 
0. LogisticRegression
1. LinearSVM
输入您选择的ID (q to quit, enter for default):0
Chooce value LogisticRegression
输入保存的文件名(Default: config.json): train_lr.json
已经将您的配置写入到 train_lr.json,你可以在该文件中查看、修改参数以便后续使用
Bye!
```

对于更细粒度的配置，如逻辑回归模型的参数，`CountVectorizer`的参数，可以在生成的`train_lr.json`中进行修改。这里使用默认的配置进行训练：

```bash
textclf --config-file train_lr.json train
```

因为数据量比较小，所以应该马上就能看到结果。训练结束后，`textclf`会在测试集上测试模型效果，同时将模型保存在`ckpts`目录下。

机器学习模型训练中的详细参数说明，请查看[文档](docs/trainer.md)。



### 加载训练完毕的模型进行测试分析

首先使用`help-config`生成`MLTesterConfig`的默认设置到`test_lr.json`：

```bash
(textclf) luo@luo-pc:~/projects$ textclf help-config
Config  有以下选择(Default: DLTrainerConfig): 
0. PreprocessConfig     预处理的设置
1. DLTrainerConfig      训练深度学习模型的设置
2. DLTesterConfig       测试深度学习模型的设置
3. MLTrainerConfig      训练机器学习模型的设置
4. MLTesterConfig       测试机器学习模型的设置
输入您选择的ID (q to quit, enter for default):4
Chooce value MLTesterConfig     测试机器学习模型的设置
输入保存的文件名(Default: config.json): test_lr.json
已经将您的配置写入到 test_lr.json,你可以在该文件中查看、修改参数以便后续使用
Bye!
```

将`test_lr.json`中的`input_file`字段修改成`query_intent_toy_data/test.csv` 的路径，然后进行测试：

```bash
textclf --config-file test_lr.json test
```

测试结束，`textclf`将会打印出准确率、每个label的`f1`值以及混淆矩阵：

```bash
Writing predicted labels to predict.csv
Acc in test file:0.56
Report:
              precision    recall  f1-score   support

       goods     0.5625    0.7500    0.6429        12
    homepage     0.5000    0.3636    0.4211        11
  infomation     0.5652    0.6500    0.6047        20
       video     0.6667    0.2857    0.4000         7

    accuracy                         0.5600        50
   macro avg     0.5736    0.5123    0.5171        50
weighted avg     0.5644    0.5600    0.5448        50

Confusion matrix:
+------------+---------+------------+--------------+---------+
|            |   goods |   homepage |   infomation |   video |
+============+=========+============+==============+=========+
| goods      |       9 |          0 |            3 |       0 |
+------------+---------+------------+--------------+---------+
| homepage   |       3 |          4 |            4 |       0 |
+------------+---------+------------+--------------+---------+
| infomation |       3 |          3 |           13 |       1 |
+------------+---------+------------+--------------+---------+
| video      |       1 |          1 |            3 |       2 |
+------------+---------+------------+--------------+---------+
```

关于机器学习模型测试中的详细参数，请查看[文档](docs/tester.md)。



### 训练TextCNN模型

训练深度学习模型TextCNN的过程与训练逻辑回归的流程大体一致。

这里简单做一下说明。先通过`help-config`进行配置，根据提示，先选择`DLTrainerConfig` ，然后再先后选择`Adam optimzer + ReduceLROnPlateau + StaticEmbeddingLayer + CNNClassifier +  CrossEntropyLoss`即可。

```bash
(textclf) luo@luo-pc:~/projects$ textclf help-config
Config  有以下选择(Default: DLTrainerConfig): 
0. PreprocessConfig	预处理的设置
1. DLTrainerConfig	训练深度学习模型的设置
2. DLTesterConfig	测试深度学习模型的设置
3. MLTrainerConfig	训练机器学习模型的设置
4. MLTesterConfig	测试机器学习模型的设置
输入您选择的ID (q to quit, enter for default):1
Chooce value DLTrainerConfig	训练深度学习模型的设置
正在设置optimizer
optimizer 有以下选择(Default: Adam): 
0. Adam
1. Adadelta
2. Adagrad
3. AdamW
4. Adamax
5. ASGD
6. RMSprop
7. Rprop
8. SGD
输入您选择的ID (q to quit, enter for default):0
Chooce value Adam
正在设置scheduler
scheduler 有以下选择(Default: NoneScheduler): 
0. NoneScheduler
1. ReduceLROnPlateau
2. StepLR
3. MultiStepLR
输入您选择的ID (q to quit, enter for default):1
Chooce value ReduceLROnPlateau
正在设置model
正在设置embedding_layer
embedding_layer 有以下选择(Default: StaticEmbeddingLayer): 
0. StaticEmbeddingLayer
1. BertEmbeddingLayer
输入您选择的ID (q to quit, enter for default):0
Chooce value StaticEmbeddingLayer
正在设置classifier
classifier 有以下选择(Default: CNNClassifier): 
0. CNNClassifier
1. LinearClassifier
输入您选择的ID (q to quit, enter for default):0
Chooce value CNNClassifier
正在设置data_loader
正在设置criterion
criterion 有以下选择(Default: CrossEntropyLoss): 
0. CrossEntropyLoss
1. FocalLoss
输入您选择的ID (q to quit, enter for default):0
Chooce value CrossEntropyLoss
输入保存的文件名(Default: config.json): train_cnn.json
已经将您的配置写入到 train_cnn.json,你可以在该文件中查看、修改参数以便后续使用
Bye!
```

然后运行：

```bash
textclf --config-file train_cnn.json train
```

即可开始训练我们配置好的`textcnn`模型。



当然，在训练结束之后，我们也可以通过`DLTesterConfig`配置来测试模型效果。而且，如果你想使用预训练的静态`embedding`如word2vec、glove只需要修改配置文件即可。

上述就是`TextCNN`的训练过程。如果你想尝试更多的模型，比如Bert，只需要在设置`DLTrainerConfig`时将`EmbeddingLayer`设置为 `BertEmbeddingLayer`，并且在生成的配置文件中手动设置一下预训练`Bert`模型的路径。这里就不再赘述了。



本节的相关文档：

[训练深度学习模型的详细参数说明](docs/dl_model.md)

[测试深度学习模型的详细参数说明](docs/tester.md)

[textclf文档](docs/README.md)




## TODO

* 加入更多的`classifier`：RCNN, TextRNN,  VDCNN, DPCNN等等。
* 完善文档
* 加入多模型集成评估和预测
* 加载训练好的模型，提供api服务
* 自动调参（？）



## 参考

[DeepText/NeuralClassifier](https://git.code.oa.com/DeepText/NeuralClassifier) 

[pytext](https://github.com/facebookresearch/pytext) 


