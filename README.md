## 介绍
功能：这是一个使用transform-xl实现的长文本分类器。
训练方式：采用预训练+微调
语言：pytorch1.0

## 分类模型结构
![image](https://github.com/xuhaiming1996/Transformer-xl-for-Dcument-Classify/tree/master/image/model.jgp)

## 文件说明
###  data
#### /data/LM
train.txt 存放分好词的语料，每一行为一篇章。
#### /data/CLASSIFY
train.label 每一行为文章标签
train.txt   每一行为已经分好词的一篇文章
valid.label 每一行为文章标签
valid.txt   每一行为已经分好词的一篇文章


###  code_for_LM
#### 运行命令
这是我预训练的命令，你可以根据自己的实际情况自己调整
这里提示：尽量采用4卡并行计算


###  code_for_CLassify
#### 运行命令
这是微调的的命令，你可以根据自己的实际情况自己调整
这里提示：尽量采用4卡并行计算，


###  results
#### /results/LM
预训练的保存路径：模型参数，词典等等
#### /results/CLASSIFY
训练分类器的保存路径:模型参数等





