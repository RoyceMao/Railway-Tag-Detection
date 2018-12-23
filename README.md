# 铁路交通号码牌小数字目标检测与识别
<br><br>
## 环境
   * Python 3.6.4
   * Keras 2.0
## 检测效果
![](https://github.com/RoyceMao/Railway-Tag-Detection/blob/master/img/aug_3_012.png)
![](https://github.com/RoyceMao/Railway-Tag-Detection/tree/master/img/EG1.png) ![](https://github.com/RoyceMao/Railway-Tag-Detection/tree/master/img/EG2.png)
## 设计思路
考虑到拍摄图片清晰度欠佳，而且小数字号码牌在图片中的占比非常低，这里采取2阶段端到端（end2end）的方式构建检测网络：<br>
1）首先，第1阶段的特征提取加RPN类似于Fasterrcnn，生成前景概率得分最佳的Top-5 proposals（一般50%有号码牌目标，50%没有）
2）然后，根据该正负proposals样本，做特征映射与裁剪crop，进行第2阶段的anchors、特征提取、小数字检测识别过程
## loss封装到layer
自定义loss并封装到第1阶段的[class,regression]输出，跳过交替式训练的中间prediction过程，实现完全end2end，提高训练效率

```
//loss layer
loss_cls = Lambda(lambda x: class_loss_cls(*x), name='cls_loss')([input_target_cls, classification])
loss_regr = Lambda(lambda x: class_loss_regr(*x), name='regr_loss')([input_target_regr, bboxes_regression])
```

## 数据集
<img src="https://github.com/RoyceMao/Railway-Tag-Detection/tree/master/img/avatar1.jpg" width="150" height="150" alt="数据集"/>
## 训练

```
//training
python train_loss_layer.py
```

## 预测

```
//prediction
python predict.py
```
