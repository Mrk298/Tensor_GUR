# Tensor_GUR
Predicting stock closing prices using the GUR network under the Tensorflow framework
#本次我们获取谷歌在2000年到2021年的股票信息，把原始数据读入进来后的一定要删除数据中的空缺值，避免对后续数据处理的影响。
# 为了方便循环神经网络的学习，需要将数据按照从旧到新的顺序排列。
#本节是通过一个序列来预测10天后的股票收盘价格。在数据表格中新添 'label' 列保存每个序列的标签值。是对一个时间点的预测。
