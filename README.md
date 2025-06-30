# Power System Transformer

## 项目介绍
本项目是一个基于Transformer架构的电力系统负荷预测模型，用于24总线电力系统的机组组合预测。

## 主要功能
- 使用Transformer模型进行时序特征提取和预测
- 支持训练和推理两种模式
- 提供GTK图形界面进行交互式操作
- 模型保存和加载功能

## 安装要求
- Python 3.12
- PyTorch 1.8+
- GTK 3.0
- 其他依赖: numpy, pandas, matplotlib

## 使用方法
### 训练模型
```bash
python LogRegressionDetailed_24Bus_24Period00.py --train <训练数据路径> --test <测试数据路径>
```

### 使用预训练模型进行预测
```python
from Load_model.py import predict

# 加载模型
model = load_pretrained_model("models/power_system_transformer_model.pth")

# 进行预测
predictions = predict(model, input_data)
```

### 图形界面
```bash
python test.py
```

## 数据格式
- 训练数据: demand24BusWBCorr24Prd.txt
- 测试数据: commitment24BusWBCorr24Prd.txt

## 模型保存
训练好的模型默认保存在`models/power_system_transformer_model.pth`

## 结果输出
预测结果保存在`results/predictions.csv`

