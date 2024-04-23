[English](README.en.md) | 简体中文

# PTQ INT8 量化

这是一个使用 TensorRT 进行快速 PTQ（Post Training Quantization）INT8 量化的脚本，支持动态和静态 Batch。

## 使用方法

首先，在 `calibration.yaml` 中配置你要量化的模型。

`calibrator.data` 是用于校准的数据路径，而 `calibrator.cache` 则是保存生成的校准文件的位置。

> 如果你选择 **动态 Batch**，务必确保 **`batch_shape`** 的维度与 **`shapes.opt`** 一致；如果你选择 **静态 Batch**，将 **`dynamic`** 设为 **`False`**，并**忽略 `shapes`**。

配置好 `calibration.yaml` 后，运行以下命令进行量化：

```bash
cd tools
python ptq_calibration.py
```

PTQ 量化后的精度与延时因模型而异，如果追求最高精度，建议使用 QAT 量化。