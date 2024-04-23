English | [简体中文](README.md)

# PTQ INT8 Quantization

This is a script for fast PTQ (Post Training Quantization) INT8 quantization using TensorRT, supporting both dynamic and static batching.

## Usage

First, configure the model you want to quantize in `calibration.yaml`.

`calibrator.data` is the path to the data used for calibration, and `calibrator.cache` is the location to save the generated calibration files.

> If you choose **dynamic batching**, ensure that the dimensions of **`batch_shape`** match **`shapes.opt`**. If you choose **static batching**, set **`dynamic`** to **`False`**, and ignore **`shapes`**.

After configuring `calibration.yaml`, run the following command to perform quantization:

```bash
cd tools
python ptq_calibration.py
```

The precision and latency after PTQ quantization vary depending on the model. For maximum precision, it is recommended to use QAT quantization.