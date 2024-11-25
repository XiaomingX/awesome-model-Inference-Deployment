# awesome-model-Inference-Deployment

**精选AI模型推理部署框架列表**

| 框架 | 开发公司 | 支持的编程语言 | 支持的框架 | 支持的量化类型 | 支持的硬件 | 支持的系统 | 应用场景 | 其他功能 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **[OpenVINO](https://docs.openvino.ai/latest/index.html)** | Intel | C, C++, Python | 支持TensorFlow、PyTorch、ONNX等 | INT8, FP16 | CPU, GPU, FPGA等 | Linux, Windows, macOS | 用于Intel硬件的AI优化工具 | 高效模型优化 |
| **[TensorRT](https://developer.nvidia.com/zh-cn/tensorrt)** | NVIDIA | C++, Python | 支持TensorFlow、PyTorch、ONNX等 | INT8, FP16 | NVIDIA GPU | Linux, Windows | 高性能推理 | 优化GPU性能 |
| **[MediaPipe](https://developers.google.com/mediapipe)** | Google | C++, JavaScript, Python | 支持TensorFlow | - | GPU, TPU | Linux, Android, iOS等 | 视频分析和实时流媒体处理 | 可定制化 |
| **[TensorFlow Lite](https://www.tensorflow.org/lite)** | Google | C++, Python, Java等 | 支持TensorFlow | INT8, FP16 | CPU, GPU, TPU等 | Linux, Android, iOS等 | 移动端机器学习 | 轻量化部署 |
| **[ONNX Runtime](https://onnxruntime.ai)** | Microsoft | C, Python等 | 支持TensorFlow、PyTorch、ONNX等 | INT8 | CPU, GPU, NPU | Linux, Windows, macOS等 | 用于Office 365、Bing等 | 通用推理引擎 |
| **[NCNN](https://github.com/Tencent/ncnn)** | 腾讯 | - | 支持TensorFlow、ONNX等 | INT8, FP16 | CPU, GPU | Linux, Android, iOS等 | 微信、QQ等应用 | 无第三方依赖 |
| **[Paddle Lite](https://www.paddlepaddle.org.cn/lite)** | 百度 | C++, Python等 | 支持PaddlePaddle | INT8, INT16 | CPU, GPU, FPGA等 | Android, iOS, Linux等 | 百度产品的AI部署 | 轻量化设计 |
| **[MNN](http://www.mnn.zone)** | 阿里巴巴 | Python | 支持TensorFlow、ONNX等 | INT8, BF16, FP16 | CPU, GPU, NPU等 | iOS, Android, Linux等 | 淘宝、优酷等产品 | 多场景支持 |
| **[TNN](https://github.com/Tencent/TNN)** | 腾讯 | - | 支持TensorFlow、ONNX等 | INT8, FP16 | CPU, GPU, NPU | Android, iOS, Linux等 | 移动端AI加速 | 高性能 |
| **[MegEngine Lite](https://megengine.org.cn)** | 旷视 | Python, C++等 | 支持MegEngine | INT8 | CPU, GPU等 | Linux, Android, iOS等 | 面向多平台推理 | 高效推理 |
| **[TVM](https://tvm.apache.org)** | 华盛顿大学 | Python, C++等 | 支持TensorFlow、ONNX等 | - | CPU, GPU, FPGA等 | Linux, Windows等 | 模型编译与优化 | 灵活扩展 |
| **[Bolt](https://huawei-noah.github.io/bolt)** | 华为 | C, Java等 | 支持TensorFlow、ONNX等 | INT8, FP16 | CPU, GPU等 | Linux, Android, iOS等 | 华为产品线 | 高效部署 |

---

### 什么是ONNX？

ONNX（开放神经网络交换格式）是一种通用的机器学习模型格式，支持在不同框架和硬件间高效运行。主要支持的框架包括TensorFlow、PyTorch、ONNX Runtime等。ONNX由微软、亚马逊、Meta等公司共同开发，用于跨平台模型部署。

### 使用场景示例：
- 从PyTorch导出模型到ONNX，然后用ONNX Runtime运行。
- 从TensorFlow导出模型到ONNX，然后用NCNN运行。
- 利用ONNX提高推理性能，适配多种硬件。

### 3. 实践

#### 3.1. ONNX 简介

ONNX 是一种广泛支持的开放格式，可以在许多框架、工具和硬件中使用。它实现了不同框架之间的互操作性，并简化了从研究到生产的流程，加快了 AI 社区的创新速度。

---

#### 3.1.1. 将 PyTorch 模型导出为 ONNX 格式并使用 ONNX Runtime 运行

以下是主要功能的实现方法：

---

**1. 导出 PyTorch 模型为 ONNX 格式**

```python
def PyTorch2ONNX(torch_model, dummy_input_to_model, onnx_save_dir, check_onnx_model=True):
    ''' 导出模型（PyTorch2ONNX） '''
    torch.onnx.export(
        torch_model,                                    # 需要转换的 PyTorch 模型
        dummy_input_to_model,                           # 模型输入（可以是单个或多个输入）
        onnx_save_dir,                                  # 保存模型的路径
        export_params=True,                             # 将训练好的参数保存到模型文件中
        opset_version=10,                               # 导出的 ONNX 版本
        do_constant_folding=True,                       # 是否执行常量折叠以优化模型
        input_names=['input'],                          # 模型的输入名称
        output_names=['output'],                        # 模型的输出名称
        dynamic_axes={                                  # 动态维度设置
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        })

    if check_onnx_model:  # 验证模型结构，确保模型合法
        onnx_model = onnx.load(onnx_save_dir)
        onnx.checker.check_model(onnx_model)
```

---

**2. 使用 ONNX Runtime 运行模型**

```python
def Run_ONNX_in_ONNX_RUNTIME(onnx_dir, img_path, img_save_path):
    ''' 使用 ONNX Runtime 运行模型并处理图像 '''
    # 读取并预处理图像
    img = Image.open(img_path)
    resize = transforms.Resize([224, 224])
    img = resize(img)
    img_ycbcr = img.convert('YCbCr')
    img_y, img_cb, img_cr = img_ycbcr.split()
    to_tensor = transforms.ToTensor()
    img_y = to_tensor(img_y)
    img_y.unsqueeze_(0)

    # 创建 ONNX Runtime 推理会话
    ort_session = onnxruntime.InferenceSession(onnx_dir)

    # 在 ONNX Runtime 中运行模型
    ort_inputs = {ort_session.get_inputs()[0].name: torchtensor2numpy(img_y)}
    ort_outs = ort_session.run(None, ort_inputs)
    img_out_y = ort_outs[0]

    # 生成输出图像
    img_out_y = Image.fromarray(np.uint8((img_out_y[0] * 255.0).clip(0, 255)[0]), mode='L')
    final_img = Image.merge(
        "YCbCr", [
            img_out_y,
            img_cb.resize(img_out_y.size, Image.BICUBIC),
            img_cr.resize(img_out_y.size, Image.BICUBIC),
        ]).convert("RGB")

    # 保存输出图像
    final_img.save(img_save_path)
```

---

完整代码请参考：[PyTorch2ONNX_Run_in_ONNX_RUNTIME.py](./src/PyTorch2ONNX/PyTorch2ONNX_Run_in_ONNX_RUNTIME.py)。
