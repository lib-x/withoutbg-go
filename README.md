# withoutbg-go

[withoutBG](https://withoutbg.com/open-model) 开放权重模型（背景去除 / alpha matting）的纯 Go 封装，基于 [pure-onnx](https://github.com/amikos-tech/pure-onnx)（purego 调用 ONNX Runtime，无需 CGo）。

```go
import withoutbg "github.com/lib-x/withoutbg-go"

r, err := withoutbg.New(withoutbg.Config{ModelPath: "withoutbg-open-weights.onnx"})
if err != nil { log.Fatal(err) }
defer r.Close()

cutout, err := r.Remove(img)  // *image.NRGBA：原尺寸，RGB 是原色，A 是蒙版
alpha, err := r.Alpha(img)    // *image.Gray：只要蒙版

// 也可以走 io.Reader / io.Writer，一行完成"读图 → 抠图 → 写 PNG"
err = r.Cutout(src, dst)      // src io.Reader, dst io.Writer
```

## 特性

- **契约优先**：模型自带的 sidecar JSON（`<model>.onnx.json`）是权威来源，画布尺寸、张量名、dtype 都从它读，并和 ONNX 图逐项校验，不匹配直接报错
- **可选哈希校验**：`Config.VerifyModelSHA256` 按 sidecar 里的 sha256 校验模型文件（455MB，默认关闭）
- **两步输出**：`Alpha` 给蒙版，`Remove` 给带 alpha 通道的抠图，都是原图尺寸；返回 `*image.NRGBA`，RGB 是原色、A 是蒙版（不是预乘的 `*image.RGBA`，原因见 docs/PARAMETERS.md）
- **流式接口**：`RemoveReader` / `AlphaReader` 直接吃 `io.Reader`，`Cutout` 一行完成读图到写 PNG，`WritePNG` 单独可用；PNG 和 JPEG 解码器已内置注册
- **纯 Go 依赖**：无 CGo；ONNX Runtime 共享库通过 `ONNXRUNTIME_LIB_PATH` 指定，或由 pure-onnx 自动下载并缓存

## 模型文件

从 [withoutbg/withoutbg-openweights-onnx](https://huggingface.co/withoutbg/withoutbg-openweights-onnx) 下载两个文件，**必须成对使用**：

| 文件 | 大小 | 说明 |
|---|---|---|
| `withoutbg-open-weights.onnx` | 454,497,726 字节 | 推理图（DepthAnythingV2 + ConvNeXt 抠图头） |
| `withoutbg-open-weights.onnx.json` | 721 字节 | sidecar：`canvas_size`、张量名、形状、sha256 |

sidecar 里的 sha256 是 `29930e48e9d5ecc56d6486c53c35a4c1470566c2a3359fa180b08c8d3c34ef0f`，下载后可以直接核对：

```bash
sha256sum withoutbg-open-weights.onnx
```

## 运行环境

ONNX Runtime 共享库（1.24.1）：

```bash
# 方式一：指定已有共享库
export ONNXRUNTIME_LIB_PATH=/path/to/libonnxruntime.so.1.24.1

# 方式二：不设置，由 pure-onnx bootstrap 自动下载到 ~/.cache/onnx-purego
```

## 模型契约

| | 名称 | 形状 | 类型 | 取值范围 |
|---|---|---|---|---|
| 输入 | `rgb` | `[1, 3, 448, 448]` | float32 | `[0,1]`，NCHW |
| 输出 | `alpha` | `[1, 1, 448, 448]` | float32 | `[0,1]` |

预处理：转 RGB → 长边缩放到 448（保持比例，双线性）→ 贴在 448×448 黑底画布左上角 → 除以 255 → HWC 转 CHW。后处理：把蒙版裁到缩放区域 → 双线性缩回原尺寸 → 量化成 8 位 alpha。

参数为什么是这些值、改了会怎样，见 [docs/PARAMETERS.md](docs/PARAMETERS.md)。

## CLI

```bash
go run ./cmd/withoutbg -model withoutbg-open-weights.onnx -o cutout.png photo.jpg
# 同时导出蒙版
go run ./cmd/withoutbg -model withoutbg-open-weights.onnx -o cutout.png -alpha mask.png photo.jpg
```

CLI 内部就是走 `Cutout` 的流式接口：文件句柄直接传给库，中间不落临时文件。

## 已知边界

- **最大 448 像素**：模型输入是固定 448×448，更大的图会先缩小，因此细发丝、细网格这类高频细节的蒙版精度受限；需要更高精度时按官方建议自行分块处理
- **黑底 letterbox**：填充是黑色（对应文档要求），不是镜像或边缘延展
- **无 GPU 执行提供者**：默认 CPU；pure-onnx 的接口没有暴露 EP 选择
- **矢量图/卡通表现差**：模型在照片上表现好（官方样图对比 IoU 0.97 / 0.99），但对平涂卡通这类训练分布外的输入会留下大片背景。做过对照实验：把黑边填充去掉（裁成正方形再推理）反而更差，所以这是模型特性，不是预处理问题
- 只在官方 `oss` 导出（version 10.0.0）上验证过；换导出时 sidecar 会跟着换，契约校验会拦住不匹配的组合

## 验证

`go test ./...`：

- 单元测试：sidecar 解析（含缺字段、dtype 不符的负例）、letterbox 几何、输入张量取值与黑底填充、蒙版裁剪与缩放、alpha 量化、NRGBA 合成、流式接口往返
- golden 测试：拿官方示例三联图（输入 / 抠图 / 蒙版）对比，三个样本前景 IoU 0.869、0.973、0.991，MAE 0.038、0.016、0.082（示例是缩放过的发布件，边缘不会逐点一致，断言留了余量）
- 样图回归：`testdata` 里三张测试照片（卡通、庆祝海报、猫）跑通并检查蒙版分布（有前景、有背景、边缘比中心透明）

模型缺失时相关测试自动跳过。

## License

库代码 Apache-2.0。模型权重来自 withoutBG（Apache-2.0），其中包含的第三方组件遵循各自条款：ConvNeXt 骨干来自 DINOv3（[DINOv3 License](https://ai.meta.com/resources/models-and-libraries/dinov3-license/)），深度模型来自 Depth Anything V2（Apache-2.0）。
