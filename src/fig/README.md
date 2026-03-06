# 豆包视觉 API：目标文字边界框识别

根据用户输入（如 `#3阳床的0.0m3/h`）解析出要定位的文字（如 `0.0m3/h`），调用豆包大模型视觉 API，在给定图片中识别该文字的**精确边界框（box）**。

## 功能

- **输入解析**：从 `#3阳床的0.0m3/h` 自动提取目标文字 `0.0m3/h` 和上下文 `#3阳床`。
- **豆包 API 调用**：使用火山方舟 `POST /api/v3/responses`，传入图片 + 构造的定位提示词。
- **边界框解析**：从模型返回的文本中解析 JSON 格式的 `x_min, y_min, x_max, y_max`（相对坐标 0~1）。

## 环境

- Python 3.9+
- 依赖：`requests`（见 `requirements.txt`）

## 安装

```bash
pip install -r requirements.txt
```

## 配置

- **API Key**：豆包 Ark API Key，可从 [火山方舟控制台](https://console.volcengine.com/ark) 获取。
- 可通过环境变量 `DOUBAO_ARK_API_KEY` 或命令行参数 `--api-key` 传入（**不要将 Key 提交到代码库**）。

## 用法

### 命令行

```bash

python doubao_box_detector.py "中间水泵P109A下面的反馈右边的0.00HZ" --image-path ./小小.png -o 小小2.png




# 使用图片 URL
export DOUBAO_ARK_API_KEY="你的API_KEY"
python doubao_box_detector.py "#3阳床的0.0m3/h" --image-url "https://example.com/screenshot.png"

# 使用本地图片
python doubao_box_detector.py "#3阳床的0.0m3/h" --image-path ./1.jpg --api-key "你的API_KEY"

# 指定模型（默认 doubao-seed-2-0-pro-260215）
python doubao_box_detector.py "#3阳床的0.0m3/h" --image-url "https://..." --model "你的模型endpoint"
```

### 输出示例

```json
{
  "x_min": 0.12,
  "y_min": 0.35,
  "x_max": 0.22,
  "y_max": 0.42,
  "found": true
}
```

若未在图中找到该文字，则 `found` 为 `false`，坐标可为 0。

### 在代码中调用

```python
from doubao_box_detector import detect_text_box, parse_user_input

# 解析输入
target, context = parse_user_input("#3阳床的0.0m3/h")  # ("0.0m3/h", "#3阳床")

# 识别边界框（图片三选一：image_url / image_path / image_base64）
box = detect_text_box(
    "#3阳床的0.0m3/h",
    image_url="https://example.com/image.png",
    api_key="你的API_KEY",
)
if box and box.found:
    print(box.x_min, box.y_min, box.x_max, box.y_max)
```

## 输入解析规则

1. 若输入包含 **「的」**：则「的」**后面**的片段作为要定位的文字（如 `#3阳床的0.0m3/h` → `0.0m3/h`）。
2. 否则匹配 **数值+单位** 模式（如 `12.5m3/h`），作为目标文字。
3. 否则整句作为目标文字。

## 说明

- 边界框坐标为**相对图片宽高的比例**（0~1），如需像素可乘以图片宽高。
- 本地图片通过 base64 传入；若你的部署不支持 data URL，请先将图片上传为可访问的 URL 再传 `image_url`。
- 模型需支持多模态（图+文）理解；你提供的 `doubao-seed-2-0-pro-260215` 为示例，请按控制台实际 endpoint 填写 `--model`。
