#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
豆包大模型视觉 API：根据用户输入（如 "#3阳床的0.0m3/h"）在图片中识别目标文字的边界框。
"""
import warnings
warnings.filterwarnings("ignore", message=".*doesn't match a supported version.*")
import re
import json
import os
import sys
import base64
import io
import requests
from typing import Optional
from dataclasses import dataclass


# 豆包 Responses API
ARK_API_URL = "https://ark.cn-beijing.volces.com/api/v3/responses"
DEFAULT_MODEL = "doubao-seed-2-0-pro-260215"


@dataclass
class BoundingBox:
    """边界框，坐标可为像素或比例 (0-1)。"""
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    found: bool = True

    def to_dict(self):
        return {"x_min": self.x_min, "y_min": self.y_min, "x_max": self.x_max, "y_max": self.y_max, "found": self.found}


def draw_box_on_image(
    image_source: str,
    box: BoundingBox,
    output_path: str,
    *,
    is_url: bool = False,
    line_width: int = 3,
    color: tuple = (255, 0, 0),
) -> str:
    """
    在图片上绘制边界框并保存。image_source 为本地路径或 URL；is_url=True 表示从 URL 下载。
    返回保存的绝对路径。
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        raise ImportError("请安装 Pillow: pip install Pillow")

    if is_url:
        r = requests.get(image_source, timeout=30)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
    else:
        img = Image.open(image_source).convert("RGB")
    w, h = img.size
    x1 = int(box.x_min * w)
    y1 = int(box.y_min * h)
    x2 = int(box.x_max * w)
    y2 = int(box.y_max * h)
    draw = ImageDraw.Draw(img)
    draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
    out = os.path.abspath(output_path)
    img.save(out)
    return out


def parse_user_input(user_input: str) -> tuple[str, str]:
    """
    从用户输入中解析出「要定位的文字」和可选上下文。
    - 多级描述用最后一个「的」分割：前面是定位描述，后面是要找的目标值。
    例如：
      "#3阳床的0.0m3/h" -> 目标 "0.0m3/h", 上下文 "#3阳床"
      "中间水泵P109A下面的反馈右边的0.00HZ" -> 目标 "0.00HZ", 上下文 "中间水泵P109A下面的反馈右边"
    """
    user_input = (user_input or "").strip()
    if not user_input:
        return "", ""

    # 规则1：若包含「的」，用最后一个「的」分割，则「的」后面是要定位的目标值，前面是描述
    if "的" in user_input:
        parts = user_input.rsplit("的", 1)  # 从右边只分一次
        context = parts[0].strip()
        target = parts[1].strip()
        if target:
            return target, context

    # 规则2：匹配常见数值+单位，如 0.0m3/h、0.00HZ、12.5m3/h
    value_unit = re.search(r"[\d.]+(?:m\d*/?h|HZ|Hz|hz)", user_input, re.IGNORECASE)
    if value_unit:
        return value_unit.group(0), user_input.replace(value_unit.group(0), "").strip()

    # 规则3：整句作为目标
    return user_input, ""


def build_grounding_prompt(target_text: str, context: str) -> str:
    """构建让模型返回边界框的提示词。"""
    if context:
        prompt = f"图片中与「{context}」相关的文字是「{target_text}」。请定位「{target_text}」在图片中的精确位置。"
    else:
        prompt = f"请在该图片中定位文字「{target_text}」的精确位置。"
    prompt += (
        "\n请只返回一个 JSON 对象，不要其他说明。"
        "\n若找到，格式为：{\"found\": true, \"x_min\": 0.0, \"y_min\": 0.0, \"x_max\": 0.0, \"y_max\": 0.0}，"
        "坐标为相对图片宽高的比例 (0~1)，即左上角 (x_min,y_min)、右下角 (x_max,y_max)。"
        "\n若未找到该文字，返回：{\"found\": false}。"
    )
    return prompt


def call_doubao_vision(
    image_url: Optional[str] = None,
    image_path: Optional[str] = None,
    image_base64: Optional[str] = None,
    prompt: str = "",
    *,
    api_key: str,
    model: str = DEFAULT_MODEL,
    timeout: int = 180,
) -> dict:
    """
    调用豆包 /api/v3/responses 多模态接口。
    图片三选一：image_url（公网 URL）、image_path（本地路径）、image_base64（base64 字符串）。
    timeout: 请求超时秒数，视觉模型可能较慢，默认 180。
    """
    content: list = []

    # 图片内容：优先 URL，其次本地文件转 base64（若 API 支持 input_image 的 url 为 data URL）
    if image_url:
        content.append({"type": "input_image", "image_url": image_url})
    elif image_path and os.path.isfile(image_path):
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        # 按扩展名设置 MIME，否则 API 可能无法正确解析 PNG 等格式
        ext = os.path.splitext(image_path)[1].lower()
        mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif", ".webp": "image/webp"}.get(ext, "image/jpeg")
        data_url = f"data:{mime};base64,{b64}"
        content.append({"type": "input_image", "image_url": data_url})
    elif image_base64:
        if not image_base64.startswith("data:"):
            image_base64 = f"data:image/jpeg;base64,{image_base64}"
        content.append({"type": "input_image", "image_url": image_base64})
    else:
        raise ValueError("必须提供 image_url、image_path 或 image_base64 之一")

    content.append({"type": "input_text", "text": prompt})

    body = {
        "model": model,
        "input": [
            {"role": "user", "content": content}
        ],
    }

    resp = requests.post(
        ARK_API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=body,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def _collect_text_from_obj(obj, texts: list) -> None:
    """递归从 dict/list 中收集所有字符串（用于兜底提取模型输出）。"""
    if isinstance(obj, str) and obj.strip():
        texts.append(obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("text", "content", "message") and isinstance(v, str) and v.strip():
                texts.append(v)
            _collect_text_from_obj(v, texts)
    elif isinstance(obj, list):
        for x in obj:
            _collect_text_from_obj(x, texts)


def _extract_text_from_response(api_response: dict) -> str:
    """从多种可能的 API 响应结构中提取文本内容。"""
    # Responses API: output.text
    if "output" in api_response:
        out = api_response["output"]
        if isinstance(out, dict) and "text" in out and out["text"]:
            return out["text"]
        if isinstance(out, dict) and "output_items" in out:
            for item in out.get("output_items", []):
                if isinstance(item, dict) and item.get("type") == "message" and "content" in item:
                    for c in item.get("content", []):
                        if isinstance(c, dict) and c.get("type") == "output_text":
                            t = c.get("text") or ""
                            if t:
                                return t
    # Chat 兼容: choices[].message.content
    if "choices" in api_response and len(api_response["choices"]) > 0:
        c = api_response["choices"][0]
        if "message" in c:
            content = c["message"].get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        t = part.get("text", "") or ""
                        if t:
                            return t
    # 兜底：递归收集所有可能是模型回复的文本，取最长一段（通常是主回复）
    collected: list = []
    _collect_text_from_obj(api_response, collected)
    if collected:
        return max(collected, key=len)
    return ""


def parse_box_from_response(api_response: dict) -> Optional[BoundingBox]:
    """
    从豆包 API 返回的 JSON 中解析出边界框。
    支持 output.text、output.output_items、choices[].message.content 等结构。
    """
    text = _extract_text_from_response(api_response)
    try:
        if not text:
            return None

        # 尝试从文本中提取 JSON 对象（兼容被 markdown 包裹或前后有说明文字的情况）
        text = text.strip()
        if "```json" in text:
            text = text.split("```json")[-1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip() if text.count("```") >= 2 else text
        # 若整段不是合法 JSON，尝试匹配含 x_min/y_min/x_max/y_max 的 JSON 对象（允许内部有嵌套或换行）
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            obj = None
            # 匹配 {...} 且内含 x_min（允许中间有任意字符）
            for m in re.finditer(r'\{[^{}]*(?:"x_min"[^{}]*"y_min"[^{}]*"x_max"[^{}]*"y_max"[^{}]*)[^{}]*\}', text):
                try:
                    obj = json.loads(m.group(0))
                    break
                except json.JSONDecodeError:
                    continue
            if obj is None:
                # 匹配更宽松：任意包含 "x_min" 的 {...}（可能含嵌套）
                depth = 0
                start = text.find('{"x_min"')
                if start < 0:
                    start = text.find('"x_min"')
                    if start >= 0:
                        start = text.rfind('{', 0, start)
                if start >= 0:
                    end = start + 1
                    depth = 1
                    for i in range(start + 1, len(text)):
                        if text[i] == '{':
                            depth += 1
                        elif text[i] == '}':
                            depth -= 1
                            if depth == 0:
                                end = i + 1
                                break
                    if depth == 0:
                        try:
                            obj = json.loads(text[start:end])
                        except json.JSONDecodeError:
                            pass
        if not isinstance(obj, dict):
            return None
        if obj.get("found") is False:
            return BoundingBox(0, 0, 0, 0, found=False)
        x_min = float(obj.get("x_min", 0))
        y_min = float(obj.get("y_min", 0))
        x_max = float(obj.get("x_max", 0))
        y_max = float(obj.get("y_max", 0))
        return BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max, found=True)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def detect_text_box(
    user_input: str,
    image_url: Optional[str] = None,
    image_path: Optional[str] = None,
    image_base64: Optional[str] = None,
    *,
    api_key: str,
    model: str = DEFAULT_MODEL,
    timeout: int = 180,
    debug: bool = False,
) -> Optional[BoundingBox]:
    """
    主流程：解析用户输入 -> 调用豆包视觉 API -> 解析边界框。
    user_input: 例如 "#3阳床的0.0m3/h"
    图片三选一：image_url / image_path / image_base64。
    timeout: API 请求超时秒数。
    debug=True 时在解析失败时打印原始 API 响应到 stderr。
    """
    target_text, context = parse_user_input(user_input)
    if not target_text:
        return None
    prompt = build_grounding_prompt(target_text, context)
    response = call_doubao_vision(
        image_url=image_url,
        image_path=image_path,
        image_base64=image_base64,
        prompt=prompt,
        api_key=api_key,
        model=model,
        timeout=timeout,
    )
    box = parse_box_from_response(response)
    if box is None and debug:
        print(" [DEBUG] API 原始响应:", file=sys.stderr)
        print(json.dumps(response, ensure_ascii=False, indent=2)[:8000], file=sys.stderr)
    return box


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="根据输入在图片中识别目标文字的边界框（豆包 API）。"
        " 多级描述用「的」连接，最后一个「的」后面是要找的目标值。"
    )
    parser.add_argument(
        "input",
        help='用户输入。例: "#3阳床的0.0m3/h" 或 "中间水泵P109A下面的反馈右边的0.00HZ"'
    )
    parser.add_argument("--image-url", help="图片公网 URL")
    parser.add_argument("--image-path", help="本地图片路径")
    parser.add_argument("--api-key", default=os.environ.get("DOUBAO_ARK_API_KEY"), help="豆包 Ark API Key")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="模型 endpoint 名称")
    parser.add_argument("--debug", action="store_true", help="解析失败时打印 API 原始响应")
    parser.add_argument("--output-image", "-o", metavar="PATH", help="将画好边界框的图片保存到该路径，便于检查")
    parser.add_argument("--timeout", type=int, default=180, metavar="SEC", help="API 请求超时秒数，默认 180（视觉模型较慢）")
    args = parser.parse_args()

    if not args.api_key:
        print("错误：请设置 --api-key 或环境变量 DOUBAO_ARK_API_KEY")
        return 1
    if not args.image_url and not args.image_path:
        print("错误：请提供 --image-url 或 --image-path")
        return 1

    box = detect_text_box(
        args.input,
        image_url=args.image_url,
        image_path=args.image_path,
        api_key=args.api_key,
        model=args.model,
        timeout=args.timeout,
        debug=args.debug,
    )
    if box is None:
        print("未能解析出边界框", end="")
        if not args.debug:
            print("（可加 --debug 查看 API 原始响应）", end="")
        print()
        return 1
    print(json.dumps(box.to_dict(), ensure_ascii=False, indent=2))
    # 成功得到 box 时，自动在图片上画出框并保存（默认路径若未指定 -o）
    if box.found:
        image_source = args.image_path or args.image_url
        if image_source:
            out_path = args.output_image
            if not out_path:
                if args.image_path:
                    base, ext = os.path.splitext(args.image_path)
                    out_path = f"{base}_box{ext}" if base else "result_box.jpg"
                else:
                    out_path = "result_box.jpg"
            try:
                saved = draw_box_on_image(
                    image_source,
                    box,
                    out_path,
                    is_url=bool(args.image_url),
                )
                print(f"已保存带框图片: {saved}")
            except Exception as e:
                print(f"保存带框图片失败: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    exit(main() or 0)
