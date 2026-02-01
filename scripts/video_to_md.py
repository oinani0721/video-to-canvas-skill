"""
混合方案：Gemini 视觉检测 + BiliNote 提示词架构

两阶段管道：
1. 阶段1：Gemini 视觉检测变化点（使用 JSON Schema 强制输出）
   - 可选：启用双通道融合模式（视觉 + 语义交叉验证）
2. 阶段2：基于变化点生成高质量笔记（借鉴 BiliNote 9 种风格）

用法:
    python video_to_md.py <视频路径> [选项]

示例:
    python video_to_md.py "教程.mp4" --style tutorial
    python video_to_md.py "讲座.mp4" --preset lecture
    python video_to_md.py "演示.mp4" --style business --fusion  # 启用双通道融合
"""

from google import genai
from google.genai import types
import subprocess
import json
import os
import re
import sys
import time
import argparse

# 尝试从 lore-engine 的 .env 加载 API Key
LORE_ENGINE_ENV = os.path.expanduser("~/lore-engine/.env")
if os.path.exists(LORE_ENGINE_ENV):
    with open(LORE_ENGINE_ENV, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                if key.startswith("GEMINI_API_KEY") and key not in os.environ:
                    os.environ[key] = value

from styles import STYLES, VIDEO_PRESETS, get_style_prompt, list_styles, list_presets
from prompt_builder import (
    PromptBuilder,
    CHANGE_DETECTION_PROMPT,
    CHANGE_DETECTION_SCHEMA
)
from prompt_builder_v2 import (
    PromptBuilderV2,
    create_tutorial_notes_prompt,
    create_lecture_notes_prompt,
    create_quick_notes_prompt,
    create_deep_notes_prompt,
    DEPTH_LEVELS
)


def parse_timestamp_to_seconds(timestamp: str) -> float:
    """将 MM:SS 或 HH:MM:SS 格式转换为秒数"""
    parts = timestamp.split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    return 0


def seconds_to_timestamp(seconds: float) -> str:
    """将秒数转换为 MM:SS 格式"""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


def extract_screenshot(video_path: str, timestamp: str, output_file: str) -> bool:
    """使用 ffmpeg 从视频中提取指定时间的截图"""
    cmd = [
        "ffmpeg", "-y",
        "-ss", timestamp,
        "-i", video_path,
        "-frames:v", "1",
        "-q:v", "2",
        output_file
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def wait_for_processing(client, video_file, max_wait: int = 300):
    """等待视频处理完成"""
    start_time = time.time()
    while video_file.state.name == "PROCESSING":
        elapsed = time.time() - start_time
        if elapsed > max_wait:
            raise TimeoutError(f"视频处理超时（>{max_wait}秒）")
        print(f"  视频处理中... ({int(elapsed)}s)")
        time.sleep(5)
        video_file = client.files.get(name=video_file.name)

    if video_file.state.name == "FAILED":
        raise RuntimeError("视频处理失败")

    return video_file


def filter_change_points(change_points: list, min_interval: float = 2.0) -> list:
    """
    过滤变化点，确保最小间隔

    Args:
        change_points: 变化点列表
        min_interval: 最小间隔（秒）

    Returns:
        过滤后的变化点列表
    """
    if not change_points:
        return []

    filtered = [change_points[0]]
    last_time = parse_timestamp_to_seconds(change_points[0]["timestamp"])

    for cp in change_points[1:]:
        current_time = parse_timestamp_to_seconds(cp["timestamp"])
        if current_time - last_time >= min_interval:
            filtered.append(cp)
            last_time = current_time

    return filtered


def phase1_detect_changes(client, video_file, density: str = "normal") -> dict:
    """
    阶段1：Gemini 视觉检测变化点

    Args:
        client: Gemini 客户端
        video_file: 上传的视频文件
        density: 检测密度 (sparse/normal/dense)

    Returns:
        包含 change_points 的字典
    """
    # 根据密度调整提示词
    density_hints = {
        "sparse": "\n\n密度要求：每分钟 1-3 个变化点，只保留最重要的变化。",
        "normal": "\n\n密度要求：每分钟 3-6 个变化点，平衡覆盖度和精简度。",
        "dense": "\n\n密度要求：每分钟 6-12 个变化点，尽可能捕捉所有变化。"
    }

    prompt = CHANGE_DETECTION_PROMPT + density_hints.get(density, density_hints["normal"])

    print("阶段 1: Gemini 视频分析 - 检测画面变化点...")

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[video_file, prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=CHANGE_DETECTION_SCHEMA
        )
    )

    # 解析 JSON 响应
    try:
        result = json.loads(response.text)
    except json.JSONDecodeError as e:
        print(f"警告：JSON 解析失败，尝试修复...")
        # 尝试提取 JSON
        text = response.text.strip()
        text = re.sub(r'^```json\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        result = json.loads(text)

    return result


def phase2_generate_notes(
    client,
    screenshots: list,
    style: str = "tutorial",
    video_summary: str = "",
    use_v2: bool = True,
    depth: str = "balanced"
) -> str:
    """
    阶段2：生成高质量笔记

    Args:
        client: Gemini 客户端
        screenshots: 截图信息列表
        style: 笔记风格 (v1) 或内容模式 (v2)
        video_summary: 视频摘要（来自阶段1）
        use_v2: 是否使用 V2 架构（Lore Engine 风格）
        depth: 输出深度 (short_hand/balanced/deep_dive)

    Returns:
        生成的 Markdown 文本
    """
    if use_v2:
        # V2 架构：Lore Engine 风格，分层组织，不漏任何知识点
        print(f"阶段 2: 生成笔记 (V2 架构, 深度: {depth})...")

        # 根据 style 选择内容模式
        mode_map = {
            "tutorial": "video_tutorial",
            "lecture": "lecture",
            "academic": "lecture",
            "code": "code_demo",
        }
        mode = mode_map.get(style, "video_tutorial")

        # 构建 V2 提示词
        prompt = (PromptBuilderV2()
                  .with_mode(mode)
                  .with_depth(depth)
                  .with_hierarchy()
                  .with_screenshots(screenshots)
                  .with_inference()
                  .with_summary()
                  .build())

        # 如果有视频摘要，添加到提示词开头
        if video_summary:
            prompt = f"## 视频主题\n{video_summary}\n\n" + prompt

    else:
        # V1 架构：原有的 BiliNote 风格
        print(f"阶段 2: 生成笔记 (V1 风格: {style})...")

        builder = (PromptBuilder()
                   .with_style(style)
                   .with_screenshots(screenshots)
                   .with_timestamps()
                   .with_ai_summary())

        if video_summary:
            builder.with_custom("视频概述", f"本视频主题：{video_summary}")

        prompt = builder.build()

    # 加载截图图片
    images = []
    for s in screenshots:
        if os.path.exists(s["path"]):
            with open(s["path"], "rb") as f:
                image_data = f.read()
            images.append(types.Part.from_bytes(data=image_data, mime_type="image/jpeg"))

    # 发送截图 + 提示词给 Gemini
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[*images, prompt]
    )

    return response.text


def main(
    video_path: str,
    output_dir: str = "output",
    style: str = "tutorial",
    preset: str = None,
    density: str = "normal",
    min_interval: float = 2.0,
    fusion: bool = False,
    min_confidence: float = 0.0,
    use_v2: bool = True,
    depth: str = "balanced"
):
    """
    混合方案主函数

    Args:
        video_path: 视频文件路径
        output_dir: 输出目录
        style: 笔记风格
        preset: 视频类型预设
        density: 变化检测密度
        min_interval: 最小截图间隔（秒）
        fusion: 是否启用双通道融合模式（视觉+语义交叉验证）
        min_confidence: 最小置信度阈值（仅融合模式有效）
        use_v2: 是否使用 V2 架构（Lore Engine 风格，推荐）
        depth: 输出深度 (short_hand/balanced/deep_dive)
    """
    # 检查视频文件
    if not os.path.exists(video_path):
        print(f"错误：视频文件不存在: {video_path}")
        sys.exit(1)

    # 应用预设配置
    if preset and preset in VIDEO_PRESETS:
        preset_config = VIDEO_PRESETS[preset]
        style = preset_config.get("style", style)
        min_interval = preset_config.get("min_interval", min_interval)
        print(f"应用预设: {preset_config['name']}")

    # 初始化客户端
    api_key = (
        os.getenv("GEMINI_API_KEY") or
        os.getenv("GEMINI_API_KEY_1") or
        os.getenv("GOOGLE_AI_API_KEY")
    )
    if not api_key:
        print("错误：请设置 GEMINI_API_KEY 环境变量")
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    # 创建输出目录
    screenshot_dir = os.path.join(output_dir, "screenshots")
    os.makedirs(screenshot_dir, exist_ok=True)

    # ========== 上传视频 ==========
    print(f"正在上传视频: {video_path}")
    file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
    print(f"  文件大小: {file_size_mb:.1f} MB")

    video_file = client.files.upload(file=video_path)
    print("上传完成，正在等待处理...")
    video_file = wait_for_processing(client, video_file)
    print("视频处理完成！")

    # ========== 阶段1：变化检测 ==========
    if fusion:
        # 双通道融合模式：视觉 + 语义交叉验证
        from fusion_detector import FusionDetector, analyze_detection_quality

        print("\n[融合模式] 启用双通道交叉验证")

        detector = FusionDetector(client, video_file)
        fused_points = detector.detect_and_fuse(
            visual_prompt=CHANGE_DETECTION_PROMPT,
            visual_schema=CHANGE_DETECTION_SCHEMA,
            min_confidence=min_confidence
        )

        # 转换为兼容格式
        change_points = detector.to_legacy_format()
        video_summary = ""

        # 输出分析结果
        analysis = analyze_detection_quality(
            detector.visual_points,
            detector.semantic_points,
            detector.fused_points
        )
        print(f"\n[分析] 匹配率: {analysis['fused_ratio']:.1%}")
        for suggestion in analysis.get("suggestions", []):
            print(f"  建议: {suggestion}")
    else:
        # 单通道模式：仅视觉检测
        result = phase1_detect_changes(client, video_file, density)
        change_points = result.get("change_points", [])
        video_summary = result.get("video_summary", "")

    print(f"检测到 {len(change_points)} 个原始变化点")

    # 过滤变化点
    change_points = filter_change_points(change_points, min_interval)
    print(f"过滤后保留 {len(change_points)} 个变化点（最小间隔: {min_interval}s）")

    if not change_points:
        print("警告：未检测到有效变化点，使用备用策略...")
        # 备用策略：每 30 秒一个截图
        # TODO(human): 实现备用截图策略
        # 当 Gemini 未能检测到变化点时，需要一个回退方案
        # 可以考虑：1) 固定间隔截图 2) 使用 FFmpeg 场景检测
        change_points = [{"timestamp": "00:00", "change_type": "other", "description": "视频开始"}]

    # ========== 提取截图 ==========
    print("正在提取截图...")
    screenshots = []
    for i, cp in enumerate(change_points):
        ts = cp["timestamp"]
        safe_ts = ts.replace(":", "-")
        output_file = os.path.join(screenshot_dir, f"{safe_ts}.jpg")

        success = extract_screenshot(video_path, ts, output_file)
        status = "[OK]" if success else "[FAIL]"
        print(f"  [{i+1}/{len(change_points)}] {status} {ts} - {cp['description'][:30]}")

        if success:
            screenshots.append({
                "timestamp": ts,
                "path": output_file,
                "desc": cp["description"],
                "type": cp["change_type"]
            })

    # ========== 阶段2：笔记生成 ==========
    markdown_text = phase2_generate_notes(
        client, screenshots, style, video_summary,
        use_v2=use_v2, depth=depth
    )

    # ========== 保存结果 ==========
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    md_path = os.path.join(output_dir, f"{video_name}.md")

    # 添加头部信息
    if use_v2:
        arch_info = f"V2 架构 (Lore Engine 风格), 深度: {depth}"
    else:
        arch_info = f"V1 架构, 风格: {STYLES.get(style, {}).get('name', style)}"

    header = f"""# {video_name}

> 由 Gemini 视觉检测 + 智能笔记生成
> 架构: {arch_info}
> 截图: {len(screenshots)} 张

---

"""

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(header + markdown_text)

    # 保存变化点信息（用于调试）
    debug_path = os.path.join(output_dir, f"{video_name}_changes.json")
    with open(debug_path, "w", encoding="utf-8") as f:
        json.dump({
            "video_summary": video_summary,
            "change_points": change_points,
            "style": style,
            "density": density
        }, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"[DONE] 完成！")
    print(f"   Markdown: {md_path}")
    print(f"   Screenshots: {screenshot_dir}")
    print(f"   截图: {len(screenshots)} 张")
    if use_v2:
        print(f"   架构: V2 (Lore Engine 风格)")
        print(f"   深度: {depth}")
    else:
        print(f"   架构: V1")
        print(f"   风格: {STYLES.get(style, {}).get('name', style)}")
    print(f"{'='*60}")

    return md_path


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="视频转笔记：Gemini 视觉检测 + 智能笔记生成",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # V2 架构（推荐，按知识结构组织）
  python video_to_md.py "教程.mp4"                              # 默认 V2 + balanced
  python video_to_md.py "教程.mp4" --depth deep_dive            # 深度模式
  python video_to_md.py "教程.mp4" --depth short_hand           # 极简模式

  # V1 架构（原 BiliNote 风格，按时间点组织）
  python video_to_md.py "教程.mp4" --v1 --style academic

  # 高级选项
  python video_to_md.py "视频.mp4" --fusion                     # 双通道融合
  python video_to_md.py "视频.mp4" --density dense              # 密集截图

V2 深度: short_hand (极简) | balanced (平衡,推荐) | deep_dive (深度)
V1 风格: minimal, detailed, academic, tutorial, business, outline, qa, summary, annotated
        """
    )

    parser.add_argument("video", nargs="?", help="视频文件路径")
    parser.add_argument("-o", "--output", default="output", help="输出目录 (默认: output)")

    # 架构选择
    arch_group = parser.add_mutually_exclusive_group()
    arch_group.add_argument("--v2", action="store_true", default=True,
                            help="使用 V2 架构 (Lore Engine 风格，推荐，默认)")
    arch_group.add_argument("--v1", action="store_true",
                            help="使用 V1 架构 (BiliNote 风格，按时间点组织)")

    # V2 参数
    parser.add_argument("--depth", default="balanced",
                        choices=["short_hand", "balanced", "deep_dive"],
                        help="V2 输出深度 (默认: balanced)")

    # V1 参数
    parser.add_argument("-s", "--style", default="tutorial",
                        choices=list(STYLES.keys()),
                        help="V1 笔记风格 (默认: tutorial)")

    # 通用参数
    parser.add_argument("-p", "--preset",
                        choices=list(VIDEO_PRESETS.keys()),
                        help="视频类型预设")
    parser.add_argument("-d", "--density", default="normal",
                        choices=["sparse", "normal", "dense"],
                        help="变化检测密度 (默认: normal)")
    parser.add_argument("--min-interval", type=float, default=2.0,
                        help="最小截图间隔秒数 (默认: 2.0)")
    parser.add_argument("--fusion", action="store_true",
                        help="启用双通道融合模式（视觉+语义交叉验证）")
    parser.add_argument("--min-confidence", type=float, default=0.0,
                        help="最小置信度阈值，仅融合模式有效 (默认: 0.0)")
    parser.add_argument("--list-styles", action="store_true",
                        help="列出所有可用风格")
    parser.add_argument("--list-presets", action="store_true",
                        help="列出所有可用预设")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 列出风格/预设
    if args.list_styles:
        list_styles()
        sys.exit(0)
    if args.list_presets:
        list_presets()
        sys.exit(0)

    # 检查视频参数
    if not args.video:
        print("错误：请提供视频文件路径")
        print("用法：python video_to_md.py <视频路径> [选项]")
        print("使用 --help 查看所有选项")
        sys.exit(1)

    # 确定使用的架构
    use_v2 = not args.v1  # 默认 V2，除非指定 --v1

    # 运行主函数
    main(
        video_path=args.video,
        output_dir=args.output,
        style=args.style,
        preset=args.preset,
        density=args.density,
        min_interval=args.min_interval,
        fusion=args.fusion,
        min_confidence=args.min_confidence,
        use_v2=use_v2,
        depth=args.depth
    )
