"""
三阶段混合管道：WhisperX/Gemini 转录 + Gemini 视觉检测 + LLM 笔记生成

三阶段架构（社区验证最佳实践）：
1. Stage 1 (Ears): 音频转录 — WhisperX (首选) 或 Gemini Audio (备选)
2. Stage 2 (Eyes): Gemini 视觉检测变化点 + FFmpeg 截图提取
3. Stage 3 (Brain): LLM 融合转录文本 + 截图 → 结构化笔记

解决的核心问题：
- 旧架构只发送截图给 LLM，完全丢失音频内容
- Gemini 直接处理 >20 分钟视频有严重幻觉风险
- 新架构通过 15 分钟分段 + 双通道融合避免信息丢失

用法:
    python video_to_md.py <视频路径> [选项]

示例:
    python video_to_md.py "教程.mp4"                              # 默认三阶段管道
    python video_to_md.py "讲座.mp4" --depth deep_dive            # 深度模式
    python video_to_md.py "视频.mp4" --no-transcribe              # 跳过转录（旧模式）
    python video_to_md.py "视频.mp4" --backend gemini             # 强制用 Gemini 转录
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
import random

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
from transcriber import transcribe, save_transcript, load_transcript, TranscriptResult
from srt_generator import generate_srt_from_transcript, translate_srt_file


def get_video_duration(video_path: str) -> float:
    """使用 ffprobe 获取视频时长（秒）"""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except (ValueError, FileNotFoundError):
        pass
    return 0.0


def fill_coverage_gaps(change_points: list, video_duration: float, gap_interval: float = 30.0) -> list:
    """
    检查变化点覆盖率，对未覆盖的时间段自动补充截图点。

    解决问题：Gemini 对 >20 分钟的视频只分析前半部分，
    后半段完全没有变化点和截图。

    Args:
        change_points: 已检测的变化点列表
        video_duration: 视频总时长（秒）
        gap_interval: 补充截图的间隔（秒），默认 30 秒

    Returns:
        补充后的变化点列表（按时间排序）
    """
    if not change_points or video_duration <= 0:
        return change_points

    # 找到最后一个变化点的时间
    last_cp_time = max(
        parse_timestamp_to_seconds(cp["timestamp"]) for cp in change_points
    )

    # 计算覆盖率
    coverage = last_cp_time / video_duration

    if coverage >= 0.85:
        # 覆盖率足够，无需补充
        return change_points

    # 覆盖率不足，需要补充
    uncovered_start = last_cp_time + gap_interval
    uncovered_duration = video_duration - uncovered_start

    if uncovered_duration <= 0:
        return change_points

    print(f"\n[!] Stage 2 coverage check:")
    print(f"  Video duration: {video_duration/60:.1f} min")
    print(f"  Last change point: {seconds_to_timestamp(last_cp_time)}")
    print(f"  Coverage: {coverage:.0%} -- {uncovered_duration/60:.1f} min uncovered")
    print(f"  -> Auto-filling screenshots (every {gap_interval:.0f}s)...")

    # 生成补充变化点
    fill_points = []
    t = uncovered_start
    while t < video_duration - 5:  # 留 5 秒余量避免越界
        fill_points.append({
            "timestamp": seconds_to_timestamp(t),
            "change_type": "auto_fill",
            "description": "Auto-fill screenshot (uncovered segment)"
        })
        t += gap_interval

    print(f"  Added {len(fill_points)} fill points "
          f"({seconds_to_timestamp(uncovered_start)} ~ {seconds_to_timestamp(min(t, video_duration))})")

    # 合并并按时间排序
    all_points = change_points + fill_points
    all_points.sort(key=lambda cp: parse_timestamp_to_seconds(cp["timestamp"]))

    return all_points


def validate_image_references(markdown_text: str, screenshot_dir: str) -> str:
    """
    验证 Markdown 中的图片引用，移除指向不存在文件的引用。

    解决问题：Stage 3 (Gemini) 有时会生成指向不存在截图的引用，
    例如视频只有 56 分钟但生成了 67:21 时间戳的截图引用。

    Args:
        markdown_text: Markdown 文本
        screenshot_dir: 截图目录路径

    Returns:
        清理后的 Markdown 文本
    """
    # 匹配 ![任意描述](screenshots/XX-XX.jpg) 格式
    pattern = r'!\[([^\]]*)\]\((screenshots/[^)]+)\)'

    removed = []

    def check_ref(match):
        desc = match.group(1)
        ref_path = match.group(2)
        # screenshots/XX-XX.jpg -> 拼接为实际路径
        full_path = os.path.join(os.path.dirname(screenshot_dir), ref_path)
        if not os.path.exists(full_path):
            # 也检查 screenshot_dir 内
            alt_path = os.path.join(screenshot_dir, os.path.basename(ref_path))
            if not os.path.exists(alt_path):
                removed.append(ref_path)
                return ""  # 移除无效引用
        return match.group(0)  # 保留有效引用

    cleaned = re.sub(pattern, check_ref, markdown_text)

    if removed:
        # 清理可能残留的空行
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        print(f"  [Post-process] Removed {len(removed)} invalid image refs:")
        for r in removed:
            print(f"    - {r}")

    return cleaned


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

    response_text = _gemini_generate_with_retry(
        client, "gemini-2.5-flash",
        [video_file, prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=CHANGE_DETECTION_SCHEMA
        )
    )

    # 解析 JSON 响应
    try:
        result = json.loads(response_text)
    except json.JSONDecodeError as e:
        print(f"警告：JSON 解析失败，尝试修复...")
        # 尝试提取 JSON
        text = response_text.strip()
        text = re.sub(r'^```json\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        result = json.loads(text)

    return result


def format_transcript_for_prompt(transcript_result: TranscriptResult, start: float = 0, end: float = float('inf')) -> str:
    """
    将转录结果格式化为提示词中的文本块

    Args:
        transcript_result: 转录结果
        start: 起始时间（秒）
        end: 结束时间（秒）

    Returns:
        格式化的转录文本
    """
    lines = []
    for seg in transcript_result.segments:
        if seg.end > start and seg.start < end:
            ts_start = seconds_to_timestamp(seg.start)
            ts_end = seconds_to_timestamp(seg.end)
            lines.append(f"[{ts_start}-{ts_end}] {seg.text}")
    return "\n".join(lines)


def phase2_generate_notes(
    client,
    screenshots: list,
    style: str = "tutorial",
    video_summary: str = "",
    use_v2: bool = True,
    depth: str = "balanced",
    transcript_result: TranscriptResult = None,
    segment_minutes: float = 15.0,
    video_duration: float = 0.0
) -> str:
    """
    Stage 3 (Brain)：融合转录文本 + 截图生成高质量笔记

    Args:
        client: Gemini 客户端
        screenshots: 截图信息列表
        style: 笔记风格 (v1) 或内容模式 (v2)
        video_summary: 视频摘要（来自阶段1）
        use_v2: 是否使用 V2 架构（Lore Engine 风格）
        depth: 输出深度 (short_hand/balanced/deep_dive)
        transcript_result: 转录结果（三阶段管道核心）
        segment_minutes: 分段时长（分钟），避免 Gemini 长上下文幻觉
        video_duration: 视频实际时长（秒，来自 ffprobe），用于校正转录时长幻觉

    Returns:
        生成的 Markdown 文本
    """
    # 判断是否需要分段处理
    # 优先使用 ffprobe 时长（可靠），回退到转录时长（可能有 Gemini 幻觉）
    effective_duration = video_duration if video_duration > 0 else (
        transcript_result.duration if transcript_result else 0
    )

    if transcript_result and effective_duration > segment_minutes * 60 * 1.2:
        # 视频超过分段阈值 120%，启用分段模式
        return _generate_notes_segmented(
            client, screenshots, style, video_summary,
            use_v2, depth, transcript_result, segment_minutes,
            video_duration
        )

    # 单段处理（视频较短或无转录）
    return _generate_notes_single(
        client, screenshots, style, video_summary,
        use_v2, depth, transcript_result
    )


def _gemini_generate_with_retry(client, model: str, contents: list, max_retries: int = 5, **kwargs):
    """
    带指数退避重试的 Gemini API 调用，处理 429 速率限制错误。

    Gemini 2.5 Flash 有 1,000,000 token/min 限制，分段生成时容易触发。
    API 返回的 429 错误会包含建议等待时间 (如 "retry in 33.8s")。
    额外的 kwargs (如 config) 会传递给 generate_content。
    """
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(model=model, contents=contents, **kwargs)
            return response.text
        except Exception as e:
            error_str = str(e)
            is_rate_limit = "429" in error_str or "RESOURCE_EXHAUSTED" in error_str

            if not is_rate_limit or attempt == max_retries - 1:
                raise

            # 从错误消息中提取建议等待时间
            wait_time = 40.0  # 默认等待
            import re as _re
            match = _re.search(r"retry.*?(\d+\.?\d*)\s*s", error_str, _re.IGNORECASE)
            if match:
                wait_time = float(match.group(1)) + 5  # 加 5 秒缓冲

            # 指数退避 + 抖动
            backoff = min(wait_time * (1.5 ** attempt), 120)
            jitter = random.uniform(0, 5)
            total_wait = backoff + jitter

            print(f"    [Rate Limit] 429 触发，等待 {total_wait:.1f}s 后重试 "
                  f"(attempt {attempt + 1}/{max_retries})...")
            time.sleep(total_wait)

    raise RuntimeError("Gemini API 重试次数已耗尽")


def _generate_notes_single(
    client,
    screenshots: list,
    style: str,
    video_summary: str,
    use_v2: bool,
    depth: str,
    transcript_result: TranscriptResult = None
) -> str:
    """单段笔记生成（用于短视频或单个分段）"""

    # 格式化转录文本
    transcript_text = ""
    if transcript_result:
        transcript_text = format_transcript_for_prompt(transcript_result)

    if use_v2:
        has_transcript = bool(transcript_text)
        channel_info = "双通道: 转录+截图" if has_transcript else "单通道: 仅截图"
        print(f"[Stage 3: Brain] 生成笔记 (V2, {depth}, {channel_info})...")

        mode_map = {
            "tutorial": "video_tutorial",
            "lecture": "lecture",
            "academic": "lecture",
            "code": "code_demo",
        }
        mode = mode_map.get(style, "video_tutorial")

        builder = (PromptBuilderV2()
                   .with_mode(mode)
                   .with_depth(depth)
                   .with_hierarchy()
                   .with_screenshots(screenshots))

        # 核心改进：注入转录文本
        if transcript_text:
            builder.with_transcript(transcript_text)

        builder.with_inference().with_summary()
        prompt = builder.build()

        if video_summary:
            prompt = f"## 视频主题\n{video_summary}\n\n" + prompt

    else:
        print(f"[Stage 3: Brain] 生成笔记 (V1 风格: {style})...")

        builder = (PromptBuilder()
                   .with_style(style)
                   .with_screenshots(screenshots)
                   .with_timestamps()
                   .with_ai_summary())

        if video_summary:
            builder.with_custom("视频概述", f"本视频主题：{video_summary}")

        # V1 架构也支持转录文本（作为自定义块注入）
        if transcript_text:
            builder.with_custom("音频转录", transcript_text)

        prompt = builder.build()

    # 加载截图图片
    images = []
    for s in screenshots:
        if os.path.exists(s["path"]):
            with open(s["path"], "rb") as f:
                image_data = f.read()
            images.append(types.Part.from_bytes(data=image_data, mime_type="image/jpeg"))

    # 发送截图 + 转录文本 + 提示词给 Gemini（带速率限制重试）
    return _gemini_generate_with_retry(
        client, "gemini-2.5-flash", [*images, prompt]
    )


def _generate_notes_segmented(
    client,
    screenshots: list,
    style: str,
    video_summary: str,
    use_v2: bool,
    depth: str,
    transcript_result: TranscriptResult,
    segment_minutes: float,
    video_duration: float = 0.0
) -> str:
    """
    分段笔记生成（用于长视频，避免 Gemini 幻觉）

    将视频按 segment_minutes 切分，每段独立生成笔记，最后合并。
    video_duration 用于截断超出实际视频时长的 chunks（防止转录时长幻觉）。
    """
    chunks = transcript_result.get_chunks(max_duration=segment_minutes * 60)

    # 使用 ffprobe 时长截断超出实际视频时长的 chunks
    # 防止 Gemini Audio 幻觉导致的转录时长 > 实际视频时长
    if video_duration > 0:
        filtered_chunks = []
        for chunk in chunks:
            if chunk["start"] >= video_duration:
                # 整个 chunk 超出视频时长，丢弃
                continue
            if chunk["end"] > video_duration:
                # chunk 部分超出，截断到实际时长
                chunk = dict(chunk)
                chunk["end"] = video_duration
            filtered_chunks.append(chunk)

        if len(filtered_chunks) < len(chunks):
            print(f"  [Duration cap] Trimmed {len(chunks) - len(filtered_chunks)} chunks "
                  f"beyond video duration ({video_duration/60:.1f}min)")
            chunks = filtered_chunks

    total_chunks = len(chunks)

    print(f"[Stage 3: Brain] 长视频分段模式: {total_chunks} 个分段 (每段 ≤{segment_minutes}min)")

    all_notes = []

    for i, chunk in enumerate(chunks):
        chunk_start = chunk["start"]
        chunk_end = chunk["end"]
        chunk_duration = (chunk_end - chunk_start) / 60

        print(f"\n  分段 {i+1}/{total_chunks}: "
              f"[{seconds_to_timestamp(chunk_start)} - {seconds_to_timestamp(chunk_end)}] "
              f"({chunk_duration:.1f}min)")

        # 筛选该时间段内的截图
        chunk_screenshots = [
            s for s in screenshots
            if chunk_start - 5 <= parse_timestamp_to_seconds(s["timestamp"]) <= chunk_end + 5
        ]
        print(f"    截图: {len(chunk_screenshots)} 张")

        # 创建该段的转录子集
        chunk_transcript = TranscriptResult(
            segments=[s for s in transcript_result.segments
                      if s.end > chunk_start and s.start < chunk_end],
            backend=transcript_result.backend,
            language=transcript_result.language
        )

        # 生成该段笔记（分段间预防性延迟避免速率限制）
        if i > 0:
            print(f"    [Cooldown] 分段间等待 10s 避免速率限制...")
            time.sleep(10)

        segment_notes = _generate_notes_single(
            client, chunk_screenshots, style,
            video_summary if i == 0 else "",  # 只在第一段加摘要
            use_v2, depth, chunk_transcript
        )
        all_notes.append(segment_notes)
        print(f"    分段 {i+1}/{total_chunks} 完成")

    # 合并所有分段笔记
    if total_chunks == 1:
        return all_notes[0]

    # 多段合并：请 LLM 整合
    print(f"\n[Stage 3: Brain] 合并 {total_chunks} 个分段笔记...")

    merge_prompt = f"""你是笔记合并专家。以下是同一个视频的 {total_chunks} 个分段笔记，
请将它们合并为一份结构完整、无重复的笔记。

合并规则：
1. 保留所有知识点，不要遗漏
2. 去除分段边界处的重复内容
3. 重新组织为统一的章节结构
4. 保留所有截图引用
5. 保留所有时间戳
6. 输出使用中文，技术术语保留英文

"""
    for i, notes in enumerate(all_notes):
        merge_prompt += f"\n{'='*40}\n## 分段 {i+1}\n{'='*40}\n{notes}\n"

    return _gemini_generate_with_retry(
        client, "gemini-2.5-flash", [merge_prompt]
    )


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
    depth: str = "balanced",
    transcribe_audio: bool = True,
    transcribe_backend: str = "auto",
    whisper_model: str = "large-v3",
    segment_minutes: float = 15.0,
    transcript_path: str = None,
    generate_srt: bool = True,
    srt_translate_lang: str = None
):
    """
    三阶段混合管道主函数

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
        transcribe_audio: 是否转录音频（三阶段管道核心）
        transcribe_backend: 转录后端 ("auto"/"faster-whisper"/"gemini")
        whisper_model: Whisper 模型大小
        segment_minutes: 分段时长（分钟）
        transcript_path: 已有转录文件路径（跳过转录步骤）
        generate_srt: 是否生成 SRT 字幕文件（默认 True）
        srt_translate_lang: SRT 翻译目标语言（如 "zh"），None 则不翻译
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

    # ========== Stage 1 (Ears): 音频转录 ==========
    transcript_result = None
    if transcribe_audio:
        if transcript_path and os.path.exists(transcript_path):
            # 加载已有转录
            print(f"\n加载已有转录: {transcript_path}")
            transcript_result = load_transcript(transcript_path)
            print(f"  片段数: {len(transcript_result.segments)}, 后端: {transcript_result.backend}")
        else:
            transcript_result = transcribe(
                video_path=video_path,
                backend=transcribe_backend,
                model_size=whisper_model,
                gemini_client=client if transcribe_backend in ("auto", "gemini") else None
            )
            # 保存转录结果
            video_name_for_transcript = os.path.splitext(os.path.basename(video_path))[0]
            transcript_save_path = os.path.join(output_dir, f"{video_name_for_transcript}_transcript.json")
            save_transcript(transcript_result, transcript_save_path)
    else:
        print("\n[Stage 1: Ears] 已跳过音频转录 (--no-transcribe)")

    # ========== Stage 1.5: 生成 SRT 字幕文件 ==========
    srt_paths = {}
    if transcript_result and generate_srt:
        video_name_for_srt = os.path.splitext(os.path.basename(video_path))[0]
        srt_path = os.path.join(output_dir, f"{video_name_for_srt}.srt")

        print(f"\n[Stage 1.5: SRT] Generating subtitles...")
        generate_srt_from_transcript(transcript_result, srt_path)
        srt_paths["en"] = srt_path
        print(f"  SRT: {srt_path} ({len(transcript_result.segments)} segments)")
        print(f"  Backend: {transcript_result.backend} (timestamps {'accurate' if transcript_result.backend == 'faster-whisper' else 'may need offset'})")

        # 翻译字幕
        if srt_translate_lang:
            translated_srt_path = os.path.join(
                output_dir, f"{video_name_for_srt}.{srt_translate_lang}.srt"
            )
            print(f"  Translating to {srt_translate_lang}...")
            result = translate_srt_file(
                srt_path, translated_srt_path, srt_translate_lang,
                gemini_client=client
            )
            if result:
                srt_paths[srt_translate_lang] = result

    # ========== Stage 2 (Eyes): 上传视频 + 视觉检测 ==========
    print(f"\n正在上传视频: {video_path}")
    file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
    print(f"  文件大小: {file_size_mb:.1f} MB")

    video_file = client.files.upload(file=video_path)
    print("上传完成，正在等待处理...")
    video_file = wait_for_processing(client, video_file)
    print("视频处理完成！")

    # ========== Stage 2 (Eyes): 变化检测 ==========
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
        change_points = [{"timestamp": "00:00", "change_type": "other", "description": "视频开始"}]

    # ========== Stage 2.5: 覆盖率检查 + 自动补充 ==========
    # 解决 Gemini 长视频只分析前半部分的问题
    # 注意：优先使用 ffprobe 时长（最可靠），转录时长可能有 Gemini 幻觉
    video_duration = get_video_duration(video_path)
    if video_duration <= 0 and transcript_result and transcript_result.duration > 0:
        video_duration = transcript_result.duration

    if video_duration > 0:
        change_points = fill_coverage_gaps(change_points, video_duration, gap_interval=30.0)

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

    # ========== Stage 3 (Brain): 笔记生成 ==========
    markdown_text = phase2_generate_notes(
        client, screenshots, style, video_summary,
        use_v2=use_v2, depth=depth,
        transcript_result=transcript_result,
        segment_minutes=segment_minutes,
        video_duration=video_duration
    )

    # ========== 保存结果 ==========
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    md_path = os.path.join(output_dir, f"{video_name}.md")

    # 添加头部信息
    if use_v2:
        arch_info = f"V2 三阶段混合管道, 深度: {depth}"
    else:
        arch_info = f"V1 架构, 风格: {STYLES.get(style, {}).get('name', style)}"

    transcript_info = ""
    if transcript_result:
        # 使用 ffprobe 时长（更可靠），Gemini 转录时长可能有幻觉
        display_duration = video_duration if video_duration > 0 else transcript_result.duration
        transcript_info = (f"\n> 转录: {transcript_result.backend} | "
                          f"{len(transcript_result.segments)} 段 | "
                          f"{display_duration/60:.1f} 分钟")

    header = f"""# {video_name}

> 由三阶段混合管道生成 (Ears → Eyes → Brain)
> 架构: {arch_info}
> 截图: {len(screenshots)} 张{transcript_info}

---

"""

    # 后处理 1：修复 Gemini 输出中反引号包裹图片引用的 bug
    # Gemini 有时会输出 `![desc](path)` 而不是 ![desc](path)
    markdown_text = re.sub(r'`(!\[.*?\]\(screenshots/[^)]+\))`', r'\1', markdown_text)

    # 后处理 2：验证图片引用，移除指向不存在文件的引用
    # 解决 Gemini 幻觉生成超出视频实际时长的截图引用
    markdown_text = validate_image_references(markdown_text, screenshot_dir)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(header + markdown_text)

    # 保存变化点信息（用于调试）
    debug_path = os.path.join(output_dir, f"{video_name}_changes.json")
    debug_data = {
        "video_summary": video_summary,
        "change_points": change_points,
        "style": style,
        "density": density,
        "pipeline": "3-stage-hybrid",
        "transcript_backend": transcript_result.backend if transcript_result else "none",
        "transcript_segments": len(transcript_result.segments) if transcript_result else 0,
        "segment_minutes": segment_minutes
    }
    with open(debug_path, "w", encoding="utf-8") as f:
        json.dump(debug_data, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"[DONE] 三阶段混合管道完成！")
    print(f"   Markdown: {md_path}")
    print(f"   Screenshots: {screenshot_dir}")
    print(f"   截图: {len(screenshots)} 张")
    if transcript_result:
        print(f"   转录: {transcript_result.backend} ({len(transcript_result.segments)} 段)")
    else:
        print(f"   转录: 已跳过")
    if srt_paths:
        for lang, path in srt_paths.items():
            print(f"   字幕 ({lang}): {path}")
    if use_v2:
        print(f"   架构: V2 三阶段混合管道")
        print(f"   深度: {depth}")
    else:
        print(f"   架构: V1")
        print(f"   风格: {STYLES.get(style, {}).get('name', style)}")
    print(f"{'='*60}")

    return md_path


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="视频转笔记：三阶段混合管道 (Ears → Eyes → Brain)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 三阶段管道（推荐，音频+视觉双通道）
  python video_to_md.py "教程.mp4"                              # 默认: 自动选择转录后端
  python video_to_md.py "讲座.mp4" --depth deep_dive            # 深度模式
  python video_to_md.py "视频.mp4" --backend gemini             # 强制 Gemini 转录

  # 跳过转录（旧模式，仅截图）
  python video_to_md.py "教程.mp4" --no-transcribe

  # 使用已有转录
  python video_to_md.py "教程.mp4" --transcript 教程_transcript.json

  # V1 架构（原 BiliNote 风格，按时间点组织）
  python video_to_md.py "教程.mp4" --v1 --style academic

  # 高级选项
  python video_to_md.py "视频.mp4" --fusion                     # 视觉双通道融合
  python video_to_md.py "视频.mp4" --density dense              # 密集截图
  python video_to_md.py "视频.mp4" --segment-minutes 10         # 10分钟分段

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
    # 转录参数（三阶段管道）
    transcript_group = parser.add_argument_group("转录选项 (Stage 1: Ears)")
    transcript_group.add_argument("--no-transcribe", action="store_true",
                                  help="跳过音频转录（退回旧模式，仅截图）")
    transcript_group.add_argument("--backend", default="auto",
                                  choices=["auto", "faster-whisper", "gemini"],
                                  help="转录后端 (默认: auto, 优先 faster-whisper)")
    transcript_group.add_argument("--whisper-model", default="large-v3",
                                  help="Whisper 模型大小 (默认: large-v3)")
    transcript_group.add_argument("--segment-minutes", type=float, default=15.0,
                                  help="长视频分段时长（分钟，默认 15）")
    transcript_group.add_argument("--transcript", default=None,
                                  help="已有转录文件路径（跳过转录步骤）")

    # SRT 字幕参数
    srt_group = parser.add_argument_group("SRT 字幕选项")
    srt_group.add_argument("--no-srt", action="store_true",
                           help="不生成 SRT 字幕文件")
    srt_group.add_argument("--srt-lang", default=None,
                           help="SRT 翻译目标语言 (如 zh, ja, ko)，默认不翻译")

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
        depth=args.depth,
        transcribe_audio=not args.no_transcribe,
        transcribe_backend=args.backend,
        whisper_model=args.whisper_model,
        segment_minutes=args.segment_minutes,
        transcript_path=args.transcript,
        generate_srt=not args.no_srt,
        srt_translate_lang=args.srt_lang
    )
