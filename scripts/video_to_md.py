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
import math
import datetime
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

def _load_env_files():
    """Load .env from skill directory, scripts dir, or legacy location"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    skill_dir = os.path.dirname(script_dir)
    for env_path in [
        os.path.join(skill_dir, ".env"),
        os.path.join(script_dir, ".env"),
        os.path.expanduser("~/lore-engine/.env"),
    ]:
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        key, value = key.strip(), value.strip()
                        if key not in os.environ:
                            os.environ[key] = value

_load_env_files()

# Fix: 确保 stdout/stderr 行缓冲，让 pipeline.log 实时输出
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(line_buffering=True)

from styles import STYLES, VIDEO_PRESETS, get_style_prompt, list_styles, list_presets
from add_video_timestamps import convert_timestamps, add_video_embed
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
import shutil


def preflight_check():
    """Validate prerequisites before running the pipeline. Fail fast with actionable errors."""
    errors = []
    warnings = []

    # Fatal: FFmpeg/ffprobe
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        errors.append("FFmpeg/ffprobe not found in PATH. Install: https://ffmpeg.org/download.html")

    # Fatal: GEMINI_API_KEY
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY_1") or os.getenv("GOOGLE_AI_API_KEY")
    if not api_key:
        errors.append("GEMINI_API_KEY not set. Add it to ~/.claude/skills/video-to-canvas/.env")

    # Warning: GPU
    try:
        import torch
        if not torch.cuda.is_available():
            warnings.append("CUDA not available — whisper will use CPU (slower)")
    except (ImportError, OSError, RuntimeError):
        warnings.append("torch not installed — whisper will use CPU (slower)")

    # Warning: faster-whisper
    try:
        import faster_whisper  # noqa: F401
    except (ImportError, OSError):
        warnings.append("faster-whisper not available — will use Gemini cloud transcription")

    # Warning: disk space
    try:
        usage = shutil.disk_usage(os.getcwd())
        free_gb = usage.free / (1024**3)
        if free_gb < 1.0:
            warnings.append(f"Low disk space: {free_gb:.1f} GB free (recommend > 1 GB)")
    except OSError:
        pass

    for w in warnings:
        print(f"  [WARNING] {w}", flush=True)

    if errors:
        print("\n[PREFLIGHT FAILED]", flush=True)
        for e in errors:
            print(f"  [FATAL] {e}", flush=True)
        sys.exit(1)

    if warnings:
        print()


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


def _split_video_segments(video_path: str, max_duration: int = 2100) -> list:
    """
    将视频分割为 ≤max_duration 秒的片段，用于分段上传 Gemini。
    使用 -c copy 快速分割（无需重编码），耗时 <10 秒。

    Args:
        video_path: 视频文件路径
        max_duration: 每段最大时长（秒），默认 2100 = 35 分钟

    Returns:
        [(segment_path, offset_seconds), ...]
        如果视频不需要分割，返回 [(video_path, 0)]
    """
    duration = get_video_duration(video_path)
    if duration <= 0:
        print(f"  [Split] Cannot determine duration, uploading whole video")
        return [(video_path, 0)]

    if duration <= max_duration:
        print(f"  [Split] Video {duration/60:.1f}min ≤ {max_duration/60:.0f}min limit, no split needed")
        return [(video_path, 0)]

    num_segments = math.ceil(duration / max_duration)
    segment_duration = duration / num_segments

    print(f"  [Split] Video {duration/60:.1f}min → {num_segments} segments "
          f"(~{segment_duration/60:.1f}min each)")

    segments = []
    for i in range(num_segments):
        offset = i * segment_duration
        seg_path = tempfile.mktemp(suffix=f'_seg{i}.mp4')

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(offset),
            "-i", video_path,
            "-t", str(segment_duration),
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            seg_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg segment split failed: {result.stderr[:300]}")

        seg_size_mb = os.path.getsize(seg_path) / (1024 * 1024)
        print(f"    Segment {i+1}/{num_segments}: offset={offset/60:.1f}min, "
              f"size={seg_size_mb:.1f}MB")
        segments.append((seg_path, offset))

    return segments


_progress_started_at = None
_progress_lock = threading.Lock()


def _update_progress(output_dir: str, stage: str, detail: str,
                     status: str = "running", error: str = None,
                     parallel_stages: list = None):
    """更新 progress.json，供 Claude 轮询监控进度（线程安全）"""
    global _progress_started_at
    with _progress_lock:
        if _progress_started_at is None:
            _progress_started_at = datetime.datetime.now().isoformat()

        progress_data = {
            "status": status,
            "stage": stage,
            "stage_detail": detail,
            "started_at": _progress_started_at,
            "updated_at": datetime.datetime.now().isoformat(),
            "error": error
        }
        if parallel_stages:
            progress_data["parallel_stages"] = parallel_stages

        progress_path = os.path.join(output_dir, "progress.json")
        with open(progress_path, "w", encoding="utf-8") as f:
            json.dump(progress_data, f, ensure_ascii=False, indent=2)


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


# ========== 后处理：表格修复 ==========

LATEX_TABLE_UNICODE = {
    r'\times': '×', r'\cdot': '·', r'\leq': '≤', r'\le': '≤',
    r'\geq': '≥', r'\ge': '≥', r'\neq': '≠', r'\ne': '≠',
    r'\approx': '≈', r'\infty': '∞', r'\pm': '±',
    r'\alpha': 'α', r'\beta': 'β', r'\gamma': 'γ', r'\delta': 'δ',
    r'\epsilon': 'ε', r'\theta': 'θ', r'\lambda': 'λ', r'\pi': 'π',
    r'\sigma': 'σ', r'\mu': 'μ', r'\omega': 'ω',
    r'\to': '→', r'\rightarrow': '→', r'\leftarrow': '←',
    r'\Rightarrow': '⇒', r'\Leftarrow': '⇐',
    r'\subset': '⊂', r'\subseteq': '⊆', r'\in': '∈',
}


def _sanitize_latex_in_cell(cell: str) -> str:
    """将表格单元格中简单 LaTeX 替换为 Unicode，保留复杂公式"""
    def replace_dollar(match):
        content = match.group(1)
        result = content
        for latex_cmd, unicode_char in LATEX_TABLE_UNICODE.items():
            result = result.replace(latex_cmd, unicode_char)
        # 如果替换后不再有 \ 命令，剥除 $ 符号
        if '\\' not in result:
            return result
        # 仍有复杂命令，保留 $...$
        return f'${result}$'
    return re.sub(r'\$([^$]+)\$', replace_dollar, cell)


def fix_tables(markdown_text: str) -> str:
    """修复 markdown 表格的渲染问题：LaTeX Unicode 化 + 结构验证"""
    lines = markdown_text.split('\n')
    result = []
    i = 0
    tables_fixed = 0

    while i < len(lines):
        # 检测表格起始（至少 2 行 | 开头，第 2 行是分隔符）
        if (i + 1 < len(lines)
                and lines[i].strip().startswith('|')
                and lines[i + 1].strip().startswith('|')
                and re.match(r'^\s*\|[\s:|-]+\|\s*$', lines[i + 1])):
            # 收集整个表格块
            table_start = i
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                table_lines.append(lines[i])
                i += 1

            # 确定头部列数
            header_cols = len(table_lines[0].split('|')) - 2  # 去掉首尾空
            if header_cols < 1:
                header_cols = len([c for c in table_lines[0].split('|') if c.strip()])

            fixed_lines = []
            for j, tl in enumerate(table_lines):
                # LaTeX Unicode 化
                cells = tl.split('|')
                sanitized = []
                for ci, cell in enumerate(cells):
                    if ci == 0 or ci == len(cells) - 1:
                        sanitized.append(cell)  # 保留首尾空
                    elif j == 1:
                        sanitized.append(cell)  # 分隔行不处理
                    else:
                        sanitized.append(_sanitize_latex_in_cell(cell))
                fixed_line = '|'.join(sanitized)

                # 列数修复：确保每行管道数一致
                actual_cols = len(sanitized) - 2
                if actual_cols > header_cols and j > 0:
                    # 多余列：截断
                    parts = fixed_line.split('|')
                    fixed_line = '|'.join(parts[:header_cols + 1]) + '|'
                elif actual_cols < header_cols and j > 0:
                    # 缺少列：补空
                    diff = header_cols - actual_cols
                    fixed_line = fixed_line.rstrip('|') + '| ' * diff + '|'

                fixed_lines.append(fixed_line)

            result.extend(fixed_lines)
            tables_fixed += 1
        else:
            result.append(lines[i])
            i += 1

    if tables_fixed > 0:
        print(f"[后处理 4.1] 修复了 {tables_fixed} 个表格（LaTeX→Unicode + 结构验证）")
    return '\n'.join(result)


# ========== 后处理：截图分布修复 ==========

def parse_timestamp_from_filename(filename: str) -> float:
    """从截图文件名（如 '03-04.jpg' 或 'screenshots/01-15-08.jpg'）提取秒数"""
    basename = os.path.basename(filename).replace('.jpg', '').replace('.png', '')
    parts = basename.split('-')
    try:
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    except (ValueError, IndexError):
        pass
    return 0


def fix_screenshot_distribution(
    markdown_text: str,
    screenshot_dir: str,
    screenshots: list,
    min_gap_seconds: float = 180.0
) -> str:
    """补充无截图的长 section + 检测截图时间戳错配"""
    lines = markdown_text.split('\n')

    # Step 1: 解析 sections
    ts_pattern = re.compile(r'\[(\d{1,2}:\d{2}(?::\d{2})?)\]')
    img_pattern = re.compile(r'!\[.*?\]\((screenshots/[^)]+)\)')
    sections = []
    current = None

    for li, line in enumerate(lines):
        heading_match = re.match(r'^(#{2,3})\s+', line)
        if heading_match:
            if current is not None:
                current['end_line'] = li
                sections.append(current)
            current = {
                'heading': line, 'level': len(heading_match.group(1)),
                'start_line': li, 'end_line': len(lines),
                'timestamps': [], 'screenshots': []
            }
        if current is not None:
            for ts_match in ts_pattern.finditer(line):
                secs = parse_timestamp_to_seconds(ts_match.group(1))
                if secs > 0:
                    current['timestamps'].append(secs)
            for img_match in img_pattern.finditer(line):
                current['screenshots'].append(img_match.group(1))

    if current is not None:
        current['end_line'] = len(lines)
        sections.append(current)

    # Step 2: 构建截图清单
    already_referenced = set()
    for sec in sections:
        already_referenced.update(sec['screenshots'])

    all_available = {}
    if os.path.isdir(screenshot_dir):
        for fn in os.listdir(screenshot_dir):
            if fn.endswith('.jpg') or fn.endswith('.png'):
                rel_path = f"screenshots/{fn}"
                ts_sec = parse_timestamp_from_filename(fn)
                all_available[rel_path] = ts_sec

    # 截图描述查找表
    desc_lookup = {}
    for ss in screenshots:
        safe_ts = ss['timestamp'].replace(':', '-')
        rel_path = f"screenshots/{safe_ts}.jpg"
        desc_lookup[rel_path] = ss.get('desc', '')

    unreferenced = {p: t for p, t in all_available.items() if p not in already_referenced}

    # Step 3+4: 识别空白段 + 自动插入
    inserted = 0
    mismatches = 0
    insert_ops = []  # (line_number, image_markdown)

    for sec in sections:
        if not sec['timestamps']:
            continue
        sec_start = min(sec['timestamps'])
        sec_end = max(sec['timestamps'])
        sec_span = sec_end - sec_start

        # 错配检测 (Issue 3)
        for ss_ref in sec['screenshots']:
            if ss_ref in all_available:
                ss_ts = all_available[ss_ref]
                tolerance = 120
                if ss_ts < sec_start - tolerance or ss_ts > sec_end + tolerance:
                    print(f"  [Warning] Misplaced screenshot: {ss_ref} (ts={ss_ts:.0f}s) "
                          f"in section [{sec_start:.0f}s-{sec_end:.0f}s]")
                    mismatches += 1

        # 空白段补图
        if len(sec['screenshots']) == 0 and sec_span >= min_gap_seconds:
            midpoint = (sec_start + sec_end) / 2
            best_path = None
            best_dist = float('inf')
            for path, ts in unreferenced.items():
                if sec_start - 30 <= ts <= sec_end + 30:
                    dist = abs(ts - midpoint)
                    if dist < best_dist:
                        best_dist = dist
                        best_path = path

            if best_path:
                desc = desc_lookup.get(best_path, '截图')
                img_md = f"\n![{desc}]({best_path})\n"
                # 在 heading 后的第一个非空内容行之后插入
                insert_line = sec['start_line'] + 1
                for k in range(sec['start_line'] + 1, min(sec['end_line'], sec['start_line'] + 5)):
                    if k < len(lines) and lines[k].strip():
                        insert_line = k + 1
                        break
                insert_ops.append((insert_line, img_md))
                del unreferenced[best_path]
                inserted += 1

    # 按倒序插入避免行号偏移
    insert_ops.sort(key=lambda x: x[0], reverse=True)
    for line_no, img_md in insert_ops:
        lines.insert(line_no, img_md)

    if inserted > 0 or mismatches > 0:
        print(f"[后处理 2.5] 截图分布: 插入 {inserted} 张, "
              f"错配警告 {mismatches} 个, 未引用 {len(unreferenced)} 张")
    return '\n'.join(lines)


def convert_timestamps_to_links(md_text: str) -> str:
    """将纯文本时间戳 [MM:SS] 转为 Media Extended 可点击格式 [MM:SS]()"""
    # 时间范围: [MM:SS-MM:SS] → [MM:SS]()-[MM:SS]()
    pattern_range = r'\[(\d{1,2}:\d{2}(?::\d{2})?)\s*[-–]\s*(\d{1,2}:\d{2}(?::\d{2})?)\](?!\()'
    md_text = re.sub(pattern_range, r'[\1]()-[\2]()', md_text)
    # 单时间戳: [MM:SS] 或 [HH:MM:SS]（排除已是链接的和图片引用的）
    pattern_single = r'\[(\d{1,2}:\d{2}(?::\d{2})?)\](?!\()'
    md_text = re.sub(pattern_single, r'[\1]()', md_text)
    return md_text


def parse_timestamp_to_seconds(timestamp: str) -> float:
    """将 MM:SS 或 HH:MM:SS 格式转换为秒数"""
    parts = timestamp.split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    return 0


def seconds_to_timestamp(seconds: float) -> str:
    """将秒数转换为 MM:SS 或 HH:MM:SS 格式"""
    total_secs = int(seconds)
    hrs = total_secs // 3600
    mins = (total_secs % 3600) // 60
    secs = total_secs % 60
    if hrs > 0:
        return f"{hrs:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


def extract_screenshot(video_path: str, timestamp: str, output_file: str) -> bool:
    """使用 ffmpeg 从视频中提取指定时间的截图"""
    cmd = [
        "ffmpeg", "-y",
        "-ss", timestamp,
        "-i", video_path,
        "-frames:v", "1",
        "-vf", "scale='min(1280,iw)':-2",
        "-q:v", "2",
        output_file
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def _extract_screenshots_parallel(video_path: str, change_points: list,
                                   screenshot_dir: str, max_workers: int = 8) -> list:
    """并行提取截图（I/O 密集型，ThreadPoolExecutor 加速）"""
    print(f"正在提取截图（并行, {max_workers} workers）...")

    def _extract_one(i, cp):
        ts = cp["timestamp"]
        safe_ts = ts.replace(":", "-")
        output_file = os.path.join(screenshot_dir, f"{safe_ts}.jpg")
        success = extract_screenshot(video_path, ts, output_file)
        return i, ts, cp.get("description", "")[:30], cp.get("change_type", ""), success, output_file

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_extract_one, i, cp)
                   for i, cp in enumerate(change_points)]
        results = [f.result() for f in as_completed(futures)]

    results.sort(key=lambda x: x[0])

    screenshots = []
    for i, ts, desc, change_type, success, output_file in results:
        status = "[OK]" if success else "[FAIL]"
        print(f"  [{i+1}/{len(change_points)}] {status} {ts} - {desc}")
        if success:
            screenshots.append({
                "timestamp": ts,
                "path": output_file,
                "desc": desc,
                "type": change_type
            })
    return screenshots


def wait_for_processing(client, video_file, max_wait: int = 300):
    """等待视频处理完成（自适应轮询间隔）"""
    start_time = time.time()
    poll_interval = 2.0
    max_interval = 10.0
    while video_file.state.name == "PROCESSING":
        elapsed = time.time() - start_time
        if elapsed > max_wait:
            raise TimeoutError(f"视频处理超时（>{max_wait}秒）")
        print(f"  视频处理中... ({int(elapsed)}s)")
        time.sleep(poll_interval)
        poll_interval = min(poll_interval * 2, max_interval)
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


def phase1_detect_changes(client, video_path: str, density: str = "normal") -> dict:
    """
    Stage 2 (Eyes): 分段上传视频到 Gemini 进行视觉变化检测。
    长视频（>35min）自动分割为多个片段分别上传，合并结果。

    Args:
        client: Gemini 客户端
        video_path: 视频文件路径（非 video_file 对象）
        density: 检测密度 (sparse/normal/dense)

    Returns:
        包含 change_points 和 video_summary 的字典
    """
    density_hints = {
        "sparse": "\n\n密度要求：每分钟 1-3 个变化点，只保留最重要的变化。",
        "normal": "\n\n密度要求：每分钟 3-6 个变化点，平衡覆盖度和精简度。",
        "dense": "\n\n密度要求：每分钟 6-12 个变化点，尽可能捕捉所有变化。"
    }
    prompt = CHANGE_DETECTION_PROMPT + density_hints.get(density, density_hints["normal"])

    # 分割视频（≤35min/段，≈540K tokens，远低于 1M 限制）
    segments = _split_video_segments(video_path, max_duration=2100)

    all_change_points = []
    video_summary = ""

    def _process_segment(i, seg_path, offset):
        """处理单个视频分段：上传 → 等待 → 检测 → 清理"""
        seg_count = len(segments)
        print(f"\n[Stage 2] Segment {i+1}/{seg_count}: uploading to Gemini...")

        vf = client.files.upload(file=seg_path)
        vf = wait_for_processing(client, vf)

        response_text = _gemini_generate_with_retry(
            client, "gemini-2.5-flash",
            [vf, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=CHANGE_DETECTION_SCHEMA
            )
        )

        try:
            result = json.loads(response_text)
        except json.JSONDecodeError:
            text = response_text.strip()
            text = re.sub(r'^```json\s*', '', text)
            text = re.sub(r'\s*```$', '', text)
            result = json.loads(text)

        seg_points = result.get("change_points", [])
        if offset > 0:
            for cp in seg_points:
                ts_seconds = parse_timestamp_to_seconds(cp["timestamp"])
                cp["timestamp"] = seconds_to_timestamp(ts_seconds + offset)

        print(f"  Segment {i+1}/{seg_count}: {len(seg_points)} change points detected")
        seg_summary = result.get("video_summary", "") if i == 0 else None

        try:
            client.files.delete(name=vf.name)
        except Exception:
            pass

        if seg_path != video_path:
            try:
                os.remove(seg_path)
            except OSError:
                pass

        return (i, seg_points, seg_summary)

    # 分段并行处理（每段独立上传+检测，无共享可变状态）
    with ThreadPoolExecutor(max_workers=min(3, len(segments))) as executor:
        futures = [executor.submit(_process_segment, i, seg_path, offset)
                   for i, (seg_path, offset) in enumerate(segments)]
        results = [f.result() for f in futures]
    results.sort(key=lambda x: x[0])
    for _, seg_points, seg_summary in results:
        all_change_points.extend(seg_points)
        if seg_summary is not None:
            video_summary = seg_summary

    print(f"\n[Stage 2] Total: {len(all_change_points)} change points across {len(segments)} segments")

    return {
        "video_summary": video_summary,
        "change_points": all_change_points
    }


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
    video_duration: float = 0.0,
    output_dir: str = None
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
            video_duration, output_dir
        )

    # 单段处理（视频较短或无转录）
    return _generate_notes_single(
        client, screenshots, style, video_summary,
        use_v2, depth, transcript_result
    )


class RateLimiter:
    """线程安全的 API 调用速率限制器（匹配 v2.1.0 的 10s 间隔）"""
    def __init__(self, min_interval: float = 10.0):
        self._lock = threading.Lock()
        self._last_call = 0.0
        self._min_interval = min_interval

    def wait(self):
        with self._lock:
            now = time.time()
            elapsed = now - self._last_call
            if elapsed < self._min_interval:
                sleep_time = self._min_interval - elapsed
                print(f"    [Rate Limit] Waiting {sleep_time:.1f}s...")
                time.sleep(sleep_time)
            self._last_call = time.time()


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


# 评估 JSON Schema（强制结构化输出）
QUALITY_EVAL_SCHEMA = types.Schema(
    type="OBJECT",
    properties={
        "coverage_score": types.Schema(type="INTEGER", description="转录内容覆盖率 1-10"),
        "structure_score": types.Schema(type="INTEGER", description="知识结构组织质量 1-10"),
        "depth_score": types.Schema(type="INTEGER", description="解释深度 1-10"),
        "accuracy_score": types.Schema(type="INTEGER", description="内容准确性 1-10"),
        "scaffolding_score": types.Schema(type="INTEGER", description="教学脚手架完整性 1-10 (blockquote/对比表/worked examples/总结)"),
        "overall_score": types.Schema(type="INTEGER", description="综合评分 1-10"),
        "missing_content": types.Schema(
            type="ARRAY",
            items=types.Schema(type="STRING"),
            description="笔记中遗漏的关键内容列表"
        ),
        "structure_issues": types.Schema(
            type="ARRAY",
            items=types.Schema(type="STRING"),
            description="结构组织问题列表"
        ),
        "scaffolding_issues": types.Schema(
            type="ARRAY",
            items=types.Schema(type="STRING"),
            description="教学脚手架缺失项（如缺少blockquote/worked example/对比表/总结）"
        ),
        "hallucinations": types.Schema(
            type="ARRAY",
            items=types.Schema(type="STRING"),
            description="可能的幻觉/不准确内容"
        ),
    },
    required=["coverage_score", "structure_score", "depth_score",
              "accuracy_score", "scaffolding_score", "overall_score",
              "missing_content", "structure_issues", "scaffolding_issues",
              "hallucinations"],
)


def _evaluate_notes_quality(client, notes: str, transcript_text: str) -> dict:
    """评估笔记质量，返回结构化评分和改进建议"""
    eval_prompt = f"""你是笔记质量评估专家。请对比原始转录文本和生成的笔记，评估笔记质量。

## 评估标准
- coverage_score: 转录中的知识点是否都出现在笔记中？(1=大量遗漏, 10=完全覆盖)
- structure_score: 是否按知识结构组织而非时间流水账？(1=纯流水账, 10=完美知识树)
- depth_score: 是否有足够的解释和推理？(1=只有标题, 10=深度解析)
- accuracy_score: 内容是否准确无幻觉？(1=大量错误, 10=完全准确)
- scaffolding_score: 教学脚手架是否完整？(1=纯文字无标注, 10=丰富的 blockquote/对比表/worked examples/总结)
  检查项：每个 ## 章节是否有 💡核心思想 blockquote？是否有 ⚠️易错点？核心算法是否有 worked example？多概念是否有对比表？章节间是否有过渡句？
- missing_content: 列出笔记中遗漏的关键知识点
- structure_issues: 列出结构组织问题
- scaffolding_issues: 列出教学脚手架缺失项（如"第3章缺少核心思想blockquote"、"A*算法缺少worked example"）
- hallucinations: 列出可能的幻觉内容

## 原始转录文本
{transcript_text}

## 生成的笔记
{notes}
"""

    response = _gemini_generate_with_retry(
        client, "gemini-2.5-flash", [eval_prompt],
        config=types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json",
            response_schema=QUALITY_EVAL_SCHEMA,
        )
    )
    return json.loads(response)


def _supplement_notes(client, notes: str, eval_result: dict,
                      transcript_text: str, screenshots: list) -> str:
    """根据评估结果定向补全笔记"""
    missing = eval_result.get("missing_content", [])
    issues = eval_result.get("structure_issues", [])
    scaffolding = eval_result.get("scaffolding_issues", [])
    hallucinations = eval_result.get("hallucinations", [])

    supplement_prompt = f"""你是笔记修订专家。请根据以下问题修订笔记。

## 需要修复的问题

### 遗漏的内容（必须补充）
{chr(10).join(f'- {m}' for m in missing) if missing else '无'}

### 结构问题（需要调整）
{chr(10).join(f'- {i}' for i in issues) if issues else '无'}

### 教学脚手架缺失（必须补充！）
{chr(10).join(f'- {s}' for s in scaffolding) if scaffolding else '无'}

### 幻觉内容（需要移除或修正）
{chr(10).join(f'- {h}' for h in hallucinations) if hallucinations else '无'}

## 修订规则
1. 保留原笔记中质量好的部分
2. 补充遗漏的知识点到合适的位置
3. 修正结构问题（按知识领域组织，不按时间）
4. **补充教学脚手架**：为每个 ## 章节添加缺失的 > 💡 核心思想、> ⚠️ 易错点 blockquote；为核心算法补充 worked examples；为多概念对比添加表格；为章节末尾添加 > 📝 小结
5. 移除或修正幻觉内容
6. 保留所有截图引用和时间戳
7. 输出完整的修订后笔记

## 原始转录文本（作为事实基准）
{transcript_text}

## 需要修订的笔记
{notes}
"""
    images = []
    for s in screenshots:
        if os.path.exists(s["path"]):
            with open(s["path"], "rb") as f:
                images.append(types.Part.from_bytes(data=f.read(), mime_type="image/jpeg"))

    return _gemini_generate_with_retry(
        client, "gemini-2.5-flash", [*images, supplement_prompt],
        config=types.GenerateContentConfig(temperature=0.2, top_p=0.9)
    )


def _generate_notes_single(
    client,
    screenshots: list,
    style: str,
    video_summary: str,
    use_v2: bool,
    depth: str,
    transcript_result: TranscriptResult = None,
    is_continuation: bool = False
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
                   .with_hierarchy())

        # 分段连贯性：非首段时禁止重复和过渡语
        if is_continuation:
            builder.with_continuity()

        # 音频主干优先（双通道融合: 音频 > 视觉）
        if transcript_text:
            builder.with_transcript(transcript_text)

        builder.with_inference()

        # 截图作为辅助，在推理之后
        builder.with_screenshots(screenshots)

        builder.with_summary()

        # lecture 模式：启用 LaTeX、表格、理解题、教学脚手架（借鉴 Lore Engine TOOLS）
        if mode == "lecture":
            builder.with_latex().with_tables().with_tricky_questions().with_teaching_scaffolding()

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
    video_duration: float = 0.0,
    output_dir: str = None
) -> str:
    """
    分段笔记生成（用于长视频，避免 Gemini 幻觉）

    将视频按 segment_minutes 切分，每段独立生成笔记，最后合并。
    每个 chunk 完成后立即保存到 chunks/ 目录，支持断点恢复。
    """
    chunks = transcript_result.get_chunks(max_duration=segment_minutes * 60)

    # 准备 chunk 缓存目录
    chunks_dir = None
    if output_dir:
        chunks_dir = os.path.join(output_dir, "chunks")
        os.makedirs(chunks_dir, exist_ok=True)

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

    all_notes = [None] * total_chunks
    rate_limiter = RateLimiter(min_interval=10.0)

    def _process_chunk(i, chunk):
        """处理单个 chunk：缓存检查 → 筛选数据 → 生成笔记 → 保存缓存"""
        chunk_start = chunk["start"]
        chunk_end = chunk["end"]
        chunk_duration = (chunk_end - chunk_start) / 60

        print(f"\n  分段 {i+1}/{total_chunks}: "
              f"[{seconds_to_timestamp(chunk_start)} - {seconds_to_timestamp(chunk_end)}] "
              f"({chunk_duration:.1f}min)")

        # 检查该 chunk 是否已有缓存（断点恢复）
        chunk_cache_path = os.path.join(chunks_dir, f"chunk_{i}.json") if chunks_dir else None
        if chunk_cache_path and os.path.exists(chunk_cache_path):
            with open(chunk_cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            print(f"    [Cache] Chunk {i+1} loaded from cache")
            return (i, cached["notes"])

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

        if output_dir:
            _update_progress(output_dir, "stage3",
                             f"Stage 3: Generating chunk {i+1}/{total_chunks}...")

        rate_limiter.wait()
        segment_notes = _generate_notes_single(
            client, chunk_screenshots, style,
            video_summary if i == 0 else "",
            use_v2, depth, chunk_transcript,
            is_continuation=(i > 0)
        )

        # 立即保存 chunk（防止中断丢失）
        if chunk_cache_path:
            with open(chunk_cache_path, "w", encoding="utf-8") as f:
                json.dump({"chunk_index": i, "notes": segment_notes}, f, ensure_ascii=False)

        print(f"    分段 {i+1}/{total_chunks} 完成")
        return (i, segment_notes)

    # Chunk 并行生成（2 workers + RateLimiter 保证 ≥10s 间隔）
    with ThreadPoolExecutor(max_workers=min(2, total_chunks)) as executor:
        futures = {executor.submit(_process_chunk, i, chunk): i
                   for i, chunk in enumerate(chunks)}
        for future in as_completed(futures):
            idx, notes = future.result()
            all_notes[idx] = notes

    # 合并所有分段笔记
    if total_chunks == 1:
        return all_notes[0]

    # 多段合并：检查缓存或请 LLM 整合
    merged_cache_path = os.path.join(chunks_dir, "merged.md") if chunks_dir else None
    if merged_cache_path and os.path.exists(merged_cache_path):
        cached_chunks = len([f for f in os.listdir(chunks_dir)
                             if f.startswith("chunk_") and f.endswith(".json")])
        if cached_chunks >= total_chunks:
            with open(merged_cache_path, "r", encoding="utf-8") as f:
                print(f"  [Cache] Merged notes loaded from cache")
                return f.read()

    print(f"\n[Stage 3: Brain] 合并 {total_chunks} 个分段笔记...")

    # 统计输入截图引用总数，用于合并 prompt 的零丢失约束
    total_ss_refs = sum(
        len(re.findall(r'!\[.*?\]\(screenshots/[^)]+\)', n))
        for n in all_notes if n
    )
    min_ss_refs = int(total_ss_refs * 0.9)

    # 统计输入中的 blockquote 和教学元素数量
    total_blockquotes = sum(
        len(re.findall(r'^>[ ]', n, re.MULTILINE))
        for n in all_notes if n
    )
    min_blockquotes = max(int(total_blockquotes * 0.9), 1)

    merge_prompt = f"""你是笔记合并专家。以下是同一个视频的 {total_chunks} 个分段笔记，
请将它们合并为一份结构完整、无重复的笔记。

## 合并规则
1. **知识结构优先**：按知识领域重新组织，不是简单拼接分段
2. **零遗漏**：保留所有知识点、截图引用、时间戳
3. **去重**：分段边界处的重复内容只保留一份
4. **统一结构**：使用 # → ## → ### → #### 层级，主题章节约 5-10 个
5. **截图零丢失（最重要！）**：输入中共有 {total_ss_refs} 个截图引用，输出中不得少于 {min_ss_refs} 个。每个 ![...](screenshots/...) 引用的路径必须原样保留，绝不能删除。
6. **保留时间戳**：所有 [MM:SS]() 格式的时间戳必须保留
7. **教学元素零丢失**：输入中共有约 {total_blockquotes} 个 blockquote（> 开头的行），输出中不得少于 {min_blockquotes} 个。
   所有 `> 💡`、`> ⚠️`、`> 📌`、`> 📝` 格式的教学标注必须保留。
   Worked examples（逐步求解过程）必须保留完整步骤，不得压缩为摘要。
8. **内容详细度保持**：合并时不要将段落压缩为简短列表。每个 ### 子章节应保持原始分段中的详细程度。
9. **输出使用中文**，技术术语保留英文

## 输出要求
- 直接输出合并后的 Markdown，不要包含元描述
- 不要写"以下是合并后的笔记"等前言

"""
    for i, notes in enumerate(all_notes):
        merge_prompt += f"\n{'='*40}\n## 分段 {i+1}\n{'='*40}\n{notes}\n"

    rate_limiter.wait()
    merged_result = _gemini_generate_with_retry(
        client, "gemini-2.5-flash", [merge_prompt],
        config=types.GenerateContentConfig(
            temperature=0.1,
            top_p=0.85,
        )
    )

    # 缓存 merge 结果（resume 时跳过重复 API 调用）
    if merged_cache_path:
        with open(merged_cache_path, "w", encoding="utf-8") as f:
            f.write(merged_result)

    return merged_result


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
    # 自动风格检测：文件名含 lecture/lec → 使用 lecture 模式
    video_basename = os.path.basename(video_path).lower()
    if style == "tutorial" and any(kw in video_basename for kw in ["lecture", "lec", "讲座", "讲课"]):
        style = "lecture"
        print(f"[Auto-detect] 检测到讲座视频，自动切换为 lecture 模式")

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
    # Preflight: validate prerequisites before anything else
    preflight_check()

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

    # 预计算视频名（用于恢复检查）
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # 初始化进度追踪
    _update_progress(output_dir, "init", "Pipeline starting...")

    # ========== 恢复检查 (Fix 6) ==========
    # 自动检测输出目录中已有的文件，跳过已完成的阶段
    transcript_json_path = os.path.join(output_dir, f"{video_name}_transcript.json")
    changes_json_path = os.path.join(output_dir, f"{video_name}_changes.json")

    if not transcript_path and os.path.exists(transcript_json_path):
        transcript_path = transcript_json_path
        print(f"[Resume] Found existing transcript: {transcript_json_path}")

    skip_stage2 = False
    saved_change_points = None
    if os.path.exists(changes_json_path) and os.path.isdir(screenshot_dir):
        existing_screenshots = [f for f in os.listdir(screenshot_dir) if f.endswith('.jpg')]
        if existing_screenshots:
            print(f"[Resume] Found existing changes.json + {len(existing_screenshots)} screenshots")
            try:
                with open(changes_json_path, "r", encoding="utf-8") as f:
                    saved_data = json.load(f)
                saved_change_points = saved_data.get("change_points", [])
                if saved_change_points:
                    skip_stage2 = True
                    print(f"[Resume] Will skip Stage 2 ({len(saved_change_points)} change points)")
            except (json.JSONDecodeError, KeyError):
                pass

    # ========== 预计算视频时长（ffprobe，快速） ==========
    video_duration = get_video_duration(video_path)

    # ========== 确定需要运行的阶段 ==========
    need_stage1 = transcribe_audio and not (transcript_path and os.path.exists(transcript_path))
    need_stage2 = not skip_stage2

    transcript_result = None
    change_points = None
    video_summary = ""
    screenshots = []

    # ----- Stage 1 线程函数（本地 faster-whisper，无 API 调用） -----
    def _do_stage1():
        _update_progress(output_dir, "stage1", "Stage 1: Audio transcription...",
                         parallel_stages=["stage1", "stage2"] if need_stage2 else None)
        result = transcribe(
            video_path=video_path,
            backend=transcribe_backend,
            model_size=whisper_model,
            gemini_client=client if transcribe_backend in ("auto", "gemini") else None
        )
        save_transcript(result, os.path.join(output_dir, f"{video_name}_transcript.json"))
        print("[Pipeline] Stage 1 complete")
        return result

    # ----- Stage 2 线程函数（Gemini API 视觉检测 + 截图） -----
    def _do_stage2():
        _update_progress(output_dir, "stage2", "Stage 2: Visual change detection...",
                         parallel_stages=["stage1", "stage2"] if need_stage1 else None)
        if fusion:
            print(f"\n[Stage 2] Uploading video for fusion mode: {video_path}")
            file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
            print(f"  File size: {file_size_mb:.1f} MB")
            video_file = client.files.upload(file=video_path)
            print("  Upload complete, waiting for processing...")
            video_file = wait_for_processing(client, video_file)
            from fusion_detector import FusionDetector, analyze_detection_quality
            print("\n[融合模式] 启用双通道交叉验证")
            detector = FusionDetector(client, video_file)
            detector.detect_and_fuse(
                visual_prompt=CHANGE_DETECTION_PROMPT,
                visual_schema=CHANGE_DETECTION_SCHEMA,
                min_confidence=min_confidence
            )
            cp = detector.to_legacy_format()
            vs = ""
            analysis = analyze_detection_quality(
                detector.visual_points, detector.semantic_points, detector.fused_points
            )
            print(f"\n[分析] 匹配率: {analysis['fused_ratio']:.1%}")
            for suggestion in analysis.get("suggestions", []):
                print(f"  建议: {suggestion}")
        else:
            result = phase1_detect_changes(client, video_path, density)
            cp = result.get("change_points", [])
            vs = result.get("video_summary", "")

        print(f"检测到 {len(cp)} 个原始变化点")
        cp = filter_change_points(cp, min_interval)
        print(f"过滤后保留 {len(cp)} 个变化点（最小间隔: {min_interval}s）")
        if not cp:
            print("警告：未检测到有效变化点，使用备用策略...")
            cp = [{"timestamp": "00:00", "change_type": "other", "description": "视频开始"}]

        # Stage 2.5: 覆盖率检查 + 自动补充
        vd = video_duration
        if vd > 0:
            cp = fill_coverage_gaps(cp, vd, gap_interval=30.0)

        # 并行截图提取
        ss = _extract_screenshots_parallel(video_path, cp, screenshot_dir)
        print(f"[Pipeline] Stage 2 complete ({len(ss)} screenshots)")
        return cp, vs, ss

    # ========== Stage 1 & Stage 2: 并行或顺序执行 ==========
    if need_stage1 and need_stage2:
        # ===== 并行模式: Stage 1 (LOCAL) + Stage 2 (API) 同时运行 =====
        print("\n[Pipeline] PARALLEL mode: Stage 1 + Stage 2 running simultaneously")
        print("  Thread A: Stage 1 (local faster-whisper)")
        print("  Thread B: Stage 2 (Gemini API visual detection)")
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_s1 = executor.submit(_do_stage1)
            future_s2 = executor.submit(_do_stage2)
            errors = {}
            for future in as_completed([future_s1, future_s2]):
                try:
                    if future is future_s1:
                        transcript_result = future.result()
                    else:
                        change_points, video_summary, screenshots = future.result()
                except Exception as e:
                    name = "Stage 1" if future is future_s1 else "Stage 2"
                    errors[name] = e
                    print(f"[Pipeline] {name} FAILED: {e}")
            if errors:
                msg = "; ".join(f"{k}: {v}" for k, v in errors.items())
                raise RuntimeError(f"Parallel pipeline failed: {msg}")
    else:
        # ===== 顺序模式: 恢复路径 =====
        if need_stage1:
            transcript_result = _do_stage1()
        elif transcript_path and os.path.exists(transcript_path):
            print(f"\n[Stage 1] Loading existing transcript: {transcript_path}")
            transcript_result = load_transcript(transcript_path)
            print(f"  Segments: {len(transcript_result.segments)}, Backend: {transcript_result.backend}")
        elif not transcribe_audio:
            print("\n[Stage 1: Ears] Skipped (--no-transcribe)")
        _update_progress(output_dir, "stage1", "Stage 1 complete")

        if need_stage2:
            change_points, video_summary, screenshots = _do_stage2()
        else:
            change_points = saved_change_points
            video_summary = ""
            print(f"\n[Stage 2] Resumed: {len(change_points)} change points from cache")
            screenshots = []
            for cp_item in change_points:
                ts = cp_item["timestamp"]
                safe_ts = ts.replace(":", "-")
                ss_file = os.path.join(screenshot_dir, f"{safe_ts}.jpg")
                if os.path.exists(ss_file):
                    screenshots.append({
                        "timestamp": ts, "path": ss_file,
                        "desc": cp_item.get("description", ""),
                        "type": cp_item.get("change_type", "")
                    })
            print(f"  Loaded {len(screenshots)} existing screenshots")

    # 确保 video_duration 有 transcript 补充
    if video_duration <= 0 and transcript_result and transcript_result.duration > 0:
        video_duration = transcript_result.duration

    # ========== SRT 生成 + 后台翻译（与 Stage 3 并行） ==========
    srt_paths = {}
    _srt_translate_future = None
    _srt_translate_executor = None
    if transcript_result and generate_srt:
        srt_path = os.path.join(output_dir, f"{video_name}.srt")
        if os.path.exists(srt_path):
            print(f"\n[SRT] Already exists, skipping: {srt_path}")
            srt_paths["en"] = srt_path
        else:
            print(f"\n[SRT] Generating subtitles...")
            generate_srt_from_transcript(transcript_result, srt_path)
            srt_paths["en"] = srt_path
            print(f"  SRT: {srt_path} ({len(transcript_result.segments)} segments)")

        if srt_translate_lang:
            translated_srt_path = os.path.join(
                output_dir, f"{video_name}.{srt_translate_lang}.srt"
            )
            if os.path.exists(translated_srt_path):
                print(f"  [Cache] Translated SRT exists: {translated_srt_path}")
                srt_paths[srt_translate_lang] = translated_srt_path
            else:
                print(f"  [Background] SRT translation to {srt_translate_lang} started...")
                _srt_translate_executor = ThreadPoolExecutor(max_workers=1)
                _srt_translate_future = _srt_translate_executor.submit(
                    translate_srt_file, srt_path, translated_srt_path,
                    srt_translate_lang, gemini_client=client
                )

    # ========== Stage 3 (Brain): 笔记生成 ==========
    _update_progress(output_dir, "stage3", "Stage 3: Generating notes...")
    markdown_text = phase2_generate_notes(
        client, screenshots, style, video_summary,
        use_v2=use_v2, depth=depth,
        transcript_result=transcript_result,
        segment_minutes=segment_minutes,
        video_duration=video_duration,
        output_dir=output_dir
    )

    # ========== 质量评估循环 ==========
    QUALITY_THRESHOLD = 7
    if transcript_result:
        transcript_for_eval = format_transcript_for_prompt(transcript_result)
        print(f"\n[Quality] 评估笔记质量...")
        eval_result = _evaluate_notes_quality(client, markdown_text, transcript_for_eval)

        overall = eval_result.get("overall_score", 10)
        scaffolding = eval_result.get("scaffolding_score", 10)
        print(f"[Quality] 评分: 覆盖={eval_result.get('coverage_score')}, "
              f"结构={eval_result.get('structure_score')}, "
              f"深度={eval_result.get('depth_score')}, "
              f"准确={eval_result.get('accuracy_score')}, "
              f"脚手架={scaffolding}, "
              f"综合={overall}")

        # 当综合评分低 OR 教学脚手架评分低时触发补全
        needs_supplement = overall < QUALITY_THRESHOLD or scaffolding < QUALITY_THRESHOLD
        if needs_supplement:
            missing_count = len(eval_result.get("missing_content", []))
            issue_count = len(eval_result.get("structure_issues", []))
            scaffolding_count = len(eval_result.get("scaffolding_issues", []))
            trigger_reason = []
            if overall < QUALITY_THRESHOLD:
                trigger_reason.append(f"综合={overall}")
            if scaffolding < QUALITY_THRESHOLD:
                trigger_reason.append(f"脚手架={scaffolding}")
            print(f"[Quality] {' & '.join(trigger_reason)} < {QUALITY_THRESHOLD}，启动定向补全... "
                  f"(缺失: {missing_count}, 结构问题: {issue_count}, 脚手架缺失: {scaffolding_count})")
            markdown_text = _supplement_notes(
                client, markdown_text, eval_result, transcript_for_eval, screenshots
            )
            print(f"[Quality] 补全完成")
        else:
            print(f"[Quality] 质量达标，跳过补全")

        # 保存评估结果
        eval_path = os.path.join(output_dir, f"{video_name}_quality.json")
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(eval_result, f, ensure_ascii=False, indent=2)

    # ========== 保存结果 ==========
    _update_progress(output_dir, "saving", "Saving results...")
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

    # 后处理 2.5：截图分布修复（补充缺失截图 + 错配检测）
    markdown_text = fix_screenshot_distribution(markdown_text, screenshot_dir, screenshots)

    # 后处理 3：修复未关闭的代码块（Lore Engine markdown_utils 移植）
    lines = markdown_text.split('\n')
    in_code = False
    for line in lines:
        if line.strip().startswith('```'):
            in_code = not in_code
    if in_code:
        markdown_text += '\n```\n'
        print("[后处理 3] 修复了未关闭的代码块")

    # 后处理 4：压缩连续 3+ 空行为 2 空行
    markdown_text = re.sub(r'\n{4,}', '\n\n\n', markdown_text)

    # 后处理 4.1：修复表格渲染（LaTeX→Unicode + 结构验证）
    markdown_text = fix_tables(markdown_text)

    # 后处理 4.5：修复伪代码语言标注（Gemini 常将伪代码标为 ```python）
    pseudocode_markers = [
        r'function ', r'procedure ', r'for each', r'if .* then',
        r'←', r':=', r'while .* do', r'end for', r'end if',
        r'end while', r'end function', r'end procedure',
    ]
    pp_lines = markdown_text.split('\n')
    pp_result = []
    pp_i = 0
    while pp_i < len(pp_lines):
        line = pp_lines[pp_i]
        m = re.match(r'^(\s*)```(\w+)\s*$', line)
        if m:
            indent, lang = m.group(1), m.group(2)
            # 收集 block 内容直到关闭
            block_lines = []
            j = pp_i + 1
            while j < len(pp_lines) and not pp_lines[j].strip().startswith('```'):
                block_lines.append(pp_lines[j])
                j += 1
            block_text = '\n'.join(block_lines)
            # 判断是否是伪代码（>=2 个伪代码特征匹配）
            hits = sum(1 for p in pseudocode_markers
                       if re.search(p, block_text, re.IGNORECASE))
            if hits >= 2:
                pp_result.append(f'{indent}```')
                print(f"[后处理 4.5] 修复伪代码语言标注: ```{lang} → ```")
            else:
                pp_result.append(line)
            pp_i += 1
        else:
            pp_result.append(line)
            pp_i += 1
    markdown_text = '\n'.join(pp_result)

    # 后处理 5：转换时间戳为 Media Extended 可点击格式
    video_filename = os.path.basename(video_path)
    print(f"\n[后处理 5] 转换时间戳为 Media Extended 格式...")
    markdown_text = convert_timestamps(markdown_text, video_filename)

    # 后处理 6：在笔记顶部添加视频嵌入
    header = add_video_embed(header, video_filename)

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

    # 等待后台 SRT 翻译完成（如果有）
    if _srt_translate_future:
        try:
            print("\n[SRT] Waiting for background translation to finish...")
            result = _srt_translate_future.result(timeout=600)
            if result:
                srt_paths[srt_translate_lang] = result
                print(f"  [SRT] Translation complete: {result}")
        except Exception as e:
            print(f"  [Warning] SRT translation failed (non-fatal): {e}")
        finally:
            _srt_translate_executor.shutdown(wait=False)

    _update_progress(output_dir, "done", "Pipeline complete!", status="completed")

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

    parser.add_argument("--daemon", action="store_true",
                        help="后台运行模式：启动管道后立即退出，进度通过 progress.json 查询")
    parser.add_argument("--list-styles", action="store_true",
                        help="列出所有可用风格")
    parser.add_argument("--list-presets", action="store_true",
                        help="列出所有可用预设")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # --daemon: 后台运行模式，脚本自己 fork 子进程
    if args.daemon:
        import subprocess as _sp
        # 构建子进程命令（移除 --daemon 防止递归）
        child_args = [a for a in sys.argv if a != '--daemon']
        child_cmd = [sys.executable] + child_args

        # 日志路径与输出目录一致
        _daemon_output_dir = args.output
        os.makedirs(_daemon_output_dir, exist_ok=True)
        log_path = os.path.join(_daemon_output_dir, "pipeline.log")

        # Windows: CREATE_NO_WINDOW + DETACHED_PROCESS; Unix: start_new_session
        kwargs = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = _sp.CREATE_NO_WINDOW | _sp.DETACHED_PROCESS
        else:
            kwargs["start_new_session"] = True

        with open(log_path, "w", encoding="utf-8") as log_file:
            proc = _sp.Popen(child_cmd, stdout=log_file, stderr=_sp.STDOUT,
                             close_fds=(sys.platform != "win32"), **kwargs)

        print(f"Pipeline started in background (PID: {proc.pid})")
        print(f"  Log: {log_path}")
        print(f"  Progress: {os.path.join(_daemon_output_dir, 'progress.json')}")
        sys.exit(0)

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

    # 运行主函数（错误时写入 progress.json）
    try:
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
    except Exception as e:
        import traceback
        traceback.print_exc()
        _update_progress(args.output, "error", str(e)[:500], status="failed", error=str(e))
        sys.exit(1)
