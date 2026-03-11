"""
本地变化检测器 — Stage 2a 升级核心

三层检测 + 三层去重架构：
┌────────────────────────────────────────────────┐
│ 检测层                                          │
│  Layer 1: SSIM 视觉粗筛（大幅画面切换）          │
│  Layer 2: OCR-diff 文本精筛（同模板PPT/代码变化） │
│  Layer 3: 定期采样兜底（图表/动画变化）           │
└────────────────────────────────────────────────┘
                      ↓
┌────────────────────────────────────────────────┐
│ 去重层                                          │
│  Layer 1: Debounce（5秒时间窗口合并）            │
│  Layer 2: OCR 文本去重（>95% 相似去除）           │
│  Layer 3: 上限控制（保留 top-K 变化帧）           │
└────────────────────────────────────────────────┘

依赖：
  pip install opencv-python-headless rapidocr-onnxruntime
  或 pip install opencv-python-headless （仅 SSIM，无 OCR）

设计原则：
  - OCR 引擎可选：有 RapidOCR 就用，没有则降级为纯视觉模式
  - 输出格式兼容 video_to_md.py 的 change_points 格式
  - 全部本地运行，$0 成本
"""

import os
import re
import subprocess
import json
import difflib
import tempfile
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor


# ========== 数据结构 ==========

@dataclass
class DetectedFrame:
    """检测到的变化帧"""
    timestamp: str          # MM:SS
    seconds: float          # 秒数
    frame_path: str         # 帧图片路径
    change_type: str        # visual / text / periodic
    description: str        # 描述
    ocr_text: str = ""      # OCR 提取的文本
    ssim_score: float = 1.0 # 与前帧的 SSIM 相似度
    text_similarity: float = 1.0  # 与前帧的文本相似度
    confidence: float = 1.0


# ========== 工具函数 ==========

def _seconds_to_timestamp(seconds: float) -> str:
    """秒数转 MM:SS"""
    total_secs = int(seconds)
    hrs = total_secs // 3600
    mins = (total_secs % 3600) // 60
    secs = total_secs % 60
    if hrs > 0:
        return f"{hrs:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


def _get_video_duration(video_path: str) -> float:
    """用 ffprobe 获取视频时长"""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        video_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return float(result.stdout.strip())
    except Exception:
        return 0.0


# ========== 帧提取 ==========

def extract_frames(video_path: str, fps: float = 1.0, output_dir: str = None,
                   scale: int = 640) -> List[Tuple[float, str]]:
    """
    用 FFmpeg 按固定帧率提取帧

    Args:
        video_path: 视频路径
        fps: 每秒提取帧数（默认1FPS）
        output_dir: 输出目录（None则用临时目录）
        scale: 缩放宽度（降低OCR/SSIM计算量）

    Returns:
        [(seconds, frame_path), ...] 按时间排序
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="vtc_frames_")
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"fps={fps},scale={scale}:-2",
        "-q:v", "3",
        "-vsync", "vfr",
        os.path.join(output_dir, "frame_%06d.jpg")
    ]

    print(f"  [帧提取] FFmpeg {fps} FPS, scale={scale}px...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg 帧提取失败: {result.stderr[:200]}")

    # 收集帧文件，计算时间戳
    frames = []
    frame_files = sorted(f for f in os.listdir(output_dir) if f.startswith("frame_"))
    for i, fname in enumerate(frame_files):
        seconds = i / fps
        frames.append((seconds, os.path.join(output_dir, fname)))

    print(f"  [帧提取] 共提取 {len(frames)} 帧")
    return frames


# ========== SSIM 计算 ==========

def _compute_ssim(img1_path: str, img2_path: str) -> float:
    """计算两张图片的 SSIM 相似度"""
    try:
        import cv2
        import numpy as np

        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
        if img1 is None or img2 is None:
            return 1.0

        # 确保尺寸一致
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        # 计算 SSIM（简化版，不依赖 skimage）
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)

        mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return float(ssim_map.mean())
    except ImportError:
        # 无 OpenCV，用像素差异近似
        return _compute_pixel_diff(img1_path, img2_path)


def _compute_pixel_diff(img1_path: str, img2_path: str) -> float:
    """无 OpenCV 时的降级方案：PIL 像素差异"""
    try:
        from PIL import Image
        import struct

        img1 = Image.open(img1_path).convert("L").resize((160, 120))
        img2 = Image.open(img2_path).convert("L").resize((160, 120))

        pixels1 = list(img1.getdata())
        pixels2 = list(img2.getdata())

        diff = sum(abs(a - b) for a, b in zip(pixels1, pixels2))
        max_diff = 255 * len(pixels1)
        similarity = 1.0 - (diff / max_diff)
        return similarity
    except Exception:
        return 1.0


# ========== OCR ==========

_ocr_engine = None
_ocr_available = None


def _init_ocr():
    """懒加载 OCR 引擎"""
    global _ocr_engine, _ocr_available
    if _ocr_available is not None:
        return _ocr_available

    # 尝试 RapidOCR
    try:
        from rapidocr_onnxruntime import RapidOCR
        _ocr_engine = RapidOCR()
        _ocr_available = True
        print("  [OCR] RapidOCR engine loaded")
        return True
    except ImportError:
        pass

    # 尝试 PaddleOCR
    try:
        from paddleocr import PaddleOCR
        _ocr_engine = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
        _ocr_available = True
        print("  [OCR] PaddleOCR engine loaded")
        return True
    except ImportError:
        pass

    _ocr_available = False
    print("  [OCR] 无 OCR 引擎可用（pip install rapidocr-onnxruntime）")
    print("  [OCR] 降级为纯视觉检测模式")
    return False


def _ocr_extract(image_path: str) -> str:
    """提取图片中的文本"""
    global _ocr_engine
    if not _init_ocr():
        return ""

    try:
        # RapidOCR 接口
        if hasattr(_ocr_engine, '__call__'):
            result, _ = _ocr_engine(image_path)
            if result:
                return "\n".join(line[1] for line in result)
            return ""

        # PaddleOCR 接口
        if hasattr(_ocr_engine, 'ocr'):
            result = _ocr_engine.ocr(image_path, cls=True)
            if result and result[0]:
                return "\n".join(line[1][0] for line in result[0])
            return ""
    except Exception as e:
        return ""

    return ""


def _text_similarity(text1: str, text2: str) -> float:
    """计算两段文本的相似度"""
    if not text1 and not text2:
        return 1.0
    if not text1 or not text2:
        return 0.0
    return difflib.SequenceMatcher(None, text1, text2).ratio()


# ========== 三层检测 ==========

def detect_changes(
    frames: List[Tuple[float, str]],
    ssim_threshold: float = 0.92,
    text_threshold: float = 0.90,
    periodic_interval: float = 30.0,
    use_ocr: bool = True
) -> List[DetectedFrame]:
    """
    三层变化检测

    Layer 1: SSIM 视觉粗筛 — 检测大幅画面切换
    Layer 2: OCR-diff 文本精筛 — 检测同模板PPT/代码变化
    Layer 3: 定期采样兜底 — 不漏图表/动画变化

    Args:
        frames: [(seconds, frame_path), ...]
        ssim_threshold: SSIM < 此值标记为视觉变化
        text_threshold: 文本相似度 < 此值标记为文本变化
        periodic_interval: 定期采样间隔（秒）
        use_ocr: 是否启用 OCR（需要 RapidOCR/PaddleOCR）

    Returns:
        检测到的变化帧列表
    """
    if not frames:
        return []

    print(f"\n[变化检测] 开始三层检测（{len(frames)} 帧）")
    print(f"  SSIM 阈值: {ssim_threshold}, 文本阈值: {text_threshold}, 定期采样: {periodic_interval}s")

    detected: List[DetectedFrame] = []
    prev_ocr_text = ""
    last_periodic_time = -periodic_interval  # 确保第一帧被采样

    # 始终保留第一帧
    first_seconds, first_path = frames[0]
    first_ocr = _ocr_extract(first_path) if use_ocr and _init_ocr() else ""
    detected.append(DetectedFrame(
        timestamp=_seconds_to_timestamp(first_seconds),
        seconds=first_seconds,
        frame_path=first_path,
        change_type="visual",
        description="视频开始",
        ocr_text=first_ocr,
        ssim_score=0.0,
        text_similarity=0.0,
        confidence=1.0
    ))
    prev_ocr_text = first_ocr
    last_periodic_time = first_seconds
    prev_path = first_path

    total = len(frames)
    visual_count = 0
    text_count = 0
    periodic_count = 0

    # OCR 并行提取（如果可用）
    ocr_texts = {}
    if use_ocr and _init_ocr():
        print("  [OCR] 批量提取文本...")
        # 不对所有帧都 OCR — 先用 SSIM 筛一遍，只对"可能变化"的帧做 OCR
        # 这里做简单并行 OCR
        def _ocr_frame(idx_path):
            idx, path = idx_path
            return idx, _ocr_extract(path)

        # 每5帧抽样 OCR，或者帧数 < 200 时全量 OCR
        if total <= 200:
            ocr_indices = list(range(total))
        else:
            ocr_indices = list(range(0, total, 3))  # 每3帧 OCR

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = executor.map(_ocr_frame, [(i, frames[i][1]) for i in ocr_indices])
            for idx, text in results:
                ocr_texts[idx] = text
        print(f"  [OCR] 完成 {len(ocr_texts)}/{total} 帧的文本提取")

    for i in range(1, total):
        seconds, frame_path = frames[i]
        is_change = False
        change_type = ""
        change_desc = ""

        # Layer 1: SSIM 视觉粗筛
        ssim = _compute_ssim(prev_path, frame_path)

        if ssim < ssim_threshold:
            is_change = True
            change_type = "visual"
            change_desc = f"画面变化(SSIM={ssim:.3f})"
            visual_count += 1

        # Layer 2: OCR-diff 文本精筛
        ocr_text = ocr_texts.get(i, "")
        text_sim = 1.0
        if ocr_text and prev_ocr_text:
            text_sim = _text_similarity(prev_ocr_text, ocr_text)
            if text_sim < text_threshold and not is_change:
                is_change = True
                change_type = "text_change"
                change_desc = f"文本变化(相似度={text_sim:.3f})"
                text_count += 1
            elif text_sim < text_threshold and is_change:
                change_type = "visual+text"
                change_desc = f"画面+文本变化(SSIM={ssim:.3f}, 文本={text_sim:.3f})"

        # Layer 3: 定期采样兜底
        if not is_change and (seconds - last_periodic_time) >= periodic_interval:
            is_change = True
            change_type = "periodic"
            change_desc = f"定期采样({seconds - last_periodic_time:.0f}s间隔)"
            periodic_count += 1

        if is_change:
            detected.append(DetectedFrame(
                timestamp=_seconds_to_timestamp(seconds),
                seconds=seconds,
                frame_path=frame_path,
                change_type=change_type,
                description=change_desc,
                ocr_text=ocr_text,
                ssim_score=ssim,
                text_similarity=text_sim,
                confidence=1.0 if change_type in ("visual+text", "visual") else 0.8
            ))
            last_periodic_time = seconds

        # 更新前帧状态
        prev_path = frame_path
        if ocr_text:
            prev_ocr_text = ocr_text

        # 进度报告
        if i % 100 == 0:
            print(f"  [{i}/{total}] 已检测 {len(detected)} 个变化点")

    print(f"\n[变化检测] 完成:")
    print(f"  视觉变化: {visual_count}")
    print(f"  文本变化: {text_count}")
    print(f"  定期采样: {periodic_count}")
    print(f"  合计: {len(detected)} 个候选帧")

    return detected


# ========== 三层去重 ==========

def deduplicate(
    detected: List[DetectedFrame],
    debounce_window: float = 5.0,
    text_dedup_threshold: float = 0.95,
    max_frames: int = 40
) -> List[DetectedFrame]:
    """
    三层去重

    Layer 1: Debounce — 时间窗口内连续变化只保留最后一帧
    Layer 2: OCR 文本去重 — 相邻帧文字 >95% 相似则去除前帧
    Layer 3: 上限控制 — 保留 top-K 帧（按置信度+变化幅度排序）

    Args:
        detected: 检测到的变化帧列表
        debounce_window: debounce 时间窗口（秒）
        text_dedup_threshold: 文本去重阈值
        max_frames: 最大保留帧数
    """
    if len(detected) <= 1:
        return detected

    print(f"\n[去重] 输入 {len(detected)} 个候选帧")

    # Layer 1: Debounce（时间窗口合并）
    debounced = []
    group = [detected[0]]

    for frame in detected[1:]:
        if frame.seconds - group[-1].seconds <= debounce_window:
            group.append(frame)
        else:
            # 保留组内最后一帧（内容最完整）
            best = group[-1]
            # 如果组内有 visual+text 类型，优先保留
            for f in group:
                if "+" in f.change_type:
                    best = f
                    break
            debounced.append(best)
            group = [frame]

    # 处理最后一组
    if group:
        best = group[-1]
        for f in group:
            if "+" in f.change_type:
                best = f
                break
        debounced.append(best)

    print(f"  Layer 1 Debounce ({debounce_window}s): {len(detected)} → {len(debounced)}")

    # Layer 2: OCR 文本去重
    if any(f.ocr_text for f in debounced):
        text_deduped = [debounced[0]]
        for frame in debounced[1:]:
            prev_text = text_deduped[-1].ocr_text
            curr_text = frame.ocr_text

            if prev_text and curr_text:
                sim = _text_similarity(prev_text, curr_text)
                if sim >= text_dedup_threshold:
                    # 文本几乎相同，跳过（保留后者，替换前者）
                    text_deduped[-1] = frame
                    continue

            text_deduped.append(frame)
        print(f"  Layer 2 OCR去重 (>{text_dedup_threshold:.0%}): {len(debounced)} → {len(text_deduped)}")
        debounced = text_deduped

    # Layer 3: 上限控制
    if len(debounced) > max_frames:
        # 按 confidence + 变化幅度排序，保留 top-K
        # 始终保留第一帧和最后一帧
        first = debounced[0]
        last = debounced[-1]
        middle = debounced[1:-1]

        # 计算每帧的得分
        for f in middle:
            score = f.confidence
            if f.change_type == "visual+text":
                score += 0.3
            elif f.change_type == "visual":
                score += 0.2
            elif f.change_type == "text_change":
                score += 0.1
            # SSIM 越低（变化越大）分数越高
            score += (1.0 - f.ssim_score) * 0.5
            f.confidence = score

        middle.sort(key=lambda f: f.confidence, reverse=True)
        kept = middle[:max_frames - 2]
        kept.sort(key=lambda f: f.seconds)  # 恢复时间顺序
        debounced = [first] + kept + [last]
        print(f"  Layer 3 上限控制 (max={max_frames}): → {len(debounced)}")

    print(f"[去重] 最终保留 {len(debounced)} 帧")
    return debounced


# ========== 输出转换 ==========

def to_change_points(detected: List[DetectedFrame]) -> List[dict]:
    """
    转换为 video_to_md.py 兼容的 change_points 格式

    输出格式：
    {
        "timestamp": "01:30",
        "change_type": "visual",
        "description": "画面变化描述",
        "ocr_text": "OCR 提取的原文文字",
        "confidence": 0.9,
        "source": "local_detector"
    }
    """
    return [
        {
            "timestamp": f.timestamp,
            "change_type": _normalize_change_type(f.change_type),
            "description": _build_description(f),
            "ocr_text": f.ocr_text,
            "confidence": f.confidence,
            "source": "local_detector"
        }
        for f in detected
    ]


def _normalize_change_type(change_type: str) -> str:
    """将本地检测类型映射到原有 schema 的 change_type"""
    mapping = {
        "visual": "slide_change",
        "text_change": "slide_change",
        "visual+text": "slide_change",
        "periodic": "other",
    }
    return mapping.get(change_type, "other")


def _build_description(frame: DetectedFrame) -> str:
    """构建描述文本（优先使用 OCR 文本）"""
    if frame.ocr_text:
        # 用 OCR 文本前100字作为描述
        text = frame.ocr_text.replace("\n", " ").strip()
        if len(text) > 100:
            text = text[:100] + "..."
        return text
    return frame.description


# ========== 主入口 ==========

def detect_local(
    video_path: str,
    fps: float = 1.0,
    ssim_threshold: float = 0.92,
    text_threshold: float = 0.90,
    periodic_interval: float = 30.0,
    debounce_window: float = 5.0,
    text_dedup_threshold: float = 0.95,
    max_frames: int = 40,
    scale: int = 640,
    temp_dir: str = None
) -> dict:
    """
    完整的本地变化检测管线

    Args:
        video_path: 视频文件路径
        fps: 帧提取频率
        ssim_threshold: SSIM 变化阈值
        text_threshold: 文本变化阈值
        periodic_interval: 定期采样间隔
        debounce_window: debounce 窗口
        text_dedup_threshold: 文本去重阈值
        max_frames: 最大保留帧数
        scale: 帧缩放宽度
        temp_dir: 临时帧目录（None 自动创建）

    Returns:
        兼容 phase1_detect_changes 的返回格式：
        {
            "video_summary": "",
            "change_points": [...]
        }
    """
    print("\n" + "=" * 60)
    print("[Stage 2a] 本地变化检测 v3.0")
    print("=" * 60)

    duration = _get_video_duration(video_path)
    print(f"  视频时长: {_seconds_to_timestamp(duration)} ({duration:.1f}s)")

    # 自动调整参数
    if duration > 3600:  # > 1小时
        fps = 0.5
        periodic_interval = 45.0
        max_frames = 60
        print(f"  [自动调整] 长视频模式: fps={fps}, periodic={periodic_interval}s, max={max_frames}")
    elif duration > 1800:  # > 30分钟
        max_frames = 50
        print(f"  [自动调整] 中等视频: max_frames={max_frames}")

    # Step 1: 帧提取
    frames = extract_frames(video_path, fps=fps, output_dir=temp_dir, scale=scale)

    # Step 2: 三层检测
    use_ocr = _init_ocr()
    detected = detect_changes(
        frames,
        ssim_threshold=ssim_threshold,
        text_threshold=text_threshold,
        periodic_interval=periodic_interval,
        use_ocr=use_ocr
    )

    # Step 3: 三层去重
    deduped = deduplicate(
        detected,
        debounce_window=debounce_window,
        text_dedup_threshold=text_dedup_threshold,
        max_frames=max_frames
    )

    # Step 4: 转换为兼容格式
    change_points = to_change_points(deduped)

    print(f"\n[Stage 2a] 完成: {len(change_points)} 个变化点")
    print("=" * 60)

    return {
        "video_summary": "",
        "change_points": change_points
    }


# ========== 清理临时帧 ==========

def cleanup_frames(temp_dir: str):
    """清理临时帧目录"""
    import shutil
    if temp_dir and os.path.exists(temp_dir) and "vtc_frames_" in temp_dir:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ========== CLI 测试 ==========

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法: python local_detector.py <视频路径> [--fps 1.0] [--max-frames 40]")
        sys.exit(1)

    video = sys.argv[1]
    fps_arg = 1.0
    max_arg = 40

    for i, arg in enumerate(sys.argv):
        if arg == "--fps" and i + 1 < len(sys.argv):
            fps_arg = float(sys.argv[i + 1])
        elif arg == "--max-frames" and i + 1 < len(sys.argv):
            max_arg = int(sys.argv[i + 1])

    result = detect_local(video, fps=fps_arg, max_frames=max_arg)

    print(f"\n检测结果:")
    for cp in result["change_points"]:
        src = cp.get("source", "")
        desc = cp["description"][:50]
        print(f"  [{cp['timestamp']}] {cp['change_type']}: {desc}")

    # 保存结果
    out_path = os.path.splitext(video)[0] + "_changes.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out_path}")
