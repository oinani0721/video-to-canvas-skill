"""
差异化描述生成器 — Stage 2b

基于 ShareGPT4Video (NeurIPS 2024) 研究：
- 详细差异描述 → +3.4 avg points
- 短描述（30-100字） → +0.2（几乎无用）
- 结论：200+ 字的 Rich Differential Caption 是关键

架构：
┌────────────────────────────────────────────────┐
│ Stage 2a: 本地变化检测 (local_detector)          │
│  → change_points + screenshots                  │
└────────────────────────────────────────────────┘
                      ↓
┌────────────────────────────────────────────────┐
│ Stage 2b: 差异化描述生成 (本模块)                │
│  • 相邻截图对比 → 200+ 字描述                    │
│  • 并发 API 调用（Gemini Flash）                  │
│  • 输出：enriched change_points                  │
└────────────────────────────────────────────────┘
                      ↓
┌────────────────────────────────────────────────┐
│ Stage 3: LLM 笔记生成（注意力锚点引导）          │
└────────────────────────────────────────────────┘

成本：~$0.001/对（Gemini Flash 图片输入），远低于视频理解
"""

import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional


# ========== 提示词 ==========

FIRST_FRAME_PROMPT = """\
Describe this screenshot from a video. Include:
1. All visible text (titles, bullet points, formulas, code) — transcribe accurately
2. Any diagrams, charts, tables, or visual elements — describe structure
3. The apparent topic or context

Output a single paragraph in Chinese, 200-400 characters. Focus on CONTENT, not appearance.\
"""

DIFF_CAPTION_PROMPT = """\
Compare these two consecutive screenshots from a video.

Image 1: previous frame [{ts_prev}]
Image 2: current frame [{ts_curr}]

Describe:
1. What specifically changed between the two frames (new text, new diagram, slide transition, code modification, etc.)
2. Key content visible in the current frame (transcribe important text, formulas, code accurately)
3. If it's a progressive change (e.g., new bullet point added, search tree expanded one level), describe the delta precisely

Output a single paragraph in Chinese, 200-400 characters. Focus on CONTENT changes, not visual styling.\
"""

DIFF_CAPTION_SCHEMA = {
    "type": "object",
    "properties": {
        "caption": {
            "type": "string",
            "description": "200-400 characters differential caption in Chinese"
        }
    },
    "required": ["caption"]
}


# ========== 图片加载 ==========

def _load_image_part(image_path: str) -> Optional[dict]:
    """Load image as Gemini-compatible Part"""
    if not os.path.exists(image_path):
        return None
    try:
        from google.genai import types
        with open(image_path, "rb") as f:
            data = f.read()

        ext = os.path.splitext(image_path)[1].lower()
        mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}.get(ext, "image/jpeg")

        return types.Part.from_bytes(data=data, mime_type=mime)
    except Exception as e:
        print(f"  [Caption] Failed to load image {image_path}: {e}")
        return None


def _get_screenshot_path(cp: dict, screenshot_dir: str) -> Optional[str]:
    """Get screenshot file path for a change point"""
    ts = cp.get("timestamp", "00:00").replace(":", "-")
    for ext in (".jpg", ".png"):
        path = os.path.join(screenshot_dir, f"{ts}{ext}")
        if os.path.exists(path):
            return path
    return None


# ========== 单对描述生成 ==========

def _caption_first_frame(client, screenshot_path: str, model: str) -> str:
    """Generate caption for the first frame (no comparison)"""
    from google.genai import types

    img_part = _load_image_part(screenshot_path)
    if img_part is None:
        return ""

    try:
        response = client.models.generate_content(
            model=model,
            contents=[img_part, FIRST_FRAME_PROMPT],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=DIFF_CAPTION_SCHEMA
            )
        )
        result = json.loads(response.text)
        return result.get("caption", "")
    except Exception as e:
        print(f"  [Caption] First frame caption failed: {e}")
        return ""


def _caption_diff_pair(
    client,
    prev_path: str,
    curr_path: str,
    ts_prev: str,
    ts_curr: str,
    model: str
) -> str:
    """Generate differential caption for a pair of adjacent screenshots"""
    from google.genai import types

    prev_img = _load_image_part(prev_path)
    curr_img = _load_image_part(curr_path)
    if prev_img is None or curr_img is None:
        return ""

    prompt = DIFF_CAPTION_PROMPT.replace("{ts_prev}", ts_prev).replace("{ts_curr}", ts_curr)

    try:
        response = client.models.generate_content(
            model=model,
            contents=[prev_img, curr_img, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=DIFF_CAPTION_SCHEMA
            )
        )
        result = json.loads(response.text)
        return result.get("caption", "")
    except Exception as e:
        print(f"  [Caption] Diff caption failed ({ts_prev} → {ts_curr}): {e}")
        return ""


# ========== 主入口 ==========

def caption_screenshots(
    client,
    change_points: List[dict],
    screenshot_dir: str,
    model: str = "gemini-2.5-flash",
    max_workers: int = 5
) -> List[dict]:
    """
    Stage 2b: Generate rich differential captions for change points.

    For each pair of adjacent screenshots, sends both images to Gemini Flash
    to get a detailed 200-400 char description of what changed.

    Args:
        client: Gemini API client
        change_points: List of change point dicts from Stage 2a
        screenshot_dir: Directory containing extracted screenshots
        model: Gemini model to use
        max_workers: Max concurrent API calls

    Returns:
        Enriched change_points with 'differential_caption' field added
    """
    if not change_points:
        return change_points

    print(f"\n[Stage 2b] Differential captioning ({len(change_points)} screenshots)...")

    # Resolve screenshot paths
    paths = []
    for cp in change_points:
        path = _get_screenshot_path(cp, screenshot_dir)
        paths.append(path)

    # Count available screenshots
    available = sum(1 for p in paths if p is not None)
    if available == 0:
        print("  [Caption] No screenshots found, skipping captioning")
        return change_points

    print(f"  [Caption] {available}/{len(change_points)} screenshots available")

    # Build tasks: (index, prev_path_or_None, curr_path, ts_prev, ts_curr)
    tasks = []
    for i, cp in enumerate(change_points):
        if paths[i] is None:
            continue
        if i == 0:
            tasks.append((i, None, paths[i], None, cp["timestamp"]))
        else:
            prev_path = paths[i - 1]
            if prev_path is None:
                # No previous screenshot; treat as first frame
                tasks.append((i, None, paths[i], None, cp["timestamp"]))
            else:
                tasks.append((i, prev_path, paths[i],
                              change_points[i - 1]["timestamp"], cp["timestamp"]))

    # Execute concurrently
    results = {}

    def _do_caption(task):
        idx, prev_path, curr_path, ts_prev, ts_curr = task
        if prev_path is None:
            caption = _caption_first_frame(client, curr_path, model)
        else:
            caption = _caption_diff_pair(client, prev_path, curr_path,
                                         ts_prev, ts_curr, model)
        return idx, caption

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_do_caption, t): t[0] for t in tasks}
        done_count = 0
        for future in as_completed(futures):
            idx, caption = future.result()
            results[idx] = caption
            done_count += 1
            if done_count % 10 == 0 or done_count == len(tasks):
                print(f"  [Caption] {done_count}/{len(tasks)} completed")

    # Enrich change_points
    captioned_count = 0
    for i, cp in enumerate(change_points):
        caption = results.get(i, "")
        if caption:
            cp["differential_caption"] = caption
            captioned_count += 1
        else:
            # Fallback: use existing description or OCR text
            cp["differential_caption"] = cp.get("ocr_text", cp.get("description", ""))

    print(f"  [Caption] {captioned_count}/{len(change_points)} enriched with differential captions")
    return change_points


# ========== 格式化输出（供 prompt_builder 使用） ==========

def format_captions_for_prompt(change_points: List[dict]) -> str:
    """
    Format differential captions for inclusion in Stage 3 prompt.

    Output format:
    ---
    [00:15] → [01:30]: 幻灯片从标题页切换到目录页，新增了5个章节标题...
    [01:30] → [02:45]: 进入第一章"搜索算法概述"，出现DFS/BFS/UCS的分类图...
    ---
    """
    if not change_points:
        return ""

    lines = []
    for i, cp in enumerate(change_points):
        ts = cp.get("timestamp", "??:??")
        caption = cp.get("differential_caption", "")
        if not caption:
            caption = cp.get("description", "")

        if i == 0:
            lines.append(f"[{ts}] (start): {caption}")
        else:
            prev_ts = change_points[i - 1].get("timestamp", "??:??")
            lines.append(f"[{prev_ts}] → [{ts}]: {caption}")

    return "\n".join(lines)


# ========== CLI test ==========

if __name__ == "__main__":
    # Dry run test — format only, no API calls
    test_points = [
        {"timestamp": "00:15", "description": "Title slide", "differential_caption": "标题页显示课程名称'CS188 人工智能导论'，副标题'搜索算法'，教授姓名和学期信息。背景为蓝色渐变，左上角有UC Berkeley校徽。"},
        {"timestamp": "01:30", "description": "Outline", "differential_caption": "从标题页切换到目录页。新增5个章节标题：1.搜索问题定义 2.DFS深度优先 3.BFS广度优先 4.UCS代价一致 5.A*启发式搜索。每个标题前有编号和图标。"},
        {"timestamp": "03:00", "description": "DFS diagram", "differential_caption": "进入第二章DFS部分。出现3层搜索树示意图，根节点S标红表示当前扩展节点，左子树A-D-E已完全展开（灰色），右子树B-F正在展开（绿色箭头指向F）。树旁边有frontier栈的状态：[F, C]。"},
    ]

    print("Formatted captions:")
    print(format_captions_for_prompt(test_points))
