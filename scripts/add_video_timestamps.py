#!/usr/bin/env python3
"""
将 Markdown 笔记中的截图时间戳转换为 Media Extended 可点击链接。

⚠️ 重要：只转换截图时间戳（来自 FFmpeg，准确），
   移除转录时间戳（来自 Gemini Audio，时间基准不准确）。

用法:
    python add_video_timestamps.py <md_file> <video_filename>
    python add_video_timestamps.py "CS 61C 5.md" "CS 61C 5.mp4"

转换规则:
    截图行: *图：描述 [07:05]*  →  *图：描述 [[video.mp4#t=425|⏱07:05]]*
    正文行: 概念说明 [02:30]   →  概念说明 [02:30]()  (Media Extended 可点击格式)
    时间范围: [02:30-03:15]     →  (移除，不适合做可点击链接)
"""

import re
import sys
import os


def time_to_seconds(time_str: str) -> int:
    """MM:SS 或 HH:MM:SS 转换为秒数"""
    parts = time_str.split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return 0


def is_screenshot_caption_line(line: str) -> bool:
    """判断是否为截图说明行（来自 Stage 2 FFmpeg，时间戳准确）"""
    stripped = line.strip()
    # 模式: *图：...* 或 *Figure:...*
    if stripped.startswith("*") and ("图：" in stripped or "图:" in stripped or "Figure:" in stripped):
        return True
    return False


def convert_timestamps(md_text: str, video_filename: str) -> str:
    """
    智能转换时间戳：
    - 截图说明行的时间戳 → 可点击 Media Extended wikilink（准确）
    - 正文中的单时间戳 → [MM:SS]() 可点击格式
    - 正文中的时间范围 → 移除（不适合做可点击链接）
    """
    lines = md_text.split("\n")
    result = []
    in_code_block = False

    # 匹配 [MM:SS] 或 [HH:MM:SS]，兼容已有 () 后缀
    ts_pattern = re.compile(r'(?<!\!)\[(\d{1,2}:\d{2}(?::\d{2})?)\](?:\(\))?')
    # 匹配 [MM:SS-MM:SS] 时间范围，兼容已有 () 后缀
    ts_range_pattern = re.compile(r'\[(\d{1,2}:\d{2}(?::\d{2})?)-(\d{1,2}:\d{2}(?::\d{2})?)\](?:\(\))?')

    screenshot_converted = 0
    inline_converted = 0
    transcript_removed = 0

    for line in lines:
        # 跳过代码块
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            result.append(line)
            continue

        if in_code_block:
            result.append(line)
            continue

        if is_screenshot_caption_line(line):
            # ✅ 截图说明行：转换为可点击链接
            def replace_screenshot_ts(match):
                ts = match.group(1)
                seconds = time_to_seconds(ts)
                return f"[[{video_filename}#t={seconds}|⏱{ts}]]"

            new_line = ts_pattern.sub(replace_screenshot_ts, line)
            if new_line != line:
                screenshot_converted += ts_pattern.findall(line).__len__()
            result.append(new_line)
        else:
            # 正文行：转换单时间戳为可点击格式，移除时间范围
            # 先移除时间范围（不适合做可点击链接）
            new_line, n1 = ts_range_pattern.subn("", line)
            # 单时间戳 → [MM:SS]() 可点击格式
            def replace_inline_ts(match):
                ts = match.group(1)
                return f"[{ts}]()"
            new_line, n2 = ts_pattern.subn(replace_inline_ts, new_line)
            inline_converted += n2
            transcript_removed += n1

            # 清理移除后可能产生的多余空格
            new_line = re.sub(r'  +', ' ', new_line)
            new_line = re.sub(r' ([。，、；：])', r'\1', new_line)

            result.append(new_line)

    print(f"  截图时间戳 → 可点击 wikilink: {screenshot_converted} 个")
    print(f"  内联时间戳 → 可点击 [MM:SS](): {inline_converted} 个")
    print(f"  时间范围 → 已移除: {transcript_removed} 个")

    return "\n".join(result)


def add_video_embed(md_text: str, video_filename: str) -> str:
    """在笔记顶部添加视频嵌入（如果还没有的话）"""
    embed_marker = f"![[{video_filename}]]"

    if embed_marker in md_text:
        return md_text

    lines = md_text.split("\n")
    insert_idx = 0

    # 跳过 frontmatter (---...---)
    if lines and lines[0].strip() == "---":
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                insert_idx = i + 1
                break

    # 跳过空行
    while insert_idx < len(lines) and not lines[insert_idx].strip():
        insert_idx += 1

    # 插入视频嵌入
    embed_block = f"\n> [!video]- 视频播放器\n> {embed_marker}\n"
    lines.insert(insert_idx, embed_block)

    return "\n".join(lines)


def main():
    if len(sys.argv) < 3:
        print("用法: python add_video_timestamps.py <md_file> <video_filename>")
        print('示例: python add_video_timestamps.py "CS 61C 5.md" "CS 61C 5.mp4"')
        sys.exit(1)

    md_path = sys.argv[1]
    video_filename = sys.argv[2]

    if not os.path.exists(md_path):
        print(f"错误: 文件不存在: {md_path}")
        sys.exit(1)

    # 检查是否有备份文件，优先从备份恢复
    backup_path = md_path + ".bak"
    if os.path.exists(backup_path):
        print(f"从备份恢复: {backup_path}")
        with open(backup_path, "r", encoding="utf-8") as f:
            md_text = f.read()
    else:
        with open(md_path, "r", encoding="utf-8") as f:
            md_text = f.read()
        # 创建备份
        with open(backup_path, "w", encoding="utf-8") as f:
            f.write(md_text)
        print(f"备份: {backup_path}")

    print(f"文件: {md_path}")
    print(f"视频: {video_filename}")

    # 转换时间戳
    md_text = convert_timestamps(md_text, video_filename)

    # 添加视频嵌入
    md_text = add_video_embed(md_text, video_filename)

    # 写入修改后的文件
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)

    # 统计
    link_count = len(re.findall(r'\[\[.*?#t=\d+\|', md_text))
    print(f"\n[OK] 完成: {link_count} 个可点击截图时间戳")
    print(f"视频嵌入: 已添加")


if __name__ == "__main__":
    main()
