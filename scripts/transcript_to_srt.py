#!/usr/bin/env python3
"""
将 transcript JSON 转换为 SRT 字幕文件。

用法:
    python transcript_to_srt.py <transcript.json> [--offset SECONDS] [--output FILE]

示例:
    python transcript_to_srt.py transcript.json --offset 225 --output "CS 61C 5.srt"
"""

import json
import sys
import os
import argparse


def seconds_to_srt_time(seconds: float) -> str:
    """将秒数转换为 SRT 时间格式 HH:MM:SS,mmm"""
    if seconds < 0:
        seconds = 0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def transcript_to_srt(transcript_path: str, offset: float = 0.0, output_path: str = None,
                      max_chars: int = 80) -> str:
    """
    将 transcript JSON 转换为 SRT 字幕。

    Args:
        transcript_path: transcript JSON 文件路径
        offset: 时间偏移量（秒），加到每个时间戳上
        output_path: 输出 SRT 文件路径
        max_chars: 每条字幕最大字符数（超长则分行）
    """
    with open(transcript_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get("segments", [])
    srt_entries = []
    idx = 1

    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue

        start = seg.get("start", 0) + offset
        end = seg.get("end", 0) + offset

        # 跳过偏移后为负数的段
        if end <= 0:
            continue
        if start < 0:
            start = 0

        # 如果文本太长，分成多行显示
        if len(text) > max_chars:
            words = text.split()
            lines = []
            current_line = []
            current_len = 0
            for word in words:
                if current_len + len(word) + 1 > max_chars and current_line:
                    lines.append(" ".join(current_line))
                    current_line = [word]
                    current_len = len(word)
                else:
                    current_line.append(word)
                    current_len += len(word) + 1
            if current_line:
                lines.append(" ".join(current_line))
            display_text = "\n".join(lines)
        else:
            display_text = text

        srt_entries.append(
            f"{idx}\n"
            f"{seconds_to_srt_time(start)} --> {seconds_to_srt_time(end)}\n"
            f"{display_text}\n"
        )
        idx += 1

    srt_content = "\n".join(srt_entries)

    # 确定输出路径
    if not output_path:
        base = os.path.splitext(transcript_path)[0]
        # 移除 _transcript 后缀
        if base.endswith("_transcript"):
            base = base[:-11]
        output_path = base + ".srt"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(srt_content)

    print(f"SRT file: {output_path}")
    print(f"Segments: {len(srt_entries)}")
    print(f"Offset: {offset}s ({offset/60:.1f}min)")
    if srt_entries:
        # 显示前3条和最后1条
        print(f"\n--- Preview (first 3) ---")
        for entry in srt_entries[:3]:
            print(entry.strip())
            print()

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Convert transcript JSON to SRT subtitles")
    parser.add_argument("transcript", help="Path to transcript JSON file")
    parser.add_argument("--offset", type=float, default=0.0,
                        help="Time offset in seconds to add to all timestamps")
    parser.add_argument("--output", "-o", help="Output SRT file path")
    parser.add_argument("--max-chars", type=int, default=80,
                        help="Max characters per subtitle line")

    args = parser.parse_args()

    if not os.path.exists(args.transcript):
        print(f"Error: file not found: {args.transcript}")
        sys.exit(1)

    transcript_to_srt(args.transcript, args.offset, args.output, args.max_chars)


if __name__ == "__main__":
    main()
