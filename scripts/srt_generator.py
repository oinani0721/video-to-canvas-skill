"""
SRT 字幕生成模块 - 从 TranscriptResult 生成精确时间戳字幕

支持:
- 从 faster-whisper 转录直接生成 SRT（精确时间戳，无需偏移）
- 从 Gemini 转录生成 SRT（需要时间偏移校正）
- 使用 Gemini API 批量翻译生成多语言字幕
"""

import os
import re
import time


def _format_srt_time(seconds: float) -> str:
    """将秒数转换为 SRT 时间格式 HH:MM:SS,mmm"""
    if seconds < 0:
        seconds = 0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def generate_srt_from_transcript(transcript_result, output_path: str, max_chars: int = 80) -> str:
    """
    从 TranscriptResult 生成 SRT 字幕文件。

    faster-whisper 的时间戳直接对应视频实际时间，无需偏移。
    Gemini 的时间戳可能有时间基准偏移，但仍直接使用（用户可通过播放器微调）。

    Args:
        transcript_result: TranscriptResult 对象
        output_path: 输出 SRT 文件路径
        max_chars: 每条字幕最大字符数（超长则分行）

    Returns:
        输出文件路径
    """
    srt_entries = []
    idx = 1

    for seg in transcript_result.segments:
        text = seg.text.strip()
        if not text:
            continue

        start = seg.start
        end = seg.end

        # 跳过无效时间段
        if end <= 0 or end <= start:
            continue

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
            f"{_format_srt_time(start)} --> {_format_srt_time(end)}\n"
            f"{display_text}\n"
        )
        idx += 1

    srt_content = "\n".join(srt_entries)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(srt_content)

    return output_path


def _parse_srt(srt_text: str) -> list:
    """解析 SRT 文件为条目列表"""
    entries = []
    blocks = re.split(r'\n\n+', srt_text.strip())

    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue
        try:
            idx = int(lines[0].strip())
            timing = lines[1].strip()
            text = '\n'.join(lines[2:])
            entries.append({
                'idx': idx,
                'timing': timing,
                'text': text
            })
        except (ValueError, IndexError):
            continue

    return entries


def _entries_to_srt(entries: list) -> str:
    """将条目列表转换回 SRT 格式"""
    blocks = []
    for e in entries:
        blocks.append(f"{e['idx']}\n{e['timing']}\n{e['text']}\n")
    return '\n'.join(blocks)


def _translate_batch(client, texts: list, target_lang: str) -> list:
    """批量翻译文本"""
    numbered = '\n'.join(f"[{i}] {t}" for i, t in enumerate(texts))

    prompt = f"""Translate the following subtitle lines to {target_lang}.
Keep the [N] numbering prefix. Only output the translations, nothing else.
Keep each translation on one line after its [N] prefix.
For very short utterances like "Sure", "Okay", "Um", translate naturally.

{numbered}"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[prompt],
    )

    result_text = response.text.strip()

    # 解析结果
    translations = {}
    for line in result_text.split('\n'):
        line = line.strip()
        m = re.match(r'\[(\d+)\]\s*(.*)', line)
        if m:
            translations[int(m.group(1))] = m.group(2)

    # 按顺序返回，缺失的用原文
    return [translations.get(i, texts[i]) for i in range(len(texts))]


def translate_srt_file(
    input_path: str,
    output_path: str,
    target_lang: str,
    gemini_client=None,
    batch_size: int = 40
) -> str:
    """
    翻译 SRT 字幕文件为目标语言。

    Args:
        input_path: 输入 SRT 文件路径
        output_path: 输出 SRT 文件路径
        target_lang: 目标语言代码 (如 "zh", "ja", "ko")
        gemini_client: Gemini 客户端（如果为 None，尝试自动创建）
        batch_size: 每批翻译的条目数

    Returns:
        输出文件路径
    """
    if gemini_client is None:
        try:
            from google import genai
            api_key = (
                os.getenv("GEMINI_API_KEY") or
                os.getenv("GEMINI_API_KEY_1") or
                os.getenv("GOOGLE_AI_API_KEY")
            )
            if not api_key:
                print("  [SRT translate] GEMINI_API_KEY not set, skipping translation")
                return None
            gemini_client = genai.Client(api_key=api_key)
        except ImportError:
            print("  [SRT translate] google-genai not installed, skipping translation")
            return None

    # 读取 SRT
    with open(input_path, "r", encoding="utf-8") as f:
        srt_text = f.read()

    entries = _parse_srt(srt_text)
    if not entries:
        print("  [SRT translate] No entries found in SRT file")
        return None

    print(f"  [SRT translate] {len(entries)} entries -> {target_lang}")

    # 批量翻译
    all_texts = [e['text'] for e in entries]
    translated = []

    for i in range(0, len(all_texts), batch_size):
        batch = all_texts[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(all_texts) + batch_size - 1) // batch_size
        print(f"    Translating batch {batch_num}/{total_batches}...")

        try:
            result = _translate_batch(gemini_client, batch, target_lang)
            translated.extend(result)
        except Exception as e:
            print(f"    Error in batch {batch_num}: {e}")
            translated.extend(batch)  # fallback to original

        if i + batch_size < len(all_texts):
            time.sleep(1)  # rate limit

    # 更新条目
    for i, e in enumerate(entries):
        if i < len(translated):
            e['text'] = translated[i]

    # 写入
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(_entries_to_srt(entries))

    print(f"  [SRT translate] Done: {output_path} ({len(translated)} entries)")
    return output_path
