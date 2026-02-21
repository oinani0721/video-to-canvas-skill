#!/usr/bin/env python3
"""
将 SRT 字幕翻译为目标语言。使用 Gemini API 批量翻译。

用法:
    python translate_srt.py <input.srt> --lang zh --output <output.srt>
"""

import re
import sys
import os
import time
import argparse

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("Error: pip install google-genai")
    sys.exit(1)

# 从 lore-engine/.env 加载 API Key
LORE_ENGINE_ENV = os.path.expanduser("~/lore-engine/.env")
if os.path.exists(LORE_ENGINE_ENV):
    with open(LORE_ENGINE_ENV, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                if key.startswith("GEMINI_API_KEY") and key not in os.environ:
                    os.environ[key] = value


def parse_srt(srt_text: str) -> list:
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


def entries_to_srt(entries: list) -> str:
    """将条目列表转换回 SRT 格式"""
    blocks = []
    for e in entries:
        blocks.append(f"{e['idx']}\n{e['timing']}\n{e['text']}\n")
    return '\n'.join(blocks)


def translate_batch(client, texts: list, target_lang: str) -> list:
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


def main():
    parser = argparse.ArgumentParser(description="Translate SRT subtitles")
    parser.add_argument("input", help="Input SRT file")
    parser.add_argument("--lang", default="zh", help="Target language (default: zh)")
    parser.add_argument("--output", "-o", help="Output SRT file")
    parser.add_argument("--batch-size", type=int, default=30, help="Batch size for translation")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found")
        sys.exit(1)

    # 确定输出路径
    if not args.output:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}.{args.lang}{ext}"

    # 初始化 Gemini
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY_1") or os.getenv("GOOGLE_AI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not set")
        sys.exit(1)
    client = genai.Client(api_key=api_key)

    # 读取 SRT
    with open(args.input, "r", encoding="utf-8") as f:
        srt_text = f.read()

    entries = parse_srt(srt_text)
    print(f"Input: {args.input} ({len(entries)} entries)")
    print(f"Target: {args.lang}")

    # 批量翻译
    all_texts = [e['text'] for e in entries]
    translated = []
    batch_size = args.batch_size

    for i in range(0, len(all_texts), batch_size):
        batch = all_texts[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(all_texts) + batch_size - 1) // batch_size
        print(f"  Translating batch {batch_num}/{total_batches} ({len(batch)} entries)...")

        try:
            result = translate_batch(client, batch, args.lang)
            translated.extend(result)
        except Exception as e:
            print(f"  Error in batch {batch_num}: {e}")
            translated.extend(batch)  # fallback to original

        if i + batch_size < len(all_texts):
            time.sleep(1)  # rate limit

    # 更新条目
    for i, e in enumerate(entries):
        if i < len(translated):
            e['text'] = translated[i]

    # 写入
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(entries_to_srt(entries))

    print(f"\nOutput: {args.output}")
    print(f"Translated: {len(translated)} entries")

    # Preview
    print(f"\n--- Preview ---")
    for e in entries[:5]:
        print(f"{e['idx']}: {e['text']}")


if __name__ == "__main__":
    main()
