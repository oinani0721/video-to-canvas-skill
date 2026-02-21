"""
音频转录模块 - 三阶段混合管道的 Stage 1 (Ears)

支持两种后端：
1. faster-whisper (首选): 本地运行，速度快，支持 VAD 去幻觉
2. Gemini Audio (备选): 云端 API，无需本地模型

输出格式：
    [{"start": 0.0, "end": 5.2, "text": "...", "words": [...]}, ...]
"""

import os
import sys
import json
import subprocess
import tempfile
import time


class TranscriptSegment:
    """单条转录片段"""
    def __init__(self, start: float, end: float, text: str):
        self.start = start
        self.end = end
        self.text = text.strip()

    def to_dict(self) -> dict:
        return {"start": self.start, "end": self.end, "text": self.text}

    def __repr__(self):
        return f"[{self.start:.1f}-{self.end:.1f}] {self.text[:50]}"


class TranscriptResult:
    """完整转录结果"""
    def __init__(self, segments: list, backend: str, language: str = ""):
        self.segments = segments  # list of TranscriptSegment
        self.backend = backend   # "faster-whisper" or "gemini"
        self.language = language
        self.duration = segments[-1].end if segments else 0.0

    def to_dict(self) -> dict:
        return {
            "backend": self.backend,
            "language": self.language,
            "duration": self.duration,
            "segment_count": len(self.segments),
            "segments": [s.to_dict() for s in self.segments]
        }

    def get_text(self) -> str:
        """获取完整纯文本"""
        return " ".join(s.text for s in self.segments)

    def get_chunks(self, max_duration: float = 900.0) -> list:
        """
        将转录结果按时间切分为多个 chunk（默认 15 分钟）

        Args:
            max_duration: 每个 chunk 的最大时长（秒），默认 900 = 15 分钟

        Returns:
            list of dict: [{"start": float, "end": float, "text": str, "segments": [...]}]
        """
        if not self.segments:
            return []

        chunks = []
        current_chunk_start = self.segments[0].start
        current_segments = []

        for seg in self.segments:
            # 如果当前 chunk 超过最大时长，切分
            if current_segments and (seg.end - current_chunk_start) > max_duration:
                chunks.append({
                    "start": current_chunk_start,
                    "end": current_segments[-1].end,
                    "text": " ".join(s.text for s in current_segments),
                    "segments": [s.to_dict() for s in current_segments]
                })
                current_chunk_start = seg.start
                current_segments = []

            current_segments.append(seg)

        # 最后一个 chunk
        if current_segments:
            chunks.append({
                "start": current_chunk_start,
                "end": current_segments[-1].end,
                "text": " ".join(s.text for s in current_segments),
                "segments": [s.to_dict() for s in current_segments]
            })

        return chunks

    def get_text_for_timerange(self, start: float, end: float) -> str:
        """获取指定时间范围内的文本"""
        texts = []
        for seg in self.segments:
            if seg.end > start and seg.start < end:
                texts.append(seg.text)
        return " ".join(texts)


def extract_audio(video_path: str, output_path: str = None, sample_rate: int = 16000) -> str:
    """
    使用 FFmpeg 从视频提取音频

    Args:
        video_path: 视频文件路径
        output_path: 输出音频路径（默认生成临时文件）
        sample_rate: 采样率（默认 16kHz，Whisper 要求）

    Returns:
        音频文件路径
    """
    if output_path is None:
        output_path = tempfile.mktemp(suffix=".wav")

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",                    # 不要视频
        "-acodec", "pcm_s16le",   # 16-bit PCM
        "-ar", str(sample_rate),  # 采样率
        "-ac", "1",               # 单声道
        output_path
    ]

    print(f"  提取音频: {video_path} → {output_path}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg 音频提取失败: {result.stderr[:500]}")

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  音频文件大小: {file_size_mb:.1f} MB")

    return output_path


def _check_faster_whisper() -> bool:
    """检查 faster-whisper 是否可用"""
    try:
        import faster_whisper
        return True
    except ImportError:
        return False


def transcribe_with_faster_whisper(
    audio_path: str,
    model_size: str = "large-v3",
    language: str = None,
    vad_filter: bool = True,
    device: str = "auto",
    compute_type: str = "auto"
) -> TranscriptResult:
    """
    使用 faster-whisper 进行本地转录

    Args:
        audio_path: 音频文件路径
        model_size: 模型大小 (tiny/base/small/medium/large-v3)
        language: 语言代码 (None=自动检测, "en", "zh" 等)
        vad_filter: 是否启用 VAD 过滤（强烈推荐，减少幻觉）
        device: 计算设备 ("auto"/"cpu"/"cuda")
        compute_type: 精度 ("auto"/"float16"/"int8")

    Returns:
        TranscriptResult
    """
    from faster_whisper import WhisperModel

    # 自动检测设备和精度
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"

    print(f"  加载 faster-whisper 模型: {model_size} ({device}/{compute_type})")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    # VAD 参数（社区验证的最佳配置）
    vad_params = None
    if vad_filter:
        vad_params = {
            "threshold": 0.5,
            "min_speech_duration_ms": 250,
            "max_speech_duration_s": 30,        # 防止超长段
            "min_silence_duration_ms": 500,
            "speech_pad_ms": 200,
        }

    print(f"  转录中 (VAD={'ON' if vad_filter else 'OFF'})...")
    start_time = time.time()

    segments_iter, info = model.transcribe(
        audio_path,
        language=language,
        vad_filter=vad_filter,
        vad_parameters=vad_params,
        word_timestamps=True,
        condition_on_previous_text=True,
    )

    detected_lang = info.language
    print(f"  检测语言: {detected_lang} (概率: {info.language_probability:.2f})")

    # 收集所有段落
    segments = []
    for seg in segments_iter:
        segments.append(TranscriptSegment(
            start=seg.start,
            end=seg.end,
            text=seg.text
        ))

    elapsed = time.time() - start_time
    print(f"  转录完成: {len(segments)} 个片段, 耗时 {elapsed:.1f}s")

    return TranscriptResult(
        segments=segments,
        backend="faster-whisper",
        language=detected_lang
    )


def transcribe_with_gemini(
    audio_path: str,
    client=None,
    language: str = None
) -> TranscriptResult:
    """
    使用 Gemini API 进行云端转录

    注意：Gemini 的时间戳精度不如 Whisper，但无需本地 GPU

    Args:
        audio_path: 音频文件路径
        client: google.genai.Client 实例
        language: 语言提示

    Returns:
        TranscriptResult
    """
    from google import genai
    from google.genai import types

    if client is None:
        api_key = (
            os.getenv("GEMINI_API_KEY") or
            os.getenv("GEMINI_API_KEY_1") or
            os.getenv("GOOGLE_AI_API_KEY")
        )
        if not api_key:
            raise ValueError("需要 GEMINI_API_KEY 环境变量")
        client = genai.Client(api_key=api_key)

    # 上传音频文件
    print(f"  上传音频到 Gemini...")
    audio_file = client.files.upload(file=audio_path)

    # 等待处理
    while audio_file.state.name == "PROCESSING":
        print(f"  处理中...")
        time.sleep(3)
        audio_file = client.files.get(name=audio_file.name)

    if audio_file.state.name == "FAILED":
        raise RuntimeError("Gemini 音频处理失败")

    lang_hint = f"语言: {language}" if language else "自动检测语言"

    prompt = f"""请将这段音频完整转录为文本。{lang_hint}

要求：
1. 输出 JSON 数组格式
2. 每个元素包含 start (秒), end (秒), text (文本)
3. 每段不超过 30 秒
4. 保留所有口语内容，不要省略
5. 时间戳尽可能精确

输出格式：
[
  {{"start": 0.0, "end": 5.2, "text": "..."}},
  {{"start": 5.2, "end": 12.1, "text": "..."}}
]

只输出 JSON，不要其他内容。"""

    print(f"  Gemini 转录中...")
    start_time = time.time()

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[audio_file, prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        )
    )

    # 解析响应
    text = response.text.strip()
    # 清理 markdown 代码块
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        raw_segments = json.loads(text)
    except json.JSONDecodeError as e:
        # 尝试修复常见 JSON 问题
        import re
        print(f"  [JSON修复] 原始错误: {e}")

        # 修复 1: 尾随逗号
        fixed = re.sub(r',\s*]', ']', text)
        fixed = re.sub(r',\s*}', '}', fixed)

        # 修复 2: 单引号 → 双引号 (key/value)
        fixed = re.sub(r"'(start|end|text)'", r'"\1"', fixed)

        try:
            raw_segments = json.loads(fixed)
        except json.JSONDecodeError:
            # 修复 3: 截断到最后一个完整的 }, 再闭合数组
            last_brace = fixed.rfind('}')
            if last_brace > 0:
                truncated = fixed[:last_brace + 1]
                # 确保以 ] 结尾
                if not truncated.rstrip().endswith(']'):
                    truncated = truncated.rstrip().rstrip(',') + '\n]'
                try:
                    raw_segments = json.loads(truncated)
                    print(f"  [JSON修复] 截断修复成功，保留到 char {last_brace}")
                except json.JSONDecodeError:
                    pass

            # 修复 4: 正则逐条提取
            if 'raw_segments' not in dir():
                pattern = r'\{\s*"start"\s*:\s*([\d.]+)\s*,\s*"end"\s*:\s*([\d.]+)\s*,\s*"text"\s*:\s*"((?:[^"\\]|\\.)*)"\s*\}'
                matches = re.findall(pattern, text, re.DOTALL)
                if matches:
                    raw_segments = [
                        {"start": float(m[0]), "end": float(m[1]), "text": m[2]}
                        for m in matches
                    ]
                    print(f"  [JSON修复] 正则提取成功: {len(raw_segments)} 个片段")
                else:
                    raise RuntimeError(f"无法解析 Gemini 转录结果 (JSON 损坏): {e}")

    segments = []
    for item in raw_segments:
        segments.append(TranscriptSegment(
            start=float(item.get("start", 0)),
            end=float(item.get("end", 0)),
            text=str(item.get("text", ""))
        ))

    elapsed = time.time() - start_time
    print(f"  转录完成: {len(segments)} 个片段, 耗时 {elapsed:.1f}s")

    # 清理上传的文件
    try:
        client.files.delete(name=audio_file.name)
    except Exception:
        pass

    return TranscriptResult(
        segments=segments,
        backend="gemini",
        language=language or "auto"
    )


def transcribe(
    video_path: str,
    backend: str = "auto",
    model_size: str = "large-v3",
    language: str = None,
    vad_filter: bool = True,
    gemini_client=None,
    keep_audio: bool = False
) -> TranscriptResult:
    """
    智能转录入口 - 自动选择最佳后端

    Args:
        video_path: 视频文件路径
        backend: 后端选择 ("auto"/"faster-whisper"/"gemini")
        model_size: Whisper 模型大小
        language: 语言提示
        vad_filter: 是否启用 VAD
        gemini_client: Gemini 客户端（用于 Gemini 后端）
        keep_audio: 是否保留提取的音频文件

    Returns:
        TranscriptResult
    """
    print("\n[Stage 1: Ears] 音频转录")
    print(f"  视频: {video_path}")

    # 提取音频
    audio_dir = os.path.dirname(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    audio_path = os.path.join(audio_dir, f"{video_name}_audio.wav")
    audio_path = extract_audio(video_path, audio_path)

    try:
        # 选择后端
        if backend == "auto":
            if _check_faster_whisper():
                backend = "faster-whisper"
                print("  后端: faster-whisper (本地, 推荐)")
            else:
                backend = "gemini"
                print("  后端: Gemini Audio (云端, faster-whisper 未安装)")

        if backend == "faster-whisper":
            result = transcribe_with_faster_whisper(
                audio_path,
                model_size=model_size,
                language=language,
                vad_filter=vad_filter
            )
        elif backend == "gemini":
            result = transcribe_with_gemini(
                audio_path,
                client=gemini_client,
                language=language
            )
        else:
            raise ValueError(f"未知后端: {backend}")

        print(f"  总时长: {result.duration:.0f}s ({result.duration/60:.1f}min)")
        print(f"  片段数: {len(result.segments)}")
        print(f"  后端: {result.backend}")

        return result

    finally:
        # 清理音频文件
        if not keep_audio and os.path.exists(audio_path):
            os.remove(audio_path)
            print(f"  已清理临时音频文件")


def save_transcript(result: TranscriptResult, output_path: str):
    """保存转录结果到 JSON 文件"""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
    print(f"  转录结果已保存: {output_path}")


def load_transcript(path: str) -> TranscriptResult:
    """从 JSON 文件加载转录结果"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = [
        TranscriptSegment(s["start"], s["end"], s["text"])
        for s in data["segments"]
    ]

    return TranscriptResult(
        segments=segments,
        backend=data.get("backend", "unknown"),
        language=data.get("language", "")
    )


# ============================================================================
# CLI 入口
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="视频音频转录工具")
    parser.add_argument("video", help="视频文件路径")
    parser.add_argument("-o", "--output", help="输出 JSON 路径")
    parser.add_argument("--backend", default="auto",
                        choices=["auto", "faster-whisper", "gemini"],
                        help="转录后端 (默认: auto)")
    parser.add_argument("--model", default="large-v3",
                        help="Whisper 模型大小 (默认: large-v3)")
    parser.add_argument("--language", default=None,
                        help="语言代码 (默认: 自动检测)")
    parser.add_argument("--no-vad", action="store_true",
                        help="禁用 VAD 过滤")
    parser.add_argument("--keep-audio", action="store_true",
                        help="保留提取的音频文件")
    parser.add_argument("--chunk-minutes", type=float, default=15.0,
                        help="分块时长（分钟，默认 15）")

    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"错误：视频文件不存在: {args.video}")
        sys.exit(1)

    result = transcribe(
        video_path=args.video,
        backend=args.backend,
        model_size=args.model,
        language=args.language,
        vad_filter=not args.no_vad,
        keep_audio=args.keep_audio
    )

    # 保存结果
    if args.output:
        output_path = args.output
    else:
        video_name = os.path.splitext(os.path.basename(args.video))[0]
        output_path = f"{video_name}_transcript.json"

    save_transcript(result, output_path)

    # 显示分块信息
    chunks = result.get_chunks(max_duration=args.chunk_minutes * 60)
    print(f"\n分块结果 ({args.chunk_minutes}min/块):")
    for i, chunk in enumerate(chunks):
        mins = (chunk["end"] - chunk["start"]) / 60
        words = len(chunk["text"].split())
        print(f"  块 {i+1}: [{chunk['start']:.0f}s - {chunk['end']:.0f}s] "
              f"({mins:.1f}min, ~{words} 词)")

    # 预览前 3 段
    print(f"\n前 3 段预览:")
    for seg in result.segments[:3]:
        print(f"  {seg}")
