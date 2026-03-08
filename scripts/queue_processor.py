#!/usr/bin/env python3
"""
Video-to-Canvas Queue Processor
顺序处理多个视频的队列管理器，复用 video_to_md.main() 管道。

用法:
  python queue_processor.py add video1.mp4 video2.mp4 [--depth balanced --srt-lang zh]
  python queue_processor.py run [--daemon]
  python queue_processor.py status
  python queue_processor.py clear [--completed | --failed | --all]
"""

import json
import sys
import os
import argparse
import tempfile
import datetime
import threading

# 队列文件默认位置：skill 目录下
SKILL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_QUEUE_PATH = os.path.join(SKILL_DIR, "queue.json")

_queue_lock = threading.Lock()


def _atomic_write(path: str, data: dict):
    """原子写入 JSON 文件（写入临时文件后重命名，避免写入竞争）"""
    dir_name = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        # Windows: 需要先删除目标文件
        if sys.platform == "win32" and os.path.exists(path):
            os.replace(tmp_path, path)
        else:
            os.rename(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def load_queue(queue_path: str = DEFAULT_QUEUE_PATH) -> dict:
    """加载队列文件，不存在则返回空队列"""
    if os.path.exists(queue_path):
        with open(queue_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "status": "idle",
        "total": 0,
        "completed": 0,
        "failed": 0,
        "current_index": None,
        "items": [],
        "updated_at": None,
    }


def save_queue(queue_path: str, data: dict):
    """保存队列文件（线程安全 + 原子写入）"""
    with _queue_lock:
        data["updated_at"] = datetime.datetime.now().isoformat()
        # 重新计算统计
        data["total"] = len(data["items"])
        data["completed"] = sum(1 for i in data["items"] if i["status"] == "completed")
        data["failed"] = sum(1 for i in data["items"] if i["status"] == "failed")
        _atomic_write(queue_path, data)


# ── add 命令 ─────────────────────────────────────────

def cmd_add(videos: list, common_args: dict, queue_path: str = DEFAULT_QUEUE_PATH):
    """将视频添加到队列"""
    queue = load_queue(queue_path)
    next_id = max((item["id"] for item in queue["items"]), default=0) + 1

    added = []
    for video in videos:
        video_path = os.path.abspath(video)
        if not os.path.isfile(video_path):
            print(f"  [SKIP] 文件不存在: {video_path}")
            continue

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(os.path.dirname(video_path), video_name)

        queue["items"].append({
            "id": next_id,
            "video": video_path,
            "output": output_dir,
            "status": "pending",
            "args": common_args,
            "added_at": datetime.datetime.now().isoformat(),
            "started_at": None,
            "completed_at": None,
            "error": None,
        })
        added.append(video_name)
        next_id += 1

    save_queue(queue_path, queue)
    print(f"[Queue] 已添加 {len(added)} 个视频到队列:")
    for name in added:
        print(f"  + {name}")
    print(f"[Queue] 队列总数: {queue['total']}  待处理: {queue['total'] - queue['completed'] - queue['failed']}")
    return len(added)


# ── run 命令 ─────────────────────────────────────────

def cmd_run(queue_path: str = DEFAULT_QUEUE_PATH):
    """顺序处理队列中所有待处理的视频"""
    # 导入管道主函数
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from video_to_md import main as pipeline_main

    queue = load_queue(queue_path)
    if not queue["items"]:
        print("[Queue] 队列为空，无需处理")
        return

    queue["status"] = "processing"
    save_queue(queue_path, queue)

    processed = 0
    while True:
        # 每次循环重新加载队列（热加载：支持运行中追加新视频）
        queue = load_queue(queue_path)

        # 找到下一个 pending 的视频
        pending = [i for i, item in enumerate(queue["items"]) if item["status"] == "pending"]
        if not pending:
            break

        idx = pending[0]
        item = queue["items"][idx]
        queue["current_index"] = idx
        item["status"] = "processing"
        item["started_at"] = datetime.datetime.now().isoformat()
        save_queue(queue_path, queue)

        video_name = os.path.splitext(os.path.basename(item["video"]))[0]
        position = f"[{queue['completed'] + queue['failed'] + 1}/{queue['total']}]"
        print(f"\n{'='*60}")
        print(f"[Queue] {position} 开始处理: {video_name}")
        print(f"{'='*60}")

        try:
            args = item.get("args", {})
            os.makedirs(item["output"], exist_ok=True)

            pipeline_main(
                video_path=item["video"],
                output_dir=item["output"],
                depth=args.get("depth", "balanced"),
                density=args.get("density", "normal"),
                min_interval=args.get("min_interval", 2.0),
                fusion=args.get("fusion", False),
                transcribe_audio=args.get("transcribe_audio", True),
                transcribe_backend=args.get("backend", "auto"),
                whisper_model=args.get("whisper_model", "large-v3"),
                segment_minutes=args.get("segment_minutes", 15.0),
                transcript_path=args.get("transcript", None),
                generate_srt=args.get("generate_srt", True),
                srt_translate_lang=args.get("srt_lang", None),
            )

            # 重新加载（可能有新增项）后更新当前项
            queue = load_queue(queue_path)
            queue["items"][idx]["status"] = "completed"
            queue["items"][idx]["completed_at"] = datetime.datetime.now().isoformat()
            processed += 1
            print(f"\n[Queue] ✓ 完成: {video_name}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            queue = load_queue(queue_path)
            queue["items"][idx]["status"] = "failed"
            queue["items"][idx]["error"] = str(e)[:500]
            queue["items"][idx]["completed_at"] = datetime.datetime.now().isoformat()
            print(f"\n[Queue] ✗ 失败: {video_name} — {e}")

        queue["current_index"] = None
        save_queue(queue_path, queue)

    # 全部处理完成
    queue = load_queue(queue_path)
    queue["status"] = "completed"
    queue["current_index"] = None
    save_queue(queue_path, queue)

    print(f"\n{'='*60}")
    print(f"[Queue] 队列处理完成!")
    print(f"  完成: {queue['completed']}  失败: {queue['failed']}  总计: {queue['total']}")
    print(f"{'='*60}")
    return processed


# ── status 命令 ──────────────────────────────────────

def cmd_status(queue_path: str = DEFAULT_QUEUE_PATH):
    """显示队列状态"""
    queue = load_queue(queue_path)
    if not queue["items"]:
        print("[Queue] 队列为空")
        return

    pending = sum(1 for i in queue["items"] if i["status"] == "pending")
    processing = sum(1 for i in queue["items"] if i["status"] == "processing")

    print(f"[Queue] 状态: {queue['status']}")
    print(f"  总计: {queue['total']}  完成: {queue['completed']}  失败: {queue['failed']}  "
          f"处理中: {processing}  待处理: {pending}")
    print()

    for item in queue["items"]:
        name = os.path.basename(item["video"])
        status_icon = {"pending": "○", "processing": "◉", "completed": "✓", "failed": "✗"}
        icon = status_icon.get(item["status"], "?")
        line = f"  {icon} [{item['id']}] {name} — {item['status']}"
        if item.get("error"):
            line += f" ({item['error'][:60]})"
        print(line)


# ── clear 命令 ──────────────────────────────────────

def cmd_clear(mode: str = "completed", queue_path: str = DEFAULT_QUEUE_PATH):
    """清理队列中的已完成/失败/全部项"""
    queue = load_queue(queue_path)
    before = len(queue["items"])

    if mode == "all":
        queue["items"] = []
    elif mode == "completed":
        queue["items"] = [i for i in queue["items"] if i["status"] != "completed"]
    elif mode == "failed":
        queue["items"] = [i for i in queue["items"] if i["status"] != "failed"]

    queue["status"] = "idle"
    queue["current_index"] = None
    save_queue(queue_path, queue)

    removed = before - len(queue["items"])
    print(f"[Queue] 已清理 {removed} 项 (模式: {mode})，剩余 {len(queue['items'])} 项")


# ── CLI 入口 ─────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Video-to-Canvas 队列处理器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
用法示例:
  # 添加视频到队列
  python queue_processor.py add video1.mp4 video2.mp4 --depth balanced --srt-lang zh

  # 后台启动队列处理
  python queue_processor.py run --daemon

  # 查看队列状态
  python queue_processor.py status

  # 清理已完成的项
  python queue_processor.py clear --completed
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # ── add ──
    add_parser = subparsers.add_parser("add", help="添加视频到队列")
    add_parser.add_argument("videos", nargs="+", help="视频文件路径（支持多个）")
    add_parser.add_argument("--depth", default="balanced",
                            choices=["short_hand", "balanced", "deep_dive"])
    add_parser.add_argument("--density", default="normal",
                            choices=["sparse", "normal", "dense"])
    add_parser.add_argument("--srt-lang", default=None, help="SRT 翻译语言")
    add_parser.add_argument("--backend", default="auto",
                            choices=["auto", "faster-whisper", "gemini"])
    add_parser.add_argument("--no-transcribe", action="store_true")
    add_parser.add_argument("--no-srt", action="store_true")
    add_parser.add_argument("--segment-minutes", type=float, default=15.0)
    add_parser.add_argument("--whisper-model", default="large-v3")

    # ── run ──
    run_parser = subparsers.add_parser("run", help="开始处理队列")
    run_parser.add_argument("--daemon", action="store_true",
                            help="后台运行模式")

    # ── status ──
    subparsers.add_parser("status", help="查看队列状态")

    # ── clear ──
    clear_parser = subparsers.add_parser("clear", help="清理队列")
    clear_group = clear_parser.add_mutually_exclusive_group()
    clear_group.add_argument("--completed", action="store_true",
                             help="清理已完成的项（默认）")
    clear_group.add_argument("--failed", action="store_true",
                             help="清理失败的项")
    clear_group.add_argument("--all", action="store_true",
                             help="清空整个队列")

    # 全局参数
    parser.add_argument("--queue-file", default=DEFAULT_QUEUE_PATH,
                        help=f"队列文件路径 (默认: {DEFAULT_QUEUE_PATH})")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    queue_path = args.queue_file

    if args.command == "add":
        common_args = {
            "depth": args.depth,
            "density": args.density,
            "srt_lang": args.srt_lang,
            "backend": args.backend,
            "transcribe_audio": not args.no_transcribe,
            "generate_srt": not args.no_srt,
            "segment_minutes": args.segment_minutes,
            "whisper_model": args.whisper_model,
        }
        cmd_add(args.videos, common_args, queue_path)

    elif args.command == "run":
        if args.daemon:
            import subprocess as _sp
            child_args = [a for a in sys.argv if a != "--daemon"]
            child_cmd = [sys.executable] + child_args

            log_path = os.path.join(SKILL_DIR, "queue.log")
            kwargs = {}
            if sys.platform == "win32":
                kwargs["creationflags"] = _sp.CREATE_NO_WINDOW | _sp.DETACHED_PROCESS
            else:
                kwargs["start_new_session"] = True

            with open(log_path, "w", encoding="utf-8") as log_file:
                proc = _sp.Popen(child_cmd, stdout=log_file, stderr=_sp.STDOUT,
                                 close_fds=(sys.platform != "win32"), **kwargs)

            print(f"[Queue] 后台处理已启动 (PID: {proc.pid})")
            print(f"  日志: {log_path}")
            print(f"  队列: {queue_path}")
            sys.exit(0)

        cmd_run(queue_path)

    elif args.command == "status":
        cmd_status(queue_path)

    elif args.command == "clear":
        if args.all:
            cmd_clear("all", queue_path)
        elif args.failed:
            cmd_clear("failed", queue_path)
        else:
            cmd_clear("completed", queue_path)


if __name__ == "__main__":
    main()
