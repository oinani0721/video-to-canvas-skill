"""
双通道变化检测 + 交叉验证融合

架构：
┌─────────────────────────────────────────────────────────┐
│  通道 A: Gemini 视觉检测                                 │
│  • 直接分析视频帧，检测画面变化                          │
│  • 优势：捕捉快速画面切换、无声视频段落                  │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  通道 B: 语音转录 + 语义边界检测                         │
│  • 使用 Gemini 内置转录 或 Whisper                       │
│  • 分析语义边界（话题切换、段落结束）                    │
│  • 优势：捕捉纯口述的概念切换                            │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  融合层                                                  │
│  • 时间窗口匹配（±3秒）                                  │
│  • 置信度评分：视觉+语义同时触发 > 单通道触发            │
│  • 输出：融合后的高置信度变化点                          │
└─────────────────────────────────────────────────────────┘
"""

from dataclasses import dataclass
from typing import List, Optional
import json


@dataclass
class ChangePoint:
    """变化点数据结构"""
    timestamp: str           # MM:SS 格式
    seconds: float           # 秒数（用于计算）
    change_type: str         # 变化类型
    description: str         # 描述
    source: str              # 来源：visual / semantic / fused
    confidence: float = 1.0  # 置信度 0-1


def parse_timestamp(ts: str) -> float:
    """MM:SS 或 HH:MM:SS 转秒数"""
    parts = ts.split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    return 0


def seconds_to_timestamp(secs: float) -> str:
    """秒数转 MM:SS"""
    mins = int(secs // 60)
    secs = int(secs % 60)
    return f"{mins:02d}:{secs:02d}"


# ========== 语义边界检测提示词 ==========
SEMANTIC_BOUNDARY_PROMPT = """
分析这个视频的语音内容，识别语义边界点。

"语义边界"定义：
1. 话题切换 - 从一个主题转到另一个主题
2. 段落结束 - 一个概念讲解完毕，开始新概念
3. 总结/过渡 - "接下来"、"总结一下"、"下面我们来看"
4. 问答转换 - 从陈述变成提问，或反之
5. 示例开始 - "比如"、"举个例子"、"我们来看一个案例"

忽略：
- 语气词、停顿
- 同一概念内的细节展开

输出 JSON 格式：
{
    "semantic_points": [
        {
            "timestamp": "MM:SS",
            "boundary_type": "topic_switch|paragraph_end|transition|qa_switch|example_start",
            "trigger_phrase": "触发识别的关键词句",
            "description": "此处语义边界的简短描述"
        }
    ],
    "transcript_summary": "语音内容的一句话摘要"
}
"""

SEMANTIC_BOUNDARY_SCHEMA = {
    "type": "object",
    "properties": {
        "semantic_points": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "timestamp": {"type": "string"},
                    "boundary_type": {
                        "type": "string",
                        "enum": ["topic_switch", "paragraph_end", "transition", "qa_switch", "example_start"]
                    },
                    "trigger_phrase": {"type": "string"},
                    "description": {"type": "string"}
                },
                "required": ["timestamp", "boundary_type", "description"]
            }
        },
        "transcript_summary": {"type": "string"}
    },
    "required": ["semantic_points"]
}


class FusionDetector:
    """
    双通道变化检测融合器

    使用方法：
        detector = FusionDetector(client, video_file)
        fused_points = detector.detect_and_fuse()
    """

    def __init__(
        self,
        client,
        video_file,
        time_window: float = 3.0,  # 时间窗口（秒）
        visual_weight: float = 0.6,
        semantic_weight: float = 0.4
    ):
        self.client = client
        self.video_file = video_file
        self.time_window = time_window
        self.visual_weight = visual_weight
        self.semantic_weight = semantic_weight

        self.visual_points: List[ChangePoint] = []
        self.semantic_points: List[ChangePoint] = []
        self.fused_points: List[ChangePoint] = []

    def detect_visual_changes(self, prompt: str, schema: dict) -> List[ChangePoint]:
        """通道 A: Gemini 视觉检测"""
        from google.genai import types

        print("  [通道 A] Gemini 视觉检测...")

        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[self.video_file, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema
            )
        )

        result = json.loads(response.text)
        points = []

        for cp in result.get("change_points", []):
            points.append(ChangePoint(
                timestamp=cp["timestamp"],
                seconds=parse_timestamp(cp["timestamp"]),
                change_type=cp["change_type"],
                description=cp["description"],
                source="visual",
                confidence=0.8
            ))

        self.visual_points = points
        print(f"    检测到 {len(points)} 个视觉变化点")
        return points

    def detect_semantic_boundaries(self) -> List[ChangePoint]:
        """通道 B: 语义边界检测"""
        from google.genai import types

        print("  [通道 B] 语义边界检测...")

        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[self.video_file, SEMANTIC_BOUNDARY_PROMPT],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=SEMANTIC_BOUNDARY_SCHEMA
            )
        )

        result = json.loads(response.text)
        points = []

        for sp in result.get("semantic_points", []):
            points.append(ChangePoint(
                timestamp=sp["timestamp"],
                seconds=parse_timestamp(sp["timestamp"]),
                change_type=sp["boundary_type"],
                description=sp["description"],
                source="semantic",
                confidence=0.7
            ))

        self.semantic_points = points
        print(f"    检测到 {len(points)} 个语义边界点")
        return points

    def fuse_points(self) -> List[ChangePoint]:
        """
        融合两个通道的变化点

        融合策略：
        1. 时间窗口内同时触发 → 高置信度 (1.0)
        2. 仅视觉触发 → 中等置信度 (0.6)
        3. 仅语义触发 → 中等置信度 (0.5)
        """
        print("  [融合层] 交叉验证...")

        fused = []
        used_semantic = set()

        # 遍历视觉变化点，寻找匹配的语义边界
        for vp in self.visual_points:
            matched_semantic = None

            for i, sp in enumerate(self.semantic_points):
                if i in used_semantic:
                    continue

                # 时间窗口匹配
                if abs(vp.seconds - sp.seconds) <= self.time_window:
                    matched_semantic = sp
                    used_semantic.add(i)
                    break

            if matched_semantic:
                # 双通道同时触发 → 高置信度
                fused.append(ChangePoint(
                    timestamp=vp.timestamp,  # 使用视觉时间戳（更精确）
                    seconds=vp.seconds,
                    change_type=f"{vp.change_type}+{matched_semantic.change_type}",
                    description=f"[视觉] {vp.description} | [语义] {matched_semantic.description}",
                    source="fused",
                    confidence=1.0
                ))
            else:
                # 仅视觉触发
                vp.confidence = self.visual_weight
                fused.append(vp)

        # 添加未匹配的语义边界点
        for i, sp in enumerate(self.semantic_points):
            if i not in used_semantic:
                sp.confidence = self.semantic_weight
                fused.append(sp)

        # 按时间排序
        fused.sort(key=lambda x: x.seconds)

        self.fused_points = fused
        print(f"    融合后共 {len(fused)} 个变化点")
        print(f"    - 双通道验证: {sum(1 for p in fused if p.source == 'fused')}")
        print(f"    - 仅视觉: {sum(1 for p in fused if p.source == 'visual')}")
        print(f"    - 仅语义: {sum(1 for p in fused if p.source == 'semantic')}")

        return fused

    def detect_and_fuse(
        self,
        visual_prompt: str,
        visual_schema: dict,
        min_confidence: float = 0.0
    ) -> List[ChangePoint]:
        """
        完整的双通道检测 + 融合流程

        Args:
            visual_prompt: 视觉检测提示词
            visual_schema: 视觉检测 JSON Schema
            min_confidence: 最小置信度阈值

        Returns:
            融合后的变化点列表
        """
        print("\n双通道变化检测启动...")

        # 通道 A: 视觉检测
        self.detect_visual_changes(visual_prompt, visual_schema)

        # 通道 B: 语义边界检测
        self.detect_semantic_boundaries()

        # 融合
        self.fuse_points()

        # 过滤低置信度点
        if min_confidence > 0:
            self.fused_points = [
                p for p in self.fused_points
                if p.confidence >= min_confidence
            ]
            print(f"    过滤后保留 {len(self.fused_points)} 个变化点（置信度 >= {min_confidence}）")

        return self.fused_points

    def to_legacy_format(self) -> List[dict]:
        """转换为原有格式，兼容 video_to_md.py"""
        return [
            {
                "timestamp": p.timestamp,
                "change_type": p.change_type,
                "description": p.description,
                "confidence": p.confidence,
                "source": p.source
            }
            for p in self.fused_points
        ]


# ========== 交叉验证分析 ==========
def analyze_detection_quality(
    visual_points: List[ChangePoint],
    semantic_points: List[ChangePoint],
    fused_points: List[ChangePoint]
) -> dict:
    """
    分析检测质量，帮助调试和优化

    返回：
    - 匹配率：双通道同时触发的比例
    - 视觉独占率：仅视觉触发的比例
    - 语义独占率：仅语义触发的比例
    - 建议：基于分析结果的调优建议
    """
    total = len(fused_points)
    if total == 0:
        return {"error": "无变化点"}

    fused_count = sum(1 for p in fused_points if p.source == "fused")
    visual_only = sum(1 for p in fused_points if p.source == "visual")
    semantic_only = sum(1 for p in fused_points if p.source == "semantic")

    analysis = {
        "total_points": total,
        "fused_count": fused_count,
        "fused_ratio": fused_count / total,
        "visual_only_count": visual_only,
        "visual_only_ratio": visual_only / total,
        "semantic_only_count": semantic_only,
        "semantic_only_ratio": semantic_only / total,
        "suggestions": []
    }

    # 生成建议
    if analysis["fused_ratio"] < 0.3:
        analysis["suggestions"].append(
            "匹配率较低，考虑增大 time_window 参数"
        )

    if analysis["visual_only_ratio"] > 0.6:
        analysis["suggestions"].append(
            "视觉变化点占主导，视频可能是快速剪辑类型，语义边界检测效果有限"
        )

    if analysis["semantic_only_ratio"] > 0.6:
        analysis["suggestions"].append(
            "语义边界点占主导，视频可能是讲座/演讲类型，画面变化较少"
        )

    if analysis["fused_ratio"] > 0.7:
        analysis["suggestions"].append(
            "双通道高度一致，检测结果可信度高"
        )

    return analysis


# ========== 测试 ==========
if __name__ == "__main__":
    # 模拟测试数据
    visual = [
        ChangePoint("00:10", 10, "slide_change", "标题页", "visual"),
        ChangePoint("01:25", 85, "code_change", "代码示例", "visual"),
        ChangePoint("02:00", 120, "diagram", "架构图", "visual"),
    ]

    semantic = [
        ChangePoint("00:12", 12, "topic_switch", "开场介绍", "semantic"),
        ChangePoint("01:30", 90, "example_start", "举例说明", "semantic"),
        ChangePoint("03:00", 180, "transition", "总结部分", "semantic"),
    ]

    # 模拟融合
    print("模拟融合测试：")
    print(f"视觉点: {[p.timestamp for p in visual]}")
    print(f"语义点: {[p.timestamp for p in semantic]}")

    # 分析
    fused = visual + semantic  # 简化测试
    analysis = analyze_detection_quality(visual, semantic, fused)
    print(f"\n分析结果: {json.dumps(analysis, ensure_ascii=False, indent=2)}")
