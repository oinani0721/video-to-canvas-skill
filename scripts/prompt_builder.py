"""
提示词构建器 - 借鉴 BiliNote 的 Builder 模式

支持：
- 9 种笔记风格
- 截图集成
- AI 摘要
- 自定义扩展
"""

from styles import STYLES, get_style_prompt


# ========== 阶段1：变化检测提示词 ==========
CHANGE_DETECTION_PROMPT = """
分析这个视频，识别所有有意义的画面变化点。

"有意义的变化"定义：
1. 幻灯片/内容切换 - 新幻灯片出现，内容显著变化
2. 图表/示意图变化 - 新图表、流程图、可视化出现
3. 代码变化 - 新代码块，代码修改
4. 白板更新 - 新的书写、绘图
5. 演示状态变化 - 软件演示展示新功能/状态
6. 动画步骤 - 动画解释的每个关键帧
7. 文字叠加 - 重要文字、注释、高亮出现

忽略：
- 无内容变化的镜头切换
- 轻微的 UI 变化（鼠标移动、滚动条）
- 视频压缩噪点

输出要求：
- 时间戳格式：MM:SS
- 按时间升序排列
- 每个变化点包含：时间戳、变化类型、简短描述
"""

# JSON Schema 强制输出格式
CHANGE_DETECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "change_points": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "timestamp": {
                        "type": "string",
                        "description": "时间戳，格式 MM:SS"
                    },
                    "change_type": {
                        "type": "string",
                        "enum": [
                            "slide_change",      # 幻灯片切换
                            "diagram",           # 图表/示意图
                            "code_change",       # 代码变化
                            "whiteboard",        # 白板更新
                            "demo_state",        # 演示状态
                            "animation",         # 动画关键帧
                            "text_overlay",      # 文字叠加
                            "ui_change",         # 界面变化
                            "other"              # 其他
                        ],
                        "description": "变化类型"
                    },
                    "description": {
                        "type": "string",
                        "description": "变化内容的简短描述（10-30字）"
                    }
                },
                "required": ["timestamp", "change_type", "description"]
            }
        },
        "video_summary": {
            "type": "string",
            "description": "视频内容的一句话摘要"
        },
        "total_duration": {
            "type": "string",
            "description": "视频总时长，格式 MM:SS"
        }
    },
    "required": ["change_points", "video_summary"]
}


# ========== 阶段2：笔记生成基础提示词 ==========
BASE_PROMPT = """
你是专业的视频笔记整理专家。

## 任务
根据提供的视频截图和变化点信息，生成结构化的 Markdown 笔记。

## 输出要求
- 使用 Markdown 格式
- 中文输出，技术术语保留英文
- 每个截图对应一个知识点章节

## 截图信息
以下是视频中的关键画面变化点：
{screenshot_list}

## 笔记结构
每个章节包含：
1. 标题（基于画面内容）
2. 时间戳标记：`*时间戳: [mm:ss]`
3. 截图引用：`![描述](screenshots/mm-ss.jpg)`
4. 详细内容描述
5. 要点列表（如果适用）
"""


class PromptBuilder:
    """
    BiliNote 风格的提示词构建器

    使用方法：
        prompt = (PromptBuilder()
            .with_style("tutorial")
            .with_screenshots(screenshots)
            .with_ai_summary()
            .build())
    """

    def __init__(self):
        self.prompt = BASE_PROMPT
        self._style = None
        self._screenshots = []
        self._ai_summary = False
        self._custom_sections = []

    def with_style(self, style: str) -> "PromptBuilder":
        """添加笔记风格"""
        self._style = style
        style_prompt = get_style_prompt(style)
        self.prompt += f"\n{style_prompt}"
        return self

    def with_screenshots(self, screenshots: list) -> "PromptBuilder":
        """
        设置可用截图列表

        Args:
            screenshots: 截图信息列表，每项包含：
                - timestamp: 时间戳 (MM:SS)
                - path: 截图文件路径
                - desc: 描述
                - type: 变化类型
        """
        self._screenshots = screenshots

        # 生成截图列表文本
        screenshot_list = []
        for i, s in enumerate(screenshots, 1):
            safe_ts = s["timestamp"].replace(":", "-")
            screenshot_list.append(
                f"{i}. [{s['timestamp']}] {s['type']}: {s['desc']}\n"
                f"   截图路径: screenshots/{safe_ts}.jpg"
            )

        self.prompt = self.prompt.replace(
            "{screenshot_list}",
            "\n".join(screenshot_list) if screenshot_list else "（无截图信息）"
        )
        return self

    def with_ai_summary(self) -> "PromptBuilder":
        """添加 AI 摘要要求"""
        self._ai_summary = True
        self.prompt += """

## AI 摘要
在笔记末尾添加一段 AI 生成的专业摘要：
- 不超过 100 字
- 概括视频的核心内容和价值
- 使用标记：`> 📝 AI 摘要：...`
"""
        return self

    def with_table_of_contents(self) -> "PromptBuilder":
        """添加目录生成要求"""
        self.prompt += """

## 目录
在笔记开头生成目录：
- 列出所有章节标题
- 使用 Markdown 链接格式
"""
        return self

    def with_timestamps(self) -> "PromptBuilder":
        """强调时间戳标记"""
        self.prompt += """

## 时间戳要求
- 每个章节标题后添加时间戳
- 格式：`## 章节标题 [MM:SS]`
- 确保时间戳与截图对应
"""
        return self

    def with_custom(self, section_name: str, content: str) -> "PromptBuilder":
        """添加自定义提示词片段"""
        self._custom_sections.append((section_name, content))
        self.prompt += f"\n\n## {section_name}\n{content}"
        return self

    def build(self) -> str:
        """构建最终提示词"""
        # 如果没有设置截图，使用占位符提示
        if "{screenshot_list}" in self.prompt:
            self.prompt = self.prompt.replace(
                "{screenshot_list}",
                "（请在调用时提供截图信息）"
            )
        return self.prompt

    def __str__(self) -> str:
        """返回当前构建状态"""
        parts = [f"PromptBuilder(style={self._style}"]
        if self._screenshots:
            parts.append(f"screenshots={len(self._screenshots)}")
        if self._ai_summary:
            parts.append("ai_summary=True")
        if self._custom_sections:
            parts.append(f"custom_sections={len(self._custom_sections)}")
        return ", ".join(parts) + ")"


# ========== 预设提示词工厂 ==========
def create_tutorial_prompt(screenshots: list) -> str:
    """创建教程风格提示词"""
    return (PromptBuilder()
            .with_style("tutorial")
            .with_screenshots(screenshots)
            .with_timestamps()
            .with_ai_summary()
            .build())


def create_lecture_prompt(screenshots: list) -> str:
    """创建讲座风格提示词"""
    return (PromptBuilder()
            .with_style("academic")
            .with_screenshots(screenshots)
            .with_table_of_contents()
            .with_ai_summary()
            .build())


def create_quick_prompt(screenshots: list) -> str:
    """创建快速摘要提示词"""
    return (PromptBuilder()
            .with_style("summary")
            .with_screenshots(screenshots)
            .build())


# ========== 测试 ==========
if __name__ == "__main__":
    # 测试 Builder
    test_screenshots = [
        {"timestamp": "00:15", "path": "ss/00-15.jpg", "desc": "标题页", "type": "slide_change"},
        {"timestamp": "01:30", "path": "ss/01-30.jpg", "desc": "核心概念", "type": "slide_change"},
    ]

    prompt = (PromptBuilder()
              .with_style("tutorial")
              .with_screenshots(test_screenshots)
              .with_ai_summary()
              .build())

    print("=" * 60)
    print("生成的提示词：")
    print("=" * 60)
    print(prompt)
