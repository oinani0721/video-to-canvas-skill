"""
笔记风格定义 - 借鉴 BiliNote 的 9 种风格

每种风格适用于不同的视频类型和学习目的。
"""

# 9 种笔记风格（来自 BiliNote prompt_builder.py）
STYLES = {
    "minimal": {
        "name": "极简模式",
        "description": "仅提取核心要点，每个知识点一句话",
        "prompt": """
## 风格要求：极简模式
- 每个知识点用一句话概括
- 去除所有冗余信息
- 只保留最核心的概念和操作
- 适合快速复习和要点回顾
"""
    },
    "detailed": {
        "name": "详细模式",
        "description": "完整记录所有内容，包含示例和解释",
        "prompt": """
## 风格要求：详细模式
- 完整记录视频中的所有内容
- 包含具体示例和操作步骤
- 保留演讲者的解释和背景知识
- 适合深入学习和完整复习
"""
    },
    "academic": {
        "name": "学术模式",
        "description": "学术风格，使用专业术语，引用格式",
        "prompt": """
## 风格要求：学术模式
- 使用专业术语和学术表达
- 逻辑清晰，层次分明
- 包含定义、原理、证明
- 适合学术课程和专业学习
"""
    },
    "tutorial": {
        "name": "教程模式",
        "description": "教程风格，步骤化，每步配图",
        "prompt": """
## 风格要求：教程模式
- 步骤化呈现，每步清晰编号
- 强调操作流程和顺序
- 突出注意事项和常见错误
- 适合软件教程和操作演示
"""
    },
    "business": {
        "name": "商务模式",
        "description": "商务风格，简洁专业，重点突出",
        "prompt": """
## 风格要求：商务模式
- 简洁专业，无冗余
- 重点突出，层次清晰
- 包含关键数据和结论
- 适合商业演示和报告
"""
    },
    "outline": {
        "name": "大纲模式",
        "description": "大纲模式，层级清晰，便于复习",
        "prompt": """
## 风格要求：大纲模式
- 使用多级列表结构
- 层级关系清晰
- 便于快速定位和复习
- 适合课程大纲和知识体系梳理
"""
    },
    "qa": {
        "name": "问答模式",
        "description": "问答模式，将内容转化为问答对",
        "prompt": """
## 风格要求：问答模式
- 将知识点转化为问答形式
- 问题具体明确
- 答案简洁准确
- 适合考试复习和知识测试
"""
    },
    "summary": {
        "name": "摘要模式",
        "description": "摘要模式，每段一个总结",
        "prompt": """
## 风格要求：摘要模式
- 每个章节配一段简短摘要
- 突出核心观点和结论
- 忽略细节，保留主旨
- 适合快速了解视频内容
"""
    },
    "annotated": {
        "name": "批注模式",
        "description": "批注模式，添加个人理解和延伸",
        "prompt": """
## 风格要求：批注模式
- 在原始内容基础上添加批注
- 使用 `> 批注：` 标记批注内容
- 包含延伸思考和关联知识
- 适合深度学习和知识扩展
"""
    }
}

# 视频类型预设
VIDEO_PRESETS = {
    "lecture": {
        "name": "讲座/演示",
        "fps": 1,
        "resolution": "LOW",
        "style": "academic",
        "min_interval": 10,
        "description": "适合讲座、演示、PPT 类视频，画面变化较慢"
    },
    "tutorial": {
        "name": "编程教程",
        "fps": 5,
        "resolution": "HIGH",
        "style": "tutorial",
        "min_interval": 3,
        "description": "适合代码演示、软件教程，需要捕捉代码细节"
    },
    "fastcut": {
        "name": "快速剪辑",
        "fps": 10,
        "resolution": "MEDIUM",
        "style": "summary",
        "min_interval": 1,
        "description": "适合抖音、TikTok 等快速剪辑视频"
    },
    "demo": {
        "name": "产品演示",
        "fps": 3,
        "resolution": "MEDIUM",
        "style": "business",
        "min_interval": 5,
        "description": "适合产品介绍、功能演示视频"
    }
}


def get_style_prompt(style_name: str) -> str:
    """获取指定风格的提示词片段"""
    style = STYLES.get(style_name, STYLES["detailed"])
    return style["prompt"]


def get_preset(preset_name: str) -> dict:
    """获取指定视频类型的预设配置"""
    return VIDEO_PRESETS.get(preset_name, VIDEO_PRESETS["tutorial"])


def list_styles() -> None:
    """打印所有可用风格"""
    print("\n可用笔记风格：")
    print("-" * 50)
    for key, style in STYLES.items():
        print(f"  {key:12} - {style['name']}: {style['description']}")


def list_presets() -> None:
    """打印所有可用预设"""
    print("\n可用视频预设：")
    print("-" * 50)
    for key, preset in VIDEO_PRESETS.items():
        print(f"  {key:10} - {preset['name']}: {preset['description']}")
