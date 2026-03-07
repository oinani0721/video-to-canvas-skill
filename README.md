# Video to Canvas Skill

> 将视频转换为 Obsidian Canvas 可视化笔记的 Claude Code Skill

[![Claude Code](https://img.shields.io/badge/Claude%20Code-Skill-blue)](https://claude.ai/code)
[![Canvas Learning System](https://img.shields.io/badge/Canvas%20Learning%20System-Compatible-green)](https://github.com/oinani0721/canvas-learning-system)

## 概述

`/video-to-canvas` 是一个端到端的 Claude Code Skill，将视频内容自动转换为结构化的 Obsidian Canvas 可视化笔记。

```
📹 视频 → 🔍 Gemini 分析 → 📸 截图提取 → 📝 MD 笔记 → 📊 Canvas 可视化
```

## 与 Canvas Learning System 的配合

### 🎯 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    Canvas Learning System                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐    ┌──────────────────┐                   │
│  │ video-to-canvas  │───▶│ Obsidian Canvas  │                   │
│  │     Skill        │    │   (可视化笔记)    │                   │
│  └──────────────────┘    └────────┬─────────┘                   │
│           │                       │                              │
│           ▼                       ▼                              │
│  ┌──────────────────┐    ┌──────────────────┐                   │
│  │   Markdown 笔记   │    │   知识图谱视图    │                   │
│  │  (结构化内容)     │    │  (可交互学习)    │                   │
│  └──────────────────┘    └──────────────────┘                   │
│           │                       │                              │
│           └───────────┬───────────┘                              │
│                       ▼                                          │
│              ┌──────────────────┐                                │
│              │  学习追踪与复习   │                                │
│              │  (Obsidian 插件) │                                │
│              └──────────────────┘                                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 🔗 配合关系

| 组件 | 功能 | 输入 | 输出 |
|------|------|------|------|
| **video-to-canvas** | 视频内容提取与可视化 | 视频文件 | .md + .canvas + screenshots |
| **Canvas Learning** | 知识组织与学习追踪 | .canvas 文件 | 学习进度、复习计划 |
| **Obsidian Vault** | 知识存储与管理 | 所有输出 | 个人知识库 |

### 🌟 核心价值

1. **视频 → 可视化知识图谱**
   - 不是简单的视频笔记，而是**知识结构的可视化表达**
   - Canvas 中的节点和边反映知识点之间的逻辑关系

2. **被动学习 → 主动探索**
   - Canvas 支持拖拽、缩放、连接
   - 学习者可以在 Canvas 中重新组织知识结构

3. **与 Obsidian 生态无缝集成**
   - 输出标准 Markdown + JSON Canvas 格式
   - 可与 Obsidian 的双向链接、Graph View 配合使用

## 安装

### 方式一：直接复制到 Claude Code skills 目录

```bash
git clone https://github.com/oinani0721/video-to-canvas-skill.git
cp -r video-to-canvas-skill ~/.claude/skills/video-to-canvas
```

### 方式二：符号链接

```bash
git clone https://github.com/oinani0721/video-to-canvas-skill.git ~/projects/video-to-canvas-skill
ln -s ~/projects/video-to-canvas-skill ~/.claude/skills/video-to-canvas
```

## 依赖

### 系统依赖

| 依赖 | 版本要求 | 安装方式 | 用途 |
|------|---------|---------|------|
| **Claude Code** | 最新版 | `npm install -g @anthropic-ai/claude-code` | Skill 运行环境 |
| **Python** | 3.8+ (推荐 3.10+) | 见下方 | 运行三阶段管道脚本 |
| **FFmpeg** | 5.0+ | 见下方 | 提取视频截图 + 音频 |
| **uv** (推荐) | 0.10+ | `pip install uv` 或 `curl -LsSf https://astral.sh/uv/install.sh \| sh` | Python 包管理（比 pip 快 100x） |

### Python 依赖

| 包名 | 用途 | 必需? |
|------|------|-------|
| `google-genai` | Gemini API 客户端（视频分析、转录、笔记生成） | 是 |
| `faster-whisper` | 本地音频转录（GPU 加速，更精准的时间戳） | 可选（回退到 Gemini 云端转录） |
| `torch` (CUDA) | PyTorch GPU 加速（配合 faster-whisper） | 可选（无 GPU 则 CPU 模式） |

### 安装步骤

#### 1. 安装 FFmpeg

```bash
# Windows (winget)
winget install Gyan.FFmpeg

# macOS (Homebrew)
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg
```

#### 2. 安装 Python + 创建虚拟环境

```bash
# 推荐使用 uv 管理 Python 和依赖
uv python install 3.10
cd ~/.claude/skills/video-to-canvas/scripts
uv venv .venv --python 3.10
```

#### 3. 安装 Python 依赖

```bash
cd ~/.claude/skills/video-to-canvas/scripts

# 必需
uv pip install google-genai --python .venv/Scripts/python.exe   # Windows
uv pip install google-genai --python .venv/bin/python            # macOS/Linux

# 可选：本地转录（推荐有 NVIDIA GPU 的用户）
uv pip install faster-whisper
uv pip install torch --index-url https://download.pytorch.org/whl/cu124  # CUDA 加速
```

#### 4. 配置 API Key

```bash
# 方式一：环境变量（推荐）
export GEMINI_API_KEY="your-gemini-api-key"

# 方式二：.env 文件
echo 'GEMINI_API_KEY=your-gemini-api-key' > ~/.claude/skills/video-to-canvas/.env
```

获取 Gemini API Key: https://aistudio.google.com/apikey

#### 5. Windows 注意事项

- SKILL.md 已配置使用 `.venv/Scripts/python.exe` 和 `PYTHONUTF8=1`，无需额外设置
- 如果 `python` 命令打开 Microsoft Store，在 **Settings > Apps > App execution aliases** 中关闭 python.exe 和 python3.exe 的开关

### 验证安装

```bash
cd ~/.claude/skills/video-to-canvas/scripts

# 检查 Python
.venv/Scripts/python.exe --version        # Windows
.venv/bin/python --version                 # macOS/Linux

# 检查核心依赖
.venv/Scripts/python.exe -c "from google import genai; print('OK:', genai.__version__)"

# 检查 FFmpeg
ffprobe -version

# 检查 GPU（可选）
.venv/Scripts/python.exe -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

## 使用

### 基础用法

```bash
# 完整流程：视频 → Canvas
/video-to-canvas /path/to/video.mp4

# 指定输出目录
/video-to-canvas /path/to/video.mp4 -o /output/dir

# 指定笔记深度
/video-to-canvas /path/to/video.mp4 --depth deep_dive
```

### 深度选项

| 选项 | 说明 |
|------|------|
| `short_hand` | 极简模式，要点列表 |
| `balanced` | 平衡模式（默认），段落+列表 |
| `deep_dive` | 深度模式，详尽解释 |

### 仅生成 Canvas（已有 MD）

```bash
/video-to-canvas --canvas-only existing-notes.md
```

## 工作流程

### Phase 1: Gemini 视觉分析

使用 Gemini 2.5 Flash 分析视频，检测画面变化点：
- 幻灯片切换
- 代码变化
- 图表出现
- UI 状态变化

### Phase 2: FFmpeg 截图提取

根据变化点时间戳，使用 FFmpeg 提取关键帧截图。

### Phase 3: 笔记生成（V2 Lore Engine 架构）

使用 V2 架构生成结构化笔记：
- **按知识结构组织**（不是时间流水账）
- **智能推理补全**（填补视频中不完整的解释）
- **分层组织**（主题 → 子主题 → 知识点）

### Phase 4: Canvas 智能生成

Claude 理解笔记内容后，生成 Canvas：
- **语义分组**：相关知识点放入同一 Group
- **智能布局**：并列关系水平排列，层级关系垂直排列
- **关系标注**：用 Edge 和标签表达知识点关系

## 输出格式

### 目录结构

```
output/
├── video-name.md          # Markdown 笔记
├── video-name.canvas      # Obsidian Canvas 文件
├── video-name_changes.json # 变化点信息（调试用）
└── screenshots/           # 截图目录
    ├── 00-30.jpg
    ├── 00-36.jpg
    └── ...
```

### Canvas 结构

```json
{
  "nodes": [
    { "type": "group", "label": "章节标题", ... },
    { "type": "text", "text": "知识点内容", ... },
    { "type": "file", "file": "screenshots/00-36.jpg", ... }
  ],
  "edges": [
    { "fromNode": "...", "toNode": "...", "label": "关系" }
  ]
}
```

## 文件结构

```
video-to-canvas-skill/
├── SKILL.md                    # 主 Skill 定义
├── README.md                   # 本文档
├── references/
│   ├── json-canvas-spec.md    # JSON Canvas 格式规范
│   └── v2-architecture.md     # V2 Lore Engine 架构说明
├── config/
│   └── default-config.json    # 默认配置
├── scripts/
│   ├── video_to_md.py         # Phase 1-3 执行脚本
│   ├── prompt_builder_v2.py   # V2 提示词构建器
│   ├── prompt_builder.py      # V1 提示词构建器
│   ├── styles.py              # 笔记风格定义
│   └── fusion_detector.py     # 双通道融合检测器
└── sub-skills/                 # 子 Skill（可选）
```

## 与其他工具的对比

| 特性 | BiliNote | Lore Engine | video-to-canvas |
|------|---------|-------------|-----------------|
| 视频分析 | 文本推断 | 不支持 | Gemini 视觉检测 |
| 笔记组织 | 时间流水账 | 知识结构 | 知识结构 |
| 可视化 | ❌ | ❌ | ✅ Canvas |
| AI 布局 | ❌ | ❌ | ✅ Claude |

## 示例

### 输入视频

Obsidian 教程视频（21分钟）

### 输出效果

```
Canvas 可视化：
┌─────────────────────────────────────────────────────────┐
│  一、核心优势                                            │
│  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐        │
│  │双向链接│──│Zettel  │  │本地存储│  │高自定义│        │
│  │[截图]  │  │[截图]  │  │[截图]  │  │[截图]  │        │
│  └────────┘  └────────┘  └────────┘  └────────┘        │
└─────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│  二、优缺点                                              │
│  ┌────────────┐          ┌────────────┐                │
│  │ 优点       │          │ 缺点       │                │
│  │ • 高效率   │          │ • 易折腾   │                │
│  │ • 全功能   │          │ • 同步难   │                │
│  └────────────┘          └────────────┘                │
└─────────────────────────────────────────────────────────┘
```

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

MIT License

## 相关项目

- [Obsidian](https://obsidian.md/) - 知识管理工具
- [JSON Canvas](https://jsoncanvas.org/) - 开放的画布格式规范
- [Claude Code](https://claude.ai/code) - Claude AI 编程助手
- [kepano/obsidian-skills](https://github.com/kepano/obsidian-skills) - Obsidian 官方 Agent Skills
