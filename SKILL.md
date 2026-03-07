---
name: video-to-canvas
description: |
  将视频转换为 Obsidian Canvas 可视化笔记。

  三阶段混合管道 (社区验证最佳实践)：
  1. Stage 1 (Ears): WhisperX/Gemini 音频转录
  2. Stage 2 (Eyes): Gemini 视觉检测 + FFmpeg 截图
  3. Stage 3 (Brain): LLM 双通道融合生成笔记
  4. Phase 4: Claude 智能布局生成 Canvas

  用法：/video-to-canvas <视频路径> [选项]

user-invocable: true

allowed-tools:
  - Read
  - Write
  - Edit
  - Glob
  - Grep
  - Bash
  - AskUserQuestion
  - Task
  - mcp__sequential-thinking__sequentialthinking

skills:
  - obsidian-canvas

hooks:
  PreToolUse:
    - matcher: "Bash"
      hooks:
        - type: command
          command: "echo '[video-to-canvas] 执行系统命令...'"
---

# Video to Canvas

将视频转换为 Obsidian Canvas 可视化笔记（三阶段混合管道）。

## 快速使用

```bash
/video-to-canvas <视频路径>
/video-to-canvas video.mp4 --depth=deep_dive
/video-to-canvas video.mp4 --no-transcribe     # 跳过转录（旧模式）
/video-to-canvas video.mp4 --layout=mindmap
```

---

## 完整工作流（三阶段混合管道）

```
📹 视频文件
     │
     ├──────────────────────┐
     ▼                      ▼
┌──────────────────┐  ┌──────────────────┐
│ Stage 1 (Ears)   │  │ Stage 2 (Eyes)   │
│ WhisperX/Gemini  │  │ Gemini 视觉检测  │
│ 音频转录         │  │ + FFmpeg 截图    │
│ → 时间戳文本     │  │ → screenshots/   │
└──────────────────┘  └──────────────────┘
     │                      │
     └──────────┬───────────┘
                ▼
┌────────────────────────────┐
│ Stage 3 (Brain)            │
│ Gemini 2.5 Flash           │
│ 转录文本 + 截图 → 笔记    │
│ 15 分钟分段避免幻觉        │
│ → 结构化 Markdown          │
└────────────────────────────┘
                │
                ▼
┌────────────────────────────┐
│ Phase 4: Claude            │
│ 语义理解 + 智能布局        │
│ → .canvas 文件             │
└────────────────────────────┘
                │
                ▼
📊 Obsidian Canvas 可视化笔记
```

### 为什么需要三阶段？

| 问题 | 旧架构 | 三阶段管道 |
|------|--------|-----------|
| 音频内容丢失 | Phase 2 只发送截图，丢失口述内容 | Stage 1 转录音频，Stage 3 融合 |
| 长视频幻觉 | Gemini >20min 严重幻觉 | 15 分钟分段处理 |
| 信息不完整 | 只有屏幕变化，无讲解内容 | 双通道：视觉+音频 |
| 截图覆盖缺口 | Gemini 只分析前半部分 | Stage 2.5 自动补充 + ffprobe 时长校正 |
| 幻觉图片引用 | LLM 编造不存在的截图路径 | 提示词约束 + 后处理验证 |

### 防护机制（自动生效）

管道内置以下防护，无需手动干预：

| 机制 | 位置 | 解决的问题 |
|------|------|-----------|
| **ffprobe 时长优先** | Stage 2.5, Stage 3 | Gemini Audio 转录时长可能幻觉（如 56min 视频报告 78min） |
| **Stage 2.5 覆盖率检查** | Stage 2 之后 | Gemini 视觉检测只分析前半部分，后半段无截图 |
| **自动补充截图** | Stage 2.5 | 未覆盖区域每 30 秒自动截图填充 |
| **分段时长截断** | Stage 3 分段 | 超出实际视频时长的转录 chunks 被丢弃 |
| **提示词截图约束** | Stage 3 Prompt | 明确告知 LLM 只能引用列表中的截图文件 |
| **后处理图片验证** | 保存 MD 前 | 扫描所有 `![](screenshots/...)` 引用，移除指向不存在文件的 |
| **反引号修复** | 保存 MD 前 | 修复 Gemini 输出的 `` `![desc](path)` `` 格式问题 |

---

## Stage 1-3: 三阶段管道 (Python 脚本)

### 执行步骤

1. 运行 Python 脚本执行三阶段管道：
   ```bash
   cd ~/.claude/skills/video-to-canvas/scripts
   PYTHONUTF8=1 .venv/Scripts/python.exe video_to_md.py "<视频路径>" -o "<输出目录>" --depth balanced
   ```

2. 等待执行完成，获取：
   - `<输出目录>/<视频名>.md` - Markdown 笔记
   - `<输出目录>/screenshots/` - 截图目录
   - `<输出目录>/<视频名>_transcript.json` - 转录结果
   - `<输出目录>/<视频名>_changes.json` - 变化点信息
   - `<输出目录>/<视频名>.srt` - 英文字幕（精确时间戳，默认生成）
   - `<输出目录>/<视频名>.<lang>.srt` - 翻译字幕（使用 `--srt-lang` 时生成）

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--depth` | 笔记深度 | balanced |
| `--density` | 检测密度 (sparse/normal/dense) | normal |
| `--min-interval` | 最小截图间隔（秒）| 2.0 |
| `--fusion` | 启用视觉双通道融合 | false |
| `--backend` | 转录后端 (auto/faster-whisper/gemini) | auto |
| `--whisper-model` | Whisper 模型大小 | large-v3 |
| `--segment-minutes` | 长视频分段时长（分钟）| 15 |
| `--transcript` | 已有转录文件路径 | - |
| `--no-transcribe` | 跳过音频转录（旧模式）| false |
| `--no-srt` | 不生成 SRT 字幕文件 | false |
| `--srt-lang` | SRT 翻译目标语言 (如 zh, ja, ko) | 不翻译 |

### 深度选项

- `short_hand`: 极简模式，要点列表
- `balanced`: 平衡模式（推荐），段落+列表
- `deep_dive`: 深度模式，详尽解释

### 转录后端选择

| 后端 | 优势 | 要求 |
|------|------|------|
| **faster-whisper** (推荐) | 速度快、VAD 去幻觉、本地运行 | `pip install faster-whisper` |
| **gemini** (备选) | 无需本地模型、零配置 | GEMINI_API_KEY |

`--backend auto` 会自动检测：优先使用 faster-whisper，未安装则回退到 Gemini。

---

## Phase 4: Canvas 智能生成

Phase 1-3 由 Python 脚本执行后，Phase 4 由 Claude 完成。

### 输入

读取 Phase 3 生成的 Markdown 笔记文件。

### 分析任务

1. **结构识别**
   ```
   # 标题 → 画布标题（不创建节点）
   ## 一级章节 → group 节点（分组容器）
   ### 二级章节 → text 节点
   #### 三级标题 → text 节点（在父节点内）
   ![描述](path) → file 节点
   ```

2. **语义分析**
   - 并列关系 → 水平排列
   - 层级关系 → 垂直排列
   - 对比关系 → 左右对称
   - 重要内容 → 颜色高亮

3. **布局决策**
   - 教程类 → 流程图布局（从上到下）
   - 概念类 → 思维导图（中心辐射）
   - 对比类 → 表格布局（左右对称）

### 输出格式

生成 JSON Canvas 格式：

```json
{
  "nodes": [
    {
      "id": "group-章节名",
      "type": "group",
      "label": "一、章节标题",
      "x": 0, "y": 0,
      "width": 800, "height": 400,
      "color": "4"
    },
    {
      "id": "text-知识点",
      "type": "text",
      "text": "### 知识点标题\n\n内容描述...",
      "x": 50, "y": 80,
      "width": 350, "height": 150
    },
    {
      "id": "img-00-36",
      "type": "file",
      "file": "<输出目录名>/screenshots/00-36.jpg",
      "x": 50, "y": 250,
      "width": 350, "height": 250
    }
  ],
  "edges": [
    {
      "id": "edge-1",
      "fromNode": "text-知识点",
      "toNode": "img-00-36",
      "fromSide": "bottom",
      "toSide": "top",
      "label": "演示"
    }
  ]
}
```

---

## JSON Canvas 格式规范

### 节点类型

| 类型 | 用途 | 必需字段 |
|------|------|---------|
| `text` | Markdown 文本 | `text` |
| `file` | 图片/文件引用 | `file` |
| `link` | 外部 URL | `url` |
| `group` | 分组容器 | `label` (可选) |

### 通用节点属性

```json
{
  "id": "unique-16-char-hex",
  "type": "text|file|link|group",
  "x": 0,
  "y": 0,
  "width": 300,
  "height": 150,
  "color": "1"  // 可选
}
```

### 颜色系统

| 值 | 颜色 | 建议用途 |
|----|------|---------|
| "1" | 红色 | 重要/警告 |
| "2" | 橙色 | 提示/注意 |
| "3" | 黄色 | 高亮重点 |
| "4" | 绿色 | 正确/完成 |
| "5" | 青色 | 引用/链接 |
| "6" | 紫色 | 概念/定义 |

也支持 hex 格式：`"#FF5733"`

### Edge 属性

```json
{
  "id": "edge-unique-id",
  "fromNode": "source-node-id",
  "toNode": "target-node-id",
  "fromSide": "bottom",  // top|right|bottom|left
  "toSide": "top",
  "toEnd": "arrow",      // arrow|none
  "label": "关系描述"    // 可选
}
```

---

## 布局算法

### 尺寸建议

| 元素 | 宽度 | 高度 |
|------|------|------|
| 标题文本节点 | 300 | 80 |
| 内容文本节点 | 400 | 150 |
| 图片节点 | 350 | 250 |
| 分组最小 | 自适应 | 自适应 |
| 节点水平间距 | 100 | - |
| 节点垂直间距 | 80 | - |

### 层级布局（默认）

```
                    [文档标题]
                        │
    ┌───────────────────┼───────────────────┐
    │                   │                   │
[Group: 章节1]    [Group: 章节2]     [Group: 章节3]
    │                   │                   │
┌───┴───┐         ┌─────┴─────┐        ┌───┴───┐
│       │         │           │        │       │
[知识点] [知识点]  [知识点]   [知识点]  [知识点] [知识点]
   │        │        │           │        │        │
[图片]   [图片]   [图片]      [图片]   [图片]   [图片]
```

### 坐标计算规则

1. **根节点**：从 (0, 0) 开始
2. **Group 内部**：padding 50px
3. **同级节点**：水平排列，间距 100px
4. **子节点**：垂直排列在父节点下方，间距 80px
5. **图片节点**：紧跟对应文本节点下方

### 避免重叠

- 计算每个 group 的实际宽高（基于内部节点）
- Group 之间保持 100px 间距
- 使用深度优先遍历计算坐标

---

## 图片路径处理

### Markdown 中的图片

```markdown
![链接演示](screenshots/00-36.jpg)
```

### 转换为 Canvas file 节点

```json
{
  "id": "img-00-36",
  "type": "file",
  "file": "<输出目录名>/screenshots/00-36.jpg",
  "x": 50,
  "y": 250,
  "width": 350,
  "height": 250
}
```

**⚠️ 重要：Vault 相对路径**

Obsidian Canvas 的 `file` 路径是**相对于 vault 根目录**的，不是相对于 `.canvas` 文件。

例如，输出目录为 `lecture1/`，用户将其复制到 vault 根目录后：
```
vault根目录/
├── lecture1/
│   ├── lecture.canvas
│   ├── lecture.md
│   └── screenshots/
│       └── 05-33.jpg
```

Canvas 中的路径应为：
```json
"file": "lecture1/screenshots/05-33.jpg"   // ✅ 正确 (vault 相对路径)
"file": "screenshots/05-33.jpg"            // ❌ 错误 (canvas 相对路径)
```

**Phase 4 生成规则**：`file` 字段必须使用 `<输出目录名>/screenshots/xx-xx.jpg` 格式。

---

## 执行流程

当用户调用 `/video-to-canvas <视频路径>` 时：

### Step 1: 参数解析

```
视频路径: <用户提供>
输出目录: 与视频同目录，或用户指定
深度: balanced (默认)
布局: hierarchical (默认)
```

### Step 2: 执行三阶段管道

```bash
cd ~/.claude/skills/video-to-canvas/scripts
PYTHONUTF8=1 .venv/Scripts/python.exe video_to_md.py "<视频路径>" -o "<输出目录>" --depth <深度>
```

等待脚本完成，检查输出：
- `<输出目录>/<视频名>.md` 存在
- `<输出目录>/screenshots/` 目录有图片
- `<输出目录>/<视频名>_transcript.json` 存在（如启用转录）

### Step 3: 读取 Markdown

```
使用 Read 工具读取生成的 .md 文件
```

### Step 4: 分析并生成 Canvas

1. 解析 Markdown 结构（标题、内容、图片）
2. 确定语义关系和布局策略
3. 计算每个节点的坐标
4. 生成 JSON Canvas
5. **file 节点路径**: 使用 `<输出目录名>/screenshots/xx-xx.jpg` (vault 相对路径)
6. **text 节点内容**: 确保 JSON 字符串中的双引号已转义 (`\"`) 或替换为全角引号

### Step 5: 写入 Canvas 文件

```
使用 Write 工具写入 <视频名>.canvas
```

### Step 6: 报告结果

```
✅ 三阶段混合管道完成！

输出文件：
├── <视频名>.md                 # Markdown 笔记（双通道融合）
├── <视频名>.canvas             # Canvas 可视化
├── <视频名>.srt                # 英文字幕（精确时间戳，默认生成）
├── <视频名>.<lang>.srt         # 翻译字幕（--srt-lang 时生成）
├── <视频名>_transcript.json    # 音频转录结果
├── <视频名>_changes.json       # 变化点信息
└── screenshots/                # 截图目录

提示：将整个输出目录复制到 Obsidian vault 中，
然后打开 .canvas 文件查看可视化笔记。
SRT 字幕文件会被 Media Extended 插件自动检测（同名同目录）。
```

---

## 错误处理

### Python 脚本执行失败

1. 检查 GEMINI_API_KEY 环境变量
2. 检查视频文件路径是否正确
3. 检查 FFmpeg 是否安装

### 转录失败

1. `--backend auto`: faster-whisper 未安装时自动回退到 Gemini
2. 安装 faster-whisper: `pip install faster-whisper`
3. 或强制使用 Gemini: `--backend gemini`
4. 使用已有转录: `--transcript existing_transcript.json`
5. 跳过转录: `--no-transcribe`

### 图片不显示

1. 确保 `screenshots/` 目录与 `.canvas` 文件在同一目录
2. 检查图片路径是否正确（相对路径）

### Canvas 节点重叠

1. 增加 `nodeSpacing` 参数
2. 使用更简化的布局

---

## 高级选项

### 指定布局

```bash
/video-to-canvas video.mp4 --layout=mindmap
```

布局选项：
- `hierarchical`: 层级布局（默认）
- `mindmap`: 思维导图
- `flowchart`: 流程图

### 指定深度

```bash
/video-to-canvas video.mp4 --depth=deep_dive
```

### 只生成 Canvas（已有 MD）

```bash
/video-to-canvas --canvas-only existing-notes.md
```

---

## 示例输出

### 输入 Markdown 片段

```markdown
# Obsidian 核心功能

## 一、双向链接

双向链接让笔记形成网状结构，而非传统的树状层级。

### 什么是双向链接？

当笔记 A 链接到笔记 B 时，B 会自动显示反向链接。

![链接演示](screenshots/00-36.jpg)

### 如何创建？

1. 输入 `[[` 触发建议
2. 选择目标笔记
3. 按 Enter 确认

![创建链接](screenshots/00-48.jpg)

## 二、本地存储

所有文件以 Markdown 格式存储在本地。

![本地文件](screenshots/01-16.jpg)
```

### 输出 Canvas

```json
{
  "nodes": [
    {
      "id": "group-双向链接",
      "type": "group",
      "label": "一、双向链接",
      "x": 0,
      "y": 0,
      "width": 850,
      "height": 700,
      "color": "4"
    },
    {
      "id": "text-intro",
      "type": "text",
      "text": "双向链接让笔记形成网状结构，而非传统的树状层级。",
      "x": 50,
      "y": 50,
      "width": 750,
      "height": 60
    },
    {
      "id": "text-what",
      "type": "text",
      "text": "### 什么是双向链接？\n\n当笔记 A 链接到笔记 B 时，B 会自动显示反向链接。",
      "x": 50,
      "y": 130,
      "width": 350,
      "height": 120
    },
    {
      "id": "img-00-36",
      "type": "file",
      "file": "<输出目录名>/screenshots/00-36.jpg",
      "x": 50,
      "y": 270,
      "width": 350,
      "height": 220
    },
    {
      "id": "text-how",
      "type": "text",
      "text": "### 如何创建？\n\n1. 输入 `[[` 触发建议\n2. 选择目标笔记\n3. 按 Enter 确认",
      "x": 450,
      "y": 130,
      "width": 350,
      "height": 120
    },
    {
      "id": "img-00-48",
      "type": "file",
      "file": "<输出目录名>/screenshots/00-48.jpg",
      "x": 450,
      "y": 270,
      "width": 350,
      "height": 220
    },
    {
      "id": "group-本地存储",
      "type": "group",
      "label": "二、本地存储",
      "x": 950,
      "y": 0,
      "width": 500,
      "height": 450,
      "color": "5"
    },
    {
      "id": "text-local",
      "type": "text",
      "text": "所有文件以 Markdown 格式存储在本地。",
      "x": 1000,
      "y": 50,
      "width": 400,
      "height": 60
    },
    {
      "id": "img-01-16",
      "type": "file",
      "file": "<输出目录名>/screenshots/01-16.jpg",
      "x": 1000,
      "y": 130,
      "width": 400,
      "height": 280
    }
  ],
  "edges": [
    {
      "id": "edge-1",
      "fromNode": "text-what",
      "toNode": "img-00-36",
      "fromSide": "bottom",
      "toSide": "top"
    },
    {
      "id": "edge-2",
      "fromNode": "text-how",
      "toNode": "img-00-48",
      "fromSide": "bottom",
      "toSide": "top"
    },
    {
      "id": "edge-3",
      "fromNode": "text-local",
      "toNode": "img-01-16",
      "fromSide": "bottom",
      "toSide": "top"
    }
  ]
}
```

---

## 验证清单

生成 Canvas 后检查：

- [ ] 所有节点 ID 唯一
- [ ] Edge 引用的节点存在
- [ ] 图片路径有效（相对路径）
- [ ] 无节点重叠
- [ ] JSON 格式正确
- [ ] Group 正确包含子节点
