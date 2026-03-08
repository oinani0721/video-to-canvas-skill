---
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
---

# Video to Canvas

将视频转换为 Obsidian Canvas 可视化笔记（三阶段混合管道）。

**用法**:
- 单视频: `/video-to-canvas <视频路径> [选项]`
- 多视频队列: `/video-to-canvas <视频1> <视频2> <视频3> [选项]`

请先读取完整的 Skill 定义文件，然后按照指示执行：

```
Read ~/.claude/skills/video-to-canvas/SKILL.md
```

读取 SKILL.md 后，判断参数中包含的视频数量：
- **单个视频** → 按照"执行流程"（Step 1-6）处理
- **多个视频** → 按照"队列模式"（Step Q1-Q7）处理

用户提供的参数: $ARGUMENTS
