# JSON Canvas 格式规范

> 来源: [jsoncanvas.org/spec/1.0](https://jsoncanvas.org/spec/1.0/) | 版本: 1.0

JSON Canvas 是 Obsidian 使用的开放格式，用于无限画布数据的互操作性。

---

## 文件结构

Canvas 文件使用 `.canvas` 扩展名，包含两个顶级数组：

```json
{
  "nodes": [...],
  "edges": [...]
}
```

- 两个数组都是可选的
- 节点按数组顺序进行 z-ordering（靠前的在下层）

---

## 节点 (Nodes)

### 通用属性

所有节点类型都包含以下属性：

| 属性 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `id` | string | ✅ | 唯一标识符（建议 16 位十六进制）|
| `type` | string | ✅ | 节点类型: text, file, link, group |
| `x` | number | ✅ | X 坐标（像素，可为负数）|
| `y` | number | ✅ | Y 坐标（像素，可为负数）|
| `width` | number | ✅ | 宽度（像素）|
| `height` | number | ✅ | 高度（像素）|
| `color` | string | ❌ | 颜色（预设 "1"-"6" 或 hex）|

### 坐标系统

- 原点 (0, 0) 在左上角
- X 向右增加
- Y 向下增加
- 节点位置为左上角坐标

---

## 节点类型详解

### 1. Text 节点

存储 Markdown 格式的文本内容。

```json
{
  "id": "text-001",
  "type": "text",
  "x": 0,
  "y": 0,
  "width": 300,
  "height": 150,
  "text": "## 标题\n\n这是 **Markdown** 内容。",
  "color": "4"
}
```

| 属性 | 类型 | 说明 |
|------|------|------|
| `text` | string | Markdown 格式文本（必需）|

**⚠️ 换行符注意**：
- ✅ 正确：`"text": "Line 1\nLine 2"`
- ❌ 错误：`"text": "Line 1\\nLine 2"` （会显示为字面量 `\n`）

---

### 2. File 节点

引用文件或附件（图片、PDF、笔记等）。

```json
{
  "id": "file-001",
  "type": "file",
  "x": 400,
  "y": 0,
  "width": 350,
  "height": 250,
  "file": "screenshots/00-36.jpg"
}
```

| 属性 | 类型 | 说明 |
|------|------|------|
| `file` | string | 文件相对路径（必需）|
| `subpath` | string | 文件内锚点，以 `#` 开头（可选）|

**路径说明**：
- 路径相对于 `.canvas` 文件所在目录
- 支持图片、PDF、Markdown 文件等

---

### 3. Link 节点

引用外部 URL。

```json
{
  "id": "link-001",
  "type": "link",
  "x": 800,
  "y": 0,
  "width": 300,
  "height": 100,
  "url": "https://obsidian.md"
}
```

| 属性 | 类型 | 说明 |
|------|------|------|
| `url` | string | 完整 URL（必需）|

---

### 4. Group 节点

视觉分组容器，用于组织其他节点。

```json
{
  "id": "group-001",
  "type": "group",
  "x": 0,
  "y": 0,
  "width": 800,
  "height": 400,
  "label": "一、核心功能",
  "color": "4"
}
```

| 属性 | 类型 | 说明 |
|------|------|------|
| `label` | string | 分组标签（可选）|
| `background` | string | 背景图片路径（可选）|
| `backgroundStyle` | string | 背景样式: cover, ratio, repeat（可选）|

**分组逻辑**：
- Group 不"包含"其他节点（JSON 层面）
- 视觉上，坐标在 group 范围内的节点会显示在其内部
- 建议：内部节点坐标 = group 坐标 + padding

---

## 边 (Edges)

连接两个节点，表示关系。

```json
{
  "id": "edge-001",
  "fromNode": "text-001",
  "toNode": "file-001",
  "fromSide": "bottom",
  "toSide": "top",
  "toEnd": "arrow",
  "label": "演示",
  "color": "4"
}
```

### Edge 属性

| 属性 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `id` | string | ✅ | 唯一标识符 |
| `fromNode` | string | ✅ | 起始节点 ID |
| `toNode` | string | ✅ | 目标节点 ID |
| `fromSide` | string | ❌ | 起点位置: top, right, bottom, left |
| `toSide` | string | ❌ | 终点位置: top, right, bottom, left |
| `fromEnd` | string | ❌ | 起点形状: none, arrow（默认 none）|
| `toEnd` | string | ❌ | 终点形状: none, arrow（默认 arrow）|
| `label` | string | ❌ | 边上的文字标签 |
| `color` | string | ❌ | 线条颜色 |

---

## 颜色系统

### 预设颜色

| 值 | 颜色 | 英文 |
|----|------|------|
| "1" | 红色 | Red |
| "2" | 橙色 | Orange |
| "3" | 黄色 | Yellow |
| "4" | 绿色 | Green |
| "5" | 青色 | Cyan |
| "6" | 紫色 | Purple |

### Hex 格式

也支持 hex 颜色值：

```json
"color": "#FF5733"
```

---

## ID 生成规则

- 建议使用 16 位十六进制字符串
- 必须在文档内唯一
- 示例：`"8a9b0c1d2e3f4a5b"`

JavaScript 生成方式：
```javascript
const id = Array.from(crypto.getRandomValues(new Uint8Array(8)))
  .map(b => b.toString(16).padStart(2, '0'))
  .join('');
```

---

## 完整示例

### 简单知识图谱

```json
{
  "nodes": [
    {
      "id": "group-main",
      "type": "group",
      "label": "Obsidian 核心功能",
      "x": 0,
      "y": 0,
      "width": 900,
      "height": 500,
      "color": "4"
    },
    {
      "id": "text-intro",
      "type": "text",
      "text": "## 双向链接\n\n让笔记形成网状结构",
      "x": 50,
      "y": 80,
      "width": 350,
      "height": 120
    },
    {
      "id": "img-demo",
      "type": "file",
      "file": "screenshots/00-36.jpg",
      "x": 50,
      "y": 220,
      "width": 350,
      "height": 230
    },
    {
      "id": "text-local",
      "type": "text",
      "text": "## 本地存储\n\n所有文件存储在本地",
      "x": 500,
      "y": 80,
      "width": 350,
      "height": 120
    },
    {
      "id": "img-local",
      "type": "file",
      "file": "screenshots/01-16.jpg",
      "x": 500,
      "y": 220,
      "width": 350,
      "height": 230
    }
  ],
  "edges": [
    {
      "id": "edge-1",
      "fromNode": "text-intro",
      "toNode": "img-demo",
      "fromSide": "bottom",
      "toSide": "top"
    },
    {
      "id": "edge-2",
      "fromNode": "text-local",
      "toNode": "img-local",
      "fromSide": "bottom",
      "toSide": "top"
    },
    {
      "id": "edge-relation",
      "fromNode": "text-intro",
      "toNode": "text-local",
      "fromSide": "right",
      "toSide": "left",
      "label": "相关",
      "color": "5"
    }
  ]
}
```

---

## 验证规则

生成 Canvas 时检查：

1. **ID 唯一性**：所有 node/edge 的 id 必须唯一
2. **引用有效性**：edge 的 fromNode/toNode 必须对应存在的节点
3. **类型正确**：node.type 必须是 text/file/link/group 之一
4. **必需字段**：所有必需字段都有值
5. **JSON 格式**：有效的 JSON 语法

---

## 布局建议

### 推荐尺寸

| 元素类型 | 宽度 | 高度 |
|---------|------|------|
| 短文本 | 200-300 | 60-80 |
| 段落文本 | 300-400 | 100-200 |
| 图片 | 300-400 | 200-300 |
| 分组（小）| 400-600 | 300-400 |
| 分组（大）| 800-1200 | 500-800 |

### 间距建议

- 节点水平间距：80-120px
- 节点垂直间距：60-100px
- Group 内部 padding：40-60px
- Group 之间间距：100-150px

### 对齐建议

- 同级节点顶部对齐
- 相关节点居中对齐
- 使用网格（50px 或 100px）对齐坐标

---

## 参考资源

- [JSON Canvas 官网](https://jsoncanvas.org/)
- [JSON Canvas 规范 v1.0](https://jsoncanvas.org/spec/1.0/)
- [GitHub - obsidianmd/jsoncanvas](https://github.com/obsidianmd/jsoncanvas)
- [官方示例文件](https://github.com/obsidianmd/jsoncanvas/blob/main/sample.canvas)
