# 任务：Hugo 博客 Mermaid 图表构建时预渲染

## 目标

将博客中所有 Mermaid 代码块在 GitHub Actions CI 构建阶段预渲染为 SVG 静态图片，使客户端无需加载 Mermaid JS 库，实现零运行时渲染。

## 约束条件

- 本项目是 Hugo + PaperMod 主题的博客，部署在 GitHub Pages
- 预渲染脚本只在 CI 环境中修改文件，不得影响仓库源码
- 博客支持 `defaultTheme = "auto"`（亮/暗主题切换），需要考虑 SVG 在两种主题下的可读性
- 使用 `@mermaid-js/mermaid-cli`（`mmdc`）作为渲染工具

## 执行步骤

### 第一步：分析现有实现

先了解当前 Mermaid 是如何引入和渲染的：

1. 查找项目中所有与 mermaid 相关的文件：
   ```
   grep -r "mermaid" --include="*.html" --include="*.js" --include="*.toml" --include="*.yaml" -l .
   ```
2. 确认当前的 mermaid 加载方式（CDN script 标签、Hugo partial、shortcode 等）
3. 查找所有包含 mermaid 代码块的 markdown 文件：
   ```
   grep -rl '```mermaid' content/
   ```
4. 查看现有的 GitHub Actions workflow 文件（`.github/workflows/` 目录）
5. **将以上发现汇报给我，等待我确认后再继续**

### 第二步：编写预渲染 Python 脚本

在 `scripts/render_mermaid.py` 创建脚本，要求：

- 用正则匹配 markdown 中的 ` ```mermaid ... ``` ` 代码块
- 对每个代码块内容取 MD5 hash 前 10 位作为文件名（相同内容不重复渲染）
- 调用 `mmdc` 将代码块渲染为 SVG，输出到 `static/mermaid-svg/` 目录
- 将原 markdown 中的 mermaid 代码块替换为 Hugo 图片引用（用 `figure` shortcode 或直接 `<img>` 标签，根据项目实际情况选择）
- 脚本应处理以下边界情况：
  - 一个 md 文件中有多个 mermaid 代码块
  - mermaid 代码块内容为空或格式异常时跳过并打印警告
  - `mmdc` 渲染失败时不中断整体流程，记录错误并跳过该块
- `mmdc` 渲染参数：`-t dark -b transparent`（或根据主题适配策略调整，见下方说明）
- 脚本顶部添加清晰的注释说明用途和注意事项

### 第三步：处理亮/暗主题适配

本博客 `defaultTheme = "auto"`，预渲染 SVG 需要在两种主题下可读。选择以下方案之一（请分析项目实际情况后选择最合适的，并说明理由）：

**方案 A：双主题 SVG + CSS 切换**
- 每个图表生成 light 和 dark 两个版本
- 用 `<picture>` + `prefers-color-scheme` 媒体查询切换
- 优点：两种主题下都完美显示
- 缺点：SVG 数量翻倍，脚本复杂度增加

**方案 B：透明背景 + CSS 滤镜**
- 只生成 dark 主题 SVG（`-b transparent`）
- 亮色模式下用 CSS `filter: invert(1)` 反转颜色
- 优点：实现简单，只需一套 SVG
- 缺点：反转后颜色可能不够精确

**方案 C：neutral 主题 + 透明背景**
- 使用 `mmdc -t neutral -b transparent`，生成中性配色 SVG
- 通过 CSS 变量控制 SVG 内文字颜色适配主题
- 需要评估 neutral 主题的 SVG 在 PaperMod 亮/暗模式下的实际表现

### 第四步：修改 GitHub Actions Workflow

修改现有的 deploy workflow 文件，在 Hugo build 步骤之前加入预渲染阶段：

```yaml
# 需要添加的步骤（插入到 hugo build 之前）：

- uses: actions/setup-node@v4
  with:
    node-version: '20'

- name: Install mermaid-cli
  run: npm install -g @mermaid-js/mermaid-cli

- uses: actions/setup-python@v5
  with:
    python-version: '3.12'

- name: Pre-render Mermaid to SVG
  run: python scripts/render_mermaid.py
```

注意事项：
- `mmdc` 依赖 Puppeteer（headless Chromium），CI 环境中可能需要安装系统依赖，注意检查是否需要 `npx puppeteer browsers install chrome`
- 确保预渲染步骤在 checkout 之后、hugo build 之前
- 如果现有 workflow 有缓存策略，考虑将 `node_modules` 也纳入缓存

### 第五步：清理客户端 Mermaid 依赖

预渲染生效后，移除客户端加载 Mermaid JS 的代码：

1. 删除或注释掉模板中的 `<script src="...mermaid...">` 标签
2. 删除或注释掉 `mermaid.initialize(...)` 相关代码
3. 如果有专门的 mermaid partial 文件（如 `layouts/partials/mermaid.html`），将其内容替换为空或删除
4. 清理 `hugo.toml` 中与 mermaid 相关的参数（如果有）
5. **不要删除文件**，先注释并标注 `<!-- DISABLED: mermaid pre-rendered in CI -->`，待验证通过后再彻底清理

### 第六步：添加 CSS 样式

在 `assets/css/extended/custom.css` 中为预渲染的 SVG 图片添加样式：

```css
/* Mermaid 预渲染 SVG 样式 */
.mermaid-svg {
  display: block;
  max-width: 100%;
  height: auto;
  margin: 1.5rem auto;
}
```

若选择了方案 B（CSS 滤镜），还需要加入主题适配样式。

### 第七步：验证

完成以上步骤后，进行以下检查：

1. 在本地安装 `mmdc` 并手动运行 `python scripts/render_mermaid.py`，确认 SVG 正确生成
2. 运行 `hugo server` 查看渲染效果
3. 检查生成的 HTML 中不再包含任何 mermaid JS 引用
4. 确认 `static/mermaid-svg/` 已加入 `.gitignore`（这些是 CI 产物，不需要提交）
5. 汇报验证结果

## 重要提醒

- 每完成一个主要步骤后暂停，向我汇报进展和发现，等待确认后再继续
- 如果发现项目的实际结构与本文档的假设不一致，优先以实际代码为准，并说明差异
- 不要创建你不确定是否需要的文件，先确认再动手
- 预渲染脚本修改的是 CI 环境中的 checkout 副本，确保不会产生 git commit 推回仓库