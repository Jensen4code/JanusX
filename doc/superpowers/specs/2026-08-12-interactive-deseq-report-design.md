# DESeq2 UpSet/火山图交互式 HTML 报告设计

日期：2026-08-12

## 目标

基于 `/share/home/jxfu26/03_counts_star/deseq_upset_20260812/` 已完成的 DESeq2、基因集合和注释结果，生成一个可交互的 HTML 报告。报告覆盖全样本与去除 `R-M-1d-3`、`S-M-1d-3` 两个样本的版本，并覆盖 main effects 与 strict matched 两种比较模型。

报告的核心用途是：从某个比较的 UpSet 交集直接进入基因列表，按效应大小查看基因及其位置、统计量和功能注释，并在同一选择上下文中检查火山图。

## 交付形式

输出目录：

`/share/home/jxfu26/03_counts_star/deseq_upset_20260812/html_report/`

主要文件：

- `index.html`：报告入口和交互逻辑；Plotly JavaScript 内嵌，不依赖外部网络。
- `data/manifest.js`：样本版本、模型、因素、比较、阈值和数据文件索引。
- `data/sets/*.js`：按样本版本、模型、因素和方向懒加载的 UpSet 与基因集合数据。
- `data/volcano/*.js`：每个比较的火山图数据；背景包含该比较的全部基因，显著点另带完整注释。
- `README.md`：打开方式、目录说明、阈值和结果来源。

数据按选择懒加载，避免在首次打开页面时把 104 个比较的全部结果一次性放入浏览器内存。数据文件采用浏览器可直接加载的 JavaScript 数据文件，因此 `index.html` 可直接双击打开；若浏览器对本地文件策略有限制，也提供本地 HTTP 服务方式。

## 页面结构与交互

### 1. 全局选择器

页面顶部提供以下联动选择器：

1. 样本版本：`all_samples` 或 `remove_R-M-1d-3_S-M-1d-3`；
2. 模型：`main_effects` 或 `strict_matched`；
3. 分析因素/类别：Disease、Concentration 或 Day；
4. 具体比较；
5. UpSet 方向：Up 或 Down。

选择变化时，UpSet、基因列表和火山图同步切换到同一比较上下文。页面显示当前样本数、比较定义、固定因素说明及差异阈值。

### 2. UpSet 区域

UpSet 区域由交集数量条形图和集合成员矩阵组成：

- 条形图显示每个交集的基因数；
- 矩阵显示该交集由哪些基因集合组成；
- 点击数量条或矩阵列，会选中对应交集；
- 选中交集后，下面的基因表立即更新为该交集的基因；
- 默认选择数量最大的非空交集；
- 若某个方向没有显著基因，显示明确的空结果说明，不绘制误导性的空图。

UpSet 输入使用已有 `gene_membership.tsv`、`intersection_table.tsv` 和 `set_manifest.tsv`，不重新定义已有基因集边界。

### 3. 基因集合列表

基因表展示当前交集的全部基因，默认按 `abs(log2FoldChange)` 从大到小排序。表格同时提供搜索、列排序和分页。

字段包括：

- `Geneid`；
- `Chr`、`Start`、`End`、`Strand`；
- `baseMean`、`log2FoldChange`、线性 FC（`2^log2FoldChange`）、`padj`、`pvalue`；
- `Direction`；
- `Gene_Source`、`Gene_Biotype`；
- `eggNOG_Description`、`eggNOG_Preferred_name`；
- `eggNOG_GOs`、`eggNOG_KEGG_ko`、`eggNOG_KEGG_Pathway`、`eggNOG_KEGG_Module`、`eggNOG_KEGG_Reaction`；
- `eggNOG_EC`、`eggNOG_COG_category`、`eggNOG_PFAMs`、`eggNOG_eggNOG_OGs`。

缺失注释统一显示为 `-`。表格中的基因可以点击，在当前页面固定显示该基因的完整注释，并同步高亮火山图中的对应点（若当前火山图已加载）。

### 4. 火山图区域

火山图使用当前具体比较的 DESeq2 全基因结果：

- 横轴为 `log2FoldChange`；
- 纵轴为 `-log10(padj)`，对 `padj` 缺失或为 0 的值采用可见性保护值；
- 背景点包含全部参与该比较的基因；
- 显著 Up 与 Down 基因使用独立颜色图层；
- 显著阈值线为 `padj < 0.05` 和 `abs(log2FoldChange) > 1`；
- 显著点悬停显示基因名、坐标、`padj`、`log2FoldChange`、线性 FC、描述、Preferred name、GO 和 KEGG 信息；
- 点击显著点后，在右侧注释面板固定显示完整注释，并提供跳转到当前基因集合/按该基因筛选表格的操作；
- 非显著点作为背景保留，悬停显示基因 ID 和基本统计量。

火山图数据按比较懒加载，每次只加载当前比较，避免 104 个比较同时占用浏览器内存。

## 数据流与实现边界

报告生成器只读现有结果，不重新计算比对、featureCounts 或 DESeq2：

1. 读取 `comparison_manifest.tsv`、两版 `results/*.tsv`、已有 UpSet 文件和 `gene_annotation_reference.tsv`；
2. 校验结果表和注释表的键为 `Geneid`，并合并坐标和功能注释；
3. 为每个方向和每个 UpSet 类别生成轻量交互数据；
4. 为每个具体比较生成火山图数据，保存全部背景点和显著点的完整注释；
5. 生成入口 HTML、数据索引和使用说明。

线性 FC 仅作为展示字段计算，DESeq2 的统计判断仍使用原始 `padj` 和 `log2FoldChange`。差异阈值固定为 `padj < 0.05`、`abs(log2FoldChange) > 1`，与已完成的基因集和 UpSet 结果保持一致。

## 异常处理

- 缺少某个结果文件、UpSet 文件或注释键时，生成器立即失败并报告具体路径；
- `padj` 缺失、非有限或为 0 时，火山图使用有限的绘图值，但表格保留原始值并标记缺失；
- 同一 `Geneid` 出现多条注释时，按现有注释表规则去重并记录重复数；
- 空的 Up/Down 集合显示空状态卡片，页面其他选择仍可用；
- 所有输出文件完成后再写入完成标记，避免用户误用不完整报告。

## 验证计划

生成完成后在 JXFU 上验证：

1. 两个样本版本各有 52 个比较（main 6 个、strict 46 个），共 104 个比较；
2. 每个比较的 Up/Down 数据索引、基因表数据和火山图数据均存在且非空（允许统计学上确实为空的方向）；
3. 随机抽查全样本 Disease、去除异常样本的 Day 及一个 strict matched 比较，核对基因数、`padj`、`log2FC`、坐标和 GO/KEGG 注释与原始 TSV 一致；
4. 检查 HTML 中没有指向外部 CDN 的资源；
5. 用 Python 解析 HTML 和数据文件，检查 JavaScript 数据索引、比较 ID 和基因 ID 的一致性；
6. 用浏览器或本地 HTTP 服务打开报告，验证选择器联动、UpSet 点击、基因表排序/搜索、火山图悬停/点击和基因列表跳转。

## 非目标

- 不改变已有差异表达阈值、样本剔除方案或统计模型；
- 不在浏览器端重新运行 DESeq2 或功能富集；
- 不把所有原始 312.5 MB 结果表重复嵌入入口页面；原始 TSV 继续作为可追溯的完整结果保存。
