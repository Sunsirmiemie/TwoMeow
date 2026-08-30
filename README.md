# TwoMeow — TechJam 2026 Conversational Search Agent

多轮对话电商搜索 Agent，参赛项目：TechJam 2026 Conversational E-Commerce Search Challenge。

**当前默认结果**（公开集 200 条，`PYTHONHASHSEED=0`，BM25-only，无 LLM）：HitRate@10 = **0.955**，MRR = **0.706546**，MTTC = **2.865**，Efficiency = **0.8135**，TechnicalScore = **0.852164**，token = **0**。相对更新仓库原始公开集结果（0.905 / 0.706437 / 3.535 / 0.813731），命中率提高 5 个百分点、MTTC 缩短 0.67 轮、技术分提高 0.038433。

**泛化警告**：第二组完全隔离的 400-ASIN 最终盲测中，当前版本 TechnicalScore 为 `0.750553`，低于原版 BM25 的 `0.760657`。因此上面的公开集提升不能被表述为已经证明的泛化提升；当前三项优化仍应视为实验方案。

本轮在原有风险感知 MMR 上重新实现三项优化：否定感知回复提纯、候选池感知的动态商品属性评分，以及带风险门控的字段分离 Dense。设计、消融和合规边界见 [`docs/THREE_OPTIMIZATIONS_20260829.md`](docs/THREE_OPTIMIZATIONS_20260829.md)。

**仓库历史最佳记录**（公开集 200 条，见 `results_optimal.json`）：HitRate@10 = **0.890**，MRR = **0.555**，TechScore = **0.763**。项目原有说明将该实验记录为无 LLM 重排，但 JSON 本身不包含足以独立验证运行配置的 provenance；本轮也没有重新跑 Dense/LLM。

**本轮可复现 BM25-only 基线**：HitRate@10 = **0.865**，MRR = **0.553**，MTTC = **3.650**，TechScore = **0.745**。完整口径、命令和改动说明见 [`docs/BASELINE_STABILIZATION.md`](docs/BASELINE_STABILIZATION.md)。

官方 Baseline TechScore = 0.107。

---

## 快速开始

### 环境配置

支持 Python `>=3.10,<3.13`。按使用场景安装：

```bash
python -m venv .venv
source .venv/bin/activate

pip install .              # 核心：BM25 + 配置加载
pip install '.[dense]'     # 可选：Dense 检索
pip install '.[test]'      # 可选：测试、构建与制品校验
```

`requirements.txt` 是 Dense、LLM 与开发/测试工具的完整聚合依赖；需要一次装齐时可运行 `pip install -r requirements.txt`。

### 本地评测

```bash
# 标准评测（推荐）：BM25-only，完全可复现，零 token
PYTHONHASHSEED=0 python scripts/run_public_eval.py --no-dense --output /tmp/twomeow-result.json

# 默认配置评测：开启 Dense 风险门控（首次运行需下载 sentence-transformers 模型）
python scripts/run_public_eval.py --output /tmp/twomeow-default.json

# 诊断用途：强制绕过风险门控（公开集上会明显降分）
python scripts/run_public_eval.py --force-dense --output /tmp/twomeow-forced-dense.json
```

### 消融实验

```bash
# 注意：当前脚本会写入或覆盖仓库根目录的 ablation_results.json。
# 它不属于安全的只读验证流程；只读验证请使用上面的 /tmp 评测命令或 pytest。
# 跑完整 6 行消融表
python scripts/run_ablation_table.py

# 分析失败 session
python scripts/inspect_failures.py --results results.json

# 预先构建 Dense 索引（避免首次评测等待）
python scripts/build_index.py
```

### 运行测试

```bash
python -m pytest -q
```

---

## 项目结构

```
TwoMeow/
├── src/
│   ├── agent/
│   │   ├── orchestrator.py      # Agent 主入口（reset / respond）
│   │   ├── router.py            # 场景分类（buying/browsing/override/boundary）
│   │   ├── state.py             # SessionMemory + SlotTracker
│   │   └── response_builder.py  # 查询构建 + 消息生成
│   ├── retrieval/
│   │   ├── catalog.py           # SQLite FTS5 索引 + 属性缓存
│   │   ├── bm25.py              # BM25 检索 + slot 硬过滤
│   │   ├── dense.py             # Dense 检索（all-MiniLM-L6-v2）
│   │   ├── hybrid.py            # BM25 + Dense + RRF 融合
│   │   └── candidate_builder.py # 候选池截断策略
│   ├── dialogue/
│   │   ├── attribute_stats.py   # 属性正则模式 + 全局熵统计
│   │   ├── entropy.py           # coverage × entropy 打分
│   │   ├── purification.py      # 否定/改口/无偏好回复提纯
│   │   ├── question_policy.py   # Clarifier（动态属性选择）
│   │   ├── early_stop.py        # 熵阈值早停（τ=0.3）
│   │   └── override.py          # Intent Override / Boundary 检测
│   ├── ranking/
│   │   ├── scorer.py            # RRF 融合算法
│   │   ├── dynamic_attributes.py # 动态属性权重与商品兼容度
│   │   ├── mmr.py               # MMR 去重选择
│   │   ├── reranker.py          # 本地字段感知 top-20 → top-10 重排（LLM 仅兼容可选路径）
│   │   ├── features.py          # 商品文本提取
│   │   └── profile_prior.py     # 用户画像先验（本地、确定性）
│   ├── evaluation/
│   │   ├── runner.py            # 评测封装
│   │   ├── analysis.py          # 分数分析
│   │   ├── failure_analysis.py  # 失败 session 分析
│   │   └── ablation.py          # 消融配置
│   └── config/
│       └── default.yaml         # 本轮接线的 Agent 主要可调参数
├── scripts/
│   ├── run_public_eval.py       # 主评测脚本
│   ├── run_ablation_table.py    # 6 配置消融表
│   ├── build_index.py           # 预构建 Dense 索引
│   └── inspect_failures.py      # 失败分析
├── tests/                       # 按模块的单元测试
├── docs/
│   ├── architecture.md          # 系统架构详解
│   ├── BASELINE_STABILIZATION.md # 基线稳定化改动与验证记录
│   ├── experiment_plan.md       # 实验记录与计划
│   ├── innovation.md            # 创新点说明
│   └── team_ownership.md        # 团队分工
├── data/
│   ├── catalog.jsonl            # 50,000 件商品目录
│   └── public_set.jsonl         # 200 条公开评测集
├── evaluator/                   # 官方评测器（不修改）
├── starter/agent.py             # 官方接口入口（re-export）
├── requirements.txt
└── .env.example
```

---

## 系统流程

每轮 `respond()` 按 6 步执行：

```
① 场景分类  → buying / browsing / override / boundary
② 槽提取    → {material: "cotton", color: "blue", ...}
③ 检索      → 动态字段 BM25 → 动态商品属性重排 → top-100
              ↘ 仅在 BM25 弱时启用字段分离 Dense + 自适应 RRF
④ 问题决策  → coverage×entropy 打分 → Early Stop(τ=0.3) → ask_attribute
⑤ 排序      → 前 20 候选 → 本地字段感知重排 → top-10
⑥ 返回      → {recommendations, ask_attribute, message}
```

**检索策略**：
- Buying：动态字段 BM25 + 价格过滤 + 每商品动态属性评分
- Browsing：先判断 BM25 当前候选量与置信度；仅在词法召回不足时启用字段分离 Dense
- Dense 启用后，身份字段与属性字段分别编码，权重随累计证据改变；BM25/Dense 融合权重还会参考本轮结果置信度

**核心创新**：
- 回复提纯：区分正向条件、否定条件和“无偏好”；意图覆盖时替换冲突属性，其他旧属性仅作为低置信度弱证据
- 动态商品属性：属性权重由证据置信度、时效性和当前候选池选择性共同决定，同时用于 50,000→100 和 top-20→10
- 风险门控字段 Dense：不拼接商品向量；语义召回仅在 BM25 薄弱时介入，离线失败自动回退
- 动态熵属性选择：每轮基于候选池实时打分，选 coverage×entropy 最高的属性提问
- Early Stop（τ=0.3）：候选池收敛时改问通配符，消除无效轮次，MTTC 从 5.1 降至 3.4
- 意图突变重置：精确检测 Override 信号，原子级清空旧槽，重新建立搜索上下文
- 字段感知重排：按确认槽位在 title/categories/features/details 中的覆盖率推进精确匹配商品

详见 `docs/architecture.md`、`docs/innovation.md` 和 `docs/BASELINE_STABILIZATION.md`。

---

## 配置说明

本轮已经接入运行时的 Agent 主要可调参数集中在 `src/config/default.yaml`，可通过 `Agent(catalog_path, config)` 的扁平 `config` 字典覆盖：

```python
agent = Agent("data/catalog.jsonl", {
    "use_dense": True,            # BM25+Dense 混合检索
    "use_dense_risk_gate": True,  # 强 BM25 场景不引入 Dense 噪声
    "use_field_aware_dense": True,# 身份/属性分别编码
    "dense_device": "auto",      # MPS 实验可显式设为 "mps"
    "dense_max_seq_length": 256,  # 多模型公平对照的统一上下文长度
    "dense_query_prefix": "",    # BGE/E5 等检索模型可配置官方前缀
    "dense_document_prefix": "",
    "use_reply_purification": True,
    "use_dynamic_attribute_scoring": True,
    "use_dynamic_entropy": True,  # 动态熵属性选择
    "use_early_stop": True,       # 熵阈值早停
    "use_override_detection": True,
    "use_llm_ranker": False,      # 可选，默认关闭
    "ranker_model": "claude-haiku-4-5-20251001",
    "use_field_aware_slot_coverage": True,
})
```

### Dense 模型选择

固定种子 MPS 对照中，E5/BGE/MPNet 相对 MiniLM 的收益很小，而且四个模型在强制 RRF 路径上都明显弱于当前风险门控路径。因此提交版继续使用体积和延迟更低的 `all-MiniLM-L6-v2`，并只在 BM25 词法召回薄弱时启用 Dense。模型选择数据和边界见 `docs/DENSE_MODEL_SELECTION_20260830.md`。

---

## 离线模式（决赛禁网）

默认方案（BM25-only）完全离线，无网络依赖，无需下载模型。以字段感知的确认约束覆盖、画像先验和风险感知 MMR 对 top-20 候选重排。

启用 Dense 时：模型/索引不可用则自动回退 BM25；默认风险门控采用惰性加载，强 BM25 场景不触发 Dense 模型加载。

---

## 依赖

核心依赖只有 `numpy` 与 `PyYAML`；`sentence-transformers`（Dense）、`anthropic`（LLM）以及测试/构建工具通过 `pyproject.toml` 的 extras 分组安装。`requirements.txt` 则提供完整聚合环境。

无外部向量数据库，全部数据结构在内存中维护。

---

## 数据说明

- `data/catalog.jsonl`：50,000 件 Amazon 服装商品，字段见比赛规范
- `data/public_set.jsonl`：200 条官方公开评测集（带 ground_truth），场景比例 40/40/15/5
- 评测集对话由 `evaluator/local_evaluator.py` 在运行时动态生成，不是预录的对话
