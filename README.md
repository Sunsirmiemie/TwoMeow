# TwoMeow — TechJam 2026 Conversational Search Agent

多轮对话电商搜索 Agent，参赛项目：TechJam 2026 Conversational E-Commerce Search Challenge。

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
pip install '.[llm]'       # 可选：LLM 重排
pip install '.[test]'      # 可选：测试、构建与制品校验
```

`requirements.txt` 是 Dense、LLM 与开发/测试工具的完整聚合依赖；需要一次装齐时可运行 `pip install -r requirements.txt`。

### 本地评测

```bash
# 默认配置评测（默认开启 Dense；仅缺少 sentence-transformers 包时自动回退 BM25）
# 若包已安装但模型未缓存，模型加载可能尝试下载或直接失败
python scripts/run_public_eval.py --output /tmp/twomeow-default.json

# BM25 only（可复现、更快，无需 sentence-transformers；避免覆盖仓库结果）
python scripts/run_public_eval.py --no-dense --output /tmp/twomeow-bm25.json

# 开启 LLM 重排（需要 ANTHROPIC_API_KEY）
export ANTHROPIC_API_KEY=sk-ant-...
python scripts/run_public_eval.py --llm-rank --output /tmp/twomeow-llm.json
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
│   │   ├── question_policy.py   # Clarifier（动态属性选择）
│   │   ├── early_stop.py        # 熵阈值早停（τ=0.3）
│   │   └── override.py          # Intent Override / Boundary 检测
│   ├── ranking/
│   │   ├── scorer.py            # RRF 融合算法
│   │   ├── reranker.py          # LLM 重排（Claude API，可选）
│   │   ├── features.py          # 商品文本提取
│   │   └── profile_prior.py     # 用户画像个性化（stub）
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
③ 检索      → BM25(100) + Dense(100) → RRF → top-100
④ 问题决策  → coverage×entropy 打分 → Early Stop(τ=0.3) → ask_attribute
⑤ 排序      → 候选截断 → top-10
⑥ 返回      → {recommendations, ask_attribute, message}
```

**检索策略**：
- Buying：BM25 only + 价格硬过滤
- Browsing：BM25/Dense 比例随槽数动态调整（0 槽 50/50，1 槽 60/40，≥2 槽 75/25）

**核心创新**：
- 动态熵属性选择：每轮基于候选池实时打分，选 coverage×entropy 最高的属性提问
- Early Stop（τ=0.3）：候选池收敛时改问通配符，消除无效轮次，MTTC 从 5.1 降至 3.4
- 意图突变重置：精确检测 Override 信号，原子级清空旧槽，重新建立搜索上下文

详见 `docs/architecture.md`、`docs/innovation.md` 和 `docs/BASELINE_STABILIZATION.md`。

---

## 配置说明

本轮已经接入运行时的 Agent 主要可调参数集中在 `src/config/default.yaml`，可通过 `Agent(catalog_path, config)` 的扁平 `config` 字典覆盖：

```python
agent = Agent("data/catalog.jsonl", {
    "use_dense": True,            # BM25+Dense 混合检索
    "use_dynamic_entropy": True,  # 动态熵属性选择
    "use_early_stop": True,       # 熵阈值早停
    "use_override_detection": True,
    "use_llm_ranker": False,      # 开启需要 ANTHROPIC_API_KEY
    "ranker_model": "claude-haiku-4-5-20251001",
})
```

---

## 离线模式（决赛禁网）

当 `ANTHROPIC_API_KEY` 未设置或网络不可用时：
- LLM 重排自动回退到检索分顺序（无需修改代码）
- BM25 与全部对话策略均在本地运行
- Dense 推理在本地执行，但首次加载模型仍要求模型已安装/缓存；严格离线复现请使用 `--no-dense`

推荐离线重排替代方案：`cross-encoder/ms-marco-MiniLM-L-6-v2`（~25MB，详见 `docs/experiment_plan.md`）

---

## 依赖

核心依赖只有 `numpy` 与 `PyYAML`；`sentence-transformers`（Dense）、`anthropic`（LLM）以及测试/构建工具通过 `pyproject.toml` 的 extras 分组安装。`requirements.txt` 则提供完整聚合环境。

无外部向量数据库，全部数据结构在内存中维护。

---

## 数据说明

- `data/catalog.jsonl`：50,000 件 Amazon 服装商品，字段见比赛规范
- `data/public_set.jsonl`：200 条官方公开评测集（带 ground_truth），场景比例 40/40/15/5
- 评测集对话由 `evaluator/local_evaluator.py` 在运行时动态生成，不是预录的对话
