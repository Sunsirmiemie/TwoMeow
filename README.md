# TwoMeow — TechJam 2026 Conversational Search Agent

多轮对话电商搜索 Agent，参赛项目：TechJam 2026 Conversational E-Commerce Search Challenge。

**当前成绩**（公开集 200 条，无 LLM 重排）：HitRate@10 = **0.890**，MRR = **0.555**，TechScore = **0.763**

官方 Baseline TechScore = 0.107。

---

## 快速开始

### 环境配置

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 本地评测

```bash
# 标准评测（BM25 + Dense，最优默认配置）
python scripts/run_public_eval.py

# BM25 only（更快，无需 sentence-transformers）
python scripts/run_public_eval.py --no-dense

# 开启 LLM 重排（需要 ANTHROPIC_API_KEY）
export ANTHROPIC_API_KEY=sk-ant-...
python scripts/run_public_eval.py --llm-rank
```

### 消融实验

```bash
# 跑完整 6 行消融表
python scripts/run_ablation_table.py

# 分析失败 session
python scripts/inspect_failures.py --results results.json

# 预先构建 Dense 索引（避免首次评测等待）
python scripts/build_index.py
```

### 运行测试

```bash
python tests/test_state.py
python tests/test_override.py
python tests/test_entropy.py
python tests/test_retrieval.py
python tests/test_ranking.py
python tests/test_contract.py
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
│       └── default.yaml         # 所有超参数
├── scripts/
│   ├── run_public_eval.py       # 主评测脚本
│   ├── run_ablation_table.py    # 6 配置消融表
│   ├── build_index.py           # 预构建 Dense 索引
│   └── inspect_failures.py      # 失败分析
├── tests/                       # 按模块的单元测试
├── docs/
│   ├── architecture.md          # 系统架构详解
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
- Browsing：BM25/Dense 比例随槽数动态调整（0槽 50/50，≥2槽 75/25）

**核心创新**：
- 动态熵属性选择：每轮基于候选池实时打分，选 coverage×entropy 最高的属性提问
- Early Stop（τ=0.3）：候选池收敛时改问通配符，消除无效轮次，MTTC 从 5.1 降至 3.4
- 意图突变重置：精确检测 Override 信号，原子级清空旧槽，重新建立搜索上下文

详见 `docs/architecture.md` 和 `docs/innovation.md`。

---

## 配置说明

所有超参数集中在 `src/config/default.yaml`，可通过 `Agent(catalog_path, config)` 的 `config` 字典覆盖：

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
- BM25 + Dense + 全部对话策略均为本地运行，无网络依赖

推荐离线重排替代方案：`cross-encoder/ms-marco-MiniLM-L-6-v2`（~25MB，详见 `docs/experiment_plan.md`）

---

## 依赖

```
anthropic          # LLM 重排（可选）
sentence-transformers  # Dense 检索（可选，--no-dense 时不需要）
numpy
```

无外部向量数据库，全部数据结构在内存中维护。

---

## 数据说明

- `data/catalog.jsonl`：50,000 件 Amazon 服装商品，字段见比赛规范
- `data/public_set.jsonl`：200 条官方公开评测集（带 ground_truth），场景比例 40/40/15/5
- 评测集对话由 `evaluator/local_evaluator.py` 在运行时动态生成，不是预录的对话
