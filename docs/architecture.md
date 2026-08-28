# 系统架构

## 总览

TwoMeow 是一个纯内存、无外部数据库依赖的多轮对话电商搜索 Agent，实现官方 `reset() / respond()` 接口。

```
用户消息 (turn N)
      │
      ▼
┌─────────────────────────────────────────────────┐
│  src/agent/orchestrator.py  (Agent.respond)      │
│                                                 │
│  ① IntentRouter   场景分类 + Override 检测       │
│  ② SlotTracker    结构化槽提取                   │
│  ③ HybridRetriever 检索 (BM25 + Dense + RRF)    │
│  ④ Clarifier      动态熵属性选择 + Early Stop     │
│  ⑤ Ranker         top-20 → 字段感知本地重排 → top-10 │
│  ⑥ 返回结果        recommendations + ask_attr    │
└─────────────────────────────────────────────────┘
      │
      ▼
评估器决定下一轮用户消息
```

## 模块依赖关系

```
dialogue/attribute_stats.py   ← 基础层（无内部依赖）
agent/state.py                ← 基础层

dialogue/entropy.py           ← attribute_stats
dialogue/override.py          ← 无内部依赖
dialogue/question_policy.py   ← entropy + attribute_stats
dialogue/early_stop.py        ← entropy

agent/router.py               ← dialogue/override
agent/response_builder.py     ← 无内部依赖

ranking/scorer.py             ← 无内部依赖
ranking/features.py           ← 无内部依赖
ranking/profile_prior.py       ← 无内部依赖
ranking/reranker.py           ← profile_prior

retrieval/catalog.py          ← dialogue/attribute_stats
retrieval/bm25.py             ← retrieval/catalog
retrieval/dense.py            ← ranking/features
retrieval/hybrid.py           ← retrieval/bm25 + retrieval/dense + ranking/scorer

agent/orchestrator.py         ← 全部上层模块
```

依赖方向单向：`dialogue/ranking → retrieval → agent`，无循环 import。

## 各层职责

### Layer 1: 对话状态（`src/agent/state.py`）

**SessionMemory** — 单会话内存：
- `slots: dict[str, str]` — 已确认约束（material/color/size/style/use_case/feature/budget）
- `scenario_type` — buying / browsing / intent_override / boundary
- `asked_attributes` — 已问过的属性（防重复）
- `history` — 近期消息记录（用于查询构建）

**SlotTracker** — 解析规则（优先级从高到低）：
1. `"what matters is: X; Y"` → 结构化多槽解析
2. `"a key requirement is: X"` → 单槽买家约束
3. `"what I need is: X"` → Override 后新约束
4. 自由文本 → 正则 fallback（不覆盖已有槽）

槽分类器完全 mirror 评估器 `classify_constraint()`，保证一致性。

### Layer 2: 场景路由（`src/agent/router.py` + `src/dialogue/override.py`）

| 信号 | 模式 | 动作 |
|------|------|------|
| Intent Override | `"ignore my earlier preference"` | 清空 slots + asked_attributes，改为 buying 轨道 |
| Boundary | `"please use your judgment"` | 设标志，跳过本属性，继续 |
| Buying | Turn 0 + `"a key requirement is:"` | scenario_type = buying |
| Browsing | Turn 0 + 其他 | scenario_type = browsing |

### Layer 3: 检索（`src/retrieval/`）

**BM25 检索**（SQLite FTS5，全内存）

字段权重：
```
title(6.0) > categories(4.0) > features(2.5) = description(2.5) > store(1.5) > details(1.0)
```

查询构建：`slot_text×2 + 当前消息 + 近3轮历史`（双重 slot 放大 BM25 精准匹配权重）

Buying 专属：追加 `price ≤ budget` 硬过滤。

**Dense 检索**（sentence-transformers all-MiniLM-L6-v2）

编码策略：`title. categories. features[:300]`（features 包含评估器意图卡的关键词）

50K 商品首次编码后缓存 `.embed_cache/*.npz`，后续直接加载。

cosine similarity via dot product（L2 归一化后等价）+ partial sort O(n log k)。

**RRF 融合**（仅 Browsing 轨道）

| 已确认槽数 | BM25 权重 | Dense 权重 | 逻辑 |
|-----------|-----------|-----------|------|
| 0 | 50% | 50% | 无约束，语义召回兜底 |
| 1 | 60% | 40% | 约束少，仍需语义 |
| ≥2 | 75% | 25% | 约束多，精确匹配为主 |

`score = w_bm25 × 1/(60+rank_bm25) + w_dense × 1/(60+rank_dense)`

Buying 轨道跳过 Dense，直接 BM25-only（评估器每轮给出精确约束文本，Dense 在此场景引入噪声）。

### Layer 4: 问题决策（`src/dialogue/`）

**信息熵打分**

对候选池中每个剩余属性：
```
score(attr) = coverage(attr) × entropy(attr)

coverage = 有该属性值的商品 / 候选池总数
entropy  = 归一化信息熵（池<10个时与全局熵加权混合）
```

全局熵来源：PDF 对 50,000 件商品的统计（use_case=0.87 > budget=0.79 > color=0.77 > …）

**Early Stop（τ=0.3）**

```python
if max(score(attr) for attr in remaining) < 0.3:
    ask "other"  # 通配，评估器返回任意剩余约束
else:
    ask argmax(score)  # 信息量最大的属性
```

当候选池已收敛（属性分布均质化），继续问特定属性只会得到 "I don't have an additional preference"（空轮）。Early Stop 将 MTTC 从 5.1 降至 3.4，HitRate 从 0.800 升至 0.890。

### Layer 5: 排序（`src/ranking/`）

**候选池截断**：slots < 2 且候选 ≥ 50 → 取 top-20（防止槽少时从过大的模糊池里猜）

**字段感知本地重排**（默认）：
- 固定取检索 top-20，输出 top-10；
- 以 BM25/RRF 分数、类目、价格、热度和确认槽位的精确覆盖率组成相关性分；
- 覆盖率检查商品 `title/categories/features/details` 是否满足已确认约束；
- 不训练模型、不调用网络或 LLM。LLM 路径仅为兼容可选分支，不是默认评测方案。

## 超参数汇总

全部集中在 `src/config/default.yaml`，代码中不出现魔法数字。

| 参数 | 值 | 说明 |
|------|-----|------|
| RRF K | 60 | 标准 RRF 常数 |
| BM25 base weight | 0.75 | Browsing ≥2 槽时 |
| Dense base weight | 0.25 | 同上 |
| entropy τ | 0.3 | Early Stop 阈值 |
| min pool for dynamic | 10 | 小于此值混合全局熵 |
| rerank top_n | 20 | 本地重排候选数 |
| use_field_aware_slot_coverage | true | 确认槽位字段覆盖开关 |
| truncated pool size | 20 | 槽少时截断 |

## 历史系统消融结果（公开集 200 条，无 LLM 重排）

| 系统 | Buying | Browsing | Override | Boundary | Overall | MRR | MTTC | TechScore |
|------|--------|----------|----------|----------|---------|-----|------|-----------|
| 官方 Baseline | - | - | - | - | 0.125 | 0.068 | 9.81 | 0.107 |
| BM25-only | 0.663 | 0.725 | 0.767 | 0.900 | 0.715 | 0.464 | 6.36 | 0.590 |
| + Dense (Hybrid) | 0.663 | 0.725 | 0.767 | 0.900 | 0.715 | 0.427 | 6.38 | 0.578 |
| + Entropy | 0.738 | 0.838 | 0.833 | 0.900 | 0.800 | 0.459 | 5.08 | 0.656 |
| + Early Stop | 0.888 | 0.900 | 0.833 | 0.900 | 0.885 | 0.536 | 3.44 | 0.755 |
| + Override | 0.888 | 0.900 | 0.867 | 0.900 | 0.890 | 0.555 | 3.40 | **0.763** |

上表用于说明历史对话策略增益。当前 rerank 为保留画像先验的字段感知精确约束覆盖与风险感知 MMR：HitRate@10=0.905、MRR=0.706437、TechnicalScore=0.813731，结果见 `results_rerank_risk_aware_mmr_profile.json`。
