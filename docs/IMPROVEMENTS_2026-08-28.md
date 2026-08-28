# 2026-08-28 改进记录：BM25 + 特征重排 + 提问策略修正

## 结果

公开集 200 条，官方评测器，BM25-only（零模型、零 token、全离线）：

| 指标 | 稳定化基线（BM25-only） | 本次改进 | 历史最优(results_optimal) |
|---|---|---|---|
| Hit Rate@10 | 0.865 | **0.920** | 0.890 |
| MRR | 0.553 | **0.580** | 0.555 |
| MTTC | 3.65 | **3.195** | 3.405 |
| Efficiency | 0.735 | **0.781** | 0.760 |
| **TechnicalScore** | 0.745 | **0.790** | 0.763 |

分场景（本次改进）：

| 场景 | 命中率 | MRR | MTTC |
|---|---|---|---|
| Buying (80) | 0.912 | 0.571 | 2.61 |
| Browsing (80) | 0.938 | 0.507 | 3.23 |
| Intent Override (30) | 0.867 | 0.760 | 4.70 |
| Boundary (10) | 1.000 | 0.693 | 3.10 |

失败会话从 27 个降到 19 个（buying 8 / browsing 6 / override 4 / boundary 1）。

## 改动清单（原文件备份在 `backups/2026-08-28/`）

1. **`src/dialogue/attribute_stats.py`**：size 提取重写。原来 `\bs\b|\bm\b` 会误匹配 "I'm"→m、"100% Textile"→100；现在只在 `size X`、S/M/L 连写、数字尺寸等明确语境中识别。
2. **`src/agent/state.py`**：槽位 fallback 模式同步修复；新增 `looking for X` 类目提取（槽位 `category`）；`SlotTracker.extract_and_update` 现在返回"是否有新信息"。
3. **`src/retrieval/catalog.py`**：索引新增 `categories` 与 `meta`（价格、评分、评分数）缓存，供过滤与重排使用。
4. **`src/retrieval/bm25.py`**：buying 场景增加类目后过滤（任一类目标签匹配即保留，不会把结果集清空）；AND 表达式已回退（见"试过但回退"）。
5. **`src/ranking/reranker.py`**：新增**无模型特征重排**——BM25/RRF 基础分 + 槽位-标题/类目词重叠(Jaccard) + 类目命中 + 评分数流行度先验 + 价格邻近度，加权求和，零训练；LLM 路径保留且失败时回退到特征重排。
6. **`src/agent/orchestrator.py`**：无 LLM 时不再把候选池截断到 20（截断是为 LLM 重排设计的，离线时反而丢召回）；提问策略保持"持续提问"（见下）。
7. **`src/config/default.yaml` / `loader.py`**：新增 `use_features` 与 `feature_weights` 配置；最终权重 base 1.0 / slot 0.8 / category 1.0 / popularity 0.15 / price 0.4。
8. **`tests/test_config.py`、`tests/test_entropy.py`**：同步更新断言（新配置键、Clarifier 无可用属性时返回 None、无 LLM 时全候选池）。

## 试过但回退的改动（重要经验）

| 尝试 | 结果 | 结论 |
|---|---|---|
| Buying 用 AND 表达式（`"leather" AND "red"`） | TechScore 0.519→0.439 | 评测器给的约束是泛化词，AND 把目标商品一起滤掉；**OR + 槽位翻倍 + BM25 排序更稳** |
| 类目短语进 FTS（`"Shoes & Jewelry"`） | 结果集被清空 | 粗类目是空格拼接的短语，FTS 按相邻 token 匹配，直接失效 |
| Boundary 场景永不提问 | Boundary 命中下降 | **Boundary 只拒绝第一次提问，后续提问会正常透露约束**——不能停 |
| 连续 2 轮无新信息就停问 | Hit/MRR 下降 | 提问不占推荐机会（每轮都推），停问没有收益反而切断了后续信息 |
| 特征权重 slot 2.2 / category 1.8 | MRR 下降 | 特征权重过大打乱 BM25 序；最终收敛到 slot 0.8 / category 1.0 |

## 复现

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install .          # 或 pip install numpy PyYAML
python scripts/run_public_eval.py --no-dense --output results_bm25_features_20260828.json
```

结果文件：`results_bm25_features_20260828.json`。

## 剩余空间（下一步）

- **MRR 头仍然很大**：88 个命中在 rank 1，但仍有 37 个命中在 6–10 名；继续调特征权重或加入更细的槽位-标题匹配能再涨 MRR（权重 0.3）。
- **Intent Override 最弱**（hit 0.867 / MTTC 4.70）：改口后只剩泛化约束（如 "leather"），需要"改口后检索策略切换"（例如改口后优先按改口前的槽位集合做差集排序）。
- 剩余 19 个失败大多卡在"意图卡里只有泛化词"的场景，可以尝试规则查询扩展（同义词词典）和 top-10 品牌/价格多样性兜底。
- 如需启用本地模型（打包随提交，不微调）：`use_dense: true`（all-MiniLM 已缓存）或本地交叉编码器重排，配合现有特征重排做融合。
