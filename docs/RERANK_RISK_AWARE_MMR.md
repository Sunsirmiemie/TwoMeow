# 重排更新记录：画像先验 + 字段感知 + 风险约束 MMR

## 目标

在不使用大模型、不改动检索与官方评测器的前提下，对检索阶段给出的候选商品做本地重排，输出最终前 10 个商品。目标是在保持相关性信号的基础上，减少前 10 名中高度相似商品的重复占位。

## 当前方案

重排位于 `src/ranking/reranker.py`，只处理已有候选，不会过滤或新增商品。流程如下：

1. **画像先验**：从会话中的 `preference_tags` 提取偏好词，并与商品标题、类目和画像文本的词集合计算重叠；该分数只作为有上界的小幅加分，不会排除任何候选。
2. **字段感知约束覆盖**：把已确认的槽位词与商品的标题、类目、features、details 文本比对。用户明确表达的颜色、材质、品类等词只要出现在任一字段，就会得到覆盖分，避免只看标题而漏掉有效商品。
3. **相关性特征融合**：以原始 BM25/RRF 分数为主，同时融合槽位覆盖、类目命中、评分数流行度和预算价格邻近度。各权重统一在 `src/config/default.yaml` 的 `ranking.feature_weights` 中配置。
4. **风险约束 MMR 选前 10**：先按上述相关性计算每个候选的分数，再逐个选择“相关性高、且与已选商品不重复”的商品。商品间相似度使用 token-set Jaccard；有效 `mmr_lambda` 上限为 0.70，避免相近商品占满前 10 个位置。

整个路径为本地规则和词项计算：不调用网络、不训练模型、token 使用量为 0。LLM 重排接口仍兼容保留，但本次公开集实验不使用它。

## 灵感来源论文

本方案的核心灵感来自 Puthiya Parambath、Vijayakumar 与 Chawla 的论文：

> **Risk Aware Ranking for Top-k Recommendations** (2019)
> https://arxiv.org/abs/1904.05325

论文讨论了 Top-k 推荐不能只优化单个商品的预测分数：最终列表还应控制不确定性和重复风险。这里没有复现论文中的学习式风险模型，而是采用可审计的本地近似：保留相关性主导的特征分，并用 MMR 对已选商品的相似度施加惩罚。这样前 10 个结果既保留命中目标的机会，也避免同类近重复商品挤占列表。

## 配置与复现

关键参数位于 `src/config/default.yaml`：

| 参数 | 当前值 | 作用 |
| --- | ---: | --- |
| `rerank_top_n` | 20 | 对检索得到的前 20 个候选进行重排 |
| `mmr_lambda` | 1.0（有效上限 0.70） | 控制相关性与去重的平衡 |
| `profile_weight` | 0.15 | 画像先验的有界加分 |
| `use_field_aware_slot_coverage` | true | 启用标题/类目/属性/详情字段覆盖 |
| `feature_weights` | 见配置 | 各相关性特征的融合权重 |

在项目根目录运行：

```bash
$env:PYTHONHASHSEED='0'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
python scripts/run_public_eval.py --output results_rerank_risk_aware_mmr_profile.json
```

## 公开集结果

使用官方公开集 200 条与官方 evaluator 运行，结果写入 `results_rerank_risk_aware_mmr_profile.json`：

| 指标 | 当前重排结果 |
| --- | ---: |
| Hit Rate@10 | 0.905 |
| MRR | 0.706437 |
| MTTC | 3.535 |
| Efficiency | 0.7465 |
| TechnicalScore | 0.813731 |
| LLM tokens | 0 |

该结果在相同固定环境变量下重复运行，汇总指标一致。评测结果应以该 JSON 文件和官方 evaluator 的实际输出为准。

## 修改范围

- 主要逻辑：`src/ranking/reranker.py`
- 画像先验：`src/ranking/profile_prior.py`
- 参数配置：`src/config/default.yaml`
- 结果文件：`results_rerank_risk_aware_mmr_profile.json`

官方 `evaluator/` 未作重构或修改。
