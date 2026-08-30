# 实验计划与结果记录

> 历史记录说明：以下”已完成实验”保留用于追溯熵、早停与 Override 的系统改进。
> **当前版本（三项优化后，2026-08-30）**公开集结果：HitRate@10=0.955、MRR=0.706129、MTTC=2.865、TechnicalScore=**0.852039**（`docs/THREE_OPTIMIZATIONS_20260829.md`）。
> **泛化警告**：与公开集零重叠的 400-ASIN 盲测（seed 20260830）TechnicalScore=0.750553，低于原版 BM25 的 0.760657；当前三项优化仍应视为实验方案。

## 已完成实验（消融分析）

### 实验设置

- 数据集：`data/public_set.jsonl`，200 条 session（Buying 80 / Browsing 80 / Override 30 / Boundary 10）
- 商品目录：50,000 件服装
- 无 LLM 重排（离线模式，`use_llm_ranker=False`）
- 运行脚本：`scripts/run_ablation_table.py`

### 结果总表

| 配置 | Buying HR | Browsing HR | Override HR | Boundary HR | Overall HR | MRR | MTTC | TechScore |
|------|-----------|-------------|-------------|-------------|------------|-----|------|-----------|
| Baseline (BM25 only) | 0.663 | 0.725 | 0.767 | 0.900 | 0.715 | 0.464 | 6.36 | 0.590 |
| + Hybrid (Dense) | 0.663 | 0.725 | 0.767 | 0.900 | 0.715 | 0.427 | 6.38 | 0.578 |
| + Entropy | 0.738 | 0.838 | 0.833 | 0.900 | 0.800 | 0.459 | 5.08 | 0.656 |
| + Early Stop (τ=0.3) | 0.888 | 0.900 | 0.833 | 0.900 | 0.885 | 0.536 | 3.44 | 0.755 |
| + Override Detection | 0.888 | 0.900 | 0.867 | 0.900 | 0.890 | 0.555 | 3.40 | **0.763** |
| Final (无 Early Stop) | 0.738 | 0.838 | 0.767 | 0.900 | 0.790 | 0.473 | 5.12 | 0.655 |

**第二轮消融（三项优化，2026-08-30；详见 `docs/THREE_OPTIMIZATIONS_20260829.md`）**

| 配置 | Overall HR | MRR | MTTC | TechScore |
|------|------------|-----|------|-----------|
| 更新仓库原始复测（Rerank + MMR 基线） | 0.905 | 0.706437 | 3.535 | 0.813731 |
| + 否定感知回复提纯 | 0.880 | 0.664893 | 3.660 | 0.786268 |
| + 动态属性权重（严格 Override，历史实验） | 0.935 | 0.688629 | 2.995 | 0.834189 |
| 最终：选择性 Override + Dense 风险门控 | **0.955** | 0.706129 | **2.865** | **0.852039** |
| 最终：选择性 Override + 强制字段 Dense | 0.920 | 0.546383 | 3.165 | 0.780615 |

### 关键结论

1. **Dense 单独加有害**：在无更好问题选择的情况下，Dense 引入噪音使 MRR 下降（0.464→0.427）。

2. **Entropy 是最大单步提升**：+0.078 TechScore。动态 coverage×entropy 让每轮问题都选最有信息量的属性，MTTC 从 6.36→5.08。

3. **Early Stop 是最大意外收益**：+0.099 TechScore，MTTC 从 5.08→3.44。候选池收敛后，继续问特定属性换来空回复；改问 `other`（通配）直接获取评估器主动披露的约束，消除无效轮次。

4. **Override Detection 精准修复 Override 场景**：Override HR 0.833→0.867，MRR 0.601→0.722，不影响其他场景。

5. **当前 "Final" 配置（无 Early Stop）次优**：Early Stop 已设为默认 `True`，最优 TechScore=0.763。

---

## 待探索实验

### 短期（预赛阶段）

#### E1: τ 超参数扫描
Early Stop 阈值目前固定 τ=0.3（来自 PDF 理论推导）。

| τ | 预期影响 |
|----|---------|
| 0.1 | 更激进停止，MTTC 更低，可能损失 HitRate |
| 0.3 | 当前最优（实验验证）|
| 0.5 | 较保守，更多问题，MTTC 略高但 HitRate 可能更稳 |

运行方式：修改 `src/dialogue/early_stop.py` 中 `TAU`，执行 `scripts/run_public_eval.py`。

#### E2: LLM 重排器（暂不采用）
> 当前方案不使用 LLM 重排。默认 `use_llm_ranker=False`，公开集 200 条结果为零 token、纯本地运行（TechScore=0.852164）。LLM 路径保留为兼容可选分支，但 prompt 信息量不足（仅 title），预期收益有限，暂不评测。

#### E3: slot_filter 扩展
> **部分完成**：动态属性权重已在 top-300→100 阶段对 material/color 等属性施加兼容度重排（`src/ranking/dynamic_attributes.py`）；BM25 `_apply_slot_filters()` 仍仅硬过滤 budget。进一步添加 material/color 硬过滤可进一步提升 Buying 场景 HitRate。

文件：`src/retrieval/bm25.py:_apply_slot_filters()`

#### E4: 用户画像接入
**当前实现**：`profile_prior.py` 在 rerank 阶段读取 `preference_tags` 并做有界本地加分；该画像先验保留在当前方案中。

### 中期（决赛阶段）

#### E5: 离线重排器（跨 network 禁用时）
替代 LLM 重排：`cross-encoder/ms-marco-MiniLM-L-6-v2`（~25MB，CPU 推理）。

需修改：`src/ranking/reranker.py` 增加 cross-encoder fallback 分支。

#### E6: BM25 slot_filter 扩展至全属性
Buying 场景 HR=0.888，仍有 ~11% miss。部分原因是 material/color slot 未做硬过滤。

#### E7: 查询改写
> **已完成**：Override 后选择性覆盖已通过 `src/dialogue/purification.py` + `override_carryover_confidence=0.35` 实现；旧槽降为低置信度弱证据，旧原句不会重新拼回查询。

---

## 实验管理规范

- 所有新实验结果保存为独立 `results_*.json`，不覆盖原始 `results.json`；当前 rerank 结果为 `results_rerank_risk_aware_mmr_profile.json`。
- 修改超参数前先在 `src/config/experiments/` 建独立 yaml
- 每次实验在本文件追加一行到结果总表
