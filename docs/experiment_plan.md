# 实验计划与结果记录

> 历史记录说明：以下“已完成实验”保留用于追溯熵、早停与 Override 的系统改进，不代表当前 rerank。当前方案为保留画像先验的字段感知精确约束覆盖与风险感知 MMR 重排，结果见 `results_rerank_risk_aware_mmr_profile.json`（TechnicalScore 0.813731）。

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

#### E2: LLM 重排器开启（联网环境）
> 非当前方案。当前 rerank 不依赖 LLM，且公开集结果使用零 token 本地重排获得。
当前 `use_llm_ranker=False`。开启后对 MRR 影响最大（31 个 session 的目标商品位于 rank 6-10）。

```bash
ANTHROPIC_API_KEY=... python scripts/run_public_eval.py --llm-rank
```

推荐模型：`claude-sonnet-4-6`（比 Haiku 语义理解更强，适合 listwise 排序）

#### E3: slot_filter 扩展
当前 `_apply_slot_filters` 仅过滤 budget。添加 material/color 硬过滤可提升 Buying 场景 HitRate。

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
当前查询 = `slot_text×2 + message + history`，未对 intent override 后做重置。
Override 后可重新构建不含旧槽的 clean query。

---

## 实验管理规范

- 所有新实验结果保存为独立 `results_*.json`，不覆盖原始 `results.json`；当前 rerank 结果为 `results_rerank_risk_aware_mmr_profile.json`。
- 修改超参数前先在 `src/config/experiments/` 建独立 yaml
- 每次实验在本文件追加一行到结果总表
