# 三项优化、消融与规则合规说明（2026-08-29）

## 基线与范围

本轮以远端最新提交 `53a2905` 为唯一基线重新实现，不复制旧工作树中的优化代码。官方入口契约保持 `starter.agent.Agent`，冻结商品目录只读，`evaluator/local_evaluator.py` 与 `data/public_set.jsonl` 未修改。

公开集原始复测（200 会话，BM25-only，`PYTHONHASHSEED=0`）为：HitRate@10 `0.905`、MRR `0.706437`、MTTC `3.535`、TechnicalScore `0.813731`。

## 优化一：否定感知的回复提纯

`src/dialogue/purification.py` 把一轮回答拆成三类已观察证据：

- 正向约束，例如 `blue`；
- 排除约束，例如 `not red`；
- 无偏好，例如 `I don't have a preference for color`。

例子：`I do not want red; blue instead.` 不再把 `red` 和 `blue` 一起送进检索。检索文本只保留 `blue instead`，会话内同时记录 `negative_slots.color={red}`。后续商品评分对红色商品施加排除风险，而不是把否定词当作正向关键词。

Intent Override 使用选择性覆盖：稳定类别保留；与新回答同属性的旧值直接替换；明确否定或无偏好的属性删除；其他旧属性降为 `0.35` 置信度，只进入查询一次且仍会被后续问题重新确认。旧原句不会重新拼回查询。这一过程不使用下一轮消息或隐藏标签。

## 优化二：动态属性权重贯穿 50,000→100→10

属性权重不是固定常数，也不是只调整 BM25/Dense 的总比例。每轮对每个当前属性 `a` 计算：

`raw_weight(a) = evidence_confidence × recency × (0.55 + 0.45 × pool_selectivity) × attribute_prior`

再对本轮全部有效属性归一化。这里：

- `evidence_confidence` 区分结构化明确回答与泛化正则提取；
- `recency` 让很旧的条件小幅衰减，但不会遗忘；
- `pool_selectivity` 看当前候选池中多少商品已经满足该值，越能区分商品的属性越重要；
- `attribute_prior` 只是很小的可审计先验，最终权重仍由当前累计证据和候选池重新计算。

动态结果作用在两个阶段：

1. **50,000→100**：累计类别信息提高 BM25 `categories/title` 字段权重；材质、颜色、尺寸、风格、用途、功能信息提高 `features/details/description` 字段权重。BM25 取回的前 300 个结果再按每商品属性兼容度重排，最后截取前 100。
2. **top-20→10**：原仓库风险感知 MMR 的 relevance 部分改用当前动态兼容度，并对命中否定条件的商品施加风险惩罚；MMR 的去重职责保持不变。

例如当前只有 `category=running shoes` 时，类别权重占主导；用户随后补充 `waterproof`，若前 300 中只有少量商品防水，`feature` 的选择性上升，它会自动获得更高权重；再补充 `not red` 后，红色商品得到排除风险。整个过程只看本轮以前已经提供的信息。

## 优化三：字段分离 Dense + 风险门控自适应融合

Dense 不再先拼接 `title + category + features` 后编码一个商品向量，而是分别建立：

- 身份向量：`title + categories + store`；
- 属性向量：`features + details + description`。

查询也分成类别/身份查询与属性查询。已知具体属性越多，属性相似度权重越高；类别明确时身份相似度权重更高。两个相似度分数在运行时加权，不拼接向量。

原始提交真正加载原版 Dense 后，公开集 TechnicalScore 为 `0.742997`，低于其 BM25-only 的 `0.813731`；当前选择性覆盖版本强制字段 Dense 时为 `0.780615`，也低于风险门控默认路径的 `0.852039`。因此默认加入可审计风险门控：

- Buying 始终使用精确 BM25；
- BM25 候选不少于 20 且已有稳定类别时，不调用 Dense；
- 只有词法候选不足，或无类别宽泛查询且 BM25 置信度很低时，才惰性加载 Dense；
- Dense 不可安装、模型不在离线缓存或加载失败时，自动回退 BM25。

门控打开 Dense 后，RRF 的来源权重由累计槽位数以及本轮两个结果列表的分数分离度共同确定。这里没有预测下一轮问题收益，也没有使用真实比赛中不可获得的未来分数。

## 完整消融结果

| 配置 | HitRate@10 | MRR | MTTC | TechnicalScore |
| --- | ---: | ---: | ---: | ---: |
| 更新仓库原始复测 | 0.905 | 0.706437 | 3.535 | 0.813731 |
| + 回复提纯 | 0.880 | 0.664893 | 3.660 | 0.786268 |
| + 动态属性（严格 Override，历史实验） | 0.935 | 0.688629 | 2.995 | 0.834189 |
| 最终选择性 Override + Dense 门控 | **0.955** | 0.706129 | **2.865** | **0.852039** |
| 最终选择性 Override + 强制字段 Dense | 0.920 | 0.546383 | 3.165 | 0.780615 |

回复提纯单项在公开集下降的主要原因是：公开评测器不生成 `not red` 这类真正的排除回答，而 Intent Override 的旧软偏好恰好来自目标商品。选择性覆盖减少了全清空造成的信息断裂，同时避免把旧原句继续当作强查询。Dense 则按强制对照结果选择风险门控，而非为了形式上使用模型而牺牲分数。

## 未见商品与过拟合审计

先从冻结目录生成 400 个与公开目标 ASIN 零重叠的开发验证样本（seed `20260829`）。该集合用于发现严格 Override 的问题，因此不能再算最终盲测。修改为选择性覆盖后，它从原版 BM25 的 `0.766886` 小幅变为 `0.767773`。

随后冻结代码和参数，再生成第二组与公开集、开发验证集都零重叠的 400-ASIN 最终盲测（seed `20260830`）：

| 最终盲测配置 | HitRate@10 | MRR | MTTC | TechnicalScore |
| --- | ---: | ---: | ---: | ---: |
| 原版 BM25 | **0.875** | **0.587855** | 3.660 | **0.760657** |
| 当前选择性覆盖版本 | 0.8725 | 0.552678 | **3.575** | 0.750553 |

结论是：当前版本在最终盲测上效率略好，但命中率、MRR 和综合分没有超过原版。因此公开集 `0.852039` 只能作为公开集内结果，不能声称已经证明泛化。本项目保留该负结果，后续参数选择应在训练/验证折上完成，并只对新的冻结测试集评测一次。

## 赛题合规检查

- 未修改官方 evaluator、公开集、starter 入口或冻结目录；
- 没有训练 RL 策略、微调基础模型或用 LLM 生成商品 ASIN；
- Dense 使用本地 `all-MiniLM-L6-v2` 推理，没有训练；
- 向量索引为本地 `.npz`，相似度计算完全在内存中，无外部向量数据库；
- 默认路径不调用 LLM，公开集 token 使用量为 0；
- 所有推荐仍由冻结目录中的合法 `parent_asin` 组成；
- 决策只使用当前和历史已观察信息，不使用未来问题、未来收益或 ground truth。

## 复现

```bash
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python -m pytest -q

python scripts/run_public_eval.py \
  --catalog data/catalog.jsonl \
  --dataset data/public_set.jsonl \
  --output /tmp/final-optimized.json

python scripts/run_public_eval.py \
  --catalog data/catalog.jsonl \
  --dataset data/public_set.jsonl \
  --force-dense \
  --output /tmp/forced-dense.json

python scripts/run_three_optimization_ablation.py \
  --catalog data/catalog.jsonl \
  --dataset data/public_set.jsonl \
  --output /tmp/three-optimization-ablation.json
```
