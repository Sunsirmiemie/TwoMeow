# TwoMeow 基线稳定化说明

## 1. 状态与边界

- 工作分支：`stabilize-baseline`
- 基准提交：`64a7db54d5185a835fa8a850f14b2f22cb1d900e`
- 记录日期：2026-08-27
- 当前状态：改动仅保存在本地工作区，尚未提交，也未推送远端
- 本轮目标：把入口、配置、关键数据边界、评测分析与安装/测试方式收敛成一条可信且可复现的基线

本轮没有覆盖 `results.json` 或 `results_optimal.json`，也没有宣称重新验证过 Dense 或真实 LLM 路径。应用代码以最小兼容改动为主，旧入口仍可使用。

## 2. 为什么要做这次稳定化

原基线存在几类会影响复现和后续优化的问题：

1. 仓库有两套 `Agent` 实现，不同入口可能执行不同逻辑。
2. 商品价格既可能是数值，也可能是 `"—"`、`"from 12.99"` 等文本；带预算搜索会把字符串与浮点数直接比较。
3. `default.yaml` 看似是统一配置源，但运行时仍有多处硬编码，命令行默认值也会无条件覆盖 YAML。
4. LLM token usage 是进程累计值，而评测器需要单次响应的 usage。
5. 失败分析读取不存在的 `first_hit_rank`，与评测器实际输出的 `hit` / `best_rank` 不一致。
6. 缺少正式的 Python 版本范围、可选依赖分组和安装后制品测试。
7. README 将历史最佳结果写成“当前成绩”，没有区分历史制品与本轮可复现结果。

因此，本轮先修复“结果是否可信、入口是否唯一、配置是否真正生效、环境是否可装”这些基础问题，再继续做召回与对话策略优化。

## 3. 具体改动

### 3.1 唯一 Agent 实现与兼容入口

`src.agent.orchestrator.Agent` 现在是唯一实现：

- `starter.agent.Agent` 继续 re-export 该类，满足官方评测入口；
- `agent.agent.Agent` 改为兼容 re-export，不再维护第二套实现；
- `run_eval.py` 与 `scripts/run_public_eval.py` 直接使用 canonical Agent；
- 测试断言三个公开入口都指向同一个类对象。

这保留了旧 import 路径，同时消除了实现漂移。

### 3.2 价格标准化与预算行为

目录装载时统一处理价格：

- 可转换且有限的值（例如 `29.99`、`"29.99"`）转为 `float`；
- 缺失值、不可解析文本、`NaN` 和无穷值转为 `None`；
- buying 场景应用预算过滤时，已知价格只保留 `price <= budget` 的商品；
- 未知价格保留在候选中，避免因为目录字段不规范而丢失潜在命中。

这是保守策略：`"from 12.99"` 当前不会猜测为 12.99，而是按未知价格处理。

### 3.3 YAML 配置真正进入运行时

新增 `src.config.load_config()`，读取嵌套的 `src/config/default.yaml`，再映射为 Agent 使用的扁平配置。当前接线覆盖：

- 检索数量与 BM25 字段权重；
- RRF 常数、基础融合权重和按槽位数量变化的 browsing 权重（0 槽 50/50、1 槽 60/40、≥2 槽使用基础权重 75/25）；
- Dense 开关、模型名和 batch size；
- LLM ranker 开关、模型名和 rerank top-N；
- entropy threshold、动态熵最小候选池；
- rerank pool 的槽位、候选数和截断阈值；
- dynamic entropy、early stop 与 override detection 开关。

合并语义如下：

1. 每次调用都从 YAML 读取并深拷贝默认值；
2. 调用方传入的是扁平 override，显式值最后生效；
3. `field_weights` 支持局部深合并，例如只覆盖 `title` 不会清空其余字段；
4. 其他列表或映射同样复制后再使用，不会反向修改调用方对象；
5. CLI 只有在用户显式提供 `--no-dense` 或 `--llm-rank` 时才生成 override；未传参数时保留 YAML 默认值。

### 3.4 单响应 token usage

`Ranker` 在每次 `rerank()` 开始时把 usage 归零，并只记录该次 API 响应的输入/输出 token：

- 连续两次调用不会累加；
- LLM 禁用、候选为空或 API 在收到响应前失败时返回 0/0；
- 已收到响应但返回内容无法解析时，保留本次响应的 usage，并回退到原检索顺序；
- 测试通过注入 fake client 验证，不访问真实网络。

### 3.5 失败分析对齐评测器 schema

失败分析现在以官方评测器实际字段为准：

- near miss：`hit is True` 且 `best_rank` 位于指定闭区间，默认 6–10；
- complete miss：`hit is False`；
- `scripts/inspect_failures.py` 输出 `sample_id` 与 `best_rank`。

### 3.6 打包、依赖与仓库卫生

- 新增 `pyproject.toml`，支持 Python `>=3.10,<3.13`；
- 核心依赖：`numpy>=1.26,<3`、`PyYAML>=6,<7`；
- 可选 extras：`dense`、`llm`、`test`；
- `requirements.txt` 是 Dense、LLM、测试和构建工具的完整聚合环境；
- wheel 包含 `src/config/default.yaml`，并提供 canonical、starter 和 legacy 三个入口；
- 新增 `.gitignore`，阻止新增的 `graft/`、Python/pytest 缓存、虚拟环境、`.env`、`.idea/`、`.embed_cache/`、`*.egg-info/`、`build/`、`dist/`、指定 catalog 副本和 `*.zip` 被跟踪。

注意：这些规则只影响尚未跟踪且匹配上述模式的文件，不会自动解除仓库中已有大文件、catalog、zip 或 IDE 文件的跟踪状态。

### 3.7 测试补强

新增或扩展的测试覆盖：

- 非数字价格与预算过滤；
- canonical Agent 入口一致性；
- YAML 默认值、扁平 override、局部字段权重合并与调用方对象隔离；
- 配置项是否真正传到 BM25、Dense、RRF、对话熵阈值和 rerank pool；
- CLI 仅生成显式 override；
- 单响应 token usage、禁用/空候选/API 异常/无效 JSON 回退；
- `hit` / `best_rank` 失败分析；
- wheel 解压后的隔离导入/构造中，入口、评测器模块和 YAML 制品可用。

## 4. RED → GREEN 记录

本轮按行为契约逐项先写失败测试，再实现最小修复：

| 契约 | RED 现象 | GREEN 结果 |
| --- | --- | --- |
| 非数字价格可参与带预算检索 | 字符串价格与浮点预算比较触发 `TypeError` | 非法价格归一为 `None`，未知价格保留 |
| 失败分析读取官方 schema | `first_hit_rank` 不存在，命中/漏召分类错误 | 使用 `hit` 与 `best_rank` |
| 所有公开入口使用同一 Agent | legacy 与 canonical 是不同类 | 三个入口均 re-export canonical 类 |
| YAML 是运行时默认配置 | 没有公开 loader，多个默认值未进入组件 | loader、CLI 和各组件完整接线 |
| token usage 属于单次响应 | 第二次调用包含第一次 token | 每次调用独立重置和记录 |
| 安装制品可独立运行 | 无标准项目元数据和 YAML 制品约束 | wheel 解压后的隔离导入/构造测试通过 |

## 5. 验证结果

### 5.1 自动化测试与制品隔离

独立验证阶段在三个受支持的 Python 次版本上运行完整套件：

| Python | 结果 |
| --- | --- |
| 3.10 | `74 passed` |
| 3.11 | `74 passed` |
| 3.12 | `74 passed` |

统一命令：

```bash
python -m pytest -q
```

其中制品测试会构建 wheel，把它解压到临时目录，移除仓库路径与 editable-import finder，再验证：

- `src.agent.orchestrator.Agent`、`starter.agent.Agent`、`agent.agent.Agent` 均来自该 wheel 且身份一致；
- `evaluator.local_evaluator` 可从制品导入；
- `default.yaml` 已打包；
- tiny catalog 可以创建 Agent。

同时通过：

- `git diff --check`；
- `results.json` 与 `results_optimal.json` 相对基准提交逐字节不变；
- 评测输出写入 `/tmp`，未写入仓库结果文件；验证产生的仓库根目录 `build/` 与 `*.egg-info/` 已在检查后精确清理，最终不存在根目录 `build/`、`dist/` 或 `*.egg-info/`。

### 5.2 本轮可复现 BM25-only 公共集基线

为避免覆盖仓库结果，输出必须显式写到临时目录：

```bash
python scripts/run_public_eval.py \
  --no-dense \
  --output /tmp/twomeow_bm25_stabilize_20260827.json
```

2026-08-27 在 Python 3.12.13、PyYAML 6.0.3、NumPy 2.4.6 下重跑 200 个公开 session，得到：

| 指标 | 数值 |
| --- | ---: |
| sample_count | 200 |
| HitRate@10 | 0.865 |
| MRR | 0.552575 |
| MTTC | 3.65 |
| Efficiency | 0.735 |
| Recommended Technical Score | 0.745273 |
| Prompt / Completion Tokens | 0 / 0 |

该结果只验证 BM25-only 路径；未加载 sentence-transformers，也未调用 LLM。

### 5.3 仓库内历史结果文件的口径

两个文件都保留原样：

| 文件 | 定位 | HitRate@10 | MRR | MTTC | TechScore |
| --- | --- | ---: | ---: | ---: | ---: |
| `results_optimal.json` | 仓库已有的历史最佳记录 | 0.890 | 0.554536 | 3.405 | 0.763261 |
| `results.json` | 较早、非 canonical 的旧结果，不再作为当前基线 | 0.790 | 0.471744 | 5.12 | 0.654123 |

历史最佳文件不是本轮新生成的结果。项目原有说明把它记录为无 LLM 实验，但 JSON 本身没有足够的运行配置 provenance 可独立证明这一点；本轮新鲜、可复现且环境口径明确的数字是上一节的 BM25-only 基线。

## 6. 已知限制与延期工作

1. **价格语义仍然保守**：`"from 12.99"` 被视为未知，没有解析币种、区间或起售价。
2. **配置只有接线与合并，没有 schema 校验**：非法类型、负阈值、缺字段或权重长度错误可能在下游才暴露。
3. **旧 `agent/` 模块树仍保留**：入口已经统一，但旧实现依赖的辅助模块尚未删除，以降低本轮兼容风险。
4. **已跟踪的大文件尚未清理**：目录副本、Dense cache、participant kit zip 和 `.idea` 文件仍在 Git 历史/索引中；新增 ignore 规则不会自动解除跟踪。
5. **Dense 的严格离线行为未验证**：sentence-transformers 可能在模型未缓存时尝试下载；本轮没有新跑完整 Dense 评测。
6. **真实 LLM 路径未验证**：仅使用 fake client 验证协议、usage 和回退；没有调用 Anthropic API。
7. **依赖尚未锁定到哈希级别**：版本范围可安装，但不同时间解析出的传递依赖可能变化。
8. **BM25 查询仍是 OR 组合**：召回较宽，槽值重复在去重后不会形成额外词频权重，需要单独实验 AND/短语/分字段策略。
9. **熵公式与论文口径仍需校对**：当前归一化分母使用候选值数量；需要和属性取值域口径做对照实验。
10. **early stop 的 `other` 策略仍需验证**：低信息增益时询问 `other` 是启发式设计，是否优于停止提问或返回结果要用分场景消融确认。

## 7. 下一步建议

按优先级推进：

1. **建立唯一的官方基线流程**：在固定环境中分别跑 BM25-only 与可离线加载的 Dense，保存命令、依赖快照、模型来源和新结果文件；明确哪个结果可以发布。
2. **补配置校验与错误信息**：对权重、阈值、路径、模型与布尔开关做启动时校验，避免静默退化。
3. **围绕 200 条公开集做检索实验**：优先比较 BM25 OR/AND/phrase、槽值字段加权与 unknown-price 策略，所有实验输出到新文件，不能覆盖历史制品。
4. **校正对话策略口径**：对熵归一化和 `other` early-stop 做分场景消融，重点观察 MTTC 与 HitRate 的交换关系。
5. **完成仓库减重**：确认数据分发要求后，解除已跟踪的 IDE、cache、重复 catalog 和 zip；如需保留大文件，改用发布制品或 LFS。
6. **生成依赖锁与离线安装包**：为 Python 3.10–3.12 生成可审计锁文件，并验证无网络安装和模型加载。
7. **最后再删除旧模块树**：先对外宣布 canonical import，再移除不再使用的 legacy 辅助模块。

## 8. 当前 Git 状态

本说明完成时仍遵循以下边界：

- 分支已创建：`stabilize-baseline`；
- 未创建 commit；
- 未设置或推送远端分支；
- `results.json` 与 `results_optimal.json` 未改动；
- 本轮记录的 BM25-only 输出保存在 `/tmp/twomeow_bm25_stabilize_20260827.json`，从未写入仓库内的结果文件。
