# TechJam 2026 对话式电商搜索比赛 · 完整介绍

> 信息源：官方参赛包（problem statement、`docs/`、`evaluator/local_evaluator.py`、`data/public_set.jsonl`）
> 整理日期：2026-08-26

## 一、这是什么比赛

TechJam 2026 Conversational E-Commerce Search Challenge——对话式电商搜索 Agent 比赛。

任务一句话：**做一个 AI 购物助手，在最多 10 轮对话内，通过提问和推荐，让用户找到他真正想买的那件商品**（目标商品是隐藏的，Agent 看不到）。

主办方基于 Amazon Reviews 2023 构造"模拟用户 + 隐藏答案"的评测环境：每条会话背后是一个真实购买记录对应的商品，评测器扮演顾客，Agent 每轮可以问一个问题、给一批推荐，直到命中目标或 10 轮耗尽。

## 二、数据

| 数据 | 说明 |
|---|---|
| 商品目录 | 冻结的 50,000 件商品，Amazon Reviews 2023 的 Clothing_Shoes_and_Jewelry 类目 |
| 可见字段 | parent_asin、title、features、description、price、categories、details、average_rating、rating_number、store |
| 评分字段 | **只有 parent_asin** |
| 公开集 | 200 条带标签开发会话，本地调试用 |
| 私有集 | 800 条，组织方保留，最终评分用 |
| 会话构成 | 匿名用户画像 + 模拟对话（**非真实购物对话**，由评测器按隐藏意图卡模拟） |
| 用户画像 | purchase_frequency、average_prior_rating、rating_style、preference_tags、summary |

关键规则：目录**严格只读**，禁止修改或注入新商品 ID；公开集与私有集的用户、目标商品互不重叠。

## 三、任务与四大支柱

**I. 核心架构：意图路由与混合管线**

- 双轨路由：快速识别 Buying（想买）→ 高精度过滤轨道锁定硬约束；Browsing（随便看看）→ 多样化稠密检索轨道做跨类目匹配；
- 管线：多路检索（关键词 + 类目 + 向量相似度）→ LLM 语义排序。

**II. 对话策略：多轮场景演化**

- 动态状态机：信息累积（增量填槽）+ 意图突变（槽位擦除重写）；
- 主动引导：候选池过大时触发检索截断，主动生成结构化澄清问题。

**III. 自进化：动态上下文编程**

- 运行时适配：对话历史做个性化上下文蒸馏，更新短期会话状态和长期用户画像；
- 自适应编排：运行时重排工作流与策略。

**IV. 评估矩阵**

- Coverage（Hit Rate@K）：检索阶段目录召回；
- Precision（MRR / Top-K Hit Rate）：目标商品排到榜首的精度；
- Efficiency（MTTC）：平均几轮找到，惩罚多余对话负担。

## 四、Agent 接口（唯一要实现的服务契约）

```python
class Agent:
    def reset(self, session_id: str, user_profile: dict) -> None: ...
    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        return {
            "message": "Do you have a material preference?",
            "ask_attribute": "material",
            "recommendations": [{"parent_asin": "B000..."}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 30}
        }
```

要点：

- `ask_attribute` 只能是 10 种之一：`category / material / color / size / style / brand / budget / feature / use_case / other`，或 `null`；
- 推荐列表最多 100 条，**只取前 10 个目录内有效且不重复的 parent_asin 参与评分**；
- 推荐必须按最好到最差排序（影响 MRR）；
- `usage` 报告 token 数（不参与核心评分，属可行性指标）；
- 抛异常、非法输出、超时 → 按未命中处理。

## 五、评测器机制（赢的关键）

评测器 = 一个会按"隐藏意图卡"回答问题的模拟用户。

1. 评测器从目标商品元数据（标题、特性、详情、材质、颜色、价格）生成"意图卡"（硬约束 + 软偏好），对 Agent 隐藏；
2. 按场景发首条消息；
3. 每轮 Agent 返回 `message + ask_attribute + recommendations`；
4. 评测器检查 top-10 是否含目标商品——**包含即命中，会话立即结束**；
5. 未命中则按问的属性回答：从意图卡里挑未披露且匹配该属性的约束（最多 2 条）；无匹配则回"我没有额外的偏好"；
6. 10 轮未命中 → 判负。

**四种场景（固定比例 40/40/15/5）：**

| 场景 | 比例 | 首条消息特征 | 关键机制 |
|---|---|---|---|
| Buying | 40% | 一开始就透露类目 + 一个硬约束 | 快速过滤锁定 |
| Browsing | 40% | "我还在随便看看" | 提问挖掘软偏好 |
| Intent Override | 15% | 给一个偏好 | 第 3 或 4 轮突然改口，旧槽位必须重置 |
| Boundary | 5% | 普通开场 | 第一次问任何属性都会被回"你自己判断"——问就是浪费轮次 |

机制细节：

- **命中即结束** → 每一轮都应返回 top-10 推荐；
- **Intent Override 会话在改口前不能命中** → 死磕首轮偏好的 Agent 必挂。

## 六、指标与评分公式

```text
HitRate@10  = 成功会话数 / 总数
MRR         = 平均(1 / 目标商品排名)，未命中记 0
MTTC        = 平均(首次命中轮数)，未命中记 11
Efficiency  = clip((11 - MTTC) / 10, 0, 1)
TechnicalScore = 0.50 × HitRate@10 + 0.30 × MRR + 0.20 × Efficiency
```

基线（官方弱 BM25）：

| 指标 | 数值 |
|---|---|
| Hit Rate@10 | 0.125 |
| MRR | 0.068034 |
| MTTC | 9.81 |
| Efficiency | 0.119 |
| **TechnicalScore** | **0.10671** |

四种场景分别报告指标，某个场景崩了会在分项里暴露。

## 七、规则与限制

**In scope**：意图检测模块；异构检索路由（权重 / 动态截断 / 槽位衰减）；运行时自适应记忆层；LLM 排序阶段的提示词或本地打分逻辑。

**Out of scope**：UI/UX 开发；基础大模型训练或全参微调；外部向量数据库集群（**必须完全内存运行**）；多模态（只处理文本）。

**硬限制**：

- 每会话最多 10 轮，超了强制终止且该会话零分；
- 目录只读，禁止结构改动或伪造 ASIN；
- 输入已清洗；目录/价格/类目树静态不变；会话隔离、无并发压力。

**模型政策**：开发期可用任何合法 LLM API 或本地模型；**决赛评分可能禁用网络**——必须声明网络依赖并说明离线兜底；API 密钥只走环境变量，严禁提交进仓库。

## 八、资源

- 冻结目录 50,000 商品（GitHub Release 下载，附 SHA256 校验）；
- 200 条公开开发会话；
- 弱 BM25 起步 Agent（Python 标准库）；
- 确定性本地评测器（`evaluator/local_evaluator.py`）；
- Agent API 契约、评测配置、基线结果、提交规则文档。

## 九、交付物

1. **Devpost 文字描述**：方案如何解决题目、开发工具、API、库和框架、数据集；
2. **公开 GitHub 仓库**：结构良好、有注释的代码；README 含项目概述、安装说明、复现步骤、局限性反思、团队贡献；
3. **演示视频**：端到端演示（后端/NLP 赛道可用 API 调用或推理分析演示），YouTube 公开，链接放 Devpost；不得使用未经授权的第三方商标或版权内容。

提交规则还要求：`Agent` 入口文件 + 依赖清单 + 安装说明 + 简短技术报告（方法、模型选择、局限）+ 延迟/token/成本披露。

## 十、评审标准

| 标准 | 权重 | 看什么 |
|---|---|---|
| Technical Execution | 35% | 代码结构、架构、API/模型使用、demo 可靠运行 |
| Innovation & Problem Insight | 20% | 原创性、问题理解深度、方案是否直击要害 |
| Impact & Relevance | 20% | 对真实用户/场景的价值，超越比赛本身 |
| Feasibility & Practicality | 15% | 资源使用合理、架构可落地、不空谈 |
| Presentation & Communication | 10% | 决赛现场故事线与答辩深度 |

注意：**TechnicalScore 只占评审的一部分**——Innovation、Impact、交付质量占了另外一半以上。

## 十一、实战含义

1. **每轮必推 top-10**（命中即结束，早推早赢）；
2. **问对属性 = 直接挖出目标商品元数据里的约束**（问 material 可能直接拿到 "leather"）；
3. **Boundary 别问、Override 要重置、Browsing 要引导**；
4. 评测器源码在本地仓库，任何策略猜想都能用 `evaluator/local_evaluator.py` 验证。
