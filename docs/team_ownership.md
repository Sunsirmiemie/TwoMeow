# 团队分工与模块归属

## 模块负责人

| 模块 | 文件 | 负责人 | 说明 |
|------|------|--------|------|
| 场景路由 & 状态管理 | `src/agent/router.py` `src/agent/state.py` `src/dialogue/override.py` | TBD | 意图分类、槽提取、Override 重置 |
| BM25 检索 | `src/retrieval/catalog.py` `src/retrieval/bm25.py` | TBD | SQLite FTS5 索引、字段权重、slot_filter |
| Dense 检索 | `src/retrieval/dense.py` | TBD | sentence-transformers 嵌入、缓存策略 |
| 混合检索 & 融合 | `src/retrieval/hybrid.py` `src/ranking/scorer.py` | TBD | RRF 融合、Buying/Browsing 轨道权重 |
| 澄清策略 | `src/dialogue/entropy.py` `src/dialogue/question_policy.py` `src/dialogue/early_stop.py` | TBD | 动态熵打分、Early Stop τ=0.3 |
| 排序器 | `src/ranking/reranker.py` `src/ranking/profile_prior.py` | TBD | LLM 重排、用户画像接入（stub） |
| 评测 & 分析 | `src/evaluation/` `scripts/` | TBD | 消融实验、失败分析、脚本维护 |
| 编排 & 接口 | `src/agent/orchestrator.py` `starter/agent.py` | TBD | 主流程、官方接口兼容 |

## 代码规范（来自目录结构建议）

- 单个业务 Python 文件控制在 **200–250 行**以内
- 一个文件只承担一个清晰职责
- 禁止超过 500 行的"万能 agent.py"
- 不允许模块间循环 import
- 所有关键超参数进入 `src/config/default.yaml`
- 所有公共函数必须有 type hints
- 核心策略写简短 docstring
- 不允许把实验逻辑散落在业务代码（用 `scripts/` 隔离）
- 官方 evaluator 文件视为外部依赖，不得重构

## 待认领任务

以下功能已有 stub，需要负责人跟进实现：

| 任务 | 文件 | 优先级 |
|------|------|--------|
| slot_filter 扩展（material/color 硬过滤） | `src/retrieval/bm25.py` | 高 |
| 用户画像个性化 | `src/ranking/profile_prior.py` | 中 |
| 离线 cross-encoder 重排 | `src/ranking/reranker.py` | 中（决赛） |
| τ 超参数扫描 | `src/dialogue/early_stop.py` | 低 |

## 提交前检查清单

- [ ] `starter/agent.py` 正确指向 `src.agent.orchestrator.Agent`
- [ ] `.env.example` 存在，`.env` 未提交
- [ ] `data/catalog.jsonl` 在 `.gitignore` 中（大文件）
- [ ] 所有测试通过：`python tests/test_*.py`
- [ ] `results.json` 已更新为最优配置的分数
- [ ] `docs/` 文档与实际代码保持同步
