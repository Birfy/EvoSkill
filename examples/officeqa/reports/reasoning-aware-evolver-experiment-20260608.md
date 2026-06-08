# Reasoning-Aware Evolver 实验报告（2026-06-08）

## 背景

上轮实验（2026-06-07）发现 iter-skill-11 在 100 条 holdout 上仅得 76%（基线 86%），退化 10 个点，根因是：
1. Skill description 命令式写法注入 system prompt 干扰全局行为
2. DeepSeek 主动触发率 0%
3. Evolver 生成的 Skill 全部是门控型（gate skill），无法修复计算公式错误

本轮针对 Evolver 本身做了两处改动，重新跑一轮演化后评估。

## 改动

### Change 1：向 Proposer 传递 Agent Reasoning

修改 `src/loop/helpers.py` 的 `build_proposer_query()`，在每条失败样本中额外附上：

```
Agent Reasoning (chain-of-thought before final answer):
<compacted agent reasoning, ≤1800 chars>
```

**目的**：让 Proposer 看到 Agent 为什么出错（推理过程中的具体错误步骤），而不只是最终错误答案的形状。

### Change 2：Skill Generator 两阶段提示

修改 `src/agent_profiles/skill_generator/prompt.py`，在系统提示中加入两个显式阶段：

- **Phase 1 — Root-Cause Analysis**：读取 Proposer 的根因分析和 Agent 推理，判断错误类型（公式错误、行/列选错、时间窗边界、格式错误等），并映射到对应的 Skill 类型（计算型、门控型、格式型、选择型）。
- **Phase 2 — Skill Design**：根据 Phase 1 的结论写 Skill，禁止默认生成门控型。

核心指令：
> "Do not default to a gate skill just because gate skills are familiar. If the agent reasoned through the correct steps but applied the wrong formula, a gate that says 'check before computing' will not fix it — only a computation skill that specifies the exact formula will."

## Evolver 运行

- 数据集：93 条训练轨迹（DeepSeek-V4-Flash，无 Skill 基线）
- 训练集基线（Evolver 内部 BT 评分）：83.87%（78/93）；直接 score_answer() 评分：89.2%（83/93）
- 迭代数：12 轮
- 最优节点：frontier-distilled（Judge 估计分 86.74%）

### 生成的 Skill（frontier-distilled）

| Skill | 类型 | 触发条件 |
|---|---|---|
| `column-scope-verification` | 选择型 | 问题请求某实体的聚合指标，但表格列头含子类限定词（如 "Railroad Account" vs "总余额"） |
| `derived-statistic-verification` | **计算型** | 问题要求命名统计操作（移动平均、CAGR、标准差、百分位数、Tukey方法等） |
| `source-revision-authority` | 验证门 | 多个源文件包含同一 entity×period×metric 的重叠数据，需交叉验证修订版本 |
| `table-title-verification` | 选择型 | 源文件含多个命名表格，需从目录匹配正确 table title |

与上轮 iter-skill-11 对比：本轮出现了 `derived-statistic-verification`（计算型），是 Change 2 两阶段设计的直接效果——Evolver 识别到计算错误根因后选择了计算型 Skill，而非门控型。

## 评估结果

在同一批 193 条轨迹（100 holdout + 93 train）上分别跑新 Skill 版和纯基线版进行对比：

| 数据集 | 新 Skill | 基线 | 差值 | Skill 触发率 |
|---|---|---|---|---|
| Holdout 100（未见任务） | **89.0%** | 86.0% | **+3.0%** | 53% |
| Train 93（训练任务） | **90.3%** | 89.2% | **+1.1%** | 63% |
| 合计 193 | **89.6%** | 87.6% | **+2.0%** | 58% |

与上轮 iter-skill-11（Holdout 76%，退化 10%）相比，本轮 Holdout +3.0%，完全扭转。

### Skill 触发分布（193 条）

| Skill | 触发次数 |
|---|---|
| derived-statistic-verification | ~45 |
| source-revision-authority | ~35 |
| table-title-verification | ~18 |
| column-scope-verification | ~14 |

触发率从上轮的 3% 提升到 58%，主要原因是 description 改为被动标签格式（"Invoke when..."）后模型能正确识别触发条件。

## Skill 质量分析

### 有效场景

- `derived-statistic-verification`：成功拦截了中间步骤精度损失、公式简化未验证等计算错误
- `table-title-verification`：在多表格文件中引导模型扫描目录再定位表格，减少选错表的错误
- `column-scope-verification`：防止将子账户列（"Railroad Account"）误当总体指标列

### 问题：`source-revision-authority` 过触发

**根因**：触发条件"多个文件有重叠数据"在 OfficeQA 多文件题中几乎普遍满足，导致触发率过高。Step 5 的"后发布优先"规则（prefer later publication date）会强制模型去找"修订版"，但部分题目本来就该用特定年份的原始数据。

**案例**：
- UID0193：Fish imports data，Skill 引导用 Oct 1939 bulletin 作为"权威版本"，但正确答案在更早的 bulletin 里
- UID0020：KL divergence，跨文件交叉比对后读到略不同的数值，导致精度丢失（0.00266 vs 正确的 0.00262）
- UID0058：4 个 Skill 同时触发互相干扰，column 选择被错误限制

### 未覆盖的根因

| 失败类型 | 说明 |
|---|---|
| VaR / 指数平滑（EWM）公式错误 | `derived-statistic-verification` 触发列表未包含 VaR、EWM，模型不识别触发条件 |
| 输出格式错误（数字列表格式） | Evolver 训练集中此类失败只有 1 条，信号太稀疏无法归纳 |
| 模型随机性 | 2/8 训练集退化案例是相同问题不同运行结果不同，非 Skill 因素 |

## Evolver 根因识别准确性

Evolver 在 12 轮迭代中对训练集失败的根因分析：

| 根因 | Evolver 是否识别 | 准确度 |
|---|---|---|
| 跨文件历史数据修订未验证 | ✅ 是（iter-1、iter-12） | 准确，有具体 bulletin 年份和值差异的分析 |
| 命名统计操作被替换且精度损失 | ✅ 是（iter-6、iter-9 成功案例） | 准确，识别出 CMA vs 均值的等价性验证缺失 |
| 多表格文件中 table title 匹配缺失 | ✅ 是（从成功案例归纳，iter-9） | 准确，从 3 个成功轨迹归纳出共同机制 |
| 列头子类限定词误选 | ✅ 是（iter-11） | 准确，识别出 "Railroad" 子账户 vs 总体的范围问题 |
| VaR / 指数平滑公式 | ❌ 否 | 训练集中各只有 1 条，无法归纳通用 Skill |
| source-revision-authority 副作用 | ❌ 否 | Judge regression sample 未覆盖被该 Skill 干扰的题型 |

## 对比汇总

| 版本 | Holdout 正确率 | 训练集正确率 | Skill 触发率 | 备注 |
|---|---|---|---|---|
| 纯基线（无 Skill） | 86.0% | 89.2% | — | DeepSeek-V4-Flash |
| iter-skill-11（旧） | 76.0% | — | ~3% | Description 命令式，system prompt 污染 |
| frontier-distilled（本轮） | **89.0%** | **90.3%** | **58%** | Description 被动标签 + 两阶段 Evolver |

## 后续方向

1. **收窄 `source-revision-authority` 触发条件**：加入"仅当问题未指定来源且存在值冲突时才应用后发布优先规则"的保护逻辑
2. **扩展 `derived-statistic-verification` 覆盖范围**：在触发条件中补充 VaR、CVaR、指数平滑（EWM/EWMA）、Holt-Winters 等
3. **Judge regression sample 改进**：增加对 source-revision-authority 类 Skill 的对抗性测试案例，使 Evolver 能检测到过触发副作用
4. **换更强模型验证**：本轮用 DeepSeek-V4-Flash，换 Claude Sonnet 或 DeepSeek-V4-Pro 跑基线，看 Skill 对更强模型的效果
