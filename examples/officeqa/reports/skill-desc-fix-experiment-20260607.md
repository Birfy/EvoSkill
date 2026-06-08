# OfficeQA Skill Description Fix — Experiment Report (2026-06-07)

## 背景

Evolver 生成的 iter-skill-11（含三个 Skill）在 100 条 holdout 上得分 **76/100**，低于无 Skill 纯基线的 **86/100**，退化 10 个点。

## 根因分析

1. **Skills 从未被调用**：100 条 holdout 里 `skill_calls=[]`，Skill 逻辑完全没有执行。
2. **Description 字段干扰全局行为**：Skill frontmatter 的 `description` 被 Claude Code 注入到每次会话的 system prompt，但内容是命令式行为规则（`"Before binding... inspect ALL sources..."`），相当于全局修改模型行为，导致 56 道本来答对的题出现格式/计算错误。
3. **DeepSeek 不主动检查 Skill**：即使 system prompt 包含 `"If skills are listed, check trigger conditions before starting work"`，DeepSeek 也直接忽略，从不主动调用 Skill tool。

## 修复方案与实验结果

| 版本 | 修改内容 | 100条Holdout | 193条合计 | Skill触发率 |
|---|---|---|---|---|
| iter-skill-11（旧） | 命令式 description | 76/100 (76%) | — | 0% |
| 纯基线 | 无 Skill | 86/100 (86%) | 169/193 (88%) | — |
| 固定 description（无提醒） | 改为 `"Invoke when [condition]"` | 87/100 (87%) | — | 0% |
| Skill 提醒（query末尾追加） | 固定描述 + query 内提醒检查 | 83/100 (83%) | 166/193 (86%) | 3% |

## 关键结论

1. **Description 格式决定性重要**：命令式描述（"Before X do Y"）比不写 Skill 还差；改为被动标签（"Invoke when..."）后，仅凭减少干扰就恢复到 87/100（+1 vs 基线）。

2. **Skill 从未主动触发**：56 道多文件题一次 Skill 调用都没有。Description 变成被动标签后，system prompt 干扰消除，正确率略微提升，但 Skill 逻辑本身贡献为零。

3. **Query 内强制提醒效果有限**：加了提醒后模型确实会逐条检查触发条件（reasoning 中可见），但触发率仍只有 3%，且触发后未带来净改善，反而因执行 Skill 流程分散注意力导致格式错误（净 -3 点）。

4. **DeepSeek 对 Skill 机制不友好**：即使强制检查，触发率极低；触发后也未显著提升准确率。Skill 框架对 DeepSeek 的 ROI 接近零。

## 后续方向

- **换更强的模型**测试 Skill 机制（如 Claude Sonnet），验证是否是 DeepSeek 的模型能力瓶颈。
- **改进 Evolver**：Skill 生成的测试集应直接检验 Skill 调用后是否带来正确率提升，而不只是检验 Skill 格式合法性。
- **重新设计 Skill 触发条件**：当前触发条件描述（≥2 files）对模型来说过于隐式，需要更明确的 task pattern 或 few-shot 示例。
