# 消融实验结果整理

## 1. 说明

本文档集中整理当前项目里已经形成结论的几类消融实验，重点突出分支/编码器消融结果，方便后续写论文、答辩或汇报时直接引用。

当前项目的主实验配置已经切换为“去掉 `uav_04` 后的四数据集版本”，但现有较完整的消融结论主要来自此前的五 UAV 阶段。因此本文按下面的方式组织：

- **分支/编码器消融**：保留历史五 UAV 编码器消融的核心汇总指标，重点看 `transformer_only`、`gcn_only`、`mlp_only`、`random_graph`、`full` 五种配置。
- **多 UAV 策略消融**：总结 group normalization、group threshold、group embedding 等策略层面的已知结论。
- **`uav_04` 定向修复消融**：整理针对最难域 `uav_04` 的三轮修复实验。
- **当前四数据集结论**：说明为什么当前主线实验最终选择去掉 `uav_04`。

需要说明的是：历史五 UAV 的原始 `encoder_ablation_5_20260502_1729` 结果目录当前不在工作区中，因此下面的分支消融表保留的是已经整理过的核心汇总指标。

## 2. 分支/编码器消融（重点）

### 2.1 配置含义

| 配置 | 含义 |
| --- | --- |
| `transformer_only` | 仅保留时序分支，关闭图分支。 |
| `gcn_only` | 仅保留图分支，关闭 Transformer 时序分支。 |
| `mlp_only` | 不走时序分支也不走图分支，仅使用输入投影后做 MLP/池化。 |
| `random_graph` | 同时保留时序与图分支，但把行为图从真实 kNN 图替换成随机图。 |
| `full` | 时序分支 + 图分支 + learned gate 融合的完整模型。 |

### 2.2 总体结果

| 配置 | ID micro-F1 | ID macro-F1 | ID mAP | OOD AUROC | OOD F1 | Macro-UAV OOD F1 | Worst-UAV OOD F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `transformer_only` | 0.9338 | 0.8299 | 0.9010 | 0.9022 | 0.8184 | 0.4762 | 0.0391 |
| `gcn_only` | 0.9118 | 0.8240 | 0.8993 | 0.7122 | 0.0770 | 0.3618 | 0.0026 |
| `mlp_only` | 0.9360 | 0.8019 | 0.9213 | 0.8881 | 0.7645 | 0.4809 | 0.0553 |
| `random_graph` | 0.9303 | 0.8202 | 0.9114 | 0.8881 | 0.8146 | 0.5181 | 0.0079 |
| `full` | 0.9317 | 0.8382 | 0.8950 | 0.8707 | 0.8065 | 0.4849 | 0.0106 |

### 2.3 分支消融结论

1. **`transformer_only` 是当前最强的开放集主力配置。**  
   它在 `OOD AUROC` 和 `OOD F1` 两项最关键的 OOD 指标上都最好，说明当前场景下真正承担主要判别能力的是时序分支。

2. **`mlp_only` 的已知类分类能力不差，但 OOD 明显不如 `transformer_only`。**  
   `mlp_only` 的 `ID micro-F1` 和 `mAP` 反而略高，但 `OOD F1`、`OOD AUROC` 都落后，说明它更像一个强 ID 分类基线，而不是更好的开放集检测器。

3. **`gcn_only` 基本失效。**  
   它的 `OOD F1=0.0770`，已经接近不可用，说明单靠行为图分支无法支撑当前异构多 UAV 开放集检测。

4. **`random_graph` 没有像预期那样崩掉，说明“有无图结构”不是唯一决定因素。**  
   `random_graph` 的整体 `OOD F1=0.8146`，甚至接近 `transformer_only`，但最差 UAV 依然很差，说明当前图分支的稳定收益并不明确，图结构本身也还没有形成可靠增益。

5. **`full` 没有稳定超过 `transformer_only`。**  
   这说明在当前任务上，完整的时序+图融合还没有把图分支的潜力真正释放出来；现阶段更像“复杂度更高，但收益不稳定”。

### 2.4 关于“纯 Transformer”和“纯 MLP”的结论

如果目标是**开放集检测/异常检测整体效果**，优先选 `transformer_only`。  
如果目标只是**已知类分类精度**，`mlp_only` 有一定竞争力，但它不是当前更优的 OOD 配置。

## 3. 多 UAV 策略消融

这一部分主要关注 normalization、threshold、group embedding 这些“训练/校准策略”，而不是主干分支本身。

### 3.1 已知关键结论

1. **group normalization 是最稳定的增益来源。**  
   它能减轻不同 UAV 之间特征尺度和样本量差异带来的偏置，是异构多 UAV 场景下最可靠的基础设置。

2. **group threshold 往往优于只用 global threshold。**  
   尤其在某些 UAV 验证窗口数量较少、分布偏移更明显时，分组阈值比单一全局阈值更稳。

3. **`group_norm_group_threshold` 是五 UAV 阶段最强的通用策略基线之一。**  
   在后续 `uav_04` 修复实验中，它也被持续拿来作为原始强基线进行对比。

### 3.2 关键对比点

| 配置 | 说明 | Overall OOD F1 | `uav_04` OOD F1 |
| --- | --- | ---: | ---: |
| `full_conservative` | 五 UAV 主实验基线 | 0.8065 | 0.0106 |
| `group_norm_group_threshold` | 强通用策略基线 | 0.8184 | 0.0704 |

### 3.3 策略层面结论

- 在五 UAV 阶段，**group normalization + group threshold** 是最值得保留的组合思路。
- 但即便如此，`uav_04` 仍然是主要瓶颈，说明问题并不只是“统一阈值不合适”，而是更深层的 OOD 分数方向与域偏移问题。

## 4. `uav_04` 定向修复消融

`uav_04` 是五 UAV 阶段最明显的困难域，因此单独做了三轮修复实验。

### 4.1 Round 1

| Experiment | Intent | Overall OOD F1 | `uav_04` AUROC | `uav_04` OOD F1 |
| --- | --- | ---: | ---: | ---: |
| `full_conservative` | 原五机主实验基线 | 0.8065 | 0.3865 | 0.0106 |
| `group_norm_group_threshold` | 原对比基线 | 0.8184 | 0.3937 | 0.0704 |
| `A3_groupnorm_groupthr_q090_ws8_s4` | 更密窗口 + raw group threshold | 0.7611 | 0.4588 | 0.3517 |
| `A4_groupnorm_conservative_k100_r085_q090` | 仅放松 conservative threshold | 0.7852 | 0.3937 | 0.0362 |
| `B1_transformeronly_raw_q090_noemb` | transformer-only + raw threshold | 0.7863 | 0.4224 | 0.0498 |
| `B4_transformeronly_raw_q090_pseudood_noemb` | transformer-only + pseudo-OOD | 0.8110 | 0.7914 | 0.6237 |

**Round 1 结论：**

1. 单纯放松 conservative floor 基本无效。
2. 更密窗口能缓解一部分问题，但整体代价不小。
3. `pseudo_ood` 方向校准是决定性因素；`B4` 第一次把 `uav_04` 拉回可用区间。

### 4.2 Round 2

| Experiment | Overall OOD F1 | `uav_04` AUROC | `uav_04` OOD F1 |
| --- | ---: | ---: | ---: |
| `A3_groupnorm_groupthr_q090_ws8_s4` | 0.7611 | 0.4588 | 0.3517 |
| `B4_transformeronly_raw_q090_pseudood_noemb` | 0.8110 | 0.7914 | 0.6237 |
| `C1_transformeronly_pseudood_raw_q090_ws8_s4` | 0.7334 | 0.5579 | 0.3586 |

**Round 2 结论：**

- 把“更密窗口”和“pseudo-OOD 修复”直接叠加没有继续变好。  
- 一旦分数方向已经修正，再继续缩窗口，反而会破坏整体平衡。

### 4.3 Round 3

| Experiment | Change | Overall OOD F1 | Overall AUROC | `uav_04` OOD F1 | `uav_04` AUROC |
| --- | --- | ---: | ---: | ---: | ---: |
| `B4` | reference | 0.8110 | 0.9202 | 0.6237 | 0.7914 |
| `F1_b4_q085` | `q_ood=0.85` | 0.7752 | 0.9202 | 0.7523 | 0.7914 |
| `F2_b4_q080` | `q_ood=0.80` | 0.7472 | 0.9202 | 0.8104 | 0.7914 |
| `F3_b4_proto_q090` | `fusion=proto` | 0.8349 | 0.9515 | 0.3226 | 0.6522 |
| `F4_b4_knn_q090` | `fusion=knn` | 0.7767 | 0.8803 | 0.4552 | 0.5876 |

**Round 3 结论：**

1. 继续降低 `q_ood` 能提升 `uav_04` 的召回和 `OOD F1`，但本质上是在调低阈值，不是在改善分数排序。
2. `F2_b4_q080` 是当前“尽可能修好 `uav_04`”时的上界配置。
3. `F3_b4_proto_q090` 给出最好看的舰队级总体 OOD 指标，但明显伤害 `uav_04` 本身。
4. `knn` 融合没有同时改善整体和 `uav_04`，不适合继续作为主方向。

## 5. 当前四数据集主线与消融结论的关系

当前主线实验已经改为去掉 `uav_04`，保留 `uav_01`、`uav_02`、`uav_03`、`uav_05` 的四数据集设置。与原五 UAV 主实验相比：

| Metric | Five UAV | No `uav_04` | Delta |
| --- | ---: | ---: | ---: |
| ID micro-F1 | 0.9317 | 0.9331 | +0.0014 |
| ID macro-F1 | 0.8382 | 0.8449 | +0.0067 |
| ID mAP | 0.8950 | 0.9108 | +0.0158 |
| OOD AUROC | 0.8707 | 0.9765 | +0.1058 |
| OOD F1 | 0.8065 | 0.8843 | +0.0778 |

这说明：

1. `uav_04` 的确是原五 UAV 设置里最主要的困难域。
2. 当前项目切换到四数据集主线是合理的，因为它显著改善了总体 OOD 质量。
3. 分支消融的结论仍然有参考价值，但如果后续论文想严格对应当前四数据集主实验，最好再对 `transformer_only`、`mlp_only`、`full` 等配置在 **no-`uav_04` 数据集** 上补跑一轮。

## 6. 建议直接写进正文的结论

1. **时序分支是当前模型的主要有效分支。**  
   `transformer_only` 在整体 OOD 指标上优于 `full`、`mlp_only` 和 `gcn_only`，说明当前开放集判别能力主要来自时序建模。

2. **纯图分支不足以支撑当前异构 UAV 开放集检测。**  
   `gcn_only` 的 OOD 表现接近失效，说明图分支当前更适合作为潜在辅助信息，而非独立主干。

3. **完整融合结构尚未稳定超过纯时序分支。**  
   `full` 没有在关键 OOD 指标上持续压过 `transformer_only`，表明当前图分支与门控融合仍有优化空间。

4. **五 UAV 阶段的最大问题不是分类，而是 `uav_04` 的 OOD 校准。**  
   `uav_04` 的 ID 识别没有明显问题，但 OOD 排序和阈值决策失效，直接拖低整体结果。

5. **`pseudo_ood` 方向校准是当前最关键的修复因素。**  
   相比单独调窗口、单独调保守阈值，方向校准对困难域恢复更有效。

## 7. 结果来源说明

- 当前工作区直接可读的定向修复与四数据集对比结果：`UAV04_EXPERIMENT_SUMMARY.md`
- 当前代码中分支定义与实验入口：`ucs_oodid/model.py`、`scripts/run_encoder_ablation.py`
- 当前 README 中的四数据集主线实验与消融命令：`README.md`

其中“分支/编码器消融”的数值表为历史五 UAV 阶段已经整理出的核心汇总指标，用于后续写作和结论归纳。
