# UAV_04（第四无人机）实验结果总汇

## 1. 数据来源与清洗结果

### 1.1 原始来源
- 原始文件：`..\UAVs-Dataset-Under-Normal-and-Cyberattacks-main\Dataset_T-ITS.csv`
- 处理说明：仅保留 `class` 字段可直接识别的干净样本，避免混入同文件中另一套模式错位的数据行。

### 1.2 清洗后 `uav_04` 单机数据
| 指标 | 数值 |
| --- | ---: |
| 原始行数 | 54783 |
| 直接可用行数 | 33102 |
| 清洗后保留行数 | 19501 |
| 去重移除 | 1584 |
| 歧义行移除 | 12017 |
| `benign` | 8721 |
| `dos` | 4785 |
| `replay`（OOD） | 5995 |

### 1.3 融入四机异构数据后的 `uav_04` 切分
`uav_04` 被并入 `data/multi_uav_hetero_4_experiment.csv` 后，对应切分如下：

| Split | Rows |
| --- | ---: |
| train | 9455 |
| val | 2026 |
| test_id | 2025 |
| test_ood | 5995 |

四机联合数据总规模为 `465935` 行，特征数为 `133`。

## 2. 五机主实验中的 `uav_04` 问题定位

五机主实验基线目录：`runs/multi_uav_hetero_5_full_conservative_20260502_1700`

### 2.1 基线结果
| 指标 | Overall | `uav_04` |
| --- | ---: | ---: |
| ID micro-F1 | 0.9317 | 0.9954 |
| OOD AUROC | 0.8707 | 0.3865 |
| OOD F1 | 0.8065 | 0.0106 |

### 2.2 去掉 `uav_04` 后的对比
对比目录：`runs/multi_uav_hetero_no_uav04_full_conservative_20260502_1821`

| Metric | Five UAV | No `uav_04` | Delta |
| --- | ---: | ---: | ---: |
| ID micro-F1 | 0.9317 | 0.9331 | +0.0014 |
| ID macro-F1 | 0.8382 | 0.8449 | +0.0067 |
| ID mAP | 0.8950 | 0.9108 | +0.0158 |
| OOD AUROC | 0.8707 | 0.9765 | +0.1058 |
| OOD F1 | 0.8065 | 0.8843 | +0.0778 |

### 2.3 结论
`uav_04` 在五机主实验中是最主要的 OOD 瓶颈。其已知类识别几乎没有问题，但 OOD 排序和阈值决策明显失效，导致整体五机 OOD 指标被明显拖低。

## 3. `uav_04` 修复实验 Round 1

目录：`runs/uav04_repair_round1_20260502_1747`

| Experiment | Intent | Overall OOD F1 | `uav_04` AUROC | `uav_04` OOD F1 |
| --- | --- | ---: | ---: | ---: |
| `full_conservative` | 原五机主实验基线 | 0.8065 | 0.3865 | 0.0106 |
| `group_norm_group_threshold` | 原对比基线 | 0.8184 | 0.3937 | 0.0704 |
| `A3_groupnorm_groupthr_q090_ws8_s4` | 更密窗口 + raw group threshold | 0.7611 | 0.4588 | 0.3517 |
| `A4_groupnorm_conservative_k100_r085_q090` | 仅放松 conservative threshold | 0.7852 | 0.3937 | 0.0362 |
| `B1_transformeronly_raw_q090_noemb` | transformer-only + raw threshold | 0.7863 | 0.4224 | 0.0498 |
| `B4_transformeronly_raw_q090_pseudood_noemb` | transformer-only + pseudo-OOD | 0.8110 | 0.7914 | 0.6237 |

### Round 1 结论
1. 单纯放松 conservative floor 基本无效。
2. 更密的窗口能部分缓解问题，但提升有限且会损伤整体表现。
3. 真正起决定作用的是 `pseudo_ood` 方向校准；`B4` 首次把 `uav_04` 的 AUROC 拉到可用区间，并且整体 OOD F1 还略高于原主实验基线。

## 4. `uav_04` 修复实验 Round 2

目录：`runs/uav04_repair_round2_20260502_1801`

| Experiment | Overall OOD F1 | `uav_04` AUROC | `uav_04` OOD F1 |
| --- | ---: | ---: | ---: |
| `A3_groupnorm_groupthr_q090_ws8_s4` | 0.7611 | 0.4588 | 0.3517 |
| `B4_transformeronly_raw_q090_pseudood_noemb` | 0.8110 | 0.7914 | 0.6237 |
| `C1_transformeronly_pseudood_raw_q090_ws8_s4` | 0.7334 | 0.5579 | 0.3586 |

### Round 2 结论
把“更密窗口”和“`B4` 的 pseudo-OOD 修复”直接叠加，并没有得到更好的结果。说明在分数方向已经修正后，再缩小窗口反而会破坏五机整体平衡。

## 5. `uav_04` 修复实验 Round 3

目录：`runs/uav04_repair_round3_20260502_1809`

| Experiment | Change | Overall OOD F1 | Overall AUROC | `uav_04` OOD F1 | `uav_04` AUROC |
| --- | --- | ---: | ---: | ---: | ---: |
| `B4` | reference | 0.8110 | 0.9202 | 0.6237 | 0.7914 |
| `F1_b4_q085` | `q_ood=0.85` | 0.7752 | 0.9202 | 0.7523 | 0.7914 |
| `F2_b4_q080` | `q_ood=0.80` | 0.7472 | 0.9202 | 0.8104 | 0.7914 |
| `F3_b4_proto_q090` | `fusion=proto` | 0.8349 | 0.9515 | 0.3226 | 0.6522 |
| `F4_b4_knn_q090` | `fusion=knn` | 0.7767 | 0.8803 | 0.4552 | 0.5876 |

### Round 3 结论
1. 降低 `q_ood` 可以持续提高 `uav_04` 的 OOD F1，但本质上是在下调阈值，不是在改善分数排序，因此整体五机 OOD F1 会明显下降。
2. `F2_b4_q080` 是当前“单独追求 `uav_04` 捕获能力”时最强的方案，`uav_04 OOD F1 = 0.8104`。
3. `F3_b4_proto_q090` 给出了当前最好看的整体舰队级指标，但会明显伤害 `uav_04` 本身，因此更像 fleet-wide 优化而不是 `uav_04` 修复。
4. `knn` 融合没有同时改善整体和 `uav_04`，不建议继续投入。

## 6. 当前总判断

### 6.1 如果目标是“整体五机最均衡”
优先保留 `B4_transformeronly_raw_q090_pseudood_noemb`：

- Overall OOD F1：`0.8110`
- `uav_04` OOD F1：`0.6237`
- `uav_04` AUROC：`0.7914`

这是目前兼顾五机整体表现和 `uav_04` 可用性的最佳折中点。

### 6.2 如果目标是“尽可能修好 `uav_04`”
优先选择 `F2_b4_q080`：

- Overall OOD F1：`0.7472`
- `uav_04` OOD F1：`0.8104`
- `uav_04` Recall：`0.7487`

代价是整体舰队级 OOD 表现会明显回落，且告警率会更高。

### 6.3 如果目标是“整体 OOD 指标最好看”
可选 `F3_b4_proto_q090`：

- Overall OOD F1：`0.8349`
- Overall AUROC：`0.9515`
- `uav_04` OOD F1：`0.3226`

该配置不适合作为 `uav_04` 修复结论，只适合作为 fleet-level 备选。

## 7. 建议写入正文的结论

1. `uav_04` 的核心问题不是 ID 分类，而是 OOD 分数方向和阈值校准失效。
2. `pseudo_ood` 方向校准是最关键的修复因素，明显强于单独调窗口或单独调保守阈值。
3. 当前最合理的主结论配置是 `B4_transformeronly_raw_q090_pseudood_noemb`。
4. 若论文或报告需要强调“`uav_04` 最佳单机恢复能力”，则补充引用 `F2_b4_q080` 作为上界结果。

## 8. 结果来源文件
- `data/uavs_normal_cyberattacks_uav04_notes.json`
- `data/multi_uav_hetero_4_notes.json`
- `runs/multi_uav_hetero_5_full_conservative_20260502_1700/RESULT_SUMMARY.md`
- `runs/multi_uav_hetero_no_uav04_full_conservative_20260502_1821/RESULT_SUMMARY.md`
- `runs/uav04_repair_round1_20260502_1747/RESULT_SUMMARY.md`
- `runs/uav04_repair_round2_20260502_1801/RESULT_SUMMARY.md`
- `runs/uav04_repair_round3_20260502_1809/RESULT_SUMMARY.md`
