# 纯 IDS 检测结果文档

## 1. 结果来源

- 离线评估目录：`runs/multi_uav_6_offline_transformer_only_pureids_20260504_0200_rerun`
- 在线回放目录：`runs/mixed_attack_6datasets_restart_20260504_0114_pureids/simulation_payload.json`
- 自动导出的明细文件：
  - `runs/multi_uav_6_offline_transformer_only_pureids_20260504_0200_rerun/metrics_summary.md`
  - `runs/multi_uav_6_offline_transformer_only_pureids_20260504_0200_rerun/metrics_summary.json`
  - `runs/multi_uav_6_offline_transformer_only_pureids_20260504_0200_rerun/per_uav_metrics.csv`
  - `runs/multi_uav_6_offline_transformer_only_pureids_20260504_0200_rerun/per_source_type_metrics.csv`

本文中的“纯 IDS”指 `transformer_only` 主干的检测配置，不包含额外结构消融结论，重点给出离线检测指标与在线回放表现。

## 2. 实验配置

| 项目 | 数值 |
| --- | --- |
| 模型 | `transformer_only` |
| 图分支 | `false` |
| OOD 融合 | `correlation_aware` |
| 归一化 | `group` |
| 阈值模式 | `group` |
| 分组列 | `uav_id` |
| 窗口模式 | `count` |
| 窗口长度 | `16` |
| 步长 | `8` |
| 随机种子 | `42` |
| 设备 | `cuda` |
| 总窗口数 | `83598` |

补充部署基准如下：

| 指标 | 数值 |
| --- | ---: |
| 参数量 | 109969 |
| 模型包大小 | 6.6837 MB |
| 权重大小 | 0.4195 MB |
| 平均窗口推理时延 | 0.1847 ms |
| 最大窗口推理时延 | 0.2911 ms |
| 吞吐率 | 5413.8819 windows/s |

## 3. 离线检测总体结果

| 指标 | 数值 |
| --- | ---: |
| 已知攻击 Micro-F1 | 0.9387 |
| 已知攻击 Macro-F1 | 0.8738 |
| 已知攻击 mAP | 0.9062 |
| Subset Accuracy | 0.7735 |
| Hamming Loss | 0.0197 |
| OOD AUROC | 0.9737 |
| OOD AUPRC | 0.9660 |
| OOD FPR95 | 0.1301 |
| OOD Precision | 0.8528 |
| OOD Recall | 0.9377 |
| OOD F1 | 0.8933 |
| 全局 OOD 阈值 | 0.8634 |
| 主导 OOD 分数分支 | `knn` |

总体上，纯 IDS 离线结果较强：已知攻击检测和未知攻击拒识都达到了可用水平，且推理时延很低，适合做轻量级部署基线。

## 4. 分 UAV 离线结果

| UAV | 数据集 | ID Micro-F1 | Present-Class Macro-F1 | OOD AUROC | OOD F1 | FPR95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `uav_01` | UAV-NDD | 0.9301 | 0.7767 | 0.9818 | 0.9307 | 0.1356 |
| `uav_02` | GCS-to-UAV Updated | 0.9811 | 0.9335 | 0.9960 | 0.4697 | 0.0106 |
| `uav_03` | ISOT Drone Dataset | 0.9705 | 0.7972 | 0.8746 | 0.6119 | 0.8762 |
| `uav_05` | UNSW-NB15 | 0.9283 | 0.8380 | 0.8401 | 0.3716 | 0.4575 |
| `uav_06` | ECU-IoFT-main | 0.9976 | 0.9984 | 1.0000 | 0.6486 | 0.0000 |
| `uav_07` | UAVIDS | 0.9839 | 0.9804 | 0.9931 | 0.9920 | 0.0278 |

说明：

- `Present-Class Macro-F1` 更适合多 UAV 异构场景，因为每个 UAV 只覆盖全局标签空间的一部分，直接看 per-UAV `Macro-F1` 会被缺失类别拉低。
- `uav_06` 与 `uav_07` 的离线结果最好，说明在这两个域上纯 IDS 识别较稳定。
- `uav_05`（`UNSW-NB15`）是当前最弱域，且它属于 `external_non_uav`，不应直接当作真实 UAV 主实验结论。
- `uav_03` 的 `FPR95=0.8762` 偏高，说明该域的阈值判别稳定性仍然不足。

## 5. 在线混合攻击回放结果

以下结果来自 `runs/mixed_attack_6datasets_restart_20260504_0114_pureids/simulation_payload.json` 中的 `ood_trace` 与汇总字段，用于反映纯 IDS 在在线多机混合攻击场景下的实际表现。

### 5.1 回放总体摘要

| 指标 | 数值 |
| --- | ---: |
| 记录数 | 432 |
| 攻击记录数 | 105 |
| 检测窗口数 | 402 |
| 告警数 | 29 |
| 误告警数 | 21 |
| 响应次数 | 6 |
| 总能耗 | 18.2529 Wh |
| 任务成功 | false |
| 攻击开始时间 | 24.0 s |
| 首次告警时间 | 25.0 s |
| 攻击后告警数 | 29 |
| 攻击前最大 OOD 分数 | 3.3364 |
| 攻击后最大 OOD 分数 | 92.2549 |

补充观察：

- 首个告警出现在攻击开始后约 1 秒。
- 误告警占全部告警的比例约为 `72.4%`（`21/29`）。
- 按窗口统计的攻击检测率仅约为 `5.9%`（`8/135`），表明在线 mixed-attack 场景明显弱于离线评估。

### 5.2 单攻击 / 混合攻击窗口检测结果

| 攻击模式 | 攻击窗口 | 检出窗口 | 漏检窗口 | 检测率 | Critical 告警 | 平均 OOD | 峰值 OOD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `single_attack` | 2 | 0 | 2 | 0.0000 | 0 | -2.9524 | -2.7194 |
| `mixed_attack` | 133 | 8 | 125 | 0.0602 | 6 | -4.9463 | 92.2549 |

### 5.3 各 UAV 在线检测结果

| UAV | 攻击窗口 | 检出攻击窗口 | 检测率 | 误告警窗口 | 峰值 OOD |
| --- | ---: | ---: | ---: | ---: | ---: |
| `uav_01` | 23 | 7 | 0.3043 | 0 | 92.2549 |
| `uav_02` | 21 | 1 | 0.0476 | 3 | 2.4270 |
| `uav_03` | 20 | 0 | 0.0000 | 2 | 1.9693 |
| `uav_05` | 25 | 0 | 0.0000 | 3 | 2.3522 |
| `uav_06` | 23 | 0 | 0.0000 | 1 | 3.3364 |
| `uav_07` | 23 | 0 | 0.0000 | 12 | 4.3696 |

从在线回放看，只有 `uav_01` 有相对明显的攻击捕获能力，其余 UAV 基本未能稳定检出；其中 `uav_07` 的误告警窗口最多。

## 6. 结论

1. 纯 IDS 离线结果是好的：`Micro-F1=0.9387`、`OOD AUROC=0.9737`、`OOD F1=0.8933`，并且平均推理时延只有 `0.1847 ms`。
2. 纯 IDS 在离线基准上已经证明具备较好的已知攻击识别与未知攻击拒识能力，可作为当前项目的轻量部署型离线基线。
3. 但在当前在线 mixed-attack 回放中，检测率只有约 `5.9%`，误告警比例约 `72.4%`，最终 `mission_success=false`，说明离线表现尚未直接转化为稳定的在线多机防御效果。
4. 因此，若正文需要强调“纯 IDS 模型本身的识别能力”，建议优先引用第 3 节与第 4 节；若需要说明真实回放效果，则必须同时引用第 5 节，并明确指出在线 mixed-attack 场景仍是当前瓶颈。
