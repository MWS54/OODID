# 6 无人机入侵检测模型训练文档

## 1. 训练目标

本次训练使用项目内现成的 `transformer_only` 轻量化 UCS-OODID 主线模型，针对 **6 架无人机联合入侵检测 / OOD 识别** 场景进行离线训练。

参与训练的无人机分组如下：

- `uav_01`
- `uav_02`
- `uav_03`
- `uav_05`
- `uav_06`
- `uav_07`

本次实验明确 **不包含** `uav_04`。

## 2. 本次训练输入数据

- 训练数据文件：`data/multi_uav_hetero_no_uav04_plus_uav06_uav07_experiment.csv`
- 数据说明文件：`data/multi_uav_hetero_no_uav04_plus_uav06_uav07_notes.json`
- 总样本数：`668870`
- 特征数：`200`

## 3. 实际训练命令

在项目目录 `D:\UAV_IDS\new\ucs_oodid_project` 下执行：

```bash
python scripts/train.py ^
  --input "data\multi_uav_hetero_no_uav04_plus_uav06_uav07_experiment.csv" ^
  --output_dir "runs\multi_uav_6_offline_transformer_only" ^
  --label_col label ^
  --timestamp_col timestamp ^
  --record_id_col record_id ^
  --group_col uav_id ^
  --normalization_mode group ^
  --ood_threshold_mode group ^
  --group_threshold_strategy conservative ^
  --group_threshold_min_ratio 1.0 ^
  --use_group_embedding ^
  --group_embedding_dim 16 ^
  --encoder_ablation transformer_only ^
  --hidden_dim 64 ^
  --num_heads 2 ^
  --num_layers 1 ^
  --q_ood 0.90 ^
  --fusion correlation_aware ^
  --id_classes benign,icmp_flooding,udp_flooding,dos,ddos,bruteforce,mitm,deauthentication,jamming,injection,ip_spoofing,payload_manipulation,recon_scanning,generic,exploits,fuzzers,sybil ^
  --ood_classes replay,reply,fake_landing,evil,video_interception,unauthorized_udp,analysis,backdoor,shellcode,worms,blackhole,wormhole ^
  --window_mode count ^
  --window_size 16 ^
  --stride 8 ^
  --epochs 10
```

## 4. 训练产物

训练输出目录：

- `runs/multi_uav_6_offline_transformer_only`

关键产物文件：

- 模型文件：`runs/multi_uav_6_offline_transformer_only/artifact.pt`
- 评估结果：`runs/multi_uav_6_offline_transformer_only/eval_report.json`
- 特征泄漏检查：`runs/multi_uav_6_offline_transformer_only/leakage_report.json`

## 5. 放入仿真软件的方法

你后续放到模拟仿真软件 `Artifact path` 的文件就是：

- `D:\UAV_IDS\new\ucs_oodid_project\runs\multi_uav_6_offline_transformer_only\artifact.pt`

建议：

1. 直接将仿真软件的 `Artifact path` 指向这个 `artifact.pt`。
2. 如果仿真软件需要你手动复制文件，就复制这个 `artifact.pt` 到对应位置。
3. 如果仿真软件还有独立的数据说明或调试界面，可同时保留 `eval_report.json` 便于查看阈值和指标，但真正必须的模型文件仍然是 `artifact.pt`。

## 6. 本次训练结果摘要

训练已成功完成：

- 退出码：`0`
- 训练耗时：约 `118.8` 秒
- 随机种子：`42`

总体 ID 检测结果：

- `micro_f1`: `0.9286`
- `macro_f1`: `0.7982`
- `hamming_loss`: `0.0232`
- `subset_accuracy`: `0.7415`
- `mAP`: `0.9088`

总体 OOD 检测结果：

- `AUROC`: `0.9893`
- `AUPR_OUT`: `0.9842`
- `FPR95`: `0.0653`
- `precision`: `0.8402`
- `TPR`: `0.9728`
- `OOD_F1`: `0.9017`

## 7. 关键训练配置

- 编码器：`transformer_only`
- 轻量化部署：`true`
- 图分支：`关闭`
- `group_col`: `uav_id`
- 归一化方式：`group`
- OOD 阈值模式：`group`
- 阈值策略：`conservative`
- 组嵌入：`启用`
- 组嵌入维度：`16`
- 无人机组数：`6`

本次模型适合当前项目的多无人机异构流量检测场景，并且已经按照 `uav_id` 分组完成窗口构造、归一化和组阈值校准。

## 8. 如需重新训练

先进入项目目录并安装依赖：

```bash
cd D:\UAV_IDS\new\ucs_oodid_project
pip install -r requirements.txt
```

然后重新执行第 3 节中的训练命令即可。

## 9. 对比实验运行方式

如果需要一键运行论文中的对比实验，可以在项目目录 `D:\UAV_IDS\new\ucs_oodid_project` 下执行以下 Windows CMD 命令：

```cmd
python scripts/run_comparison_experiments.py ^
  --input "data\multi_uav_hetero_no_uav04_plus_uav06_uav07_experiment.csv" ^
  --output_root "runs\comparison_experiments_6uav" ^
  --label_col label ^
  --timestamp_col timestamp ^
  --record_id_col record_id ^
  --group_col uav_id ^
  --normalization_mode group ^
  --id_classes benign,icmp_flooding,udp_flooding,dos,ddos,bruteforce,mitm,deauthentication,jamming,injection,ip_spoofing,payload_manipulation,recon_scanning,generic,exploits,fuzzers,sybil ^
  --ood_classes replay,reply,fake_landing,evil,video_interception,unauthorized_udp,analysis,backdoor,shellcode,worms,blackhole,wormhole ^
  --window_mode count ^
  --window_size 16 ^
  --stride 8 ^
  --q_ood 0.90 ^
  --epochs 10 ^
  --seed 42
```

运行完成后，主要输出文件位于：

- `runs\comparison_experiments_6uav\known_detection_table.csv`
- `runs\comparison_experiments_6uav\ood_detection_table.csv`
- `runs\comparison_experiments_6uav\known_detection_table.tex`
- `runs\comparison_experiments_6uav\ood_detection_table.tex`
- `runs\comparison_experiments_6uav\comparison_summary.json`

## 10. 同源/一致数据集实验

如果需要把每个 `uav_id` 单独过滤出来，分别作为单源数据集运行同源实验，可以在项目目录 `D:\UAV_IDS\new\ucs_oodid_project` 下执行以下 Windows CMD 命令：

```cmd
python scripts\run_homogeneous_experiments.py ^
  --input "data\multi_uav_hetero_no_uav04_plus_uav06_uav07_experiment.csv" ^
  --output_root "runs\homogeneous_experiments_6uav" ^
  --label_col label ^
  --timestamp_col timestamp ^
  --record_id_col record_id ^
  --group_col uav_id ^
  --id_classes benign,icmp_flooding,udp_flooding,dos,ddos,bruteforce,mitm,deauthentication,jamming,injection,ip_spoofing,payload_manipulation,recon_scanning,generic,exploits,fuzzers,sybil ^
  --ood_classes replay,reply,fake_landing,evil,video_interception,unauthorized_udp,analysis,backdoor,shellcode,worms,blackhole,wormhole ^
  --window_mode count ^
  --window_size 16 ^
  --stride 8 ^
  --normalization_mode global ^
  --q_ood 0.90 ^
  --epochs 10 ^
  --seed 42 ^
  --methods random_forest,xgboost,transformer_only_mean,ucs_oodid
```

说明：

- 同源实验会自动按 `uav_id` 切分输入数据，并对每个 group 单独调用对比实验脚本。
- 同源实验默认推荐 `--normalization_mode global`，因为每次运行时只保留一个 group。
- 如果需要进一步比较同源结果与异质联合训练结果，可以继续执行下面的对比命令。

同源 vs 异质对比命令：

```cmd
python scripts\compare_homogeneous_heterogeneous.py ^
  --homogeneous_root "runs\homogeneous_experiments_6uav" ^
  --heterogeneous_root "runs\comparison_experiments_6uav" ^
  --output_dir "runs\homo_hetero_comparison" ^
  --focus_methods random_forest,xgboost,transformer_only_mean,ucs_oodid
```

主要输出文件：

- `runs\homogeneous_experiments_6uav\homogeneous_main_table.csv`
- `runs\homogeneous_experiments_6uav\homogeneous_main_table.tex`
- `runs\homo_hetero_comparison\homogeneous_vs_heterogeneous_table.csv`
- `runs\homo_hetero_comparison\homogeneous_vs_heterogeneous_table.tex`
