# 面向无人机端部署的轻量级 Transformer 加密流量入侵检测与 OOD 识别

This repository now defaults to a lightweight `transformer_only` pipeline for encrypted UAV traffic intrusion detection and OOD recognition. The current mainline is designed for onboard deployment with a smaller Transformer backbone and without default graph construction.

Historical graph-capable code is intentionally preserved:

- `transformer_only` is the recommended default mainline.
- `full`, `gcn_only`, `random_graph`, and `mlp_only` remain available for historical ablations and future extensions.
- GCN and behavior-graph modules are no longer the default research narrative of the repository.

## Install

```bash
cd ucs_oodid_project
pip install -r requirements.txt
```

## 仿真系统简介

This repository ships with a lightweight UAV mission simulator, an optional realtime UCS-OODID shell, and a Streamlit dashboard for visualization. The simulator models mission phases, per-UAV mission contexts, attack schedules, link quality, traffic load, energy draw, and response actions, then can feed the generated records into the online detector.

## 默认演示场景配置

The shared demo scene is stored in `configs/demo_scene_default.yaml`. It defines the default dashboard mission, multi-UAV runtime parameters, and detector bootstrap settings. The lower-level simulator profile templates and per-UAV attack plans remain in `configs/simulator_default.yaml`.

## 单 UAV 运行方式

```bash
python scripts/simulate_live_demo.py \
  --config configs/demo_scene_default.yaml \
  --uav_count 1 \
  --uav_id uav_demo_01 \
  --duration_s 24 \
  --output_json runs/demo_single_uav.json
```

## 多 UAV 运行方式

```bash
python scripts/simulate_live_demo.py \
  --config configs/demo_scene_default.yaml \
  --duration_s 48 \
  --output_json runs/demo_multi_uav.json
```

## 启动 Dashboard

Launch the visualization dashboard:

```bash
python scripts/launch_dashboard.py
```

Or run Streamlit directly:

```bash
streamlit run app/streamlit_app.py
```

The dashboard uses `configs/demo_scene_default.yaml` as its built-in demo defaults and provides four views:

- `Mission`: configure UAV count, mission parameters, attack settings, and detection settings.
- `Live Monitor`: inspect current mission phase, `battery_soc`, speed/altitude, link quality, throughput, OOD scores, alerts, and cumulative energy.
- `Replay`: scrub the mission timeline and replay attack / OOD / alert evolution.
- `Analysis`: review final metrics, attack-alert summaries, throughput totals, mission phase durations, and energy breakdown charts.

Use the `Run built-in demo scene` button on the `Mission` page to populate the dashboard with the shared default demo scenario.

## Current Mainline

The default training profile is deployment-oriented:

- `encoder_ablation=transformer_only`
- `hidden_dim=64`
- `num_heads=2`
- `num_layers=1`
- `q_ood=0.90`
- `fusion=correlation_aware`

The graph configuration and `gcn_layers` are still kept in configs and code so that historical ablations continue to run, but `transformer_only` does not build behavior graphs and does not use the GCN branch.

## UAV Dataset Mapping

Default three-UAV setup:

- `UAV-NDD` -> `uav_01`
- `GCS-to-UAV Updated` -> `uav_02`
- `ISOT Drone Dataset` -> `uav_03`

Optional non-mainline reference domain:

- `UNSW-NB15` -> `uav_05` as an external wired domain, not the default onboard UAV experiment.
- `ECU-IoFT` -> `uav_06` as an additional packet-level wireless domain.
- `UAVIDS` -> `uav_07` as an additional flow-level ad hoc UAV domain.

## Three-UAV Data Preparation

Prepare `UAV-NDD` as `uav_01`:

```bash
python scripts/prepare_uav_ndd_case1.py \
  --input data/UAV-NDD.csv \
  --output data/uav_ndd_case1_experiment.csv \
  --notes_json data/uav_ndd_case1_notes.json
```

Prepare `GCS-to-UAV Updated` as `uav_02`:

```bash
python scripts/prepare_gcs_to_uav_updated.py \
  --input data/GCS-to-UAV-Updated.csv \
  --output data/gcs_to_uav_updated_experiment.csv \
  --notes_json data/gcs_to_uav_updated_notes.json
```

Prepare `ISOT Drone Dataset` as `uav_03`:

```bash
python scripts/prepare_isot_drone.py \
  --input_root "..\ISOT Drone Dataset\Dataset\new_feature_csv" \
  --output data/isot_drone_uav03_experiment.csv \
  --notes_json data/isot_drone_uav03_notes.json \
  --target_rows 35599
```

Merge the three prepared UAV datasets:

```bash
python scripts/prepare_multi_uav_hetero.py \
  --uav_ndd_csv data/uav_ndd_case1_experiment.csv \
  --gcs_csv data/gcs_to_uav_updated_experiment.csv \
  --third_csv data/isot_drone_uav03_experiment.csv \
  --third_uav_id uav_03 \
  --third_dataset_name isot_drone \
  --output data/multi_uav_hetero_3_experiment.csv \
  --notes_json data/multi_uav_hetero_3_notes.json
```

Optional external wired-domain preparation with `UNSW-NB15` (`uav_05`):

```bash
python scripts/prepare_unsw_nb15.py \
  --train_input "..\NUSW\UNSW_NB15_training-set.csv" \
  --test_input "..\NUSW\UNSW_NB15_testing-set.csv" \
  --output data/unsw_nb15_uav05_experiment.csv \
  --notes_json data/unsw_nb15_uav05_notes.json
```

Optional packet-level wireless preparation with `ECU-IoFT` (`uav_06`):

```bash
python scripts/prepare_ecu_ioft.py \
  --input "..\ECU-IoFT-main\dataset\ECU-IoFT-Dataset.csv" \
  --output data/ecu_ioft_uav06_experiment.csv \
  --notes_json data/ecu_ioft_uav06_notes.json
```

Optional flow-level ad hoc UAV preparation with `UAVIDS` (`uav_07`):

```bash
python scripts/prepare_uavids.py \
  --input "..\UAVIDS\UAVIDS-2025.csv" \
  --output data/uavids_uav07_experiment.csv \
  --notes_json data/uavids_uav07_notes.json
```

## Train The Transformer-Only Mainline

```bash
python scripts/train.py \
  --input data/multi_uav_hetero_3_experiment.csv \
  --output_dir runs/uav_onboard_transformer_only \
  --label_col label \
  --timestamp_col timestamp \
  --record_id_col record_id \
  --group_col uav_id \
  --normalization_mode group \
  --ood_threshold_mode group \
  --group_threshold_strategy conservative \
  --group_threshold_min_ratio 1.0 \
  --use_group_embedding \
  --group_embedding_dim 16 \
  --encoder_ablation transformer_only \
  --hidden_dim 64 \
  --num_heads 2 \
  --num_layers 1 \
  --q_ood 0.90 \
  --fusion correlation_aware \
  --id_classes benign,icmp_flooding,udp_flooding,dos,ddos,bruteforce,mitm,deauthentication,jamming,recon_scanning \
  --ood_classes replay,fake_landing,evil,video_interception,unauthorized_udp \
  --window_mode count \
  --window_size 16 \
  --stride 8 \
  --epochs 10
```

`eval_report.json` now includes:

- `deployment_profile`
- `id_test_by_group`
- `ood_test_by_group`

## Run Detection

```bash
python scripts/detect.py \
  --input data/multi_uav_hetero_3_experiment.csv \
  --artifact runs/uav_onboard_transformer_only/artifact.pt \
  --output_jsonl runs/uav_onboard_transformer_only/detections.jsonl \
  --record_scores_json runs/uav_onboard_transformer_only/record_scores.json \
  --summary_json runs/uav_onboard_transformer_only/group_detection_summary.json \
  --group_col uav_id
```

## Benchmark Onboard Deployment

```bash
python scripts/benchmark_onboard.py \
  --artifact runs/uav_onboard_transformer_only/artifact.pt \
  --input data/multi_uav_hetero_3_experiment.csv \
  --group_col uav_id \
  --batch_size 256 \
  --warmup_runs 5 \
  --repeat_runs 20
```

The benchmark saves `benchmark_report.json` with deployment indicators such as parameter count, model size, average per-window inference time, and throughput.

## Historical Ablations Still Supported

The following encoder modes remain available and should continue to work for compatibility and ablation studies:

- `full`
- `gcn_only`
- `random_graph`
- `mlp_only`
- `transformer_only`

Run the shared encoder-ablation driver:

```bash
python scripts/run_encoder_ablation.py \
  --input data/multi_uav_hetero_3_experiment.csv \
  --output_root runs/encoder_ablation_uav_onboard \
  --group_col uav_id \
  --label_col label \
  --timestamp_col timestamp \
  --record_id_col record_id \
  --hidden_dim 64 \
  --num_heads 2 \
  --num_layers 1 \
  --q_ood 0.90 \
  --fusion correlation_aware \
  --id_classes benign,icmp_flooding,udp_flooding,dos,ddos,bruteforce,mitm,deauthentication,jamming,recon_scanning \
  --ood_classes replay,fake_landing,evil,video_interception,unauthorized_udp \
  --window_mode count \
  --window_size 16 \
  --stride 8 \
  --epochs 10 \
  --normalization_mode group \
  --ood_threshold_mode group \
  --group_threshold_strategy conservative \
  --group_threshold_min_ratio 1.0 \
  --use_group_embedding
```

## Feature And Leakage Policy

The model uses metadata-only encrypted traffic features. The following fields are reserved for grouping, statistics, or calibration and are not used as detection features:

- `uav_id`
- `domain_id`
- `source_type`
- `direction_type`
- `scenario_role`
- `original_group_id`

Payload-derived and semantic-leakage fields are also dropped by default.

## Main Scripts

- `scripts/train.py`: train, calibrate, and evaluate the default Transformer-only mainline.
- `scripts/detect.py`: run online intrusion detection and OOD rejection.
- `scripts/benchmark_onboard.py`: export deployment-oriented benchmark metrics for onboard use.
- `scripts/run_encoder_ablation.py`: compare `transformer_only`, `full`, `gcn_only`, `random_graph`, and `mlp_only`.
- `scripts/edge_benchmark.py`: legacy module-level latency benchmark.
- `scripts/offline_triage.py`: offline clustering and analyst-facing report generation.

## Notes

- `--group_col uav_id` keeps windows inside the same UAV and avoids cross-UAV mixing.
- Homogeneous experiments such as `scripts/run_homogeneous_experiments.py` default to `--normalization_mode global` because each run filters down to a single group.
- Heterogeneous experiments such as `scripts/run_comparison_experiments.py` default to `--normalization_mode group` so each UAV/group can use its own scaler with a global fallback.
- `transformer_only` is the default branch for current experiments.
- GCN and graph-building code remain in the repository to keep historical tests and ablations compatible.
## Legacy Research Appendix

The sections below are kept as historical paper-style notes for the original Transformer-GCN-oriented UCS-OODID pipeline. They are not the default mainline of this repository.

**UCS-OODID: A Temporal-Topological Framework for Open-Set Intrusion Detection in Encrypted UAV Control Links** (historical paper framing)

It includes:

- metadata-only leakage-control preprocessing;
- count/time/adaptive mixed-window construction with window-record mappings;
- in-window kNN behavioral graph construction and graph ablation variants;
- Transformer temporal branch;
- dense GCN behavioral-topology branch;
- learnable gate fusion;
- attention-based multi-instance pooling;
- multi-label ID prediction;
- ID-only temperature, class-threshold, OOD-score, fusion-weight, and OOD-threshold calibration;
- confidence, energy, prototype-distance, and kNN-consistency OOD evidence scores;
- correlation-aware, mean, hard-voting, variance-weighted, and single-score OOD fusion;
- online detection and OOD rejection;
- record-level suspiciousness aggregation for mixed OOD windows;
- offline OOD clustering and analyst-facing Markdown report generation;
- mixed-window, open-set, fusion, graph-ablation, mission-phase proxy, and edge-latency experiment scripts;
- synthetic data generator for end-to-end verification.

## Install

```bash
cd ucs_oodid_project
pip install -r requirements.txt
```

The code avoids PyTorch Geometric and implements dense adjacency GCN directly, because the paper's default window size is small (`N=16`).

## Quick synthetic run

```bash
python scripts/make_synthetic.py --output examples/synthetic_uav.csv --records 5000 --seed 42

python scripts/train.py \
  --input examples/synthetic_uav.csv \
  --output_dir runs/synthetic \
  --label_col label \
  --timestamp_col timestamp \
  --record_id_col record_id \
  --id_classes benign,dos,mitm,spoof,replay,injection,scan \
  --ood_classes unknown_probe,unknown_burst \
  --window_size 16 \
  --stride 8 \
  --epochs 5 \
  --batch_size 128

python scripts/detect.py \
  --input examples/synthetic_uav.csv \
  --artifact runs/synthetic/artifact.pt \
  --output_jsonl runs/synthetic/detections.jsonl \
  --record_scores_json runs/synthetic/record_scores.json

python scripts/offline_triage.py \
  --detections runs/synthetic/detections.jsonl \
  --output_dir runs/synthetic/offline_report
```

## Multi-UAV Heterogeneous Traffic Monitoring

This project can simulate two heterogeneous UAVs by jointly using two different prepared datasets:

- `UAV-NDD CSV` -> `uav_01`
- `GCS-to-UAV Updated` -> `uav_02`

Use the following end-to-end workflow.

1. Prepare UAV-NDD:

```bash
python scripts/prepare_uav_ndd_case1.py \
  --input data/UAV-NDD.csv \
  --output data/uav_ndd_case1_experiment.csv \
  --notes_json data/uav_ndd_case1_notes.json
```

2. Prepare GCS-to-UAV Updated:

```bash
python scripts/prepare_gcs_to_uav_updated.py \
  --input data/GCS-to-UAV-Updated.csv \
  --output data/gcs_to_uav_updated_experiment.csv \
  --notes_json data/gcs_to_uav_updated_notes.json
```

3. Merge them into one heterogeneous multi-UAV dataset:

```bash
python scripts/prepare_multi_uav_hetero.py \
  --uav_ndd_csv data/uav_ndd_case1_experiment.csv \
  --gcs_csv data/gcs_to_uav_updated_experiment.csv \
  --output data/multi_uav_hetero_experiment.csv \
  --notes_json data/multi_uav_hetero_notes.json
```

Optional: add ISOT as a third UAV (`uav_03`) at a size that matches `gcs_to_uav_updated_experiment.csv`:

```bash
python scripts/prepare_isot_drone.py \
  --input_root "..\ISOT Drone Dataset\Dataset\new_feature_csv" \
  --output data/isot_drone_uav03_experiment.csv \
  --notes_json data/isot_drone_uav03_notes.json \
  --target_rows 35599

python scripts/prepare_multi_uav_hetero.py \
  --uav_ndd_csv data/uav_ndd_case1_experiment.csv \
  --gcs_csv data/gcs_to_uav_updated_experiment.csv \
  --third_csv data/isot_drone_uav03_experiment.csv \
  --third_uav_id uav_03 \
  --third_dataset_name isot_drone \
  --output data/multi_uav_hetero_3_experiment.csv \
  --notes_json data/multi_uav_hetero_3_notes.json
```

Optional legacy step: add `UAVs-Dataset-Under-Normal-and-Cyberattacks-main` as a fourth UAV (`uav_04`) using only the clean packet-level rows with direct class labels. If you are following the current main experiment setup, skip this dataset and keep the command below only as a historical reference:

```bash
python scripts/prepare_uavs_normal_cyberattacks.py \
  --input "..\UAVs-Dataset-Under-Normal-and-Cyberattacks-main\Dataset_T-ITS.csv" \
  --output data/uavs_normal_cyberattacks_uav04_experiment.csv \
  --notes_json data/uavs_normal_cyberattacks_uav04_notes.json

python scripts/prepare_multi_uav_hetero.py \
  --uav_ndd_csv data/uav_ndd_case1_experiment.csv \
  --gcs_csv data/gcs_to_uav_updated_experiment.csv \
  --third_csv data/isot_drone_uav03_experiment.csv \
  --third_uav_id uav_03 \
  --third_dataset_name isot_drone \
  --fourth_csv data/uavs_normal_cyberattacks_uav04_experiment.csv \
  --fourth_uav_id uav_04 \
  --fourth_dataset_name uavs_normal_cyberattacks \
  --output data/multi_uav_hetero_4_experiment.csv \
  --notes_json data/multi_uav_hetero_4_notes.json
```

Historical multi-domain setup: keep the other four datasets and do not include `uav_04` in the merged experiment:

```bash
python scripts/prepare_unsw_nb15.py \
  --train_input "..\NUSW\UNSW_NB15_training-set.csv" \
  --test_input "..\NUSW\UNSW_NB15_testing-set.csv" \
  --output data/unsw_nb15_uav05_experiment.csv \
  --notes_json data/unsw_nb15_uav05_notes.json

python scripts/prepare_multi_uav_hetero.py \
  --uav_ndd_csv data/uav_ndd_case1_experiment.csv \
  --gcs_csv data/gcs_to_uav_updated_experiment.csv \
  --third_csv data/isot_drone_uav03_experiment.csv \
  --third_uav_id uav_03 \
  --third_dataset_name isot_drone \
  --fifth_csv data/unsw_nb15_uav05_experiment.csv \
  --fifth_uav_id uav_05 \
  --fifth_dataset_name unsw_nb15 \
  --output data/multi_uav_hetero_no_uav04_experiment.csv \
  --notes_json data/multi_uav_hetero_no_uav04_notes.json
```

Extended seven-source setup with `ECU-IoFT` and `UAVIDS`:

```bash
python scripts/prepare_ecu_ioft.py \
  --input "..\ECU-IoFT-main\dataset\ECU-IoFT-Dataset.csv" \
  --output data/ecu_ioft_uav06_experiment.csv \
  --notes_json data/ecu_ioft_uav06_notes.json

python scripts/prepare_uavids.py \
  --input "..\UAVIDS\UAVIDS-2025.csv" \
  --output data/uavids_uav07_experiment.csv \
  --notes_json data/uavids_uav07_notes.json

python scripts/prepare_multi_uav_hetero.py \
  --uav_ndd_csv data/uav_ndd_case1_experiment.csv \
  --gcs_csv data/gcs_to_uav_updated_experiment.csv \
  --third_csv data/isot_drone_uav03_experiment.csv \
  --third_uav_id uav_03 \
  --third_dataset_name isot_drone \
  --fourth_csv data/uavs_normal_cyberattacks_uav04_experiment.csv \
  --fourth_uav_id uav_04 \
  --fourth_dataset_name uavs_normal_cyberattacks \
  --fifth_csv data/unsw_nb15_uav05_experiment.csv \
  --fifth_uav_id uav_05 \
  --fifth_dataset_name unsw_nb15 \
  --sixth_csv data/ecu_ioft_uav06_experiment.csv \
  --sixth_uav_id uav_06 \
  --sixth_dataset_name ecu_ioft \
  --seventh_csv data/uavids_uav07_experiment.csv \
  --seventh_uav_id uav_07 \
  --seventh_dataset_name uavids \
  --output data/multi_uav_hetero_7_experiment.csv \
  --notes_json data/multi_uav_hetero_7_notes.json
```

If you want to keep the historical `no_uav04` convention while still adding the two new datasets, reuse the same command and append `--exclude_uav_ids uav_04`, then change the output filenames accordingly.

If you already have a five-input merge command and only want to drop `uav_04`, you can also keep the original inputs and append `--exclude_uav_ids uav_04`.

4. Historical joint multi-UAV training example:

```bash
python scripts/train.py \
  --input data/multi_uav_hetero_no_uav04_experiment.csv \
  --output_dir runs/multi_uav_hetero_no_uav04_full_conservative \
  --label_col label \
  --timestamp_col timestamp \
  --record_id_col record_id \
  --group_col uav_id \
  --normalization_mode group \
  --ood_threshold_mode group \
  --group_threshold_strategy conservative \
  --group_threshold_min_ratio 1.0 \
  --use_group_embedding \
  --group_embedding_dim 16 \
  --id_classes benign,icmp_flooding,udp_flooding,dos,ddos,bruteforce,mitm,deauthentication,jamming,injection,ip_spoofing,payload_manipulation,recon_scanning,generic,exploits,fuzzers \
  --ood_classes replay,reply,fake_landing,evil,video_interception,unauthorized_udp,analysis,backdoor,shellcode,worms \
  --window_mode count \
  --window_size 16 \
  --stride 8 \
  --epochs 10
```

5. Run detection:

```bash
python scripts/detect.py \
  --input data/multi_uav_hetero_no_uav04_experiment.csv \
  --artifact runs/multi_uav_hetero_no_uav04_full_conservative/artifact.pt \
  --output_jsonl runs/multi_uav_hetero_no_uav04_full_conservative/detections.jsonl \
  --record_scores_json runs/multi_uav_hetero_no_uav04_full_conservative/record_scores.json \
  --summary_json runs/multi_uav_hetero_no_uav04_full_conservative/group_detection_summary.json \
  --group_col uav_id
```

Notes:

- `--group_col uav_id` means windows are built only within the same UAV, so records from different UAVs are never mixed into the same window.
- Homogeneous experiments default to `--normalization_mode global`; this keeps single-group runs simple and avoids unnecessary group-specific preprocessing.
- Heterogeneous experiments default to `--normalization_mode group`; it fits one scaler per UAV/group plus a global fallback scaler.
- Group normalization can reduce training bias caused by one UAV contributing much more data or having a very different feature scale than the others.
- `multi_uav_hetero_no_uav04_experiment.csv` is the current recommended merged dataset; it keeps `uav_01`, `uav_02`, `uav_03`, and `uav_05` while leaving `uav_04` out of the experiment.
- The merged dataset uses a joint aligned feature space; features missing from one dataset are filled with `0`.
- `eval_report.json` includes `id_test_by_group` and `ood_test_by_group` so you can inspect per-UAV performance independently.
- `group_detection_summary.json` summarizes online detection alerts for each UAV during inference.

Quick homogeneous smoke test (Windows CMD):

```cmd
python scripts\run_homogeneous_experiments.py ^
  --input "data\multi_uav_hetero_no_uav04_plus_uav06_uav07_experiment.csv" ^
  --output_root "runs\homogeneous_smoke_test" ^
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
  --epochs 1 ^
  --seed 42 ^
  --groups uav_01 ^
  --methods random_forest,ucs_oodid
```

This smoke test should generate:

- `runs\homogeneous_smoke_test\homogeneous_summary.json`
- `runs\homogeneous_smoke_test\homogeneous_ood_detection_table.csv`
- `runs\homogeneous_smoke_test\homogeneous_main_table.csv`

Optional third-UAV training example with ISOT included:

```bash
python scripts/train.py \
  --input data/multi_uav_hetero_3_experiment.csv \
  --output_dir runs/multi_uav_hetero_3_joint \
  --label_col label \
  --timestamp_col timestamp \
  --record_id_col record_id \
  --group_col uav_id \
  --normalization_mode group \
  --id_classes benign,icmp_flooding,udp_flooding,dos,ddos,bruteforce,mitm,deauthentication,jamming,injection,ip_spoofing,payload_manipulation,recon_scanning \
  --ood_classes replay,fake_landing,evil,video_interception,unauthorized_udp \
  --window_mode count \
  --window_size 16 \
  --stride 8 \
  --epochs 10
```

Alternative: keep all five prepared inputs in the merge command, but explicitly exclude `uav_04` at merge time:

```bash
python scripts/prepare_multi_uav_hetero.py \
  --uav_ndd_csv data/uav_ndd_case1_experiment.csv \
  --gcs_csv data/gcs_to_uav_updated_experiment.csv \
  --third_csv data/isot_drone_uav03_experiment.csv \
  --third_uav_id uav_03 \
  --third_dataset_name isot_drone \
  --fourth_csv data/uavs_normal_cyberattacks_uav04_experiment.csv \
  --fourth_uav_id uav_04 \
  --fourth_dataset_name uavs_normal_cyberattacks \
  --fifth_csv data/unsw_nb15_uav05_experiment.csv \
  --fifth_uav_id uav_05 \
  --fifth_dataset_name unsw_nb15 \
  --exclude_uav_ids uav_04 \
  --output data/multi_uav_hetero_no_uav04_experiment.csv \
  --notes_json data/multi_uav_hetero_no_uav04_notes.json
```

Notes for the current four-dataset setup:

- `reply` is a small OOD label slice already present in `gcs_to_uav_updated_experiment.csv`, so keep it in `--ood_classes` alongside `replay`.
- `full_conservative` remains the safest default when one UAV has much fewer validation/OOD windows than the others.
- `--exclude_uav_ids uav_04` preserves the remaining UAV IDs as `uav_01`, `uav_02`, `uav_03`, and `uav_05`.

Optional UAV domain embedding:

```bash
python scripts/train.py \
  --input data/multi_uav_hetero_no_uav04_experiment.csv \
  --output_dir runs/multi_uav_hetero_no_uav04_group_emb \
  --label_col label \
  --timestamp_col timestamp \
  --record_id_col record_id \
  --group_col uav_id \
  --use_group_embedding \
  --group_embedding_dim 16 \
  --normalization_mode group \
  --ood_threshold_mode group \
  --id_classes benign,icmp_flooding,udp_flooding,dos,ddos,bruteforce,mitm,deauthentication,jamming,injection,ip_spoofing,payload_manipulation,recon_scanning,generic,exploits,fuzzers \
  --ood_classes replay,reply,fake_landing,evil,video_interception,unauthorized_udp,analysis,backdoor,shellcode,worms \
  --window_mode count \
  --window_size 16 \
  --stride 8 \
  --epochs 10
```

Seven-source training example with `ECU-IoFT` and `UAVIDS` included:

```bash
python scripts/train.py \
  --input data/multi_uav_hetero_7_experiment.csv \
  --output_dir runs/multi_uav_hetero_7_joint \
  --label_col label \
  --timestamp_col timestamp \
  --record_id_col record_id \
  --group_col uav_id \
  --normalization_mode group \
  --ood_threshold_mode group \
  --group_threshold_strategy conservative \
  --group_threshold_min_ratio 1.0 \
  --use_group_embedding \
  --group_embedding_dim 16 \
  --id_classes benign,icmp_flooding,udp_flooding,dos,ddos,bruteforce,mitm,deauthentication,jamming,injection,ip_spoofing,payload_manipulation,recon_scanning,generic,exploits,fuzzers,sybil \
  --ood_classes replay,reply,fake_landing,evil,video_interception,unauthorized_udp,analysis,backdoor,shellcode,worms,blackhole,wormhole \
  --window_mode count \
  --window_size 16 \
  --stride 8 \
  --epochs 10
```

Conservative group-threshold calibration:

```bash
python scripts/train.py \
  --input data/multi_uav_hetero_no_uav04_experiment.csv \
  --output_dir runs/multi_uav_no_uav04_full_conservative \
  --label_col label \
  --timestamp_col timestamp \
  --record_id_col record_id \
  --group_col uav_id \
  --normalization_mode group \
  --ood_threshold_mode group \
  --group_threshold_strategy conservative \
  --group_threshold_min_ratio 1.0 \
  --group_threshold_shrink_k 1000 \
  --group_threshold_min_samples 10 \
  --use_group_embedding \
  --group_embedding_dim 16 \
  --id_classes benign,icmp_flooding,udp_flooding,dos,ddos,bruteforce,mitm,deauthentication,jamming,injection,ip_spoofing,payload_manipulation,recon_scanning,generic,exploits,fuzzers \
  --ood_classes replay,reply,fake_landing,evil,video_interception,unauthorized_udp,analysis,backdoor,shellcode,worms \
  --window_mode count \
  --window_size 16 \
  --stride 8 \
  --epochs 10
```

Notes for conservative group thresholds:

- `conservative` first smooths each UAV-specific group threshold toward the global OOD threshold, then enforces a lower bound of `global_threshold * group_threshold_min_ratio`.
- This is useful when one UAV has fewer validation windows and its raw group quantile is calibrated much lower than the global threshold.
- In practice it helps suppress the `uav_02` style low-threshold, high-false-positive failure mode that can degrade OOD-F1.

## Cross-UAV Generalization

Cross-UAV generalization is not joint training. The model is trained on one UAV and directly tested on another UAV to measure how well it transfers to a new UAV source.

Example: `UAV-NDD -> GCS-to-UAV Updated`

```bash
python scripts/run_cross_uav_generalization.py \
  --source_csv runs/data_prepare_verify/uav_ndd_case1_experiment.csv \
  --target_csv runs/data_prepare_verify/gcs_to_uav_updated_experiment.csv \
  --source_uav_id uav_01 \
  --target_uav_id uav_02 \
  --output_dir runs/cross_uav_uavndd_to_gcs \
  --id_classes benign,icmp_flooding,udp_flooding,dos,ddos,bruteforce,mitm,deauthentication,jamming,recon_scanning \
  --ood_classes replay,fake_landing,evil \
  --window_mode count \
  --window_size 16 \
  --stride 8 \
  --epochs 10
```

Reverse direction: `GCS-to-UAV Updated -> UAV-NDD`

```bash
python scripts/run_cross_uav_generalization.py \
  --source_csv runs/data_prepare_verify/gcs_to_uav_updated_experiment.csv \
  --target_csv runs/data_prepare_verify/uav_ndd_case1_experiment.csv \
  --source_uav_id uav_02 \
  --target_uav_id uav_01 \
  --output_dir runs/cross_uav_gcs_to_uavndd \
  --id_classes benign,icmp_flooding,udp_flooding,dos,ddos,bruteforce,mitm,deauthentication,jamming,recon_scanning \
  --ood_classes replay,fake_landing,evil \
  --window_mode count \
  --window_size 16 \
  --stride 8 \
  --epochs 10
```

## Multi-UAV Ablation

```bash
python scripts/run_multi_uav_ablation.py \
  --input data/multi_uav_hetero_no_uav04_experiment.csv \
  --output_root runs/multi_uav_ablation_no_uav04 \
  --group_col uav_id \
  --id_classes benign,icmp_flooding,udp_flooding,dos,ddos,bruteforce,mitm,deauthentication,jamming,injection,ip_spoofing,payload_manipulation,recon_scanning,generic,exploits,fuzzers \
  --ood_classes replay,reply,fake_landing,evil,video_interception,unauthorized_udp,analysis,backdoor,shellcode,worms \
  --window_mode count \
  --window_size 16 \
  --stride 8 \
  --epochs 10
```

## Encoder Ablation

Use the encoder ablation driver below to compare `transformer_only`, `gcn_only`, `mlp_only`, `random_graph`, and `full` under the same dataset split, random seed, and calibration settings:

```bash
python scripts/run_encoder_ablation.py \
  --input data/multi_uav_hetero_no_uav04_experiment.csv \
  --output_root runs/encoder_ablation_no_uav04 \
  --group_col uav_id \
  --label_col label \
  --timestamp_col timestamp \
  --record_id_col record_id \
  --id_classes benign,icmp_flooding,udp_flooding,dos,ddos,bruteforce,mitm,deauthentication,jamming,injection,ip_spoofing,payload_manipulation,recon_scanning,generic,exploits,fuzzers \
  --ood_classes replay,reply,fake_landing,evil,video_interception,unauthorized_udp,analysis,backdoor,shellcode,worms \
  --window_mode count \
  --window_size 16 \
  --stride 8 \
  --epochs 10 \
  --normalization_mode group \
  --ood_threshold_mode group \
  --group_threshold_strategy conservative \
  --group_threshold_min_ratio 1.0 \
  --use_group_embedding
```

This generates `encoder_ablation_summary.csv` and `encoder_ablation_summary.json` in the selected `output_root`.

## Dataset format

The generic loader accepts CSV, JSONL, JSON, and Parquet. Each record should contain a label column and metadata columns. Typical columns:

- `timestamp`
- `record_id`
- `label`
- packet size statistics
- inter-arrival time statistics
- direction ratio
- burst statistics
- session duration
- packet/byte counts
- transport-level timing fields

Payload-derived and semantic-leakage fields are dropped by default using pattern matching, including payload bytes, message IDs, command IDs, application-layer fields, IP identifiers, and uncontrolled ports.

## Main scripts

- `scripts/train.py`: training, ID-only calibration, and evaluation.
- `scripts/detect.py`: online mixed-window detection and OOD rejection.
- `scripts/offline_triage.py`: offline clustering, behavioral profile summary, and report generation.
- `scripts/run_experiments.py`: runs the paper-style RQ1--RQ7 protocols over provided datasets.
- `scripts/fusion_ablation.py`: compares OOD fusion strategies.
- `scripts/graph_ablation.py`: compares behavioral graph variants.
- `scripts/edge_benchmark.py`: measures module-level latency.
- `scripts/make_synthetic.py`: creates a metadata-only UAV-like dataset for testing.

## Reproducibility policy implemented

- OOD samples are not used in training, temperature calibration, threshold calibration, OOD score normalization, fusion-weight estimation, or hyperparameter selection.
- The ID validation split is used for temperature calibration, class thresholds, prototype/bank construction, score normalization, score-correlation matrix, fusion weights, and OOD threshold.
- Unknown classes are only used for final testing or online detection simulation.

## Notes

This code is a research implementation matching the paper's design. Public UAV datasets differ in schemas, so dataset-specific conversion may be needed before using the generic metadata loader. Use the leakage-control configuration to explicitly separate removed fields and retained metadata features for every dataset.

## 2026-04-30 extended implementation notes

This enhanced version keeps the original UCS-OODID method chain and adds the missing paper-protocol tooling that was not fully automated in the first package.

### Correctness fixes

- **Padding-aware windows**: `WindowedData` now carries `valid_mask`. Time-based and adaptive windows pad to a fixed length, but padded records are ignored by Transformer attention, GCN graph construction, MIL pooling, record-head loss, and record-level suspiciousness ranking.
- **ID-only hard voting**: `OODCalibrator` now stores per-score thresholds from the ID validation split. Test-time hard voting no longer recomputes quantiles on the test set.
- **Record attribution**: repeated padded IDs are ignored when ranking suspicious records.

### Dataset conversion

Use `scripts/convert_dataset.py` to convert public datasets into the canonical schema:

```bash
python scripts/convert_dataset.py \
  --input raw.csv \
  --output data/canonical.csv \
  --dataset cicids2017 \
  --label_col Label \
  --timestamp_col Timestamp \
  --notes_json data/canonical_notes.json
```

Supported adapter profiles are `generic`, `cicids2017`, `ustc-tfc2016`, `mavlink`, `uav-gcs-ids`, and `uav-cyber-physical`. The adapter normalizes `label`, `timestamp`, and `record_id`; leakage-sensitive fields are still removed by `MetadataPreprocessor` and reported in `leakage_report.json`.

### Paper experiment drivers

```bash
# RQ2/RQ5 leave-one-class-out
python scripts/run_loco.py --input data/canonical.csv --output_dir runs/loco --train_args "--epochs 10"

# RQ5 attack-family hold-out
python scripts/run_family_holdout.py \
  --input data/canonical.csv \
  --output_dir runs/family \
  --family_map_json configs/family_map.json \
  --train_args "--epochs 10"

# RQ2/RQ5 cross-dataset unknown
python scripts/run_cross_dataset_ood.py \
  --source data/source.csv \
  --target data/target_unknown.csv \
  --output_dir runs/cross_dataset \
  --train_args "--epochs 10"

# RQ1 mixed-window attack-ratio experiments
python scripts/run_mixed_window_ratios.py \
  --output_dir runs/mixed_ratios \
  --ratios 0.05,0.10,0.20,0.50 \
  --train_args "--epochs 10"

# RQ6 mission-phase proxy evaluation
python scripts/mission_phase_eval.py \
  --input data/canonical.csv \
  --artifact runs/default/artifact.pt \
  --output_dir runs/mission_phase_proxy \
  --phase_col mission_phase_proxy

# RQ7 record-level suspiciousness-ranking metrics
python scripts/evaluate_attribution.py \
  --record_scores_json runs/default/record_scores.json \
  --records data/canonical.csv \
  --output_json runs/default/attribution_metrics.json

# Paper tables and score-correlation heatmap
python scripts/make_figures_tables.py \
  --results_dir runs \
  --output_dir paper_outputs \
  --artifact runs/default/artifact.pt
```

The script `make_figures_tables.py` can generate `score_correlation_heatmap.pdf`, matching the LaTeX figure name used in the manuscript.

## Simulation Experiment Demonstration

https://private-user-images.githubusercontent.com/164164346/594583186-a51e0273-ccc1-4cb3-97aa-288b57f9e143.mp4?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzkxNzk4NjksIm5iZiI6MTc3OTE3OTU2OSwicGF0aCI6Ii8xNjQxNjQzNDYvNTk0NTgzMTg2LWE1MWUwMjczLWNjYzEtNGNiMy05N2FhLTI4OGI1N2Y5ZTE0My5tcDQ_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwNTE5JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDUxOVQwODMyNDlaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT0xYjk0NTNhNDFkM2UyMjYyYmM1ODY5NmNhMGI4MDRiOWEyY2JkMTgyM2VhY2RkYjA5N2IyODZmYjE0ZDE2Y2JmJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZyZXNwb25zZS1jb250ZW50LXR5cGU9dmlkZW8lMkZtcDQifQ.PAjxKFm_eUET2FQxJZoHvz6uR9i1o4tBq7x-XD_RYhg
