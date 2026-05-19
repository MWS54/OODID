from __future__ import annotations

import importlib
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ucs_oodid.realtime import OnlineDetector, StreamingWindowBuffer
from ucs_oodid.dataset_registry import (
    SIMULATION_DATASET_BINDINGS,
    bounded_dataset_count,
    dataset_display_for_uav as registry_dataset_display_for_uav,
    dataset_display_name as registry_dataset_display_name,
    dataset_name_for_uav as registry_dataset_name_for_uav,
    default_simulation_uav_ids,
)
from ucs_oodid.io import save_json
from ucs_oodid.simulator.attack_replay import ATTACK_REPLAY_MODES, ATTACK_TYPE_LABEL_ALIASES, AttackReplayPool
from ucs_oodid.simulator.attacks import SUPPORTED_ATTACK_TYPES, AttackInjector
from ucs_oodid.simulator.demo_config import load_demo_scene_config
from ucs_oodid.simulator.engine import SimulationConfig, SimulationEngine
from ucs_oodid.simulator.entities import Attacker, GCS, UAV
from ucs_oodid.simulator.mission import SUPPORTED_MISSION_CONTEXTS
from ucs_oodid.simulator.response import SUPPORTED_RESPONSE_ACTIONS
from ucs_oodid.simulator.scenario import AttackEvent as ScenarioAttackEvent
from ucs_oodid.simulator.scenario import ScenarioConfig, build_fleet

DEFAULT_CONFIG: dict[str, Any] = load_demo_scene_config()
DEFAULT_CONFIG["attack_mode"] = "mixed_attack"
PURE_IDS_ARTIFACT_PATH = str(
    (ROOT / "runs/multi_uav_6_offline_transformer_only_pureids_20260504_0200_rerun/artifact.pt").resolve()
)
PURE_IDS_CSV_PATH = "data/multi_uav_hetero_no_uav04_plus_uav06_uav07_experiment.csv"
PURE_IDS_WINDOW_SIZE = 16
PURE_IDS_STRIDE = 8
PURE_IDS_REPLAY_RECORDS_PER_STEP = 186
SIMULATION_MIXED_REPLAY_LABEL = "Simulation mixed replay"
PURE_IDS_CSV_REPLAY_LABEL = "Pure IDS CSV original-order replay"
DATA_SOURCE_MODE_OPTIONS: tuple[str, ...] = (
    SIMULATION_MIXED_REPLAY_LABEL,
    PURE_IDS_CSV_REPLAY_LABEL,
)
DATA_SOURCE_MODE_WIDGET_KEY = "cfg_data_source_mode"
REQUIRED_WIDGET_CONFIG_DEFAULTS: dict[str, Any] = {
    "uav_count": 6,
    "duration_s": 600.0,
    "dt_s": 1.0,
    "seed": 42,
    "scenario_profile_path": "configs/simulator_default.yaml",
    "start_spacing_s": 3.0,
    "route_length_m": 600.0,
    "hover_duration_s": 10.0,
    "cruise_altitude_m": 90.0,
    "cruise_speed_mps": 17.0,
    "battery_capacity_wh": 180.0,
    "attack_mode": "mixed_attack",
    "attack_type": "jamming_proxy",
    "attack_start_s": 18.0,
    "attack_end_s": 42.0,
    "attack_intensity": 0.92,
    "attack_replay_mode": "loop",
    "records_per_uav_per_step": PURE_IDS_REPLAY_RECORDS_PER_STEP,
    "target_records_per_uav": 0,
    "enable_online_detection": True,
    "response_strategy": "conservative_mode",
    "artifact_path": PURE_IDS_ARTIFACT_PATH,
    "use_offline_calibrator_for_demo": True,
    "bootstrap_duration_s": 48.0,
    "window_size": PURE_IDS_WINDOW_SIZE,
    "stride": PURE_IDS_STRIDE,
    "dataset_replay_mode": "",
    "pure_ids_csv_path": "",
    "pure_ids_group_col": "uav_id",
    "pure_ids_replay_records_per_uav_per_step": PURE_IDS_REPLAY_RECORDS_PER_STEP,
    "pure_ids_replay_stop_when_exhausted": True,
    "pure_ids_replay_keep_original_order": True,
    "pure_ids_replay_no_mixing": True,
    "pure_ids_replay_split_filter": ["test_id", "test_ood"],
    "pure_ids_replay_reset_buffer_on_split_change": True,
    "bank_k": 5,
    "bootstrap_q_ood": 0.90,
    "bootstrap_threshold_margin": 0.05,
    "top_records": 5,
    "export_summary_only": False,
}
for key, value in REQUIRED_WIDGET_CONFIG_DEFAULTS.items():
    DEFAULT_CONFIG.setdefault(key, value)

ATTACK_MODE_OPTIONS: tuple[str, ...] = ("single_attack", "mixed_attack")
FIXED_UAV_DATASET_BINDINGS: tuple[dict[str, Any], ...] = tuple(
    {
        **dict(row),
        "dataset_display": registry_dataset_display_name(str(row["dataset_name"]), annotate_role=True),
    }
    for row in SIMULATION_DATASET_BINDINGS
)
FIXED_UAV_METADATA_BY_ID: dict[str, dict[str, Any]] = {
    str(row["uav_id"]): dict(row) for row in FIXED_UAV_DATASET_BINDINGS
}
UAV_ID_ORDER: tuple[str, ...] = tuple(str(row["uav_id"]) for row in FIXED_UAV_DATASET_BINDINGS)
UAV_ID_ORDER_INDEX: dict[str, int] = {uav_id: idx for idx, uav_id in enumerate(UAV_ID_ORDER)}
DEFAULT_DASHBOARD_SELECTED_UAV_IDS: tuple[str, ...] = UAV_ID_ORDER
DEFAULT_CONFIG["uav_count"] = bounded_dataset_count(DEFAULT_CONFIG.get("uav_count", 3))
DEFAULT_ATTACK_PLAN_SETTINGS: dict[str, dict[str, Any]] = {
    "uav_01": {
        "attack_types": ["jamming_proxy"],
        "attack_start_s": 18.0,
        "attack_duration_s": 24.0,
        "attack_intensity": 0.92,
    },
    "uav_02": {
        "attack_types": ["replay"],
        "attack_start_s": 26.0,
        "attack_duration_s": 14.0,
        "attack_intensity": 0.70,
    },
    "uav_03": {
        "attack_types": ["command_injection"],
        "attack_start_s": 34.0,
        "attack_duration_s": 16.0,
        "attack_intensity": 0.88,
    },
    "uav_05": {
        "attack_types": ["scan"],
        "attack_start_s": 42.0,
        "attack_duration_s": 18.0,
        "attack_intensity": 0.85,
    },
    "uav_06": {
        "attack_types": ["command_injection"],
        "attack_start_s": 46.0,
        "attack_duration_s": 14.0,
        "attack_intensity": 0.82,
    },
    "uav_07": {
        "attack_types": ["flood"],
        "attack_start_s": 50.0,
        "attack_duration_s": 16.0,
        "attack_intensity": 0.84,
    },
}
CANONICAL_ATTACK_OPTIONS: tuple[str, ...] = tuple(attack for attack in SUPPORTED_ATTACK_TYPES if attack != "benign")
ATTACK_LABEL_ALIASES = {
    "jamming_proxy": "jamming",
    "flood": "udp_flooding",
    "scan": "recon_scanning",
    "command_injection": "injection",
    "replay": "replay",
    "benign": "benign",
}

RECORD_COLUMNS = [
    "timestamp",
    "uav_id",
    "mission_phase",
    "mission_context",
    "battery_soc",
    "speed",
    "altitude",
    "rssi",
    "snr",
    "latency_ms",
    "loss_rate",
    "distance_to_gcs",
    "wind_level",
    "obstacle_factor",
    "bytes_up",
    "bytes_down",
    "attack_active",
    "attack_type",
    "cpu_load",
    "board_temperature_c",
    "response_action",
    "response_time",
    "response_reason",
    "flight_energy_wh",
    "communication_energy_wh",
    "detection_energy_wh",
    "total_energy_wh",
    "cumulative_energy_wh",
    "sim_time",
    "dataset_name",
    "attack_source_dataset",
    "record_kind",
    "record_id",
    "label",
    "throughput_bytes",
]
DETECTION_COLUMNS = [
    "simulation_time_s",
    "window_id",
    "group_id",
    "raw_ood_score",
    "normalized_ood_score",
    "ood_score",
    "threshold",
    "ood_threshold",
    "threshold_source",
    "is_ood",
    "ood_alert",
    "known_attack_alert",
    "known_attack_pred_labels",
    "alert_reason",
    "alert_level",
    "has_alert",
    "false_alert",
    "attack_active",
    "ground_truth_is_ood",
    "window_partition",
    "attack_types",
    "dominant_ood_source",
    "response_action",
    "response_time",
    "response_reason",
    "response_triggered",
    "record_count",
]
ALERT_REASON_ORDER = ["known_attack", "ood", "known_attack+ood", "none"]
STREAMLIT_EXPORT_DIR = ROOT / "runs" / "streamlit_dashboard_exports"
STREAMLIT_EXPORT_LATEST_FILENAME = "streamlit_simulation_result_latest.json"
RESPONSE_COLUMNS = ["uav_id", "response_action", "response_time", "response_reason"]
ALERT_LEVEL_RANK = {"normal": 0, "watch": 1, "warning": 2, "critical": 3}
PHASE_ORDER = ["idle", "takeoff", "climb", "cruise", "hover", "return_home", "land", "emergency"]
WIDGET_KEYS = {name: f"cfg_{name}" for name in DEFAULT_CONFIG}
SELECTED_UAVS_WIDGET_KEY = "cfg_selected_uav_ids"
PENDING_MISSION_WIDGET_SYNC_KEY = "_pending_mission_widget_sync"


st.set_page_config(page_title="UAV IDS Dashboard", layout="wide")

SIMULATION_ENABLED = True
SIMULATION_DISABLED_MESSAGE = (
    "Mission simulation is currently disabled. The dashboard will not run the built-in demo "
    "or execute the current mission."
)


def clone_default_config() -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    config["pure_ids_replay_split_filter"] = list(DEFAULT_CONFIG["pure_ids_replay_split_filter"])
    config["per_uav_attack_plans"] = clone_default_attack_plans()
    return config


def clone_dashboard_default_config() -> dict[str, Any]:
    config = clone_default_config()
    config["selected_uav_ids"] = list(DEFAULT_DASHBOARD_SELECTED_UAV_IDS)
    config["uav_count"] = len(config["selected_uav_ids"])
    return config


def clone_pure_ids_csv_replay_config() -> dict[str, Any]:
    config = clone_dashboard_default_config()
    config["dataset_replay_mode"] = "pure_ids_csv"
    config["pure_ids_csv_path"] = PURE_IDS_CSV_PATH
    config["pure_ids_group_col"] = "uav_id"
    config["pure_ids_replay_records_per_uav_per_step"] = PURE_IDS_REPLAY_RECORDS_PER_STEP
    config["pure_ids_replay_stop_when_exhausted"] = True
    config["pure_ids_replay_keep_original_order"] = True
    config["pure_ids_replay_no_mixing"] = True
    config["pure_ids_replay_split_filter"] = ["test_id", "test_ood"]
    config["pure_ids_replay_reset_buffer_on_split_change"] = True
    config["enable_online_detection"] = True
    config["records_per_uav_per_step"] = PURE_IDS_REPLAY_RECORDS_PER_STEP
    config["target_records_per_uav"] = 0
    config["artifact_path"] = PURE_IDS_ARTIFACT_PATH
    config["window_size"] = PURE_IDS_WINDOW_SIZE
    config["stride"] = PURE_IDS_STRIDE
    config["use_offline_calibrator_for_demo"] = True
    config["export_summary_only"] = False
    return config


def clone_pure_ids_demo_config() -> dict[str, Any]:
    return clone_pure_ids_csv_replay_config()


def bounded_uav_count(value: Any) -> int:
    return bounded_dataset_count(value)


def attack_plan_widget_key(uav_id: str, field: str) -> str:
    return f"cfg_{uav_id}_{field}"


def normalize_attack_selection(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple, set)):
        values = list(value)
    else:
        values = [str(value)]
    ordered: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = str(item).strip().lower()
        if text and text not in seen:
            ordered.append(text)
            seen.add(text)
    return ordered


def normalize_pure_ids_replay_split_filter(value: Any) -> list[str]:
    if value is None:
        return list(DEFAULT_CONFIG["pure_ids_replay_split_filter"])
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple, set)):
        values = list(value)
    else:
        values = [str(value)]
    ordered: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = str(item).strip().lower()
        if text in {"id", "test_id"}:
            text = "test_id"
        elif text in {"ood", "test_ood"}:
            text = "test_ood"
        if text and text not in seen:
            ordered.append(text)
            seen.add(text)
    return ordered or list(DEFAULT_CONFIG["pure_ids_replay_split_filter"])


def pure_ids_mode_overrides() -> dict[str, Any]:
    return {
        "dataset_replay_mode": "pure_ids_csv",
        "pure_ids_csv_path": PURE_IDS_CSV_PATH,
        "pure_ids_group_col": "uav_id",
        "pure_ids_replay_records_per_uav_per_step": PURE_IDS_REPLAY_RECORDS_PER_STEP,
        "pure_ids_replay_stop_when_exhausted": True,
        "pure_ids_replay_keep_original_order": True,
        "pure_ids_replay_no_mixing": True,
        "pure_ids_replay_split_filter": ["test_id", "test_ood"],
        "pure_ids_replay_reset_buffer_on_split_change": True,
        "enable_online_detection": True,
        "records_per_uav_per_step": PURE_IDS_REPLAY_RECORDS_PER_STEP,
        "target_records_per_uav": 0,
        "artifact_path": PURE_IDS_ARTIFACT_PATH,
        "window_size": PURE_IDS_WINDOW_SIZE,
        "stride": PURE_IDS_STRIDE,
        "use_offline_calibrator_for_demo": True,
        "export_summary_only": False,
    }


def data_source_mode_label_for_config(config: Mapping[str, Any] | None) -> str:
    dataset_replay_mode = (
        ""
        if not isinstance(config, Mapping)
        else str(config.get("dataset_replay_mode", "") or "").strip().lower()
    )
    return PURE_IDS_CSV_REPLAY_LABEL if dataset_replay_mode == "pure_ids_csv" else SIMULATION_MIXED_REPLAY_LABEL


def apply_data_source_mode_widget_overrides(mode_label: str) -> None:
    if mode_label == PURE_IDS_CSV_REPLAY_LABEL:
        for key, value in pure_ids_mode_overrides().items():
            st.session_state[WIDGET_KEYS[key]] = value
    else:
        st.session_state[WIDGET_KEYS["dataset_replay_mode"]] = ""
        st.session_state[WIDGET_KEYS["use_offline_calibrator_for_demo"]] = True


def apply_data_source_mode_to_config(config: Mapping[str, Any], mode_label: str) -> dict[str, Any]:
    merged = dict(config)
    if mode_label == PURE_IDS_CSV_REPLAY_LABEL:
        merged.update(pure_ids_mode_overrides())
    else:
        merged["dataset_replay_mode"] = ""
        merged["use_offline_calibrator_for_demo"] = True
    merged["pure_ids_replay_split_filter"] = normalize_pure_ids_replay_split_filter(
        merged.get("pure_ids_replay_split_filter")
    )
    return merged


def uav_metadata_for_uav(uav_id: str) -> dict[str, Any]:
    canonical_uav_id = canonical_uav_id_text(uav_id)
    return dict(FIXED_UAV_METADATA_BY_ID.get(canonical_uav_id, {"uav_id": canonical_uav_id}))


def canonical_uav_id_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return str(value).strip()


def uav_source_type(uav_id: str) -> str:
    return str(uav_metadata_for_uav(uav_id).get("source_type", "") or "")


def uav_simulation_role(uav_id: str) -> str:
    return str(uav_metadata_for_uav(uav_id).get("simulation_role", "") or "")


def uav_source_note(uav_id: str) -> str:
    return str(uav_metadata_for_uav(uav_id).get("source_note", "") or "")


def source_label_for_uav(uav_id: str) -> str:
    canonical_uav_id = canonical_uav_id_text(uav_id)
    if not canonical_uav_id:
        return ""
    dataset_display = dataset_display_for_uav(canonical_uav_id, annotate_role=False)
    return f"{canonical_uav_id} | {dataset_display}" if dataset_display else canonical_uav_id


def uav_selection_label(uav_id: str) -> str:
    spec = uav_metadata_for_uav(uav_id)
    dataset_display = dataset_display_for_uav(uav_id, annotate_role=False)
    return (
        f"{str(uav_id).strip()} | {dataset_display} | "
        f"dataset_name={spec.get('dataset_name', '')} | source_type={spec.get('source_type', '')}"
    )


def ordered_uav_ids(values: Sequence[Any]) -> list[str]:
    normalized = {canonical_uav_id_text(value) for value in values if canonical_uav_id_text(value)}
    ordered = [uav_id for uav_id in UAV_ID_ORDER if uav_id in normalized]
    extras = sorted(value for value in normalized if value not in UAV_ID_ORDER_INDEX)
    return ordered + extras


def normalize_selected_uav_ids(
    value: Any,
    *,
    fallback_count: int | None = None,
    allow_empty: bool = False,
) -> list[str]:
    if value is None:
        candidates: list[Any] = []
    elif isinstance(value, str):
        candidates = [value]
    elif isinstance(value, (list, tuple, set)):
        candidates = list(value)
    else:
        candidates = [value]
    selected = ordered_uav_ids(candidates)
    if selected or allow_empty:
        return selected
    if fallback_count is not None:
        return default_simulation_uav_ids(bounded_uav_count(fallback_count))
    return []


def selected_uav_ids_for_config(config: Mapping[str, Any] | None) -> list[str]:
    if not isinstance(config, Mapping):
        return []
    if "selected_uav_ids" in config:
        return normalize_selected_uav_ids(config.get("selected_uav_ids"), allow_empty=True)
    return normalize_selected_uav_ids(
        None,
        fallback_count=int(config.get("uav_count", DEFAULT_CONFIG["uav_count"])),
    )


def active_uav_specs(
    uav_count: int | None = None,
    *,
    selected_uav_ids: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    if selected_uav_ids is not None:
        return [uav_metadata_for_uav(uav_id) for uav_id in normalize_selected_uav_ids(selected_uav_ids, allow_empty=True)]
    count = bounded_uav_count(uav_count if uav_count is not None else 1)
    return [dict(row) for row in FIXED_UAV_DATASET_BINDINGS[:count]]


def attach_uav_metadata(frame: pd.DataFrame, *, uav_col: str = "uav_id") -> pd.DataFrame:
    if frame.empty or uav_col not in frame.columns:
        return frame.copy()
    work = frame.copy()
    work["dataset_name"] = work[uav_col].map(lambda value: dataset_name_for_uav(canonical_uav_id_text(value)))
    work["dataset_display"] = work[uav_col].map(lambda value: dataset_display_for_uav(canonical_uav_id_text(value), annotate_role=False))
    work["source_type"] = work[uav_col].map(lambda value: uav_source_type(canonical_uav_id_text(value)))
    work["simulation_role"] = work[uav_col].map(lambda value: uav_simulation_role(canonical_uav_id_text(value)))
    work["source_note"] = work[uav_col].map(lambda value: uav_source_note(canonical_uav_id_text(value)))
    work["source_label"] = work[uav_col].map(lambda value: source_label_for_uav(canonical_uav_id_text(value)))
    return work


def sort_frame_by_uav_order(
    frame: pd.DataFrame,
    *,
    uav_col: str = "uav_id",
    prefix_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    work = frame.copy()
    sort_columns = [column for column in list(prefix_cols or ()) if column in work.columns]
    if uav_col not in work.columns:
        if sort_columns:
            return work.sort_values(sort_columns, kind="stable").reset_index(drop=True)
        return work.reset_index(drop=True)
    rank_map = {uav_id: idx for idx, uav_id in enumerate(ordered_uav_ids(work[uav_col].tolist()))}
    work["_uav_sort_order"] = work[uav_col].map(lambda value: rank_map.get(canonical_uav_id_text(value), len(rank_map)))
    work = work.sort_values([*sort_columns, "_uav_sort_order"], kind="stable").drop(columns="_uav_sort_order")
    return work.reset_index(drop=True)


def dataset_name_for_uav(uav_id: str) -> str:
    return registry_dataset_name_for_uav(str(uav_id).strip())


def dataset_display_name(dataset_name: str, *, annotate_role: bool = True) -> str:
    return registry_dataset_display_name(str(dataset_name).strip(), annotate_role=annotate_role)


def dataset_display_for_uav(uav_id: str, *, annotate_role: bool = True) -> str:
    return registry_dataset_display_for_uav(str(uav_id).strip(), annotate_role=annotate_role)


def attack_type_summary(attack_types: list[str] | tuple[str, ...]) -> str:
    tokens = normalize_attack_selection(list(attack_types))
    if not tokens:
        return "benign"
    if len(tokens) == 1:
        return tokens[0]
    return " + ".join(tokens)


def clone_default_attack_plans() -> dict[str, dict[str, Any]]:
    plans: dict[str, dict[str, Any]] = {}
    for row in FIXED_UAV_DATASET_BINDINGS:
        uav_id = row["uav_id"]
        defaults = DEFAULT_ATTACK_PLAN_SETTINGS.get(
            uav_id,
            {
                "attack_types": ["replay"],
                "attack_start_s": 10.0,
                "attack_duration_s": 10.0,
                "attack_intensity": 1.0,
            },
        )
        plans[uav_id] = {
            "uav_id": uav_id,
            "source_dataset": row["dataset_name"],
            "source_dataset_display": row["dataset_display"],
            "attack_types": list(defaults["attack_types"]),
            "attack_start_s": float(defaults["attack_start_s"]),
            "attack_duration_s": float(defaults["attack_duration_s"]),
            "attack_intensity": float(defaults["attack_intensity"]),
        }
    return plans


def merge_attack_plans(existing: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    plans = clone_default_attack_plans()
    if not isinstance(existing, Mapping):
        return plans
    for uav_id, plan in existing.items():
        canonical_uav_id = str(uav_id).strip()
        if canonical_uav_id not in plans or not isinstance(plan, Mapping):
            continue
        plans[canonical_uav_id]["attack_types"] = normalize_attack_selection(
            plan.get("attack_types", plans[canonical_uav_id]["attack_types"])
        ) or list(plans[canonical_uav_id]["attack_types"])
        plans[canonical_uav_id]["attack_start_s"] = float(
            plan.get("attack_start_s", plans[canonical_uav_id]["attack_start_s"])
        )
        plans[canonical_uav_id]["attack_duration_s"] = float(
            plan.get("attack_duration_s", plans[canonical_uav_id]["attack_duration_s"])
        )
        plans[canonical_uav_id]["attack_intensity"] = float(
            plan.get("attack_intensity", plans[canonical_uav_id]["attack_intensity"])
        )
    return plans


def merge_dashboard_config(existing: Mapping[str, Any] | None) -> dict[str, Any]:
    config = clone_default_config()
    if not isinstance(existing, Mapping):
        config["selected_uav_ids"] = normalize_selected_uav_ids(
            None,
            fallback_count=int(config["uav_count"]),
        )
        config["uav_count"] = len(config["selected_uav_ids"])
        return config
    for key, value in existing.items():
        if key == "per_uav_attack_plans":
            config[key] = merge_attack_plans(value if isinstance(value, Mapping) else None)
            continue
        config[str(key)] = value
    if "selected_uav_ids" in existing:
        config["selected_uav_ids"] = normalize_selected_uav_ids(existing.get("selected_uav_ids"), allow_empty=True)
    else:
        config["selected_uav_ids"] = normalize_selected_uav_ids(
            None,
            fallback_count=int(config.get("uav_count", DEFAULT_CONFIG["uav_count"])),
        )
    config["uav_count"] = len(config["selected_uav_ids"])
    config["attack_mode"] = str(config.get("attack_mode", DEFAULT_CONFIG["attack_mode"]) or DEFAULT_CONFIG["attack_mode"]).strip()
    if config["attack_mode"] not in ATTACK_MODE_OPTIONS:
        config["attack_mode"] = DEFAULT_CONFIG["attack_mode"]
    config["use_offline_calibrator_for_demo"] = True
    config["per_uav_attack_plans"] = merge_attack_plans(config.get("per_uav_attack_plans"))
    return config


def sync_selected_uav_widget_state(
    selected_uav_ids: Sequence[str] | None,
    *,
    fallback_count: int,
    overwrite: bool,
) -> None:
    if overwrite or SELECTED_UAVS_WIDGET_KEY not in st.session_state:
        st.session_state[SELECTED_UAVS_WIDGET_KEY] = normalize_selected_uav_ids(
            selected_uav_ids,
            fallback_count=fallback_count,
            allow_empty=True,
        )


def sync_attack_plan_widget_state(plans: Mapping[str, Any] | None, *, overwrite: bool) -> None:
    merged = merge_attack_plans(plans if isinstance(plans, Mapping) else None)
    for uav_id, plan in merged.items():
        selected_types = normalize_attack_selection(plan.get("attack_types", [])) or list(DEFAULT_ATTACK_PLAN_SETTINGS[uav_id]["attack_types"])
        single_key = attack_plan_widget_key(uav_id, "attack_type_single")
        multi_key = attack_plan_widget_key(uav_id, "attack_types_multi")
        start_key = attack_plan_widget_key(uav_id, "attack_start_s")
        duration_key = attack_plan_widget_key(uav_id, "attack_duration_s")
        intensity_key = attack_plan_widget_key(uav_id, "attack_intensity")
        if overwrite or single_key not in st.session_state:
            st.session_state[single_key] = selected_types[0]
        if overwrite or multi_key not in st.session_state:
            st.session_state[multi_key] = selected_types
        if overwrite or start_key not in st.session_state:
            st.session_state[start_key] = float(plan["attack_start_s"])
        if overwrite or duration_key not in st.session_state:
            st.session_state[duration_key] = float(plan["attack_duration_s"])
        if overwrite or intensity_key not in st.session_state:
            st.session_state[intensity_key] = float(plan["attack_intensity"])


def sync_widget_state(config: dict[str, Any], *, overwrite: bool) -> None:
    for name, widget_key in WIDGET_KEYS.items():
        if name not in config:
            continue
        value = config[name]
        if name == "uav_count":
            value = bounded_uav_count(value)
        if overwrite or widget_key not in st.session_state:
            st.session_state[widget_key] = value
    # This hidden flag should follow the normalized config instead of stale session state.
    st.session_state[WIDGET_KEYS["use_offline_calibrator_for_demo"]] = bool(
        config.get("use_offline_calibrator_for_demo", True)
    )
    if overwrite or DATA_SOURCE_MODE_WIDGET_KEY not in st.session_state:
        st.session_state[DATA_SOURCE_MODE_WIDGET_KEY] = data_source_mode_label_for_config(config)
    sync_selected_uav_widget_state(
        config.get("selected_uav_ids"),
        fallback_count=int(config.get("uav_count", DEFAULT_CONFIG["uav_count"])),
        overwrite=overwrite,
    )
    sync_attack_plan_widget_state(config.get("per_uav_attack_plans"), overwrite=overwrite)


def queue_mission_widget_sync(config: Mapping[str, Any] | None) -> None:
    st.session_state[PENDING_MISSION_WIDGET_SYNC_KEY] = merge_dashboard_config(config)


def apply_pending_mission_widget_sync() -> None:
    pending = st.session_state.pop(PENDING_MISSION_WIDGET_SYNC_KEY, None)
    if pending is None:
        return
    config = merge_dashboard_config(pending if isinstance(pending, Mapping) else st.session_state.get("dashboard_config"))
    st.session_state.dashboard_config = config
    sync_widget_state(config, overwrite=True)


def init_session_state() -> None:
    if "dashboard_config" not in st.session_state:
        st.session_state.dashboard_config = clone_dashboard_default_config()
    config = merge_dashboard_config(st.session_state.get("dashboard_config"))
    st.session_state.dashboard_config = config
    sync_widget_state(config, overwrite=False)
    if "dashboard_payload" not in st.session_state:
        st.session_state.dashboard_payload = None
    if not SIMULATION_ENABLED:
        st.session_state.dashboard_payload = None
    if "dashboard_error" not in st.session_state:
        st.session_state.dashboard_error = ""
    if "live_time_index" not in st.session_state:
        st.session_state.live_time_index = 0
    if "replay_time_index" not in st.session_state:
        st.session_state.replay_time_index = 0
    if "live_selected_uav" not in st.session_state:
        st.session_state.live_selected_uav = "uav_01"
    if "replay_selected_uav" not in st.session_state:
        st.session_state.replay_selected_uav = "uav_01"
    if "analysis_selected_uav" not in st.session_state:
        st.session_state.analysis_selected_uav = "uav_01"


@st.cache_resource(show_spinner=False)
def cached_bootstrap_artifact(
    uav_count: int,
    selected_uav_ids: tuple[str, ...],
    route_length_m: float,
    hover_duration_s: float,
    cruise_altitude_m: float,
    cruise_speed_mps: float,
    dt_s: float,
    seed: int,
    scenario_profile_path: str,
    bootstrap_duration_s: float,
    window_size: int,
    stride: int,
    bank_k: int,
    bootstrap_q_ood: float,
    bootstrap_threshold_margin: float,
) -> dict[str, Any]:
    from scripts.simulate_live_demo import build_bootstrap_artifact

    args = SimpleNamespace(
        uav_id="uav_bootstrap_01",
        uav_count=int(len(selected_uav_ids) or uav_count),
        selected_uav_ids=list(selected_uav_ids),
        duration_s=float(bootstrap_duration_s),
        dt_s=float(dt_s),
        seed=int(seed),
        scenario_profile_path=str(scenario_profile_path),
        route_length_m=float(route_length_m),
        hover_duration_s=float(hover_duration_s),
        cruise_altitude_m=float(cruise_altitude_m),
        cruise_speed_mps=float(cruise_speed_mps),
        battery_capacity_wh=180.0,
        start_spacing_s=3.0,
        attack_type="benign",
        attack_start_s=0.0,
        attack_end_s=0.0,
        attack_intensity=0.0,
        response_strategy="alert_only",
        artifact_out="",
        bootstrap_duration_s=float(bootstrap_duration_s),
        window_size=int(window_size),
        stride=int(stride),
        bank_k=int(bank_k),
        bootstrap_q_ood=float(bootstrap_q_ood),
        bootstrap_threshold_margin=float(bootstrap_threshold_margin),
    )
    return build_bootstrap_artifact(args)


@st.cache_resource(show_spinner=False)
def cached_attack_replay_pool(seed: int) -> AttackReplayPool | None:
    return AttackReplayPool.from_default_paths(seed=int(seed), allow_missing=True)


def attack_type_options_for_uav(uav_id: str, seed: int) -> list[str]:
    default_options = list(CANONICAL_ATTACK_OPTIONS)
    pool = cached_attack_replay_pool(int(seed))
    if pool is None or not pool.has_binding(str(uav_id).strip()):
        return default_options
    available_labels = {str(label).strip().lower() for label in pool.available_labels(str(uav_id).strip())}
    options: list[str] = []
    for attack_type in CANONICAL_ATTACK_OPTIONS:
        aliases = set(ATTACK_TYPE_LABEL_ALIASES.get(attack_type, ()))
        aliases.add(attack_type)
        if available_labels.intersection({str(alias).strip().lower() for alias in aliases}):
            options.append(attack_type)
    return options or default_options


def current_selected_uav_ids_from_widgets() -> list[str]:
    return normalize_selected_uav_ids(
        st.session_state.get(SELECTED_UAVS_WIDGET_KEY),
        allow_empty=True,
    )


def current_per_uav_attack_plans_from_widgets(attack_mode: str) -> dict[str, dict[str, Any]]:
    existing_config = st.session_state.get("dashboard_config")
    existing_plans = None
    if isinstance(existing_config, Mapping):
        existing_plans = existing_config.get("per_uav_attack_plans")
    plans = merge_attack_plans(existing_plans if isinstance(existing_plans, Mapping) else None)
    for row in FIXED_UAV_DATASET_BINDINGS:
        uav_id = row["uav_id"]
        single_key = attack_plan_widget_key(uav_id, "attack_type_single")
        multi_key = attack_plan_widget_key(uav_id, "attack_types_multi")
        single_selection = str(st.session_state.get(single_key, "")).strip().lower()
        multi_selection = normalize_attack_selection(st.session_state.get(multi_key, []))
        if attack_mode == "single_attack":
            selected_types = multi_selection[:1] or ([single_selection] if single_selection else [])
        else:
            selected_types = multi_selection or ([single_selection] if single_selection else [])
        if not selected_types:
            selected_types = list(plans[uav_id]["attack_types"])
        plans[uav_id] = {
            "uav_id": uav_id,
            "source_dataset": row["dataset_name"],
            "source_dataset_display": row["dataset_display"],
            "attack_types": selected_types,
            "attack_start_s": float(st.session_state.get(attack_plan_widget_key(uav_id, "attack_start_s"), plans[uav_id]["attack_start_s"])),
            "attack_duration_s": float(
                st.session_state.get(attack_plan_widget_key(uav_id, "attack_duration_s"), plans[uav_id]["attack_duration_s"])
            ),
            "attack_intensity": float(
                st.session_state.get(attack_plan_widget_key(uav_id, "attack_intensity"), plans[uav_id]["attack_intensity"])
            ),
        }
    return plans


def current_config_from_widgets() -> dict[str, Any]:
    attack_mode = str(st.session_state[WIDGET_KEYS["attack_mode"]]).strip().lower()
    if attack_mode not in ATTACK_MODE_OPTIONS:
        attack_mode = DEFAULT_CONFIG["attack_mode"]
    selected_uav_ids = current_selected_uav_ids_from_widgets()
    config = {
        "uav_count": len(selected_uav_ids),
        "selected_uav_ids": selected_uav_ids,
        "duration_s": float(st.session_state[WIDGET_KEYS["duration_s"]]),
        "dt_s": float(st.session_state[WIDGET_KEYS["dt_s"]]),
        "records_per_uav_per_step": int(st.session_state[WIDGET_KEYS["records_per_uav_per_step"]]),
        "target_records_per_uav": int(st.session_state[WIDGET_KEYS["target_records_per_uav"]]),
        "seed": int(st.session_state[WIDGET_KEYS["seed"]]),
        "scenario_profile_path": str(st.session_state[WIDGET_KEYS["scenario_profile_path"]]).strip(),
        "start_spacing_s": float(st.session_state[WIDGET_KEYS["start_spacing_s"]]),
        "route_length_m": float(st.session_state[WIDGET_KEYS["route_length_m"]]),
        "hover_duration_s": float(st.session_state[WIDGET_KEYS["hover_duration_s"]]),
        "cruise_altitude_m": float(st.session_state[WIDGET_KEYS["cruise_altitude_m"]]),
        "cruise_speed_mps": float(st.session_state[WIDGET_KEYS["cruise_speed_mps"]]),
        "battery_capacity_wh": float(st.session_state[WIDGET_KEYS["battery_capacity_wh"]]),
        "attack_type": str(st.session_state[WIDGET_KEYS["attack_type"]]),
        "attack_start_s": float(st.session_state[WIDGET_KEYS["attack_start_s"]]),
        "attack_end_s": float(st.session_state[WIDGET_KEYS["attack_end_s"]]),
        "attack_intensity": float(st.session_state[WIDGET_KEYS["attack_intensity"]]),
        "attack_replay_mode": str(st.session_state[WIDGET_KEYS["attack_replay_mode"]]),
        "enable_online_detection": bool(st.session_state[WIDGET_KEYS["enable_online_detection"]]),
        "response_strategy": str(st.session_state[WIDGET_KEYS["response_strategy"]]),
        "artifact_path": str(st.session_state[WIDGET_KEYS["artifact_path"]]).strip(),
        "use_offline_calibrator_for_demo": bool(st.session_state[WIDGET_KEYS["use_offline_calibrator_for_demo"]]),
        "bootstrap_duration_s": float(st.session_state[WIDGET_KEYS["bootstrap_duration_s"]]),
        "window_size": int(st.session_state[WIDGET_KEYS["window_size"]]),
        "stride": int(st.session_state[WIDGET_KEYS["stride"]]),
        "dataset_replay_mode": str(st.session_state[WIDGET_KEYS["dataset_replay_mode"]]).strip(),
        "pure_ids_csv_path": str(st.session_state[WIDGET_KEYS["pure_ids_csv_path"]]).strip(),
        "pure_ids_group_col": str(st.session_state[WIDGET_KEYS["pure_ids_group_col"]]).strip(),
        "pure_ids_replay_records_per_uav_per_step": int(
            st.session_state[WIDGET_KEYS["pure_ids_replay_records_per_uav_per_step"]]
        ),
        "pure_ids_replay_stop_when_exhausted": bool(
            st.session_state[WIDGET_KEYS["pure_ids_replay_stop_when_exhausted"]]
        ),
        "pure_ids_replay_keep_original_order": bool(
            st.session_state[WIDGET_KEYS["pure_ids_replay_keep_original_order"]]
        ),
        "pure_ids_replay_no_mixing": bool(st.session_state[WIDGET_KEYS["pure_ids_replay_no_mixing"]]),
        "pure_ids_replay_split_filter": normalize_pure_ids_replay_split_filter(
            st.session_state[WIDGET_KEYS["pure_ids_replay_split_filter"]]
        ),
        "pure_ids_replay_reset_buffer_on_split_change": bool(
            st.session_state[WIDGET_KEYS["pure_ids_replay_reset_buffer_on_split_change"]]
        ),
        "bank_k": int(st.session_state[WIDGET_KEYS["bank_k"]]),
        "bootstrap_q_ood": float(st.session_state[WIDGET_KEYS["bootstrap_q_ood"]]),
        "bootstrap_threshold_margin": float(st.session_state[WIDGET_KEYS["bootstrap_threshold_margin"]]),
        "top_records": int(st.session_state[WIDGET_KEYS["top_records"]]),
        "export_summary_only": bool(st.session_state[WIDGET_KEYS["export_summary_only"]]),
        "attack_mode": attack_mode,
        "per_uav_attack_plans": current_per_uav_attack_plans_from_widgets(attack_mode),
    }
    return apply_data_source_mode_to_config(
        config,
        str(st.session_state.get(DATA_SOURCE_MODE_WIDGET_KEY, SIMULATION_MIXED_REPLAY_LABEL)),
    )


def resolve_input_path(path_text: str) -> Path:
    candidate = Path(path_text).expanduser()
    if candidate.is_absolute():
        return candidate
    return ROOT / candidate


def validate_artifact_path(path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix and suffix not in {".pt", ".pth"}:
        raise ValueError(
            f"Artifact path must point to a trained model file like artifact.pt, not {path.name}."
        )
    try:
        with path.open("rb") as handle:
            prefix = handle.read(64).lstrip()
    except OSError:
        return
    if prefix.startswith((b"{", b"[")):
        raise ValueError(
            f"Artifact path points to a text report, not a PyTorch model artifact: {path.name}. "
            "Please select the training output artifact.pt file."
        )


def build_uavs(config: dict[str, Any]) -> list[UAV]:
    uav_ids = selected_uav_ids_for_config(config)
    if not uav_ids:
        raise ValueError("Select at least one UAV to build the mission fleet.")
    return build_fleet(config, uav_ids=uav_ids)


def build_runtime_scenario_config(config: Mapping[str, Any]) -> ScenarioConfig:
    attack_mode = str(config.get("attack_mode", DEFAULT_CONFIG["attack_mode"]) or DEFAULT_CONFIG["attack_mode"]).strip().lower()
    if attack_mode not in ATTACK_MODE_OPTIONS:
        raise ValueError(f"Unsupported attack mode: {attack_mode}")
    plans = merge_attack_plans(config.get("per_uav_attack_plans") if isinstance(config.get("per_uav_attack_plans"), Mapping) else None)
    selected_uav_ids = selected_uav_ids_for_config(config)
    active_specs = active_uav_specs(selected_uav_ids=selected_uav_ids)
    if not active_specs:
        raise ValueError("Select at least one UAV before running the mission.")
    replay_mode = str(config.get("attack_replay_mode", "sequential") or "sequential").strip().lower()
    events: list[ScenarioAttackEvent] = []
    for spec in active_specs:
        uav_id = spec["uav_id"]
        plan = plans[uav_id]
        selected_types = normalize_attack_selection(plan.get("attack_types", []))
        if attack_mode == "single_attack":
            if len(selected_types) != 1:
                raise ValueError(f"{uav_id} must select exactly one attack type in single_attack mode.")
        else:
            if not selected_types:
                raise ValueError(f"{uav_id} must select at least one attack type in mixed_attack mode.")
        start_time = float(plan.get("attack_start_s", 0.0))
        duration = float(plan.get("attack_duration_s", 0.0))
        intensity = float(plan.get("attack_intensity", 0.0))
        if start_time < 0.0:
            raise ValueError(f"{uav_id} attack start must be non-negative.")
        if duration <= 0.0:
            raise ValueError(f"{uav_id} attack duration must be positive.")
        if intensity <= 0.0:
            raise ValueError(f"{uav_id} attack intensity must be positive.")
        events.append(
            ScenarioAttackEvent(
                uav_id=uav_id,
                source_dataset=spec["dataset_name"],
                attack_types=selected_types,
                start_time=start_time,
                duration=duration,
                intensity=intensity,
                replay_mode=replay_mode,
                is_ood=True,
            )
        )
    if attack_mode == "mixed_attack" and len(active_specs) == 1 and len(events[0].attack_types) < 2:
        raise ValueError("Mixed attack mode with one UAV requires selecting at least two attack types.")
    return ScenarioConfig(
        name=f"dashboard_{attack_mode}_{len(active_specs)}uav",
        scenario_type=attack_mode,
        attack_events=tuple(events),
        description="Runtime scenario generated from the Streamlit dashboard.",
        metadata={"source": "streamlit_dashboard"},
    )


def scenario_schedule_rows(scenario_config: ScenarioConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, event in enumerate(scenario_config.attack_events):
        rows.append(
            {
                "uav_id": event.uav_id,
                "attack_type": attack_type_summary(list(event.attack_types)),
                "start_s": float(event.start_time),
                "end_s": float(event.end_time),
                "duration_s": float(event.end_time - event.start_time),
                "intensity": float(event.intensity),
                "source_dataset": str(event.source_dataset),
                "source_dataset_display": dataset_display_name(str(event.source_dataset)),
                "replay_mode": str(event.replay_mode),
                "plan_id": f"{event.uav_id}_event_{idx + 1}",
                "attack_count": len(event.attack_types),
            }
        )
    return rows


def attack_schedule_frame_for_config(config: Mapping[str, Any]) -> pd.DataFrame:
    if str(config.get("dataset_replay_mode", "") or "").strip().lower() == "pure_ids_csv":
        return pd.DataFrame(
            columns=[
                "uav_id",
                "attack_type",
                "start_s",
                "end_s",
                "duration_s",
                "intensity",
                "source_dataset",
                "source_dataset_display",
                "replay_mode",
                "plan_id",
                "attack_count",
            ]
        )
    scenario_config = build_runtime_scenario_config(config)
    rows = scenario_schedule_rows(scenario_config)
    if not rows:
        return pd.DataFrame(
            columns=[
                "uav_id",
                "attack_type",
                "start_s",
                "end_s",
                "duration_s",
                "intensity",
                "source_dataset",
                "source_dataset_display",
                "replay_mode",
                "plan_id",
                "attack_count",
            ]
        )
    return sort_frame_by_uav_order(pd.DataFrame(rows), prefix_cols=["start_s"])


def configure_demo_detector_for_artifact_thresholds(detector: OnlineDetector) -> OnlineDetector:
    detector.use_artifact_calibrator_decision = True
    detector.score_threshold_mode = "artifact_ood_calibrator"
    if isinstance(getattr(detector, "threshold_config", None), dict):
        detector.threshold_config["threshold_mode"] = detector.score_threshold_mode
    return detector


def build_detector(config: dict[str, Any]) -> OnlineDetector | None:
    if not bool(config["enable_online_detection"]):
        return None

    simulation_group_col = "uav_id"
    simulation_buffer = StreamingWindowBuffer(
        mode="count",
        window_size=int(config["window_size"]),
        stride=int(config["stride"]),
        timestamp_col="timestamp",
        record_id_col="record_id",
        group_col=simulation_group_col,
    )
    artifact_path = str(config["artifact_path"]).strip()
    if artifact_path:
        resolved = resolve_input_path(artifact_path)
        if not resolved.exists():
            raise FileNotFoundError(f"Artifact not found: {resolved}")
        validate_artifact_path(resolved)
        try:
            detector = OnlineDetector.from_artifact_path(
                str(resolved),
                top_records=int(config["top_records"]),
                group_col=simulation_group_col,
            )
        except (pickle.UnpicklingError, OSError, EOFError, RuntimeError, ValueError, TypeError) as exc:
            raise ValueError(
                f"Failed to load model artifact from {resolved}. "
                "Please select a valid training output artifact.pt file. "
                f"Original error: {exc}"
            ) from exc
        detector.buffer = simulation_buffer
        detector.group_col = simulation_group_col
        detector.pre.group_col = simulation_group_col
        detector.window_config = {
            **detector.window_config,
            "mode": simulation_buffer.mode,
            "size": int(simulation_buffer.window_size),
            "stride": int(simulation_buffer.stride),
            "time_seconds": float(simulation_buffer.time_seconds),
            "adaptive_min_size": int(simulation_buffer.adaptive_min_size),
            "adaptive_max_size": int(simulation_buffer.adaptive_max_size),
        }
    else:
        artifact = cached_bootstrap_artifact(
            uav_count=int(config["uav_count"]),
            selected_uav_ids=tuple(selected_uav_ids_for_config(config)),
            route_length_m=float(config["route_length_m"]),
            hover_duration_s=float(config["hover_duration_s"]),
            cruise_altitude_m=float(config["cruise_altitude_m"]),
            cruise_speed_mps=float(config["cruise_speed_mps"]),
            dt_s=float(config["dt_s"]),
            seed=int(config["seed"]),
            scenario_profile_path=str(config["scenario_profile_path"]).strip(),
            bootstrap_duration_s=float(config["bootstrap_duration_s"]),
            window_size=int(config["window_size"]),
            stride=int(config["stride"]),
            bank_k=int(config["bank_k"]),
            bootstrap_q_ood=float(config["bootstrap_q_ood"]),
            bootstrap_threshold_margin=float(config["bootstrap_threshold_margin"]),
        )
        detector = OnlineDetector(
            artifact,
            top_records=int(config["top_records"]),
            group_col=simulation_group_col,
            buffer=simulation_buffer,
        )
    return configure_demo_detector_for_artifact_thresholds(detector)


def latest_simulator_engine_classes() -> tuple[type[Any], type[Any]]:
    module = importlib.import_module("ucs_oodid.simulator.engine")
    importlib.invalidate_caches()
    module = importlib.reload(module)
    return module.SimulationConfig, module.SimulationEngine


def record_to_row(record: Any) -> dict[str, Any]:
    row = dict(record.to_dict())
    slot_suffix = ""
    if row.get("attack_replay_slot") not in {None, ""}:
        slot_suffix = f":slot{int(row['attack_replay_slot'])}"
    if row.get("source_record_id") not in {None, ""}:
        row["record_id"] = f"{row['uav_id']}:{float(row['timestamp']):.3f}:{row['source_record_id']}{slot_suffix}"
    else:
        row["record_id"] = f"{row['uav_id']}:{float(row['timestamp']):.3f}{slot_suffix}"
    attack_active = bool(row.get("attack_active", False))
    attack_type = str(row.get("attack_type", "benign") or "benign")
    if not attack_active or attack_type == "benign":
        row["label"] = "benign"
    else:
        row["label"] = ATTACK_LABEL_ALIASES.get(attack_type, attack_type)
    row["throughput_bytes"] = int(row.get("bytes_up", 0)) + int(row.get("bytes_down", 0))
    return row


def _normalize_known_attack_pred_labels(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return str(value).strip()
    if isinstance(value, (list, tuple, set)):
        labels = [str(item).strip() for item in value if str(item).strip()]
        return ", ".join(labels)
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return str(value).strip()


def _normalize_alert_reason_value(
    value: Any,
    *,
    known_attack_alert: bool,
    ood_alert: bool,
) -> str:
    text = str(value or "").strip().lower()
    if text in {"known_attack", "ood", "known_attack+ood", "none"}:
        return text
    if known_attack_alert and ood_alert:
        return "known_attack+ood"
    if known_attack_alert:
        return "known_attack"
    if ood_alert:
        return "ood"
    return "none"


def ensure_detection_alert_channel_fields(detections: pd.DataFrame) -> pd.DataFrame:
    work = detections.copy()
    if "known_attack_alert" not in work.columns:
        work["known_attack_alert"] = False
    work["known_attack_alert"] = work["known_attack_alert"].fillna(False).astype(bool)

    if "ood_alert" not in work.columns:
        if "is_ood" in work.columns:
            work["ood_alert"] = work["is_ood"].fillna(False).astype(bool)
        else:
            work["ood_alert"] = False
    elif "is_ood" in work.columns:
        work["ood_alert"] = work["ood_alert"].fillna(work["is_ood"].fillna(False)).astype(bool)
    else:
        work["ood_alert"] = work["ood_alert"].fillna(False).astype(bool)

    if "known_attack_pred_labels" not in work.columns:
        work["known_attack_pred_labels"] = ""
    work["known_attack_pred_labels"] = work["known_attack_pred_labels"].map(_normalize_known_attack_pred_labels)

    if "alert_reason" not in work.columns:
        work["alert_reason"] = ""
    work["alert_reason"] = [
        _normalize_alert_reason_value(
            value,
            known_attack_alert=bool(known_attack_alert),
            ood_alert=bool(ood_alert),
        )
        for value, known_attack_alert, ood_alert in zip(
            work["alert_reason"].tolist(),
            work["known_attack_alert"].tolist(),
            work["ood_alert"].tolist(),
        )
    ]
    return work


def frame_to_json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    return list(json.loads(frame.to_json(orient="records", date_format="iso")))


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.DataFrame):
        return frame_to_json_records(value)
    if isinstance(value, pd.Series):
        return _json_safe_value(value.to_dict())
    if isinstance(value, (str, int, float, bool)) or value is None:
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
        return value
    if hasattr(value, "item"):
        try:
            return _json_safe_value(value.item())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, bytearray)):
        try:
            return _json_safe_value(value.tolist())
        except TypeError:
            pass
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def _record_count_for_export(value: Any) -> int:
    if isinstance(value, pd.DataFrame):
        return int(len(value.index))
    if value is None:
        return 0
    try:
        return int(len(value))
    except TypeError:
        return 0


def build_streamlit_export_paths(
    *,
    output_dir: str | Path | None = None,
    timestamp: datetime | None = None,
) -> dict[str, str]:
    base_dir = Path(output_dir) if output_dir is not None else STREAMLIT_EXPORT_DIR
    moment = timestamp or datetime.now()
    stamp = moment.strftime("%Y%m%d_%H%M%S_%f")
    snapshot_path = base_dir / f"streamlit_simulation_result_{stamp}.json"
    latest_path = base_dir / STREAMLIT_EXPORT_LATEST_FILENAME
    return {
        "dashboard_json": str(snapshot_path),
        "dashboard_json_latest": str(latest_path),
    }


def build_streamlit_export_payload(
    payload: Mapping[str, Any],
    *,
    export_paths: Mapping[str, str] | None = None,
    timestamp: datetime | None = None,
) -> dict[str, Any]:
    moment = timestamp or datetime.now()
    raw_config = payload.get("config", {})
    config = raw_config if isinstance(raw_config, Mapping) else {}
    export_summary_only = bool(config.get("export_summary_only", False))
    records = payload.get("records", pd.DataFrame())
    detections = payload.get("detections", pd.DataFrame())
    if isinstance(detections, pd.DataFrame):
        detections = ensure_detection_alert_channel_fields(detections)
    export_payload = {
        "export_source": "streamlit_dashboard",
        "saved_at": moment.isoformat(timespec="seconds"),
        "config": _json_safe_value(config),
        "summary": _json_safe_value(payload.get("summary", {})),
        "detections": _json_safe_value(detections),
        "responses": _json_safe_value(payload.get("responses", pd.DataFrame())),
        "attack_schedule": _json_safe_value(payload.get("attack_schedule", pd.DataFrame())),
    }
    if export_summary_only:
        summary = payload.get("summary", {})
        summary_record_count = (
            int(summary.get("record_count", 0) or 0) if isinstance(summary, Mapping) else _record_count_for_export(records)
        )
        export_payload["records"] = []
        export_payload["records_omitted"] = True
        export_payload["records_omitted_count"] = summary_record_count
        export_payload["time_points"] = []
    else:
        export_payload["time_points"] = _json_safe_value(payload.get("time_points", []))
        export_payload["records"] = _json_safe_value(records)
        export_payload["output_files"] = _json_safe_value(dict(export_paths or {}))
    return export_payload


def save_streamlit_payload_json(
    payload: Mapping[str, Any],
    *,
    output_dir: str | Path | None = None,
) -> dict[str, str]:
    moment = datetime.now()
    export_paths = build_streamlit_export_paths(output_dir=output_dir, timestamp=moment)
    export_payload = build_streamlit_export_payload(payload, export_paths=export_paths, timestamp=moment)
    save_json(export_payload, export_paths["dashboard_json"])
    save_json(export_payload, export_paths["dashboard_json_latest"])
    return dict(export_paths)


def format_output_path(path_text: str) -> str:
    try:
        path = Path(path_text).resolve()
        return str(path.relative_to(ROOT.resolve())).replace("\\", "/")
    except (OSError, ValueError):
        return str(path_text)


def records_frame(records: list[Any]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=RECORD_COLUMNS)
    frame = pd.DataFrame(record_to_row(record) for record in records)
    frame = attach_uav_metadata(frame)
    return sort_frame_by_uav_order(frame, prefix_cols=["timestamp"])


def detections_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=DETECTION_COLUMNS)

    normalized: list[dict[str, Any]] = []
    for row in rows:
        truth = row.get("ground_truth") if isinstance(row.get("ground_truth"), dict) else {}
        truth_attack_active_raw = row.get("gt_attack_active")
        if truth_attack_active_raw is None:
            truth_attack_active_raw = truth.get("attack_active")
        truth_is_ood_raw = row.get("ground_truth_is_ood")
        if truth_is_ood_raw is None:
            truth_is_ood_raw = truth.get("is_ood")
        record_ids = list(row.get("record_ids", []) or [])
        group_id = row.get("group_id")
        if group_id in {None, ""} and record_ids:
            group_id = str(record_ids[0]).split(":", 1)[0]
        alert_level = str(row.get("alert_level", "normal") or "normal").strip().lower()
        normalized_ood_score = row.get("normalized_ood_score")
        if normalized_ood_score is None:
            normalized_ood_score = row.get("ood_score")
        raw_ood_score = row.get("raw_ood_score")
        if raw_ood_score is None:
            raw_ood_score = row.get("ood_score")
        threshold = row.get("threshold")
        if threshold is None:
            threshold = row.get("ood_threshold")
        is_ood = bool(row.get("is_ood", False))
        known_attack_alert = bool(row.get("known_attack_alert", False))
        known_attack_pred_labels = _normalize_known_attack_pred_labels(row.get("known_attack_pred_labels", []))
        ood_alert_raw = row.get("ood_alert")
        ood_alert = bool(ood_alert_raw) if ood_alert_raw is not None else is_ood
        alert_reason = _normalize_alert_reason_value(
            row.get("alert_reason"),
            known_attack_alert=known_attack_alert,
            ood_alert=ood_alert,
        )
        has_alert_raw = row.get("has_alert")
        has_alert = (
            bool(has_alert_raw)
            if has_alert_raw is not None
            else bool(known_attack_alert or ood_alert or alert_level in {"warning", "critical"})
        )
        false_alert_raw = row.get("false_alert")
        false_alert = bool(false_alert_raw) if false_alert_raw is not None else bool(has_alert and not bool(truth_attack_active_raw))
        if truth_is_ood_raw is not None:
            window_partition = "ood" if bool(truth_is_ood_raw) else ("attack" if bool(truth_attack_active_raw) else "benign")
        else:
            window_partition = "attack" if bool(truth_attack_active_raw) else "benign"
        normalized.append(
            {
                "simulation_time_s": float(row.get("simulation_time_s", 0.0)),
                "window_id": int(row.get("window_id", 0)),
                "group_id": None if group_id in {None, ""} else str(group_id),
                "raw_ood_score": float(raw_ood_score or 0.0),
                "normalized_ood_score": float(normalized_ood_score or 0.0),
                "ood_score": float(normalized_ood_score or 0.0),
                "threshold": float(threshold or 0.0),
                "ood_threshold": float(threshold or 0.0),
                "threshold_source": str(row.get("threshold_source", row.get("ood_threshold_source", "global")) or "global"),
                "is_ood": is_ood,
                "ood_alert": ood_alert,
                "known_attack_alert": known_attack_alert,
                "known_attack_pred_labels": known_attack_pred_labels,
                "alert_reason": alert_reason,
                "alert_level": alert_level,
                "has_alert": has_alert,
                "false_alert": false_alert,
                "attack_active": bool(truth_attack_active_raw) if truth_attack_active_raw is not None else False,
                "ground_truth_is_ood": None if truth_is_ood_raw is None else bool(truth_is_ood_raw),
                "window_partition": window_partition,
                "attack_types": ", ".join(str(item) for item in truth.get("attack_types", []) if str(item).strip()),
                "dominant_ood_source": row.get("dominant_ood_source"),
                "response_action": row.get("response_action"),
                "response_time": row.get("response_time"),
                "response_reason": row.get("response_reason"),
                "response_triggered": bool(row.get("response_triggered", False)),
                "record_count": int(len(record_ids)),
            }
        )
    frame = pd.DataFrame(normalized)
    frame = ensure_detection_alert_channel_fields(frame)
    frame = attach_uav_metadata(frame, uav_col="group_id")
    return sort_frame_by_uav_order(frame, uav_col="group_id", prefix_cols=["simulation_time_s"])


def responses_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=RESPONSE_COLUMNS)
    frame = pd.DataFrame(rows)
    return frame.sort_values("response_time", kind="stable").reset_index(drop=True)


def latest_records_by_uav(records: pd.DataFrame, current_time_s: float | None = None) -> pd.DataFrame:
    if records.empty:
        return records.copy()
    work = records
    if current_time_s is not None:
        work = work[work["timestamp"] <= float(current_time_s)]
    if work.empty:
        return pd.DataFrame(columns=records.columns)
    latest = sort_frame_by_uav_order(work, prefix_cols=["timestamp"]).groupby("uav_id", as_index=False).tail(1)
    return sort_frame_by_uav_order(latest)


def per_uav_energy(records: pd.DataFrame) -> pd.DataFrame:
    if records.empty:
        return pd.DataFrame(columns=["uav_id", "flight_energy_wh", "communication_energy_wh", "detection_energy_wh", "total_energy_wh"])
    grouped = records.groupby("uav_id", as_index=False)[
        ["flight_energy_wh", "communication_energy_wh", "detection_energy_wh", "total_energy_wh"]
    ].sum()
    return sort_frame_by_uav_order(grouped)


def phase_durations(records: pd.DataFrame, dt_s: float) -> pd.DataFrame:
    if records.empty:
        return pd.DataFrame(columns=["uav_id", "mission_phase", "duration_s"])
    work = records.groupby(["uav_id", "mission_phase"], as_index=False).size()
    work["duration_s"] = work["size"] * float(dt_s)
    work["mission_phase"] = pd.Categorical(work["mission_phase"], categories=PHASE_ORDER, ordered=True)
    work["_uav_sort_order"] = work["uav_id"].map(lambda value: UAV_ID_ORDER_INDEX.get(str(value).strip(), len(UAV_ID_ORDER_INDEX)))
    work = work.sort_values(["_uav_sort_order", "mission_phase"], kind="stable").drop(columns="_uav_sort_order")
    return work.reset_index(drop=True)


def per_uav_analysis(records: pd.DataFrame, detections: pd.DataFrame) -> pd.DataFrame:
    if records.empty:
        return pd.DataFrame()
    work_records = attach_uav_metadata(records)
    latest = latest_records_by_uav(work_records).set_index("uav_id")
    agg = work_records.groupby("uav_id").agg(
        dataset_name=("dataset_name", "last"),
        dataset_display=("dataset_display", "last"),
        source_type=("source_type", "last"),
        simulation_role=("simulation_role", "last"),
        mission_context=("mission_context", "last"),
        avg_rssi=("rssi", "mean"),
        avg_latency_ms=("latency_ms", "mean"),
        avg_loss_rate=("loss_rate", "mean"),
        avg_cpu_load=("cpu_load", "mean"),
        peak_board_temperature_c=("board_temperature_c", "max"),
        record_count=("uav_id", "size"),
        total_bytes_up=("bytes_up", "sum"),
        total_bytes_down=("bytes_down", "sum"),
        total_energy_wh=("total_energy_wh", "sum"),
        attack_record_count=("attack_active", "sum"),
    )
    agg["current_phase"] = latest["mission_phase"]
    agg["battery_soc"] = latest["battery_soc"]
    agg["altitude"] = latest["altitude"]
    agg["speed"] = latest["speed"]
    if not detections.empty:
        work_detections = attach_uav_metadata(ensure_detection_alert_channel_fields(detections), uav_col="group_id")
        work_detections["dual_alert_window"] = work_detections["known_attack_alert"] & work_detections["ood_alert"]
        det = work_detections.groupby("group_id").agg(
            window_count=("group_id", "size"),
            ood_windows=("is_ood", "sum"),
            ood_alert_windows=("ood_alert", "sum"),
            known_attack_alert_windows=("known_attack_alert", "sum"),
            alert_windows=("has_alert", "sum"),
            dual_alert_windows=("dual_alert_window", "sum"),
            mean_ood_score=("ood_score", "mean"),
            peak_ood_score=("ood_score", "max"),
            false_alert_windows=("has_alert", lambda values: 0),
        )
        false_alerts = work_detections.assign(false_alert_window=work_detections["has_alert"] & ~work_detections["attack_active"])
        det["false_alert_windows"] = false_alerts.groupby("group_id")["false_alert_window"].sum()
        agg = agg.join(det, how="left")
    agg = agg.fillna(
        {
            "window_count": 0,
            "ood_windows": 0,
            "ood_alert_windows": 0,
            "known_attack_alert_windows": 0,
            "alert_windows": 0,
            "dual_alert_windows": 0,
            "mean_ood_score": 0.0,
            "peak_ood_score": 0.0,
            "false_alert_windows": 0,
        }
    )
    return sort_frame_by_uav_order(agg.reset_index().rename(columns={"index": "uav_id"}))


def detections_with_phase(records: pd.DataFrame, detections: pd.DataFrame) -> pd.DataFrame:
    if records.empty or detections.empty:
        return pd.DataFrame()
    work = detections.copy()
    work = work[work["group_id"].notna()].copy()
    if work.empty:
        return pd.DataFrame()
    work = work.rename(columns={"group_id": "uav_id"})
    # merge_asof requires the "on" key to be globally sorted even when a "by" key is present.
    det_keys = sort_frame_by_uav_order(work[["uav_id", "simulation_time_s"]], prefix_cols=["simulation_time_s"])
    record_keys = (
        records[["uav_id", "timestamp", "mission_phase", "mission_context"]]
        .rename(columns={"timestamp": "simulation_time_s"})
    )
    record_keys = sort_frame_by_uav_order(record_keys, prefix_cols=["simulation_time_s"])
    merged = pd.merge_asof(
        det_keys,
        record_keys,
        on="simulation_time_s",
        by="uav_id",
        direction="backward",
    )
    merged = merged.merge(work, on=["uav_id", "simulation_time_s"], how="right")
    merged["mission_phase"] = pd.Categorical(merged["mission_phase"], categories=PHASE_ORDER, ordered=True)
    return sort_frame_by_uav_order(merged, prefix_cols=["simulation_time_s"])


def phase_false_positive_rates(records: pd.DataFrame, detections: pd.DataFrame) -> pd.DataFrame:
    work = detections_with_phase(records, detections)
    if work.empty:
        return pd.DataFrame(columns=["mission_phase", "benign_windows", "false_positive_windows", "false_positive_rate"])
    work["false_positive_window"] = work["has_alert"] & ~work["attack_active"]
    work["benign_window"] = ~work["attack_active"]
    grouped = work.groupby("mission_phase", as_index=False, observed=False).agg(
        benign_windows=("benign_window", "sum"),
        false_positive_windows=("false_positive_window", "sum"),
    )
    grouped["false_positive_rate"] = grouped.apply(
        lambda row: 0.0 if float(row["benign_windows"]) <= 0.0 else float(row["false_positive_windows"]) / float(row["benign_windows"]),
        axis=1,
    )
    return grouped.sort_values("mission_phase", kind="stable").reset_index(drop=True)


def per_uav_ood_comparison(detections: pd.DataFrame) -> pd.DataFrame:
    if detections.empty:
        return pd.DataFrame(columns=["uav_id", "ood_windows", "alert_windows", "false_alert_windows", "peak_ood_score"])
    work = detections[detections["group_id"].notna()].copy()
    if work.empty:
        return pd.DataFrame(columns=["uav_id", "ood_windows", "alert_windows", "false_alert_windows", "peak_ood_score"])
    work["false_alert_window"] = work["has_alert"] & ~work["attack_active"]
    comparison = work.groupby("group_id", as_index=False).agg(
        ood_windows=("is_ood", "sum"),
        alert_windows=("has_alert", "sum"),
        false_alert_windows=("false_alert_window", "sum"),
        peak_ood_score=("ood_score", "max"),
    )
    comparison = comparison.rename(columns={"group_id": "uav_id"})
    return sort_frame_by_uav_order(comparison)


def _safe_binary_auroc(labels: pd.Series, scores: pd.Series) -> float | None:
    work = pd.DataFrame({"label": labels, "score": scores}).dropna()
    if work.empty:
        return None
    if work["label"].nunique() < 2:
        return None
    try:
        return float(roc_auc_score(work["label"].astype(int), work["score"].astype(float)))
    except ValueError:
        return None


def observed_ood_label_series(detections: pd.DataFrame) -> tuple[pd.Series, str]:
    if "ground_truth_is_ood" in detections.columns and detections["ground_truth_is_ood"].notna().any():
        return detections["ground_truth_is_ood"], "ground_truth.is_ood"
    if "gt_attack_active" in detections.columns and detections["gt_attack_active"].notna().any():
        return detections["gt_attack_active"], "gt_attack_active_fallback"
    return detections["attack_active"], "attack_active_fallback"


def per_uav_diagnostic_rows(
    records: pd.DataFrame,
    detections: pd.DataFrame,
    detector: OnlineDetector | None = None,
) -> list[dict[str, Any]]:
    if detections.empty:
        return []
    work = attach_uav_metadata(detections, uav_col="group_id")
    work = work[work["group_id"].notna()].copy()
    if work.empty:
        return []

    attack_col = "gt_attack_active" if "gt_attack_active" in work.columns and work["gt_attack_active"].notna().any() else "attack_active"
    if attack_col in work.columns:
        work["diagnostic_gt_attack_active"] = work[attack_col].fillna(False).astype(bool)
    else:
        work["diagnostic_gt_attack_active"] = False
    if "ground_truth_is_ood" in work.columns and work["ground_truth_is_ood"].notna().any():
        work["diagnostic_truth_is_ood"] = work["ground_truth_is_ood"].fillna(False).astype(bool)
        label_source = "ground_truth.is_ood"
    else:
        work["diagnostic_truth_is_ood"] = False
        label_source = "missing_ground_truth.is_ood"
    work["diagnostic_window_partition"] = work.apply(
        lambda row: "ood"
        if bool(row.get("diagnostic_truth_is_ood", False))
        else ("attack" if bool(row.get("diagnostic_gt_attack_active", False)) else "benign"),
        axis=1,
    )
    score_col = (
        "ood_score"
        if "ood_score" in work.columns
        else ("normalized_ood_score" if "normalized_ood_score" in work.columns else "raw_ood_score")
    )
    raw_score_col = "raw_ood_score" if "raw_ood_score" in work.columns else score_col
    normalized_score_col = "normalized_ood_score" if "normalized_ood_score" in work.columns else score_col

    def _diag_map(name: str) -> dict[str, Any]:
        return {str(group_id): value for group_id, value in (live_diagnostics.get(name, {}) or {}).items()}

    def _subset_text_value(subset: pd.DataFrame, column: str, default: str = "") -> str:
        if column in subset.columns and subset[column].notna().any():
            return str(subset[column].dropna().mode().iloc[0] if not subset[column].dropna().mode().empty else subset[column].dropna().iloc[0])
        return default

    def _mean_or_none(series: pd.Series) -> float | None:
        clean = pd.to_numeric(series, errors="coerce").dropna()
        if clean.empty:
            return None
        return float(clean.mean())

    def _quantile_or_none(series: pd.Series, q: float) -> float | None:
        clean = pd.to_numeric(series, errors="coerce").dropna()
        if clean.empty:
            return None
        return float(clean.quantile(q))

    def _peak_or_none(series: pd.Series) -> float | None:
        clean = pd.to_numeric(series, errors="coerce").dropna()
        if clean.empty:
            return None
        return float(clean.max())

    threshold_map: dict[str, float] = {}
    raw_threshold_map: dict[str, float] = {}
    normalized_threshold_map: dict[str, float] = {}
    threshold_source_map: dict[str, str] = {}
    global_threshold = None
    live_diagnostics: dict[str, Any] = {}
    if detector is not None and hasattr(detector, "simulation_diagnostics"):
        live_diagnostics = dict(detector.simulation_diagnostics() or {})
        threshold_map = {
            str(group_id): float(value)
            for group_id, value in (live_diagnostics.get("per_uav_threshold", {}) or {}).items()
            if value is not None
        }
        raw_threshold_map = {
            str(group_id): float(value)
            for group_id, value in (live_diagnostics.get("per_uav_raw_threshold", {}) or {}).items()
            if value is not None
        }
        normalized_threshold_map = {
            str(group_id): float(value)
            for group_id, value in (live_diagnostics.get("per_uav_normalized_threshold", {}) or {}).items()
            if value is not None
        }
        threshold_source_map = {
            str(group_id): str(value)
            for group_id, value in (live_diagnostics.get("per_uav_threshold_source", {}) or {}).items()
            if value not in {None, ""}
        }
    if detector is not None and hasattr(detector, "threshold_config"):
        threshold_config = getattr(detector, "threshold_config", {}) or {}
        raw_thresholds = threshold_config.get("group_ood_thresholds", {})
        for group_id, value in raw_thresholds.items():
            threshold_map.setdefault(str(group_id), float(value))
        if threshold_config.get("global_ood_threshold") is not None:
            global_threshold = float(threshold_config["global_ood_threshold"])

    raw_q50_map = _diag_map("per_uav_raw_score_q50")
    raw_q90_map = _diag_map("per_uav_raw_score_q90")
    raw_q95_map = _diag_map("per_uav_raw_score_q95")
    raw_q99_map = _diag_map("per_uav_raw_score_q99")
    normalized_q50_map = _diag_map("per_uav_normalized_score_q50")
    normalized_q90_map = _diag_map("per_uav_normalized_score_q90")
    normalized_q95_map = _diag_map("per_uav_normalized_score_q95")
    normalized_q99_map = _diag_map("per_uav_normalized_score_q99")
    benign_q50_map = _diag_map("per_uav_benign_score_q50")
    benign_q90_map = _diag_map("per_uav_benign_score_q90")
    benign_q95_map = _diag_map("per_uav_benign_score_q95")
    benign_q99_map = _diag_map("per_uav_benign_score_q99")
    default_score_mode = str(live_diagnostics.get("score_mode", "") or "").strip()
    default_threshold_mode = str(live_diagnostics.get("threshold_mode", "") or "").strip()
    default_score_direction = str(live_diagnostics.get("score_direction", "") or "").strip()

    rows: list[dict[str, Any]] = []
    for uav_id in ordered_uav_ids(work["group_id"].tolist()):
        subset = work[work["group_id"] == uav_id].copy()
        if subset.empty:
            continue
        window_count = int(len(subset))
        pred_alert_window_count = int(subset["has_alert"].fillna(False).sum())
        pred_ood_window_count = int(subset["is_ood"].fillna(False).sum()) if "is_ood" in subset.columns else pred_alert_window_count
        benign_mask = subset["diagnostic_window_partition"] == "benign"
        attack_mask = subset["diagnostic_window_partition"] == "attack"
        ood_mask = subset["diagnostic_window_partition"] == "ood"
        benign_window_count = int(benign_mask.sum())
        gt_attack_window_count = int(attack_mask.sum())
        gt_ood_window_count = int(ood_mask.sum())
        false_alert_window_count = (
            int(subset["false_alert"].fillna(False).sum())
            if "false_alert" in subset.columns
            else int((subset["has_alert"].fillna(False) & benign_mask).sum())
        )
        threshold_value = threshold_map.get(str(uav_id))
        if threshold_value is None:
            threshold_col = "threshold" if "threshold" in subset.columns else "ood_threshold"
            if threshold_col in subset.columns and subset[threshold_col].notna().any():
                threshold_value = float(subset[threshold_col].mean())
            elif global_threshold is not None:
                threshold_value = float(global_threshold)
        raw_threshold = raw_threshold_map.get(str(uav_id))
        if raw_threshold is None and "raw_threshold" in subset.columns and subset["raw_threshold"].notna().any():
            raw_threshold = float(subset["raw_threshold"].mean())
        normalized_threshold = normalized_threshold_map.get(str(uav_id))
        if normalized_threshold is None and "normalized_threshold" in subset.columns and subset["normalized_threshold"].notna().any():
            normalized_threshold = float(subset["normalized_threshold"].mean())
        threshold_source = threshold_source_map.get(str(uav_id))
        if not threshold_source:
            source_col = "threshold_source" if "threshold_source" in subset.columns else "ood_threshold_source"
            if source_col in subset.columns and subset[source_col].notna().any():
                threshold_source = str(subset[source_col].dropna().iloc[0])
            else:
                threshold_source = "global"
        score_mode = default_score_mode or _subset_text_value(subset, "score_mode", "normalized")
        threshold_mode = default_threshold_mode or _subset_text_value(subset, "threshold_mode")
        score_direction = default_score_direction or _subset_text_value(subset, "score_direction", "higher_is_more_anomalous")
        benign_score_mean = _mean_or_none(subset.loc[benign_mask, score_col]) if score_col in subset.columns else None
        attack_score_mean = _mean_or_none(subset.loc[attack_mask, score_col]) if score_col in subset.columns else None
        ood_score_mean = _mean_or_none(subset.loc[ood_mask, score_col]) if score_col in subset.columns else None
        score_separation = None if benign_score_mean is None or attack_score_mean is None else float(attack_score_mean - benign_score_mean)
        direction_warning = bool(
            benign_score_mean is not None
            and attack_score_mean is not None
            and float(attack_score_mean) < float(benign_score_mean)
        )
        raw_ood_score_mean = _mean_or_none(subset[raw_score_col]) if raw_score_col in subset.columns else None
        normalized_ood_score_mean = _mean_or_none(subset[normalized_score_col]) if normalized_score_col in subset.columns else None
        rows.append(
            {
                "uav_id": str(uav_id),
                "dataset_name": str(subset["dataset_name"].dropna().iloc[0]) if "dataset_name" in subset.columns and subset["dataset_name"].notna().any() else registry_dataset_name_for_uav(str(uav_id)),
                "dataset_display": str(subset["dataset_display"].dropna().iloc[0]) if "dataset_display" in subset.columns and subset["dataset_display"].notna().any() else dataset_display_for_uav(str(uav_id), annotate_role=False),
                "source_type": str(subset["source_type"].dropna().iloc[0]) if "source_type" in subset.columns and subset["source_type"].notna().any() else uav_source_type(str(uav_id)),
                "simulation_role": str(subset["simulation_role"].dropna().iloc[0]) if "simulation_role" in subset.columns and subset["simulation_role"].notna().any() else uav_simulation_role(str(uav_id)),
                "score_partition_label_source": label_source,
                "window_count": window_count,
                "benign_window_count": benign_window_count,
                "gt_attack_window_count": gt_attack_window_count,
                "gt_ood_window_count": gt_ood_window_count,
                "pred_alert_window_count": pred_alert_window_count,
                "pred_ood_window_count": pred_ood_window_count,
                "false_alert_window_count": false_alert_window_count,
                "threshold": threshold_value,
                "raw_threshold": raw_threshold,
                "normalized_threshold": normalized_threshold,
                "threshold_source": threshold_source,
                "score_mode": score_mode,
                "threshold_mode": threshold_mode,
                "score_direction": score_direction,
                "id_score_mean": benign_score_mean,
                "benign_score_mean": benign_score_mean,
                "attack_score_mean": attack_score_mean,
                "ood_score_mean": ood_score_mean,
                "score_separation": score_separation,
                "direction_warning": direction_warning,
                "reversed_score_separation": None if score_separation is None else float(-score_separation),
                "raw_ood_score_mean": raw_ood_score_mean,
                "normalized_ood_score_mean": normalized_ood_score_mean,
                "peak_ood_score": _peak_or_none(subset[score_col]) if score_col in subset.columns else None,
                "raw_score_q50": raw_q50_map.get(str(uav_id), _quantile_or_none(subset[raw_score_col], 0.50) if raw_score_col in subset.columns else None),
                "raw_score_q90": raw_q90_map.get(str(uav_id), _quantile_or_none(subset[raw_score_col], 0.90) if raw_score_col in subset.columns else None),
                "raw_score_q95": raw_q95_map.get(str(uav_id), _quantile_or_none(subset[raw_score_col], 0.95) if raw_score_col in subset.columns else None),
                "raw_score_q99": raw_q99_map.get(str(uav_id), _quantile_or_none(subset[raw_score_col], 0.99) if raw_score_col in subset.columns else None),
                "normalized_score_q50": normalized_q50_map.get(str(uav_id), _quantile_or_none(subset[normalized_score_col], 0.50) if normalized_score_col in subset.columns else None),
                "normalized_score_q90": normalized_q90_map.get(str(uav_id), _quantile_or_none(subset[normalized_score_col], 0.90) if normalized_score_col in subset.columns else None),
                "normalized_score_q95": normalized_q95_map.get(str(uav_id), _quantile_or_none(subset[normalized_score_col], 0.95) if normalized_score_col in subset.columns else None),
                "normalized_score_q99": normalized_q99_map.get(str(uav_id), _quantile_or_none(subset[normalized_score_col], 0.99) if normalized_score_col in subset.columns else None),
                "benign_score_q50": benign_q50_map.get(str(uav_id), _quantile_or_none(subset.loc[benign_mask, raw_score_col], 0.50) if raw_score_col in subset.columns else None),
                "benign_score_q90": benign_q90_map.get(str(uav_id), _quantile_or_none(subset.loc[benign_mask, raw_score_col], 0.90) if raw_score_col in subset.columns else None),
                "benign_score_q95": benign_q95_map.get(str(uav_id), _quantile_or_none(subset.loc[benign_mask, raw_score_col], 0.95) if raw_score_col in subset.columns else None),
                "benign_score_q99": benign_q99_map.get(str(uav_id), _quantile_or_none(subset.loc[benign_mask, raw_score_col], 0.99) if raw_score_col in subset.columns else None),
                "alert_rate": 0.0 if window_count <= 0 else float(pred_alert_window_count) / float(window_count),
                "false_alert_rate_on_benign_windows": 0.0
                if benign_window_count <= 0
                else float(false_alert_window_count) / float(benign_window_count),
            }
        )
    return rows


def score_direction_diagnostics(
    detections: pd.DataFrame,
    detector: OnlineDetector | None = None,
) -> dict[str, Any]:
    configured_report: list[dict[str, Any]] = []
    configured_directions: dict[str, float] = {}
    configured_label_source = "none"
    if detector is not None and hasattr(detector, "ood_cal"):
        ood_cal = detector.ood_cal
        configured_report = [dict(item) for item in getattr(ood_cal, "direction_report", [])]
        configured_directions = {str(name): float(value) for name, value in getattr(ood_cal, "directions", {}).items()}
        configured_label_source = str(getattr(ood_cal, "direction_label_source", None) or "none")

    if detections.empty:
        return {
            "configured_label_source": configured_label_source,
            "configured_directions": configured_directions,
            "configured_report": configured_report,
            "observed_label_source": "unavailable",
            "fused_ood_score_raw_auroc": None,
            "fused_ood_score_effective_auroc": None,
            "fused_ood_score_flip_suspected": False,
            "per_uav_fused_ood_score_raw_auroc": {},
        }

    observed_labels, observed_label_source = observed_ood_label_series(detections)
    fused_raw_auroc = _safe_binary_auroc(observed_labels, detections["ood_score"])
    fused_flip_suspected = fused_raw_auroc is not None and fused_raw_auroc < 0.5
    fused_effective_auroc = None if fused_raw_auroc is None else (1.0 - fused_raw_auroc if fused_flip_suspected else fused_raw_auroc)

    per_uav_raw_auroc: dict[str, float | None] = {}
    if "group_id" in detections.columns:
        for uav_id in ordered_uav_ids(detections["group_id"].dropna().tolist()):
            subset = detections[detections["group_id"] == uav_id]
            if subset.empty:
                continue
            subset_labels, _ = observed_ood_label_series(subset)
            per_uav_raw_auroc[str(uav_id)] = _safe_binary_auroc(subset_labels, subset["ood_score"])

    return {
        "configured_label_source": configured_label_source,
        "configured_directions": configured_directions,
        "configured_report": configured_report,
        "observed_label_source": observed_label_source,
        "fused_ood_score_raw_auroc": fused_raw_auroc,
        "fused_ood_score_effective_auroc": fused_effective_auroc,
        "fused_ood_score_flip_suspected": fused_flip_suspected,
        "per_uav_fused_ood_score_raw_auroc": per_uav_raw_auroc,
    }


def build_detection_diagnostics(
    records: pd.DataFrame,
    detections: pd.DataFrame,
    detector: OnlineDetector | None = None,
) -> dict[str, Any]:
    detections = ensure_detection_alert_channel_fields(detections)
    rows = per_uav_diagnostic_rows(records, detections, detector=detector)
    detector_diagnostics = dict(detector.simulation_diagnostics() or {}) if detector is not None and hasattr(detector, "simulation_diagnostics") else {}
    ids_energy_wh = float(records["detection_energy_wh"].sum()) if not records.empty and "detection_energy_wh" in records else 0.0
    total_energy_wh = float(records["total_energy_wh"].sum()) if not records.empty and "total_energy_wh" in records else 0.0
    ids_energy_ratio = 0.0 if total_energy_wh <= 0.0 else ids_energy_wh / total_energy_wh
    truth_attack_active = (
        detections.get("gt_attack_active", detections.get("attack_active", pd.Series(False, index=detections.index)))
        if not detections.empty
        else pd.Series(False, dtype=bool)
    )
    truth_is_ood = (
        detections.get("ground_truth_is_ood", pd.Series(False, index=detections.index))
        if not detections.empty
        else pd.Series(False, dtype=bool)
    )
    if not detections.empty:
        truth_attack_active = truth_attack_active.fillna(False).astype(bool)
        truth_is_ood = truth_is_ood.fillna(False).astype(bool)
    gt_attack_window_count = int((truth_attack_active & ~truth_is_ood).sum()) if not detections.empty else 0
    gt_ood_window_count = int(truth_is_ood.sum()) if not detections.empty else 0
    benign_window_count = max(int(len(detections)) - gt_attack_window_count - gt_ood_window_count, 0)

    def _rate_by_source_type(source_type: str | None) -> float | None:
        if not rows:
            return None
        selected = [
            row for row in rows
            if (source_type is None and row["source_type"] != "external_non_uav")
            or (source_type is not None and row["source_type"] == source_type)
        ]
        benign_window_count = sum(int(row["benign_window_count"]) for row in selected)
        false_alert_windows = sum(int(row["false_alert_window_count"]) for row in selected)
        if benign_window_count <= 0:
            return None
        return float(false_alert_windows) / float(benign_window_count)

    return {
        "gt_attack_window_count": gt_attack_window_count,
        "gt_ood_window_count": gt_ood_window_count,
        "benign_window_count": benign_window_count,
        "pred_alert_window_count": int(detections["has_alert"].sum()) if not detections.empty else 0,
        "pred_known_attack_alert_window_count": int(detections["known_attack_alert"].sum()) if not detections.empty else 0,
        "pred_ood_window_count": int(detections["ood_alert"].sum()) if not detections.empty else 0,
        "pred_dual_alert_window_count": int((detections["known_attack_alert"] & detections["ood_alert"]).sum())
        if not detections.empty
        else 0,
        "false_alert_window_count": int(detections["false_alert"].sum())
        if not detections.empty and "false_alert" in detections
        else (
            int(
                (
                    detections["has_alert"]
                    & ~detections.get(
                        "gt_attack_active",
                        detections.get("attack_active", pd.Series(False, index=detections.index)),
                    )
                ).sum()
            )
            if not detections.empty
            else 0
        ),
        "per_uav_diagnostic_rows": rows,
        "model_input_columns": list(detector_diagnostics.get("model_input_columns", [])),
        "per_uav_threshold": {row["uav_id"]: row["threshold"] for row in rows},
        "per_uav_threshold_source": {row["uav_id"]: row["threshold_source"] for row in rows},
        "per_uav_raw_threshold": {row["uav_id"]: row["raw_threshold"] for row in rows},
        "per_uav_normalized_threshold": {row["uav_id"]: row["normalized_threshold"] for row in rows},
        "per_uav_benign_window_count": {row["uav_id"]: row["benign_window_count"] for row in rows},
        "per_uav_gt_attack_window_count": {row["uav_id"]: row["gt_attack_window_count"] for row in rows},
        "per_uav_gt_ood_window_count": {row["uav_id"]: row["gt_ood_window_count"] for row in rows},
        "per_uav_pred_alert_window_count": {row["uav_id"]: row["pred_alert_window_count"] for row in rows},
        "per_uav_pred_ood_window_count": {row["uav_id"]: row["pred_ood_window_count"] for row in rows},
        "per_uav_id_score_mean": {row["uav_id"]: row["id_score_mean"] for row in rows},
        "per_uav_benign_score_mean": {row["uav_id"]: row["benign_score_mean"] for row in rows},
        "per_uav_attack_score_mean": {row["uav_id"]: row["attack_score_mean"] for row in rows},
        "per_uav_raw_ood_score_mean": {row["uav_id"]: row["raw_ood_score_mean"] for row in rows},
        "per_uav_normalized_ood_score_mean": {row["uav_id"]: row["normalized_ood_score_mean"] for row in rows},
        "per_uav_ood_score_mean": {row["uav_id"]: row["ood_score_mean"] for row in rows},
        "per_uav_score_separation": {row["uav_id"]: row["score_separation"] for row in rows},
        "per_uav_direction_warning": {row["uav_id"]: row["direction_warning"] for row in rows},
        "per_uav_reversed_score_separation": {row["uav_id"]: row["reversed_score_separation"] for row in rows},
        "per_uav_alert_rate": {row["uav_id"]: row["alert_rate"] for row in rows},
        "per_uav_false_alert_rate_on_benign_windows": {
            row["uav_id"]: row["false_alert_rate_on_benign_windows"] for row in rows
        },
        "per_uav_raw_score_q50": {row["uav_id"]: row["raw_score_q50"] for row in rows},
        "per_uav_raw_score_q90": {row["uav_id"]: row["raw_score_q90"] for row in rows},
        "per_uav_raw_score_q95": {row["uav_id"]: row["raw_score_q95"] for row in rows},
        "per_uav_raw_score_q99": {row["uav_id"]: row["raw_score_q99"] for row in rows},
        "per_uav_normalized_score_q50": {row["uav_id"]: row["normalized_score_q50"] for row in rows},
        "per_uav_normalized_score_q90": {row["uav_id"]: row["normalized_score_q90"] for row in rows},
        "per_uav_normalized_score_q95": {row["uav_id"]: row["normalized_score_q95"] for row in rows},
        "per_uav_normalized_score_q99": {row["uav_id"]: row["normalized_score_q99"] for row in rows},
        "per_uav_benign_score_q50": {row["uav_id"]: row["benign_score_q50"] for row in rows},
        "per_uav_benign_score_q90": {row["uav_id"]: row["benign_score_q90"] for row in rows},
        "per_uav_benign_score_q95": {row["uav_id"]: row["benign_score_q95"] for row in rows},
        "per_uav_benign_score_q99": {row["uav_id"]: row["benign_score_q99"] for row in rows},
        "score_mode": detector_diagnostics.get("score_mode"),
        "threshold_mode": detector_diagnostics.get("threshold_mode"),
        "score_direction": detector_diagnostics.get("score_direction"),
        "normalized_threshold": detector_diagnostics.get("normalized_threshold"),
        "score_direction_diagnostics": score_direction_diagnostics(detections, detector=detector),
        "ids_energy_wh": ids_energy_wh,
        "ids_energy_ratio": ids_energy_ratio,
        "uav_only_false_alert_rate": _rate_by_source_type(None),
        "external_non_uav_false_alert_rate": _rate_by_source_type("external_non_uav"),
    }


def build_summary(
    config: dict[str, Any],
    result: Any,
    records: pd.DataFrame,
    detections: pd.DataFrame,
    responses: pd.DataFrame,
    detector: OnlineDetector | None = None,
) -> dict[str, Any]:
    diagnostics = dict(getattr(result, "diagnostics", {}) or {}) if isinstance(getattr(result, "diagnostics", None), Mapping) else {}
    summary_uav_ids = ordered_uav_ids(diagnostics.get("summary_uav_ids", [])) if diagnostics else []
    summary = {
        "uav_count": int(records["uav_id"].nunique())
        if not records.empty
        else (len(summary_uav_ids) if summary_uav_ids else int(config["uav_count"])),
        "record_count": int(len(records)) if not records.empty else int(diagnostics.get("summary_record_count", 0) or 0),
        "window_count": int(len(detections)),
        "response_count": int(len(responses)),
        "dataset_replay_mode": str(config.get("dataset_replay_mode", "") or ""),
        "pure_ids_csv_path": str(config.get("pure_ids_csv_path", "") or ""),
        "pure_ids_group_col": str(config.get("pure_ids_group_col", "") or ""),
        "pure_ids_replay_records_per_uav_per_step": int(config.get("pure_ids_replay_records_per_uav_per_step", 0) or 0),
        "pure_ids_replay_stop_when_exhausted": bool(config.get("pure_ids_replay_stop_when_exhausted", False)),
        "pure_ids_replay_keep_original_order": bool(config.get("pure_ids_replay_keep_original_order", False)),
        "pure_ids_replay_no_mixing": bool(config.get("pure_ids_replay_no_mixing", False)),
        "pure_ids_replay_split_filter": normalize_pure_ids_replay_split_filter(
            config.get("pure_ids_replay_split_filter", [])
        ),
        "pure_ids_replay_reset_buffer_on_split_change": bool(
            config.get("pure_ids_replay_reset_buffer_on_split_change", False)
        ),
        "artifact_path": str(config.get("artifact_path", "") or ""),
        "window_size": int(config.get("window_size", 0) or 0),
        "stride": int(config.get("stride", 0) or 0),
        "export_summary_only": bool(config.get("export_summary_only", False)),
        "mission_success": bool(result.mission_success),
        "attack_record_count": int(result.attack_record_count)
        if not records.empty
        else int(diagnostics.get("summary_attack_record_count", result.attack_record_count) or 0),
        "alert_count": int(result.alert_count),
        "false_alert_count": int(result.false_alert_count),
        "average_alert_delay": float(result.average_alert_delay),
        "total_energy_wh": float(result.total_energy_wh),
        "peak_ood_score": float(detections["ood_score"].max()) if not detections.empty else 0.0,
        "mean_rssi": float(records["rssi"].mean())
        if not records.empty
        else float(diagnostics.get("summary_mean_rssi", 0.0) or 0.0),
        "mean_latency_ms": float(records["latency_ms"].mean())
        if not records.empty
        else float(diagnostics.get("summary_mean_latency_ms", 0.0) or 0.0),
        "mean_loss_rate": float(records["loss_rate"].mean())
        if not records.empty
        else float(diagnostics.get("summary_mean_loss_rate", 0.0) or 0.0),
        "mean_cpu_load": float(records["cpu_load"].mean())
        if not records.empty and "cpu_load" in records
        else float(diagnostics.get("summary_mean_cpu_load", 0.0) or 0.0),
        "peak_board_temperature_c": float(records["board_temperature_c"].max())
        if not records.empty and "board_temperature_c" in records
        else float(diagnostics.get("summary_peak_board_temperature_c", 0.0) or 0.0),
        "total_bytes_up": int(records["bytes_up"].sum())
        if not records.empty
        else int(diagnostics.get("summary_total_bytes_up", 0) or 0),
        "total_bytes_down": int(records["bytes_down"].sum())
        if not records.empty
        else int(diagnostics.get("summary_total_bytes_down", 0) or 0),
    }
    summary.update(build_detection_diagnostics(records, detections, detector=detector))
    if not records.empty:
        summary["uav_ids"] = ordered_uav_ids(records["uav_id"].unique().tolist())
    elif summary_uav_ids:
        summary["uav_ids"] = summary_uav_ids
    if records.empty:
        ids_energy_wh = float(diagnostics.get("summary_ids_energy_wh", summary.get("ids_energy_wh", 0.0)) or 0.0)
        summary["ids_energy_wh"] = ids_energy_wh
        summary["ids_energy_ratio"] = 0.0 if float(summary["total_energy_wh"]) <= 0.0 else ids_energy_wh / float(summary["total_energy_wh"])
    if diagnostics:
        summary.update(
            {
                str(key): value
                for key, value in diagnostics.items()
                if str(key) == "dataset_replay_mode" or str(key).startswith("pure_ids_")
            }
        )
    return summary


def run_simulation(config: dict[str, Any]) -> dict[str, Any]:
    if not SIMULATION_ENABLED:
        raise RuntimeError(SIMULATION_DISABLED_MESSAGE)
    dataset_replay_mode = str(config.get("dataset_replay_mode", "") or "").strip().lower()
    scenario_config = None if dataset_replay_mode == "pure_ids_csv" else build_runtime_scenario_config(config)
    detector = build_detector(config)
    uavs = build_uavs(config)
    export_summary_only = bool(config.get("export_summary_only", False))
    SimulationConfigCls, SimulationEngineCls = latest_simulator_engine_classes()
    attack_replay_pool = None
    if dataset_replay_mode != "pure_ids_csv":
        attack_replay_pool = cached_attack_replay_pool(int(config["seed"]))
        if attack_replay_pool is not None:
            attack_replay_pool.reset()
    engine = SimulationEngineCls(
        uavs=uavs,
        gcs=GCS(gcs_id="gcs_dashboard"),
        attacker=Attacker(attacker_id="attacker_dashboard", x_m=220.0, y_m=60.0),
        attack_injector=AttackInjector(),
        config=SimulationConfigCls(
            duration_s=float(config["duration_s"]),
            dt_s=float(config["dt_s"]),
            records_per_uav_per_step=int(config["records_per_uav_per_step"]),
            target_records_per_uav=int(config["target_records_per_uav"]),
            seed=int(config["seed"]),
            enable_online_detection=bool(config["enable_online_detection"]),
            response_strategy=str(config["response_strategy"]),
            scenario_profile_path=str(config["scenario_profile_path"]).strip(),
            attack_replay_mode=str(config["attack_replay_mode"]),
            export_summary_only=export_summary_only,
            dataset_replay_mode=str(config.get("dataset_replay_mode", "")).strip(),
            pure_ids_csv_path=str(config.get("pure_ids_csv_path", "")).strip(),
            pure_ids_group_col=str(config.get("pure_ids_group_col", "uav_id")).strip(),
            pure_ids_replay_records_per_uav_per_step=int(
                config.get("pure_ids_replay_records_per_uav_per_step", PURE_IDS_REPLAY_RECORDS_PER_STEP)
            ),
            pure_ids_replay_stop_when_exhausted=bool(config.get("pure_ids_replay_stop_when_exhausted", True)),
            pure_ids_replay_keep_original_order=bool(config.get("pure_ids_replay_keep_original_order", True)),
            pure_ids_replay_no_mixing=bool(config.get("pure_ids_replay_no_mixing", True)),
            pure_ids_replay_split_filter=tuple(
                normalize_pure_ids_replay_split_filter(config.get("pure_ids_replay_split_filter"))
            ),
            pure_ids_replay_reset_buffer_on_split_change=bool(
                config.get("pure_ids_replay_reset_buffer_on_split_change", True)
            ),
        ),
        online_detector=detector,
        attack_replay_pool=attack_replay_pool,
        scenario_config=scenario_config,
    )
    result = engine.run()
    detections = detections_frame(list(result.online_detection_results))
    responses = responses_frame(list(result.response_events))
    attack_schedule = attack_schedule_frame_for_config(config)
    if export_summary_only:
        records = pd.DataFrame(columns=RECORD_COLUMNS)
        time_points: list[float] = []
    else:
        records = records_frame(result.records)
        time_points = sorted(float(value) for value in records["timestamp"].unique()) if not records.empty else []
    return {
        "config": dict(config),
        "records": records,
        "detections": detections,
        "responses": responses,
        "attack_schedule": attack_schedule,
        "summary": build_summary(config, result, records, detections, responses, detector=detector),
        "time_points": time_points,
    }


def payload_uav_ids(payload: Mapping[str, Any]) -> list[str]:
    records = payload.get("records", pd.DataFrame())
    if isinstance(records, pd.DataFrame) and not records.empty and "uav_id" in records.columns:
        return ordered_uav_ids(records["uav_id"].tolist())
    summary = payload.get("summary", {})
    if isinstance(summary, Mapping):
        diagnostic_rows = summary.get("per_uav_diagnostic_rows", [])
        if isinstance(diagnostic_rows, Sequence) and not isinstance(diagnostic_rows, (str, bytes, bytearray)):
            uav_ids = ordered_uav_ids(
                [row.get("uav_id") for row in diagnostic_rows if isinstance(row, Mapping) and row.get("uav_id")]
            )
            if uav_ids:
                return uav_ids
        uav_ids = ordered_uav_ids(summary.get("uav_ids", []))
        if uav_ids:
            return uav_ids
    config = payload.get("config", {})
    if isinstance(config, Mapping):
        uav_ids = selected_uav_ids_for_config(config)
        if uav_ids:
            return uav_ids
    return [UAV_ID_ORDER[0]]


def payload_is_summary_only(payload: Mapping[str, Any]) -> bool:
    config = payload.get("config", {})
    records = payload.get("records", pd.DataFrame())
    return bool(isinstance(config, Mapping) and config.get("export_summary_only", False)) and isinstance(records, pd.DataFrame) and records.empty


def store_payload(payload: dict[str, Any]) -> None:
    export_paths = save_streamlit_payload_json(payload)
    payload = dict(payload)
    payload["export_paths"] = export_paths
    st.session_state.dashboard_payload = payload
    st.session_state.dashboard_error = ""
    merged_config = merge_dashboard_config(payload["config"])
    st.session_state.dashboard_config = merged_config
    queue_mission_widget_sync(merged_config)
    time_points = payload["time_points"]
    uav_ids = payload_uav_ids(payload)
    st.session_state.live_time_index = max(len(time_points) - 1, 0)
    st.session_state.replay_time_index = 0
    st.session_state.live_selected_uav = uav_ids[0]
    st.session_state.replay_selected_uav = uav_ids[0]
    st.session_state.analysis_selected_uav = uav_ids[0]


def run_and_store(config: dict[str, Any]) -> None:
    with st.spinner("Running mission simulation and building dashboard data..."):
        payload = run_simulation(config)
    store_payload(payload)


def ensure_payload_or_offer_demo() -> dict[str, Any] | None:
    if not SIMULATION_ENABLED:
        st.info(SIMULATION_DISABLED_MESSAGE)
        return None
    payload = st.session_state.dashboard_payload
    if payload is not None:
        return payload
    st.info("No simulation data yet. Run the built-in demo scene to populate Live Monitor, Replay, and Analysis.")
    if st.button("Run built-in demo scene", type="primary", key="empty_state_demo"):
        try:
            config = clone_pure_ids_csv_replay_config()
            sync_widget_state(config, overwrite=True)
            run_and_store(config)
            st.success("Demo finished — charts and metrics are shown below on this page.")
        except Exception as exc:  # pragma: no cover - UI guard
            st.session_state.dashboard_error = str(exc)
            st.exception(exc)
        return st.session_state.dashboard_payload
    return None


def format_pct(value: float) -> str:
    return f"{value:.1f}%"


def format_bytes(value: int) -> str:
    return f"{int(value):,} B"


def format_wh(value: float) -> str:
    return f"{value:.2f} Wh"


def empty_figure(title: str, message: str, *, height: int = 320) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, showarrow=False, xref="paper", yref="paper")
    fig.update_layout(template="plotly_white", title=title, height=height)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def add_time_cursor(fig: go.Figure, current_time_s: float | None) -> go.Figure:
    if current_time_s is not None:
        fig.add_vline(x=float(current_time_s), line_dash="dash", line_color="#ff7f0e", opacity=0.8)
    return fig


def selected_records(records: pd.DataFrame, uav_id: str, current_time_s: float | None = None, truncate: bool = False) -> pd.DataFrame:
    work = records[records["uav_id"] == uav_id].copy()
    if current_time_s is not None and truncate:
        work = work[work["timestamp"] <= float(current_time_s)]
    return work.sort_values("timestamp", kind="stable").reset_index(drop=True)


def selected_detections(detections: pd.DataFrame, uav_id: str, current_time_s: float | None = None, truncate: bool = False) -> pd.DataFrame:
    if detections.empty:
        return ensure_detection_alert_channel_fields(detections)
    work = ensure_detection_alert_channel_fields(detections)
    if "group_id" in work.columns and work["group_id"].notna().any():
        filtered = work[work["group_id"] == uav_id]
        if not filtered.empty:
            work = filtered
    if current_time_s is not None and truncate:
        work = work[work["simulation_time_s"] <= float(current_time_s)]
    return work.sort_values("simulation_time_s", kind="stable").reset_index(drop=True)


def selected_responses(responses: pd.DataFrame, uav_id: str, current_time_s: float | None = None, truncate: bool = False) -> pd.DataFrame:
    if responses.empty:
        return responses.copy()
    work = responses[responses["uav_id"] == uav_id].copy()
    if current_time_s is not None and truncate:
        work = work[work["response_time"] <= float(current_time_s)]
    return work.sort_values("response_time", kind="stable").reset_index(drop=True)


def latest_detection_at(detections: pd.DataFrame, uav_id: str, current_time_s: float) -> pd.Series | None:
    work = selected_detections(detections, uav_id, current_time_s=current_time_s, truncate=True)
    if work.empty:
        return None
    return work.iloc[-1]


def fleet_status_frame(records: pd.DataFrame, detections: pd.DataFrame, current_time_s: float) -> pd.DataFrame:
    columns = [
        "uav_id",
        "mission_phase",
        "battery_soc",
        "attack_active",
        "attack_type",
        "ood_score",
        "alert_level",
        "alert_reason",
        "known_attack_alert",
        "ood_alert",
    ]
    snapshot = latest_records_by_uav(records, current_time_s=current_time_s)
    if snapshot.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for row in sort_frame_by_uav_order(snapshot).to_dict("records"):
        uav_id = str(row["uav_id"])
        detection_row = latest_detection_at(detections, uav_id, current_time_s)
        rows.append(
            {
                "uav_id": uav_id,
                "mission_phase": str(row.get("mission_phase", "n/a")),
                "battery_soc": float(row.get("battery_soc", 0.0)),
                "attack_active": bool(row.get("attack_active", False)),
                "attack_type": str(row.get("attack_type", "benign") or "benign"),
                "ood_score": 0.0 if detection_row is None else float(detection_row.get("ood_score", 0.0)),
                "alert_level": "normal" if detection_row is None else str(detection_row.get("alert_level", "normal") or "normal"),
                "alert_reason": "none" if detection_row is None else str(detection_row.get("alert_reason", "none") or "none"),
                "known_attack_alert": False if detection_row is None else bool(detection_row.get("known_attack_alert", False)),
                "ood_alert": False if detection_row is None else bool(detection_row.get("ood_alert", detection_row.get("is_ood", False))),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def fleet_metric_chart(
    frame: pd.DataFrame,
    x_col: str,
    y_col: str,
    series_col: str,
    title: str,
    *,
    labels: dict[str, str] | None = None,
    category_orders: dict[str, list[str]] | None = None,
    current_time_s: float | None = None,
    height: int = 320,
) -> go.Figure:
    if frame.empty:
        return empty_figure(title, "No data available.")
    work = frame.copy()
    work = work[work[series_col].notna()].copy()
    if work.empty:
        return empty_figure(title, "No data available.")
    fig = px.line(
        work.sort_values([x_col, series_col], kind="stable"),
        x=x_col,
        y=y_col,
        color=series_col,
        markers=True,
        title=title,
        labels=labels or {},
        category_orders=category_orders or {},
    )
    add_time_cursor(fig, current_time_s)
    fig.update_layout(template="plotly_white", height=height, legend=dict(orientation="h"))
    return fig


def fleet_attack_timeline_chart(schedule: pd.DataFrame, current_time_s: float | None = None) -> go.Figure:
    if schedule.empty:
        return empty_figure("Attack Timeline by Data Source", "No attack schedule available.")
    work = attach_uav_metadata(schedule)
    source_order = [source_label_for_uav(uav_id) for uav_id in ordered_uav_ids(work["uav_id"].tolist())]
    fig = px.timeline(
        work,
        x_start="start_s",
        x_end="end_s",
        y="source_label",
        color="attack_type",
        title="Attack Timeline by Data Source",
        labels={"source_label": "Data source", "attack_type": "Attack type"},
        category_orders={"source_label": source_order},
        hover_data=["uav_id", "dataset_name", "source_type", "intensity", "replay_mode"],
    )
    add_time_cursor(fig, current_time_s)
    fig.update_layout(template="plotly_white", height=max(260, 100 + 60 * len(work["source_label"].unique())))
    fig.update_xaxes(title_text="Time (s)")
    return fig


def fleet_alert_event_chart(detections: pd.DataFrame, current_time_s: float | None = None) -> go.Figure:
    if detections.empty:
        return empty_figure("Alert Trigger Points by Data Source", "Online detection is disabled or no alerts were emitted.")
    alerts = ensure_detection_alert_channel_fields(detections[detections["has_alert"]].copy())
    if alerts.empty:
        return empty_figure("Alert Trigger Points by Data Source", "No alert trigger points were recorded.")
    alerts = attach_uav_metadata(alerts, uav_col="group_id")
    alerts["source_label"] = alerts["group_id"].map(lambda value: source_label_for_uav(str(value).strip()) if pd.notna(value) else "fleet")
    source_order = [source_label_for_uav(uav_id) for uav_id in ordered_uav_ids(alerts["group_id"].dropna().tolist())]
    if "fleet" in alerts["source_label"].tolist():
        source_order.append("fleet")
    fig = px.scatter(
        alerts,
        x="simulation_time_s",
        y="source_label",
        color="alert_reason",
        symbol="alert_level",
        title="Alert Trigger Points by Data Source",
        labels={"simulation_time_s": "Time (s)", "source_label": "Data source", "alert_reason": "Alert channel"},
        category_orders={"source_label": source_order, "alert_reason": ALERT_REASON_ORDER},
        hover_data=[
            "group_id",
            "dataset_name",
            "source_type",
            "ood_score",
            "attack_types",
            "known_attack_pred_labels",
            "known_attack_alert",
            "ood_alert",
        ],
    )
    add_time_cursor(fig, current_time_s)
    fig.update_layout(template="plotly_white", height=320, legend=dict(orientation="h"))
    return fig


def split_attack_tokens(value: Any) -> list[str]:
    tokens: list[str] = []
    for raw_token in str(value or "").replace("+", ",").split(","):
        text = str(raw_token).strip().lower()
        if text and text != "benign":
            tokens.append(text)
    return normalize_attack_selection(tokens)


def attack_scene_frame(records: pd.DataFrame) -> pd.DataFrame:
    columns = ["timestamp", "active_uav_count", "scene_attack_types", "attack_scene_mode"]
    if records.empty:
        return pd.DataFrame(columns=columns)
    active = records[records["attack_active"]].copy()
    if active.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for timestamp, group in active.groupby("timestamp", sort=True):
        scene_attack_types: list[str] = []
        for attack_type in group["attack_type"].tolist():
            scene_attack_types.extend(split_attack_tokens(attack_type))
        unique_types = normalize_attack_selection(scene_attack_types)
        rows.append(
            {
                "timestamp": float(timestamp),
                "active_uav_count": int(group["uav_id"].nunique()),
                "scene_attack_types": attack_type_summary(unique_types),
                "attack_scene_mode": "mixed_attack"
                if int(group["uav_id"].nunique()) > 1 or len(unique_types) > 1
                else "single_attack",
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values("timestamp", kind="stable").reset_index(drop=True)


def detections_with_attack_scene(records: pd.DataFrame, detections: pd.DataFrame) -> pd.DataFrame:
    if detections.empty:
        return pd.DataFrame(columns=[*DETECTION_COLUMNS, "active_uav_count", "scene_attack_types", "attack_scene_mode"])
    work = detections.copy()
    scene = attack_scene_frame(records).rename(columns={"timestamp": "simulation_time_s"})
    if scene.empty:
        work["active_uav_count"] = 0
        work["scene_attack_types"] = ""
        work["attack_scene_mode"] = work["attack_active"].map(lambda active: "single_attack" if bool(active) else "benign")
        return work
    work = work.merge(scene, on="simulation_time_s", how="left")
    work["active_uav_count"] = work["active_uav_count"].fillna(0).astype(int)
    work["scene_attack_types"] = work["scene_attack_types"].fillna("")
    work["attack_scene_mode"] = work.apply(
        lambda row: "benign"
        if not bool(row.get("attack_active", False))
        else str(row.get("attack_scene_mode", "") or "single_attack"),
        axis=1,
    )
    return work


def empty_detection_mode_row(attack_mode: str) -> dict[str, Any]:
    return {
        "attack_mode": attack_mode,
        "attack_windows": 0,
        "detected_windows": 0,
        "missed_windows": 0,
        "detection_rate": 0.0,
        "mean_ood_score": 0.0,
        "peak_ood_score": 0.0,
        "critical_alerts": 0,
    }


def detection_mode_summary(records: pd.DataFrame, detections: pd.DataFrame) -> pd.DataFrame:
    if detections.empty:
        return pd.DataFrame([empty_detection_mode_row(mode) for mode in ATTACK_MODE_OPTIONS])
    work = detections_with_attack_scene(records, detections)
    rows: list[dict[str, Any]] = []
    for attack_mode in ATTACK_MODE_OPTIONS:
        subset = work[work["attack_scene_mode"] == attack_mode].copy()
        if subset.empty:
            rows.append(empty_detection_mode_row(attack_mode))
            continue
        attack_windows = int(len(subset))
        detected_windows = int(subset["has_alert"].sum())
        rows.append(
            {
                "attack_mode": attack_mode,
                "attack_windows": attack_windows,
                "detected_windows": detected_windows,
                "missed_windows": max(attack_windows - detected_windows, 0),
                "detection_rate": 0.0 if attack_windows <= 0 else float(detected_windows) / float(attack_windows),
                "mean_ood_score": float(subset["ood_score"].mean()),
                "peak_ood_score": float(subset["ood_score"].max()),
                "critical_alerts": int((subset["alert_level"] == "critical").sum()),
            }
        )
    return pd.DataFrame(rows)


def detection_mode_row(summary_frame: pd.DataFrame, attack_mode: str) -> dict[str, Any]:
    if summary_frame.empty:
        return empty_detection_mode_row(attack_mode)
    matched = summary_frame[summary_frame["attack_mode"] == attack_mode]
    if matched.empty:
        return empty_detection_mode_row(attack_mode)
    return dict(matched.iloc[0].to_dict())


def per_uav_dataset_detection_comparison(records: pd.DataFrame, detections: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "uav_id",
        "dataset",
        "attack_windows",
        "detected_windows",
        "false_alert_windows",
        "detection_rate",
        "mean_ood_score",
        "peak_ood_score",
    ]
    if records.empty:
        return pd.DataFrame(columns=columns)
    scene_detections = detections_with_attack_scene(records, detections) if not detections.empty else pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for uav_id in ordered_uav_ids(records["uav_id"].unique().tolist()):
        det = (
            scene_detections[scene_detections["group_id"] == uav_id].copy()
            if not scene_detections.empty and scene_detections["group_id"].notna().any()
            else pd.DataFrame()
        )
        attack_windows = int(det["attack_active"].sum()) if not det.empty else 0
        detected_windows = int((det["has_alert"] & det["attack_active"]).sum()) if not det.empty else 0
        false_alert_windows = int((det["has_alert"] & ~det["attack_active"]).sum()) if not det.empty else 0
        rows.append(
            {
                "uav_id": str(uav_id),
                "dataset": dataset_display_for_uav(str(uav_id)),
                "attack_windows": attack_windows,
                "detected_windows": detected_windows,
                "false_alert_windows": false_alert_windows,
                "detection_rate": 0.0 if attack_windows <= 0 else float(detected_windows) / float(attack_windows),
                "mean_ood_score": 0.0 if det.empty else float(det["ood_score"].mean()),
                "peak_ood_score": 0.0 if det.empty else float(det["ood_score"].max()),
            }
        )
    return sort_frame_by_uav_order(pd.DataFrame(rows, columns=columns))


def aggregate_detection_metrics_by_group(
    detections: pd.DataFrame,
    group_cols: Sequence[str],
) -> pd.DataFrame:
    columns = [
        *group_cols,
        "window_count",
        "attack_windows",
        "detected_attack_windows",
        "ood_windows",
        "alert_windows",
        "false_alert_windows",
        "detection_rate",
        "mean_ood_score",
        "peak_ood_score",
    ]
    if detections.empty:
        return pd.DataFrame(columns=columns)
    work = detections.copy()
    work["detected_attack_window"] = work["has_alert"] & work["attack_active"]
    work["false_alert_window"] = work["has_alert"] & ~work["attack_active"]
    grouped = work.groupby(list(group_cols), as_index=False, dropna=False).agg(
        window_count=("has_alert", "size"),
        attack_windows=("attack_active", "sum"),
        detected_attack_windows=("detected_attack_window", "sum"),
        ood_windows=("is_ood", "sum"),
        alert_windows=("has_alert", "sum"),
        false_alert_windows=("false_alert_window", "sum"),
        mean_ood_score=("ood_score", "mean"),
        peak_ood_score=("ood_score", "max"),
    )
    grouped["detection_rate"] = grouped.apply(
        lambda row: 0.0
        if float(row["attack_windows"]) <= 0.0
        else float(row["detected_attack_windows"]) / float(row["attack_windows"]),
        axis=1,
    )
    return grouped


def per_dataset_metrics_frame(records: pd.DataFrame, detections: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "uav_ids",
        "dataset_name",
        "dataset_display",
        "source_type",
        "simulation_role",
        "source_note",
        "uav_count",
        "record_count",
        "attack_record_count",
        "window_count",
        "attack_windows",
        "detected_attack_windows",
        "ood_windows",
        "alert_windows",
        "false_alert_windows",
        "detection_rate",
        "mean_ood_score",
        "peak_ood_score",
        "total_energy_wh",
        "mean_battery_soc",
    ]
    if records.empty:
        return pd.DataFrame(columns=columns)
    work_records = attach_uav_metadata(records)
    work_detections = attach_uav_metadata(detections, uav_col="group_id") if not detections.empty else pd.DataFrame()
    group_cols = ["dataset_name", "dataset_display", "source_type", "simulation_role", "source_note"]
    record_metrics = work_records.groupby(group_cols, as_index=False, dropna=False).agg(
        uav_ids=("uav_id", lambda values: ", ".join(ordered_uav_ids(values.tolist()))),
        sort_uav_id=("uav_id", lambda values: (ordered_uav_ids(values.tolist()) or [""])[0]),
        uav_count=("uav_id", "nunique"),
        record_count=("uav_id", "size"),
        attack_record_count=("attack_active", "sum"),
        total_energy_wh=("total_energy_wh", "sum"),
        mean_battery_soc=("battery_soc", "mean"),
    )
    detection_metrics = aggregate_detection_metrics_by_group(
        work_detections[work_detections["group_id"].notna()].copy() if not work_detections.empty else work_detections,
        group_cols,
    )
    merged = record_metrics.merge(detection_metrics, on=group_cols, how="left")
    merged = merged.fillna(
        {
            "window_count": 0,
            "attack_windows": 0,
            "detected_attack_windows": 0,
            "ood_windows": 0,
            "alert_windows": 0,
            "false_alert_windows": 0,
            "detection_rate": 0.0,
            "mean_ood_score": 0.0,
            "peak_ood_score": 0.0,
        }
    )
    merged = sort_frame_by_uav_order(merged, uav_col="sort_uav_id")
    return merged.drop(columns="sort_uav_id")[columns]


def per_source_type_metrics_frame(records: pd.DataFrame, detections: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "source_type",
        "simulation_role",
        "uav_ids",
        "dataset_names",
        "uav_count",
        "dataset_count",
        "record_count",
        "attack_record_count",
        "window_count",
        "attack_windows",
        "detected_attack_windows",
        "ood_windows",
        "alert_windows",
        "false_alert_windows",
        "detection_rate",
        "mean_ood_score",
        "peak_ood_score",
        "total_energy_wh",
        "mean_battery_soc",
    ]
    if records.empty:
        return pd.DataFrame(columns=columns)
    work_records = attach_uav_metadata(records)
    work_detections = attach_uav_metadata(detections, uav_col="group_id") if not detections.empty else pd.DataFrame()
    group_cols = ["source_type", "simulation_role"]
    record_metrics = work_records.groupby(group_cols, as_index=False, dropna=False).agg(
        uav_ids=("uav_id", lambda values: ", ".join(ordered_uav_ids(values.tolist()))),
        sort_uav_id=("uav_id", lambda values: (ordered_uav_ids(values.tolist()) or [""])[0]),
        dataset_names=("dataset_display", lambda values: ", ".join(dict.fromkeys(str(value) for value in values if str(value).strip()))),
        uav_count=("uav_id", "nunique"),
        dataset_count=("dataset_name", "nunique"),
        record_count=("uav_id", "size"),
        attack_record_count=("attack_active", "sum"),
        total_energy_wh=("total_energy_wh", "sum"),
        mean_battery_soc=("battery_soc", "mean"),
    )
    detection_metrics = aggregate_detection_metrics_by_group(
        work_detections[work_detections["group_id"].notna()].copy() if not work_detections.empty else work_detections,
        group_cols,
    )
    merged = record_metrics.merge(detection_metrics, on=group_cols, how="left")
    merged = merged.fillna(
        {
            "window_count": 0,
            "attack_windows": 0,
            "detected_attack_windows": 0,
            "ood_windows": 0,
            "alert_windows": 0,
            "false_alert_windows": 0,
            "detection_rate": 0.0,
            "mean_ood_score": 0.0,
            "peak_ood_score": 0.0,
        }
    )
    merged = sort_frame_by_uav_order(merged, uav_col="sort_uav_id")
    return merged.drop(columns="sort_uav_id")[columns]


def external_ood_metrics_frame(records: pd.DataFrame, detections: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "uav_ids",
        "dataset_name",
        "dataset_display",
        "source_type",
        "simulation_role",
        "source_note",
        "uav_count",
        "record_count",
        "attack_record_count",
        "window_count",
        "attack_windows",
        "detected_attack_windows",
        "ood_windows",
        "alert_windows",
        "false_alert_windows",
        "detection_rate",
        "mean_ood_score",
        "peak_ood_score",
        "total_energy_wh",
        "mean_battery_soc",
    ]
    work = per_dataset_metrics_frame(records, detections)
    if work.empty:
        return pd.DataFrame(columns=columns)
    external = work[
        (work["simulation_role"] == "external_ood") | (work["source_type"] == "external_non_uav")
    ].copy()
    if external.empty:
        return pd.DataFrame(columns=columns)
    return external[columns].reset_index(drop=True)


def uav_dataset_detection_chart(records: pd.DataFrame, detections: pd.DataFrame) -> go.Figure:
    work = per_uav_dataset_detection_comparison(records, detections)
    if work.empty:
        return empty_figure("UAV / Dataset Detection Comparison", "No detection windows are available.")
    uav_order = ordered_uav_ids(work["uav_id"].tolist())
    fig = px.bar(
        work,
        x="uav_id",
        y="detection_rate",
        color="dataset",
        text=work["detection_rate"].map(lambda value: f"{value:.0%}"),
        title="UAV / Dataset Detection Comparison",
        labels={"uav_id": "UAV", "detection_rate": "Detection rate", "dataset": "Dataset"},
        category_orders={"uav_id": uav_order},
        hover_data=["attack_windows", "detected_windows", "false_alert_windows", "mean_ood_score", "peak_ood_score"],
    )
    fig.update_traces(textposition="outside")
    fig.add_trace(
        go.Scatter(
            x=work["uav_id"],
            y=work["peak_ood_score"],
            mode="lines+markers",
            name="peak_ood_score",
            yaxis="y2",
            marker=dict(color="#d62728"),
            line=dict(color="#d62728"),
        )
    )
    fig.update_layout(
        template="plotly_white",
        height=360,
        legend=dict(orientation="h"),
        yaxis=dict(tickformat=".0%", range=[0.0, max(float(work["detection_rate"].max()) * 1.2, 1.0)]),
        yaxis2=dict(title="Peak OOD score", overlaying="y", side="right"),
    )
    return fig


def energy_share_frame(records: pd.DataFrame) -> pd.DataFrame:
    columns = ["component", "energy_wh", "share"]
    if records.empty:
        return pd.DataFrame(columns=columns)
    totals = {
        "Flight": float(records["flight_energy_wh"].sum()),
        "Communication": float(records["communication_energy_wh"].sum()),
        "Detection": float(records["detection_energy_wh"].sum()),
    }
    total_energy = sum(totals.values())
    rows = [
        {
            "component": component,
            "energy_wh": energy_wh,
            "share": 0.0 if total_energy <= 0.0 else energy_wh / total_energy,
        }
        for component, energy_wh in totals.items()
    ]
    return pd.DataFrame(rows, columns=columns)


def energy_share_chart(records: pd.DataFrame) -> go.Figure:
    work = energy_share_frame(records)
    if work.empty:
        return empty_figure("Energy Share", "No energy data available.")
    fig = px.pie(
        work,
        names="component",
        values="energy_wh",
        hole=0.55,
        title="Energy Share",
    )
    fig.update_traces(textinfo="percent+label")
    fig.update_layout(template="plotly_white", height=360, legend=dict(orientation="h"))
    return fig


def line_chart(
    frame: pd.DataFrame,
    x_col: str,
    y_cols: list[str],
    title: str,
    *,
    labels: dict[str, str] | None = None,
    current_time_s: float | None = None,
    height: int = 320,
) -> go.Figure:
    if frame.empty:
        return empty_figure(title, "No data available.")
    labels = labels or {}
    fig = go.Figure()
    for column in y_cols:
        fig.add_trace(
            go.Scatter(
                x=frame[x_col],
                y=frame[column],
                mode="lines+markers",
                name=labels.get(column, column),
            )
        )
    add_time_cursor(fig, current_time_s)
    fig.update_layout(template="plotly_white", title=title, height=height, legend=dict(orientation="h"))
    return fig


def attack_timeline_chart(records: pd.DataFrame, uav_id: str, current_time_s: float | None = None, truncate: bool = False) -> go.Figure:
    work = selected_records(records, uav_id, current_time_s=current_time_s, truncate=truncate)
    if work.empty:
        return empty_figure("Attack Timeline", "No attack data available.")
    plot_frame = work[["timestamp", "attack_active", "attack_type"]].copy()
    plot_frame["attack_value"] = plot_frame["attack_active"].astype(int)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=plot_frame["timestamp"],
            y=plot_frame["attack_value"],
            mode="lines",
            line_shape="hv",
            fill="tozeroy",
            name="attack_active",
        )
    )
    markers = plot_frame[plot_frame["attack_active"]]
    if not markers.empty:
        fig.add_trace(
            go.Scatter(
                x=markers["timestamp"],
                y=markers["attack_value"],
                mode="markers",
                marker=dict(color="#d62728", size=8),
                text=markers["attack_type"],
                name="attack_type",
            )
        )
    add_time_cursor(fig, current_time_s)
    fig.update_layout(template="plotly_white", title="Attack Timeline", height=300)
    fig.update_yaxes(tickvals=[0, 1], ticktext=["benign", "attack"])
    return fig


def ood_timeline_chart(detections: pd.DataFrame, uav_id: str, current_time_s: float | None = None, truncate: bool = False) -> go.Figure:
    work = ensure_detection_alert_channel_fields(selected_detections(detections, uav_id, current_time_s=current_time_s, truncate=truncate))
    if work.empty:
        return empty_figure("OOD Score Timeline", "Online detection is disabled or no windows were emitted.")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=work["simulation_time_s"],
            y=work["ood_score"],
            mode="lines+markers",
            name="OOD score",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=work["simulation_time_s"],
            y=work["ood_threshold"],
            mode="lines",
            line=dict(dash="dash"),
            name="threshold",
        )
    )
    ood_only_alerts = work[work["ood_alert"] & ~work["known_attack_alert"]]
    known_only_alerts = work[work["known_attack_alert"] & ~work["ood_alert"]]
    dual_alerts = work[work["known_attack_alert"] & work["ood_alert"]]
    if not ood_only_alerts.empty:
        fig.add_trace(
            go.Scatter(
                x=ood_only_alerts["simulation_time_s"],
                y=ood_only_alerts["ood_score"],
                mode="markers",
                marker=dict(color="#d62728", size=10),
                name="OOD alert",
            )
        )
    if not known_only_alerts.empty:
        fig.add_trace(
            go.Scatter(
                x=known_only_alerts["simulation_time_s"],
                y=known_only_alerts["ood_score"],
                mode="markers",
                marker=dict(color="#ff7f0e", size=10, symbol="diamond"),
                name="Known-attack alert",
            )
        )
    if not dual_alerts.empty:
        fig.add_trace(
            go.Scatter(
                x=dual_alerts["simulation_time_s"],
                y=dual_alerts["ood_score"],
                mode="markers",
                marker=dict(color="#9467bd", size=11, symbol="star"),
                name="Known+OOD alert",
            )
        )
    add_time_cursor(fig, current_time_s)
    fig.update_layout(template="plotly_white", title="OOD Score Timeline", height=320, legend=dict(orientation="h"))
    return fig


def alert_timeline_chart(
    detections: pd.DataFrame,
    responses: pd.DataFrame,
    uav_id: str,
    current_time_s: float | None = None,
    truncate: bool = False,
) -> go.Figure:
    work = selected_detections(detections, uav_id, current_time_s=current_time_s, truncate=truncate)
    response_work = selected_responses(responses, uav_id, current_time_s=current_time_s, truncate=truncate)
    if work.empty and response_work.empty:
        return empty_figure("Alert Timeline", "No alerts or responses recorded.")
    fig = go.Figure()
    if not work.empty:
        work = work.copy()
        work["alert_rank"] = work["alert_level"].map(ALERT_LEVEL_RANK).fillna(0)
        fig.add_trace(
            go.Scatter(
                x=work["simulation_time_s"],
                y=work["alert_rank"],
                mode="lines+markers",
                line_shape="hv",
                name="alert_level",
            )
        )
    if not response_work.empty:
        fig.add_trace(
            go.Scatter(
                x=response_work["response_time"],
                y=[3.2] * len(response_work),
                mode="markers",
                marker=dict(symbol="star", size=12, color="#ff7f0e"),
                text=response_work["response_action"],
                name="response",
            )
        )
    add_time_cursor(fig, current_time_s)
    fig.update_layout(template="plotly_white", title="Alert Timeline", height=300, legend=dict(orientation="h"))
    fig.update_yaxes(tickvals=[0, 1, 2, 3], ticktext=["normal", "watch", "warning", "critical"])
    return fig


def energy_breakdown_chart(records: pd.DataFrame) -> go.Figure:
    work = per_uav_energy(records)
    if work.empty:
        return empty_figure("Energy Consumption Breakdown", "No energy data available.")
    uav_order = ordered_uav_ids(work["uav_id"].tolist())
    melted = work.melt(
        id_vars="uav_id",
        value_vars=["flight_energy_wh", "communication_energy_wh", "detection_energy_wh"],
        var_name="component",
        value_name="energy_wh",
    )
    fig = px.bar(
        melted,
        x="uav_id",
        y="energy_wh",
        color="component",
        barmode="stack",
        title="Energy Consumption Breakdown",
        labels={"uav_id": "UAV", "energy_wh": "Energy (Wh)", "component": "Component"},
        category_orders={"uav_id": uav_order},
    )
    fig.update_layout(template="plotly_white", height=360)
    return fig


def phase_duration_chart(records: pd.DataFrame, dt_s: float) -> go.Figure:
    work = phase_durations(records, dt_s)
    if work.empty:
        return empty_figure("Mission Phase Duration", "No mission phase data available.")
    uav_order = ordered_uav_ids(work["uav_id"].tolist())
    fig = px.bar(
        work,
        x="uav_id",
        y="duration_s",
        color="mission_phase",
        category_orders={"mission_phase": PHASE_ORDER},
        title="Mission Phase Duration",
        labels={"uav_id": "UAV", "duration_s": "Duration (s)", "mission_phase": "Phase"},
    )
    fig.update_xaxes(categoryorder="array", categoryarray=uav_order)
    fig.update_layout(template="plotly_white", height=360)
    return fig


def throughput_summary_chart(records: pd.DataFrame) -> go.Figure:
    if records.empty:
        return empty_figure("Fleet Throughput Summary", "No throughput data available.")
    work = (
        records.groupby("uav_id", as_index=False)[["bytes_up", "bytes_down"]]
        .sum()
    )
    work = sort_frame_by_uav_order(work).melt(id_vars="uav_id", var_name="direction", value_name="bytes")
    uav_order = ordered_uav_ids(work["uav_id"].tolist())
    fig = px.bar(
        work,
        x="uav_id",
        y="bytes",
        color="direction",
        barmode="group",
        title="Fleet Throughput Summary",
        labels={"uav_id": "UAV", "bytes": "Bytes", "direction": "Direction"},
        category_orders={"uav_id": uav_order},
    )
    fig.update_layout(template="plotly_white", height=360)
    return fig


def alert_summary_chart(records: pd.DataFrame, detections: pd.DataFrame) -> go.Figure:
    if records.empty:
        return empty_figure("Attack vs Alert Summary", "No summary data available.")
    attack_steps = records.groupby("uav_id", as_index=False)["attack_active"].sum().rename(columns={"attack_active": "attack_steps"})
    if detections.empty:
        attack_steps["alert_windows"] = 0
    else:
        alert_steps = detections.groupby("group_id", as_index=False)["has_alert"].sum().rename(columns={"group_id": "uav_id", "has_alert": "alert_windows"})
        attack_steps = attack_steps.merge(alert_steps, on="uav_id", how="left").fillna({"alert_windows": 0})
    attack_steps = sort_frame_by_uav_order(attack_steps)
    melted = attack_steps.melt(id_vars="uav_id", var_name="series", value_name="count")
    uav_order = ordered_uav_ids(melted["uav_id"].tolist())
    fig = px.bar(
        melted,
        x="uav_id",
        y="count",
        color="series",
        barmode="group",
        title="Attack vs Alert Summary",
        labels={"uav_id": "UAV", "count": "Count", "series": "Series"},
        category_orders={"uav_id": uav_order},
    )
    fig.update_layout(template="plotly_white", height=360)
    return fig


def phase_false_positive_chart(records: pd.DataFrame, detections: pd.DataFrame) -> go.Figure:
    work = phase_false_positive_rates(records, detections)
    if work.empty:
        return empty_figure("False Positive Rate by Flight Phase", "No benign detection windows are available.")
    fig = px.bar(
        work,
        x="mission_phase",
        y="false_positive_rate",
        text="false_positive_windows",
        category_orders={"mission_phase": PHASE_ORDER},
        title="False Positive Rate by Flight Phase",
        labels={
            "mission_phase": "Flight phase",
            "false_positive_rate": "False positive rate",
            "false_positive_windows": "False alerts",
        },
    )
    fig.update_traces(texttemplate="%{text}", textposition="outside")
    fig.update_layout(template="plotly_white", height=360)
    fig.update_yaxes(tickformat=".1%")
    return fig


def uav_ood_comparison_chart(detections: pd.DataFrame) -> go.Figure:
    work = per_uav_ood_comparison(detections)
    if work.empty:
        return empty_figure("Per-UAV OOD Detection Comparison", "No OOD windows are available.")
    uav_order = ordered_uav_ids(work["uav_id"].tolist())
    melted = work.melt(
        id_vars="uav_id",
        value_vars=["ood_windows", "alert_windows", "false_alert_windows"],
        var_name="metric",
        value_name="count",
    )
    fig = px.bar(
        melted,
        x="uav_id",
        y="count",
        color="metric",
        barmode="group",
        title="Per-UAV OOD Detection Comparison",
        labels={"uav_id": "UAV", "count": "Windows", "metric": "Metric"},
        category_orders={"uav_id": uav_order},
    )
    peak_trace = go.Scatter(
        x=work["uav_id"],
        y=work["peak_ood_score"],
        mode="lines+markers",
        name="peak_ood_score",
        yaxis="y2",
        marker=dict(color="#d62728"),
        line=dict(color="#d62728"),
    )
    fig.add_trace(peak_trace)
    fig.update_layout(
        template="plotly_white",
        height=360,
        yaxis2=dict(title="Peak OOD score", overlaying="y", side="right"),
    )
    return fig


def attack_schedule_preview(config: dict[str, Any]) -> go.Figure:
    schedule = attack_schedule_frame_for_config(config)
    if schedule.empty:
        return empty_figure("Attack Schedule Preview", "No attack is scheduled in the current mission.")
    work = attach_uav_metadata(schedule)
    source_order = [source_label_for_uav(uav_id) for uav_id in ordered_uav_ids(work["uav_id"].tolist())]
    fig = px.timeline(
        work,
        x_start="start_s",
        x_end="end_s",
        y="source_label",
        color="attack_type",
        title="Attack Schedule Preview",
        labels={"source_label": "Data source", "attack_type": "Attack"},
        category_orders={"source_label": source_order},
        hover_data=["uav_id", "dataset_name", "source_type", "intensity", "replay_mode"],
    )
    fig.update_layout(template="plotly_white", height=max(220, 90 + 40 * len(work["source_label"].unique())))
    fig.update_xaxes(title_text="Time (s)")
    return fig


def uav_preview_table(config: dict[str, Any]) -> pd.DataFrame:
    uavs = build_uavs(config)
    schedule = attack_schedule_frame_for_config(config)
    pure_ids_mode = str(config.get("dataset_replay_mode", "") or "").strip().lower() == "pure_ids_csv"
    grouped_schedule: dict[str, list[dict[str, Any]]] = {}
    if not schedule.empty:
        for row in schedule.to_dict("records"):
            grouped_schedule.setdefault(str(row["uav_id"]), []).append(row)
    rows: list[dict[str, Any]] = []
    for idx, uav in enumerate(uavs):
        attack_rows = grouped_schedule.get(uav.uav_id, [])
        attack_plan = (
            "; ".join(f"{row['attack_type']} {row['start_s']:.0f}-{row['end_s']:.0f}s" for row in attack_rows)
            if attack_rows
            else ("csv_driven_labels" if pure_ids_mode else "benign")
        )
        rows.append(
            {
                "uav_id": uav.uav_id,
                "dataset_display": dataset_display_for_uav(uav.uav_id, annotate_role=False),
                "dataset_name": dataset_name_for_uav(uav.uav_id),
                "source_type": uav_source_type(uav.uav_id),
                "mission_context": str(uav.mission_context),
                "start_delay_s": float(uav.start_delay_s),
                "route_length_m": float(uav.route_length_m),
                "cruise_altitude_m": float(uav.cruise_altitude_m),
                "cruise_speed_mps": float(uav.cruise_speed_mps),
                "hover_duration_s": float(uav.hover_duration_s),
                "battery_capacity_wh": float(uav.battery_capacity_wh),
                "attack_plan": attack_plan,
                "stagger_index": idx,
            }
        )
    return pd.DataFrame(rows)


def render_snapshot_metrics(
    records: pd.DataFrame,
    detections: pd.DataFrame,
    responses: pd.DataFrame,
    uav_id: str,
    current_time_s: float,
) -> None:
    snapshot = latest_records_by_uav(records, current_time_s=current_time_s)
    snapshot = snapshot[snapshot["uav_id"] == uav_id]
    if snapshot.empty:
        st.warning("No snapshot available for the selected UAV and time.")
        return
    row = snapshot.iloc[0]
    detection_row = latest_detection_at(detections, uav_id, current_time_s)
    response_row = selected_responses(responses, uav_id, current_time_s=current_time_s, truncate=True)
    response_count = len(response_row)
    ood_score = 0.0 if detection_row is None else float(detection_row["ood_score"])
    alert_label = "normal" if detection_row is None else str(detection_row["alert_level"])
    alert_reason = "none" if detection_row is None else str(detection_row.get("alert_reason", "none") or "none")
    known_attack_alert = False if detection_row is None else bool(detection_row.get("known_attack_alert", False))
    ood_alert = False if detection_row is None else bool(detection_row.get("ood_alert", detection_row.get("is_ood", False)))
    known_attack_pred_labels = "" if detection_row is None else str(detection_row.get("known_attack_pred_labels", "") or "")

    cols = st.columns(4)
    cols[0].metric("UAV phase / context", f"{str(row['mission_phase'])} / {str(row.get('mission_context', 'n/a'))}")
    cols[1].metric("battery_soc", format_pct(float(row["battery_soc"])))
    cols[2].metric("speed / altitude", f"{float(row['speed']):.1f} m/s / {float(row['altitude']):.1f} m")
    cols[3].metric("bytes_up / bytes_down", f"{format_bytes(int(row['bytes_up']))} / {format_bytes(int(row['bytes_down']))}")

    cols = st.columns(5)
    cols[0].metric(
        "RSSI / latency / loss",
        f"{float(row['rssi']):.1f} dBm / {float(row['latency_ms']):.1f} ms / {float(row['loss_rate']) * 100.0:.1f}%",
    )
    cols[1].metric("OOD score", f"{ood_score:.3f}")
    cols[2].metric("Alert level / reason", f"{alert_label} / {alert_reason}")
    cols[3].metric("Known / OOD alert", f"{'Yes' if known_attack_alert else 'No'} / {'Yes' if ood_alert else 'No'}")
    cols[4].metric("Responses triggered", str(response_count))

    if known_attack_pred_labels:
        st.caption(f"Known attack predicted labels: `{known_attack_pred_labels}`")

    cols = st.columns(4)
    cols[0].metric(
        "distance / wind / obstacle",
        f"{float(row.get('distance_to_gcs', 0.0)):.1f} m / {float(row.get('wind_level', 0.0)):.2f} / {float(row.get('obstacle_factor', 0.0)):.2f}",
    )
    cols[1].metric("Cumulative energy", format_wh(float(row["cumulative_energy_wh"])))
    cols[2].metric("Mission context", str(row.get("mission_context", "n/a")))
    cols[3].metric("Attack state", str(row.get("attack_type", "benign")))


def render_mission_page() -> None:
    apply_pending_mission_widget_sync()
    st.title("Mission")
    st.caption(
        "Select any subset of the 6 fixed UAV/data-source bindings for mission simulation. "
        "`uav_04` is intentionally hidden and skipped."
    )
    if not SIMULATION_ENABLED:
        st.warning(
            "Mission simulation is turned off in the current dashboard build. "
            "You can still inspect the mission configuration and preview, but no run will be executed."
        )

    demo_col, info_col = st.columns([1, 2])
    if demo_col.button(
        "Run built-in demo scene",
        type="primary",
        width="stretch",
        disabled=not SIMULATION_ENABLED,
        help=None if SIMULATION_ENABLED else SIMULATION_DISABLED_MESSAGE,
    ):
        try:
            config = clone_pure_ids_csv_replay_config()
            sync_widget_state(config, overwrite=True)
            run_and_store(config)
            st.success(
                "Built-in demo completed. Open **Live Monitor**, **Replay**, or **Analysis** in the sidebar to view results. "
                "This run replays the full filtered CSV with online detection and may take several minutes the first time."
            )
        except Exception as exc:  # pragma: no cover - UI guard
            st.session_state.dashboard_error = str(exc)
            st.exception(exc)
    info_col.info(
        "Fixed bindings: "
        "`uav_01` -> UAV-NDD, "
        "`uav_02` -> GCS-to-UAV Updated, "
        "`uav_03` -> ISOT Drone Dataset, "
        "`uav_05` -> UNSW-NB15, "
        "`uav_06` -> ECU-IoFT-main, "
        "`uav_07` -> UAVIDS. "
        "Each UAV keeps a fixed `dataset_name` / `source_type` binding, and `uav_04` is not shown."
    )

    st.selectbox("Data Source Mode", options=list(DATA_SOURCE_MODE_OPTIONS), key=DATA_SOURCE_MODE_WIDGET_KEY)
    selected_data_source_mode = str(st.session_state.get(DATA_SOURCE_MODE_WIDGET_KEY, SIMULATION_MIXED_REPLAY_LABEL))
    pure_ids_mode = selected_data_source_mode == PURE_IDS_CSV_REPLAY_LABEL
    apply_data_source_mode_widget_overrides(selected_data_source_mode)
    if pure_ids_mode:
        st.info(
            "当前模式下攻击标签来自 CSV 的 `split` / `label` / `recommended_partition`，"
            "手动攻击配置不参与数据生成。"
        )
        st.caption(
            f"CSV: `{PURE_IDS_CSV_PATH}` | split filter: `test_id`, `test_ood` | artifact: `{PURE_IDS_ARTIFACT_PATH}`"
        )

    st.subheader("UAV Fleet")
    st.multiselect(
        "Participating UAVs",
        options=list(UAV_ID_ORDER),
        format_func=uav_selection_label,
        key=SELECTED_UAVS_WIDGET_KEY,
        help="Select one or more UAVs that will participate in the simulated mission.",
    )
    selected_uav_ids = current_selected_uav_ids_from_widgets()
    st.session_state[WIDGET_KEYS["uav_count"]] = len(selected_uav_ids)
    cols = st.columns(4)
    cols[0].metric("Selected UAVs", str(len(selected_uav_ids)))
    cols[1].number_input("Duration (s)", min_value=10.0, max_value=600.0, step=1.0, key=WIDGET_KEYS["duration_s"])
    cols[2].number_input("dt (s)", min_value=0.1, max_value=10.0, step=0.1, key=WIDGET_KEYS["dt_s"])
    cols[3].number_input("Random seed", min_value=0, max_value=99999, step=1, key=WIDGET_KEYS["seed"])
    st.caption("Available participants: `uav_01`, `uav_02`, `uav_03`, `uav_05`, `uav_06`, `uav_07`.")
    cols = st.columns(3)
    cols[0].number_input(
        "Records / UAV / second",
        min_value=0,
        max_value=100000,
        step=1,
        key=WIDGET_KEYS["records_per_uav_per_step"],
        help="Used when Target records / UAV is 0; otherwise the per-step emit count is derived from the target total.",
    )
    cols[1].number_input(
        "Target records / UAV",
        min_value=0,
        max_value=1000000,
        step=1,
        key=WIDGET_KEYS["target_records_per_uav"],
        help="If greater than 0, evenly distribute this many replayed records across all simulation steps for each UAV.",
    )
    cols[2].checkbox(
        "Export summary only",
        key=WIDGET_KEYS["export_summary_only"],
        disabled=pure_ids_mode,
        help="When enabled, the saved JSON omits full records and writes records_omitted_count instead.",
    )

    st.text_input(
        "Scenario profile path",
        key=WIDGET_KEYS["scenario_profile_path"],
        help="YAML file that defines mission templates, channel factors, benign drift, thermal model, and per-UAV attack plans.",
    )
    st.caption(
        "Mission context templates available in the default profile: "
        + ", ".join(f"`{name}`" for name in SUPPORTED_MISSION_CONTEXTS)
    )

    cols = st.columns(5)
    cols[0].number_input("Start spacing (s)", min_value=0.0, max_value=30.0, step=1.0, key=WIDGET_KEYS["start_spacing_s"])
    cols[1].number_input("Route length (m)", min_value=50.0, max_value=5000.0, step=10.0, key=WIDGET_KEYS["route_length_m"])
    cols[2].number_input("Hover duration (s)", min_value=0.0, max_value=300.0, step=1.0, key=WIDGET_KEYS["hover_duration_s"])
    cols[3].number_input("Cruise altitude (m)", min_value=10.0, max_value=500.0, step=5.0, key=WIDGET_KEYS["cruise_altitude_m"])
    cols[4].number_input("Cruise speed (m/s)", min_value=1.0, max_value=60.0, step=0.5, key=WIDGET_KEYS["cruise_speed_mps"])

    st.subheader("Dataset-Driven Attack Replay")
    cols = st.columns(2)
    cols[0].selectbox(
        "Attack mode",
        options=list(ATTACK_MODE_OPTIONS),
        key=WIDGET_KEYS["attack_mode"],
        disabled=pure_ids_mode,
    )
    cols[1].selectbox(
        "Replay mode",
        options=list(ATTACK_REPLAY_MODES),
        key=WIDGET_KEYS["attack_replay_mode"],
        disabled=pure_ids_mode,
    )

    attack_mode = str(st.session_state[WIDGET_KEYS["attack_mode"]])
    mission_duration_s = float(st.session_state[WIDGET_KEYS["duration_s"]])
    selected_uav_count = len(selected_uav_ids)
    seed_value = int(st.session_state[WIDGET_KEYS["seed"]])

    replay_pool = None if pure_ids_mode else cached_attack_replay_pool(seed_value)
    if (not pure_ids_mode) and replay_pool is None:
        st.warning("Default replay datasets are unavailable, so attack windows will fall back to synthetic records.")

    if not pure_ids_mode:
        if attack_mode == "mixed_attack":
            st.caption("`mixed_attack` allows one or more attack types per UAV. With only one UAV, select at least two attack types.")
        else:
            st.caption("`single_attack` uses exactly one attack type per UAV.")

    if not selected_uav_ids:
        st.warning("Select at least one UAV to configure attack replay and run the mission.")

    for spec in active_uav_specs(selected_uav_ids=selected_uav_ids):
        uav_id = spec["uav_id"]
        options = attack_type_options_for_uav(uav_id, seed_value)
        if not options:
            options = list(CANONICAL_ATTACK_OPTIONS)
        single_key = attack_plan_widget_key(uav_id, "attack_type_single")
        multi_key = attack_plan_widget_key(uav_id, "attack_types_multi")
        if str(st.session_state.get(single_key, "")) not in options:
            st.session_state[single_key] = options[0]
        current_multi = [item for item in normalize_attack_selection(st.session_state.get(multi_key, [])) if item in options]
        if not current_multi:
            default_multi_count = 2 if attack_mode == "mixed_attack" and selected_uav_count == 1 else 1
            st.session_state[multi_key] = options[: min(default_multi_count, len(options))]
        with st.container(key=f"mission_attack_plan_{uav_id}"):
            st.markdown(f"**{uav_id}** · `{dataset_display_for_uav(uav_id, annotate_role=False)}`")
            cols = st.columns([1.2, 1.1, 1.8, 1.0, 1.0, 1.0])
            cols[0].text_input(
                "dataset_name",
                value=str(spec.get("dataset_name", "")),
                disabled=True,
                key=attack_plan_widget_key(uav_id, "dataset_name"),
            )
            cols[1].text_input(
                "source_type",
                value=str(spec.get("source_type", "")),
                disabled=True,
                key=attack_plan_widget_key(uav_id, "source_type"),
            )
            cols[2].multiselect(
                "Attack type",
                options=options,
                key=multi_key,
                max_selections=1 if attack_mode == "single_attack" else None,
                disabled=pure_ids_mode,
                help=(
                    "Select exactly one attack type for this UAV."
                    if attack_mode == "single_attack"
                    else "Select one or more attack types for this UAV."
                ),
            )
            normalized_selection = normalize_attack_selection(st.session_state.get(multi_key, []))
            if normalized_selection:
                st.session_state[single_key] = normalized_selection[0]
            cols[3].number_input(
                "Start time (s)",
                min_value=0.0,
                max_value=max(mission_duration_s, 1.0),
                step=1.0,
                key=attack_plan_widget_key(uav_id, "attack_start_s"),
                disabled=pure_ids_mode,
            )
            cols[4].number_input(
                "Duration (s)",
                min_value=1.0,
                max_value=max(mission_duration_s, 1.0),
                step=1.0,
                key=attack_plan_widget_key(uav_id, "attack_duration_s"),
                disabled=pure_ids_mode,
            )
            cols[5].number_input(
                "Intensity",
                min_value=0.1,
                max_value=8.0,
                step=0.05,
                key=attack_plan_widget_key(uav_id, "attack_intensity"),
                disabled=pure_ids_mode,
            )
            if uav_id == "uav_05":
                st.info("UNSW-NB15 is used as an external non-UAV OOD/generalization dataset.")

    st.subheader("Detection Parameters")
    cols = st.columns(4)
    cols[0].checkbox("Enable online detection", key=WIDGET_KEYS["enable_online_detection"], disabled=pure_ids_mode)
    cols[1].selectbox("Response strategy", options=list(SUPPORTED_RESPONSE_ACTIONS), key=WIDGET_KEYS["response_strategy"])
    cols[2].number_input("Window size", min_value=2, max_value=64, step=1, key=WIDGET_KEYS["window_size"], disabled=pure_ids_mode)
    cols[3].number_input("Stride", min_value=1, max_value=32, step=1, key=WIDGET_KEYS["stride"], disabled=pure_ids_mode)

    cols = st.columns(5)
    cols[0].number_input("Battery capacity (Wh)", min_value=20.0, max_value=500.0, step=5.0, key=WIDGET_KEYS["battery_capacity_wh"])
    cols[1].number_input("Bootstrap benign duration (s)", min_value=10.0, max_value=300.0, step=1.0, key=WIDGET_KEYS["bootstrap_duration_s"])
    cols[2].number_input("Bank k", min_value=1, max_value=32, step=1, key=WIDGET_KEYS["bank_k"])
    cols[3].slider("Bootstrap q_ood", min_value=0.50, max_value=0.99, step=0.01, key=WIDGET_KEYS["bootstrap_q_ood"])
    cols[4].slider("Threshold margin", min_value=0.0, max_value=0.50, step=0.01, key=WIDGET_KEYS["bootstrap_threshold_margin"])

    cols = st.columns([3, 1])
    cols[0].text_input(
        "Artifact path (optional)",
        key=WIDGET_KEYS["artifact_path"],
        disabled=pure_ids_mode,
        help="Use an existing artifact.pt; leave blank to bootstrap a demo detector.",
    )
    cols[1].number_input("Top suspicious records", min_value=1, max_value=20, step=1, key=WIDGET_KEYS["top_records"])

    config = current_config_from_widgets()

    run_col, preview_col = st.columns([1, 2])
    if run_col.button(
        "Run current mission",
        width="stretch",
        disabled=(not SIMULATION_ENABLED or not selected_uav_ids),
        help=None if SIMULATION_ENABLED else SIMULATION_DISABLED_MESSAGE,
    ):
        try:
            run_and_store(config)
            export_paths = (
                st.session_state.dashboard_payload.get("export_paths", {})
                if isinstance(st.session_state.dashboard_payload, Mapping)
                else {}
            )
            latest_json_path = str(export_paths.get("dashboard_json_latest", "") or export_paths.get("dashboard_json", "")).strip()
            success_message = (
                "Simulation completed. Open Analysis to inspect the summary-only run."
                if bool(config.get("export_summary_only", False))
                else "Simulation completed. Open Live Monitor, Replay, or Analysis to inspect the run."
            )
            if latest_json_path:
                st.success(
                    f"{success_message} Saved JSON: {format_output_path(latest_json_path)}"
                )
            else:
                st.success(success_message)
        except Exception as exc:  # pragma: no cover - UI guard
            st.session_state.dashboard_error = str(exc)
            st.exception(exc)
    preview_col.caption(
        "The mission preview updates immediately; the simulation only runs when you click the button."
        if SIMULATION_ENABLED
        else "The mission preview updates immediately, but mission execution is currently disabled."
    )

    left, right = st.columns([1, 1])
    try:
        preview_table = uav_preview_table(config)
        preview_chart = attack_schedule_preview(config)
    except Exception as exc:  # pragma: no cover - UI guard
        st.warning(f"Preview unavailable: {exc}")
    else:
        with left:
            st.markdown("**UAV plan preview**")
            st.dataframe(preview_table, width="stretch")
        with right:
            st.plotly_chart(preview_chart, width="stretch")

    payload = st.session_state.dashboard_payload
    if payload is not None:
        summary = payload["summary"]
        st.subheader("Latest Run Summary")
        cols = st.columns(5)
        cols[0].metric("Mission success", "Yes" if summary["mission_success"] else "No")
        cols[1].metric("Alerts", str(summary["alert_count"]))
        cols[2].metric("Responses", str(summary["response_count"]))
        cols[3].metric("Total energy", format_wh(summary["total_energy_wh"]))
        cols[4].metric("Peak OOD score", f"{summary['peak_ood_score']:.3f}")


def render_live_monitor_page() -> None:
    st.title("Live Monitor")
    payload = ensure_payload_or_offer_demo()
    if payload is None:
        return

    records = payload["records"]
    detections = payload["detections"]
    responses = payload["responses"]
    time_points = payload["time_points"]
    if not time_points:
        if payload_is_summary_only(payload):
            st.info("This run was loaded in summary-only mode. Timeline samples were skipped to keep the dashboard fast.")
        else:
            st.warning("The latest payload does not contain simulation samples.")
        return

    uav_ids = ordered_uav_ids(records["uav_id"].unique().tolist())
    if str(st.session_state.live_selected_uav) not in uav_ids:
        st.session_state.live_selected_uav = uav_ids[0]
    control_cols = st.columns([1, 2, 1, 1])
    control_cols[0].selectbox("Selected UAV", options=uav_ids, key="live_selected_uav")
    control_cols[1].slider(
        "Live time cursor",
        min_value=0,
        max_value=len(time_points) - 1,
        key="live_time_index",
        help="Shows the mission as if the stream had reached this point in time.",
    )
    if control_cols[2].button("Step +1", width="stretch"):
        st.session_state.live_time_index = min(st.session_state.live_time_index + 1, len(time_points) - 1)
    if control_cols[3].button("Jump to latest", width="stretch"):
        st.session_state.live_time_index = len(time_points) - 1

    current_time_s = float(time_points[st.session_state.live_time_index])
    selected_uav = str(st.session_state.live_selected_uav)
    st.caption(f"Live cursor: `t = {current_time_s:.1f}s`")

    st.markdown("**Current UAV status**")
    st.dataframe(fleet_status_frame(records, detections, current_time_s), width="stretch")

    render_snapshot_metrics(records, detections, responses, selected_uav, current_time_s)

    st.markdown("**Fleet telemetry**")
    fleet_snapshot = latest_records_by_uav(records, current_time_s=current_time_s)[
        ["uav_id", "mission_context", "speed", "altitude", "cpu_load", "board_temperature_c", "cumulative_energy_wh"]
    ]
    st.dataframe(fleet_snapshot, width="stretch")

    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            line_chart(
                selected_records(records, selected_uav, current_time_s=current_time_s, truncate=True),
                "timestamp",
                ["battery_soc", "cumulative_energy_wh"],
                "Battery and Cumulative Energy",
                labels={"battery_soc": "battery_soc (%)", "cumulative_energy_wh": "cumulative_energy_wh"},
                current_time_s=current_time_s,
            ),
            width="stretch",
        )
        st.plotly_chart(
            line_chart(
                selected_records(records, selected_uav, current_time_s=current_time_s, truncate=True),
                "timestamp",
                ["speed", "altitude"],
                "Speed and Altitude",
                labels={"speed": "speed (m/s)", "altitude": "altitude (m)"},
                current_time_s=current_time_s,
            ),
            width="stretch",
        )
        st.plotly_chart(attack_timeline_chart(records, selected_uav, current_time_s=current_time_s, truncate=True), width="stretch")
        st.plotly_chart(ood_timeline_chart(detections, selected_uav, current_time_s=current_time_s, truncate=True), width="stretch")
    with right:
        st.plotly_chart(
            line_chart(
                selected_records(records, selected_uav, current_time_s=current_time_s, truncate=True),
                "timestamp",
                ["bytes_up", "bytes_down"],
                "Throughput Timeline",
                labels={"bytes_up": "bytes_up", "bytes_down": "bytes_down"},
                current_time_s=current_time_s,
            ),
            width="stretch",
        )
        st.plotly_chart(
            line_chart(
                selected_records(records, selected_uav, current_time_s=current_time_s, truncate=True),
                "timestamp",
                ["rssi", "latency_ms", "loss_rate"],
                "Link Quality Timeline",
                labels={"rssi": "RSSI (dBm)", "latency_ms": "latency (ms)", "loss_rate": "loss_rate"},
                current_time_s=current_time_s,
            ),
            width="stretch",
        )
        st.plotly_chart(
            alert_timeline_chart(detections, responses, selected_uav, current_time_s=current_time_s, truncate=True),
            width="stretch",
        )
        st.plotly_chart(
            line_chart(
                selected_records(records, selected_uav, current_time_s=current_time_s, truncate=True),
                "timestamp",
                ["flight_energy_wh", "communication_energy_wh", "detection_energy_wh"],
                "Per-step Energy",
                labels={
                    "flight_energy_wh": "flight",
                    "communication_energy_wh": "communication",
                    "detection_energy_wh": "detection",
                },
                current_time_s=current_time_s,
            ),
            width="stretch",
        )


def render_replay_page() -> None:
    st.title("Replay")
    payload = ensure_payload_or_offer_demo()
    if payload is None:
        return

    records = payload["records"]
    detections = payload["detections"]
    responses = payload["responses"]
    schedule = payload["attack_schedule"]
    time_points = payload["time_points"]
    if not time_points:
        if payload_is_summary_only(payload):
            st.info("This run was loaded in summary-only mode. Replay samples were skipped to keep the dashboard fast.")
        else:
            st.warning("The latest payload does not contain simulation samples.")
        return

    uav_ids = ordered_uav_ids(records["uav_id"].unique().tolist())
    if str(st.session_state.replay_selected_uav) not in uav_ids:
        st.session_state.replay_selected_uav = uav_ids[0]
    control_cols = st.columns([1, 2, 1, 1])
    control_cols[0].selectbox("Selected UAV", options=uav_ids, key="replay_selected_uav")
    control_cols[1].slider(
        "Replay timeline",
        min_value=0,
        max_value=len(time_points) - 1,
        key="replay_time_index",
        help="Scrub through the mission timeline.",
    )
    if control_cols[2].button("Previous", width="stretch"):
        st.session_state.replay_time_index = max(st.session_state.replay_time_index - 1, 0)
    if control_cols[3].button("Next", width="stretch"):
        st.session_state.replay_time_index = min(st.session_state.replay_time_index + 1, len(time_points) - 1)

    current_time_s = float(time_points[st.session_state.replay_time_index])
    selected_uav = str(st.session_state.replay_selected_uav)
    st.caption(f"Replay cursor: `t = {current_time_s:.1f}s`")

    st.markdown("**Replay snapshot**")
    st.dataframe(fleet_status_frame(records, detections, current_time_s), width="stretch")

    render_snapshot_metrics(records, detections, responses, selected_uav, current_time_s)

    st.plotly_chart(
        fleet_attack_timeline_chart(schedule, current_time_s=current_time_s),
        width="stretch",
    )

    left, right = st.columns(2)
    with left:
        ood_frame = attach_uav_metadata(detections, uav_col="group_id") if not detections.empty else detections.copy()
        ood_frame = ood_frame[ood_frame["group_id"].notna()].copy() if not ood_frame.empty else ood_frame
        source_order = [source_label_for_uav(uav_id) for uav_id in ordered_uav_ids(ood_frame["group_id"].tolist())] if not ood_frame.empty else []
        st.plotly_chart(
            fleet_metric_chart(
                ood_frame,
                "simulation_time_s",
                "ood_score",
                "source_label",
                "OOD Score by Data Source",
                labels={"simulation_time_s": "Time (s)", "ood_score": "OOD score", "source_label": "Data source"},
                category_orders={"source_label": source_order},
                current_time_s=current_time_s,
            ),
            width="stretch",
        )
    with right:
        st.plotly_chart(fleet_alert_event_chart(detections, current_time_s=current_time_s), width="stretch")


def render_analysis_page() -> None:
    st.title("Analysis")
    payload = ensure_payload_or_offer_demo()
    if payload is None:
        return

    records = payload["records"]
    detections = payload["detections"]
    responses = payload["responses"]
    summary = payload["summary"]
    config = payload["config"]
    summary_only_mode = payload_is_summary_only(payload)
    mode_summary = detection_mode_summary(records, detections)
    single_summary = detection_mode_row(mode_summary, "single_attack")
    mixed_summary = detection_mode_row(mode_summary, "mixed_attack")
    per_uav_metrics = per_uav_analysis(records, detections)
    per_dataset_metrics = per_dataset_metrics_frame(records, detections)
    per_source_type_metrics = per_source_type_metrics_frame(records, detections)
    external_ood_metrics = external_ood_metrics_frame(records, detections)

    cols = st.columns(6)
    cols[0].metric("Mission success", "Yes" if summary["mission_success"] else "No")
    cols[1].metric("Alerts", str(summary["alert_count"]))
    cols[2].metric("False alerts", str(summary["false_alert_count"]))
    cols[3].metric("Responses", str(summary["response_count"]))
    cols[4].metric("Average alert delay", f"{summary['average_alert_delay']:.2f} s")
    cols[5].metric("Total energy", format_wh(summary["total_energy_wh"]))

    cols = st.columns(6)
    cols[0].metric("Peak OOD score", f"{summary['peak_ood_score']:.3f}")
    cols[1].metric("Mean RSSI", f"{summary['mean_rssi']:.2f} dBm")
    cols[2].metric("Mean latency", f"{summary['mean_latency_ms']:.2f} ms")
    cols[3].metric("Mean loss rate", f"{summary['mean_loss_rate'] * 100.0:.2f}%")
    cols[4].metric("Mean CPU load", f"{summary['mean_cpu_load'] * 100.0:.1f}%")
    cols[5].metric("Peak board temp", f"{summary['peak_board_temperature_c']:.1f} C")

    cols = st.columns(6)
    cols[0].metric("Pred alert windows", str(int(summary.get("pred_alert_window_count", 0))))
    cols[1].metric("Known-attack alerts", str(int(summary.get("pred_known_attack_alert_window_count", 0))))
    cols[2].metric("OOD alerts", str(int(summary.get("pred_ood_window_count", 0))))
    cols[3].metric("Dual-channel alerts", str(int(summary.get("pred_dual_alert_window_count", 0))))
    cols[4].metric("False alert windows", str(int(summary.get("false_alert_window_count", 0))))
    cols[5].metric("IDS energy", format_wh(float(summary.get("ids_energy_wh", 0.0))))

    rate_cols = st.columns(3)
    rate_cols[0].metric("IDS / total energy", f"{float(summary.get('ids_energy_ratio', 0.0)) * 100.0:.2f}%")
    uav_only_false_alert_rate = summary.get("uav_only_false_alert_rate")
    external_false_alert_rate = summary.get("external_non_uav_false_alert_rate")
    rate_cols[1].metric(
        "UAV-only false alert rate",
        "n/a" if uav_only_false_alert_rate is None else f"{float(uav_only_false_alert_rate) * 100.0:.2f}%",
    )
    rate_cols[2].metric(
        "external_non_uav false alert rate",
        "n/a" if external_false_alert_rate is None else f"{float(external_false_alert_rate) * 100.0:.2f}%",
    )

    st.caption(f"Current replay attack mode: `{config.get('attack_mode', 'single_attack')}`")
    st.caption("Alert channels: `known_attack` = classified known attack, `ood` = unknown/OOD alert, `known_attack+ood` = both channels fired.")

    if not summary_only_mode:
        st.markdown("**per_uav_metrics**")
        st.dataframe(per_uav_metrics, width="stretch")

        metrics_left, metrics_right = st.columns(2)
        with metrics_left:
            st.markdown("**per_dataset_metrics**")
            st.dataframe(per_dataset_metrics, width="stretch")
        with metrics_right:
            st.markdown("**per_source_type_metrics**")
            st.dataframe(per_source_type_metrics, width="stretch")

    diagnostic_rows = summary.get("per_uav_diagnostic_rows", [])
    if diagnostic_rows:
        st.markdown("**ood_diagnostics_per_uav**")
        st.dataframe(sort_frame_by_uav_order(pd.DataFrame(diagnostic_rows)), width="stretch")
    direction_diagnostics = summary.get("score_direction_diagnostics", {})
    if isinstance(direction_diagnostics, Mapping):
        st.markdown("**score_direction_diagnostics**")
        report_rows = direction_diagnostics.get("configured_report", [])
        overview = {key: value for key, value in direction_diagnostics.items() if key != "configured_report"}
        diag_left, diag_right = st.columns([1.4, 1.6])
        with diag_left:
            st.json(overview)
        with diag_right:
            if isinstance(report_rows, list) and report_rows:
                st.dataframe(pd.DataFrame(report_rows), width="stretch")
            else:
                st.info("No score-direction calibration report is available for the current detector.")

    if summary_only_mode:
        st.info("This run was loaded in summary-only mode. Record-level tables, timelines, and heavy comparison charts were skipped.")
        return

    st.markdown("**external_ood_metrics**")
    if external_ood_metrics.empty:
        st.info("No external OOD/generalization source was active in the current mission.")
    else:
        st.dataframe(external_ood_metrics, width="stretch")

    left, right = st.columns(2)
    with left:
        st.markdown("**Single-attack detection results**")
        if int(single_summary["attack_windows"]) <= 0:
            st.info("No single-attack windows were present in the current replay.")
        cols = st.columns(4)
        cols[0].metric("Attack windows", str(int(single_summary["attack_windows"])))
        cols[1].metric("Detected windows", str(int(single_summary["detected_windows"])))
        cols[2].metric("Detection rate", f"{float(single_summary['detection_rate']) * 100.0:.1f}%")
        cols[3].metric("Peak OOD", f"{float(single_summary['peak_ood_score']):.3f}")
        cols = st.columns(3)
        cols[0].metric("Missed windows", str(int(single_summary["missed_windows"])))
        cols[1].metric("Critical alerts", str(int(single_summary["critical_alerts"])))
        cols[2].metric("Mean OOD", f"{float(single_summary['mean_ood_score']):.3f}")
    with right:
        st.markdown("**Mixed-attack detection results**")
        if int(mixed_summary["attack_windows"]) <= 0:
            st.info("No mixed-attack windows were present in the current replay.")
        cols = st.columns(4)
        cols[0].metric("Attack windows", str(int(mixed_summary["attack_windows"])))
        cols[1].metric("Detected windows", str(int(mixed_summary["detected_windows"])))
        cols[2].metric("Detection rate", f"{float(mixed_summary['detection_rate']) * 100.0:.1f}%")
        cols[3].metric("Peak OOD", f"{float(mixed_summary['peak_ood_score']):.3f}")
        cols = st.columns(3)
        cols[0].metric("Missed windows", str(int(mixed_summary["missed_windows"])))
        cols[1].metric("Critical alerts", str(int(mixed_summary["critical_alerts"])))
        cols[2].metric("Mean OOD", f"{float(mixed_summary['mean_ood_score']):.3f}")

    energy_left, energy_right = st.columns(2)
    with energy_left:
        st.plotly_chart(energy_share_chart(records), width="stretch")
    with energy_right:
        st.plotly_chart(energy_breakdown_chart(records), width="stretch")

    if not detections.empty:
        analysis_uav_ids = ordered_uav_ids(records["uav_id"].unique().tolist())
        if str(st.session_state.analysis_selected_uav) not in analysis_uav_ids:
            st.session_state.analysis_selected_uav = analysis_uav_ids[0]
        selected_uav = st.selectbox("Analysis focus UAV", options=analysis_uav_ids, key="analysis_selected_uav")
        cols = st.columns(2)
        with cols[0]:
            st.plotly_chart(ood_timeline_chart(detections, selected_uav), width="stretch")
        with cols[1]:
            st.plotly_chart(alert_timeline_chart(detections, responses, selected_uav), width="stretch")


def render_sidebar() -> str:
    st.sidebar.title("UAV IDS Dashboard")
    page_options = ["Mission", "Live Monitor", "Replay", "Analysis"] if SIMULATION_ENABLED else ["Mission"]
    page = st.sidebar.radio("Page", page_options)
    payload = st.session_state.dashboard_payload
    if not SIMULATION_ENABLED:
        st.sidebar.caption("Mission simulation is disabled.")
    elif payload is None:
        st.sidebar.caption("No run loaded.")
    else:
        summary = payload["summary"]
        attack_mode = str(payload["config"].get("attack_mode", "single_attack"))
        st.sidebar.caption(
            f"{summary['uav_count']} UAVs | mode={attack_mode} | alerts={summary['alert_count']} | "
            f"energy={summary['total_energy_wh']:.2f} Wh"
        )
        export_paths = payload.get("export_paths", {}) if isinstance(payload, Mapping) else {}
        latest_json_path = str(export_paths.get("dashboard_json_latest", "") or export_paths.get("dashboard_json", "")).strip()
        if latest_json_path:
            st.sidebar.caption(f"Latest JSON: `{format_output_path(latest_json_path)}`")
    return page


def main() -> None:
    init_session_state()
    page = render_sidebar()

    if st.session_state.dashboard_error:
        st.error(st.session_state.dashboard_error)

    if page == "Mission":
        render_mission_page()
    elif page == "Live Monitor":
        render_live_monitor_page()
    elif page == "Replay":
        render_replay_page()
    else:
        render_analysis_page()


if __name__ == "__main__":
    main()
