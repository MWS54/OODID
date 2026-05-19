from __future__ import annotations

from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

SIMULATION_DATASET_BINDINGS: tuple[dict[str, Any], ...] = (
    {
        "uav_id": "uav_01",
        "dataset_name": "uav_ndd",
        "dataset_display": "UAV-NDD",
        "dataset_role": "core_uav",
        "source_type": "uav",
        "simulation_role": "uav_replay",
        "data_filename": "uav_ndd_case1_experiment.csv",
        "aliases": ("UAV-NDD Dataset",),
    },
    {
        "uav_id": "uav_02",
        "dataset_name": "gcs_to_uav_updated",
        "dataset_display": "GCS-to-UAV Updated",
        "dataset_role": "core_uav",
        "source_type": "uav",
        "simulation_role": "uav_replay",
        "data_filename": "gcs_to_uav_updated_experiment.csv",
        "aliases": ("GCS to UAV Updated", "GCS-to-UAV Updated Dataset"),
    },
    {
        "uav_id": "uav_03",
        "dataset_name": "isot_drone",
        "dataset_display": "ISOT Drone Dataset",
        "dataset_role": "core_uav",
        "source_type": "uav",
        "simulation_role": "uav_replay",
        "data_filename": "isot_drone_uav03_experiment.csv",
        "aliases": ("ISOT Drone",),
    },
    {
        "uav_id": "uav_05",
        "dataset_name": "unsw_nb15",
        "dataset_display": "UNSW-NB15",
        "dataset_role": "external_non_uav",
        "source_type": "external_non_uav",
        "simulation_role": "external_ood",
        "source_note": "External OOD/generalization source, not real UAV flight traffic.",
        "data_filename": "unsw_nb15_uav05_experiment.csv",
        "aliases": ("UNSW NB15",),
    },
    {
        "uav_id": "uav_06",
        "dataset_name": "ecu_ioft",
        "dataset_display": "ECU-IoFT-main",
        "dataset_role": "extended_uav",
        "source_type": "uav_iot_wifi",
        "simulation_role": "uav_replay",
        "data_filename": "ecu_ioft_uav06_experiment.csv",
        "aliases": ("ECU-IoFT", "ECU IoFT", "ECU-IoFT Main", "ECU IoFT Main"),
    },
    {
        "uav_id": "uav_07",
        "dataset_name": "uavids",
        "dataset_display": "UAVIDS",
        "dataset_role": "extended_uav",
        "source_type": "uav",
        "simulation_role": "uav_replay",
        "data_filename": "uavids_uav07_experiment.csv",
        "aliases": ("UAVIDS Dataset",),
    },
)

DATASET_METADATA_BY_UAV = {row["uav_id"]: dict(row) for row in SIMULATION_DATASET_BINDINGS}
DATASET_METADATA_BY_NAME = {row["dataset_name"]: dict(row) for row in SIMULATION_DATASET_BINDINGS}

DEFAULT_ATTACK_DATASET_PATHS: dict[str, Path] = {
    row["dataset_name"]: ROOT / "data" / str(row["data_filename"])
    for row in SIMULATION_DATASET_BINDINGS
}

DEFAULT_UAV_DATASET_BINDINGS: dict[str, str] = {
    row["uav_id"]: row["dataset_name"]
    for row in SIMULATION_DATASET_BINDINGS
}


def bounded_dataset_count(value: Any) -> int:
    return max(1, min(int(value), len(SIMULATION_DATASET_BINDINGS)))


def active_simulation_dataset_bindings(uav_count: int) -> list[dict[str, Any]]:
    count = bounded_dataset_count(uav_count)
    return [dict(row) for row in SIMULATION_DATASET_BINDINGS[:count]]


def default_simulation_uav_ids(uav_count: int) -> list[str]:
    return [str(row["uav_id"]) for row in active_simulation_dataset_bindings(uav_count)]


def dataset_name_for_uav(uav_id: str) -> str:
    row = DATASET_METADATA_BY_UAV.get(str(uav_id).strip())
    return "" if row is None else str(row["dataset_name"])


def canonical_dataset_name(dataset_name: str) -> str:
    text = str(dataset_name).strip()
    if not text:
        return ""
    if text in DATASET_METADATA_BY_NAME:
        return text
    return DATASET_NAME_ALIASES.get(_normalize_alias_token(text), text)


def dataset_role_for_name(dataset_name: str) -> str:
    row = DATASET_METADATA_BY_NAME.get(canonical_dataset_name(dataset_name))
    if row is None:
        return "unknown"
    return str(row.get("dataset_role", "unknown"))


def dataset_role_for_uav(uav_id: str) -> str:
    return dataset_role_for_name(dataset_name_for_uav(uav_id))


def dataset_display_name(dataset_name: str, *, annotate_role: bool = False) -> str:
    canonical_name = canonical_dataset_name(dataset_name)
    row = DATASET_METADATA_BY_NAME.get(canonical_name)
    display = canonical_name or "n/a"
    role = "unknown"
    if row is not None:
        display = str(row["dataset_display"])
        role = str(row.get("dataset_role", "unknown"))
    if annotate_role and role == "external_non_uav":
        return f"{display} [external_non_uav]"
    return display


def dataset_display_for_uav(uav_id: str, *, annotate_role: bool = False) -> str:
    canonical_uav_id = str(uav_id).strip()
    row = DATASET_METADATA_BY_UAV.get(canonical_uav_id)
    if row is None:
        return dataset_display_name(dataset_name_for_uav(canonical_uav_id), annotate_role=annotate_role)
    return dataset_display_name(str(row["dataset_name"]), annotate_role=annotate_role)


def _normalize_alias_token(value: Any) -> str:
    return "".join(ch for ch in str(value).strip().lower() if ch.isalnum())


def dataset_name_aliases() -> dict[str, str]:
    aliases: dict[str, str] = {}
    for row in SIMULATION_DATASET_BINDINGS:
        dataset_name = str(row["dataset_name"])
        candidates = [
            dataset_name,
            row["dataset_display"],
            row["uav_id"],
            *tuple(row.get("aliases", ())),
        ]
        for candidate in candidates:
            token = _normalize_alias_token(candidate)
            if token:
                aliases[token] = dataset_name
    return aliases


DATASET_NAME_ALIASES = dataset_name_aliases()


__all__ = [
    "DATASET_METADATA_BY_NAME",
    "DATASET_METADATA_BY_UAV",
    "DATASET_NAME_ALIASES",
    "DEFAULT_ATTACK_DATASET_PATHS",
    "DEFAULT_UAV_DATASET_BINDINGS",
    "SIMULATION_DATASET_BINDINGS",
    "active_simulation_dataset_bindings",
    "bounded_dataset_count",
    "canonical_dataset_name",
    "dataset_display_for_uav",
    "dataset_display_name",
    "dataset_name_for_uav",
    "dataset_role_for_name",
    "dataset_role_for_uav",
    "default_simulation_uav_ids",
]
