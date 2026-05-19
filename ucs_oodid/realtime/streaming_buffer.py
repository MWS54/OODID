from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional

GLOBAL_STREAM_KEY = "__stream__global__"


def _normalize_group_id(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _to_record(record: Mapping[str, object]) -> dict[str, object]:
    return dict(record)


@dataclass(frozen=True)
class BufferedWindow:
    window_id: int
    group_id: str | None
    records: list[dict[str, object]]
    mode: str
    target_length: int
    start_marker: float
    end_marker: float
    record_id_col: str = "record_id"
    timestamp_col: str = "timestamp"

    @property
    def valid_count(self) -> int:
        return len(self.records)

    @property
    def record_ids(self) -> list[str]:
        out: list[str] = []
        for idx, record in enumerate(self.records):
            value = record.get(self.record_id_col, idx)
            out.append(str(value))
        return out

    @property
    def timestamps(self) -> list[float]:
        out: list[float] = []
        for record in self.records:
            value = record.get(self.timestamp_col)
            if value is None:
                continue
            out.append(float(value))
        return out


class StreamingWindowBuffer:
    """Realtime sliding-window cache that emits windows as records arrive."""

    def __init__(
        self,
        mode: str = "count",
        window_size: int = 32,
        stride: int = 16,
        time_seconds: float = 2.0,
        adaptive_min_size: int = 8,
        adaptive_max_size: int = 64,
        timestamp_col: str = "timestamp",
        record_id_col: str = "record_id",
        group_col: str | None = None,
    ) -> None:
        self.mode = str(mode or "count").strip().lower()
        if self.mode not in {"count", "time", "adaptive"}:
            raise ValueError(f"Unsupported streaming window mode: {mode}")
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.time_seconds = float(time_seconds)
        self.adaptive_min_size = int(adaptive_min_size)
        self.adaptive_max_size = int(adaptive_max_size)
        self.timestamp_col = str(timestamp_col)
        self.record_id_col = str(record_id_col)
        self.group_col = str(group_col).strip() or None if group_col is not None else None
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        if self.stride <= 0:
            raise ValueError("stride must be positive")
        if self.mode == "time" and self.time_seconds <= 0.0:
            raise ValueError("time_seconds must be positive for time windows")
        if self.mode == "adaptive":
            if self.adaptive_min_size <= 0 or self.adaptive_max_size <= 0:
                raise ValueError("adaptive window sizes must be positive")
            if self.adaptive_min_size > self.adaptive_max_size:
                raise ValueError("adaptive_min_size must be <= adaptive_max_size")

        self._buffers: Dict[str, List[dict[str, object]]] = {}
        self._buffer_offsets: Dict[str, int] = {}
        self._next_index_start: Dict[str, int] = {}
        self._next_time_start: Dict[str, float] = {}
        self._window_counter = 0

    @property
    def expected_window_length(self) -> int:
        if self.mode == "adaptive":
            return int(self.adaptive_max_size)
        return int(self.window_size)

    @property
    def time_stride_seconds(self) -> float:
        return self.time_seconds / 2.0

    def reset(self) -> None:
        self._buffers.clear()
        self._buffer_offsets.clear()
        self._next_index_start.clear()
        self._next_time_start.clear()
        self._window_counter = 0

    def append(self, record: Mapping[str, object]) -> list[BufferedWindow]:
        payload = _to_record(record)
        key = self._group_key(payload)
        buffer = self._buffers.setdefault(key, [])
        buffer.append(payload)
        if self.mode == "time":
            return self._emit_ready_time_windows(key)
        return self._emit_ready_index_windows(key)

    def extend(self, records: list[Mapping[str, object]]) -> list[BufferedWindow]:
        windows: list[BufferedWindow] = []
        for record in records:
            windows.extend(self.append(record))
        return windows

    def _group_key(self, record: Mapping[str, object]) -> str:
        if not self.group_col:
            return GLOBAL_STREAM_KEY
        return _normalize_group_id(record.get(self.group_col)) or GLOBAL_STREAM_KEY

    def _window_group_id(self, key: str) -> str | None:
        return None if key == GLOBAL_STREAM_KEY else key

    def _emit_ready_index_windows(self, key: str) -> list[BufferedWindow]:
        buffer = self._buffers.setdefault(key, [])
        offset = self._buffer_offsets.setdefault(key, 0)
        next_start = self._next_index_start.setdefault(key, offset)
        windows: list[BufferedWindow] = []

        while True:
            local_start = next_start - offset
            if local_start < 0 or local_start >= len(buffer):
                break
            if self.mode == "count":
                if local_start + self.window_size > len(buffer):
                    break
                valid_records = buffer[local_start : local_start + self.window_size]
                raw_end = float(next_start + self.window_size)
            else:
                valid_records, raw_end = self._adaptive_window_records(buffer, local_start, next_start)
                if valid_records is None:
                    break
            windows.append(
                self._build_window(
                    key=key,
                    records=valid_records,
                    start_marker=float(next_start),
                    end_marker=float(raw_end),
                )
            )
            next_start += self.stride

        self._next_index_start[key] = next_start
        self._prune_index_buffer(key)
        return windows

    def _adaptive_window_records(
        self,
        buffer: list[dict[str, object]],
        local_start: int,
        absolute_start: int,
    ) -> tuple[list[dict[str, object]] | None, float]:
        available = len(buffer) - local_start
        if available < self.window_size:
            return None, float(absolute_start)

        anchor = buffer[local_start : local_start + self.window_size]
        if len(anchor) >= 2:
            start_ts = self._record_timestamp(anchor[0])
            end_ts = self._record_timestamp(anchor[-1])
            local_span = max(end_ts - start_ts, 1e-6)
            local_rate = len(anchor) / local_span
            size = int(
                min(
                    max(
                        self.window_size * (1.0 + 0.5 / max(local_rate, 1e-6)),
                        self.adaptive_min_size,
                    ),
                    self.adaptive_max_size,
                )
            )
        else:
            size = int(self.window_size)
        valid_end = min(local_start + size, len(buffer))
        return buffer[local_start:valid_end], float(absolute_start + size)

    def _emit_ready_time_windows(self, key: str) -> list[BufferedWindow]:
        buffer = self._buffers.setdefault(key, [])
        if not buffer:
            return []
        first_ts = self._record_timestamp(buffer[0])
        next_start = self._next_time_start.setdefault(key, first_ts)
        latest_ts = self._record_timestamp(buffer[-1])
        windows: list[BufferedWindow] = []

        while latest_ts >= next_start + self.time_seconds:
            window_end = next_start + self.time_seconds
            valid_records = [
                record
                for record in buffer
                if next_start <= self._record_timestamp(record) < window_end
            ]
            if valid_records:
                windows.append(
                    self._build_window(
                        key=key,
                        records=valid_records[: self.window_size],
                        start_marker=float(next_start),
                        end_marker=float(window_end),
                    )
                )
            next_start += self.time_stride_seconds

        self._next_time_start[key] = next_start
        self._prune_time_buffer(key)
        return windows

    def _prune_index_buffer(self, key: str) -> None:
        buffer = self._buffers.get(key)
        if buffer is None:
            return
        offset = self._buffer_offsets.get(key, 0)
        next_start = self._next_index_start.get(key, offset)
        drop_count = max(next_start - offset, 0)
        if drop_count <= 0:
            return
        if drop_count >= len(buffer):
            self._buffers[key] = []
            self._buffer_offsets[key] = offset + len(buffer)
            return
        self._buffers[key] = buffer[drop_count:]
        self._buffer_offsets[key] = offset + drop_count

    def _prune_time_buffer(self, key: str) -> None:
        buffer = self._buffers.get(key)
        if not buffer:
            return
        keep_from = self._next_time_start.get(key)
        if keep_from is None:
            return
        first_keep_idx = 0
        for idx, record in enumerate(buffer):
            if self._record_timestamp(record) >= keep_from:
                first_keep_idx = idx
                break
        else:
            self._buffers[key] = []
            return
        if first_keep_idx > 0:
            self._buffers[key] = buffer[first_keep_idx:]

    def _record_timestamp(self, record: Mapping[str, object]) -> float:
        if self.timestamp_col not in record:
            raise ValueError(f"Streaming record is missing timestamp column {self.timestamp_col!r}")
        return float(record[self.timestamp_col])

    def _build_window(
        self,
        key: str,
        records: list[dict[str, object]],
        start_marker: float,
        end_marker: float,
    ) -> BufferedWindow:
        window = BufferedWindow(
            window_id=self._window_counter,
            group_id=self._window_group_id(key),
            records=[dict(record) for record in records],
            mode=self.mode,
            target_length=self.expected_window_length,
            start_marker=float(start_marker),
            end_marker=float(end_marker),
            record_id_col=self.record_id_col,
            timestamp_col=self.timestamp_col,
        )
        self._window_counter += 1
        return window
