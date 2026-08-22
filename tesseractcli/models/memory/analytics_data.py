from __future__ import annotations

from typing import NamedTuple


class UsageCount(NamedTuple):
    name: str | None
    total: int


class TokenUsage(NamedTuple):
    input_tokens: int
    output_tokens: int
    total_tokens: int


class TokenUsageByName(NamedTuple):
    name: str | None
    input_tokens: int
    output_tokens: int
    total_tokens: int


class ToolStats(NamedTuple):
    name: str
    total_calls: int
    success_count: int
    success_rate: float


class SessionMessageCount(NamedTuple):
    session_id: str
    message_count: int


class SessionStatusCount(NamedTuple):
    status: str
    total: int


class LevelCount(NamedTuple):
    level: int
    total: int


class CompressionRatio(NamedTuple):
    raw_message_count: int
    active_node_count: int
    ratio: (
        float | None
    )  # raw_message_count / active_node_count, None if no active nodes
