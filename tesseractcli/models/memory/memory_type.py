from enum import Enum


class MemoryType(str, Enum):
    SUMMARY = "SUMMARY"
    FACT = "FACT"
    PREFERENCE = "PREFERENCE"
    DECISION = "DECISION"
