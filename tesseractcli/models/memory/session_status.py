from enum import Enum


class SessionStatus(str, Enum):
    ACTIVE = "ACTIVE"
    DELETED = "DELETED"
