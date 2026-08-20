from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import CheckConstraint, DateTime, Index, Text, func, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from tesseractcli.memory.schema.base import Base

if TYPE_CHECKING:
    from tesseractcli.memory.schema.memory_nodes import MemoryNode
    from tesseractcli.memory.schema.messages import Messages


class Sessions(Base):
    __tablename__ = "sessions"

    session_id: Mapped[str] = mapped_column(
        Text,
        primary_key=True,
    )

    workspace_path: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    name: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    last_active_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    status: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        server_default=text("'ACTIVE'"),
    )

    messages: Mapped[list[Messages]] = relationship(
        "Messages",
        back_populates="session",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    memory_nodes: Mapped[list[MemoryNode]] = relationship(
        "MemoryNode",
        back_populates="session",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    __table_args__ = (
        Index(
            "idx_sessions_workspace_status_active",
            "workspace_path",
            "status",
            "last_active_at",
        ),
        CheckConstraint(
            "status IN ('ACTIVE', 'DELETED')", name="ck_sessions_status_valid"
        ),
    )
