from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Text,
    func,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from tesseractcli.memory.schema.base import Base

if TYPE_CHECKING:
    from tesseractcli.memory.schema.memory_node_sources import MemoryNodeSource
    from tesseractcli.memory.schema.sessions import Sessions


class MemoryNode(Base):
    __tablename__ = "memory_nodes"

    id: Mapped[str] = mapped_column(
        Text,
        primary_key=True,
        nullable=False,
    )

    session_id: Mapped[str] = mapped_column(
        Text,
        ForeignKey("sessions.session_id", ondelete="CASCADE"),
        nullable=False,
    )
    level: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    memory_type: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        server_default=text("'SUMMARY'"),
    )

    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    token_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
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

    # ----- relations
    session: Mapped[Sessions] = relationship(
        "Sessions",
        back_populates="memory_nodes",
    )

    sources: Mapped[list[MemoryNodeSource]] = relationship(
        "MemoryNodeSource",
        back_populates="memory_node",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    __table_args__ = (
        Index("idx_mem_nodes_session", "session_id"),
        Index("idx_mem_nodes_session_level", "session_id", "level"),
        Index("idx_mem_nodes_status", "status"),
        CheckConstraint("level >= 1", name="ck_mem_nodes_level_positive"),
        CheckConstraint(
            "memory_type IN ('SUMMARY', 'FACT', 'PREFERENCE', 'DECISION')",
            name="ck_mem_nodes_type_valid",
        ),
        CheckConstraint(
            "status IN ('ACTIVE', 'SUPERSEDED', 'INVALIDATED', 'DELETED')",
            name="ck_mem_nodes_status_valid",
        ),
    )

    @property
    def is_active(self) -> bool:
        return self.status == "ACTIVE"
