from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Index, Integer, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from tesseractcli.memory.schema.base import Base

if TYPE_CHECKING:
    from tesseractcli.memory.schema.model_meta import ModelMeta
    from tesseractcli.memory.schema.sessions import Sessions
    from tesseractcli.memory.schema.tools_meta import ToolMeta


class Messages(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    session_id: Mapped[str] = mapped_column(
        Text,
        ForeignKey("sessions.session_id", ondelete="CASCADE"),
        nullable=False,
    )

    role: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    # ---Relations

    session: Mapped[Sessions] = relationship(
        "Sessions",
        back_populates="messages",
    )

    model_meta: Mapped[ModelMeta | None] = relationship(
        "ModelMeta",
        back_populates="message",
        cascade="all, delete-orphan",
        uselist=False,
    )

    tools_meta: Mapped[list[ToolMeta]] = relationship(
        "ToolMeta",
        back_populates="message",
        cascade="all, delete-orphan",
    )

    # ---
    __table_args__ = (
        CheckConstraint(
            "role IN ('user', 'assistant', 'system', 'tool')", name="ck_message_role"
        ),
        Index("idx_messages_session_id", "session_id"),
        Index("idx_messages_session_timestamp", "session_id", "timestamp"),
        Index("idx_messages_session_role", "session_id", "role"),
    )
