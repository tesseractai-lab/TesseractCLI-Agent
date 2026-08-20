from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import Boolean, CheckConstraint, ForeignKey, Index, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from tesseractcli.memory.schema.base import Base

if TYPE_CHECKING:
    from tesseractcli.memory.schema.messages import Messages


class ToolMeta(Base):
    __tablename__ = "tools_meta"

    tool_call_id: Mapped[str] = mapped_column(
        Text,
        primary_key=True,
        nullable=False,
    )

    message_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("messages.id", ondelete="CASCADE"),
        nullable=False,
    )

    name: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    args: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    success: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default="0",
    )

    message: Mapped[Messages] = relationship(
        "Messages",
        back_populates="tools_meta",
        uselist=False,
    )

    __table_args__ = (
        Index("idx_tools_meta_message_id", "message_id"),
        Index("idx_tools_meta_message_name", "message_id", "name"),
        Index("idx_tools_meta_success", "success"),
        CheckConstraint("success IN (0, 1)", name="ck_tools_meta_success_bool"),
        CheckConstraint("name != ''", name="ck_tools_meta_name_not_empty"),
    )
