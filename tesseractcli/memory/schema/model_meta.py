from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    REAL,
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
    from tesseractcli.memory.schema.messages import Messages


class ModelMeta(Base):
    __tablename__ = "model_meta"

    message_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("messages.id", ondelete="CASCADE"),
        primary_key=True,
        autoincrement=False,
    )

    model_name: Mapped[str | None] = mapped_column(Text, nullable=True)
    pack_name: Mapped[str | None] = mapped_column(Text, nullable=True)

    temperature: Mapped[float] = mapped_column(
        REAL,
        nullable=False,
        server_default=text("0.7"),
    )

    max_tokens: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        server_default=text("4096"),
    )

    input_tokens: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    output_tokens: Mapped[int | None] = mapped_column(
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

    message: Mapped[Messages] = relationship(
        "Messages",
        back_populates="model_meta",
        uselist=False,
    )

    __table_args__ = (
        CheckConstraint(
            "temperature >= 0.0 AND temperature <= 2.0", name="ck_model_meta_temp_range"
        ),
        CheckConstraint("max_tokens > 0", name="ck_model_meta_tokens_positive"),
        CheckConstraint(
            "input_tokens >= 0 OR input_tokens IS NULL",
            name="ck_model_meta_input_non_neg",
        ),
        CheckConstraint(
            "output_tokens >= 0 OR output_tokens IS NULL",
            name="ck_model_meta_output_non_neg",
        ),
        Index("idx_model_meta_message_id", "message_id"),
    )

    @property
    def total_tokens(self) -> int | None:
        if self.input_tokens is not None and self.output_tokens is not None:
            return self.input_tokens + self.output_tokens
        return None
