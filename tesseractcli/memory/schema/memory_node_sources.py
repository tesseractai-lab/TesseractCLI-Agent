from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import CheckConstraint, ForeignKey, Index, PrimaryKeyConstraint, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from tesseractcli.memory.schema.base import Base

if TYPE_CHECKING:
    from tesseractcli.memory.schema.memory_nodes import MemoryNode


class MemoryNodeSource(Base):
    __tablename__ = "memory_node_sources"

    memory_node_id: Mapped[str] = mapped_column(
        Text,
        ForeignKey("memory_nodes.id", ondelete="CASCADE"),
        nullable=False,
    )

    source_type: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    source_id: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    memory_node: Mapped[MemoryNode] = relationship(
        "MemoryNode",
        back_populates="sources",
    )

    __table_args__ = (
        PrimaryKeyConstraint(
            "memory_node_id", "source_type", "source_id", name="pk_memory_node_sources"
        ),
        Index("idx_node_src_source_lookup", "source_type", "source_id"),
        Index("idx_node_src_node_id", "memory_node_id"),
        CheckConstraint(
            "source_type IN ('message', 'memory_node')", name="ck_node_src_type_valid"
        ),
    )
