from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from tesseractcli.memory.schema.memory_node_sources import MemoryNodeSource


class MemoryNodeSourceRepository:
    """Pure CRUD layer for MemoryNodeSource (the join table linking a
    MemoryNode back to the raw messages/nodes it was compressed from)."""

    def __init__(self, sessionConn: AsyncSession) -> None:
        self._s = sessionConn

    async def add_source(
        self, memory_node_id: str, source_type: str, source_id: str
    ) -> MemoryNodeSource:
        source = MemoryNodeSource(
            memory_node_id=memory_node_id,
            source_type=source_type,
            source_id=source_id,
        )

        self._s.add(source)
        await self._s.flush()

        return source

    async def add_sources(
        self, memory_node_id: str, sources: list[tuple[str, str]]
    ) -> list[MemoryNodeSource]:
        """Bulk variant — a node is usually compressed from several sources at once.
        `sources` is a list of (source_type, source_id) pairs."""
        rows = [
            MemoryNodeSource(
                memory_node_id=memory_node_id,
                source_type=source_type,
                source_id=source_id,
            )
            for source_type, source_id in sources
        ]

        self._s.add_all(rows)
        await self._s.flush()

        return rows

    async def list_sources_for_node(
        self, memory_node_id: str
    ) -> list[MemoryNodeSource]:
        result = await self._s.execute(
            select(MemoryNodeSource).where(
                MemoryNodeSource.memory_node_id == memory_node_id
            )
        )
        return list(result.scalars().all())
