from __future__ import annotations

from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from tesseractcli.memory.schema.memory_nodes import MemoryNode
from tesseractcli.models.memory.memory_node_status import MemoryNodeStatus
from tesseractcli.models.memory.memory_type import MemoryType


class MemoryNodeRepository:
    """Pure CRUD layer for MemoryNode. Deciding *when* to compress, and
    which raw messages/nodes feed a new node, is the compression
    service's job — this only stores/retrieves nodes."""

    def __init__(self, sessionConn: AsyncSession) -> None:
        self._s = sessionConn

    async def create(
        self,
        session_id: str,
        level: int,
        content: str,
        memory_type: MemoryType = MemoryType.SUMMARY,
        token_count: int | None = None,
    ) -> MemoryNode:
        node = MemoryNode(
            id=str(uuid4()),
            session_id=session_id,
            level=level,
            memory_type=memory_type,
            content=content,
            token_count=token_count,
        )

        self._s.add(node)
        await self._s.flush()
        await self._s.refresh(node)

        return node

    async def get_node(self, node_id: str) -> MemoryNode | None:
        return await self._s.get(MemoryNode, node_id)

    async def list_active_by_level(
        self, session_id: str, level: int
    ) -> list[MemoryNode]:
        result = await self._s.execute(
            select(MemoryNode)
            .where(
                MemoryNode.session_id == session_id,
                MemoryNode.level == level,
                MemoryNode.status == MemoryNodeStatus.ACTIVE,
            )
            .order_by(MemoryNode.created_at.asc())
        )
        return list(result.scalars().all())

    async def get_max_active_level(self, session_id: str) -> int:
        """Highest level with at least one ACTIVE node for this session, or 0 if none."""
        result = await self._s.execute(
            select(MemoryNode.level)
            .where(
                MemoryNode.session_id == session_id,
                MemoryNode.status == MemoryNodeStatus.ACTIVE,
            )
            .order_by(MemoryNode.level.desc())
            .limit(1)
        )
        level = result.scalars().first()
        return level if level is not None else 0

    async def set_status(
        self, node_id: str, status: MemoryNodeStatus
    ) -> MemoryNode | None:
        """Generic status transition (SUPERSEDED / INVALIDATED / DELETED)."""
        node = await self._s.get(MemoryNode, node_id)

        if node is None:
            return None

        node.status = status
        await self._s.flush()

        return node
