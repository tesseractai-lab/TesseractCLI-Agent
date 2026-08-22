from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from tesseractcli.memory.schema.tools_meta import ToolMeta


class ToolMetaRepository:
    """Pure CRUD layer for ToolMeta (one row per tool call within a message)."""

    def __init__(self, sessionConn: AsyncSession) -> None:
        self._s = sessionConn

    async def create(
        self,
        tool_call_id: str,
        message_id: int,
        name: str,
        args: str | None = None,
        success: bool = False,
    ) -> ToolMeta:
        meta = ToolMeta(
            tool_call_id=tool_call_id,
            message_id=message_id,
            name=name,
            args=args,
            success=success,
        )

        self._s.add(meta)
        await self._s.flush()
        await self._s.refresh(meta)

        return meta

    async def get_by_id(self, tool_call_id: str) -> ToolMeta | None:
        return await self._s.get(ToolMeta, tool_call_id)

    async def list_by_message(self, message_id: int) -> list[ToolMeta]:
        result = await self._s.execute(
            select(ToolMeta).where(ToolMeta.message_id == message_id)
        )
        return list(result.scalars().all())

    async def set_success(self, tool_call_id: str, success: bool) -> ToolMeta | None:
        """Args are known at call-time; success is only known after execution finishes."""
        meta = await self._s.get(ToolMeta, tool_call_id)

        if meta is None:
            return None

        meta.success = success
        await self._s.flush()

        return meta
