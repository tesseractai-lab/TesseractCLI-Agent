from __future__ import annotations

from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from tesseractcli.memory.schema.messages import Messages

_DEFAULT_LIST_LIMIT: int = 200


class MessageRepository:
    """Pure CRUD layer for Messages. Compression/summarization decisions
    (what gets folded into a MemoryNode) belong to the compression
    service, not here."""

    def __init__(self, sessionConn: AsyncSession) -> None:
        self._s = sessionConn

    async def create(self, session_id: str, role: str, content: str) -> Messages:
        message = Messages(
            session_id=session_id,
            role=role,
            content=content,
        )

        self._s.add(message)
        await self._s.flush()
        await self._s.refresh(message)

        return message

    async def get_message(self, message_id: int) -> Messages | None:
        return await self._s.get(Messages, message_id)

    async def list_by_session(
        self,
        session_id: str,
        limit: int | None = _DEFAULT_LIST_LIMIT,
        offset: int = 0,
    ) -> list[Messages]:
        """Oldest-first — matches how a session's raw history is replayed on Resume."""
        query = (
            select(Messages)
            .where(Messages.session_id == session_id)
            .order_by(Messages.timestamp.asc())
            .offset(offset)
        )
        if limit is not None:
            query = query.limit(limit)

        result = await self._s.execute(query)
        return list(result.scalars().all())

    async def count_by_session(self, session_id: str) -> int:
        """Used by the compression service to check token/message thresholds."""
        result = await self._s.execute(
            select(func.count())
            .select_from(Messages)
            .where(Messages.session_id == session_id)
        )
        return result.scalar_one()
