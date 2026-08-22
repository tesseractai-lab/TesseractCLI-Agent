from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from tesseractcli.memory.schema.sessions import Sessions
from tesseractcli.models.memory.session_status import SessionStatus

_DEFAULT_LIST_LIMIT: int = 15  # matches earlier agreed range (~15-20)


class SessionRepository:
    """Pure CRUD layer for Sessions. No runtime/business-logic knowledge
    (e.g. "is this the active session") belongs here — that's the
    caller/service layer's job."""

    def __init__(self, sessionConn: AsyncSession) -> None:
        self._s = sessionConn

    # ---- CRUD

    async def create(self, workspace_path: str, session_name: str) -> Sessions:
        session = Sessions(
            session_id=str(uuid4()),
            workspace_path=workspace_path,
            name=session_name,
        )

        self._s.add(session)
        await self._s.flush()
        await self._s.refresh(session)

        return session

    async def get_session(self, session_id: str) -> Sessions | None:
        return await self._s.get(Sessions, session_id)

    async def list_sessions(
        self, workspace_path: str, limit: int = _DEFAULT_LIST_LIMIT
    ) -> list[Sessions]:
        result = await self._s.execute(
            select(Sessions)
            .where(
                Sessions.workspace_path == workspace_path,
                Sessions.status == SessionStatus.ACTIVE,
            )
            .order_by(Sessions.last_active_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def update_last_active(self, session_id: str) -> Sessions | None:
        session = await self._s.get(Sessions, session_id)

        if session is None:
            return None

        session.last_active_at = datetime.now(UTC)
        await self._s.flush()

        return session

    async def rename_session(self, session_id: str, name: str) -> Sessions | None:
        session = await self._s.get(Sessions, session_id)

        if session is None:
            return None

        session.name = name
        await self._s.flush()

        return session

    async def soft_delete(self, session_id: str) -> Sessions | None:
        session = await self._s.get(Sessions, session_id)

        if session is None:
            return None

        session.status = SessionStatus.DELETED
        await self._s.flush()

        return session
