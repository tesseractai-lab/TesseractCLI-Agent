from __future__ import annotations

from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from tesseractcli.memory.schema.memory_nodes import MemoryNode
from tesseractcli.memory.schema.messages import Messages
from tesseractcli.memory.schema.model_meta import ModelMeta
from tesseractcli.memory.schema.sessions import Sessions
from tesseractcli.memory.schema.tools_meta import ToolMeta
from tesseractcli.models.memory.analytics_data import (
    CompressionRatio,
    LevelCount,
    SessionMessageCount,
    SessionStatusCount,
    TokenUsage,
    TokenUsageByName,
    ToolStats,
    UsageCount,
)
from tesseractcli.models.memory.memory_node_status import MemoryNodeStatus


class UsageAnalytics:
    """Read-only aggregation/reporting queries spanning ModelMeta, Messages,
    Sessions, ToolMeta, and MemoryNode. Not a per-entity repository — no
    create/update/delete here, only reads. Scope is always one of:
    session / workspace / global, where the metric supports it."""

    def __init__(self, sessionConn: AsyncSession) -> None:
        self._s = sessionConn

    # ---- packs / models (counts)

    async def count_packs_by_session(self, session_id: str) -> list[UsageCount]:
        return await self._count_model_meta_by(
            ModelMeta.pack_name, session_id=session_id
        )

    async def count_packs_by_workspace(self, workspace_path: str) -> list[UsageCount]:
        return await self._count_model_meta_by(
            ModelMeta.pack_name, workspace_path=workspace_path
        )

    async def count_packs_global(self) -> list[UsageCount]:
        return await self._count_model_meta_by(ModelMeta.pack_name)

    async def count_models_by_session(self, session_id: str) -> list[UsageCount]:
        return await self._count_model_meta_by(
            ModelMeta.model_name, session_id=session_id
        )

    async def count_models_by_workspace(self, workspace_path: str) -> list[UsageCount]:
        return await self._count_model_meta_by(
            ModelMeta.model_name, workspace_path=workspace_path
        )

    async def count_models_global(self) -> list[UsageCount]:
        return await self._count_model_meta_by(ModelMeta.model_name)

    # ---- token & cost usage

    async def total_tokens_by_session(self, session_id: str) -> TokenUsage:
        return await self._sum_tokens(session_id=session_id)

    async def total_tokens_by_workspace(self, workspace_path: str) -> TokenUsage:
        return await self._sum_tokens(workspace_path=workspace_path)

    async def total_tokens_global(self) -> TokenUsage:
        return await self._sum_tokens()

    async def tokens_by_pack_global(self) -> list[TokenUsageByName]:
        return await self._sum_tokens_by(ModelMeta.pack_name)

    async def tokens_by_model_global(self) -> list[TokenUsageByName]:
        return await self._sum_tokens_by(ModelMeta.model_name)

    # ---- tool usage & reliability

    async def tool_stats_global(self) -> list[ToolStats]:
        return await self._tool_stats()

    async def tool_stats_by_session(self, session_id: str) -> list[ToolStats]:
        return await self._tool_stats(session_id=session_id)

    async def tool_stats_by_workspace(self, workspace_path: str) -> list[ToolStats]:
        return await self._tool_stats(workspace_path=workspace_path)

    # ---- session activity

    async def message_counts_by_workspace(
        self, workspace_path: str
    ) -> list[SessionMessageCount]:
        result = await self._s.execute(
            select(Messages.session_id, func.count().label("count"))
            .join(Sessions, Sessions.session_id == Messages.session_id)
            .where(Sessions.workspace_path == workspace_path)
            .group_by(Messages.session_id)
            .order_by(func.count().desc())
        )
        return [
            SessionMessageCount(session_id=row[0], message_count=row[1])
            for row in result.all()
        ]

    async def role_distribution_by_session(self, session_id: str) -> list[UsageCount]:
        return await self._count_messages_by(Messages.role, session_id=session_id)

    async def role_distribution_by_workspace(
        self, workspace_path: str
    ) -> list[UsageCount]:
        return await self._count_messages_by(
            Messages.role, workspace_path=workspace_path
        )

    async def role_distribution_global(self) -> list[UsageCount]:
        return await self._count_messages_by(Messages.role)

    async def session_status_counts_by_workspace(
        self, workspace_path: str
    ) -> list[SessionStatusCount]:
        result = await self._s.execute(
            select(Sessions.status, func.count().label("count"))
            .where(Sessions.workspace_path == workspace_path)
            .group_by(Sessions.status)
        )
        return [SessionStatusCount(status=row[0], total=row[1]) for row in result.all()]

    # ---- memory compression health

    async def node_counts_by_level(self, session_id: str) -> list[LevelCount]:
        result = await self._s.execute(
            select(MemoryNode.level, func.count().label("count"))
            .where(
                MemoryNode.session_id == session_id,
                MemoryNode.status == MemoryNodeStatus.ACTIVE,
            )
            .group_by(MemoryNode.level)
            .order_by(MemoryNode.level.asc())
        )
        return [LevelCount(level=row[0], total=row[1]) for row in result.all()]

    async def compression_ratio(self, session_id: str) -> CompressionRatio:
        raw_count_result = await self._s.execute(
            select(func.count())
            .select_from(Messages)
            .where(Messages.session_id == session_id)
        )
        raw_count = raw_count_result.scalar_one()

        node_count_result = await self._s.execute(
            select(func.count())
            .select_from(MemoryNode)
            .where(
                MemoryNode.session_id == session_id,
                MemoryNode.status == MemoryNodeStatus.ACTIVE,
            )
        )
        node_count = node_count_result.scalar_one()

        ratio = (raw_count / node_count) if node_count > 0 else None
        return CompressionRatio(
            raw_message_count=raw_count, active_node_count=node_count, ratio=ratio
        )

    # ---- shared query builders

    async def _count_model_meta_by(
        self,
        column,
        *,
        session_id: str | None = None,
        workspace_path: str | None = None,
    ) -> list[UsageCount]:
        query = select(column, func.count().label("count")).select_from(ModelMeta)
        query = self._join_model_meta_scope(query, session_id, workspace_path)
        query = query.group_by(column).order_by(func.count().desc())

        result = await self._s.execute(query)
        return [UsageCount(name=row[0], total=row[1]) for row in result.all()]

    async def _sum_tokens(
        self,
        *,
        session_id: str | None = None,
        workspace_path: str | None = None,
    ) -> TokenUsage:
        query = select(
            func.coalesce(func.sum(ModelMeta.input_tokens), 0),
            func.coalesce(func.sum(ModelMeta.output_tokens), 0),
        ).select_from(ModelMeta)
        query = self._join_model_meta_scope(query, session_id, workspace_path)

        result = await self._s.execute(query)
        input_tokens, output_tokens = result.one()
        return TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )

    async def _sum_tokens_by(self, column) -> list[TokenUsageByName]:
        result = await self._s.execute(
            select(
                column,
                func.coalesce(func.sum(ModelMeta.input_tokens), 0),
                func.coalesce(func.sum(ModelMeta.output_tokens), 0),
            )
            .group_by(column)
            .order_by(func.sum(ModelMeta.input_tokens).desc())
        )
        return [
            TokenUsageByName(
                name=row[0],
                input_tokens=row[1],
                output_tokens=row[2],
                total_tokens=row[1] + row[2],
            )
            for row in result.all()
        ]

    async def _tool_stats(
        self,
        *,
        session_id: str | None = None,
        workspace_path: str | None = None,
    ) -> list[ToolStats]:
        query = select(
            ToolMeta.name,
            func.count().label("total"),
            func.sum(func.cast(ToolMeta.success, type_=func.count().type)).label(
                "success"
            ),
        ).select_from(ToolMeta)

        if workspace_path is not None:
            query = (
                query.join(Messages, Messages.id == ToolMeta.message_id)
                .join(Sessions, Sessions.session_id == Messages.session_id)
                .where(Sessions.workspace_path == workspace_path)
            )
        elif session_id is not None:
            query = query.join(Messages, Messages.id == ToolMeta.message_id).where(
                Messages.session_id == session_id
            )

        query = query.group_by(ToolMeta.name).order_by(func.count().desc())

        result = await self._s.execute(query)
        stats = []
        for name, total, success in result.all():
            success = success or 0
            stats.append(
                ToolStats(
                    name=name,
                    total_calls=total,
                    success_count=success,
                    success_rate=(success / total) if total > 0 else 0.0,
                )
            )
        return stats

    async def _count_messages_by(
        self,
        column,
        *,
        session_id: str | None = None,
        workspace_path: str | None = None,
    ) -> list[UsageCount]:
        query = select(column, func.count().label("count")).select_from(Messages)

        if workspace_path is not None:
            query = query.join(
                Sessions, Sessions.session_id == Messages.session_id
            ).where(Sessions.workspace_path == workspace_path)
        elif session_id is not None:
            query = query.where(Messages.session_id == session_id)

        query = query.group_by(column).order_by(func.count().desc())

        result = await self._s.execute(query)
        return [UsageCount(name=row[0], total=row[1]) for row in result.all()]

    @staticmethod
    def _join_model_meta_scope(
        query, session_id: str | None, workspace_path: str | None
    ):
        if workspace_path is not None:
            return (
                query.join(Messages, Messages.id == ModelMeta.message_id)
                .join(Sessions, Sessions.session_id == Messages.session_id)
                .where(Sessions.workspace_path == workspace_path)
            )
        if session_id is not None:
            return query.join(Messages, Messages.id == ModelMeta.message_id).where(
                Messages.session_id == session_id
            )
        return query
