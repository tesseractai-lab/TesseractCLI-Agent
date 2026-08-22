from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from tesseractcli.memory.schema.model_meta import ModelMeta


class ModelMetaRepository:
    """Pure CRUD layer for ModelMeta (1:1 with a Messages row)."""

    def __init__(self, sessionConn: AsyncSession) -> None:
        self._s = sessionConn

    async def create(
        self,
        message_id: int,
        model_name: str | None = None,
        pack_name: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> ModelMeta:
        meta = ModelMeta(
            message_id=message_id,
            model_name=model_name,
            pack_name=pack_name,
            temperature=temperature,
            max_tokens=max_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        self._s.add(meta)
        await self._s.flush()
        await self._s.refresh(meta)

        return meta

    async def get_by_message_id(self, message_id: int) -> ModelMeta | None:
        return await self._s.get(ModelMeta, message_id)

    async def update_usage(
        self, message_id: int, input_tokens: int, output_tokens: int
    ) -> ModelMeta | None:
        """Token counts often arrive after the row is created (once the LLM
        response finishes), so this is a separate step rather than folded into create()."""
        meta = await self._s.get(ModelMeta, message_id)

        if meta is None:
            return None

        meta.input_tokens = input_tokens
        meta.output_tokens = output_tokens
        await self._s.flush()

        return meta
