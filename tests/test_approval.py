"""Unit tests for `services/approval.py`."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from tesseractcli.services.approval import ApprovalService


@pytest.mark.asyncio
async def test_request_suspends_until_resolved():
    service = ApprovalService()
    results = []

    async def requester():
        approved = await service.request("write_file", {"path": "x.py"}, Path("/tmp"))
        results.append(approved)

    task = asyncio.ensure_future(requester())
    await asyncio.sleep(0)  # let requester() reach the await
    assert service.awaiting is True
    assert service.pending_tool == "write_file"

    service.resolve(True)
    await task

    assert results == [True]
    assert service.awaiting is False
    assert service.pending_tool is None


@pytest.mark.asyncio
async def test_resolve_false_is_returned_correctly():
    service = ApprovalService()

    async def requester():
        return await service.request("exec_tool", {"cmd": "rm -rf /"}, Path("/tmp"))

    task = asyncio.ensure_future(requester())
    await asyncio.sleep(0)
    service.resolve(False)
    assert await task is False


@pytest.mark.asyncio
async def test_overlapping_requests_raise():
    service = ApprovalService()

    async def hold_open():
        await service.request("a", {}, Path("/tmp"))

    task = asyncio.ensure_future(hold_open())
    await asyncio.sleep(0)

    with pytest.raises(RuntimeError):
        await service.request("b", {}, Path("/tmp"))

    service.resolve(True)
    await task


def test_resolve_with_nothing_pending_is_a_no_op():
    service = ApprovalService()
    service.resolve(True)  # must not raise
    assert service.awaiting is False
