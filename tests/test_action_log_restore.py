"""ActionLog 跨 checkpoint 恢复的类型守恒。

背景：session.metadata 是跨持久化的无类型边界。正常路径由
AgentSession.to_dict/from_dict 用 ``__type__`` 标记还原，但旧 checkpoint
（3.2 前无标记）、第三方 SessionStore、外部代码都可能留下裸 dict。
读侧必须自证类型，否则第一次 ``.append`` 就 AttributeError，
且异常发生在工具执行后的收尾路径上，表现为「调完工具没输出就断」。
"""

from __future__ import annotations

import json

import pytest

from wuwei.agent.session import AgentSession
from wuwei.runtime.action_log import (
    ACTION_LOG_KEY,
    ACTION_LOG_TYPE,
    ActionLog,
    coerce_action_log,
    get_or_create_log,
)


class FakeSession:
    """只提供 metadata 的最小替身，等价于报告里的复现。"""

    def __init__(self, metadata: dict | None = None) -> None:
        self.metadata = metadata if metadata is not None else {}


def bare_snapshot(count: int = 1) -> dict:
    log = ActionLog()
    for step in range(1, count + 1):
        log.append(run_id="r1", agent="main", step=step, kind="llm", summary=f"step-{step}")
    return log.to_dict()


def tagged_snapshot(count: int = 1) -> dict:
    return {"__type__": ACTION_LOG_TYPE, "value": bare_snapshot(count)}


def append(session, step: int) -> None:
    get_or_create_log(session).append(
        run_id="r1", agent="main", step=step, kind="tool", summary=f"after-{step}"
    )


# -- 正常路径：官方 checkpoint 往返 -----------------------------------------


def test_official_roundtrip_keeps_type_and_history() -> None:
    session = AgentSession(session_id="s1")
    get_or_create_log(session).append(
        run_id="r1", agent="main", step=1, kind="llm", summary="before"
    )

    # checkpointer 会做一次 JSON 往返，模拟真实落盘/读回。
    raw = json.loads(json.dumps(session.to_dict(), ensure_ascii=False))
    assert raw["metadata"][ACTION_LOG_KEY]["__type__"] == ACTION_LOG_TYPE

    restored = AgentSession.from_dict(raw)
    assert isinstance(restored.metadata[ACTION_LOG_KEY], ActionLog)
    append(restored, 2)
    assert len(restored.metadata[ACTION_LOG_KEY]) == 2


def test_repeated_roundtrip_does_not_drift() -> None:
    session = AgentSession(session_id="s1")
    append(session, 1)
    for step in range(2, 6):
        session = AgentSession.from_dict(json.loads(json.dumps(session.to_dict())))
        append(session, step)
    assert len(session.metadata[ACTION_LOG_KEY]) == 5


# -- 协议外形态：读侧守卫 ---------------------------------------------------


def test_bare_dict_is_restored_without_crashing() -> None:
    """报告的复现：裸 to_dict() 直接塞进 metadata，恢复后第一次 append。"""
    session = FakeSession()
    get_or_create_log(session).append(
        run_id="r1", agent="main", step=1, kind="llm", summary="before"
    )
    session.metadata = {ACTION_LOG_KEY: session.metadata[ACTION_LOG_KEY].to_dict()}

    append(session, 2)

    log = session.metadata[ACTION_LOG_KEY]
    assert isinstance(log, ActionLog)
    assert len(log) == 2, "旧条目不能被静默丢掉"
    assert log.entries[0].summary == "before"


def test_tagged_dict_does_not_lose_history() -> None:
    """只判 isinstance(dict) 就 from_dict 的话，带标记形态会被当成空日志。"""
    session = FakeSession({ACTION_LOG_KEY: tagged_snapshot(3)})
    append(session, 4)

    log = session.metadata[ACTION_LOG_KEY]
    assert isinstance(log, ActionLog)
    assert len(log) == 4
    assert [e.summary for e in log.entries[:3]] == ["step-1", "step-2", "step-3"]


def test_from_dict_accepts_both_shapes() -> None:
    assert len(ActionLog.from_dict(bare_snapshot(2))) == 2
    assert len(ActionLog.from_dict(tagged_snapshot(2))) == 2


def test_legacy_checkpoint_without_type_marker() -> None:
    """3.2 之前的快照没有 __type__ 包装，恢复时必须一并收敛。"""
    saved = AgentSession(session_id="legacy").to_dict()
    saved["metadata"][ACTION_LOG_KEY] = bare_snapshot(2)

    restored = AgentSession.from_dict(saved)
    assert isinstance(restored.metadata[ACTION_LOG_KEY], ActionLog)
    append(restored, 3)
    assert len(restored.metadata[ACTION_LOG_KEY]) == 3


def test_to_dict_normalizes_bare_dict() -> None:
    """写侧也规范化，避免脏形态在快照之间一直传下去。"""
    session = AgentSession(session_id="s1", metadata={ACTION_LOG_KEY: bare_snapshot(1)})
    saved = session.to_dict()
    assert saved["metadata"][ACTION_LOG_KEY]["__type__"] == ACTION_LOG_TYPE
    assert len(saved["metadata"][ACTION_LOG_KEY]["value"]["entries"]) == 1


# -- 回填、创建与 fail loud --------------------------------------------------


def test_backfill_keeps_identity() -> None:
    session = FakeSession({ACTION_LOG_KEY: bare_snapshot(1)})
    first = get_or_create_log(session)
    second = get_or_create_log(session)

    assert first is second, "不回填会每次重建，子代理可能写到不同实例上"
    assert isinstance(session.metadata[ACTION_LOG_KEY], ActionLog)


def test_none_creates_empty_log() -> None:
    session = FakeSession()
    assert isinstance(get_or_create_log(session), ActionLog)
    assert isinstance(session.metadata[ACTION_LOG_KEY], ActionLog)


def test_existing_instance_is_returned_unchanged() -> None:
    original = ActionLog()
    session = FakeSession({ACTION_LOG_KEY: original})
    assert get_or_create_log(session) is original


@pytest.mark.parametrize("bad", ["oops", 42, [], object()])
def test_unknown_shape_fails_loud(bad) -> None:
    """宁可在入口报清楚，也不要等 append 时抛 AttributeError。"""
    session = FakeSession({ACTION_LOG_KEY: bad})
    with pytest.raises(TypeError, match="action_log"):
        get_or_create_log(session)


def test_malformed_snapshot_is_not_treated_as_empty_log() -> None:
    with pytest.raises(ValueError, match="entries"):
        coerce_action_log({"total": 7})


def test_tagged_snapshot_without_value_fails_loud() -> None:
    with pytest.raises(ValueError, match="value"):
        coerce_action_log({"__type__": ACTION_LOG_TYPE, "value": None})


def test_coerce_accepts_mapping_subclass() -> None:
    from collections import OrderedDict

    log = coerce_action_log(OrderedDict(bare_snapshot(1)))
    assert isinstance(log, ActionLog)
    assert len(log) == 1
