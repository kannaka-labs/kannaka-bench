"""kannaka-bench unit tests — plain python, synthetic fixtures, no downloads.
Run: python tests/test_bench.py
Covers: LongMemEval and LoCoMo loaders turn the dataset shapes into items in
history order with the right gold ids and evidence level; metrics judge hits
at session vs turn level; the recency floor behaves; the kannaka adapter
parses the binary's output (stubbed) and always runs off the swarm; a run
end-to-end with the recency adapter writes results + manifest and the report
renders a losses-and-wins table.
"""
import json
import os
import sys
import tempfile
from datetime import timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bench import metrics, report  # noqa: E402
from bench.adapters import kannaka as kad  # noqa: E402
from bench.adapters.base import MemoryItem  # noqa: E402
from bench.adapters.baselines import RecencyAdapter  # noqa: E402
from bench.datasets import locomo, longmemeval  # noqa: E402

LME = [{
    "question_id": "q1", "question_type": "single-session-user", "question": "What did I buy?",
    "answer": "a kayak", "question_date": "2023/05/20 (Sat) 10:00",
    "haystack_dates": ["2023/05/10 (Wed) 09:00", "2023/05/01 (Mon) 08:00"],
    "haystack_session_ids": ["s_late", "s_early"],
    "haystack_sessions": [
        [{"role": "user", "content": "I bought a kayak", "has_answer": True}, {"role": "assistant", "content": "Nice!"}],
        [{"role": "user", "content": "weather is fine"}, {"role": "assistant", "content": ""}],
    ],
    "answer_session_ids": ["s_late"],
}]

LOCO = [{
    "qa": [{"question": "When did Caroline go?", "answer": "7 May 2023", "evidence": ["D1:2"], "category": 2},
           {"question": "adversarial?", "answer": "n/a", "evidence": [], "category": 5}],
    "conversation": {
        "speaker_a": "Caroline", "speaker_b": "Melanie",
        "session_1_date_time": "1:56 pm on 8 May, 2023",
        "session_1": [{"speaker": "Caroline", "dia_id": "D1:1", "text": "Hi"},
                      {"speaker": "Melanie", "dia_id": "D1:2", "text": "I went to the group", "blip_caption": "a rainbow flag"}],
        "session_2_date_time": "9:10 am on 25 May, 2023",
        "session_2": [{"speaker": "Caroline", "dia_id": "D2:1", "text": "Ran a race"}],
    },
}]


def test_longmemeval_loader_orders_history_and_keeps_gold():
    qs = longmemeval.questions_from(LME)
    q = qs[0]
    assert [it.id for it in q.items] == ["s_early#0", "s_late#0", "s_late#1"], "history order by date; empty turns dropped"
    assert q.items[1].text == "user: I bought a kayak" and q.items[1].when.tzinfo == timezone.utc
    assert q.gold_ids == {"s_late"} and q.gold_level == "session"
    assert q.asked_at.hour == 10 and q.qtype == "single-session-user"
    # without answer_session_ids the has_answer flags decide
    lme2 = [dict(LME[0], answer_session_ids=[])]
    assert longmemeval.questions_from(lme2)[0].gold_ids == {"s_late"}


def test_longmemeval_limit_is_per_type():
    data = [dict(LME[0], question_id=f"q{i}", question_type=t) for i, t in enumerate(
        ["a", "a", "a", "b", "a", "b", "c"])]
    qs = longmemeval.questions_from(data, limit=2)
    assert [(q.id, q.qtype) for q in qs] == [("q0", "a"), ("q1", "a"), ("q3", "b"), ("q5", "b"), ("q6", "c")]
    assert len(longmemeval.questions_from(data, limit=None)) == 7


def test_locomo_loader_turn_level():
    convs = locomo.conversations_from(LOCO)
    cid, items, qs = convs[0]
    assert [it.id for it in items] == ["D1:1", "D1:2", "D2:1"]
    assert items[1].text == "Melanie: I went to the group [shared an image: a rainbow flag]"
    assert items[0].when.day == 8 and items[0].when.hour == 13 and items[2].when.day == 25
    assert qs[0].gold_ids == {"D1:2"} and qs[0].gold_level == "turn" and qs[0].qtype == "temporal"
    assert qs[1].qtype == "adversarial" and qs[1].gold_ids == set()


def test_metrics_levels():
    assert metrics.any_hit_at_k(["s_late#3", "x#1"], {"s_late"}, "session", 5) is True
    assert metrics.any_hit_at_k(["s_late#3"], {"s_late"}, "session", 0) is False
    assert metrics.recall_at_k(["D1:2", "D9:9"], {"D1:2", "D2:1"}, "turn", 5) == 0.5
    assert metrics.mrr(["a#1", "b#1", "gold#7"], {"gold"}, "session") == 1 / 3
    assert metrics.mrr([], {"gold"}, "session") == 0.0
    assert metrics.percentile([1, 2, 3, 4], 0.5) == 2.5 and metrics.percentile([], 0.9) == 0.0


def test_recency_is_the_tail():
    with tempfile.TemporaryDirectory() as d:
        r = RecencyAdapter()
        r.open(d)
        r.ingest([MemoryItem(id=f"s#{i}", text=f"t{i}") for i in range(10)])
        hits = r.recall("anything", 3)
        assert [h.id for h in hits] == ["s#9", "s#8", "s#7"]
        assert r.footprint_bytes() > 0


def test_kannaka_adapter_parses_and_stays_off_the_swarm():
    with tempfile.TemporaryDirectory() as d:
        a = kad.KannakaAdapter(bin_path="kannaka-stub")
        a.batch = "0"                       # the per-item path, on purpose
        a.open(d)
        assert a.env["KANNAKA_NATS_URL"] == "nats://127.0.0.1:1"
        assert a.env["KANNAKA_DATA_DIR"].startswith(d)
        calls = []

        def fake_run(args):
            calls.append(args)
            if args[0] == "remember":
                return "[nats] Warning: could not connect\n11111111-2222-3333-4444-555555555555\n"
            return ('HrmStore initialized with 1 memories\n'
                    '[{"id":"11111111-2222-3333-4444-555555555555","content":"user: I bought a kayak","similarity":0.9},'
                    '{"id":"deadbeef-0000-0000-0000-000000000000","content":"a dream","similarity":0.1}]\n')
        a._run = fake_run
        a.ingest([MemoryItem(id="s_late#0", text="user: I bought a kayak",
                             when=longmemeval.parse_date("2023/05/10 (Wed) 09:00"))])
        assert calls[0][:2] == ["remember", "user: I bought a kayak"] and "--observed" in calls[0] and "2023-05-10T09:00:00Z" in calls[0]
        hits = a.recall("what did I buy", 5)
        assert [h.id for h in hits] == ["s_late#0", "kannaka:deadbeef-0000-0000-0000-000000000000"]
        assert hits[0].score == 0.9


def test_kannaka_adapter_keeps_a_json_line_whole_across_u2028():
    """kannaka-memory #1045 was this parser, not the binary: a memory whose text carries
    U+2028 (LINE SEPARATOR; present in LongMemEval's ShareGPT turns) is emitted raw by
    serde_json, and Python's str.splitlines() splits the JSON array there, so every
    --top-k that reached that row parsed as ZERO hits. One store, k>=17, every time."""
    with tempfile.TemporaryDirectory() as d:
        a = kad.KannakaAdapter(bin_path="kannaka-stub")
        a.batch = "0"
        a.open(d)
        a.by_kid["11111111-2222-3333-4444-555555555555"] = "s_1#0"
        a.by_kid["22222222-2222-3333-4444-555555555555"] = "s_2#0"
        array = ('[{"id":"11111111-2222-3333-4444-555555555555","content":"grateful to learn and grow.  I still have","similarity":0.9},'
                 '{"id":"22222222-2222-3333-4444-555555555555","content":"next line  and NEL too","similarity":0.5}]')
        assert len(array.splitlines()) == 4          # the trap, stated
        a._run = lambda args: "HrmStore initialized with 2 memories\n" + array + "\n"
        hits = a.recall("what did I learn", 5)
        assert [h.id for h in hits] == ["s_1#0", "s_2#0"]
        # the batch path reads the same shape
        a._batch_run = lambda args, timeout: (0, array + "\n" + array + "\n", "")
        many = a.recall_many(["q1", "q2"], 5)
        assert [[h.id for h in hs] for hs in many] == [["s_1#0", "s_2#0"], ["s_1#0", "s_2#0"]]


def test_kannaka_minilm_adapter_sets_the_encoder():
    with tempfile.TemporaryDirectory() as d:
        a = kad.KannakaMinilmAdapter(bin_path="kannaka-stub")
        a.open(d)
        assert a.name == "kannaka_minilm"
        assert a.env["KANNAKA_ENCODER"] == "ollama" and a.env["KANNAKA_ENCODER_MODEL"] == "all-minilm"
        assert a.env["KANNAKA_ENCODER_DIM"] == "384" and a.env["KANNAKA_NATS_URL"] == "nats://127.0.0.1:1"
        plain = kad.KannakaAdapter(bin_path="kannaka-stub")
        plain.open(d)
        assert "KANNAKA_ENCODER" not in plain.env or plain.env["KANNAKA_ENCODER"] != "ollama"


def test_kannaka_adapter_batch_and_fallback():
    items = [MemoryItem(id="s#0", text="alpha"), MemoryItem(id="s#1", text="beta")]
    with tempfile.TemporaryDirectory() as d:
        a = kad.KannakaAdapter(bin_path="kannaka-stub")
        a.open(d)
        seen = []

        def batch_ok(args, timeout):
            seen.append(args[:2])
            if args[0] == "remember":
                ndjson = open(args[2], encoding="utf-8").read().splitlines()
                assert json.loads(ndjson[0]) == {"content": "alpha"}
                return 0, "aaaaaaaa-0000-0000-0000-000000000001\naaaaaaaa-0000-0000-0000-000000000002\n", "remember --batch: 2 stored, 0 failed\n"
            return 0, '[{"id":"aaaaaaaa-0000-0000-0000-000000000002","content":"beta","similarity":0.7}]\n[]\n', ""
        a._batch_run = batch_ok
        a.ingest(items)
        assert a.batch_ok is True and a.by_kid["aaaaaaaa-0000-0000-0000-000000000002"] == "s#1"
        got = a.recall_many(["b?", "zzz"], 3)
        assert [[h.id for h in hs] for hs in got] == [["s#1"], []]
        assert seen == [["remember", "--batch"], ["recall", "--batch"]]

        # an older binary: --batch unknown -> per-item path takes over
        b = kad.KannakaAdapter(bin_path="kannaka-stub")
        b.open(d)
        b._batch_run = lambda args, timeout: (2, "", "remember: unknown flag: --batch\n")
        per_item = []
        b._run = lambda args: (per_item.append(args[0]) or "bbbbbbbb-0000-0000-0000-000000000001\n")
        b.ingest(items[:1])
        assert b.batch_ok is False and per_item == ["remember"]
        assert b.by_kid["bbbbbbbb-0000-0000-0000-000000000001"] == "s#0"


def test_run_end_to_end_with_recency_and_report():
    from bench import run as runmod
    with tempfile.TemporaryDirectory() as d:
        qs = longmemeval.questions_from(LME)
        rows = []
        ad = RecencyAdapter()
        runmod.run_store(ad, os.path.join(d, "store"), qs[0].items, qs, 2, "session", rows, "fixture")
        assert len(rows) == 1 and rows[0]["any_hit_at_k"] is True, rows   # the tail holds s_late
        out = os.path.join(d, "run")
        os.makedirs(out)
        with open(os.path.join(out, "results.jsonl"), "w") as f:
            f.write(json.dumps(rows[0]) + "\n")
            f.write(json.dumps({**rows[0], "adapter": "broken", "error": "boom"}) + "\n")
        json.dump({"run_id": "t", "dataset": {"dataset": "fixture", "sha256": "abc"}, "k": 2, "stores": 1,
                   "host": "h", "commit": "c0ffee"}, open(os.path.join(out, "manifest.json"), "w"))
        text = report.render(out)
        assert "| recency" in text and "| broken | 0 |" in text, text
        assert "1.000" in text


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("all bench tests passed")
