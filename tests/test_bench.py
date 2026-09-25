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


def test_shards_cover_every_store_once():
    from bench import run as runmod
    stores = [(f"s{i}", [], [], "session") for i in range(11)]
    parts = [runmod.shard_stores(stores, f"{i}/3") for i in range(3)]
    ids = [st[0] for p in parts for st in p]
    assert sorted(ids) == sorted(st[0] for st in stores) and len(ids) == len(set(ids)), ids
    assert [st[0] for st in parts[1]] == ["s1", "s4", "s7", "s10"]
    assert runmod.shard_stores(stores, None) == stores
    for bad in ("3/3", "-1/2", "0/0"):
        try:
            runmod.parse_shard(bad)
        except ValueError:
            continue
        raise AssertionError(f"--shard {bad} accepted")


def test_merge_joins_shards_and_refuses_duplicates():
    from bench import merge
    with tempfile.TemporaryDirectory() as d:
        base = {"adapters": ["recency"], "k": 15, "session_cap": 0, "limit": None, "commit": "c",
                "stores_total": 2, "dataset": {"sha256": "x"}, "wall_s": 1.0, "started_at": "a", "finished_at": "b"}
        dirs = []
        for i in range(2):
            sd = os.path.join(d, f"sh{i}")
            os.makedirs(sd)
            json.dump(dict(base, shard=f"{i}/2", stores=1, wall_s=10.0 * (i + 1)), open(os.path.join(sd, "manifest.json"), "w"))
            with open(os.path.join(sd, "results.jsonl"), "w") as f:
                f.write(json.dumps({"adapter": "recency", "question_id": f"q{i}"}) + "\n")
            dirs.append(sd)
        out = merge.merge(os.path.join(d, "all"), dirs)
        assert out["rows"] == 2 and out["stores"] == 2, out
        m = json.load(open(os.path.join(d, "all", "manifest.json")))
        assert m["wall_s"] == 20.0 and len(m["shards"]) == 2 and m["run_id"] == "all"
        for bad in ([dirs[0], dirs[0]], [dirs[0], "overlap"]):
            if bad[1] == "overlap":       # a different shard that re-ran q0
                bad[1] = os.path.join(d, "overlap")
                os.makedirs(bad[1])
                json.dump(dict(base, shard="1/2", stores=1), open(os.path.join(bad[1], "manifest.json"), "w"))
                with open(os.path.join(dirs[0], "results.jsonl")) as src, open(os.path.join(bad[1], "results.jsonl"), "w") as dst:
                    dst.write(src.read())
            try:
                merge.merge(os.path.join(d, "bad"), bad)
            except ValueError:
                pass
            else:
                raise AssertionError(f"merge accepted {bad}")


def test_stats_intervals_and_pairing():
    from bench import stats
    lo, hi = stats.wilson(5, 5)
    assert abs(lo - 0.5655) < 1e-3 and hi == 1.0, (lo, hi)      # 5/5 is NOT certainty
    lo, hi = stats.wilson(50, 100)
    assert abs(lo - 0.4038) < 1e-3 and abs(hi - 0.5962) < 1e-3, (lo, hi)
    rows = []
    for i in range(40):
        t = "a" if i < 20 else "b"
        rows.append({"adapter": "X", "question_id": f"q{i}", "qtype": t, "gold": ["g"], "any_hit_at_k": True,
                     "recall_at_k": 1.0, "evidence_coverage_at_k": 1.0, "mrr": 1.0})
        rows.append({"adapter": "Y", "question_id": f"q{i}", "qtype": t, "gold": ["g"], "any_hit_at_k": i % 2 == 0,
                     "recall_at_k": 0.5, "evidence_coverage_at_k": None, "mrr": 0.5})
    rows.append({"adapter": "Y", "question_id": "err", "qtype": "a", "error": "boom"})
    rows.append({"adapter": "Y", "question_id": "nogold", "qtype": "a", "gold": [], "any_hit_at_k": False})
    res = stats.analyse(rows, ("X", "Y"), n_boot=2000)
    assert res["overall"]["Y"]["n"] == 40, "error and gold-less rows are unscored"
    assert res["overall"]["Y"]["evidence_coverage_at_k"] is None
    p = res["paired"]["any_hit_at_k"]
    assert p["n"] == 40 and p["a_better"] == 20 and p["b_better"] == 0 and p["ties"] == 20, p
    assert p["lo"] > 0, "X beats Y on every question it differs: the interval must exclude 0"
    assert res["paired"]["evidence_coverage_at_k"] is None, "evid pairs only where both have it"
    sub = stats.analyse(rows, ("X", "Y"), ids={"q0", "q1"}, n_boot=500)
    assert sub["overall"]["X"]["n"] == 2
    assert "hit@k" in stats.render(res, ("X", "Y"))
    # --at-k rescores from the ranked hits: gold at rank 3 is a hit at 3, a miss at 2
    r = {"adapter": "Z", "question_id": "t", "qtype": "a", "gold": ["D1:3"], "gold_level": "turn", "k": 15,
         "hits": ["D1:1", "D1:2", "D1:3"], "any_hit_at_k": True, "recall_at_k": 1.0, "mrr": 1 / 3,
         "evidence_coverage_at_k": 1.0}
    assert stats.at_k([r], 3)[0]["any_hit_at_k"] is True and stats.at_k([r], 2)[0]["any_hit_at_k"] is False
    assert stats.at_k([r], 2)[0]["recall_at_k"] == 0.0 and stats.at_k([r], 2)[0]["mrr"] == 0.0
    assert stats.at_k([r], 2)[0]["evidence_coverage_at_k"] is None


def test_longmemeval_load_by_ids_and_stream():
    import importlib.util
    with tempfile.TemporaryDirectory() as d:
        data = [dict(LME[0], question_id=f"q{i}", question_type=("t1" if i % 2 else "t2")) for i in range(6)]
        path = os.path.join(d, "longmemeval_s.json")
        json.dump(data, open(path, "w"))
        old_dir, old_bytes = longmemeval.DATA_DIR, longmemeval.STREAM_BYTES
        longmemeval.DATA_DIR = d
        try:
            qs, meta = longmemeval.load("longmemeval_s", limit=1)
            assert [q.id for q in qs] == ["q0", "q1"] and not meta["streamed"]
            qs, meta = longmemeval.load("longmemeval_s", limit=1, question_ids={"q3", "q4"})
            assert [q.id for q in qs] == ["q3", "q4"] and not meta["limit_is_per_type"]
            if importlib.util.find_spec("ijson"):
                longmemeval.STREAM_BYTES = 1          # force the streaming path
                qs2, meta2 = longmemeval.load("longmemeval_s", limit=2)
                qs1, _ = (longmemeval.questions_from(data, 2), None)
                assert meta2["streamed"] and [q.id for q in qs2] == [q.id for q in qs1], "stream == stratified"
                assert [it.id for it in qs2[0].items] == [it.id for it in qs1[0].items]
                qs3, _ = longmemeval.load("longmemeval_s", question_ids={"q5"})
                assert [q.id for q in qs3] == ["q5"] and meta2["questions_total"] == 6
        finally:
            longmemeval.DATA_DIR, longmemeval.STREAM_BYTES = old_dir, old_bytes


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("all bench tests passed")
