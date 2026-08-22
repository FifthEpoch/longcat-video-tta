#!/usr/bin/env python3
"""Offline idea 7 (trust region) + idea 9 (hybrid router). No GPU.

Trust: on existing cand logs, reject a pick whose motion < 0.8× cand0.
This is not a counterfactual 30 s tail — per-chunk only.

Router: per-video pick among already-generated methods using only
prefix_motion. Oracle = best tail among {notta, rolling, seed, appear, live}.

    python3 -u wan_experiment/scripts/resim_v2v_trust_hybrid.py
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


TRUST = 0.8
STILL = 0.012
HOT = 0.03


def _usable(rec: dict) -> bool:
    return bool(
        rec.get("ok")
        and not rec.get("skipped")
        and rec.get("tail_motion") is not None
    )


def _key(rec: dict, fallback: str = "") -> str:
    k = rec.get("file_name") or rec.get("stem") or fallback
    return Path(str(k)).name if k else fallback


def _load(d: Path) -> dict[str, dict]:
    out = {}
    if not d.is_dir():
        return out
    for p in sorted(d.glob("*.json")):
        if p.name in {"summary.json", "joined.json"} or "vbench" in p.name:
            continue
        try:
            rec = json.loads(p.read_text())
        except Exception:
            continue
        if not _usable(rec):
            continue
        out[_key(rec, p.stem)] = rec
    return out


def _motion(c: dict) -> float:
    if c.get("temporal_motion") is not None:
        try:
            return float(c["temporal_motion"])
        except (TypeError, ValueError):
            pass
    sig = (c.get("free") or c.get("signals") or {})
    try:
        return float(sig.get("temporal_motion"))
    except (TypeError, ValueError):
        return float("nan")


def _prefix(rec: dict) -> float:
    v = rec.get("prefix_motion")
    if v is not None:
        try:
            return float(v)
        except (TypeError, ValueError):
            pass
    chunks = rec.get("chunks") or []
    if chunks:
        try:
            return float(chunks[0].get("prefix_motion"))
        except (TypeError, ValueError):
            pass
    return float("nan")


def _cand0(cands):
    for c in cands:
        if int(c.get("cand", -1)) == 0:
            return c
    return cands[0] if cands else None


def trust_resim(method_dir: Path) -> dict:
    rows = _load(method_dir)
    n_chunk = n_change = n_reject = n_keep = 0
    for rec in rows.values():
        for ch in rec.get("chunks") or []:
            cands = ch.get("candidates") or ch.get("cands") or []
            if len(cands) < 2:
                continue
            n_chunk += 1
            orig = int(ch.get("chosen_cand", 0) or 0)
            c0 = _cand0(cands)
            m0 = _motion(c0) if c0 else float("nan")
            chosen = None
            for c in cands:
                if int(c.get("cand", -1)) == orig:
                    chosen = c
                    break
            cm = _motion(chosen) if chosen else float("nan")
            if m0 != m0 or cm != cm:
                continue
            if cm < TRUST * m0:
                n_reject += 1
                if orig != 0:
                    n_change += 1
            else:
                n_keep += 1
    return {
        "dir": str(method_dir),
        "n_videos": len(rows),
        "n_search_chunks": n_chunk,
        "would_reject": n_reject,
        "would_change_from_orig": n_change,
        "keep": n_keep,
    }


def _tail(rec):
    try:
        return float(rec["tail_motion"])
    except (TypeError, ValueError, KeyError):
        return float("nan")


def router_resim(arms: dict[str, dict[str, dict]]) -> None:
    keys = None
    for mapping in arms.values():
        s = set(mapping)
        keys = s if keys is None else keys & s
    keys = sorted(keys or [])
    print(f"\n## Idea 9 hybrid router (offline, n={len(keys)})\n")
    if len(keys) < 4:
        print("not enough paired videos across arms.")
        return
    oracle_tails = []
    rule_tails = []
    notta_tails = []
    rolling_tails = []
    picks = {"notta": 0, "rolling_notta": 0}
    oracle_pick = {k: 0 for k in arms}
    for k in keys:
        scores = {name: _tail(mapping[k]) for name, mapping in arms.items()}
        scores = {n: v for n, v in scores.items() if v == v}
        if "notta" not in scores:
            continue
        best = max(scores, key=scores.get)
        oracle_pick[best] = oracle_pick.get(best, 0) + 1
        oracle_tails.append(scores[best])
        notta_tails.append(scores["notta"])
        if "rolling_notta" in scores:
            rolling_tails.append(scores["rolling_notta"])
        pm = _prefix(arms.get("notta", {}).get(k) or next(iter(arms.values()))[k])
        if pm == pm and pm >= STILL and "rolling_notta" in scores:
            rule, rule_t = "rolling_notta", scores["rolling_notta"]
        else:
            rule, rule_t = "notta", scores["notta"]
        picks[rule] = picks.get(rule, 0) + 1
        rule_tails.append(rule_t)
    def med(xs):
        xs = [x for x in xs if x == x]
        return statistics.median(xs) if xs else None
    print("| Policy | tail median | vs notta |")
    print("|---|---:|---:|")
    nm = med(notta_tails)
    print(f"| notta | {nm:.4f} | — |")
    if rolling_tails:
        rm = med(rolling_tails)
        print(f"| always rolling | {rm:.4f} | {(rm / nm - 1) * 100:+.1f}% |")
    ru = med(rule_tails)
    oc = med(oracle_tails)
    print(f"| prefix rule (still→notta, live→rolling) | {ru:.4f} | {(ru / nm - 1) * 100:+.1f}% |")
    print(f"| oracle best-of-arms | {oc:.4f} | {(oc / nm - 1) * 100:+.1f}% |")
    print("")
    print("Oracle arm counts:", {k: v for k, v in oracle_pick.items() if v})
    print("Rule picks:", picks)
    print(
        "This is **not** a new generate. Oracle uses GT-of-tails we already "
        "have. The prefix rule is the only deployable row."
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    args = ap.parse_args()
    r = args.root
    print("# Idea 7 trust-region resim (per-chunk, not a 30 s tail)\n")
    print("| Method dir | videos | search chunks | reject | change orig | keep |")
    print("|---|---:|---:|---:|---:|---:|")
    dirs = [
        r / "wan_experiment/results/v2v_panda_bakeoff_8v/seed_bon_h30s_shard0",
        r / "wan_experiment/results/v2v_panda_confirm_32v/seed_bon_h30s_shard0",
        r / "wan_experiment/results/v2v_panda_ideas_8v/appear_bon_h30s_shard0",
        r / "wan_experiment/results/v2v_panda_forward_32v/appear_bon_h30s_shard0",
        r / "wan_experiment/results/v2v_panda_live_32v/live_bon_h30s_shard0",
    ]
    for d in dirs:
        if not d.is_dir():
            print(f"| `{d}` | missing | — | — | — | — |")
            continue
        st = trust_resim(d)
        print(
            f"| `{d.name}` ({Path(st['dir']).parent.name}) | {st['n_videos']} | "
            f"{st['n_search_chunks']} | {st['would_reject']} | "
            f"{st['would_change_from_orig']} | {st['keep']} |"
        )
    print(
        "\nIf `change orig` is ~0, the original picker already stayed in "
        "the trust region. If reject is high on seed/appear-32, trust would "
        "have blocked the identity damper — still not a 30 s proof.\n"
    )

    arms = {}
    pairs = [
        ("notta", r / "wan_experiment/results/v2v_panda_confirm_32v/notta_h30s_shard0"),
        ("rolling_notta", r / "wan_experiment/results/v2v_panda_forward_32v/rolling_notta_h30s_shard0"),
        ("seed_bon", r / "wan_experiment/results/v2v_panda_confirm_32v/seed_bon_h30s_shard0"),
        ("appear_bon", r / "wan_experiment/results/v2v_panda_forward_32v/appear_bon_h30s_shard0"),
        ("live_bon", r / "wan_experiment/results/v2v_panda_live_32v/live_bon_h30s_shard0"),
    ]
    for name, d in pairs:
        rows = _load(d)
        if rows:
            arms[name] = rows
        else:
            print(f"router skip missing {d}")
    router_resim(arms)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
