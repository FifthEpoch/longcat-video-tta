#!/usr/bin/env python3
"""SF-family mechanism tables + next-action cells. Login CPU.

    python3 -u wan_experiment/scripts/analyze_v2v_sf_family_dissect.py \
      --family-dir wan_experiment/results/v2v_panda_sf_family_32v \
      --notta-dir wan_experiment/results/v2v_panda_confirm_32v \
      --rolling-dir wan_experiment/results/v2v_panda_forward_32v

Playbook: paper_tables/2026-08-24_wan_v2v_sf_family_dissect.md
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


FAMILY = (
    "sf_rewind", "sf_sick_search", "sf_pseudo", "sf_always_search",
    "rf_always_search", "sf_sink",
)
NAMED = ("panda_0004.mp4", "panda_0027.mp4", "0004", "0027")
H1_FLICKER = 0.972
EXACT_EPS = 1e-8


def _median(xs):
    xs = [float(x) for x in xs if x is not None and x == x]
    if not xs:
        return None
    return statistics.median(xs)


def _usable(rec: dict) -> bool:
    return bool(
        rec.get("ok")
        and not rec.get("skipped")
        and rec.get("tail_motion") is not None
    )


def _load_dir(d: Path) -> dict[str, dict]:
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
        key = str(rec.get("file_name") or rec.get("stem") or p.stem)
        out[key] = rec
    return out


def _load_vbench(method_dir: Path) -> dict | None:
    for name in ("vbench_full/joined.json", "vbench_full/summary.json"):
        p = method_dir / name
        if not p.is_file():
            continue
        data = json.loads(p.read_text())
        pop = data.get("population") or {}
        out = {}
        for dim in (
            "subject_consistency", "imaging_quality", "dynamic_degree",
            "temporal_flickering",
        ):
            cell = pop.get(dim)
            if isinstance(cell, dict) and cell.get("median") is not None:
                out[dim] = float(cell["median"])
        if out:
            return out
    return None


def _chunks(rec: dict) -> list[dict]:
    return list(rec.get("chunks") or [])


def _tail(rec: dict):
    t = rec.get("tail_motion")
    return float(t) if t is not None and t == t else None


def _exact(a, b) -> bool:
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) < EXACT_EPS


def _wl(keys, method_rows, base_rows):
    w = l = t = 0
    for k in keys:
        a, b = _tail(method_rows[k]), _tail(base_rows[k])
        if a is None or b is None:
            continue
        if _exact(a, b):
            t += 1
        elif a > b:
            w += 1
        else:
            l += 1
    return w, l, t


def _rewind_cov(rec: dict) -> dict:
    n_trig = n_acc = n_rej = 0
    later_freeze = False
    accepted_ci = []
    for ch in _chunks(rec):
        rw = ch.get("rewind") or {}
        reason = ch.get("gate_reason") or ""
        if rw or reason in ("sf_rewind_accept", "sf_rewind_reject"):
            n_trig += 1
            if rw.get("accepted") or reason == "sf_rewind_accept":
                n_acc += 1
                accepted_ci.append(int(ch.get("chunk", -1)))
            else:
                n_rej += 1
        if accepted_ci and ch.get("last_sick") and int(ch.get("chunk", -1)) > max(accepted_ci):
            later_freeze = True
    return {
        "trig": n_trig, "acc": n_acc, "rej": n_rej,
        "later_freeze": later_freeze,
    }


def _sick_cov(rec: dict) -> dict:
    n_sick = n_search = n_trust = 0
    for ch in _chunks(rec):
        if ch.get("last_sick"):
            n_sick += 1
        reason = ch.get("gate_reason") or ""
        if reason == "sick_search" or int(ch.get("search_k") or 1) > 1:
            n_search += 1
        if reason == "look_trust_reject":
            n_trust += 1
    return {"sick": n_sick, "search": n_search, "trust_rej": n_trust}


def _pseudo_cov(rec: dict) -> dict:
    fire = False
    rows = None
    for ch in _chunks(rec):
        if ch.get("pseudo_fire"):
            fire = True
        if ch.get("pseudo_rows"):
            rows = ch.get("pseudo_rows")
        if (ch.get("gate_reason") or "") == "pseudo_fire":
            fire = True
    return {"fire": fire, "rows": rows}


def _cell(
    method: str,
    n: int,
    exact_sf: int,
    fire_n: int,
    act_n: int,
    tail_m,
    tail_sf,
    w_sf: int,
    l_sf: int,
    iq_ok: bool,
    subj_ok: bool,
    flicker,
    dyn,
    always_on: bool,
) -> str:
    if n <= 0 or tail_m is None or tail_sf is None:
        return "PENDING"
    twitch = (
        dyn is not None and dyn >= 0.5
        and flicker is not None and flicker <= H1_FLICKER + 0.002
        and not subj_ok
    )
    if twitch:
        return "G"
    if always_on and tail_m <= tail_sf and l_sf >= w_sf:
        return "H"
    if not always_on and (exact_sf >= 16 or fire_n < 8):
        return "A"
    if not always_on and fire_n >= 8 and act_n <= 1 and tail_m <= tail_sf:
        return "B"
    motion = tail_m > tail_sf
    if motion and not (iq_ok and subj_ok):
        return "D"
    if motion and (w_sf + l_sf) >= 8 and abs(w_sf - l_sf) <= 4:
        return "E"
    if motion and iq_ok and subj_ok and (always_on or fire_n >= 8) and exact_sf < 16:
        return "F"
    if fire_n >= 8 and act_n >= 1 and tail_m <= tail_sf:
        return "C"
    if motion:
        return "E"
    return "C"


NEXT = {
    "A": "Dead gate. New sensor (prefix-relative / 2-chunk trend / later hold-out). Not a bigger k.",
    "B": "Sensor lives, actuator idle. Second resample, two-chunk rewind, or drop trust-0.8 veto.",
    "C": "Acted and tail did not rise. Pick-score anti-aligned. Add a veto; do not climb motion alone.",
    "D": "Tail up, quality tax. Keep the lever; add identity veto. HOLD no-scale.",
    "E": "Mixed videos. Cluster wounds (0004 vs 0027). Next widget is for that cluster.",
    "F": "Letter win. HOLD N=32. Fill invention sentence from coverage/conditional. Do not scale tonight.",
    "G": "H1 twitch (Dyn up + flicker ~0.972 + subject down). NO. Not a motion method.",
    "H": "Always-on harm. Drop on SF. Lever is host-specific.",
    "PENDING": "Wait for 32/32 + VBench 7/7.",
}


def _fmt(x, nd=4):
    if x is None:
        return "—"
    return f"{x:.{nd}f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family-dir", type=Path, required=True)
    ap.add_argument("--also-dir", type=Path, action="append", default=[])
    ap.add_argument("--notta-dir", type=Path, required=True)
    ap.add_argument("--rolling-dir", type=Path, required=True)
    args = ap.parse_args()
    search_roots = [args.family_dir, *args.also_dir]

    notta = _load_dir(args.notta_dir / "notta_h30s_shard0")
    rolling = _load_dir(args.rolling_dir / "rolling_notta_h30s_shard0")
    if not notta:
        raise SystemExit(f"no SF notta sidecars under {args.notta_dir}")

    vb_sf = _load_vbench(args.notta_dir / "notta_h30s_shard0")
    vb_rf = _load_vbench(args.rolling_dir / "rolling_notta_h30s_shard0")

    print("# SF-family dissection")
    print()
    print("Playbook: paper_tables/2026-08-24_wan_v2v_sf_family_dissect.md")
    print(f"SF notta n={len(notta)}  RF rolling n={len(rolling)}")
    print()

    win_sets = {}
    rows_out = []

    def _find_method(method: str):
        for root in search_roots:
            d = root / f"{method}_h30s_shard0"
            mapping = _load_dir(d)
            if mapping:
                return mapping, d
        return {}, None

    for method in FAMILY:
        mapping, method_dir = _find_method(method)
        if not mapping:
            continue
        keys = sorted(k for k in notta if k in mapping)
        tails_m = [_tail(mapping[k]) for k in keys]
        tails_s = [_tail(notta[k]) for k in keys]
        tail_m = _median(tails_m)
        tail_s = _median(tails_s)
        keys_rf = [k for k in keys if k in rolling]
        w_sf, l_sf, t_sf = _wl(keys, mapping, notta)
        w_rf, l_rf, t_rf = _wl(keys_rf, mapping, rolling) if keys_rf else (0, 0, 0)
        exact_sf = t_sf

        fire_n = act_n = later = 0
        always_on = method in (
            "sf_sink", "sf_always_search", "rf_always_search",
        )
        for k in keys:
            rec = mapping[k]
            if method == "sf_rewind":
                c = _rewind_cov(rec)
                if c["trig"]:
                    fire_n += 1
                if c["acc"]:
                    act_n += 1
                if c["later_freeze"]:
                    later += 1
            elif method == "sf_sick_search":
                c = _sick_cov(rec)
                if c["sick"] or c["search"]:
                    fire_n += 1
                if c["search"]:
                    act_n += 1
            elif method == "sf_pseudo":
                c = _pseudo_cov(rec)
                if c["fire"]:
                    fire_n += 1
                    act_n += 1
            else:
                fire_n += 1
                act_n += 1

        cond_fire, cond_quiet = [], []
        for k in keys:
            rec = mapping[k]
            fired = False
            if method == "sf_rewind":
                fired = _rewind_cov(rec)["trig"] > 0
            elif method == "sf_sick_search":
                c = _sick_cov(rec)
                fired = (c["sick"] or c["search"]) > 0
            elif method == "sf_pseudo":
                fired = _pseudo_cov(rec)["fire"]
            else:
                fired = True
            dt = None
            a, b = _tail(rec), _tail(notta[k])
            if a is not None and b is not None and b > 0:
                dt = (a - b) / b
            (cond_fire if fired else cond_quiet).append(dt)

        vb = _load_vbench(method_dir) if method_dir is not None else None
        iq_ok = subj_ok = True
        flicker = dyn = None
        if vb and vb_sf:
            if "imaging_quality" in vb and "imaging_quality" in vb_sf:
                iq_ok = vb["imaging_quality"] >= vb_sf["imaging_quality"] - 1.0
            if "subject_consistency" in vb and "subject_consistency" in vb_sf:
                subj_ok = vb["subject_consistency"] >= vb_sf["subject_consistency"] - 0.02
            flicker = vb.get("temporal_flickering")
            dyn = vb.get("dynamic_degree")

        cell = _cell(
            method, len(keys), exact_sf, fire_n, act_n,
            tail_m, tail_s, w_sf, l_sf, iq_ok, subj_ok,
            flicker, dyn, always_on,
        )
        rel = None
        if tail_m is not None and tail_s is not None and tail_s > 0:
            rel = (tail_m - tail_s) / tail_s

        wins = {
            k for k in keys
            if _tail(mapping[k]) is not None
            and _tail(notta[k]) is not None
            and _tail(notta[k]) > 0
            and (_tail(mapping[k]) - _tail(notta[k])) / _tail(notta[k]) > 0.10
        }
        win_sets[method] = wins

        deltas = []
        for k in keys:
            a, b = _tail(mapping[k]), _tail(notta[k])
            if a is None or b is None:
                continue
            deltas.append((a - b, k, a, b))
        deltas.sort()
        worst = deltas[:3]
        best = list(reversed(deltas[-3:]))

        rows_out.append({
            "method": method, "n": len(keys), "tail": tail_m, "rel": rel,
            "w": w_sf, "l": l_sf, "t": t_sf, "exact": exact_sf,
            "fire": fire_n, "act": act_n, "later": later,
            "w_rf": w_rf, "l_rf": l_rf, "t_rf": t_rf,
            "cell": cell, "vb": vb, "iq_ok": iq_ok, "subj_ok": subj_ok,
            "cond_fire": _median(cond_fire), "cond_quiet": _median(cond_quiet),
            "worst": worst, "best": best, "mapping": mapping, "keys": keys,
        })

    print("## Headline + coverage")
    print()
    print(
        "| Method | n | tail | vs SF | W/L/tie vs SF | exact-SF | "
        "fire | act | vs RF W/L/tie | cell | next |"
    )
    print("|---|---:|---:|---:|---|---:|---:|---:|---|---|---|")
    print(
        f"| notta (SF) | {len(notta)} | {_fmt(_median([_tail(r) for r in notta.values()]))} "
        f"| — | — | — | — | — | — | baseline | — |"
    )
    if rolling:
        tr = _median([_tail(r) for r in rolling.values()])
        ts = _median([_tail(r) for r in notta.values()])
        rel_r = (tr - ts) / ts if tr and ts and ts > 0 else None
        wr, lr, t_ = _wl(sorted(k for k in notta if k in rolling), rolling, notta)
        print(
            f"| rolling_notta | {len(rolling)} | {_fmt(tr)} | {_fmt(rel_r, 3)} "
            f"| {wr}/{lr}/{t_} | — | — | — | — | compare | host, not ours |"
        )
    for r in rows_out:
        print(
            f"| {r['method']} | {r['n']} | {_fmt(r['tail'])} | {_fmt(r['rel'], 3)} "
            f"| {r['w']}/{r['l']}/{r['t']} | {r['exact']} | {r['fire']} | {r['act']} "
            f"| {r['w_rf']}/{r['l_rf']}/{r['t_rf']} | {r['cell']} | {NEXT[r['cell']]} |"
        )

    print()
    print("## Conditional tail Δrel vs SF (fired vs quiet)")
    print()
    print("| Method | fired median Δrel | quiet median Δrel | later-freeze after rewind |")
    print("|---|---:|---:|---:|")
    for r in rows_out:
        print(
            f"| {r['method']} | {_fmt(r['cond_fire'], 3)} | {_fmt(r['cond_quiet'], 3)} "
            f"| {r['later'] if r['method'] == 'sf_rewind' else '—'} |"
        )

    print()
    print("## VBench full clip")
    print()
    print("| Method | subject | IQ | Dyn | flicker | IQ bar | subject bar |")
    print("|---|---:|---:|---:|---:|---|---|")
    if vb_sf:
        print(
            f"| notta (SF) | {_fmt(vb_sf.get('subject_consistency'))} | "
            f"{_fmt(vb_sf.get('imaging_quality'), 2)} | "
            f"{_fmt(vb_sf.get('dynamic_degree'), 2)} | "
            f"{_fmt(vb_sf.get('temporal_flickering'))} | — | — |"
        )
    if vb_rf:
        print(
            f"| rolling_notta | {_fmt(vb_rf.get('subject_consistency'))} | "
            f"{_fmt(vb_rf.get('imaging_quality'), 2)} | "
            f"{_fmt(vb_rf.get('dynamic_degree'), 2)} | "
            f"{_fmt(vb_rf.get('temporal_flickering'))} | — | — |"
        )
    for r in rows_out:
        vb = r["vb"] or {}
        print(
            f"| {r['method']} | {_fmt(vb.get('subject_consistency'))} | "
            f"{_fmt(vb.get('imaging_quality'), 2)} | "
            f"{_fmt(vb.get('dynamic_degree'), 2)} | "
            f"{_fmt(vb.get('temporal_flickering'))} | "
            f"{'hold' if r['iq_ok'] else 'FAIL'} | "
            f"{'hold' if r['subj_ok'] else 'FAIL'} |"
        )

    print()
    print("## Named + extreme videos (tail)")
    print()
    print("| Video | SF | RF | rewind | sick | pseudo | always | sink |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    named_keys = []
    for k in sorted(notta):
        stem = k.lower()
        if "0004" in stem or "0027" in stem:
            named_keys.append(k)
    extra = []
    for r in rows_out:
        extra.extend([k for _, k, _, _ in r["worst"] + r["best"]])
    show = []
    for k in named_keys + extra:
        if k not in show:
            show.append(k)

    def _t(mapping, k):
        rec = mapping.get(k)
        return _fmt(_tail(rec), 5) if rec else "—"

    by_m = {r["method"]: r["mapping"] for r in rows_out}
    for k in show:
        print(
            f"| {k} | {_t(notta, k)} | {_t(rolling, k)} | "
            f"{_t(by_m.get('sf_rewind', {}), k)} | "
            f"{_t(by_m.get('sf_sick_search', {}), k)} | "
            f"{_t(by_m.get('sf_pseudo', {}), k)} | "
            f"{_t(by_m.get('sf_always_search', {}), k)} | "
            f"{_t(by_m.get('sf_sink', {}), k)} |"
        )

    print()
    print("## Win-set overlap (beat SF by >10%)")
    print()
    names = list(win_sets)
    print("| " + " | ".join(["method"] + names) + " | n_win |")
    print("|---|" + "|".join(["---:" ] * (len(names) + 1)) + "|")
    for a in names:
        cells = []
        for b in names:
            sa, sb = win_sets[a], win_sets[b]
            if not sa and not sb:
                cells.append("—")
            else:
                inter = len(sa & sb)
                union = len(sa | sb) or 1
                cells.append(f"{inter}/{union}")
        print(f"| {a} | " + " | ".join(cells) + f" | {len(win_sets[a])} |")
    print()
    print("Jaccard < 0.5 between two HOLD arms → one combine N=32 is allowed.")
    print("Jaccard ≥ 0.7 → keep the cheaper actuator.")
    print()
    print("## Next actions")
    print()
    for r in rows_out:
        print(f"- **{r['method']} cell {r['cell']}:** {NEXT[r['cell']]}")
    print()
    print("If every arm is A/B/C/H: next wave is one new sensor or one")
    print("combine, N=32, same playbook. No TTC. No I2V. No DROP retune.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
