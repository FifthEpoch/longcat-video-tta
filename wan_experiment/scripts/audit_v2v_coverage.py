#!/usr/bin/env python3
"""Coverage + insight audit on already-generated V2V sidecars. No GPU.

    python3 -u wan_experiment/scripts/audit_v2v_coverage.py

Reads cluster result dirs. Cite medians. Prints markdown + optional --out.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


STILL = 0.012
SERIES = (
    ("bakeoff_8v", "wan_experiment/results/v2v_panda_bakeoff_8v"),
    ("confirm_32v", "wan_experiment/results/v2v_panda_confirm_32v"),
    ("quiet_32v", "wan_experiment/results/v2v_panda_quiet_32v"),
    ("tricks_8v", "wan_experiment/results/v2v_panda_tricks_8v"),
    ("lineage_8v", "wan_experiment/results/v2v_panda_lineage_8v"),
    ("ideas_8v", "wan_experiment/results/v2v_panda_ideas_8v"),
    ("live_32v", "wan_experiment/results/v2v_panda_live_32v"),
    ("forward_32v", "wan_experiment/results/v2v_panda_forward_32v"),
)


def _usable(rec: dict) -> bool:
    return bool(
        rec.get("ok")
        and not rec.get("skipped")
        and rec.get("tail_motion") is not None
    )


def _key(rec: dict, fallback: str = "") -> str:
    k = rec.get("file_name") or rec.get("stem") or fallback
    return Path(str(k)).name if k else fallback


def _load_method(d: Path) -> dict[str, dict]:
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


def _prefix(rec: dict):
    v = rec.get("prefix_motion")
    if v is not None:
        try:
            return float(v)
        except (TypeError, ValueError):
            pass
    chunks = rec.get("chunks") or []
    if chunks:
        v = chunks[0].get("prefix_motion")
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return float("nan")


def _tail(rec: dict) -> float:
    try:
        return float(rec["tail_motion"])
    except (TypeError, ValueError, KeyError):
        return float("nan")


def _spearman(xs, ys):
    pairs = [
        (x, y) for x, y in zip(xs, ys)
        if x == x and y == y
    ]
    n = len(pairs)
    if n < 3:
        return None
    def ranks(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        r = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            mid = 0.5 * (i + j) + 1.0
            for k in range(i, j + 1):
                r[order[k]] = mid
            i = j + 1
        return r
    rx = ranks([p[0] for p in pairs])
    ry = ranks([p[1] for p in pairs])
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    denx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    deny = math.sqrt(sum((b - my) ** 2 for b in ry))
    if denx < 1e-12 or deny < 1e-12:
        return None
    return num / (denx * deny)


def _noise_u(rec: dict):
    chunks = rec.get("chunks") or []
    us = []
    for ch in chunks:
        for c in ch.get("cands") or [ch]:
            ns = c.get("noise_stats") or {}
            u = ns.get("eps_mean_abs")
            if u is not None:
                try:
                    us.append(float(u))
                except (TypeError, ValueError):
                    pass
    return us


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    lines = [
        "# V2V coverage audit (already-run tests)",
        "",
        "Sidecars only. Cite medians. No new generate.",
        "",
    ]

    # Discover method dirs
    catalog = []
    for label, rel in SERIES:
        series = args.root / rel
        if not series.is_dir():
            lines.append(f"- `{label}` **missing** `{rel}`")
            continue
        for d in sorted(series.glob("*_h30s_shard*")):
            rows = _load_method(d)
            if not rows:
                continue
            method = d.name.split("_h30s_")[0]
            catalog.append((label, method, d, rows))

    notta = {}
    for label, method, d, rows in catalog:
        if method == "notta":
            notta[(label, "notta")] = rows
            if "8v" in label and "bakeoff" in label:
                notta["bakeoff"] = rows
            if "confirm" in label:
                notta["confirm"] = rows

    lines += [
        "## Per-method tails",
        "",
        "| Series | Method | N | tail med | vs paired notta | still W/L | live W/L | bit=notta |",
        "|---|---|---:|---:|---:|---|---|---:|",
    ]

    insights = []
    for label, method, d, rows in catalog:
        keys = sorted(rows)
        tails = [_tail(rows[k]) for k in keys]
        med = statistics.median([t for t in tails if t == t]) if tails else None
        base = None
        if "32" in label:
            base = notta.get("confirm")
        if base is None:
            base = notta.get("bakeoff")
        if method == "notta":
            base = None
        n_bit = still_w = still_l = live_w = live_l = 0
        deltas = []
        prefs = []
        if base:
            for k in keys:
                if k not in base:
                    continue
                a, b = _tail(rows[k]), _tail(base[k])
                if a != a or b != b:
                    continue
                if abs(a - b) < 1e-6:
                    n_bit += 1
                pm = _prefix(base[k])
                if pm != pm:
                    pm = _prefix(rows[k])
                win = a > b + 1e-8
                loss = a < b - 1e-8
                if pm == pm and pm < STILL:
                    still_w += int(win)
                    still_l += int(loss)
                else:
                    live_w += int(win)
                    live_l += int(loss)
                deltas.append(a - b)
                prefs.append(pm)
        vs = "—"
        if base and deltas:
            bt = [ _tail(base[k]) for k in keys if k in base ]
            bt = [t for t in bt if t == t]
            if med is not None and bt:
                vs = f"{(med / statistics.median(bt) - 1) * 100:+.1f}%"
        lines.append(
            f"| {label} | {method} | {len(rows)} | "
            f"{'—' if med is None else f'{med:.4f}'} | {vs} | "
            f"{still_w}/{still_l} | {live_w}/{live_l} | {n_bit} |"
        )
        rho = _spearman(prefs, deltas) if prefs and deltas else None
        if rho is not None and method != "notta":
            insights.append(
                f"- `{label}/{method}` Spearman(prefix, Δtail)={rho:+.2f} "
                f"(n={len(deltas)})"
            )

    lines += ["", "## Prefix vs Δtail", "", *(insights or ["- no paired deltas"])]

    # noise_probe U_t
    lines += ["", "## Idea 3 leftover: U_t on noise_probe", ""]
    found_u = False
    for label, method, d, rows in catalog:
        if method != "noise_probe":
            continue
        found_u = True
        all_u = []
        per = []
        for k, rec in rows.items():
            us = _noise_u(rec)
            if us:
                mu = statistics.mean(us)
                all_u.extend(us)
                per.append((k, mu, _tail(rec), _prefix(rec)))
        if not all_u:
            lines.append("- noise_probe present but no `eps_mean_abs` in sidecars.")
            continue
        lines.append(
            f"- n={len(per)} videos, {len(all_u)} chunk stats, "
            f"U median={statistics.median(all_u):.5g} "
            f"min={min(all_u):.5g} max={max(all_u):.5g}"
        )
        xs = [p[1] for p in per]
        ys = [p[2] for p in per]
        rho = _spearman(xs, ys)
        lines.append(f"- Spearman(U_mean, tail)={rho if rho is None else f'{rho:+.2f}'}")
        xs2 = [p[3] for p in per]
        rho2 = _spearman(xs2, xs)
        lines.append(
            f"- Spearman(prefix, U_mean)={rho2 if rho2 is None else f'{rho2:+.2f}'}"
        )
        if rho2 is not None and abs(rho2) < 0.2 and max(all_u) - min(all_u) < 0.01:
            lines.append("- **U_t is flat. Trigger is dead on this DMD. Do not revive noise_bon.**")
    if not found_u:
        lines.append("- no noise_probe dir.")

    lines += [
        "",
        "## What this means",
        "",
        "- Search methods that bit-match seed/notta on stills are skip-gates, not controllers.",
        "- rolling vs SF: if still W/L is majority and live is mixed, the host helps stills keep motion.",
        "- Spearman(prefix, Δtail) < 0 on a search method = identity damper on live prefixes.",
        "",
    ]
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
