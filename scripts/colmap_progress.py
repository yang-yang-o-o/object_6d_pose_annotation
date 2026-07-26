#!/usr/bin/env python3
"""Filter COLMAP stdout/stderr: highlight progress with ETA."""
from __future__ import annotations

import re
import sys
import time

view_re = re.compile(r"Processing view\s+(\d+)\s*/\s*(\d+)")
undist_re = re.compile(r"Undistorting image\s*\[(\d+)/(\d+)\]")
geom_re = re.compile(r"geom_consistency:\s*([01])")


def fmt_eta(sec: float) -> str:
    if sec != sec or sec < 0 or sec > 1e7:
        return "?"
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def main():
    t0 = time.time()
    last_t = t0
    rates: list[float] = []
    global_done = 0  # across passes
    n_views = None
    geom = 0
    last_key = None

    for raw in sys.stdin:
        sys.stdout.write(raw)
        sys.stdout.flush()
        line = raw.rstrip("\n")

        gm = geom_re.search(line)
        if gm:
            geom = int(gm.group(1))

        m = undist_re.search(line)
        if m:
            i, n = int(m.group(1)), int(m.group(2))
            now = time.time()
            key = ("u", i)
            if last_key != key:
                if last_key and last_key[0] == "u":
                    rates.append(now - last_t)
                    rates = rates[-30:]
                last_t = now
                last_key = key
            avg = sum(rates) / len(rates) if rates else 0
            remain = (n - i) * avg
            print(
                f"[progress] Undistort {i}/{n} ({100.0 * i / n:.0f}%) "
                f"ETA {fmt_eta(remain)} | elapsed {fmt_eta(now - t0)}",
                flush=True,
            )
            continue

        m = view_re.search(line)
        if m:
            i, n = int(m.group(1)), int(m.group(2))
            n_views = n
            now = time.time()
            key = ("v", geom, i)
            if last_key != key:
                if last_key and last_key[0] == "v":
                    dt = now - last_t
                    if dt > 0.2:
                        rates.append(dt)
                        rates = rates[-30:]
                last_t = now
                last_key = key
                # count completions: when we see view i, completed is (pass)*n + (i-1)
                global_done = geom * n + (i - 1)

            avg = sum(rates) / len(rates) if rates else 0
            total = 2 * n  # photometric + geometric
            remain_steps = max(0, total - (geom * n + i))
            remain = remain_steps * avg
            pct = 100.0 * (geom * n + i) / total
            phase = "geometric" if geom else "photometric"
            print(
                f"[progress] PatchMatch {phase} {i}/{n} | {pct:.0f}% "
                f"({geom * n + i}/{total}) | {avg:.1f}s/view | "
                f"ETA {fmt_eta(remain)} | elapsed {fmt_eta(now - t0)}",
                flush=True,
            )
            continue

        if "Elapsed time:" in line or "Fusing image" in line or "StereoFusion" in line:
            print(f"[progress] {line.strip()}", flush=True)


if __name__ == "__main__":
    main()
