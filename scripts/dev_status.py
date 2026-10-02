"""What is finished, what is running, and what is left - for one or more queues.

    python scripts/dev_status.py w9 w9b
    python scripts/dev_status.py w9 --todo

A run at epochs=100 takes about 4,085 s, so a six-arm queue is most of a day and the
question "how far along is this" comes up repeatedly. Reading it off three tmux panes
means reading three partial views, and a pane that was interrupted shows nothing at all.

A run counts as FINISHED when its checkpoint exists, which is the same test the generated
queue scripts use to decide whether to skip - so this report and the queue cannot disagree
about what is done. A run counts as IN FLIGHT when its directory and log exist but its
checkpoint does not, because scripts/train.py writes the log from the start and the
checkpoint only at the end. That also identifies a run that DIED: it stays in flight with a
log whose last line is not the final one, and --todo will not list it, so the count is the
thing to read rather than the list.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def state(tag: str) -> tuple[str, float | None]:
    """finished | in flight | not started, and the age in seconds of the newest file."""
    run_dir = os.path.join(ROOT, "results", "runs", tag)
    checkpoint = os.path.join(run_dir, "checkpoint.pt")
    log = os.path.join(run_dir, "train.log")
    if os.path.exists(checkpoint):
        return "finished", time.time() - os.path.getmtime(checkpoint)
    if os.path.exists(log):
        return "in flight", time.time() - os.path.getmtime(log)
    return "not started", None


def human(seconds: float | None) -> str:
    if seconds is None:
        return ""
    if seconds < 90:
        return f"{seconds:.0f}s ago"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m ago"
    return f"{seconds / 3600:.1f}h ago"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="+", help="queue names under results/dev/")
    parser.add_argument("--todo", action="store_true",
                        help="list every run that has not started, with its GPU")
    args = parser.parse_args()

    for name in args.names:
        # Development queues live under results/dev/, the final runs under results/final/,
        # and a bare name does not say which. Both are tried rather than making the caller
        # remember: telling someone to run `dev_status.py final` and having it answer "the
        # queue has not been generated" about a queue that exists is worse than a second
        # stat() call.
        candidates = [os.path.join(ROOT, "results", "dev", name, "manifest.json"),
                      os.path.join(ROOT, "results", name, "manifest.json")]
        path = next((p for p in candidates if os.path.exists(p)), None)
        if path is None:
            print(f"{name}: no manifest at "
                  f"{' or '.join(os.path.relpath(p, ROOT) for p in candidates)}")
            continue
        with open(path, encoding="utf-8") as handle:
            jobs = json.load(handle)["jobs"]

        states = {job["tag"]: state(job["tag"]) for job in jobs}
        counts = {"finished": 0, "in flight": 0, "not started": 0}
        for kind, _ in states.values():
            counts[kind] += 1
        print(f"{name}: {counts['finished']}/{len(jobs)} finished, "
              f"{counts['in flight']} in flight, {counts['not started']} not started")

        for gpu in sorted({job["gpu"] for job in jobs}):
            mine = [job for job in jobs if job["gpu"] == gpu]
            done = sum(1 for job in mine if states[job["tag"]][0] == "finished")
            flight = [job["tag"] for job in mine if states[job["tag"]][0] == "in flight"]
            # The newest touch on an in-flight log says whether that GPU is working or
            # whether its queue died: a live run writes a line every epoch.
            age = human(min((states[t][1] for t in flight), default=None))
            print(f"  gpu {gpu}: {done}/{len(mine)} finished"
                  + (f"   running {flight[0]} (log {age})" if flight else "   idle"))

        if args.todo:
            for job in jobs:
                if states[job["tag"]][0] != "finished":
                    print(f"    {states[job['tag']][0]:11s} {job['tag']:34s} gpu "
                          f"{job['gpu']}")


if __name__ == "__main__":
    main()
