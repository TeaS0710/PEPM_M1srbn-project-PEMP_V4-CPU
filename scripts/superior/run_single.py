#!/usr/bin/env python3
import argparse
import subprocess
import sys
import time
from pathlib import Path

try:
    import psutil  # type: ignore
except ImportError:
    psutil = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single core pipeline (prepare/train/evaluate) via `make run`."
    )
    parser.add_argument("--exp-id", required=True, help="Experiment id (for logging only).")
    parser.add_argument("--run-id", required=True, help="Run id (for logging only).")

    parser.add_argument(
        "--profile",
        required=True,
        help="Core profile name (configs/profiles/*.yml).",
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=["prepare", "train", "evaluate", "pipeline"],
        help="Core stage to run.",
    )

    parser.add_argument(
        "--make-var",
        action="append",
        default=[],
        help="Extra MAKEVAR in the form KEY=VALUE. Can be repeated.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override in the form key=value (passed to OVERRIDES=...). Can be repeated.",
    )

    parser.add_argument(
        "--log-path",
        required=True,
        help="Path to the log file for this run.",
    )

    parser.add_argument(
        "--max-ram-mb",
        type=int,
        default=None,
        help=(
            "Hard RAM limit per run (MB). If exceeded (and psutil is available), "
            "the run is killed and exit code 99 is returned."
        ),
    )

    return parser.parse_args()


def build_cmd(args: argparse.Namespace) -> list[str]:
    cmd: list[str] = [
        "make",
        "run",
        f"STAGE={args.stage}",
        f"PROFILE={args.profile}",
    ]

    for mv in args.make_var:
        # mv is expected to be "KEY=VALUE"
        cmd.append(str(mv))

    if args.override:
        overrides_str = " ".join(str(o) for o in args.override)
        cmd.append(f"OVERRIDES={overrides_str}")

    return cmd


def main() -> int:
    args = parse_args()
    cmd = build_cmd(args)

    log_path = Path(args.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"[run_single] exp_id={args.exp_id} run_id={args.run_id}\n")
        log_file.write(f"[run_single] CMD: {' '.join(cmd)}\n")
        log_file.flush()

        # Pas de limite dure ou psutil absent => simple Popen.wait()
        if args.max_ram_mb is None or psutil is None:
            if args.max_ram_mb is not None and psutil is None:
                msg = "[run_single] psutil not installed – RAM monitoring disabled despite max-ram-mb flag\n"
                print(msg.strip(), file=sys.stderr)
                log_file.write(msg)
                log_file.flush()

            proc = subprocess.Popen(cmd, stdout=log_file, stderr=log_file)
            return_code = proc.wait()
            return return_code

        # Limite dure active
        limit_bytes = int(args.max_ram_mb) * 1024 * 1024
        proc = subprocess.Popen(cmd, stdout=log_file, stderr=log_file)

        try:
            ps_proc = psutil.Process(proc.pid)  # type: ignore[attr-defined]
        except Exception:
            # Si on n'arrive pas à attacher psutil, fallback sans monitoring
            proc = subprocess.Popen(cmd, stdout=log_file, stderr=log_file)
            return proc.wait()

        peak_rss = 0

        while True:
            ret = proc.poll()
            if ret is not None:
                return_code = ret
                break

            try:
                rss = ps_proc.memory_info().rss  # type: ignore[attr-defined]
            except Exception:
                # Le process a pu se terminer entre-temps
                break

            peak_rss = max(peak_rss, rss)
            if rss > limit_bytes:
                msg = "[run_single] RAM limit exceeded, killing run\n"
                print(msg.strip(), file=sys.stderr)
                log_file.write(msg)
                log_file.flush()
                try:
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                except Exception:
                    pass
                return_code = 99
                break

            time.sleep(1.0)

        log_file.write(f"[run_single] peak_rss_mb={peak_rss / (1024 * 1024):.2f}\n")
        log_file.flush()
        return return_code


if __name__ == "__main__":
    sys.exit(main())
