"""
hallucination_pipeline/run_pipeline.py

Run Step 1, Step 2, and Step 4 for all four generators.

Full variants by default (CTGAN, ARF, RTVAE, DDPM), or the four _CORE variants
with --core.

The step scripts read GENERATOR from config.pipeline_config at import time, so
the wrapper rewrites that one line before each run and restores the original
afterwards — including on Ctrl-C or a crash. Nothing else in the config is
touched, and running a step script by hand afterwards behaves exactly as
before, picking up whatever GENERATOR the config says.

Each run is a separate subprocess: a failure in one generator does not stop the
others, and memory is released between runs.

Deliberately excludes Step 3 (LLM). It costs real money per call and is meant
to be run one generator at a time with an eye on the spend cap, not batched
across all eight here. Run step3_llm_filter.py and step3b_merge_llm.py by hand
afterward, per generator, same as before.

Step 4 is included by default: it only reshapes files Steps 1-3 already
wrote (no LLM calls, nothing recomputed), so there's no cost or safety reason
to keep it manual the way Step 3 is. It never writes to any Step 1/2/3 output
file — read-only against all of them, new files only — so running it here
cannot overwrite LLM results. Its Expert+LLM per-chain output will simply be
skipped (with a message, not an error) for any generator Step 3 hasn't been
run on yet.

Two ways to drive it:

  VS CODE PLAY BUTTON — edit the SETTINGS block below. RUN_VARIANTS picks
  "full" (pfa), "core" (pfp) or "both"; the other constants cover the same
  ground as the command-line flags. No arguments needed.

  COMMAND LINE — any flag given overrides the corresponding constant:
      python run_pipeline.py                    # uses the SETTINGS below
      python run_pipeline.py --variants core    # pfp only
      python run_pipeline.py --variants both    # pfa then pfp, 8 generators
      python run_pipeline.py --steps 2          # step 2 only
      python run_pipeline.py --dry-run          # show the plan, run nothing
      python run_pipeline.py --stop-on-error    # abort on first failure

Console output is streamed live and written to logs/step{N}_{GENERATOR}.log.
To run a single generator, set GENERATOR in the config and run the step script
directly — that is what this wrapper is temporarily overriding.
"""

import os
import re
import sys
import time
import argparse
import subprocess
from datetime import datetime, timedelta

# ============================================================
# CONFIG
# ============================================================

HERE = os.path.dirname(os.path.abspath(__file__))
# This file lives inside hallucination_pipeline/, one level below repo
# root -- config/ is a sibling of hallucination_pipeline/, not a child of it,
# so this goes up one level first.
CONFIG_PATH = os.path.join(HERE, "..", "config", "pipeline_config.py")
LOG_DIR = os.path.join(HERE, "logs")

# Order matters: step 2 reads the step 1 summary, and step 4 reads step 1 +
# step 2's ID files (and step 3's summary, if present).
STEPS = {
    1: "step1_hr_pilgram.py",
    2: "step2_expert_rules.py",
    4: "step4_per_chain_summary.py",
}

GENERATORS_FULL = ["CTGAN", "ARF", "RTVAE", "DDPM"]
GENERATORS_CORE = [f"{g}_CORE" for g in GENERATORS_FULL]

# ============================================================
# SETTINGS — edit these when running from the VS Code play button.
# A command-line flag, if given, overrides the matching constant.
# ============================================================

# Which variable set to run:
#   "full"  -> pfa: CTGAN, ARF, RTVAE, DDPM
#   "core"  -> pfp: the four _CORE variants
#   "both"  -> all eight, full first then core
RUN_VARIANTS = "both"

# Which steps, in order. [1, 2, 4] runs step 1, step 2, then step 4 per
# generator (step 3 is deliberately never in here -- see the module docstring).
RUN_STEPS = [4]

# True: print the plan and exit without running anything.
DRY_RUN = False

# True: abort the whole batch on the first failure.
# False: carry on and report failures in the summary at the end.
STOP_ON_ERROR = False

VARIANT_CHOICES = {
    "full": GENERATORS_FULL,
    "core": GENERATORS_CORE,
    "both": GENERATORS_FULL + GENERATORS_CORE,
}

# Matches a top-level assignment like:  GENERATOR       = "RTVAE"
# An optional trailing "# ..." comment -- as in the real config file, e.g.
# GENERATOR = "ARF_CORE" #ARF, CTGAN, RTVAE, DDPM -- is captured separately
# in its own group so it can be preserved across rewrites rather than
# stripped, and so its presence doesn't stop the line from matching at all.
GENERATOR_LINE = re.compile(
    r'^(GENERATOR\s*=\s*)([\'"])(.*?)\2[ \t]*(#.*)?$', re.M)


# ============================================================
# CONFIG PATCHING
# ============================================================

def read_config():
    if not os.path.exists(CONFIG_PATH):
        raise SystemExit(f"Config not found: {CONFIG_PATH}")
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return f.read()


def write_config(src):
    with open(CONFIG_PATH, "w", encoding="utf-8", newline="") as f:
        f.write(src)


def current_generator(src):
    m = GENERATOR_LINE.search(src)
    return m.group(3) if m else None


def set_generator(src, generator):
    def repl(m):
        comment = m.group(4) or ""
        sep = " " if comment else ""
        return f'{m.group(1)}"{generator}"{sep}{comment}'

    new, n = GENERATOR_LINE.subn(repl, src, count=1)
    if n != 1:
        raise SystemExit(
            f'Could not find a top-level GENERATOR = "..." line in\n'
            f'  {CONFIG_PATH}\n'
            f'The wrapper needs that line to switch generators.')
    return new


# ============================================================
# RUNNING
# ============================================================

def run_step(step, log_path):
    """Run one step script in a subprocess, streaming output to screen + log."""
    script = os.path.join(HERE, STEPS[step])
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    # Windows defaults piped stdout to cp1252, which cannot encode the check
    # marks and dashes the step scripts print. Without this the child crashes
    # on its first such print — after doing all the work.
    env["PYTHONIOENCODING"] = "utf-8"

    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"# {STEPS[step]} | {datetime.now():%Y-%m-%d %H:%M:%S}\n\n")
        log.flush()

        proc = subprocess.Popen(
            [sys.executable, script],
            cwd=HERE, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        for line in proc.stdout:
            sys.stdout.write("    " + line)
            sys.stdout.flush()
            log.write(line)
        proc.wait()

    elapsed = time.time() - t0
    ok = proc.returncode == 0
    return ok, elapsed, ("" if ok else f"exit code {proc.returncode}")


def fmt(seconds):
    return str(timedelta(seconds=int(seconds)))


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", choices=list(VARIANT_CHOICES), default=None,
                    help=f"full (pfa) / core (pfp) / both "
                         f"(default: RUN_VARIANTS = {RUN_VARIANTS!r})")
    ap.add_argument("--steps", nargs="*", type=int, default=None,
                    choices=list(STEPS),
                    help=f"default: RUN_STEPS = {RUN_STEPS}")
    ap.add_argument("--stop-on-error", action="store_true", default=None,
                    help="abort on the first failure "
                         "(default: continue and report at the end)")
    ap.add_argument("--dry-run", action="store_true", default=None)
    args = ap.parse_args()

    # Flags win when supplied; otherwise fall back to the SETTINGS block so the
    # play button and the command line behave identically.
    variants = args.variants or RUN_VARIANTS
    if variants not in VARIANT_CHOICES:
        raise SystemExit(f"RUN_VARIANTS must be one of "
                         f"{sorted(VARIANT_CHOICES)}, got {variants!r}")
    generators = VARIANT_CHOICES[variants]
    steps = sorted(args.steps if args.steps is not None else RUN_STEPS)
    dry_run = DRY_RUN if args.dry_run is None else args.dry_run
    stop_on_error = (STOP_ON_ERROR if args.stop_on_error is None
                     else args.stop_on_error)

    if not steps:
        raise SystemExit("No steps selected — check RUN_STEPS.")

    original_src = read_config()
    original_generator = current_generator(original_src)
    if original_generator is None:
        raise SystemExit(
            f'No top-level GENERATOR = "..." line found in\n  {CONFIG_PATH}')

    print(f"Project    : {HERE}")
    print(f"Python     : {sys.executable}")
    label = {"full": "full (pfa)", "core": "core (pfp)",
             "both": "full + core (pfa + pfp)"}[variants]
    print(f"Variants   : {label}")
    print(f"Generators : {', '.join(generators)}")
    print(f"Steps      : {', '.join(f'{s} ({STEPS[s]})' for s in steps)}")
    print(f"Runs       : {len(generators) * len(steps)}")
    print(f"\nThe GENERATOR line in pipeline_config.py (currently "
          f'"{original_generator}") is rewritten before each run and restored\n'
          f"at the end. Do not edit that file while this is running.")

    missing = [STEPS[s] for s in steps
               if not os.path.exists(os.path.join(HERE, STEPS[s]))]
    if missing:
        raise SystemExit(f"\nMissing script(s): {', '.join(missing)}")

    if dry_run:
        print("\n[DRY RUN] would run, in order:")
        for g in generators:
            for s in steps:
                print(f"    {STEPS[s]:26s} {g}")
        print(f'\n    then restore GENERATOR = "{original_generator}"')
        return

    os.makedirs(LOG_DIR, exist_ok=True)
    results = []
    t_start = time.time()

    try:
        for gi, generator in enumerate(generators, start=1):
            write_config(set_generator(original_src, generator))

            for s in steps:
                print(f"\n{'=' * 70}\n"
                      f"[{gi}/{len(generators)}] {generator} — step {s}\n"
                      f"{'=' * 70}")

                log_path = os.path.join(LOG_DIR, f"step{s}_{generator}.log")
                ok, elapsed, err = run_step(s, log_path)
                results.append({"generator": generator, "step": s, "ok": ok,
                                "elapsed": elapsed, "error": err})

                status = "OK" if ok else f"FAILED — {err}"
                print(f"\n  -> {status}  ({fmt(elapsed)})  log: {log_path}")

                if not ok:
                    if stop_on_error:
                        raise SystemExit(
                            f"\nStopping: {generator} step {s} failed.")
                    if s == 1 and 2 in steps:
                        print("  [WARN] step 2 reads step 1's summary and "
                              "will probably fail too.")
                    if s in (1, 2) and 4 in steps:
                        print(f"  [WARN] step 4 reads step {s}'s ID file and "
                              f"will probably produce incomplete output too.")

    except KeyboardInterrupt:
        print("\n\nInterrupted.")
    finally:
        # Always put the config back exactly as it was.
        write_config(original_src)
        print(f'\nRestored GENERATOR = "{original_generator}"')

    # ---- summary ----
    print(f"\n{'=' * 70}\nSUMMARY  (total {fmt(time.time() - t_start)})\n"
          f"{'=' * 70}")
    print(f"{'generator':<14}{'step':<6}{'status':<10}{'elapsed':<12}error")
    for r in results:
        print(f"{r['generator']:<14}{r['step']:<6}"
              f"{'OK' if r['ok'] else 'FAILED':<10}"
              f"{fmt(r['elapsed']):<12}{r['error']}")

    failed = [r for r in results if not r["ok"]]
    print(f"\n{len(results) - len(failed)}/{len(results)} succeeded")
    if failed:
        print("failed: " + ", ".join(f"{r['generator']} step {r['step']}"
                                     for r in failed))
        sys.exit(1)

    print("\nNext: run step3_llm_filter.py + step3b_merge_llm.py by hand per "
          "generator (then rerun step 4 for that generator to pick up the "
          "Expert+LLM per-chain file).")


if __name__ == "__main__":
    main()