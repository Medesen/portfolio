"""CLI router: dispatches `mlctl <command> [hydra overrides...]` to Hydra entry points.

Commands are registered in a dict rather than an if/elif tree, so adding one
is a single line and the usage text stays in sync automatically.
"""

import importlib
import sys

COMMANDS: dict[str, tuple[str, str]] = {
    "train": ("mlctl.train", "Train a model on the diabetes dataset"),
    "evaluate": ("mlctl.evaluate", "Re-score a finished run on its held-out test split"),
}


def usage() -> str:
    lines = ["Usage: mlctl <command> [hydra overrides...]", "", "Commands:"]
    for name, (_, description) in COMMANDS.items():
        lines.append(f"  {name:<10} {description}")
    lines += [
        "",
        "Examples:",
        "  mlctl train",
        "  mlctl train model=ridge model.alpha=10.0",
        "  mlctl train --config-name=gbm_tuned",
        "  mlctl train -m model=ridge,gbm seed=0,1,2",
        "  mlctl evaluate run_dir=outputs/baseline/gbm/seed_42",
        "  mlctl train --help",
    ]
    return "\n".join(lines)


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(usage())
        raise SystemExit(0 if len(sys.argv) >= 2 else 1)

    command = sys.argv[1]
    if command not in COMMANDS:
        print(f"Unknown command: {command}\n\n{usage()}")
        raise SystemExit(1)

    # Drop the command name; everything after it belongs to Hydra
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    module_name, _ = COMMANDS[command]
    importlib.import_module(module_name).main()
