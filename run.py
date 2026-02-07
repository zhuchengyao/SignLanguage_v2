import importlib
import sys


COMMAND_MODULES = {
    "train-vqvae": "src.commands.train.vqvae",
    "train-gpt": "src.commands.train.gpt",
    "train-ae2d": "src.commands.train.ae2d",
    "train-flow2d": "src.commands.train.flow2d",
    "infer-t2m": "src.commands.infer.t2m",
    "infer-flow2d-text": "src.commands.infer.flow2d_text",
    "eval-vqvae": "src.commands.eval.vqvae",
    "eval-gpt-bleu": "src.commands.eval.gpt_bleu",
    "data-extract-displacements": "src.commands.data.extract_displacements",
    "data-avg-displacements": "src.commands.data.avg_displacements",
    "viz-flow2d-samples": "src.commands.viz.flow2d_samples",
    "viz-gt-text": "src.commands.viz.gt_text",
    "viz-p0-points": "src.commands.viz.p0_points",
    "viz-reconstruct-deltas": "src.commands.viz.reconstruct_from_deltas",
    "debug-memory-leak": "src.commands.debug.memory_leak",
    "verify-latent-step1": "src.commands.verify.latent_step1",
    "verify-latent2d-step1": "src.commands.verify.latent2d_step1",
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print("Usage: python run.py <command> [args...]")
        print("")
        print("Available commands:")
        for cmd in sorted(COMMAND_MODULES.keys()):
            print(f"  {cmd}")
        return

    command = sys.argv[1]
    remaining = sys.argv[2:]
    if command not in COMMAND_MODULES:
        print(f"Unknown command: {command}")
        print("Run `python run.py --help` for available commands.")
        raise SystemExit(2)

    module_name = COMMAND_MODULES[command]
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        missing = exc.name or "unknown"
        print(
            f"Failed to import `{module_name}` because dependency `{missing}` is missing.\n"
            "Install dependencies first: `pip install -r requirements.txt`."
        )
        raise SystemExit(1) from exc
    if not hasattr(module, "main"):
        raise RuntimeError(f"{module_name} has no main() entrypoint")

    sys.argv = [command, *remaining]
    module.main()


if __name__ == "__main__":
    main()
