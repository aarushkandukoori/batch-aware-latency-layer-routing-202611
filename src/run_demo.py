'''End-to-end demo on SYNTHETIC data.

Reference implementation of a PROPOSAL. Prints toy numbers from a made-up task and made-up costs.
They are NOT benchmark results and say nothing about real hardware.

Usage (from the repo root):  python -m src.run_demo [--steps 400] [--batch-size 16]
Uses data/synthetic/ if present (see data/generate_synthetic.py), otherwise generates in memory.
'''
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.baselines import run_all  # noqa: E402
from src.cost_tables import make_synthetic_cost_table  # noqa: E402
from src.synthetic_tasks import load_synthetic, make_markov_tokens, make_need_matrix  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=400)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--budget-frac', type=float, default=0.5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--synthetic-dir', default=str(ROOT / 'data' / 'synthetic'))
    args = ap.parse_args()

    loaded = load_synthetic(args.synthetic_dir)
    if loaded is None:
        print('No files in', args.synthetic_dir, '- generating SYNTHETIC data in memory.')
        tokens = make_markov_tokens(seed=args.seed)
        need = make_need_matrix(seed=args.seed)
        table = make_synthetic_cost_table(seed=args.seed)
    else:
        tokens, need, table = loaded

    print('SYNTHETIC toy setting. Values below are NOT benchmark results.')
    print('Costs:', table.label)
    rows = run_all(tokens, need, table, B=args.batch_size, steps=args.steps,
                   budget_frac=args.budget_frac, seed=args.seed)
    print(f"{'method':36s} {'toy_task_loss':>14s} {'layers':>7s} {'modeled_T_B':>12s}")
    for r in rows:
        print(f"{r['name']:36s} {r['task_loss']:14.4f} {r['layers']:7.2f} {r['modeled_T_B']:12.4f}")


if __name__ == '__main__':
    main()
