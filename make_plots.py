"""generate all plots"""

import argparse

from utils.plots import *


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        '--logs_dir',
        help='directory containing the logs files',
        type=str,
        required=True
    )
    parser.add_argument(
        '--history_dir',
        help='directory containing the clients sampler history files',
        type=str,
        required=True
    )
    parser.add_argument(
        '--save_dir',
        help='directory to save all the plots',
        type=str,
        required=True
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    plot_all_logs(logs_dir=args.logs_dir, save_dir=args.save_dir)
    plot_all_history(history_dir=args.history_dir, save_dir=args.save_dir)
    plot_participation_heatmap(
        history_dir=args.history_dir,
        save_dir=args.save_dir
    )

    print(f"results are saved to {args.save_dir}")
