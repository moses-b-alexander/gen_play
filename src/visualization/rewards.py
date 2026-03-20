
import numpy as np


def show_reward_histogram(rewards: list[float], bin_width: float) -> None:
    bins = int((2 - 0) / bin_width)

    hist, bin_edges = np.histogram(rewards, bins=bins, range=(0, 2))
    max_count = hist.max()

    for i in range(bins):
        left = bin_edges[i]
        right = bin_edges[i + 1]
        count = hist[i]
        bar = "=" * int(count)
        print(f"[{left:.2f}, {right:.2f}): {bar} ({count})")

    return None
