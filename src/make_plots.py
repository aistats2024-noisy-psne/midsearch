#!/usr/bin/env python

import pandas as pd
import os

from psne import DATA_DIR
from psne.plotting import plot_h1_sweep

if __name__ == "__main__":
    exp_types = [
        1,
        2,
        3,
        4,
        5,
        6,
    ]
    for exp_type in exp_types:
        filenames = [
            f
            for f in os.listdir(DATA_DIR)
            if (f.startswith(f"experiment_{exp_type}") and f.endswith(".csv"))
        ]
        if not filenames:
            print(f"No data to plot for {exp_type=}.")
            continue
        # Plots most recent version of data for each experiment
        filename = sorted(filenames)[-1]
        plot_df = pd.read_csv(os.path.join(DATA_DIR, filename))
        print(f"Making plot {exp_type} from {filename}.")
        svg_filename = os.path.join(DATA_DIR, f"h1_sweep_experiment_{exp_type}.svg")
        plot_h1_sweep(plot_df, svg_filename, exp_type=exp_type)
