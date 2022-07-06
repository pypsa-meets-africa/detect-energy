import logging
import os

import geopandas as gpd
import numpy as np
import pandas as pd
from _helpers import configure_logging





if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake, sets_path_to_root

        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        snakemake = mock_snakemake("cycle_train", general_dataset="duke2bangladesh")
        sets_path_to_root("detect_energy")
    configure_logging(snakemake)