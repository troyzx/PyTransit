"""
This module provides functions for reading Hubble spoc from a pickle file.
"""


from pathlib import Path
from typing import Union
import pickle
import os
import pandas as pd


def read_hubble_spoc(datadir: Union[Path, str]):
    """
    Read the extracted light curves from a pickle file.

    Parameters:
    - datadir: The directory path where the pickle file is located.

    Returns:
    - data_lc: The loaded data from the pickle file.
    """

    # Read lc data from extracted_light_curves.pickle
    file_path_lc = os.path.join(datadir, "extracted_light_curves.pickle")

    # Open the pickle file in read mode
    with open(file_path_lc, "rb") as file:
        # Load the data from the pickle file
        data_lc = pickle.load(file)
        df_lc = pd.DataFrame(
            {
                "time": data_lc["bjd_tdb_array"],
                "flux": data_lc["white"]["flux"],
                "flux_err": data_lc["white"]["error"],
                "data_scan": data_lc["spectrum_direction_array"],
            }
        )

    # Read par data from fitting_results.pickle
    file_path_par = os.path.join(datadir, "fitting_results.pickle")

    # Open the pickle file in read mode
    with open(file_path_par, "rb") as file:
        # Load the data from the pickle file
        data_par = pickle.load(file)

    light_curve_fitted = data_par
    keys = list(light_curve_fitted["lightcurves"]["white"]["parameters"].keys())
    values = light_curve_fitted["lightcurves"]["white"]["parameters_final"]
    fitted_parameters = dict(zip(keys, values))
    df_par = pd.DataFrame.from_dict(
        fitted_parameters, orient="index", columns=["iraclis_rlt"]
    )
    # df_par.T.to_csv(os.path.join(dir, "par.csv"), index=False)

    return df_lc, df_par.T
