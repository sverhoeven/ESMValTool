"""Module that provides common plot functions."""

from ._plot import (get_dataset_style, get_path_to_mpl_style, global_contourf,
                    multi_dataset_scatterplot, quickplot, scatterplot)

__all__ = [
    'get_path_to_mpl_style',
    'get_dataset_style',
    'global_contourf',
    'quickplot',
    'multi_dataset_scatterplot',
    'scatterplot',
]
