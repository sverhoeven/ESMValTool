#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Diagnostic script to create individual GBRT models for many climate models.

Description
-----------
This diagnostic creates "Gradient Boosted Regression Trees" (GBRT) models to
predict future climate for multiple climate models.

Author
------
Manuel Schlund (DLR, Germany)

Project
-------
CRESCENDO

Configuration options in recipe
-------------------------------
see esmvaltool.gbrt module.

"""

import logging
import os

from esmvaltool.diag_scripts.gbrt import GBRTModel
from esmvaltool.diag_scripts.shared import (group_metadata, run_diagnostic,
                                            select_metadata)

logger = logging.getLogger(os.path.basename(__file__))


def main(cfg):
    """Run the diagnostic."""
    input_data = cfg['input_data'].values()
    preselection = cfg.get('metadata_preselection', {})
    group = preselection.get('group')
    input_data = select_metadata(input_data, **preselection.get('select', {}))
    grouped_datasets = group_metadata(input_data, group)
    for attr in grouped_datasets:
        logger.info("Processing %s", attr)
        if group is not None:
            metadata = {group: attr}
        else:
            metadata = {}
        gbrt_model = GBRTModel(cfg, root_dir=attr, **metadata)

        # TODO
        for d in grouped_datasets[attr]:
            print(d['filename'])

        # Fit and predict
        gbrt_model.fit()
        gbrt_model.predict()

        # Plots
        gbrt_model.plot_feature_importance()
        gbrt_model.plot_partial_dependence()
        gbrt_model.plot_prediction_error()


# Run main function when this script is called
if __name__ == '__main__':
    with run_diagnostic() as config:
        main(config)
