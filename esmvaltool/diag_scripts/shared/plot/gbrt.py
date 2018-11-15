"""Common plot functions for GBRT diagnostics."""


import logging
import os

import numpy as np
from sklearn.ensemble import partial_dependence as skl
import matplotlib
matplotlib.use('Agg')  # noqa
import matplotlib.pyplot as plt  # noqa


logger = logging.getLogger(os.path.basename(__file__))


def plot_feature_importance(clf, feature_names, cfg,
                            filename='feature_importance'):
    """Plot feature importance."""
    if not cfg['write_plots']:
        return
    (_, axes) = plt.subplots()
    feature_importance = clf.feature_importances_
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + 0.5
    axes.barh(pos, feature_importance[sorted_idx], align='center')
    axes.set_title('Variable Importance')
    axes.set_xlabel('Relative Importance')
    axes.set_yticks(pos)
    axes.set_yticklabels(feature_names[sorted_idx])
    new_path = os.path.join(cfg['plot_dir'],
                            filename + '.' + cfg['output_file_type'])
    plt.savefig(new_path, orientation='landscape', bbox_inches='tight')
    logger.info("Wrote %s", new_path)
    plt.close()


def plot_partial_dependence(clf, x_train, feature_names, cfg,
                            filename='partial_dependence'):
    """Plot partial dependence."""
    if not cfg['write_plots']:
        return
    for (idx, feature_name) in enumerate(feature_names):
        (_, [axes]) = skl.plot_partial_dependence(clf, x_train, [idx])
        axes.set_title('Partial dependence')
        axes.set_xlabel(feature_name)
        axes.set_ylabel('Eastward wind')
        new_path = os.path.join(cfg['plot_dir'],
                                filename + '_{}.{}'.format(
                                    feature_name, cfg['output_file_type']))
        plt.savefig(new_path, orientation='landscape', bbox_inches='tight')
        logger.info("Wrote %s", new_path)
        plt.close()


def plot_prediction_error(clf, test_score, cfg,
                          filename='prediction_error'):
    """Plot prediction error during training procedure."""
    if not cfg['write_plots']:
        return
    (_, axes) = plt.subplots()
    axes.plot(np.arange(len(clf.train_score_)) + 1, clf.train_score_, 'b-',
              label='Training Set Deviance')
    axes.plot(np.arange(len(test_score)) + 1, test_score, 'r-',
              label='Test Set Deviance')
    axes.legend(loc='upper right')
    axes.set_title('Deviance')
    axes.set_xlabel('Boosting Iterations')
    axes.set_ylabel('Deviance')
    new_path = os.path.join(cfg['plot_dir'],
                            filename + '.' + cfg['output_file_type'])
    plt.savefig(new_path, orientation='landscape', bbox_inches='tight')
    logger.info("Wrote %s", new_path)
    plt.close()
