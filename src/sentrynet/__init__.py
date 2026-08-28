"""SentryNet — unsupervised anomaly detection on structured network flows.

Two rules govern every module here:

1. Learned models and the preprocessing pipeline are fit on **confirmed Normal rows only**.
2. ``PortScan`` never participates in hyperparameter or threshold selection; it is the
   held-out *unseen attack* used once, on the final test set.
"""

from sentrynet.config import Config, load_config

__all__ = ["Config", "load_config", "__version__"]
__version__ = "1.0.0"
