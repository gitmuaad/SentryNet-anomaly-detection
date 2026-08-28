"""SentryNet: unsupervised anomaly detection for network flow data.

Models are fit on confirmed-normal rows only. PortScan is excluded from training and
tuning and used once, on the final test, as an unseen-attack check.
"""

from sentrynet.config import Config, load_config

__all__ = ["Config", "load_config", "__version__"]
__version__ = "1.0.0"
