# read version from installed package
from importlib.metadata import version
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

__version__ = version("cluster_aware_splitter")
__package__ = __file__

logger.info(f"{__package__} version {__version__}.")