import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="Trying to unpickle estimator", module="sklearn")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
