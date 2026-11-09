"""Central configuration for EUR/USD Informer experiments."""
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "EURUSD60_processed.csv"
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
RESULTS_DIR = PROJECT_DIR / "results"
PLOTS_DIR = PROJECT_DIR / "plots"

# Input variables from historical candles and technical indicators.
# This follows the uploaded notebook, including Close as an input feature.
DATE_COL = "date"
CLOSE_COL = "Close"
LOG_RETURN_COL = "log_return"
RSI_COL = "RSI_14"
ATR_COL = "ATR_14"
FEATURE_COLS = ["Open", "High", "Low", CLOSE_COL, "Volume", LOG_RETURN_COL, RSI_COL, ATR_COL]
USE_TIME_FEATURES = True
TARGET_MODE = "log_return_to_close"
TARGET_COLS = [LOG_RETURN_COL]

SEQ_LENS = [48, 96, 192, 384]
PRED_LENS = [1, 4, 8]
SAMPLE_TARGET_PREDICTIONS = 100
SAMPLE_ALIGNMENT_SEQ_LEN = max(SEQ_LENS)
SAMPLE_PREDICTION_END = None  # None means use the final available dataset row
MODELING_START = "2010-04-07 00:00:00"

# Chronological train/validation/test split over modeling rows.
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Informer architecture (preserved from the uploaded notebook).
D_MODEL = 64
N_HEADS = 4
E_LAYERS = 2
D_LAYERS = 1
D_FF = 256
FACTOR = 5
DROPOUT = 0.01
ATTN = "prob"
EMBED = "fixed"
ACTIVATION = "gelu"
DISTIL = True
OUTPUT_ATTENTION = False

# Optimization.
BATCH_SIZE = 32
TRAIN_EPOCHS = 50
LEARNING_RATE = 2e-4
LOSS_FUNCTION = "mse"
USE_EARLY_STOPPING = True
PATIENCE = 5
MIN_DELTA = 1e-6
RESTORE_BEST_WEIGHTS = True
SEED = 42
NUM_WORKERS = 0
MAX_TRAIN_BATCHES = None  # Set to an integer only for a short smoke-test run.
MAX_EVAL_BATCHES = None
