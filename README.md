# Model A

Model A is a tiny SPY/IEX market-microstructure machine learning research MVP. It predicts short-term directional movement of the SPY ETF using five consecutive one-minute summaries of market microstructure features.

## Overview

- **Input**: Five consecutive one-minute summaries ending at decision time
- **Features per minute**: 4 features (queue imbalance, signed trade-volume imbalance, relative spread, midpoint return)
- **Total model inputs**: 5 × 4 = 20 features
- **Model architecture**: Frozen 20 → 8 tanh → 1 MLP
- **Prediction horizon**: 5 minutes
- **Target**: UP if future midpoint > decision-time midpoint, DOWN otherwise
- **Training data**: SPY on Alpaca IEX feed, 2026-08-18 through 2026-08-24
- **Observation window**: 09:35-12:00 America/New_York

**Important**: This is a research prototype, not automated trading software. Current results do not establish a profitable trading edge.

## Repository Structure

```
model-a/
├── src/
│   └── model_a/           # Core source code
│       ├── __init__.py
│       ├── cache_builder.py    # Build historical feature cache
│       ├── cache_loader.py     # Load cached features
│       ├── cli.py              # Command-line interface utilities
│       ├── data_loader.py      # Load market event data
│       ├── features.py         # Feature calculation logic
│       ├── model.py            # Model architecture
│       ├── prediction.py       # Prediction utilities
│       ├── preprocessing.py     # Data preprocessing and scaling
│       ├── sequences.py        # Sequence construction
│       ├── simple_cache.py     # Simplified cache builder
│       ├── simple_train.py     # Simplified training utilities
│       ├── trade_signing.py    # Trade signing logic
│       └── training.py         # Model training logic
├── tests/                    # Unit and integration tests
│   ├── test_features.py
│   ├── test_integration.py
│   ├── test_leakage.py
│   ├── test_model.py
│   ├── test_sequences.py
│   └── ...
├── docs/
│   └── MODEL_A_MVP_PRD.md     # Product requirements document
├── artifacts/                 # Generated artifacts (not in Git)
│   ├── model_a/
│   │   ├── models/            # Saved model files
│   │   ├── scalers/           # Saved scaler files
│   │   ├── predictions/       # Prediction outputs
│   │   └── minute_features.csv # Cached features
│   └── spy_iex_august/        # Historical market data
│       └── market_data/
│           └── events.csv      # Large historical events file
├── build_cache.py            # Build historical feature cache
├── run_real_mvp.py           # Train model and run demonstration
├── replay_model_a.py         # Replay on historical Alpaca data
├── live_model_a.py           # Live prediction using Alpaca API
├── predict_0955.py           # Historical prediction demonstration
├── pyproject.toml            # Project configuration
├── AGENTS.md                 # Development guidelines
└── README.md                 # This file
```

## Requirements

- **Python**: 3.9 or higher
- **Git**: For cloning the repository
- **Operating system**: macOS or Linux (Windows may work but is not tested)
- **Python packages** (installed automatically):
  - numpy>=1.24.0
  - pandas>=2.0.0
  - torch>=2.0.0
  - pytz>=2023.3
  - scikit-learn>=1.3.0
  - alpaca-trade-api>=3.0.0

### Alpaca Account Requirements

For live prediction and historical replay using Alpaca IEX data:
- Alpaca account with market data access
- Alpaca API key ID
- Alpaca API secret key

## Run Model A on a New Computer

For live prediction on a fresh Linux (or macOS) computer:

```bash
# Clone the repository
git clone https://github.com/sarthak-wadhawan/model-a.git
cd model-a

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -e .

# Configure Alpaca credentials
export APCA_API_KEY_ID="your_alpaca_key"
export APCA_API_SECRET_KEY="your_alpaca_secret"

# Run live prediction
python3 live_model_a.py --symbol SPY
```

**Important notes**:
- NO retraining is needed - the frozen model and scaler ship with the repository
- NO historical events.csv is needed - live prediction uses Alpaca IEX API directly
- The frozen model is in `models/model_a_v1/model.pt`
- The frozen scaler is in `models/model_a_v1/scaler.json`

**Example symbols**:
```bash
python3 live_model_a.py --symbol SPY   # Trained symbol
python3 live_model_a.py --symbol NVDA  # Experimental (out-of-distribution)
python3 live_model_a.py --symbol AAPL  # Experimental (out-of-distribution)
python3 live_model_a.py --symbol MSFT  # Experimental (out-of-distribution)
```

**Note**: Model A was trained on SPY data. Predictions on other symbols are experimental out-of-distribution tests.

## Fresh Installation

### macOS/Linux

```bash
# Clone the repository
git clone https://github.com/sarthak-wadhawan/model-a.git
cd model-a

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade packaging tools
python -m pip install --upgrade pip setuptools wheel

# Install the project in editable mode
pip install -e .

# (Optional) Install development dependencies for testing
pip install -e ".[dev]"
```

### Windows

```bash
# Clone the repository
git clone https://github.com/sarthak-wadhawan/model-a.git
cd model-a

# Create a virtual environment
python -m venv .venv
.venv\Scripts\activate

# Upgrade packaging tools
python -m pip install --upgrade pip setuptools wheel

# Install the project in editable mode
pip install -e .

# (Optional) Install development dependencies for testing
pip install -e ".[dev]"
```

## Alpaca Credentials

The project requires Alpaca API credentials for live prediction and historical replay. Set the following environment variables:

```bash
export APCA_API_KEY_ID="your_alpaca_key_id"
export APCA_API_SECRET_KEY="your_alpaca_secret_key"
```

**Important**: Never commit credentials to the repository. The `.gitignore` file ensures `.env` files are not tracked.

You can also create a `.env` file in the repository root:

```
APCA_API_KEY_ID=your_alpaca_key_id
APCA_API_SECRET_KEY=your_alpaca_secret_key
```

Then load it with:

```bash
# Using python-dotenv (if installed)
python -c "from dotenv import load_dotenv; load_dotenv()"
```

## Market Data

### Historical Data

The large historical events.csv file (approximately 4GB) is intentionally not included in the GitHub repository. To run the historical training pipeline:

1. Obtain the historical SPY IEX market data for August 2026
2. Place it at: `artifacts/spy_iex_august/market_data/events.csv`
3. The expected format is Alpaca IEX quote/trade events in CSV format

### Live Data

Live prediction and historical replay use the Alpaca IEX API directly and do not require local historical data files.

## Running Tests

After installation, run the test suite to verify the setup:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test modules
python -m pytest tests/test_features.py -v
python -m pytest tests/test_model.py -v
python -m pytest tests/test_sequences.py -v
```

**Note**: Tests that require the historical `events.csv` file will fail if the data is not present. Unit tests for features, model architecture, and sequences should pass without external data.

## Build Historical Feature Cache

If you have the historical events.csv file, build the feature cache:

```bash
python3 build_cache.py
```

This script:
- Reads from: `artifacts/spy_iex_august/market_data/events.csv`
- Outputs to: `artifacts/model_a/minute_features.csv`
- Processes dates: 2026-08-18, 2026-08-19, 2026-08-20, 2026-08-21, 2026-08-24, 2026-08-25
- Creates minute-level feature summaries for the observation window (09:35-12:00 ET)

## Train / Run Model A

Train the model and run the historical demonstration:

```bash
python3 run_real_mvp.py
```

This script:
- Loads cached features from: `artifacts/model_a/minute_features.csv`
- Trains on dates: 2026-08-18, 2026-08-19, 2026-08-20, 2026-08-21
- Validates on: 2026-08-24
- Demonstrates prediction on: 2026-08-25 09:55 ET
- Saves model to: `artifacts/model_a/models/model.pt`
- Saves scaler to: `artifacts/model_a/scalers/scaler.json`
- Saves prediction to: `artifacts/model_a/predictions/demo_prediction.json`

## Historical Replay

Replay the model on historical Alpaca IEX data for any symbol and date:

```bash
python3 replay_model_a.py \
  --symbol NVDA \
  --date 2026-08-25 \
  --decision-time 13:00 \
  --timezone America/Chicago
```

Arguments:
- `--symbol`: Stock/ETF ticker symbol (e.g., NVDA, SPY, AAPL)
- `--date`: Date in YYYY-MM-DD format
- `--decision-time`: Decision time in HH:MM format
- `--timezone`: Timezone (default: America/Chicago)

**Important**: Non-SPY symbols are out-of-distribution experiments because the frozen model was trained on SPY data.

## Live Prediction

Run live prediction using the frozen model:

```bash
python3 live_model_a.py --symbol NVDA
```

Examples:
```bash
python3 live_model_a.py --symbol SPY
python3 live_model_a.py --symbol AAPL
python3 live_model_a.py --symbol MSFT
```

This script:
- Uses actual Alpaca IEX quotes/trades
- Selects the latest completed usable minute
- Predicts T+5 (5 minutes ahead)
- Saves Markdown output to: `artifacts/model_a/predictions/`
- Does not retrain the model

## Prediction Outputs

Prediction reports are saved in `artifacts/model_a/predictions/`:

- `latest_<SYMBOL>_live.md`: Most recent prediction for the symbol
- `<SYMBOL>_<timestamp>_live.md`: Timestamped archival copies

Reports include:
- Decision time and target time in multiple timezones
- Input minute summaries with feature values
- Raw 20-feature vector
- Model prediction (P(UP), P(DOWN), direction)
- Whether the symbol/time is out-of-distribution

## Reproducing Historical MVP

To reproduce the historical MVP demonstration:

1. Place historical events.csv at: `artifacts/spy_iex_august/market_data/events.csv`
2. Build the feature cache: `python3 build_cache.py`
3. Train the model: `python3 run_real_mvp.py`

Expected directory structure:
```
artifacts/
└── spy_iex_august/
    └── market_data/
        └── events.csv
```

The historical demonstration will:
- Train on 2026-08-18 through 2026-08-21
- Validate on 2026-08-24
- Demonstrate prediction on 2026-08-25 09:55 ET

## Known Limitations

- **Training data**: Model was trained only on SPY data
- **Data source**: IEX is not consolidated full-market data
- **Dataset size**: Very small development dataset (5 days)
- **Validation**: Historical validation is limited
- **Out-of-distribution**: Predictions on other stocks are out-of-distribution experiments
- **Performance**: Current results do not establish a profitable trading edge
- **Scope**: Research prototype only, not production trading software

## Troubleshooting

### Wrong Python version

**Problem**: `Package 'model-a' requires a different Python: 3.x not in '>=3.9'`

**Solution**: Install Python 3.9 or higher. On macOS, use Homebrew:
```bash
brew install python@3.11
python3.11 -m venv .venv
```

### Missing virtual environment

**Problem**: Module import errors or Python not found

**Solution**: Ensure the virtual environment is activated:
```bash
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows
```

### Missing Alpaca credentials

**Problem**: `Alpaca API credentials not found in environment`

**Solution**: Set the required environment variables:
```bash
export APCA_API_KEY_ID="your_key"
export APCA_API_SECRET_KEY="your_secret"
```

### Missing events.csv

**Problem**: `FileNotFoundError: [Errno 2] No such file or directory: 'artifacts/spy_iex_august/market_data/events.csv'`

**Solution**: The historical data file is not in the repository. Either:
- Obtain and place the historical data at the expected path
- Use live prediction or historical replay instead (which use Alpaca API directly)

### Module import errors

**Problem**: `ModuleNotFoundError: No module named 'model_a'`

**Solution**: Ensure the project is installed in editable mode:
```bash
pip install -e .
```

If using scripts directly, ensure you're in the repository root directory.

### Incorrect working directory

**Problem**: Scripts fail to find relative paths

**Solution**: Always run scripts from the repository root:
```bash
cd /path/to/model-a
python3 build_cache.py
```

### No IEX data returned

**Problem**: `ERROR: No usable Alpaca IEX data found for SYMBOL`

**Solution**:
- Verify Alpaca credentials are correct
- Check that the market is open for the symbol
- Try a different symbol or time
- Verify your Alpaca account has IEX data access

### Market closed / unavailable recent minute

**Problem**: Live prediction fails with no available data

**Solution**: The script searches for the most recent completed minute with data. If the market is closed or no recent data is available, prediction will fail. Try during market hours.

### Git clone/setup problems

**Problem**: `fatal: repository 'https://github.com/sarthak-wadhawan/model-a.git' not found`

**Solution**: Verify the repository URL is correct and you have internet access. Check that the repository is public or you have authentication if it's private.

## Development

See `AGENTS.md` for development guidelines and `docs/MODEL_A_MVP_PRD.md` for the product requirements document.

## License

[Add your license here]
