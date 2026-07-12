# pypelite

Pypelite turns ordinary Python functions into resumable, inspectable pipeline
steps. It offers more structure than
[joblib](https://joblib.readthedocs.io/) without pulling your code into an
[Airflow](https://airflow.apache.org/)-style DAG, scheduler, or deployment.
The code is the pipeline.

```sh
pip install pypelite
```

[API reference](https://edgy-raven.github.io/pypelite/) ·
[Agent guide](AGENTS.md)

## Pipeline

Decorate the boundaries worth keeping, then call the functions normally:

```python
import pypelite

@pypelite.checkpoint()
def load_records(path):
    return read_records(path)

@pypelite.checkpoint()
def build_features(records_df):
    return make_model_features(records_df)

@pypelite.checkpoint()
def train_model(features_df):
    return fit_price_model(features_df)

with pypelite.pipeline("runs/price-model"):
    records_df = load_records("records.parquet")
    features_df = build_features(records_df)
    model = train_model(features_df)
```

Results live in the run archive, so a failed or interrupted program resumes
from completed steps. A later run can target only the work that should change:

```python
with pypelite.pipeline(
    "runs/experiment",
    refresh=["build_features"],
    skip=["train_model"],
    clean=["predict"],
    until="build_features",
):
    run_price_model()
```

## Archive Management

The archive is deliberately readable: each cached function owns a directory,
checkpoints keep one result, and stages keep one result per key.

```text
archive/
├── load_records/
│   ├── artifact.pkl
│   └── meta.json
└── load_price/
    ├── AAPL~7d3a4c1f2b80.pkl
    └── meta.json
```

Named archives let independent pipelines share durable inputs while keeping
their run-specific outputs separate:

```python
market = pypelite.Archive("archives/market")

@pypelite.stage(archive="market")
def load_price(symbol):
    return market_api.price(symbol)

with pypelite.pipeline("runs/model-a", archives={"market": market}):
    aapl = load_price("AAPL")

with pypelite.pipeline("runs/model-b", archives={"market": market}):
    aapl = load_price("AAPL")
```

Formats resolve from the named archive to the default archive, then pickle, so
specialized storage composes without making every pipeline configure it.

## Vectorization and Batching

Collection handling keeps the same per-item cache. Vectorize when the function
accepts one item but callers have many:

```python
@pypelite.stage(vectorize="symbol", workers=4)
def load_price(symbol):
    return market_api.price(symbol)
```

Use batching when the function itself accepts a collection:

```python
@pypelite.stage(key="symbol", batch="symbols", workers=4)
def load_prices(symbols):
    return [market_api.price(symbol) for symbol in symbols]
```

A batched checkpoint instead combines worker results into one artifact:

```python
@pypelite.checkpoint(batch="records", batch_size=50, workers=4)
def score_dataset(records):
    return model.score(records)
```

## Command-Line Controls

The supplied parser exposes the same run controls without duplicating CLI
plumbing in every project:

```python
args = pypelite.argument_parser().parse_args()

with pypelite.pipeline("runs/model", **vars(args)):
    run_model()
```
