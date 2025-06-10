import pandas as pd


def update_metrics_csv(csv_path: str, metrics: dict[str, float | int]):
    df = pd.read_csv(csv_path)
    df = df.append(metrics, ignore_index=True)
    df.to_csv(csv_path, index=False)
