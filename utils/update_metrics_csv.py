import pandas as pd
from datetime import datetime
import shutil


def update_metrics_csv(csv_path: str, metrics: dict[str, float | int], backup: bool = False):
    
    if backup:
        backup_path = csv_path + ".backup"
        shutil.copy2(csv_path, backup_path)
        
    df = pd.read_csv(csv_path)
    metrics['datetime'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    m_df = pd.DataFrame(metrics, index=[0])
    m_df = m_df[['datetime'] + [col for col in m_df.columns if col != 'datetime']]
    
    df = pd.concat([df, m_df], ignore_index=True)
    df.sort_values(by='datetime', ascending=False, inplace=True)
    
    df.to_csv(csv_path, index=False)
    
    return df