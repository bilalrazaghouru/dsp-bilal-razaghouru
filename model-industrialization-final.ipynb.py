import pandas as pd
from house_prices.train import build_model

train_df = pd.read_csv('../data/train.csv')
metrics = build_model(train_df)
print(metrics)

