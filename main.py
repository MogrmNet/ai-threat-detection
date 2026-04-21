import pandas as pd
from sklearn.ensemble import IsolationForest

# Sample dataset (network-like behavior)
data = {
    "packets": [10, 12, 11, 300, 9, 8, 400],
    "duration": [1, 1.2, 0.9, 10, 1.1, 0.8, 12]
}

df = pd.DataFrame(data)

# Train model
model = IsolationForest(contamination=0.2)
df["anomaly"] = model.fit_predict(df)

print(df)
