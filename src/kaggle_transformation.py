import pandas as pd

df = pd.read_csv("submissions/test_predictions.csv") # Change this path if needed!
df.insert(0, 'id', range(len(df)))
df.to_csv("submissions/test_predictions_kaggle_fixed.csv", index=False)

print("Fixed! Now upload 'test_predictions_kaggle_fixed.csv' to Kaggle.")