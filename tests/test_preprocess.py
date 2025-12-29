import sys
from pathlib import Path

# Allow tests to import from src
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from data_loader import DataLoader
from pipelines.preprocess_pipeline import PreprocessPipeline

loader = DataLoader("data/01-raw")
data = loader.load_main_dataset()   # <-- variable name is data

pipeline = PreprocessPipeline("data/02-preprocessed")
X, y = pipeline.run(data)           # <-- pass data here



print(X.shape)
print(y.shape)
print("Total missing:", X.isnull().sum().sum())

# Save preprocessed data
assert Path("data/02-preprocessed/X_preprocessed.csv").exists()
assert Path("data/02-preprocessed/y_preprocessed.csv").exists()

print("Columns:", X.columns.tolist())
print("Total columns:", X.shape[1])