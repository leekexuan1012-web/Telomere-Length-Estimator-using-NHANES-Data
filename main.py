import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")

# --- Step 1: load each file ---
telo = pd.read_sas(DATA_DIR / "TELO_A.xpt", format="xport")
demo = pd.read_sas(DATA_DIR / "DEMO.xpt", format="xport")
bmx  = pd.read_sas(DATA_DIR / "BMX.xpt", format="xport")
smq  = pd.read_sas(DATA_DIR / "SMQ.xpt", format="xport")
paq  = pd.read_sas(DATA_DIR / "PAQ.xpt", format="xport")
alq  = pd.read_sas(DATA_DIR / "ALQ.xpt", format="xport")

# --- Step 2: merge them all on SEQN ---
df = telo.merge(demo, on="SEQN", how="left")
df = df.merge(bmx, on="SEQN", how="left")
df = df.merge(smq, on="SEQN", how="left")
df = df.merge(paq, on="SEQN", how="left")
df = df.merge(alq, on="SEQN", how="left")

# --- Step 3: select just age + target, and clean ---
cols = ["SEQN", "TELOMEAN", "RIDAGEYR"]
df_model = df[cols].copy()
df_model = df_model.dropna()

print("Rows after cleaning:", df_model.shape)
print(df_model.head())
# --- Step 4: define X (input) and y (target) ---
X = df_model[["RIDAGEYR"]]   # double brackets = keep it as a DataFrame, not a Series
y = df_model["TELOMEAN"]
# --- Step 5: split into train/test ---
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Training rows:", X_train.shape[0])
print("Testing rows:", X_test.shape[0])