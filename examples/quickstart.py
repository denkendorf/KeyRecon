import pandas as pd
from keyrecon import KeyRecon

records = pd.read_csv("examples/synthetic_records.csv", keep_default_na=False)

model = KeyRecon(language="en", mode="reference")
predictions = model.fit_reconstruct_missing(records)

print(model.fit_summary_)
print(predictions[["record_id", "canonical_key", "cks_score", "rank"]])
