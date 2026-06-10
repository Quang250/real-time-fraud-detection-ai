# Model Artifacts

This folder stores trained model artifacts locally.

Expected local files after training:

```text
models/
|-- final_fraud_model.pkl
|-- best_threshold.pkl
`-- feature_list.pkl
```

These artifacts are generated outputs and are not committed to GitHub by
default.

To reproduce them, run the modeling workflow:

```text
notebooks/03_modeling.ipynb
```

or use the training script if you want to refactor the notebook workflow into a
scripted run:

```bash
python -m src.train_model
```
