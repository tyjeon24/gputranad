# GPUTranAD
PyTorch Lightning wrapper library for TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series.

GPU supported.

# How to use
```python
from gputranad import setup, TranAD, TranADLitModel

train_dataloader = setup("dataset/P-1_train.npy")

batch_sample = next(iter(train_dataloader))[0]
n_features = batch_sample.shape[-1]
model = TranAD(n_feats=n_features).double()
lit_model = TranADLitModel(model)

trainer = L.Trainer(max_epochs=NUM_EPOCHS)
trainer.fit(lit_model, train_dataloader)
```
