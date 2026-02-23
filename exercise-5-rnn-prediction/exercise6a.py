import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import lightning.pytorch as pl

from pytorch_forecasting import TimeSeriesDataSet, Baseline, DeepAR
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.data.examples import generate_ar_data
from pytorch_forecasting.metrics import MAE, MultivariateNormalDistributionLoss

from lightning.pytorch.callbacks import EarlyStopping

pl.seed_everything(42)

data = generate_ar_data(seasonality=10.0, timesteps=400, n_series=100, seed=42)
data["static"] = 2
data["date"] = pd.Timestamp("2020-01-01") + pd.to_timedelta(data.time_idx, "D")
data = data.astype(dict(series=str))

max_encoder_length = 60
max_prediction_length = 20

training_cutoff = data["time_idx"].max() - max_prediction_length

training = TimeSeriesDataSet(
    data[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="value",
    categorical_encoders={"series": NaNLabelEncoder().fit(data.series)},
    group_ids=["series"],
    static_categoricals=["series"],
    time_varying_unknown_reals=["value"],
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
)

validation = TimeSeriesDataSet.from_dataset(
    training, data, min_prediction_idx=training_cutoff + 1)

batch_size = 128
train_dataloader = training.to_dataloader(train=True,
                                          batch_size=batch_size,
                                          num_workers=0,
                                          batch_sampler="synchronized")
val_dataloader = validation.to_dataloader(train=False,
                                          batch_size=batch_size,
                                          num_workers=0,
                                          batch_sampler="synchronized")

_ = Baseline().predict(val_dataloader,
                       trainer_kwargs=dict(accelerator="cpu"),
                       return_y=True)

trainer = pl.Trainer(accelerator="cpu",
                     gradient_clip_val=1e-1,
                     max_epochs=30,
                     enable_checkpointing=True)
net = DeepAR.from_dataset(
    training,
    learning_rate=1e-2,
    hidden_size=30,
    rnn_layers=2,
    optimizer="Adam",
)

early_stop_callback = EarlyStopping(monitor="val_loss",
                                    min_delta=1e-4,
                                    patience=10,
                                    mode="min",
                                    verbose=False)
trainer = pl.Trainer(
    max_epochs=30,
    accelerator="cpu",
    enable_model_summary=True,
    gradient_clip_val=0.1,
    callbacks=[early_stop_callback],
    limit_train_batches=50,
    enable_checkpointing=True,
)

trainer.fit(net, train_dataloader, val_dataloader)
best_model_path = trainer.checkpoint_callback.best_model_path
best_model = DeepAR.load_from_checkpoint(best_model_path)

predictions = best_model.predict(val_dataloader,
                                 trainer_kwargs=dict(accelerator="cpu"),
                                 return_y=True)

pred_out = predictions.output.detach().cpu().numpy()
true_y = predictions.y[0].detach().cpu().numpy()

mse_per_sample = np.mean((pred_out - true_y)**2, axis=1)

raw = best_model.predict(val_dataloader,
                         mode="raw",
                         return_x=True,
                         trainer_kwargs=dict(accelerator="cpu"))
index_info = validation.x_to_index(raw.x)
series_ids = index_info["series"].to_numpy()
prediction_start_idx = index_info["time_idx"].to_numpy()

mse_df = pd.DataFrame({
    "series": series_ids,
    "prediction_start_idx": prediction_start_idx,
    "mse": mse_per_sample
})
mse_df.to_csv("deepar_mse_per_sample.csv", index=False)

raw_predictions = best_model.predict(val_dataloader,
                                     mode="raw",
                                     return_x=True,
                                     n_samples=100,
                                     trainer_kwargs=dict(accelerator="cpu"))

series = validation.x_to_index(raw_predictions.x)["series"]
save_count = min(20, len(series))
for idx in range(save_count):
    fig = best_model.plot_prediction(raw_predictions.x,
                                     raw_predictions.output,
                                     idx=idx,
                                     add_loss_to_title=True)
    plt.suptitle(f"Series: {series.iloc[idx]}")
    plt.tight_layout()
    plt.savefig(f"deepar_pred_{idx:02d}.png")
    plt.close(fig)

avg_mse = mse_per_sample.mean()
print(f"average_mse_validation: {avg_mse:.6f}")
print("mse_per_sample saved to deepar_mse_per_sample.csv")
print(f"saved {save_count} prediction plots deepar_pred_##.png")
