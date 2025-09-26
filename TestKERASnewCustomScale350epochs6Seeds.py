# ============================================
# Heart Failure (Binary) — Keras vs DDESONN parity
# 50-seed run with accuracy table at end
# ============================================

import os, random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras import regularizers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ---------- Config ----------
USE_TIME_SPLIT = True
DATA_PATH = r"C:/Users/wfky1/Downloads/heart_failure_clinical_records.csv"

LR = 0.125
L1_LAMBDA = 0.00028
EPOCHS = 360
BATCH_SIZE = 16
BN_MOMENTUM = 0.9
BN_EPS = 1e-6
BN_BETA_INIT = tf.keras.initializers.Constant(0.6)
BN_GAMMA_INIT = tf.keras.initializers.Constant(0.6)
CUSTOM_SCALE = 1.04349

feature_cols = [
    "age","anaemia","creatinine_phosphokinase","diabetes","ejection_fraction",
    "high_blood_pressure","platelets","serum_creatinine","serum_sodium",
    "sex","smoking","time"
]
target_col = "DEATH_EVENT"

# ---------- Load data ----------
df = pd.read_csv(DATA_PATH).dropna()
X_all = df[feature_cols].values
y_all = df[target_col].values.astype(int)

# ---------- Run multiple seeds ----------
results = []

for seed in range(1, 1000):   # 50 seeds
    print(f"\n=== Seed {seed} ===")

    # Reproducibility
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Split
    if USE_TIME_SPLIT:
        n = len(X_all)
        n_train = int(0.70 * n)
        n_val   = int(0.15 * n)

        X_train, y_train = X_all[:n_train], y_all[:n_train]
        X_val,   y_val   = X_all[n_train:n_train+n_val], y_all[n_train:n_train+n_val]
        X_test,  y_test  = X_all[n_train+n_val:], y_all[n_train+n_val:]
    else:
        X_tr, X_hold, y_tr, y_hold = train_test_split(
            X_all, y_all, test_size=0.30, stratify=y_all, random_state=seed
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_hold, y_hold, test_size=0.50, stratify=y_hold, random_state=seed
        )
        X_train, y_train = X_tr, y_tr

    # Scale
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # ---------- Build model (no input_shape in Dense) ----------
    KERNEL_INIT = VarianceScaling(
        scale=CUSTOM_SCALE, mode='fan_in',
        distribution='truncated_normal', seed=seed  # per-seed init
    )

    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(64, activation="relu",
              kernel_initializer=KERNEL_INIT,
              kernel_regularizer=regularizers.l1(L1_LAMBDA)),
        BatchNormalization(momentum=BN_MOMENTUM, epsilon=BN_EPS,
                           beta_initializer=BN_BETA_INIT,
                           gamma_initializer=BN_GAMMA_INIT),
        Dropout(0.10),

        Dense(32, activation="relu",
              kernel_initializer=KERNEL_INIT,
              kernel_regularizer=regularizers.l1(L1_LAMBDA)),
        BatchNormalization(momentum=BN_MOMENTUM, epsilon=BN_EPS,
                           beta_initializer=BN_BETA_INIT,
                           gamma_initializer=BN_GAMMA_INIT),
        Dropout(0.00),

        Dense(1, activation="sigmoid",
              kernel_initializer=KERNEL_INIT,
              kernel_regularizer=regularizers.l1(L1_LAMBDA))
    ])

    model.compile(optimizer=Adagrad(learning_rate=LR),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])

    # Train
    model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        verbose=0
    )

    # Evaluate
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    val_loss,   val_acc   = model.evaluate(X_val,   y_val,   verbose=0)
    test_loss,  test_acc  = model.evaluate(X_test,  y_test,  verbose=0)

    results.append({
        "seed": seed,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "test_acc": test_acc
    })

# ---------- Table at end ----------
acc_table = pd.DataFrame(results)
print("\n=== Accuracy table over 50 seeds ===")
print(acc_table.to_string(index=False))

print("\n=== Summary ===")
print(acc_table.describe().round(4))
