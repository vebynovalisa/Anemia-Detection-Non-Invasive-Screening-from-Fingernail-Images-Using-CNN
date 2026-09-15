import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import ast, warnings
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_score, recall_score,
    f1_score, accuracy_score
)

import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks
from tensorflow.keras.applications import DenseNet169, InceptionV3, Xception
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


# Konfigurasi
DATASET1_DIR  = "dataset1"
DATASET2_DIR  = "dataset2"
OUTPUT_DIR    = "output_results"

IMG_SIZE      = (224, 224)
BATCH_SIZE    = 16
EPOCHS        = 50
LEARNING_RATE = 1e-4
SEED          = 42
HB_THRESHOLD  = 11.0

TRAIN_RATIO   = 0.70
VAL_RATIO     = 0.15
TEST_RATIO    = 0.15

CLASS_NAMES   = ["anemic", "non_anemic"]
NUM_CLASSES   = 2

os.makedirs(OUTPUT_DIR, exist_ok=True)
tf.random.set_seed(SEED)
np.random.seed(SEED)


# 1. Baca CSV
def read_csv_auto(csv_path):
    for sep in [",", ";", "\t"]:
        try:
            df = pd.read_csv(csv_path, sep=sep, encoding="utf-8")
            if len(df.columns) >= 2:
                df.columns = [c.strip() for c in df.columns]
                print(f"-> CSV separator: '{sep}' | Kolom: {list(df.columns)}")
                return df
        except Exception:
            continue
    try:
        df = pd.read_csv(csv_path, sep=",", encoding="latin-1")
        df.columns = [c.strip() for c in df.columns]
        return df
    except Exception as e:
        print(f"Gagal baca CSV: {e}")
        return pd.DataFrame()


# 2. Preprocessing Gambar
def illumination_normalization(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab_norm = cv2.merge([clahe.apply(l), a, b])
    return cv2.cvtColor(lab_norm, cv2.COLOR_LAB2RGB)


def extract_nail_by_bbox(image, bbox_str):
    try:
        bbox = ast.literal_eval(str(bbox_str))
        if isinstance(bbox[0], list):
            bbox = bbox[0]
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        h, w = image.shape[:2]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        if (x2 - x1) < 10 or (y2 - y1) < 10:
            return image
        return cv2.resize(image[y1:y2, x1:x2], IMG_SIZE)
    except Exception:
        return image


def extract_nail_ycbcr(image):
    ycbcr = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    mask = cv2.inRange(ycbcr,
                       np.array([0, 133, 77],   dtype=np.uint8),
                       np.array([255, 173, 127], dtype=np.uint8))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    if np.sum(mask) < (image.shape[0] * image.shape[1] * 0.05):
        return image
    return cv2.bitwise_and(image, image, mask=mask)


def preprocess_image(img_path, bbox_str=None):
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = illumination_normalization(img)
    if bbox_str is not None and str(bbox_str).strip().lower() not in ["none", "nan", ""]:
        img = extract_nail_by_bbox(img, bbox_str)
    else:
        img = extract_nail_ycbcr(img)
    if img.shape[:2] != IMG_SIZE:
        img = cv2.resize(img, IMG_SIZE)
    if len(img.shape) == 2:
        img = np.stack([img]*3, axis=-1)
    return img.astype(np.float32) / 255.0


# 3. Kumpulkan Path Dataset
def collect_dataset1_paths(dataset_dir):
    paths, labels, bboxes = [], [], []
    dataset_path = Path(dataset_dir)

    print("\n[Dataset 1] scanning...")
    for label_idx, class_name in enumerate(CLASS_NAMES):
        class_path = dataset_path / class_name
        if not class_path.exists():
            print(f"Folder '{class_name}' tidak ditemukan!")
            continue
        all_files = list(class_path.iterdir())
        img_files = [p for p in all_files if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        img_files = sorted(list(set(img_files)))
        print(f"-> {class_name}: {len(img_files)} gambar")
        for p in img_files:
            paths.append(str(p))
            labels.append(label_idx)
            bboxes.append(None)

    print(f"Dataset 1: {len(paths)} file")
    return paths, labels, bboxes


def collect_dataset2_paths(dataset_dir):
    paths, labels, bboxes = [], [], []
    dataset_path = Path(dataset_dir)
    photo_dir    = dataset_path / "photo"
    csv_path     = dataset_path / "metadata.csv"

    if not csv_path.exists():
        print("metadata.csv tidak ditemukan!")
        return paths, labels, bboxes
    if not photo_dir.exists():
        print("Folder 'photo' tidak ditemukan!")
        return paths, labels, bboxes

    print("\n[Dataset 2] scanning...")
    df = read_csv_auto(csv_path)
    if df.empty:
        return paths, labels, bboxes

    hb_col = None
    for col in df.columns:
        if "HB" in col.upper() and "LEVEL" in col.upper():
            hb_col = col
            break
        if col.upper() in ["HB", "HB_LEVEL", "HEMOGLOBIN", "HGB"]:
            hb_col = col
            break

    if hb_col is None:
        print(f"Kolom HB tidak ditemukan! Kolom yang ada: {list(df.columns)}")
        return paths, labels, bboxes

    print(f"-> Kolom HB: '{hb_col}'")
    df = df.dropna(subset=[hb_col])
    df[hb_col] = pd.to_numeric(df[hb_col], errors="coerce")
    df = df.dropna(subset=[hb_col])

    if df[hb_col].mean() > 30:
        print("-> Nilai Hb dalam g/L terdeteksi, otomatis dibagi 10 ke g/dL...")
        df[hb_col] = df[hb_col] / 10.0

    df["label"] = (df[hb_col] >= HB_THRESHOLD).astype(int)
    print(f"-> anemic     (Hb < {HB_THRESHOLD}): {(df['label']==0).sum()}")
    print(f"-> non_anemic (Hb >= {HB_THRESHOLD}): {(df['label']==1).sum()}")

    bbox_col = None
    for col in df.columns:
        if "NAIL" in col.upper() and ("BOX" in col.upper() or "BOU" in col.upper()):
            bbox_col = col
            break
    if bbox_col:
        print(f"-> Nail bbox kolom: '{bbox_col}'")

    skipped = 0
    for _, row in df.iterrows():
        pid = str(row["PATIENT_ID"]).strip()
        img_path = None
        for ext in [".jpg", ".jpeg", ".png", ".JPG", ".PNG"]:
            c = photo_dir / f"{pid}{ext}"
            if c.exists():
                img_path = c
                break
        if img_path is None:
            skipped += 1
            continue
        paths.append(str(img_path))
        labels.append(int(row["label"]))
        bboxes.append(str(row[bbox_col]) if bbox_col else None)

    if skipped > 0:
        print(f"{skipped} foto tidak ditemukan (skip)")
    print(f"Dataset 2: {len(paths)} file")
    return paths, labels, bboxes


# 4. Custom Data Generator
class AnemiaDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, paths, labels, bboxes, batch_size=16, augment=False, shuffle=True):
        self.paths      = paths
        self.labels     = labels
        self.bboxes     = bboxes
        self.batch_size = batch_size
        self.augment    = augment
        self.shuffle    = shuffle
        self.indices    = np.arange(len(self.paths))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return max(1, len(self.paths) // self.batch_size)

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        X, y = [], []
        for i in batch_idx:
            img = preprocess_image(self.paths[i], self.bboxes[i])
            if img is None:
                img = np.zeros((*IMG_SIZE, 3), dtype=np.float32)
            if self.augment:
                img = self._augment(img)
            X.append(img)
            y.append(self.labels[i])
        return np.array(X, dtype=np.float32), to_categorical(y, NUM_CLASSES)

    def _augment(self, img):
        if np.random.rand() > 0.5:
            img = img[:, ::-1, :]
        factor = np.random.uniform(0.8, 1.2)
        img = np.clip(img * factor, 0, 1)
        if np.random.rand() > 0.5:
            angle = np.random.uniform(-20, 20)
            M = cv2.getRotationMatrix2D((IMG_SIZE[0]//2, IMG_SIZE[1]//2), angle, 1)
            img = cv2.warpAffine(img, M, IMG_SIZE)
        return img

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


# 5. Build Model
def build_model(model_name):
    input_shape = (*IMG_SIZE, 3)
    base_kwargs = dict(weights="imagenet", include_top=False, input_shape=input_shape)

    if   model_name == "DenseNet169": base = DenseNet169(**base_kwargs)
    elif model_name == "InceptionV3":  base = InceptionV3(**base_kwargs)
    elif model_name == "Xception":     base = Xception(**base_kwargs)
    else: raise ValueError(f"Model '{model_name}' tidak dikenal.")

    base.trainable = False
    inp = tf.keras.Input(shape=input_shape)
    x   = base(inp, training=False)
    x   = layers.GlobalAveragePooling2D()(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dense(256, activation="relu")(x)
    x   = layers.Dropout(0.5)(x)
    x   = layers.Dense(128, activation="relu")(x)
    x   = layers.Dropout(0.3)(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    return Model(inp, out, name=model_name), base


def unfreeze_top(base, n=30):
    base.trainable = True
    for layer in base.layers[:-n]:
        layer.trainable = False


# 6. Training
def train_model(name, train_gen, val_gen, val_paths, val_labels, val_bboxes):
    print(f"\n[ TRAINING: {name} ]")

    model, base  = build_model(name)
    weights_path = os.path.join(OUTPUT_DIR, f"{name}_best.weights.h5")

    print("-> Loading validation data...")
    X_val, y_val = [], []
    for p, l, b in zip(val_paths, val_labels, val_bboxes):
        img = preprocess_image(p, b)
        if img is None:
            img = np.zeros((*IMG_SIZE, 3), dtype=np.float32)
        X_val.append(img)
        y_val.append(l)
    X_val   = np.array(X_val, dtype=np.float32)
    y_val_c = to_categorical(y_val, NUM_CLASSES)
    print(f"-> Validation: {len(X_val)} gambar")

    cb = [
        callbacks.EarlyStopping(monitor="val_loss", patience=10,
                                restore_best_weights=False, verbose=1),
        callbacks.ModelCheckpoint(weights_path, monitor="val_accuracy",
                                  save_best_only=True, save_weights_only=True,
                                  verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                    patience=5, min_lr=1e-7, verbose=1),
    ]

    print("\nFase 1: Training classification head...")
    model.compile(optimizer=Adam(LEARNING_RATE),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    h1 = model.fit(train_gen, epochs=20,
                   validation_data=(X_val, y_val_c), callbacks=cb, verbose=1)

    if os.path.exists(weights_path):
        model.load_weights(weights_path)

    print("\nFase 2: Fine-tuning 30 layer teratas...")
    unfreeze_top(base, n=30)
    model.compile(optimizer=Adam(LEARNING_RATE / 10),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    h2 = model.fit(train_gen, epochs=EPOCHS,
                   validation_data=(X_val, y_val_c), callbacks=cb, verbose=1)

    if os.path.exists(weights_path):
        model.load_weights(weights_path)

    history = {k: h1.history[k] + h2.history[k] for k in h1.history}
    return model, history, X_val, np.array(y_val)


# 7. Evaluasi
def evaluate_model(model, name, test_paths, test_labels, test_bboxes):
    print(f"\n[ EVALUASI: {name} ]")

    print("-> Loading test data...")
    X_te, y_te = [], []
    for p, l, b in zip(test_paths, test_labels, test_bboxes):
        img = preprocess_image(p, b)
        if img is None:
            img = np.zeros((*IMG_SIZE, 3), dtype=np.float32)
        X_te.append(img)
        y_te.append(l)
    X_te = np.array(X_te, dtype=np.float32)
    y_te = np.array(y_te)

    y_prob = model.predict(X_te, batch_size=BATCH_SIZE, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    acc  = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_te, y_pred,    average="weighted", zero_division=0)
    f1   = f1_score(y_te, y_pred,        average="weighted", zero_division=0)
    fpr, tpr, _ = roc_curve(y_te, y_prob[:, 0], pos_label=0)
    roc_auc     = auc(fpr, tpr)

    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  AUC-ROC   : {roc_auc:.4f}")
    print(f"\n{classification_report(y_te, y_pred, target_names=CLASS_NAMES, digits=4)}")

    return {"accuracy": acc, "precision": prec, "recall": rec,
            "f1": f1, "auc_roc": roc_auc, "fpr": fpr, "tpr": tpr,
            "y_pred": y_pred, "y_prob": y_prob, "y_true": y_te}


# 8. Visualisasi
def plot_dataset_distribution(labels):
    counts = [np.sum(np.array(labels) == i) for i in range(NUM_CLASSES)]
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = ["#E53935", "#43A047"]
    wedges, texts, autotexts = ax.pie(counts, labels=CLASS_NAMES, colors=colors,
                                       autopct="%1.1f%%", startangle=90)
    for i, (c, n) in enumerate(zip(counts, CLASS_NAMES)):
        texts[i].set_text(f"{n}\n({c} gambar)")
    ax.set_title("Distribusi Dataset Gabungan", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "dataset_distribution.png"),
                dpi=150, bbox_inches="tight")
    plt.close()


def plot_training_history(history, name):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Training History - {name}", fontsize=13, fontweight="bold")
    ep = range(1, len(history["accuracy"]) + 1)
    axes[0].plot(ep, history["accuracy"],     label="Train", color="#1565C0")
    axes[0].plot(ep, history["val_accuracy"], label="Val",   color="#E53935")
    axes[0].set_title("Accuracy"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(ep, history["loss"],     label="Train", color="#1565C0")
    axes[1].plot(ep, history["val_loss"], label="Val",   color="#E53935")
    axes[1].set_title("Loss"); axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{name}_history.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  History: {name}_history.png")


def plot_confusion_matrix(y_true, y_pred, name):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                annot_kws={"size": 14})
    ax.set_title(f"Confusion Matrix - {name}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{name}_confusion_matrix.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix: {name}_confusion_matrix.png")


def plot_roc_all(all_metrics):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"DenseNet169": "#E91E63", "InceptionV3": "#1565C0", "Xception": "#2E7D32"}
    for name, m in all_metrics.items():
        ax.plot(m["fpr"], m["tpr"],
                label=f"{name}  (AUC = {m['auc_roc']:.4f})",
                color=colors.get(name, "gray"), linewidth=2)
    ax.plot([0,1],[0,1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "roc_curve_comparison.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  ROC: roc_curve_comparison.png")


def plot_metrics_bar(all_metrics):
    metric_keys   = ["accuracy", "precision", "recall", "f1", "auc_roc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]
    model_names   = list(all_metrics.keys())
    colors = ["#E91E63", "#1565C0", "#2E7D32"]
    x = np.arange(len(metric_labels)); w = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (mname, col) in enumerate(zip(model_names, colors)):
        vals = [all_metrics[mname][k] for k in metric_keys]
        bars = ax.bar(x + i*w, vals, w, label=mname, color=col, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + w/2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x + w); ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0, 1.15); ax.set_ylabel("Score"); ax.grid(axis="y", alpha=0.3)
    ax.set_title("Model Performance Comparison", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "metrics_comparison.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Metrics: metrics_comparison.png")


def save_summary(all_metrics, l1, l2):
    path = os.path.join(OUTPUT_DIR, "results_summary.txt")
    with open(path, "w") as f:
        f.write("LAPORAN HASIL EVALUASI ANEMIA DETECTION\n\n")
        f.write("DETAIL DATASET:\n")
        f.write("Dataset 1 (Folder Mendeley/Ghana)\n")
        f.write(f"  Total Data : {len(l1)} gambar\n")
        f.write(f"  Anemic     : {l1.count(0)} gambar\n")
        f.write(f"  Non-Anemic : {l1.count(1)} gambar\n\n")
        f.write("Dataset 2 (Folder Photo + Metadata)\n")
        f.write(f"  Total Data : {len(l2)} gambar\n")
        f.write(f"  Ambang HB  : < {HB_THRESHOLD} g/dL (Dikategorikan Anemic)\n")
        f.write(f"  Anemic     : {l2.count(0)} gambar\n")
        f.write(f"  Non-Anemic : {l2.count(1)} gambar\n\n")
        f.write(f"Total Keseluruhan Data : {len(l1) + len(l2)} gambar\n\n")
        f.write("PERFORMA MODEL:\n")
        f.write(f"{'Model':<15} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'AUC-ROC':>10}\n")
        for name, m in all_metrics.items():
            f.write(f"{name:<15} {m['accuracy']:>10.4f} {m['precision']:>10.4f} "
                    f"{m['recall']:>10.4f} {m['f1']:>10.4f} {m['auc_roc']:>10.4f}\n")
        best = max(all_metrics, key=lambda x: all_metrics[x]["accuracy"])
        f.write(f"\nKesimpulan:\nModel terbaik adalah {best} "
                f"dengan Akurasi {all_metrics[best]['accuracy']*100:.2f}%\n")
    print("  Summary: results_summary.txt")


# 9. Main
def main():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    p1, l1, b1 = collect_dataset1_paths(DATASET1_DIR)
    p2, l2, b2 = collect_dataset2_paths(DATASET2_DIR)

    all_paths  = p1 + p2
    all_labels = l1 + l2
    all_bboxes = b1 + b2

    if len(all_paths) == 0:
        print("\nTidak ada data. Cek path dataset!")
        return

    total = len(all_paths)
    print(f"\nTOTAL DATA GABUNGAN: {total} gambar")
    print(f"Anemic     : {sum(1 for l in all_labels if l==0)}")
    print(f"Non-Anemic : {sum(1 for l in all_labels if l==1)}\n")

    plot_dataset_distribution(all_labels)

    indices = np.arange(total)
    idx_tr, idx_tmp = train_test_split(
        indices, test_size=(VAL_RATIO + TEST_RATIO),
        stratify=all_labels, random_state=SEED)
    idx_val, idx_te = train_test_split(
        idx_tmp, test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
        stratify=[all_labels[i] for i in idx_tmp], random_state=SEED)

    tr_p  = [all_paths[i]  for i in idx_tr]
    tr_l  = [all_labels[i] for i in idx_tr]
    tr_b  = [all_bboxes[i] for i in idx_tr]
    val_p = [all_paths[i]  for i in idx_val]
    val_l = [all_labels[i] for i in idx_val]
    val_b = [all_bboxes[i] for i in idx_val]
    te_p  = [all_paths[i]  for i in idx_te]
    te_l  = [all_labels[i] for i in idx_te]
    te_b  = [all_bboxes[i] for i in idx_te]

    print(f"Split: Train={len(tr_p)} | Val={len(val_p)} | Test={len(te_p)}")

    train_gen = AnemiaDataGenerator(tr_p, tr_l, tr_b,
                                    batch_size=BATCH_SIZE,
                                    augment=True, shuffle=True)

    all_metrics = {}
    for model_name in ["InceptionV3", "Xception", "DenseNet169"]:
        model, history, _, _ = train_model(
            model_name, train_gen, None, val_p, val_l, val_b)
        metrics = evaluate_model(model, model_name, te_p, te_l, te_b)
        all_metrics[model_name] = metrics

        plot_training_history(history, model_name)
        plot_confusion_matrix(metrics["y_true"], metrics["y_pred"], model_name)

        del model
        tf.keras.backend.clear_session()

    plot_roc_all(all_metrics)
    plot_metrics_bar(all_metrics)
    save_summary(all_metrics, l1, l2)

    print("\n[ HASIL EVALUASI MODEL ]\n")
    print(f"  {'Model':<15} {'Accuracy':>10} {'F1':>8} {'AUC-ROC':>10}")
    for name, m in all_metrics.items():
        print(f"  {name:<15} {m['accuracy']*100:>9.2f}%"
              f" {m['f1']:>8.4f} {m['auc_roc']:>10.4f}")
    best = max(all_metrics, key=lambda x: all_metrics[x]["accuracy"])
    print(f"\n  Best Model : {best} ({all_metrics[best]['accuracy']*100:.2f}%)")
    print(f"  Output: '{OUTPUT_DIR}/'\n")


if __name__ == "__main__":
    main()
