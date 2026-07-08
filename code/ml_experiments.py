"""Shared, reproducible training pipelines used by all notebooks in this folder."""

from __future__ import annotations

import gc
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = Path(
    os.getenv(
        "IMAGE_DATA_DIR",
        "C:/Users/WHS/Desktop/SVM-Image-Classification-master/images4",
    )
)
DEFAULT_RESNET_WEIGHTS = Path(
    os.getenv(
        "DUAL_RESNET_WEIGHTS",
        "C:/Users/WHS/Desktop/Dual-ResNet18-SVM/result/resnet18_d1.pth",
    )
)
DEFAULT_TIME_DATA_DIR = Path(
    os.getenv(
        "IMAGE_TIME_DATA_DIR",
        "C:/Users/WHS/Desktop/SVM-Image-Classification-master/images3/val",
    )
)
SMOKE_TEST = os.getenv("SMOKE_TEST", "0").lower() in {"1", "true", "yes"}


@dataclass
class TorchConfig:
    model_name: str
    checkpoint: str
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    val_ratio: float = 0.30
    seed: int = 42
    freeze_backbone: bool = False
    data_dir: Path = DEFAULT_DATA_DIR
    time_data_dir: Path = DEFAULT_TIME_DATA_DIR


@dataclass
class KerasConfig:
    model_name: str
    checkpoint: str
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    val_ratio: float = 0.30
    seed: int = 42
    data_dir: Path = DEFAULT_DATA_DIR


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _artifact_path(filename: str) -> Path:
    folder = ROOT / "artifacts" / ("smoke" if SMOKE_TEST else "full")
    folder.mkdir(parents=True, exist_ok=True)
    return folder / filename


def _write_result(name: str, result: Dict[str, Any]) -> Path:
    path = _artifact_path(f"{name}_result.json")
    serializable = {key: value for key, value in result.items() if key != "model"}
    path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _plot_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    classes: Sequence[str],
    filename: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    labels = list(range(len(classes)))
    matrix = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    figure, axis = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(matrix, display_labels=classes).plot(
        ax=axis, cmap="Blues", values_format=".2f", colorbar=False
    )
    axis.set_title("Normalized confusion matrix")
    figure.tight_layout()
    figure.savefig(_artifact_path(filename), dpi=120)
    plt.close(figure)


def _stratified_indices(targets: Sequence[int], val_ratio: float, seed: int):
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(targets))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_ratio,
        random_state=seed,
        shuffle=True,
        stratify=np.asarray(targets),
    )
    if not SMOKE_TEST:
        return train_idx.tolist(), val_idx.tolist()

    targets_array = np.asarray(targets)

    def take_per_class(source: np.ndarray, count: int) -> List[int]:
        selected: List[int] = []
        for label in sorted(set(targets_array.tolist())):
            selected.extend(source[targets_array[source] == label][:count].tolist())
        return selected

    per_class = max(1, int(os.getenv("SMOKE_SAMPLES_PER_CLASS", "2")))
    return take_per_class(train_idx, per_class), take_per_class(val_idx, 1)


def _build_torch_model(model_name: str, class_count: int, pretrained: bool):
    import torch
    import torch.nn as nn
    from torchvision import models

    if SMOKE_TEST and model_name == "vgg11":
        from torchvision.models.vgg import cfgs, make_layers

        class CompactVGG(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = make_layers(cfgs["A"], batch_norm=False)
                self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
                self.classifier = nn.Linear(512, class_count)

            def forward(self, inputs):
                features = self.avgpool(self.features(inputs))
                return self.classifier(torch.flatten(features, 1))

        compact_model = CompactVGG()
        return compact_model, compact_model.classifier

    local_models = {
        "convnext_tiny": (
            "torchvision_py39.models.convnext",
            "convnext_tiny",
            "ConvNeXt_Tiny_Weights",
        ),
        "mobilenet_v3_large": (
            "torchvision_py39.models.mobilenetv3",
            "mobilenet_v3_large",
            "MobileNet_V3_Large_Weights",
        ),
        "regnet_y_400mf": (
            "torchvision_py39.models.regnet",
            "regnet_y_400mf",
            "RegNet_Y_400MF_Weights",
        ),
        "swin_t": (
            "torchvision_py39.models.swin_transformer",
            "swin_t",
            "Swin_T_Weights",
        ),
        "efficientnet_v2_s": (
            "torchvision_py39.models.efficientnet",
            "efficientnet_v2_s",
            "EfficientNet_V2_S_Weights",
        ),
        "efficientnet_v2_m": (
            "torchvision_py39.models.efficientnet",
            "efficientnet_v2_m",
            "EfficientNet_V2_M_Weights",
        ),
        "efficientnet_v2_l": (
            "torchvision_py39.models.efficientnet",
            "efficientnet_v2_l",
            "EfficientNet_V2_L_Weights",
        ),
    }
    if model_name in local_models:
        import importlib

        module_name, constructor_name, weights_name = local_models[model_name]
        module = importlib.import_module(module_name)
        constructor = getattr(module, constructor_name)
        weights = getattr(module, weights_name).DEFAULT if pretrained else None
        try:
            model = constructor(weights=weights)
        except Exception as error:
            if not pretrained:
                raise
            print(
                f"{model_name}: pretrained weights unavailable ({error!r}); "
                "using random initialization."
            )
            model = constructor(weights=None)
    else:
        constructor = getattr(models, model_name)
        try:
            model = constructor(weights="DEFAULT" if pretrained else None)
        except (TypeError, ValueError):
            model = constructor(pretrained=pretrained)

    if model_name in {"alexnet", "vgg11"}:
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, class_count)
        head = model.classifier[-1]
    elif model_name == "densenet121":
        model.classifier = nn.Linear(model.classifier.in_features, class_count)
        head = model.classifier
    elif model_name in {
        "mobilenet_v2",
        "mobilenet_v3_large",
        "efficientnet_v2_s",
        "efficientnet_v2_m",
        "efficientnet_v2_l",
    }:
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, class_count)
        head = model.classifier[-1]
    elif model_name == "convnext_tiny":
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, class_count)
        head = model.classifier[-1]
    elif model_name == "swin_t":
        model.head = nn.Linear(model.head.in_features, class_count)
        head = model.head
    elif model_name == "squeezenet1_1":
        model.classifier[1] = nn.Conv2d(
            model.classifier[1].in_channels, class_count, kernel_size=1
        )
        model.num_classes = class_count
        head = model.classifier[1]
    else:
        model.fc = nn.Linear(model.fc.in_features, class_count)
        head = model.fc
    return model, head


def _single_image_classification_time(model, dataset, device) -> float:
    """Return mean model-only inference time for batch-size-one classification."""
    import torch
    from torch.utils.data import DataLoader, Subset

    if SMOKE_TEST:
        dataset = Subset(dataset, range(min(len(dataset), 4)))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    model.eval()
    total_seconds = 0.0
    total_images = 0
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            logits = model(inputs)
            _ = logits.argmax(dim=1)
            if device.type == "cuda":
                torch.cuda.synchronize()
            total_seconds += time.perf_counter() - started
            total_images += inputs.size(0)
    if total_images == 0:
        raise RuntimeError("No images are available for classification timing.")
    return total_seconds / total_images


def run_torch_experiment(config: TorchConfig) -> Dict[str, Any]:
    """Train and evaluate one torchvision classifier without leaking validation data."""
    import torch
    import torch.nn as nn
    from sklearn.metrics import accuracy_score, recall_score
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms

    started = time.perf_counter()
    _seed_everything(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.set_num_threads(max(1, int(os.getenv("TORCH_NUM_THREADS", "2"))))

    data_dir = Path(config.data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    val_transform = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize]
    )
    index_dataset = datasets.ImageFolder(str(data_dir))
    train_idx, val_idx = _stratified_indices(
        index_dataset.targets, config.val_ratio, config.seed
    )
    train_data = Subset(datasets.ImageFolder(str(data_dir), train_transform), train_idx)
    val_data = Subset(datasets.ImageFolder(str(data_dir), val_transform), val_idx)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = min(config.batch_size, max(len(train_data), 1))
    if SMOKE_TEST:
        batch_size = min(batch_size, max(1, int(os.getenv("SMOKE_BATCH_SIZE", "1"))))
    loader_options = {
        "batch_size": batch_size,
        "num_workers": max(0, int(os.getenv("DATA_LOADER_WORKERS", "0"))),
        "pin_memory": device.type == "cuda",
    }
    loaders = {
        "train": DataLoader(train_data, shuffle=True, **loader_options),
        "val": DataLoader(val_data, shuffle=False, **loader_options),
    }

    pretrained = not SMOKE_TEST and os.getenv("PRETRAINED", "1") != "0"
    model, head = _build_torch_model(
        config.model_name, len(index_dataset.classes), pretrained
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if config.freeze_backbone or SMOKE_TEST:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in head.parameters():
            parameter.requires_grad = True
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        (p for p in model.parameters() if p.requires_grad),
        lr=config.learning_rate,
        momentum=0.9,
        weight_decay=1e-4,
    )
    epochs = 1 if SMOKE_TEST else int(os.getenv("EPOCHS", str(config.epochs)))
    best_accuracy = -1.0
    best_y_true: List[int] = []
    best_y_pred: List[int] = []
    history: List[Dict[str, float]] = []
    checkpoint_path = _artifact_path(config.checkpoint)

    training_started = time.perf_counter()
    for epoch in range(epochs):
        row: Dict[str, float] = {"epoch": epoch + 1}
        for phase in ("train", "val"):
            training = phase == "train"
            model.train(training)
            total_loss = 0.0
            all_true: List[int] = []
            all_pred: List[int] = []
            for inputs, labels in loaders[phase]:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(training):
                    logits = model(inputs)
                    loss = criterion(logits, labels)
                    if training:
                        loss.backward()
                        optimizer.step()
                total_loss += loss.item() * inputs.size(0)
                all_true.extend(labels.detach().cpu().tolist())
                all_pred.extend(logits.argmax(dim=1).detach().cpu().tolist())

            row[f"{phase}_loss"] = total_loss / max(len(all_true), 1)
            row[f"{phase}_accuracy"] = accuracy_score(all_true, all_pred)
            row[f"{phase}_macro_recall"] = recall_score(
                all_true, all_pred, average="macro", zero_division=0
            )
            if phase == "val" and row["val_accuracy"] > best_accuracy:
                best_accuracy = row["val_accuracy"]
                best_y_true = all_true.copy()
                best_y_pred = all_pred.copy()
                # Avoid a second, potentially huge, in-memory parameter copy.
                # A smoke run only needs to verify serialization, so it stores the
                # small classification head; full runs store the complete model.
                checkpoint_state = head.state_dict() if SMOKE_TEST else model.state_dict()
                torch.save(checkpoint_state, checkpoint_path)
        history.append(row)
        print(
            f"{config.model_name}: epoch {epoch + 1}/{epochs} | "
            f"train_acc={row['train_accuracy']:.3f} | val_acc={row['val_accuracy']:.3f}"
        )
    training_seconds = time.perf_counter() - training_started

    accuracy = accuracy_score(best_y_true, best_y_pred)
    macro_recall = recall_score(
        best_y_true, best_y_pred, average="macro", zero_division=0
    )
    time_data_dir = Path(config.time_data_dir)
    timing_data = (
        datasets.ImageFolder(str(time_data_dir), val_transform)
        if time_data_dir.is_dir()
        else val_data
    )
    classification_seconds = _single_image_classification_time(
        model, timing_data, device
    )
    _plot_confusion_matrix(
        best_y_true,
        best_y_pred,
        index_dataset.classes,
        f"{config.model_name}_confusion.png",
    )
    result: Dict[str, Any] = {
        "framework": "pytorch",
        "model": config.model_name,
        "classes": index_dataset.classes,
        "train_samples": len(train_data),
        "validation_samples": len(val_data),
        "epochs": epochs,
        "accuracy": float(accuracy),
        "parameter_count": int(parameter_count),
        "accuracy_percent": float(accuracy * 100.0),
        "training_time_seconds": round(training_seconds, 6),
        "classification_time_seconds_per_image": round(
            classification_seconds, 9
        ),
        "macro_recall": float(macro_recall),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "checkpoint": str(checkpoint_path),
        "history": history,
    }
    _write_result(config.model_name, result)
    print(
        f"{config.model_name}: parameters={parameter_count:,}, "
        f"accuracy={result['accuracy_percent']:.3f}%, "
        f"training={result['training_time_seconds']:.3f}s, "
        "classification="
        f"{result['classification_time_seconds_per_image']:.6f}s/image"
    )
    del model, optimizer, loaders
    gc.collect()
    return result


def run_keras_experiment(config: KerasConfig) -> Dict[str, Any]:
    """Train and evaluate a Keras application model with streaming image batches."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    from sklearn.metrics import accuracy_score, recall_score
    from tensorflow import keras

    started = time.perf_counter()
    _seed_everything(config.seed)
    tf.random.set_seed(config.seed)
    data_dir = Path(config.data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")

    applications = {
        "InceptionV3": (keras.applications.InceptionV3, keras.applications.inception_v3.preprocess_input),
        "NASNetMobile": (keras.applications.NASNetMobile, keras.applications.nasnet.preprocess_input),
        "Xception": (keras.applications.Xception, keras.applications.xception.preprocess_input),
    }
    constructor, preprocess = applications[config.model_name]
    if SMOKE_TEST:
        batch_size = max(1, int(os.getenv("SMOKE_BATCH_SIZE", "1")))
    else:
        batch_size = config.batch_size
    train_generator = keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocess,
        rotation_range=10,
        zoom_range=0.10,
        horizontal_flip=True,
        validation_split=config.val_ratio,
    ).flow_from_directory(
        str(data_dir),
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode="sparse",
        subset="training",
        shuffle=True,
        seed=config.seed,
    )
    val_generator = keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocess, validation_split=config.val_ratio
    ).flow_from_directory(
        str(data_dir),
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode="sparse",
        subset="validation",
        shuffle=False,
        seed=config.seed,
    )

    base_model = constructor(
        weights=None if SMOKE_TEST or os.getenv("PRETRAINED", "1") == "0" else "imagenet",
        include_top=False,
        input_shape=(224, 224, 3),
        pooling="avg",
    )
    base_model.trainable = False
    model = keras.Sequential(
        [base_model, keras.layers.Dropout(0.2), keras.layers.Dense(train_generator.num_classes, activation="softmax")]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    epochs = 1 if SMOKE_TEST else int(os.getenv("EPOCHS", str(config.epochs)))
    fit_kwargs: Dict[str, Any] = {}
    if SMOKE_TEST:
        fit_kwargs.update(steps_per_epoch=1, validation_steps=1)
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        verbose=2,
        **fit_kwargs,
    )
    checkpoint_path = _artifact_path(config.checkpoint)
    model.save_weights(str(checkpoint_path))

    val_generator.reset()
    prediction_steps = 1 if SMOKE_TEST else len(val_generator)
    probabilities = model.predict(val_generator, steps=prediction_steps, verbose=0)
    y_pred = probabilities.argmax(axis=1)
    y_true = val_generator.classes[: len(y_pred)]
    classes = [name for name, _ in sorted(val_generator.class_indices.items(), key=lambda item: item[1])]
    accuracy = accuracy_score(y_true, y_pred)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    keras.backend.clear_session()
    del model, base_model
    gc.collect()
    _plot_confusion_matrix(y_true, y_pred, classes, f"{config.model_name}_confusion.png")
    result: Dict[str, Any] = {
        "framework": "tensorflow",
        "model": config.model_name,
        "classes": classes,
        "train_samples": int(train_generator.samples),
        "validation_samples": int(len(y_true) if SMOKE_TEST else val_generator.samples),
        "epochs": epochs,
        "accuracy": float(accuracy),
        "macro_recall": float(macro_recall),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "checkpoint": str(checkpoint_path),
        "history": {key: [float(v) for v in values] for key, values in history.history.items()},
    }
    _write_result(config.model_name, result)
    print(
        f"{config.model_name}: accuracy={accuracy:.3f}, recall={macro_recall:.3f}, "
        f"time={result['elapsed_seconds']:.1f}s"
    )
    gc.collect()
    return result


def _image_files(folder: Path) -> Iterable[Path]:
    allowed = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
    return sorted(path for path in folder.iterdir() if path.suffix.lower() in allowed)


def run_dual_resnet_svm(
    data_dir: Path = DEFAULT_DATA_DIR,
    weights_path: Path = DEFAULT_RESNET_WEIGHTS,
) -> Dict[str, Any]:
    """Extract all four Gaussian-pyramid scales in batches and tune a linear SVM."""
    import cv2
    import joblib
    import torch
    import torch.nn as nn
    from PIL import Image
    from sklearn.metrics import accuracy_score, recall_score
    from sklearn.model_selection import GridSearchCV, train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from torchvision import models, transforms

    started = time.perf_counter()
    _seed_everything(42)
    torch.manual_seed(42)
    data_dir, weights_path = Path(data_dir), Path(weights_path)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"ResNet weights do not exist: {weights_path}")

    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 4)
    model.load_state_dict(torch.load(str(weights_path), map_location="cpu"))
    extractor = nn.Sequential(*list(model.children())[:-1]).eval()
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    classes = sorted(path.name for path in data_dir.iterdir() if path.is_dir())
    paths: List[Path] = []
    labels: List[int] = []
    for label, class_name in enumerate(classes):
        class_paths = list(_image_files(data_dir / class_name))
        if SMOKE_TEST:
            class_paths = class_paths[: max(3, int(os.getenv("SMOKE_SAMPLES_PER_CLASS", "3")))]
        paths.extend(class_paths)
        labels.extend([label] * len(class_paths))

    features: List[np.ndarray] = []
    default_feature_batch = "1" if SMOKE_TEST else "8"
    batch_size = max(1, int(os.getenv("FEATURE_BATCH_SIZE", default_feature_batch)))
    for start in range(0, len(paths), batch_size):
        tensors = []
        for path in paths[start : start + batch_size]:
            bgr = cv2.imread(str(path))
            if bgr is None:
                raise ValueError(f"Unable to read image: {path}")
            image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pyramid = [image]
            for _ in range(3):
                pyramid.append(cv2.pyrDown(pyramid[-1]))
            tensors.extend(transform(Image.fromarray(level)) for level in pyramid)
        with torch.no_grad():
            batch_features = extractor(torch.stack(tensors)).flatten(1).numpy()
        features.extend(batch_features.reshape(-1, 4 * batch_features.shape[1]))

    x = np.asarray(features)
    y = np.asarray(labels)
    del extractor, model, tensors
    gc.collect()
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.30, random_state=42, shuffle=True, stratify=y
    )
    pipeline = make_pipeline(StandardScaler(), SVC())
    parameters = {
        "svc__kernel": ["linear"] if SMOKE_TEST else ["linear", "rbf", "poly", "sigmoid"],
        "svc__C": [1.0] if SMOKE_TEST else [1.0, 10.0, 100.0, 1000.0],
    }
    classifier = GridSearchCV(
        pipeline, parameters, cv=2 if SMOKE_TEST else 5, n_jobs=1, scoring="accuracy"
    )
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_val)
    accuracy = accuracy_score(y_val, y_pred)
    macro_recall = recall_score(y_val, y_pred, average="macro", zero_division=0)
    model_path = _artifact_path("svm_model.pkl")
    joblib.dump(classifier.best_estimator_, model_path)
    _plot_confusion_matrix(y_val, y_pred, classes, "dual_resnet_svm_confusion.png")
    result: Dict[str, Any] = {
        "framework": "pytorch+sklearn",
        "model": "Dual-ResNet18-SVM",
        "classes": classes,
        "samples": len(paths),
        "feature_dimensions": int(x.shape[1]),
        "accuracy": float(accuracy),
        "macro_recall": float(macro_recall),
        "best_parameters": classifier.best_params_,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "checkpoint": str(model_path),
    }
    _write_result("Dual-ResNet18-SVM", result)
    print(
        f"Dual-ResNet18-SVM: accuracy={accuracy:.3f}, recall={macro_recall:.3f}, "
        f"features={x.shape[1]}, time={result['elapsed_seconds']:.1f}s"
    )
    return result
