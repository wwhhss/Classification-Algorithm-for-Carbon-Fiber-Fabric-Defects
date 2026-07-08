"""Run every optimized experiment and write one machine-readable summary."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one tiny batch per model without downloading pretrained weights.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional model names to run (for example: resnet18 Xception).",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.smoke:
        os.environ["SMOKE_TEST"] = "1"

    from ml_experiments import (
        KerasConfig,
        TorchConfig,
        run_dual_resnet_svm,
        run_keras_experiment,
        run_torch_experiment,
    )

    experiments = [
        ("alexnet", lambda: run_torch_experiment(TorchConfig("alexnet", "alexnet.pth"))),
        (
            "densenet121",
            lambda: run_torch_experiment(
                TorchConfig("densenet121", "DenseNet121.pth", freeze_backbone=True)
            ),
        ),
        ("mobilenet_v2", lambda: run_torch_experiment(TorchConfig("mobilenet_v2", "Mobilenet.pth"))),
        ("resnet18", lambda: run_torch_experiment(TorchConfig("resnet18", "resnet18.pth"))),
        ("resnet34", lambda: run_torch_experiment(TorchConfig("resnet34", "ResNet34.pth"))),
        ("resnet50", lambda: run_torch_experiment(TorchConfig("resnet50", "ResNet50.pth"))),
        (
            "shufflenet_v2_x1_0",
            lambda: run_torch_experiment(TorchConfig("shufflenet_v2_x1_0", "ShuffleNet.pth")),
        ),
        (
            "squeezenet1_1",
            lambda: run_torch_experiment(TorchConfig("squeezenet1_1", "SqueezeNet.pth", epochs=50)),
        ),
        ("vgg11", lambda: run_torch_experiment(TorchConfig("vgg11", "VGG11.pth"))),
        (
            "convnext_tiny",
            lambda: run_torch_experiment(
                TorchConfig("convnext_tiny", "ConvNeXt.pth", freeze_backbone=True)
            ),
        ),
        (
            "mobilenet_v3_large",
            lambda: run_torch_experiment(
                TorchConfig("mobilenet_v3_large", "MobileNetV3.pth", freeze_backbone=True)
            ),
        ),
        (
            "regnet_y_400mf",
            lambda: run_torch_experiment(
                TorchConfig("regnet_y_400mf", "RegNet.pth", freeze_backbone=True)
            ),
        ),
        (
            "swin_t",
            lambda: run_torch_experiment(
                TorchConfig("swin_t", "SwinTransformer.pth", freeze_backbone=True)
            ),
        ),
        ("InceptionV3", lambda: run_keras_experiment(KerasConfig("InceptionV3", "Inception_weights.h5"))),
        (
            "NASNetMobile",
            lambda: run_keras_experiment(KerasConfig("NASNetMobile", "NASNetMobile_weights.h5")),
        ),
        ("Xception", lambda: run_keras_experiment(KerasConfig("Xception", "Xception_weights.h5"))),
        ("Dual-ResNet18-SVM", run_dual_resnet_svm),
    ]
    requested = set(args.only or [])
    selected_names = [name for name, _ in experiments if not requested or name in requested]
    if not args.worker:
        results = []
        for name in selected_names:
            command = [sys.executable, str(Path(__file__).resolve())]
            if args.smoke:
                command.append("--smoke")
            command.extend(["--worker", "--only", name])
            completed = subprocess.run(command, cwd=str(Path(__file__).resolve().parent))
            output_dir = Path("artifacts") / ("smoke" if args.smoke else "full")
            worker_result = json.loads(
                (output_dir / "run_summary.json").read_text(encoding="utf-8")
            )
            results.extend(worker_result)
            if completed.returncode and worker_result[0].get("status") != "failed":
                results[-1] = {
                    "name": name,
                    "status": "failed",
                    "error": f"worker exit {completed.returncode}",
                }
        summary_path = output_dir / "run_summary.json"
        summary_path.write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        passed = sum(item["status"] == "ok" for item in results)
        print(f"\nCompleted: {passed}/{len(results)} passed. Summary: {summary_path.resolve()}")
        return 0 if passed == len(results) else 1

    results = []
    for name, runner in experiments:
        if requested and name not in requested:
            continue
        print(f"\n===== {name} =====", flush=True)
        try:
            results.append({"name": name, "status": "ok", "result": runner()})
        except Exception as error:  # Keep the batch running so all failures are visible.
            traceback.print_exc()
            results.append({"name": name, "status": "failed", "error": str(error)})

    output_dir = Path("artifacts") / ("smoke" if args.smoke else "full")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "run_summary.json"
    summary_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    passed = sum(item["status"] == "ok" for item in results)
    print(f"\nCompleted: {passed}/{len(results)} passed. Summary: {summary_path.resolve()}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
