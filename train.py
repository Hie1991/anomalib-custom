# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Getting Started with Anomalib Training using the Python API.

This example shows the basic steps to train an anomaly detection model
using the Anomalib Python API.
"""

from src.anomalib.data import MVTecAD
from src.anomalib.engine import Engine
from src.anomalib.models import (
    Cfa, Cflow, Csflow, Dfkde, Dfm, Draem, Dsr,
    EfficientAd, Fastflow, Fre, Ganomaly, Padim,
    Patchcore, ReverseDistillation, Stfpm,
    Supersimplenet, Uflow, VlmAd, WinClip,
    AiVad, Fuvas
)

def main():
    # 1. DataModule の作成（MVTecAD 例）
    datamodule = MVTecAD(
        root="./datasets/MVTecAD",
        category="bottle",
        train_batch_size=32,
        eval_batch_size=32,
        num_workers=6,  # ← DataLoader がワーカープロセスを 6 個使う
    )

    # 2. モデルの初期化
    model = EfficientAd()

    # 3. Engine を作って学習実行（max_epochs=10 で 10 エポック学習）
    engine = Engine(max_epochs=100)
    engine.fit(datamodule=datamodule, model=model)


# Windows で num_workers>0 の DataLoader を使う場合は、必ずこの「main ガード」を入れる
if __name__ == "__main__":
    main()