import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

def visualize_prediction(pred):
    """
    pred.image_path   : Path もしくは文字列、あるいはそれらのリスト
    pred.anomaly_map  : numpy.ndarray。形状は (H_model, W_model) または (N, H_model, W_model)
    pred.pred_label   : int もしくは torch.Tensor、あるいはそれらのリスト
    pred.pred_score   : float もしくは torch.Tensor、あるいはそれらのリスト
    """

    def _visualize_single(img_path, am, label, score):
        # 1) Path に変換（str なら Path()）
        img_path = Path(img_path)

        # 2) 元画像を読み込む（オリジナル解像度）
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)
        h_img, w_img = img_np.shape[:2]

        # 3) 異常マップを正規化してカラーマップ化 → PIL 画像に変換してリサイズ
        am_min, am_max = am.min(), am.max()
        if am_max - am_min > 1e-6:
            am_norm = (am - am_min) / (am_max - am_min)
        else:
            am_norm = np.zeros_like(am)

        am_colored = plt.get_cmap("jet")(am_norm)[..., :3]
        am_colored = (am_colored * 255).astype(np.uint8)
        am_pil = Image.fromarray(am_colored)
        am_resized = am_pil.resize((w_img, h_img), resample=Image.BILINEAR)
        am_resized_np = np.array(am_resized)

        # 4) オーバーレイ画像を作成
        overlay = (0.5 * img_np + 0.5 * am_resized_np).astype(np.uint8)

        # 5) ラベル・スコアを Python の型にキャスト
        #    （Tensor の場合は .item() でスカラーを取り出す）
        if hasattr(score, "item"):
            score_val = score.item()
        else:
            score_val = float(score)  # float 値であることを想定

        if hasattr(label, "item"):
            label_val = label.item()
        else:
            label_val = int(label)    # もともと int 型ならそのまま

        # 6) Matplotlib で表示
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img_np)
        axes[0].set_title(f"Input Image\n{img_path.name}")
        axes[0].axis("off")

        axes[1].imshow(am_resized_np)
        axes[1].set_title(f"Anomaly Map (colored)\n(resized to {w_img}×{h_img})")
        axes[1].axis("off")

        axes[2].imshow(overlay)
        axes[2].set_title("Overlay (alpha=0.5)")
        axes[2].axis("off")

        fig.suptitle(f"Label: {label_val}, Score: {score_val:.3f}", fontsize=14)
        plt.tight_layout()
        plt.show()

    # バッチ処理された場合はリスト／タプル扱い
    if isinstance(pred.image_path, (list, tuple)):
        for idx, img_path in enumerate(pred.image_path):
            am = pred.anomaly_map[idx]  # (H_model, W_model)
            # pred.pred_label, pred.pred_score がリストかタプルなら idx 番目を取り出し
            label = (pred.pred_label[idx]
                     if isinstance(pred.pred_label, (list, tuple))
                     else pred.pred_label)
            score = (pred.pred_score[idx]
                     if isinstance(pred.pred_score, (list, tuple))
                     else pred.pred_score)
            _visualize_single(img_path, am, label, score)
    else:
        # 単一画像の場合
        am = pred.anomaly_map         # (H_model, W_model)
        label = pred.pred_label
        score = pred.pred_score
        _visualize_single(pred.image_path, am, label, score)


# --- 使い方の例 ---
# 以下は predict 実行後の predictions リストに対して可視化を行う流れです。

from src.anomalib.data import PredictDataset, MVTecAD
from src.anomalib.engine import Engine
from src.anomalib.models import (
    Cfa, Cflow, Csflow, Dfkde, Dfm, Draem, Dsr,
    EfficientAd, Fastflow, Fre, Ganomaly, Padim,
    Patchcore, ReverseDistillation, Stfpm,
    Supersimplenet, Uflow, VlmAd, WinClip,
    AiVad, Fuvas
)

def main():
    # 1) まずは推論を行う（以前と同様のコード）
    model = Fastflow()
    engine = Engine()

    dataset = PredictDataset(
        path=Path("datasets/MVTecAD/bottle/test/broken_large"),
        image_size=(256, 256),
    )

    predictions = engine.predict(
        model=model,
        dataset=dataset,
        ckpt_path="results/Fastflow/MVTecAD/bottle/v13/weights/lightning/model.ckpt",
    )

    # 2) 予測結果があれば、可視化関数で順番に表示
    if predictions is not None:
        for pred in predictions:
            visualize_prediction(pred)


if __name__ == "__main__":
    main()