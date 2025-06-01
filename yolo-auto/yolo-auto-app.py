import sys
import os
import json
import yaml
import shutil
import subprocess
import stat
from pathlib import Path
from glob import glob
import random
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import cv2
import numpy as np
from PIL import Image
import torch

class YOLOTrainingApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # ─── 追加①: スクリーンの利用可能領域に合わせてウィンドウを初期化 ───
        screen = QApplication.primaryScreen()
        if screen:
            rect = screen.availableGeometry()
            # 画面の90%サイズで初期化
            self.resize(int(rect.width() * 0.9), int(rect.height() * 0.9))
        else:
            # フォールバック：幅1200, 高さ800
            self.resize(1200, 800)

        self.setWindowTitle("YOLO Training & Evaluation App")
        # （従来の setGeometry は削除）
        
        # データ管理用変数
        self.project_path = ""         # GUI で選択されたプロジェクトフォルダ
        self.current_model_type = ""   # "segment" or "detect"
        self.current_model_size = ""   # モデルサイズ (例："n", "s", ...)
        self.class_names = []          # 検出クラス名リスト
        self.trained_model_path = ""   # 学習済みモデル(.pt) のパス
        self.current_pixmap = None     # 現在表示中の QPixmap を保持（リサイズ時に再スケールするため）

        self.init_ui()
    
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # メインレイアウト
        main_layout = QHBoxLayout(central_widget)
        
        # 左パネル：操作パネル
        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel, 1)
        
        # 右パネル：結果表示
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, 2)
        
    def create_left_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        #――――――――――――――――――
        # プロジェクト設定
        project_group = QGroupBox("Project Settings")
        project_layout = QVBoxLayout(project_group)
        
        # プロジェクトフォルダ選択
        folder_layout = QHBoxLayout()
        self.project_label = QLabel("No project selected")
        self.project_btn = QPushButton("Select Project Folder")
        self.project_btn.clicked.connect(self.select_project_folder)
        folder_layout.addWidget(self.project_label)
        folder_layout.addWidget(self.project_btn)
        project_layout.addLayout(folder_layout)
        
        # モデル選択
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x",
            "yolov8n-seg", "yolov8s-seg", "yolov8m-seg", "yolov8l-seg", "yolov8x-seg",
            "yolov5n", "yolov5s", "yolov5m", "yolov5l", "yolov5x",
            "yolov5n-seg", "yolov5s-seg", "yolov5m-seg", "yolov5l-seg", "yolov5x-seg"
        ])
        self.model_combo.currentTextChanged.connect(self.model_changed)
        model_layout.addWidget(self.model_combo)
        project_layout.addLayout(model_layout)
        
        layout.addWidget(project_group)
        
        #――――――――――――――――――
        # アノテーション
        annotation_group = QGroupBox("Annotation")
        annotation_layout = QVBoxLayout(annotation_group)
        
        self.labelme_btn = QPushButton("Launch LabelMe")
        self.labelme_btn.clicked.connect(self.launch_labelme)
        annotation_layout.addWidget(self.labelme_btn)
        
        self.convert_btn = QPushButton("Convert to YOLO Format")
        self.convert_btn.clicked.connect(self.convert_to_yolo)
        annotation_layout.addWidget(self.convert_btn)
        
        layout.addWidget(annotation_group)
        
        #――――――――――――――――――
        # 学習設定
        training_group = QGroupBox("Training")
        training_layout = QVBoxLayout(training_group)
        
        # エポック数
        epochs_layout = QHBoxLayout()
        epochs_layout.addWidget(QLabel("Epochs:"))
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(100)
        epochs_layout.addWidget(self.epochs_spin)
        training_layout.addLayout(epochs_layout)
        
        # バッチサイズ
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("Batch Size:"))
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 128)
        self.batch_spin.setValue(16)
        batch_layout.addWidget(self.batch_spin)
        training_layout.addLayout(batch_layout)
        
        # 画像サイズ
        imgsz_layout = QHBoxLayout()
        imgsz_layout.addWidget(QLabel("Image Size:"))
        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(320, 1280)
        self.imgsz_spin.setValue(640)
        self.imgsz_spin.setSingleStep(32)
        imgsz_layout.addWidget(self.imgsz_spin)
        training_layout.addLayout(imgsz_layout)
        
        self.train_btn = QPushButton("Start Training")
        self.train_btn.clicked.connect(self.start_training)
        training_layout.addWidget(self.train_btn)
        
        layout.addWidget(training_group)
        
        #――――――――――――――――――
        # 評価
        evaluation_group = QGroupBox("Evaluation")
        evaluation_layout = QVBoxLayout(evaluation_group)
        
        self.select_image_btn = QPushButton("Select Test Image")
        self.select_image_btn.clicked.connect(self.select_test_image)
        evaluation_layout.addWidget(self.select_image_btn)
        
        self.evaluate_btn = QPushButton("Run Inference")
        self.evaluate_btn.clicked.connect(self.run_inference)
        evaluation_layout.addWidget(self.evaluate_btn)
        
        layout.addWidget(evaluation_group)
        
        #――――――――――――――――――
        # プログレスバー
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
        # ログ
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(150)
        log_layout.addWidget(self.log_text)
        layout.addWidget(log_group)
        
        layout.addStretch()
        return panel
        
    def create_right_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 結果表示エリア
        self.result_tabs = QTabWidget()
        
        # 画像表示タブ
        self.image_tab = QWidget()
        image_layout = QVBoxLayout(self.image_tab)
        self.image_label = QLabel("No image loaded")
        self.image_label.setAlignment(Qt.AlignCenter)
        # 枠線と最小高さを指定
        self.image_label.setStyleSheet("border: 1px solid gray; min-height: 400px;")
        self.image_label.setScaledContents(False)  # 自動拡大縮小は自前で行う
        image_layout.addWidget(self.image_label)
        
        # 結果情報
        self.result_info = QTextEdit()
        self.result_info.setMaximumHeight(100)
        image_layout.addWidget(self.result_info)
        
        self.result_tabs.addTab(self.image_tab, "Results")
        
        # トレーニング結果タブ
        self.training_tab = QWidget()
        training_layout = QVBoxLayout(self.training_tab)
        self.training_results = QTextEdit()
        training_layout.addWidget(self.training_results)
        self.result_tabs.addTab(self.training_tab, "Training Results")
        
        layout.addWidget(self.result_tabs)
        
        return panel
        
    def log_message(self, message):
        """ログメッセージを追加"""
        self.log_text.append(f"[{QTime.currentTime().toString()}] {message}")
        QApplication.processEvents()
        
    def select_project_folder(self):
        """プロジェクトフォルダを選択"""
        folder = QFileDialog.getExistingDirectory(self, "Select Project Folder")
        if folder:
            self.project_path = folder
            self.project_label.setText(os.path.basename(folder))
            self.setup_project_structure()
            self.log_message(f"Project folder selected: {folder}")
            
    def setup_project_structure(self):
        """プロジェクトフォルダ構造を作成"""
        if not self.project_path:
            return
            
        # 必要なフォルダを作成
        folders = [
            "images", "annotations", "labels", 
            "train", "train/images", "train/labels",
            "val", "val/images", "val/labels",
            "models", "runs"
        ]
        
        for folder in folders:
            folder_path = os.path.join(self.project_path, folder)
            try:
                self.safe_mkdir(folder_path)
                self.log_message(f"Created folder: {folder}")
            except Exception as e:
                self.log_message(f"Error creating {folder}: {str(e)}")
                
        self.log_message("Project structure setup completed")
        
    def model_changed(self):
        """モデル選択変更時の処理"""
        model_name = self.model_combo.currentText()
        if "seg" in model_name:
            self.current_model_type = "segment"
        else:
            self.current_model_type = "detect"
            
        if "v8" in model_name:
            self.current_model_size = model_name.replace("yolov8", "").replace("-seg", "")
        else:
            self.current_model_size = model_name.replace("yolov5", "").replace("-seg", "")
            
        self.log_message(f"Model changed to: {model_name} (Type: {self.current_model_type})")
        
    def launch_labelme(self):
        """LabelMeを起動"""
        if not self.project_path:
            QMessageBox.warning(self, "Warning", "Please select a project folder first")
            return
            
        images_path = os.path.join(self.project_path, "images")
        annotations_path = os.path.join(self.project_path, "annotations")
        
        # LabelMeを起動
        try:
            cmd = f"labelme {images_path} --output {annotations_path} --nodata"
            subprocess.Popen(cmd, shell=True)
            self.log_message("LabelMe launched")
        except Exception as e:
            self.log_message(f"Error launching LabelMe: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to launch LabelMe: {str(e)}")
            
    def convert_to_yolo(self):
        """LabelMe形式からYOLO形式に変換"""
        if not self.project_path:
            QMessageBox.warning(self, "Warning", "Please select a project folder first")
            return
            
        #――――――――――――――――――
        # ① プロジェクト直下にある元画像ファイルをすべて images フォルダへコピー
        images_path = os.path.join(self.project_path, "images")
        # まず images フォルダがあるかチェック (setup_project_structure で作成済み)
        if not os.path.exists(images_path):
            self.safe_mkdir(images_path)
        # コピー可能な拡張子：.jpg, .jpeg, .png
        copied_count = 0
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for img_src in glob(os.path.join(self.project_path, ext)):
                try:
                    dst = os.path.join(images_path, os.path.basename(img_src))
                    shutil.copy2(img_src, dst)
                    copied_count += 1
                except Exception as e:
                    self.log_message(f"Error copying image {img_src}: {str(e)}")
        self.log_message(f"Copied {copied_count} images to /images")
        
        #――――――――――――――――――
        # ② アノテーション（JSONなど）がある annotations フォルダ・labels フォルダを準備
        annotations_path = os.path.join(self.project_path, "annotations")
        labels_path = os.path.join(self.project_path, "labels")
        
        if not os.path.exists(annotations_path) or not os.listdir(annotations_path):
            QMessageBox.warning(self, "Warning", "No annotations found. Please annotate images first.")
            return
        
        try:
            if not os.path.exists(labels_path):
                self.safe_mkdir(labels_path)
            else:
                # 既存のラベルファイルをクリア
                for file in os.listdir(labels_path):
                    if file.endswith('.txt'):
                        file_path = os.path.join(labels_path, file)
                        try:
                            self.safe_remove(file_path)
                        except Exception:
                            self.log_message(f"Could not remove {file_path}")
        except Exception as e:
            self.log_message(f"Error preparing labels folder: {str(e)}")
            QMessageBox.critical(self, "Error", f"Could not prepare labels folder: {str(e)}")
            return
            
        #――――――――――――――――――
        # ③ LabelMe JSON → YOLO 形式への変換を実行
        try:
            self.convert_labelme_to_yolo(annotations_path, labels_path)
            
            # ④ データセット分割 (train/val) を実行
            self.split_dataset()
            
            # ⑤ YAML 設定ファイルを作成
            self.create_yaml_config()
            
            self.log_message("Conversion to YOLO format completed successfully")
            QMessageBox.information(self, "Success", "Conversion to YOLO format completed!")
            
        except Exception as e:
            self.log_message(f"Error converting to YOLO format: {str(e)}")
            QMessageBox.critical(self, "Error", f"Conversion failed: {str(e)}")
            
    def convert_labelme_to_yolo(self, annotations_path, labels_path):
        """
        LabelMe JSON から YOLO 形式の txt に変換。
        polygon, rectangle, circle, point, linestring, mask, line の shape_type に対応。
        """
        import base64
        from io import BytesIO
        from PIL import Image
        import numpy as np

        # 1) 全 JSON を走査してクラス名を収集
        all_classes = set()
        json_files = [f for f in os.listdir(annotations_path) if f.endswith('.json')]
        for json_file in json_files:
            try:
                with open(os.path.join(annotations_path, json_file), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for shape in data.get('shapes', []):
                    all_classes.add(shape['label'])
            except Exception as e:
                self.log_message(f"Error reading {json_file} for class scan: {e}")
        sorted_classes = sorted(all_classes)
        self.class_names = sorted_classes

        converted_count = 0
        for json_file in json_files:
            try:
                json_path = os.path.join(annotations_path, json_file)
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                img_h = data.get('imageHeight')
                img_w = data.get('imageWidth')

                txt_filename = os.path.splitext(json_file)[0] + '.txt'
                txt_path = os.path.join(labels_path, txt_filename)

                lines = []
                for shape in data.get('shapes', []):
                    label = shape['label']
                    class_id = sorted_classes.index(label)
                    shape_type = shape.get('shape_type')
                    points = shape.get('points', [])

                    # --- polygon ---
                    if shape_type == 'polygon' and len(points) >= 3:
                        if self.current_model_type == "segment":
                            # セグメンテーション用に正規化された頂点リストをそのまま使用
                            norm_pts = []
                            for x, y in points:
                                x_norm = max(0, min(1, x / img_w))
                                y_norm = max(0, min(1, y / img_h))
                                norm_pts.extend([x_norm, y_norm])
                            line = f"{class_id} " + " ".join(f"{v:.6f}" for v in norm_pts)
                            lines.append(line)
                            continue
                        else:
                            # 検出用にポリゴンを外接矩形に変換
                            xs = [p[0] for p in points]
                            ys = [p[1] for p in points]
                            x1, x2 = min(xs), max(xs)
                            y1, y2 = min(ys), max(ys)

                    # --- rectangle ---
                    elif shape_type == 'rectangle' and len(points) == 2:
                        (x1, y1), (x2, y2) = points

                    # --- circle ---
                    elif shape_type == 'circle' and len(points) == 2:
                        (cx, cy), (px, py) = points
                        r = ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5
                        x1, y1 = cx - r, cy - r
                        x2, y2 = cx + r, cy + r

                        if self.current_model_type == "segment":
                            # 矩形ポリゴンをセグメンテーション用に使用
                            norm_pts = [
                                x1 / img_w, y1 / img_h,
                                x1 / img_w, y2 / img_h,
                                x2 / img_w, y2 / img_h,
                                x2 / img_w, y1 / img_h
                            ]
                            line = f"{class_id} " + " ".join(f"{v:.6f}" for v in norm_pts)
                            lines.append(line)
                            continue

                    # --- point ---
                    elif shape_type == 'point' and len(points) == 1:
                        (x, y) = points[0]
                        x1, y1, x2, y2 = x, y, x, y

                        if self.current_model_type == "segment":
                            # 点を小さな矩形ポリゴンとして扱う
                            eps = 1  # 1px 閾値
                            x1p, y1p = max(0, x - eps), max(0, y - eps)
                            x2p, y2p = min(img_w, x + eps), min(img_h, y + eps)
                            norm_pts = [
                                x1p / img_w, y1p / img_h,
                                x1p / img_w, y2p / img_h,
                                x2p / img_w, y2p / img_h,
                                x2p / img_w, y1p / img_h
                            ]
                            line = f"{class_id} " + " ".join(f"{v:.6f}" for v in norm_pts)
                            lines.append(line)
                            continue

                    # --- linestring or line ---
                    elif shape_type in ('linestring', 'line') and len(points) >= 2:
                        xs = [p[0] for p in points]
                        ys = [p[1] for p in points]
                        x1, x2 = min(xs), max(xs)
                        y1, y2 = min(ys), max(ys)

                        if self.current_model_type == "segment":
                            # 線分の外接矩形をセグメンテーション用に使用
                            norm_pts = [
                                x1 / img_w, y1 / img_h,
                                x1 / img_w, y2 / img_h,
                                x2 / img_w, y2 / img_h,
                                x2 / img_w, y1 / img_h
                            ]
                            line = f"{class_id} " + " ".join(f"{v:.6f}" for v in norm_pts)
                            lines.append(line)
                            continue

                    # --- mask ---
                    elif shape_type == 'mask' and shape.get('mask'):
                        # base64 データをデコードして bbox を計算
                        mask_data = shape.get('mask')
                        # "data:image/png;base64,xxxx..." の形式を想定
                        if ',' in mask_data:
                            b64 = mask_data.split(',', 1)[1]
                        else:
                            b64 = mask_data
                        try:
                            mask_bytes = base64.b64decode(b64)
                            mask_img = Image.open(BytesIO(mask_bytes)).convert('L')
                            mask_arr = np.array(mask_img)
                            ys, xs = np.where(mask_arr > 0)
                            if len(xs) and len(ys):
                                x1, x2 = xs.min(), xs.max()
                                y1, y2 = ys.min(), ys.max()
                            else:
                                continue
                        except Exception as e:
                            self.log_message(f"Error decoding mask in {json_file}: {e}")
                            continue

                        if self.current_model_type == "segment":
                            # マスクの矩形をセグメンテーション用に使用
                            norm_pts = [
                                x1 / img_w, y1 / img_h,
                                x1 / img_w, y2 / img_h,
                                x2 / img_w, y2 / img_h,
                                x2 / img_w, y1 / img_h
                            ]
                            line = f"{class_id} " + " ".join(f"{v:.6f}" for v in norm_pts)
                            lines.append(line)
                            continue

                    else:
                        # 未対応の shape_type は無視
                        continue

                    # --------------------
                    # ここから検出用 (detect) の YOLO bbox 出力ロジック
                    # x1,y1,x2,y2 が定義されていることを前提に処理
                    x_center = ((x1 + x2) / 2) / img_w
                    y_center = ((y1 + y2) / 2) / img_h
                    width = abs(x2 - x1) / img_w
                    height = abs(y2 - y1) / img_h
                    line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                    lines.append(line)

                # txt に書き込む（lines が空でなければ改行を追加）
                with open(txt_path, 'w', encoding='utf-8') as f:
                    if lines:
                        f.write("\n".join(lines) + "\n")

                converted_count += 1

            except Exception as e:
                self.log_message(f"Error converting {json_file}: {e}")

        self.log_message(f"Conversion completed: {converted_count} files, {len(sorted_classes)} classes found")
        self.log_message(f"Classes: {sorted_classes}")
        
    def split_dataset(self):
        """
        データセットを 8:2 の割合で訓練用と検証用に分割し、
        画像ファイルと対応ラベルをそれぞれ train/images, train/labels,
        val/images, val/labels にコピーする
        """
        images_path = os.path.join(self.project_path, "images")
        labels_path = os.path.join(self.project_path, "labels")
        
        train_images_path = os.path.join(self.project_path, "train", "images")
        train_labels_path = os.path.join(self.project_path, "train", "labels")
        val_images_path = os.path.join(self.project_path, "val", "images")
        val_labels_path = os.path.join(self.project_path, "val", "labels")
        
        # 既存の train/val フォルダをクリーンアップして再作成
        for folder in [train_images_path, train_labels_path, val_images_path, val_labels_path]:
            if os.path.exists(folder):
                try:
                    shutil.rmtree(folder)
                except Exception as e:
                    self.log_message(f"Error removing folder {folder}: {str(e)}")
            self.safe_mkdir(folder)
        
        # images/ フォルダから画像ファイルを一覧取得
        if not os.path.exists(images_path):
            self.log_message(f"Images folder not found: {images_path}")
            return
            
        image_files = [
            f for f in os.listdir(images_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ]
        
        if not image_files:
            self.log_message("No image files found in images folder")
            return
        
        self.log_message(f"Found {len(image_files)} images in /images")
        
        # 8:2 の割合でランダムにシャッフルして分割
        random.shuffle(image_files)
        split_idx = max(1, int(len(image_files) * 0.8))  # 80% を train、残りを val
        train_files = image_files[:split_idx]
        val_files = image_files[split_idx:]
        
        # ※もし画像が 1 枚しかない場合も最低 1 枚は train に入るようにする
        if len(image_files) == 1:
            train_files = image_files
            val_files = []
        
        self.log_message(f"Train count: {len(train_files)}, Val count: {len(val_files)}")
        
        # 訓練用ファイルをコピー
        train_copied = 0
        for img_file in train_files:
            try:
                # 画像コピー
                src_img = os.path.join(images_path, img_file)
                dst_img = os.path.join(train_images_path, img_file)
                shutil.copy2(src_img, dst_img)
                train_copied += 1
                
                # ラベルコピー（存在しない場合は空ファイルを作成）
                label_file = os.path.splitext(img_file)[0] + '.txt'
                src_label = os.path.join(labels_path, label_file)
                dst_label = os.path.join(train_labels_path, label_file)
                if os.path.exists(src_label):
                    shutil.copy2(src_label, dst_label)
                else:
                    # ラベルがない場合は空ファイルを作成
                    with open(dst_label, 'w', encoding='utf-8') as f:
                        pass
            except Exception as e:
                self.log_message(f"Error copying train file {img_file}: {str(e)}")
                
        # 検証用ファイルをコピー
        val_copied = 0
        for img_file in val_files:
            try:
                # 画像コピー
                src_img = os.path.join(images_path, img_file)
                dst_img = os.path.join(val_images_path, img_file)
                shutil.copy2(src_img, dst_img)
                val_copied += 1
                
                # ラベルコピー（存在しない場合は空ファイルを作成）
                label_file = os.path.splitext(img_file)[0] + '.txt'
                src_label = os.path.join(labels_path, label_file)
                dst_label = os.path.join(val_labels_path, label_file)
                if os.path.exists(src_label):
                    shutil.copy2(src_label, dst_label)
                else:
                    # ラベルがない場合は空ファイルを作成
                    with open(dst_label, 'w', encoding='utf-8') as f:
                        pass
            except Exception as e:
                self.log_message(f"Error copying val file {img_file}: {str(e)}")
        
        self.log_message(f"Dataset split completed: {train_copied} train images, {val_copied} val images")
        
    def create_yaml_config(self):
        """YOLO用のYAMLファイルを作成"""
        yaml_path = os.path.join(self.project_path, "dataset.yaml")
        
        config = {
            'path': os.path.abspath(self.project_path).replace('\\', '/'),
            'train': 'train/images',
            'val': 'val/images',
            'nc': len(self.class_names),
            'names': self.class_names
        }
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            
        self.log_message(f"Created dataset.yaml with {len(self.class_names)} classes: {self.class_names}")
        self.log_message(f"Dataset path: {config['path']}")
        
        # フォルダの存在確認と個数ログ
        train_path = os.path.join(self.project_path, "train", "images")
        val_path = os.path.join(self.project_path, "val", "images")
        
        if os.path.exists(train_path):
            train_count = len([f for f in os.listdir(train_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
            self.log_message(f"Train images: {train_count}")
        else:
            self.log_message(f"Warning: Train path does not exist: {train_path}")
            
        if os.path.exists(val_path):
            val_count = len([f for f in os.listdir(val_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
            self.log_message(f"Val images: {val_count}")
        else:
            self.log_message(f"Warning: Val path does not exist: {val_path}")
            
    def start_training(self):
        """学習を開始"""
        if not self.project_path:
            QMessageBox.warning(self, "Warning", "Please select a project folder first")
            return
            
        yaml_path = os.path.join(self.project_path, "dataset.yaml")
        if not os.path.exists(yaml_path):
            QMessageBox.warning(self, "Warning", "Please convert annotations to YOLO format first")
            return
            
        # 学習用スレッドを開始
        self.training_thread = TrainingThread(
            model=self.model_combo.currentText(),
            data=yaml_path,
            epochs=self.epochs_spin.value(),
            batch=self.batch_spin.value(),
            imgsz=self.imgsz_spin.value(),
            project=os.path.join(self.project_path, "runs")
        )
        self.training_thread.progress.connect(self.update_progress)
        self.training_thread.finished.connect(self.training_finished)
        self.training_thread.log.connect(self.log_message)
        self.training_thread.start()
        
        self.train_btn.setEnabled(False)
        self.log_message("Training started...")
        
    def update_progress(self, value):
        """プログレスバーを更新"""
        self.progress_bar.setValue(value)
        
    def training_finished(self, model_path):
        """学習完了時の処理"""
        self.train_btn.setEnabled(True)
        self.trained_model_path = model_path
        self.progress_bar.setValue(100)
        self.log_message(f"Training completed. Model saved to: {model_path}")
        
        # 結果を表示
        results_path = os.path.dirname(model_path)
        results_text = f"Training completed!\n\nModel saved to: {model_path}\n\n"
        
        # results.txt があれば読み込む
        results_file = os.path.join(results_path, "results.txt")
        if os.path.exists(results_file):
            with open(results_file, 'r', encoding='utf-8') as f:
                results_text += f.read()
                
        self.training_results.setText(results_text)
        self.result_tabs.setCurrentIndex(1)  # Training Results タブに切り替え
        
    def select_test_image(self):
        """テスト画像を選択"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Test Image", "",
            "Image Files (*.jpg *.jpeg *.png *.bmp)"
        )
        if file_path:
            self.test_image_path = file_path
            # 選択直後は単に元画像を表示
            self.display_image(file_path, results=None)
            self.log_message(f"Test image selected: {os.path.basename(file_path)}")
            
    def display_image(self, image_input, results=None):
        """
        画像を表示。引数 image_input は
         - 文字列パス (str) の場合は「そのまま画像を読み込んで表示」
         - numpy.ndarray (RGB) の場合は「その配列を QImage に変換して表示」
        results はあくまで使わない (描画済みの numpy.array を直接渡す想定)。
        
        リサイズ時には self.current_pixmap を再スケールして常にアスペクト比を維持します。
        """
        if isinstance(image_input, str) and results is None:
            # ファイルパスから直接読み込むケース
            pixmap = QPixmap(image_input)
        elif isinstance(image_input, np.ndarray):
            # numpy.ndarray (RGB) を QPixmap に変換するケース
            rgb = image_input
            height, width, channel = rgb.shape
            bytes_per_line = 3 * width
            q_image = QImage(rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
        else:
            # 予期せぬタイプは扱わない
            return
        
        # 表示用に保持
        self.current_pixmap = pixmap
        
        # ラベルサイズに合わせてアスペクト比を維持しつつスケール
        scaled = pixmap.scaled(
            self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled)
        
    def resizeEvent(self, event):
        """
        ウィンドウサイズが変更された際に呼ばれる。表示中の QPixmap があればリサイズして反映する。
        """
        super().resizeEvent(event)
        if self.current_pixmap:
            scaled = self.current_pixmap.scaled(
                self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled)
        
    def run_inference(self):
        """推論を実行"""
        if not hasattr(self, 'test_image_path'):
            QMessageBox.warning(self, "Warning", "Please select a test image first")
            return

        # 学習済みモデルパスがまだセットされていない場合は、自動で探す
        if not self.trained_model_path:
            latest = self._find_latest_model_path()
            if latest:
                self.trained_model_path = latest
                self.log_message(f"Found latest trained model: {latest}")
            else:
                QMessageBox.warning(self, "Warning", "No trained model found. Please train a model first.")
                return

        try:
            # YOLOv8/v5 の推論
            if "v8" in self.model_combo.currentText():
                from ultralytics import YOLO
                model = YOLO(self.trained_model_path)
                results = model(self.test_image_path)
                det = results[0]  # 1画像分だけ扱う
            else:
                # YOLOv5 の場合
                model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.trained_model_path)
                results = model(self.test_image_path)
                det = results[0]  # 1画像分だけ扱う

            # ─── セグメンテーションモデルの場合は det.plot() に任せ、さらにクラス名を上書き ───
            if self.current_model_type == "segment":
                # 1) マスク＋ボックスを描画した overlay_img を取得
                overlay_img = det.plot()
                # （det.plot() が RGB で返ってくる場合が多いので、BGR に変換してから文字を描く）
                img_to_draw = cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR)

                # 2) YOLOv8/v5 のどちらでも「ボックス座標」「クラスID」「信頼度(conf)」を取る
                try:
                    # YOLOv8 の場合:
                    #   det.boxes.xyxy → shape=(n,4),   det.boxes.cls → shape=(n,)
                    #   det.boxes.conf → shape=(n,)
                    boxes = det.boxes.xyxy.cpu().numpy().astype(int)      # (n,4)
                    classes = det.boxes.cls.cpu().numpy().astype(int)     # (n,)
                    confidences = det.boxes.conf.cpu().numpy().astype(float)  # (n,)
                except Exception:
                    # 万一上記が取れなかったら空リストにしておく
                    boxes = []
                    classes = []
                    confidences = []

                # 3) クラス名と信頼度を重ねて描画
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box
                    cls_id = classes[i]
                    conf = confidences[i]  # 0.0 ～ 1.0 の float

                    # クラス名が範囲内かチェック
                    if cls_id < len(self.class_names):
                        class_name = self.class_names[cls_id]
                    else:
                        class_name = f"class_{cls_id}"

                    # 「クラス名 + (信頼度 0.xx)」という表示例
                    text = f"{class_name} {conf:.2f}"

                    # フォント設定
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.6
                    thickness = 2

                    # テキストサイズを事前に取得して背景を描く（可読性向上のため）
                    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                    text_w, text_h = text_size

                    # テキスト背景用の四角形を描画
                    background_tl = (x1, max(y1 - text_h - 4, 0))
                    background_br = (x1 + text_w, y1)
                    cv2.rectangle(img_to_draw, background_tl, background_br, (0, 0, 0), cv2.FILLED)

                    # テキスト自体を白で描画
                    text_org = (x1, y1 - 4)
                    cv2.putText(
                        img_to_draw,
                        text,
                        text_org,
                        font,
                        font_scale,
                        (255, 255, 255),  # 白
                        thickness,
                        lineType=cv2.LINE_AA
                    )

                # 4) 最終的に OpenCV (BGR) から RGB に戻して表示
                final_rgb = cv2.cvtColor(img_to_draw, cv2.COLOR_BGR2RGB)
                self.display_image(final_rgb, results=None)
            else:
                # …（検出モデルのときは従来どおり det.plot() を表示する, または同様にクラス+confを描画してもOK）
                output_plot = det.plot()
                self.display_image(output_plot, results=None)

            # 検出/セグメンテーション結果の詳細を表示
            result_text = self.format_results(results)
            self.result_info.setText(result_text)

            self.result_tabs.setCurrentIndex(0)  # Results タブに切り替え
            self.log_message("Inference completed")

        except Exception as e:
            self.log_message(f"Error during inference: {str(e)}")
            QMessageBox.critical(self, "Error", f"Inference failed: {str(e)}")
            
    def format_results(self, results):
        """推論結果をフォーマット"""
        if "v8" in self.model_combo.currentText():
            # YOLOv8の結果
            result = results[0]
            if self.current_model_type == "segment":
                # セグメンテーションの場合、マスク数を表示
                count = result.masks.data.shape[0] if (hasattr(result, "masks") and result.masks is not None) else 0
                text = f"Segmented {count} objects (masks) in the image.\n\n"
            else:
                # 普通の検出 (detect)
                text = f"Detected {len(result.boxes)} objects:\n\n"
                for i, box in enumerate(result.boxes):
                    conf = float(box.conf)
                    cls = int(box.cls)
                    class_name = self.class_names[cls] if cls < len(self.class_names) else f"class_{cls}"
                    text += f"{i+1}. {class_name}: {conf:.3f}\n"
        else:
            # YOLOv5 の結果
            det = results[0]
            if self.current_model_type == "segment" and hasattr(det, "masks") and det.masks is not None:
                count = det.masks.data.shape[0]
                text = f"Segmented {count} objects (masks) in the image.\n\n"
            else:
                df = det.pandas().xyxy[0]
                text = f"Detected {len(df)} objects:\n\n"
                for i, row in df.iterrows():
                    conf = row['confidence']
                    class_name = row['name']
                    text += f"{i+1}. {class_name}: {conf:.3f}\n"
                
        return text

    #――――――――――――――――――
    # 追加②: 最新モデルを探すヘルパーを追加
    def _find_latest_model_path(self):
        """
        self.project_path/runs ディレクトリ内の "train*" フォルダを列挙し、
        更新日時が最新のフォルダ配下にある weights/best.pt のパスを返す。
        見つからなかった場合は None を返す。
        """
        if not self.project_path:
            return None
        
        runs_root = os.path.join(self.project_path, "runs")
        if not os.path.exists(runs_root):
            return None
        
        # runs/train, runs/train2, runs/train3... を探す
        pattern = os.path.join(runs_root, "train*")
        run_dirs = glob(pattern)
        if not run_dirs:
            return None
        
        # 更新日時（os.path.getmtime）が最新のディレクトリを選択
        latest_run = max(run_dirs, key=os.path.getmtime)
        candidate = os.path.join(latest_run, "weights", "best.pt")
        if os.path.exists(candidate):
            return candidate
        return None

    #――――――――――――――――――
    # 以下、パーミッション調整付きのヘルパー関数
    def safe_mkdir(self, path):
        """
        ディレクトリ作成を試み、PermissionError発生時は親ディレクトリのパーミッションを変更して再試行する
        """
        try:
            os.makedirs(path, exist_ok=True)
        except PermissionError:
            parent = os.path.dirname(path) or "."
            try:
                # 親ディレクトリを 0o755 に設定
                os.chmod(parent, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
                os.makedirs(path, exist_ok=True)
            except Exception as e:
                raise e
        # 作成後に自ディレクトリのパーミッションを 0o755 に設定
        try:
            os.chmod(path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
        except Exception:
            pass  # 失敗しても無視

    def safe_remove(self, file_path):
        """
        ファイル削除を試み、PermissionError発生時はファイルのパーミッションを変更して再試行する
        """
        try:
            os.remove(file_path)
        except PermissionError:
            try:
                # ファイルを 0o666 に設定して再削除
                os.chmod(file_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH | stat.S_IWOTH)
                os.remove(file_path)
            except Exception as e:
                raise e


class TrainingThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(str)
    log = pyqtSignal(str)
    
    def __init__(self, model, data, epochs, batch, imgsz, project):
        super().__init__()
        self.model = model
        self.data = data
        self.epochs = epochs
        self.batch = batch
        self.imgsz = imgsz
        self.project = project
        
    def run(self):
        try:
            if "v8" in self.model:
                # --- YOLOv8 での学習 ---
                from ultralytics import YOLO
                model = YOLO(f"{self.model}.pt")

                results = model.train(
                    data=self.data,
                    epochs=self.epochs,
                    batch=self.batch,
                    imgsz=self.imgsz,
                    project=self.project,  # e.g. "runs/train"
                    name="train"          # 最初は "train"。自動で train2, train3... が作られる
                )

                # ↓ ここで results から “実際に使われた run ディレクトリ” を取得できる場合はそれを使うと確実です。
                #   YOLOv8 の場合、results には run フォルダのパスが格納されているので、
                #   直接以下のように書くことも可能です（ultralytics==8.x の実装を前提とする例）。
                #
                # ✓ 例: results[0].path が "runs/train3" のような絶対パスを返す場合
                #
                try:
                    # results は通常リスト状になっていて、1 件だけ返ってくるので results[0] を使う
                    run_dir = results[0].path  # e.g. "runs/train3"
                    model_path = os.path.join(run_dir, "weights", "best.pt")
                except Exception:
                    # 万が一 results から取得できない場合は、glob で探す fallback
                    run_dir = self._find_latest_run_dir()
                    model_path = os.path.join(run_dir, "weights", "best.pt")

            else:
                # --- YOLOv5 での学習 ---
                cmd = [
                    "python", "-m", "yolov5.train",
                    "--data", self.data,
                    "--cfg", f"yolov5{self.model[-1]}.yaml",
                    "--weights", f"{self.model}.pt",
                    "--epochs", str(self.epochs),
                    "--batch-size", str(self.batch),
                    "--img", str(self.imgsz),
                    "--project", self.project,  # e.g. "runs/train"
                    "--name", "train"           # 最初は "train"。２回目以降は train2, train3...
                ]

                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                while True:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        self.log.emit(output.strip())

                # 学習完了後、project 配下の最新 train* フォルダを探す
                run_dir = self._find_latest_run_dir()
                model_path = os.path.join(run_dir, "weights", "best.pt")

            # 最終的に最新の best.pt パスを emit
            self.finished.emit(model_path)

        except Exception as e:
            self.log.emit(f"Training error: {str(e)}")


    def _find_latest_run_dir(self):
        """
        self.project フォルダ内にある "train" で始まるディレクトリを列挙し、
        作成日時（または更新日時）が最新のものを返す。
        例: "runs/train3", "runs/train2", "runs/train" の中で最新を選ぶ。
        """
        # "self.project" の直下にある "train*" フォルダをすべて取得
        # 例: runs/train, runs/train2, runs/train3 ...
        pattern = os.path.join(self.project, "train*")
        run_dirs = glob(pattern)

        if not run_dirs:
            # 万が一フォルダが見つからなければ自己責任で train フォルダを返す
            return os.path.join(self.project, "train")

        # 更新日時が最も新しいディレクトリを選択
        latest_dir = max(run_dirs, key=os.path.getmtime)
        return latest_dir


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = YOLOTrainingApp()
    window.show()
    sys.exit(app.exec_())