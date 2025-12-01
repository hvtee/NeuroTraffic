import sys
import cv2
import numpy as np
import os
import ast
from PyQt6.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton,
                             QVBoxLayout, QWidget, QTextEdit, QFileDialog, QHBoxLayout)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from ultralytics import YOLO
import configparser


class VideoWorker(QThread):
    frame_ready = pyqtSignal(np.ndarray, dict)
    finished = pyqtSignal()

    def __init__(self, video_path, conf_threshold, classes, model_path, to_show, to_save, img_size):
        super().__init__()
        self.video_path = video_path
        self.conf_threshold = conf_threshold
        self.classes = classes
        self.model_path = model_path
        self.to_show = to_show
        self.to_save = to_save
        self.img_size = img_size
        self._run_flag = True
        self._pause_flag = False

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return

        frame_idx = 0

        while self._run_flag and cap.isOpened():
            if self._pause_flag:
                self.msleep(100)
                continue

            ret, frame = cap.read()
            if not ret:
                break

            results = self.model(
                frame,
                conf=self.conf_threshold,
                show=self.to_show,
                save=self.to_save,
                classes=self.classes,
                imgsz=self.img_size
            )
            detections = results[0].boxes

            # Filter detections: accept any class in self.classes
            if isinstance(self.classes, list):
                car_detections = [
                    box for box in detections
                    if int(box.cls.item()) in self.classes
                ]
            else:
                car_detections = [
                    box for box in detections
                    if int(box.cls.item()) == self.classes
                ]

            annotated_frame = results[0].plot()
            total_cars = len(car_detections)

            stats = {
                "total": total_cars,
                "moving": 0,
                "stopped": 0,
                "frame": frame_idx,
                "filename": os.path.basename(self.video_path)
            }

            self.frame_ready.emit(annotated_frame, stats)
            self.msleep(30)
            frame_idx += 1

        cap.release()
        self.finished.emit()

    def stop(self):
        self._run_flag = False
        self.wait()

    def pause(self):
        self._pause_flag = True

    def resume(self):
        self._pause_flag = False

    @property
    def model(self):
        model = YOLO(self.model_path)

        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)

        return model


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NeuroTraffic")
        self.resize(1280, 720)

        config = self.load_config()
        self.model_path = config["model_path"]
        self.conf_threshold = config["conf_threshold"]
        self.classes = config["classes"]
        self.to_show = config["to_show"]
        self.to_save = config["to_save"]
        self.img_size = config["img_size"]

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        self.file_label = QLabel("No file selected")
        self.file_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.file_label.setMaximumHeight(24)

        self.browse_button = QPushButton("Select Video File")
        self.browse_button.setMaximumHeight(24)
        self.browse_button.clicked.connect(self.select_video)

        self.start_button = QPushButton("Start")
        self.start_button.setMaximumHeight(24)
        self.start_button.clicked.connect(self.start_processing)

        self.pause_button = QPushButton("Pause")
        self.pause_button.setMaximumHeight(24)
        self.pause_button.setEnabled(False)
        self.pause_button.clicked.connect(self.toggle_pause)

        self.stop_button = QPushButton("Stop")
        self.stop_button.setMaximumHeight(24)
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_processing)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.browse_button)
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.pause_button)
        button_layout.addWidget(self.stop_button)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")

        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMaximumHeight(100)
        self.stats_text.setStyleSheet("font-family: Courier; font-size: 12px;")

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_layout.addWidget(self.file_label)
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.video_label, 1)
        main_layout.addWidget(self.stats_text)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.worker = None
        self.current_video_path = None
        self.is_paused = False

    def parse_value(self, value):
        """Parse string value from config: bool, float, int, or list"""
        value = value.strip()
        if value.lower() in ("true", "false"):
            return value.lower() == "true"
        if value.startswith("[") and value.endswith("]"):
            try:
                return ast.literal_eval(value)
            except (ValueError, SyntaxError):
                pass
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            return value

    def load_config(self):
        config_file = "config.ini"
        config = configparser.ConfigParser()
        defaults = {
            "model_path": "yolo11m.pt",
            "conf_threshold": "0.5",
            "classes": "[2, 5, 7]",
            "to_show": "False",
            "to_save": "False",
            "img_size": "320"
        }

        # Load from file or use defaults
        if os.path.exists(config_file):
            try:
                config.read(config_file, encoding='utf-8')
                if 'settings' in config:
                    raw = dict(config['settings'])
                else:
                    print(f"Section '[settings]' not found in {config_file}. Using defaults.")
                    raw = defaults
            except Exception as e:
                print(f"Error reading config: {e}. Using defaults.")
                raw = defaults
        else:
            print(f"Config file '{config_file}' not found. Using defaults.")
            raw = defaults

        # Parse each value
        parsed = {}
        for key, value in raw.items():
            parsed[key] = self.parse_value(value)
        return parsed

    def select_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.flv *.wmv *.webm)"
        )
        if file_path:
            self.current_video_path = file_path
            self.file_label.setText(os.path.basename(file_path))
            self.start_button.setEnabled(True)
            self.pause_button.setEnabled(False)
            self.stop_button.setEnabled(False)

    def start_processing(self):
        if not self.current_video_path or not os.path.exists(self.current_video_path):
            self.stats_text.setText("Error: Please select a valid video file.")
            return

        self.stats_text.clear()
        self.stats_text.append(f"Processing: {os.path.basename(self.current_video_path)}")

        if self.worker and self.worker.isRunning():
            self.worker.stop()

        self.worker = VideoWorker(
            video_path=self.current_video_path,
            conf_threshold=self.conf_threshold,
            classes=self.classes,
            model_path=self.model_path,
            to_show=self.to_show,
            to_save=self.to_save,
            img_size=self.img_size
        )

        self.worker.frame_ready.connect(self.update_frame)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

        self.start_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.stop_button.setEnabled(True)
        self.is_paused = False
        self.pause_button.setText("Pause")

    def toggle_pause(self):
        if not self.worker:
            return
        if self.is_paused:
            self.worker.resume()
            self.pause_button.setText("Pause")
        else:
            self.worker.pause()
            self.pause_button.setText("Resume")
        self.is_paused = not self.is_paused

    def stop_processing(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
        self.cleanup_ui()

    def update_frame(self, frame, stats):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img).scaled(
            self.video_label.width(),
            self.video_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio
        )
        self.video_label.setPixmap(pixmap)

        text = (
            f"File: {stats['filename']}\n"
            f"Frame: {stats['frame']}\n"
            f"Cars total: {stats['total']}\n"
            f"Moving: {stats['moving']}\n"
            f"Stopped: {stats['stopped']}"
        )
        self.stats_text.setText(text)

    def on_finished(self):
        self.stats_text.append("\nProcessing completed.")
        self.cleanup_ui()

    def cleanup_ui(self):
        self.start_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.video_label.clear()
        self.is_paused = False
        self.pause_button.setText("Pause")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    try:
        window = MainWindow()
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        print(f"Application failed to start: {e}")
        sys.exit(1)
