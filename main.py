import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFrame, QComboBox, QMessageBox, QProgressBar,
                             QStackedWidget, QGraphicsDropShadowEffect, QSizePolicy, QInputDialog, QLineEdit)
from PyQt6.QtGui import QFont, QPixmap, QImage, QColor, QIcon
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
import cv2
import numpy as np
from ultralytics import YOLO
import time
import pyttsx3
import threading
import winsound

class ModelThread(QThread):
    model_loaded = pyqtSignal(object)
    model_error = pyqtSignal(str)

    def run(self):
        try:
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yolov10m.pt')
            model = YOLO(model_path)
            self.model_loaded.emit(model)
        except Exception as e:
            self.model_error.emit(str(e))

class CameraThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    error_occurred = pyqtSignal(str)

    def __init__(self, url, parent=None):
        super().__init__(parent)
        self.url = url
        self.running = False
        self.cap = None

    def run(self):
        self.running = True
        reconnect_delay = 1  # Start with 1 second delay
        max_reconnect_delay = 30  # Maximum delay of 30 seconds

        while self.running:
            try:
                if self.cap is None or not self.cap.isOpened():
                    self.connect_to_camera()

                if self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret and frame is not None and frame.size > 0:
                        self.frame_ready.emit(frame)
                        reconnect_delay = 1  # Reset delay on successful frame read
                    else:
                        raise Exception("Failed to grab valid frame")
                else:
                    raise Exception("Camera is not opened")

            except Exception as e:
                print(f"Error in camera thread: {str(e)}")
                self.error_occurred.emit(str(e))
                if self.cap is not None:
                    self.cap.release()
                self.cap = None

                # Wait before trying to reconnect
                time.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)  # Exponential backoff

    def connect_to_camera(self):
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'  # Changed to TCP
        self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

    def stop(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
        self.wait()

class AirportSecurityGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI-Powered Unattended Baggage Detection")
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QLabel {
                color: #ffffff;
            }
            QPushButton {
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #005a9e;
            }
            QPushButton:pressed {
                background-color: #004275;
            }
            QFrame {
                background-color: #2d2d2d;
                border-radius: 10px;
            }
            QComboBox {
                background-color: #FF8000;
                color: white;
                border: none;
                padding: 5px;
                border-radius: 5px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: url(dropdown_arrow.png);
                width: 12px;
                height: 12px;
            }
            QComboBox QAbstractItemView {
                background-color: #1e1e1e;
                color: white;
                selection-background-color: #FF8000;
            }
            QProgressBar {
                border: 2px solid #FF8000;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #FF8000;
            }
        """)
        self.setMinimumSize(800, 600)

        icon = QIcon("logo.png")
        self.setWindowIcon(icon)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        self.stacked_widget = QStackedWidget()
        self.main_layout.addWidget(self.stacked_widget)

        self.loading_page = QWidget()
        self.main_page = QWidget()

        self.setup_loading_page()
        self.setup_main_page()

        self.stacked_widget.addWidget(self.loading_page)
        self.stacked_widget.addWidget(self.main_page)

        self.detection_active = False
        self.camera_index = 0
        self.setup_model()

        self.last_frame_time = time.time()

        self.camera_type = "Webcam"
        self.camera_address = 0
        self.camera_username = None
        self.camera_password = None
        self.camera_protocol = "rtsp"
        self.cap = None
        self.camera_opened_message_shown = False
        self.exit_requested = False

        self.camera_thread = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Update every 30 ms (approx. 33 fps)
        self.reconnect_timer = QTimer(self)
        self.reconnect_timer.timeout.connect(self.try_reconnect)
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 3

        

        self.setup_camera()

    def setup_loading_page(self):
        layout = QVBoxLayout(self.loading_page)
        loading_label = QLabel("Loading AI Model...")
        loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        loading_label.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        self.loading_progress = QProgressBar()
        self.loading_progress.setRange(0, 0)
        layout.addWidget(loading_label)
        layout.addWidget(self.loading_progress)

    def setup_main_page(self):
        layout = QVBoxLayout(self.main_page)

        header_layout = QHBoxLayout()
        header_text = QLabel("AI-Powered Unattended Baggage Detection")
        header_text.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        header_text.setStyleSheet("color: #FF8000; background-color: rgba(30, 30, 30, 200);")
        header_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(header_text)
        layout.addLayout(header_layout)

        content_layout = QHBoxLayout()
        layout.addLayout(content_layout)

        video_container = QFrame()
        video_container.setStyleSheet("background-color: #2d2d2d; border-radius: 15px;")
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 0, 0, 150))
        shadow.setOffset(0, 0)
        video_container.setGraphicsEffect(shadow)
        video_layout = QVBoxLayout(video_container)
        self.video_feed = QLabel()
        self.video_feed.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_feed.setStyleSheet("background-color: #1e1e1e; border: 2px solid #FF8000; border-radius: 10px;")
        self.video_feed.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        video_layout.addWidget(self.video_feed)
        
        self.detection_status = QLabel("Detection Stopped")
        self.detection_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.detection_status.setFont(QFont("Arial", 14))
        self.detection_status.setStyleSheet("color: #ff4444; margin-top: 10px;")
        video_layout.addWidget(self.detection_status)
        
        content_layout.addWidget(video_container, 3)

        side_panel = QFrame()
        side_panel.setStyleSheet("background-color: #2d2d2d; border-radius: 15px;")
        side_layout = QVBoxLayout(side_panel)
        content_layout.addWidget(side_panel, 1)

        info_panel = QFrame()
        info_layout = QVBoxLayout(info_panel)

        self.person_count = QLabel("Persons: 0")
        self.threat_count = QLabel("Potential Threats: 0")
        self.unattended_count = QLabel("Unattended Bags: 0")

        for label in [self.person_count, self.threat_count, self.unattended_count]:
            label.setFont(QFont("Arial", 14))
            label.setAlignment(Qt.AlignmentFlag.AlignLeft)
            label.setStyleSheet("background-color: #3d3d3d; padding: 10px; border-radius: 5px; margin: 5px 0;")
            info_layout.addWidget(label)

        side_layout.addWidget(info_panel)

        control_panel = QFrame()
        control_layout = QVBoxLayout(control_panel)

        self.start_button = QPushButton("Start AI Detection")
        self.start_button.setStyleSheet("background-color: #00FF00;")
        self.stop_button = QPushButton("Stop AI Detection")
        self.stop_button.setStyleSheet("background-color: #FF0000;")
        self.exit_button = QPushButton("Exit")
        self.exit_button.setStyleSheet("background-color: #000000; border-radius: 15px; padding: 5px;")
        self.exit_button.setFixedSize(80, 30)

        self.configure_camera_button = QPushButton("Configure Camera")
        self.configure_camera_button.clicked.connect(self.configure_camera)

        self.camera_selector = QComboBox()
        self.camera_selector.addItems(["Webcam", "IP Camera", "HDMI Camera"])
        self.camera_selector.currentIndexChanged.connect(self.change_camera_type)
        self.camera_selector.setStyleSheet("background-color: #FF8000;")

        for widget in [self.start_button, self.stop_button, self.configure_camera_button, self.camera_selector, self.exit_button]:
            widget.setFont(QFont("Arial", 12))
            widget.setFixedHeight(50)
            control_layout.addWidget(widget)

        side_layout.addWidget(control_panel)

        self.start_button.clicked.connect(self.start_detection)
        self.stop_button.clicked.connect(self.stop_detection)
        self.exit_button.clicked.connect(self.close)

    def setup_model(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)

        self.model_thread = ModelThread()
        self.model_thread.model_loaded.connect(self.on_model_loaded)
        self.model_thread.model_error.connect(self.on_model_error)
        self.model_thread.start()

    def on_model_loaded(self, model):
        self.model = model
        self.target_objects = ['person', 'suitcase', 'handbag', 'backpack', 'scissors','trash']
        self.luggage_tracker = {}
        self.person_tracker = {}
        self.continuous_alert = False
        self.stacked_widget.setCurrentIndex(1)
        self.setup_camera()

    def on_model_error(self, error):
        QMessageBox.critical(self, "Error", f"Failed to load AI model: {error}")
        self.close()

    def change_camera_type(self):
        self.camera_type = self.camera_selector.currentText()
        self.configure_camera()

    def configure_camera(self):
        if self.camera_type == "IP Camera":
            address, ok = QInputDialog.getText(self, "IP Camera Configuration", "Enter IP camera address (e.g., 192.168.1.100:8080):")
            if ok and address:
                self.camera_address = address
                protocol, ok = QInputDialog.getItem(self, "IP Camera Protocol", "Select protocol:", ["rtsp", "http"], 0, False)
                if ok:
                    self.camera_protocol = protocol
                    self.camera_username, ok = QInputDialog.getText(self, "IP Camera Authentication", "Enter username (leave blank if not required):")
                    if ok:
                        self.camera_password, ok = QInputDialog.getText(self, "IP Camera Authentication", "Enter password (leave blank if not required):", QLineEdit.EchoMode.Password)
                        if ok:
                            self.setup_camera()
                        else:
                            self.camera_type = "Webcam"
                            self.camera_selector.setCurrentText("Webcam")
                    else:
                        self.camera_type = "Webcam"
                        self.camera_selector.setCurrentText("Webcam")
                else:
                    self.camera_type = "Webcam"
                    self.camera_selector.setCurrentText("Webcam")
            else:
                self.camera_type = "Webcam"
                self.camera_selector.setCurrentText("Webcam")
        elif self.camera_type == "HDMI Camera":
            device_index, ok = QInputDialog.getInt(self, "HDMI Camera Configuration", "Enter device index (usually 0, 1, or 2):", 0, 0, 10)
            if ok:
                self.camera_address = device_index
                self.setup_camera()
            else:
                self.camera_type = "Webcam"
                self.camera_selector.setCurrentText("Webcam")
        else:  # Webcam
            self.camera_address = 0
            self.setup_camera()

    def construct_ip_camera_url(self):
        if ':' not in self.camera_address:
            # If no port is specified, add default ports
            if self.camera_protocol == "rtsp":
                self.camera_address += ":554"
            elif self.camera_protocol == "http":
                self.camera_address += ":80"
        
        # Construct the full URL
        if self.camera_username and self.camera_password:
            url = f"{self.camera_protocol}://{self.camera_username}:{self.camera_password}@{self.camera_address}"
        else:
            url = f"{self.camera_protocol}://{self.camera_address}"
        
        # Remove the default path for RTSP
        if self.camera_protocol == "rtsp":
            url = url.replace("/h264_ulaw.sdp", "")
        elif self.camera_protocol == "http":
            url += "/video"
        
        return url

    def try_reconnect(self):
        if self.reconnect_attempts < self.max_reconnect_attempts:
            print(f"Attempting to reconnect... (Attempt {self.reconnect_attempts + 1})")
            self.reconnect_attempts += 1
            self.setup_camera()
        else:
            print("Max reconnection attempts reached. Please check your camera connection.")
            self.reconnect_timer.stop()
            self.reconnect_attempts = 0
            QMessageBox.critical(self, "Connection Error", "Failed to reconnect to the camera after multiple attempts. Please check your camera connection and restart the application.")


    def setup_camera(self):
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()

        try:
            if self.camera_type == "Webcam":
                self.cap = cv2.VideoCapture(0)
            elif self.camera_type == "IP Camera":
                ip_camera_url = self.construct_ip_camera_url()
                print(f"Attempting to connect to IP camera at: {ip_camera_url}")
                
                os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'
                
                self.cap = cv2.VideoCapture(ip_camera_url, cv2.CAP_FFMPEG)
                
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                
            elif self.camera_type == "HDMI Camera":
                self.cap = cv2.VideoCapture(self.camera_address)
            
            if self.cap is None or not self.cap.isOpened():
                raise ValueError(f"Could not open camera: {self.camera_type}")
            
            # Test reading a frame
            for _ in range(5):
                ret, frame = self.cap.read()
                if ret and frame is not None and frame.size > 0:
                    break
                time.sleep(1)
            
            if not ret or frame is None or frame.size == 0:
                raise ValueError(f"Could not read valid frame from camera: {self.camera_type}")
            
            print(f"Successfully opened {self.camera_type}")
            self.reconnect_timer.stop()
            self.reconnect_attempts = 0
            self.timer.start(30)  # Start updating frames
        except Exception as e:
            error_message = f"Failed to open camera: {str(e)}"
            print(error_message)
            if self.reconnect_attempts < self.max_reconnect_attempts:
                self.reconnect_timer.start(5000)
            else:
                QMessageBox.warning(self, "Camera Error", error_message + "\n\nReverting to default webcam.")
                self.camera_type = "Webcam"
                self.camera_selector.setCurrentText("Webcam")
                self.camera_address = 0
                self.setup_camera()
            
    def start_detection(self):
        if hasattr(self, 'model'):
            self.detection_active = True
            self.detection_status.setText("Detection Active")
            self.detection_status.setStyleSheet("color: #44ff44; margin-top: 10px;")
            print("Detection started")
        else:
            QMessageBox.warning(self, "Model Error", "YOLO model not loaded. Please wait for the model to finish loading.")

    def stop_detection(self):
        self.detection_active = False
        self.detection_status.setText("Detection Stopped")
        self.detection_status.setStyleSheet("color: #ff4444; margin-top: 10px;")
        print("Detection stopped")

    def update_frame(self):
        if self.exit_requested or not hasattr(self, 'cap') or self.cap is None:
            return

        try:
            ret, frame = self.cap.read()
            if not ret or frame is None or frame.size == 0:
                if not self.exit_requested:
                    raise ValueError("Failed to grab valid frame")
                else:
                    return

            self.last_frame_time = time.time()

            if self.detection_active and hasattr(self, 'model'):
                frame = self.process_frame(frame)

            video_feed_size = self.video_feed.size()
            h, w = frame.shape[:2]
            aspect_ratio = w / h

            if self.isMaximized():
                max_width = video_feed_size.width()
                max_height = video_feed_size.height()
            else:
                max_width = int(video_feed_size.width() * 0.95)
                max_height = int(video_feed_size.height() * 0.95)

            if max_width / max_height > aspect_ratio:
                new_height = max_height
                new_width = int(new_height * aspect_ratio)
            else:
                new_width = max_width
                new_height = int(new_width / aspect_ratio)

            frame = cv2.resize(frame, (new_width, new_height))
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            qt_image = QImage(rgb_image.data, w, h, ch * w, QImage.Format.Format_RGB888)
            self.video_feed.setPixmap(QPixmap.fromImage(qt_image))

        except Exception as e:
            if not self.exit_requested:
                print(f"Error updating frame: {str(e)}")
                self.detection_active = False
                self.detection_status.setText("Camera Error")
                self.detection_status.setStyleSheet("color: #ff4444; margin-top: 10px;")
                QMessageBox.critical(self, "Camera Error", f"Error occurred while reading from camera: {str(e)}\n\nPlease check your camera connection and restart the application.")
                if hasattr(self, 'cap') and self.cap is not None:
                    self.cap.release()
                self.cap = None

    def process_frame(self, frame):
        results = self.model(frame, classes=[k for k, v in self.model.names.items() if v in self.target_objects])

        person_count = 0
        potential_threat_count = 0
        unattended_bag_count = 0
        current_time = time.time()

        annotated_frame = frame.copy()
        threat_detected = False
        unattended_bag_detected = False
        
        self.person_tracker.clear()

        for r in results:
            boxes = r.boxes
            for box in boxes:
                c = box.cls
                class_name = self.model.names[int(c)]
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center = ((x1 + x2) // 2, (y1 + y2) // 2)

                if class_name == 'person':
                    person_count += 1
                    self.person_tracker[center] = True
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f"Person {person_count}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                elif class_name in ['suitcase', 'handbag', 'backpack']:
                    if center in self.luggage_tracker:
                        time_detected = current_time - self.luggage_tracker[center]
                        if time_detected >= 5:
                            if not any(np.linalg.norm(np.array(center) - np.array(p)) < 100 for p in self.person_tracker.keys()):
                                unattended_bag_count += 1
                                potential_threat_count += 1
                                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                cv2.putText(annotated_frame, 'UNATTENDED BAG', (x1, y1 - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                threat_detected = True
                                unattended_bag_detected = True
                            else:
                                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                                cv2.putText(annotated_frame, class_name, (x1, y1 - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        else:
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            cv2.putText(annotated_frame, class_name, (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    else:
                        self.luggage_tracker[center] = current_time
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(annotated_frame, class_name, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                elif class_name in ['scissors']:
                    potential_threat_count += 1
                    threat_detected = True
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(annotated_frame, 'THREAT', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        self.luggage_tracker = {k: v for k, v in self.luggage_tracker.items() if current_time - v < 3}

        if unattended_bag_detected and not self.continuous_alert:
            self.speak_alert(f"Alert!, unattended bag{'s' if unattended_bag_count > 1 else ''} detected")
            self.continuous_alert = True
            threading.Thread(target=self.beep_alert, daemon=True).start()
        elif threat_detected and not self.continuous_alert:
            self.speak_alert("Potential threat detected")
            self.continuous_alert = True
            threading.Thread(target=self.beep_alert, daemon=True).start()
        elif not threat_detected and not unattended_bag_detected:
            self.continuous_alert = False

        self.person_count.setText(f"Persons: {person_count}")
        self.threat_count.setText(f"Potential Threats: {potential_threat_count}")
        self.unattended_count.setText(f"Unattended Bags: {unattended_bag_count}")

        return annotated_frame

    def speak_alert(self, message):
        try:
            self.engine.say(message)
            self.engine.runAndWait()
        except Exception as e:
            print(f"Error in text-to-speech: {str(e)}")

    def beep_alert(self):
        while self.continuous_alert:
            try:
                winsound.Beep(1000, 500)
                time.sleep(0.5)
            except Exception as e:
                print(f"Error in beep alert: {str(e)}")
                break

    def closeEvent(self, event):
        self.exit_requested = True
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
        self.stop_detection()
        if hasattr(self, 'timer'):
            self.timer.stop()
        print("Program exited successfully")
        event.accept()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_frame()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AirportSecurityGUI()
    window.show()
    sys.exit(app.exec())