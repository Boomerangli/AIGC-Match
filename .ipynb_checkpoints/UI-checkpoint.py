import sys
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QPushButton,
    QFileDialog,
    QVBoxLayout,
    QWidget,
    QHBoxLayout,
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import numpy as np
from PIL import Image


class ImageSegmentationViewer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("医学图像分割系统")
        self.setGeometry(100, 100, 1200, 600)

        # Initialize main widget
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)

        # Layouts
        self.layout = QVBoxLayout()
        self.button_layout = QHBoxLayout()
        self.image_layout = QHBoxLayout()

        # Buttons
        self.upload_button = QPushButton("上传图像")
        self.upload_button.clicked.connect(self.upload_image)
        self.button_layout.addWidget(self.upload_button)

        self.quit_button = QPushButton("退出系统")
        self.quit_button.clicked.connect(self.close)
        self.button_layout.addWidget(self.quit_button)

        # Image Display Areas
        self.original_label = QLabel("原始图像")
        self.original_label.setAlignment(Qt.AlignCenter)
        self.mask_label = QLabel("真值掩码")
        self.mask_label.setAlignment(Qt.AlignCenter)
        self.predicted_label = QLabel("预测结果")
        self.predicted_label.setAlignment(Qt.AlignCenter)

        self.image_layout.addWidget(self.original_label)
        self.image_layout.addWidget(self.mask_label)
        self.image_layout.addWidget(self.predicted_label)

        # Add layouts to main layout
        self.layout.addLayout(self.button_layout)
        self.layout.addLayout(self.image_layout)

        self.main_widget.setLayout(self.layout)

    def upload_image(self):
        # Select an image file
        image_path, _ = QFileDialog.getOpenFileName(
            self, "选择图像文件", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )

        if image_path:
            # Display Original Image
            self.display_image(image_path, self.original_label)

            # Dummy Example: Create and Display True Mask and Predicted Mask
            true_mask = self.create_dummy_mask(image_path)
            pred_mask = self.create_dummy_mask(image_path, predicted=True)

            self.display_mask(true_mask, self.mask_label)
            self.display_mask(pred_mask, self.predicted_label)

    def display_image(self, image_path, label):
        pixmap = QPixmap(image_path)
        label.setPixmap(pixmap.scaled(300, 300, Qt.KeepAspectRatio))

    def create_dummy_mask(self, image_path, predicted=False):
        # Dummy mask generator: Load original image, apply some transformation
        img = Image.open(image_path).convert("L")  # Grayscale
        img = img.resize((300, 300))

        # Create a dummy mask: Predicted or True
        img_array = np.array(img)
        if predicted:
            img_array = (img_array > 128).astype(np.uint8) * 255  # Thresholded mask
        else:
            img_array = (img_array < 128).astype(np.uint8) * 255  # Inverted mask
        return img_array

    def display_mask(self, mask_array, label):
        height, width = mask_array.shape
        bytes_per_line = width
        q_image = QImage(
            mask_array.data, width, height, bytes_per_line, QImage.Format_Grayscale8
        )
        pixmap = QPixmap.fromImage(q_image)
        label.setPixmap(pixmap.scaled(300, 300, Qt.KeepAspectRatio))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = ImageSegmentationViewer()
    viewer.show()
    sys.exit(app.exec_())
