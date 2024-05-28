from PyQt5.QtWidgets import * 
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class ModelTrainingDialog(QWidget):

    signal_model_name = pyqtSignal(str)

    def __init__(self, dataHandler, parent=None):

        super().__init__(parent)

        self.dataHandler = dataHandler
        self.loss_train = []
        self.loss_valid = []
        self.epoch = []

        self.model_combo_box = QComboBox()
        self.model_combo_box.addItems(['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'])

        self.filter_spin_box = QSpinBox()
        self.filter_spin_box.setValue(64)

        self.kernel_combo_box = QComboBox()
        self.kernel_combo_box.addItems(['1', '3', '5', '7', '9', '11'])
        self.kernel_combo_box.setCurrentText('3')

        self.batch_size_training_spin_box = QSpinBox()
        self.batch_size_training_spin_box.setValue(32)

        self.number_of_epochs_spin_box = QSpinBox()
        self.number_of_epochs_spin_box.setValue(50)

        self.reset_weight_check_box = QCheckBox()

        self.start_button = QPushButton('Start Training')
        self.stop_button = QPushButton('Stop')
        self.progress_bar = QProgressBar()


        # self.setStyle(QCommonStyle())
        # form_layout = QFormLayout()
        # form_layout.addRow('Model', self.model_combo_box)
        # form_layout.addRow('Number of Filters', self.filter_spin_box)
        # form_layout.addRow('Kernel Size', self.kernel_combo_box)
        # form_layout.addRow('Batch Size Training', self.batch_size_training_spin_box)
        # form_layout.addRow('Number of Epochs', self.number_of_epochs_spin_box)
        # form_layout.addRow(self.start_button, self.abort_button)
        # form_layout.addRow(self.progress_bar)

        grid_layout = QGridLayout()
        grid_layout.addWidget(QLabel('Model'), 0, 0)
        grid_layout.addWidget(self.model_combo_box, 0, 1)
        grid_layout.addWidget(QLabel('Number of Filters'), 1, 0)
        grid_layout.addWidget(self.filter_spin_box, 1, 1)
        grid_layout.addWidget(QLabel('Kernel Size'), 2, 0)
        grid_layout.addWidget(self.kernel_combo_box, 2, 1)
        grid_layout.addWidget(QLabel('Batch Size Training'), 3, 0)
        grid_layout.addWidget(self.batch_size_training_spin_box, 3, 1)
        grid_layout.addWidget(QLabel('Number of Epochs'), 4, 0)
        grid_layout.addWidget(self.number_of_epochs_spin_box, 4, 1)
        grid_layout.addWidget(QLabel('Reset Model Weights'), 5, 0)
        grid_layout.addWidget(self.reset_weight_check_box, 5, 1)

        grid_layout.addWidget(self.start_button, 6, 0,1,2)
        grid_layout.addWidget(self.stop_button, 7, 0,1,2)
        grid_layout.addWidget(self.progress_bar, 8, 0,1,2)

        self.formWidget = QFrame()
        self.formWidget.setLayout(grid_layout)
        self.formWidget.setFrameStyle(QFrame.Panel | QFrame.Raised)
        self.formWidget.setFixedSize(self.formWidget.minimumSizeHint())

        self.plotWidget = LossPlot()
        self.plotWidget.setFrameStyle(QFrame.Panel | QFrame.Raised)
        self.plotWidget.setFixedHeight(self.formWidget.height())
        self.plotWidget.setFixedWidth(int(self.formWidget.height()*1.5))
        self.plotWidget.figure.tight_layout()
        self.plotWidget.figure.subplots_adjust(left=0.15,bottom=0.1)

        hbox_layout = QHBoxLayout()
        hbox_layout.addWidget(self.formWidget)
        hbox_layout.addWidget(self.plotWidget)

        self.setLayout(hbox_layout)

        # connections
        self.start_button.clicked.connect(self.start_training)
        self.stop_button.clicked.connect(self.stop_training)
        self.dataHandler.signal_progress.connect(self.progress_bar.setValue)
        self.dataHandler.signal_update_loss.connect(self.update_loss)
        self.dataHandler.signal_training_complete.connect(self.on_training_complete)

    def start_training(self):

        model_name = self.model_combo_box.currentText() + '-' + str(self.filter_spin_box.value()) + '-' + str(self.kernel_combo_box.currentText())
        self.signal_model_name.emit(model_name)

        self.loss_train = []
        self.loss_valid = []
        self.epoch = []

        self.model_combo_box.setEnabled(False)
        self.filter_spin_box.setEnabled(False)
        self.kernel_combo_box.setEnabled(False)
        self.batch_size_training_spin_box.setEnabled(False)
        self.number_of_epochs_spin_box.setEnabled(False)
        self.reset_weight_check_box.setEnabled(False)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        self.dataHandler.start_training(
            self.model_combo_box.currentText(),
            self.filter_spin_box.value(),
            int(self.kernel_combo_box.currentText()),
            self.batch_size_training_spin_box.value(),
            self.number_of_epochs_spin_box.value(),
            self.reset_weight_check_box.isChecked()
        )


    def stop_training(self):

        self.dataHandler.stop_training()
        self.model_combo_box.setEnabled(True)
        self.filter_spin_box.setEnabled(True)
        self.kernel_combo_box.setEnabled(True)
        self.batch_size_training_spin_box.setEnabled(True)
        self.number_of_epochs_spin_box.setEnabled(True)
        self.reset_weight_check_box.setEnabled(True)
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def update_loss(self, epoch, loss_train, loss_valid):

        self.loss_train.append(loss_train)
        self.loss_valid.append(loss_valid)
        self.epoch.append(epoch)

        self.plotWidget.ax.clear()
        self.plotWidget.ax.plot(self.epoch, self.loss_train, 'r', label='Training Loss')
        self.plotWidget.ax.plot(self.epoch, self.loss_valid, 'b', label='Validation Loss')
        self.plotWidget.ax.legend()
        self.plotWidget.ax.set_yscale('log')
        self.plotWidget.canvas.draw()

    def on_training_complete(self):
        self.dataHandler.stop_training()
        self.model_combo_box.setEnabled(True)
        self.filter_spin_box.setEnabled(True)
        self.kernel_combo_box.setEnabled(True)
        self.batch_size_training_spin_box.setEnabled(True)
        self.number_of_epochs_spin_box.setEnabled(True)
        self.reset_weight_check_box.setEnabled(True)
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)


'''
class ModelTrainingDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Create the dock widgets
        dock_left = QDockWidget("Left Dock")
        dock_right = QDockWidget("Right Dock")

        # Add the widgets to the dock widgets
        dock_left.setWidget(QLabel("Left Widget"))
        dock_right.setWidget(QLabel("Right Widget"))

        # Create the dock area
        dock_area = QMainWindow()
        dock_area.addDockWidget(Qt.LeftDockWidgetArea, dock_left)
        dock_area.addDockWidget(Qt.RightDockWidgetArea, dock_right)

        # Add the dock area to the dialog layout
        layout = QVBoxLayout()
        layout.addWidget(dock_area)
        self.setLayout(layout)
'''


class LossPlot(QFrame):

    def __init__(self, parent=None):

        super().__init__(parent)
        
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)

        self.ax = self.figure.subplots()
        # ax.get_xaxis().set_ticks([])
        # ax.tick_params(bottom=False)
        self.ax.set_xlabel('Epoch')
        self.ax.set_ylabel('Loss')
        self.ax.set_yscale('log')
        self.ax.legend()
        # self.canvas.setMinimumSize(self.canvas.sizeHint())

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)


if __name__ == '__main__':

    import threading
    import time
    import sys

    class Dummy(QObject):

        signal_progress = pyqtSignal(int)
        signal_update_loss = pyqtSignal(int,float,float)
        signal_training_complete = pyqtSignal()

        def __init__(self):
            QObject.__init__(self)
            self.stop_requested = False

        def start_training(self, *args, **kwargs):
            self.stop_requested = False
            self.thread = threading.Thread(target=self.train)
            self.thread.start()
    
        def stop_training(self):
            self.stop_requested = True

        def train(self):
            for i in range(100):
                if self.stop_requested:
                    self.signal_training_complete.emit()
                    break
                else:
                    time.sleep(0.1)
                    self.signal_progress.emit(i+1)
                    self.signal_update_loss.emit(i,i**2,i**2.1)
            self.signal_training_complete.emit()

    class MainWindow(QMainWindow):

        def __init__(self, parent=None):
            super().__init__(parent)

            self.dummy = Dummy()

            self.parameter_dialog = None
            self.parameter_dialog = ModelTrainingDialog(self.dummy)

            self.setWindowTitle('Training Parameters')

            self.show_dialog_button = QPushButton('Show Dialog')
            self.show_dialog_button.clicked.connect(self.show_dialog)
            self.setCentralWidget(self.show_dialog_button)


        def show_dialog(self):
            # if not self.parameter_dialog:
            #     self.parameter_dialog = ModelTrainingDialog(self,self.dummy)
            self.parameter_dialog.show()

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())