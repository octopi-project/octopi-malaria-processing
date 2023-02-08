# https://stackoverflow.com/questions/45896291/how-to-show-image-and-text-at-same-cell-in-qtablewidget-in-pyqt

from PyQt5.QtWidgets import * 
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import QtCore
import os
import numpy as np
import pandas as pd

class CustomWidget(QWidget):
    def __init__(self, text, img, parent=None):
        QWidget.__init__(self, parent)

        self._text = text
        self._img = img

        self.setLayout(QVBoxLayout())
        self.lbPixmap = QLabel(self)
        self.lbText = QLabel(self)
        self.lbText.setAlignment(Qt.AlignCenter)

        self.layout().addWidget(self.lbPixmap)
        self.layout().addWidget(self.lbText)

        self.initUi()

    def initUi(self):
        # self.lbPixmap.setPixmap(QPixmap.fromImage(self._img).scaled(self.lbPixmap.size(),Qt.KeepAspectRatio))
        self.lbPixmap.setScaledContents(True)
        self.lbPixmap.setPixmap(QPixmap.fromImage(self._img))
        self.lbText.setText(self._text)

    @pyqtProperty(str)
    def img(self):
        return self._img

    @img.setter
    def total(self, value):
        if self._img == value:
            return
        self._img = value
        self.initUi()

    @pyqtProperty(str)
    def text(self):
        return self._text

    @text.setter
    def text(self, value):
        if self._text == value:
            return
        self._text = value
        self.initUi()

class TableWidget(QTableWidget):
    def __init__(self, rows = 20, columns = 10, parent=None):
        QTableWidget.__init__(self, parent)
        self.parent = parent
        self.setColumnCount(columns)
        self.setRowCount(rows)        
        self.cellClicked.connect(self.onCellClicked)

    def populate(self, images, texts):
        for i in range(self.rowCount()):
            for j in range(self.columnCount()):
                random_image = np.random.randint(0, 255, size=(256, 256, 3), dtype=np.uint8)
                qimage = QImage(random_image.data, random_image.shape[1], random_image.shape[0], QImage.Format_RGB888)
                lb = CustomWidget(str(i) + '_' + str(j), qimage)
                self.setCellWidget(i, j, lb)
        self.resizeColumnsToContents()
        self.resizeRowsToContents()
        # self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setFixedWidth(self.horizontalHeader().length()+80)
        self.setFixedHeight(5*self.rowHeight(0)+40)

    @pyqtSlot(int, int)
    def onCellClicked(self, row, column):
        w = self.cellWidget(row, column)
        print(w.text, w.img)

class GalleryViewWidget(QFrame):
    def __init__(self, rows = 10, columns = 10, parent=None,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tableWidget = TableWidget(rows,columns,parent=self)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setMinimum(0)
        self.slider.setValue(0)
        self.slider.setSingleStep(1)

        self.entry = QSpinBox()
        self.entry.setMinimum(0) 
        self.entry.setValue(0)
        
        vbox = QVBoxLayout()
        vbox.addWidget(self.tableWidget)
        grid = QGridLayout()
        grid.addWidget(self.entry,0,0)
        grid.addWidget(self.slider,0,1)
        vbox.addLayout(grid)
        self.setLayout(vbox)

        # self.tableWidget.populate(None,None)
        
        # connections
        self.entry.valueChanged.connect(self.slider.setValue)
        self.slider.valueChanged.connect(self.entry.setValue)
        self.entry.valueChanged.connect(self.update_page)

    def update_page(self):
        self.tableWidget.populate(None,None)

    def set_total_pages(self,n):
        self.slider.setMaximum(n)
        self.entry.setMaximum(n) 

    def populate_page0(self):
        self.tableWidget.populate(None,None)


class DataLoaderWidget(QFrame):
    def __init__(self, dataHandler, main=None, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.dataHandler = dataHandler
            self.setFrameStyle(QFrame.Panel | QFrame.Raised)
            # create widgets
            self.button_load_images = QPushButton('Load Images')
            self.button_load_predictions = QPushButton('Load Predictions')
            self.button_load_embeddings = QPushButton('Load Embeddings')
            self.button_load_images.setIcon(QIcon('icon/folder.png'))
            self.button_load_predictions.setIcon(QIcon('icon/folder.png'))
            self.button_load_embeddings.setIcon(QIcon('icon/folder.png'))
            self.lineEdit_images = QLineEdit()
            self.lineEdit_images.setText('click the button to load')
            self.lineEdit_images.setReadOnly(True)
            self.lineEdit_predictions = QLineEdit()
            self.lineEdit_predictions.setText('click the button to load')
            self.lineEdit_predictions.setReadOnly(True)
            self.lineEdit_embeddings = QLineEdit()
            self.lineEdit_embeddings.setText('click the button to load')
            self.lineEdit_embeddings.setReadOnly(True)
            # layout
            layout = QGridLayout()
            layout.addWidget(self.button_load_images,0,0)
            layout.addWidget(self.button_load_predictions,1,0)
            layout.addWidget(self.button_load_embeddings,2,0)
            layout.addWidget(self.lineEdit_images,0,1)
            layout.addWidget(self.lineEdit_predictions,1,1)
            layout.addWidget(self.lineEdit_embeddings,2,1)
            self.setLayout(layout)
            # connect
            self.button_load_images.clicked.connect(self.load_images)
            self.button_load_predictions.clicked.connect(self.load_predictions)
            self.button_load_embeddings.clicked.connect(self.load_embeddings)

    def load_images(self):
        dialog = QFileDialog()
        filename, _filter = dialog.getOpenFileName(None,'Open File','.','npy files (*.npy)')
        if filename:
            self.config_filename = filename
            if(os.path.isfile(filename)==True):
                if self.dataHandler.load_images(filename) == 0:
                    self.lineEdit_images.setText(filename)

    def load_predictions(self):
        dialog = QFileDialog()
        filename, _filter = dialog.getOpenFileName(None,'Open File','.','csv files (*.csv)')
        if filename:
            self.config_filename = filename
            if(os.path.isfile(filename)==True):
                if self.dataHandler.load_output(filename) == 0:
                    self.lineEdit_predictions.setText(filename)

    def load_embeddings(self):
        dialog = QFileDialog()
        filename, _filter = dialog.getOpenFileName(None,'Open File','.','npy files (*.npy)')
        if filename:
            self.config_filename = filename
            if(os.path.isfile(filename)==True):
                self.lineEdit_embeddings.setText(filename)
                self.dataHandler.load_embeddings(filename)

class DataHandler(QObject):

    signal_populate_page0 = pyqtSignal()
    signal_set_total_page_count = pyqtSignal(int)

    def __init__(self):
        QObject.__init__(self)
        self.images = None
        self.output_pd = None
        self.embeddings = None
        self.images_loaded = False
        self.output_loaded = False
        self.embeddings_loaded = False
        self.n_images_per_page = None

    def load_images(self,path):
        self.images = np.load(path)
        
        if self.output_loaded:
            if self.images.shape[0] != self.output_pd.shape[0]:
                print('! dimension mismatch')
                return 1

        self.images_loaded = True
        if self.images_loaded & self.output_loaded == True:
            self.signal_set_total_page_count.emit(int(np.ceil(self.get_number_of_rows()/self.n_images_per_page)))
            self.signal_populate_page0.emit()
        return 0

    def load_output(self,path):
        self.output_pd = pd.read_csv(path)
        
        if self.images_loaded:
            if self.images.shape[0] != self.output_pd.shape[0]:
                print('! dimension mismatch')
                return 1

        self.output_loaded = True
        if self.images_loaded & self.output_loaded == True:
            self.signal_set_total_page_count.emit(int(np.ceil(self.get_number_of_rows()/self.n_images_per_page)))
            self.signal_populate_page0.emit()
        return 0

    def load_embeddings(self,path):
        self.embeddings = np.load(path)
        self.embeddings_loaded = True

    def set_number_of_images_per_page(self,n):
        self.n_images_per_page = n

    def get_number_of_rows(self):
        return self.images.shape[0] 


class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        num_rows = 20
        num_cols = 10

        # core
        self.dataHandler = DataHandler()
        self.dataHandler.set_number_of_images_per_page(num_rows*num_cols)

        # widgets
        self.dataLoaderWidget = DataLoaderWidget(self.dataHandler)
        self.gallery = GalleryViewWidget(num_rows,num_cols)

        layout = QVBoxLayout()
        layout.addWidget(self.dataLoaderWidget)
        layout.addWidget(self.gallery)

        self.centralWidget = QWidget()
        self.centralWidget.setLayout(layout)
        # self.centralWidget.setFixedWidth(self.centralWidget.minimumSizeHint().width())
        self.setCentralWidget(self.centralWidget)

        # connect
        self.dataHandler.signal_populate_page0.connect(self.gallery.populate_page0)
        self.dataHandler.signal_set_total_page_count.connect(self.gallery.set_total_pages)

    def closeEvent(self, event):
        event.accept()

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    # tw = TableWidget()
    # tw.populate(None,None)
    # tw.show()
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())