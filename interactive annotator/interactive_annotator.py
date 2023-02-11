# https://stackoverflow.com/questions/45896291/how-to-show-image-and-text-at-same-cell-in-qtablewidget-in-pyqt

from PyQt5.QtWidgets import * 
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import QtCore
import os
from datetime import datetime
import numpy as np
import pandas as pd
import cv2
import glob
import torch
from sklearn.neighbors import KNeighborsClassifier
import models
import utils

##########################################################
################  Default configurations  ################
##########################################################
num_rows = 6
MAX_NUM_ROWS_DISPLAYED_PER_PAGE = 6
num_cols = 10
SCALE_FACTOR = 8
model_spec = {'model':'resnet18','n_channels':4,'n_filters':64,'n_classes':1,'kernel_size':3,'stride':1,'padding':1}
batch_size_inference = 2048
KNN_METRIC = 'cosine'
COLOR_DICT = {0:QColor(150,200,250),1:QColor(250,200,200),9:QColor(250,250,200)} # 0: nonparasites, 1: parasites, 9: not sure

# on mac
# num_rows = 2
# num_cols = 4

##########################################################
#### start of loading machine specific configurations ####
##########################################################
config_files = glob.glob('.' + '/' + 'configurations*.txt')
if config_files:
    if len(config_files) > 1:
        print('multiple configuration files found, the program will exit')
        exit()
    exec(open(config_files[0]).read())
##########################################################
##### end of loading machine specific configurations #####
##########################################################


def generate_overlay(image):
    images_fluorescence = image[:,2::-1,:,:]
    images_dpc = image[:,3,:,:]
    images_dpc = np.expand_dims(images_dpc, axis=1)
    images_dpc = np.repeat(images_dpc, 3, axis=1) # make it rgb
    images_overlay = 0.64*images_fluorescence + 0.36*images_dpc
    images = images_overlay.transpose(0,2,3,1)
    # frame = np.hstack([img_dpc,img_fluorescence,img_overlay]).astype('uint8')
    # print(images.shape)
    return images

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
        self.lbPixmap.setPixmap(QPixmap.fromImage(self._img))
        self.lbPixmap.setScaledContents(True)
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
    def __init__(self, rows = 10, columns = 10, parent=None):
        QTableWidget.__init__(self, parent)
        self.parent = parent
        self.num_cols = columns
        self.num_rows = rows
        self.setColumnCount(columns)
        self.setRowCount(rows)        
        self.cellClicked.connect(self.onCellClicked)

    def populate_simulate(self, images, texts):
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
        self.setFixedHeight(5*self.rowHeight(0)+80)

    def populate(self, images, texts, annotations):
        for i in range(self.rowCount()):
            for j in range(self.columnCount()):
                idx = i*self.num_cols + j
                if idx >= images.shape[0]:
                    self.setCellWidget(i, j, None)
                else:
                    image = images[idx,].astype(np.uint8)
                    # resize
                    scale_factor = SCALE_FACTOR
                    new_height, new_width = int(image.shape[0] * scale_factor), int(image.shape[1] * scale_factor)
                    image = cv2.resize(image,(new_width, new_height),interpolation=cv2.INTER_NEAREST)
                    # image = np.random.randint(0, 255, size=(128, 128, 3), dtype=np.uint8)
                    # print(image.shape)
                    text = texts[idx]
                    qimage = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)
                    lb = CustomWidget(text, qimage)
                    self.setCellWidget(i, j, lb)
                    # set color according to annotation
                    if annotations[idx] != -1:
                        self.setItem(i, j, QTableWidgetItem())
                        self.item(i,j).setBackground(COLOR_DICT[annotations[idx]])
                    else:
                        # clear background - to improve the code
                        self.setItem(i, j, QTableWidgetItem())
                        palette = self.palette()
                        default_color = palette.color(QPalette.Window)
                        self.item(i,j).setBackground(default_color)
        self.resizeColumnsToContents()
        self.resizeRowsToContents()
        self.setFixedWidth(self.horizontalHeader().length()+80)
        self.setFixedHeight(min(self.num_rows,MAX_NUM_ROWS_DISPLAYED_PER_PAGE)*self.rowHeight(0)+80)
        # self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

    def get_selected_cells(self):
        list_of_selected_cells = []
        for index in self.selectedIndexes():
             # list_of_selected_cells.append((index.row(),index.column()))
             list_of_selected_cells.append( index.row()*self.num_cols + index.column() )
        return(list_of_selected_cells)

    @pyqtSlot(int, int)
    def onCellClicked(self, row, column):
        w = self.cellWidget(row, column)
        print(w.text)

###########################################################################################
#####################################  Gallery View  ######################################
###########################################################################################

class GalleryViewWidget(QFrame):

    signal_switchTab = pyqtSignal()
    signal_similaritySearch = pyqtSignal(np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray)

    def __init__(self, rows = 10, columns = 10, dataHandler=None, dataHandler2=None, is_main_gallery=False, parent=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataHandler = dataHandler
        self.dataHandler2 = dataHandler2 # secondary datahandler (in the similarity search gallery, this is linked to the full data)
        self.is_main_gallery = is_main_gallery
        self.image_id = None # for storing the ID of currently displayed images

        self.tableWidget = TableWidget(rows,columns,parent=self)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setMinimum(0)
        self.slider.setValue(0)
        self.slider.setSingleStep(1)

        self.entry = QSpinBox()
        self.entry.setMinimum(0) 
        self.entry.setValue(0)

        self.btn_search = QPushButton('Search Similar Images')

        vbox = QVBoxLayout()
        vbox.addWidget(self.tableWidget)
        grid = QGridLayout()
        grid.addWidget(self.entry,0,0)
        grid.addWidget(self.slider,0,1)
        # if self.is_main_gallery:
        grid.addWidget(self.btn_search,2,0,2,2)
        vbox.addLayout(grid)
        self.setLayout(vbox)
        
        # connections
        self.entry.valueChanged.connect(self.slider.setValue)
        self.slider.valueChanged.connect(self.entry.setValue)
        self.entry.valueChanged.connect(self.update_page)
        self.btn_search.clicked.connect(self.do_similarity_search)

    def update_page(self):
        # self.tableWidget.populate_simulate(None,None)
        if self.dataHandler is not None:
            images,texts,self.image_id,annotations = self.dataHandler.get_page(self.entry.value())
            self.tableWidget.populate(images,texts,annotations)
        else:
            self.tableWidget.populate_simulate(None,None)

    def set_total_pages(self,n):
        self.slider.setMaximum(n-1)
        self.entry.setMaximum(n-1) 
        self.slider.setTickInterval(int(np.ceil(n/10)))

    def populate_page0(self):
        self.entry.setValue(0)
        self.update_page()

    def do_similarity_search(self):
        selected_images = self.tableWidget.get_selected_cells()
        # ensure only one image is selected
        if len(selected_images) != 1:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText("No image or more than one images selected. Please select one image.")
            msg.exec_()
            return
        '''
        # check if embedding has been loaded
        if self.dataHandler.embeddings_loaded != True:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText("Load the embeddings first.")
            msg.exec_()
            return
        '''
        # convert to global id
        selected_image = self.image_id[selected_images[0]]
        # find similar images
        k = 200
        print( 'finding ' + str(k) + ' images similar to ' + str(selected_image) )
        if self.is_main_gallery:
            images, indices, scores, distances, annotations = self.dataHandler.find_similar_images(selected_image,k)
        else:
            images, indices, scores, distances, annotations = self.dataHandler2.find_similar_images(selected_image,k)
        # emit the results
        self.signal_similaritySearch.emit(images,indices,scores,distances,annotations)
        self.signal_switchTab.emit()

###########################################################################################
##################################  Data Loader Widget  ###################################
###########################################################################################

class DataLoaderWidget(QFrame):

    def __init__(self, dataHandler, main=None, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.dataHandler = dataHandler
            self.setFrameStyle(QFrame.Panel | QFrame.Raised)
            # create widgets
            self.button_load_model = QPushButton('Load Model')
            self.button_load_images = QPushButton('Load Images')
            self.button_load_annotations = QPushButton('Load Annotations')
            self.button_load_model.setIcon(QIcon('icon/folder.png'))
            self.button_load_images.setIcon(QIcon('icon/folder.png'))
            self.button_load_annotations.setIcon(QIcon('icon/folder.png'))
            self.lineEdit_images = QLineEdit()
            self.lineEdit_images.setText('click the button to load')
            self.lineEdit_images.setReadOnly(True)
            self.lineEdit_annotations = QLineEdit()
            self.lineEdit_annotations.setText('click the button to load')
            self.lineEdit_annotations.setReadOnly(True)
            self.lineEdit_model = QLineEdit()
            self.lineEdit_model.setText('click the button to load')
            self.lineEdit_model.setReadOnly(True)
            # layout
            layout = QGridLayout()
            layout.addWidget(self.button_load_images,1,0)
            layout.addWidget(self.button_load_annotations,2,0)
            layout.addWidget(self.button_load_model,0,0)
            layout.addWidget(self.lineEdit_images,1,1)
            layout.addWidget(self.lineEdit_annotations,2,1)
            layout.addWidget(self.lineEdit_model,0,1)
            self.setLayout(layout)
            # connect
            self.button_load_images.clicked.connect(self.load_images)
            self.button_load_annotations.clicked.connect(self.load_annotations)
            self.button_load_model.clicked.connect(self.load_model)

    def load_images(self):
        dialog = QFileDialog()
        filename, _filter = dialog.getOpenFileName(None,'Open File','.','npy files (*.npy)')
        if filename:
            self.config_filename = filename
            if(os.path.isfile(filename)==True):
                if self.dataHandler.load_images(filename) == 0:
                    self.lineEdit_images.setText(filename)

    def load_annotations(self):
        dialog = QFileDialog()
        filename, _filter = dialog.getOpenFileName(None,'Open File','.','csv files (*.csv)')
        if filename:
            self.config_filename = filename
            if(os.path.isfile(filename)==True):
                if self.dataHandler.load_annotations(filename) == 0:
                    self.lineEdit_annotations.setText(filename)

    def load_model(self):
        dialog = QFileDialog()
        filename, _filter = dialog.getOpenFileName(None,'Open File','.','model (*.pt)')
        if filename:
            self.config_filename = filename
            if(os.path.isfile(filename)==True):
                self.lineEdit_model.setText(filename)
                self.dataHandler.load_model(filename)

###########################################################################################
#####################################  Data Handaler  #####################################
###########################################################################################

class DataHandler(QObject):

    signal_populate_page0 = pyqtSignal()
    signal_set_total_page_count = pyqtSignal(int)

    def __init__(self,is_for_similarity_search=False):
        QObject.__init__(self)
        self.is_for_similarity_search = is_for_similarity_search
        self.images = None
        self.image_path = None
        self.data_pd = None
        self.data_pd = None # annotation + prediction score + index
        self.embeddings = None
        self.model_loaded = False
        self.images_loaded = False
        self.embeddings_loaded = False
        self.annotations_loaded = False
        self.n_images_per_page = None
        self.spot_idx_sorted = None

    def load_model(self,path):
        self.model = models.ResNet(model=model_spec['model'],n_channels=model_spec['n_channels'],n_filters=model_spec['n_filters'],
            n_classes=model_spec['n_classes'],kernel_size=model_spec['kernel_size'],stride=model_spec['stride'],padding=model_spec['padding'])
        if torch.cuda.is_available():
            self.model.load_state_dict(torch.load(path))
        else:
            self.model.load_state_dict(torch.load(path,map_location=torch.device('cpu')))
        self.model_loaded = True
        # if the images are already loaded, run the model
        if self.images_loaded:
            self.run_mode()
            self.signal_populate_page0.emit()

    def load_images(self,path):

        self.images = np.load(path)
        self.image_path = path
        
        if self.annotations_loaded:
            if self.images.shape[0] != self.data_pd.shape[0]:
                print('! dimension mismatch')
                return 1
        else:
            self.data_pd = pd.DataFrame({'index':np.arange(self.images.shape[0]),'annotation':-1})
            self.data_pd.set_index('index')
            # print(self.data_pd)

        self.images_loaded = True
        
        # run the model if the model has been loaded
        if self.model_loaded == True:
            self.run_model()

        # display the images
        self.signal_set_total_page_count.emit(int(np.ceil(self.get_number_of_rows()/self.n_images_per_page)))
        self.signal_populate_page0.emit()
        
        return 0

    def run_model(self):
        predictions, features = utils.generate_predictions_and_features(self.model,self.images,batch_size_inference)
        output_pd = pd.DataFrame({'index':np.arange(self.images.shape[0]),'output':predictions[:,0]})
        if 'output' in self.data_pd:
            self.data_pd = self.data_pd.drop(columns=['output'])
        self.data_pd = self.data_pd.merge(output_pd,on='index')
        print(self.data_pd)
        
        # sort the predictions
        self.data_pd = self.data_pd.sort_values('output',ascending=False)
        self.spot_idx_sorted = self.data_pd['index'].to_numpy().astype(int)
        
        self.signal_set_total_page_count.emit(int(np.ceil(self.get_number_of_rows()/self.n_images_per_page)))
        self.signal_populate_page0.emit()

        # embeddings
        self.embeddings = features
        self.embeddings_loaded = True
        self.neigh = KNeighborsClassifier(metric=KNN_METRIC)
        self.neigh.fit(self.embeddings, np.zeros(self.embeddings.shape[0]))

    def load_annotations(self,path):
        # load the annotation
        if self.data_pd is not None:
            annotation_pd = pd.read_csv(path,index_col='index')
            self.data_pd = self.data_pd.drop(columns=['annotation'])
            self.data_pd = self.data_pd.merge(annotation_pd,on='index')
            self.signal_populate_page0.emit() # update the display
        else:
            self.data_pd = pd.read_csv(path)

        # size match check
        if self.images_loaded:
            if self.images.shape[0] != self.data_pd.shape[0]:
                print('! dimension mismatch')
                return 1

        # sort the annotations
        self.data_pd = self.data_pd.sort_values('annotation',ascending=False)
        self.spot_idx_sorted = self.data_pd['index'].to_numpy().astype(int)
        self.annotations_loaded = True
        
        # update the display if images have been loaded already
        if self.images_loaded:
            self.signal_populate_page0.emit()
        
        return 0
        
    def set_number_of_images_per_page(self,n):
        self.n_images_per_page = n

    def get_number_of_rows(self):
        return self.images.shape[0]

    def get_page(self,page_number):
        idx_start = self.n_images_per_page*page_number
        idx_end = min(self.n_images_per_page*(page_number+1),self.images.shape[0])
        images = generate_overlay(self.images[self.spot_idx_sorted[idx_start:idx_end],:,:,:])
        texts = []
        image_id = []
        annotations = []
        for i in range(idx_start,idx_end):
            # texts.append( '[' + str(i) + ']  ' + str(self.data_pd.iloc[i]['index']) + ': ' + "{:.2f}".format(self.data_pd.iloc[i]['output']))
            if self.is_for_similarity_search:
                texts.append( '[' + str(i) + ']  : ' + "{:.2f}".format(self.data_pd.iloc[i]['scores']) + '\n{:.1e}'.format(self.data_pd.iloc[i]['distances'])) # .at would use the idx before sorting # + '{:.1e}'.format(self.data_pd.iloc[i]['distances'])

            else:
                texts.append( '[' + str(i) + ']  : ' + "{:.2f}".format(self.data_pd.iloc[i]['output'])) # .at would use the idx before sorting
            image_id.append(int(self.data_pd.iloc[i]['index']))
            annotations.append(int(self.data_pd.iloc[i]['annotation']))
        return images, texts, image_id, annotations

    def find_similar_images(self,idx,k=200):
        self.neigh.set_params(n_neighbors=k+1)
        distances, indices = self.neigh.kneighbors(self.embeddings[idx,].reshape(1, -1), return_distance=True)
        distances = distances.squeeze()
        indices = indices.squeeze()
        images = self.images[indices,]
        scores = self.data_pd.loc[indices]['output'].to_numpy() # use the presorting idx
        annotations = self.data_pd.loc[indices]['annotation'].to_numpy() # use the presorting idx
        return images[1:,], indices[1:], scores[1:], distances[1:], annotations[1:]

    def populate_similarity_search(self,images,indices,scores,distances,annotations):
        self.images = images
        self.data_pd = pd.DataFrame({'index':indices, 'scores':scores, 'distances':distances, 'annotation':annotations})
        # print(self.data_pd)
        self.images_loaded = True
        # self.spot_idx_sorted = self.data_pd['index'].to_numpy().astype(int)
        self.spot_idx_sorted = np.arange(self.images.shape[0]).astype(int)
        self.signal_set_total_page_count.emit(int(np.ceil(self.get_number_of_rows()/self.n_images_per_page)))
        self.signal_populate_page0.emit()

    def update_annotation(self,index,annotation):
        # condition = df['index'].isin(index), df.loc[condition, 'annotation'] = annotation 
        # to-do: support dealing with multiple datasets using multi-level indexing
        self.data_pd.loc[index,'annotation'] = annotation

    def save_annotations(self):
        if self.image_path:
            # remove the prediction score
            if 'output' in self.data_pd.columns:
                self.data_pd = self.data_pd.drop(columns=['output'])
            # save the annotations
            current_time = datetime.now().strftime('%Y-%m-%d_%H-%M')
            self.data_pd.to_csv(os.path.splitext(self.image_path)[0] + '_annotations_' + str(sum(self.data_pd['annotation']!=-1)) + '_' + str(self.data_pd.shape[0])  + '_' + current_time + '.csv')
            # remove the temporary file
            tmp_file = os.path.dirname(self.image_path)+'_tmp.csv'
            if os.path.exists(tmp_file):
                os.remove(tmp_file)

###########################################################################################
#####################################  Main Window  #######################################
###########################################################################################

class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # core
        self.dataHandler = DataHandler()
        self.dataHandler.set_number_of_images_per_page(num_rows*num_cols)

        self.dataHandler_similarity = DataHandler(is_for_similarity_search=True)
        self.dataHandler_similarity.set_number_of_images_per_page(num_rows*num_cols)

        # widgets
        self.dataLoaderWidget = DataLoaderWidget(self.dataHandler)
        self.gallery = GalleryViewWidget(num_rows,num_cols,self.dataHandler,is_main_gallery=True)
        self.gallery_similarity = GalleryViewWidget(num_rows,num_cols,self.dataHandler_similarity,dataHandler2=self.dataHandler)

        # tab widget
        self.gallery_tab = QTabWidget()
        self.gallery_tab.addTab(self.gallery,'Full Dataset')
        self.gallery_tab.addTab(self.gallery_similarity,'Similarity Search')

        layout = QVBoxLayout()
        layout.addWidget(self.dataLoaderWidget)
        # layout.addWidget(self.gallery)
        layout.addWidget(self.gallery_tab)

        self.centralWidget = QWidget()
        self.centralWidget.setLayout(layout)
        # self.centralWidget.setFixedWidth(self.centralWidget.minimumSizeHint().width())
        self.setCentralWidget(self.centralWidget)

        # connect
        self.dataHandler.signal_set_total_page_count.connect(self.gallery.set_total_pages)
        self.dataHandler.signal_populate_page0.connect(self.gallery.populate_page0)
        
        self.dataHandler_similarity.signal_set_total_page_count.connect(self.gallery_similarity.set_total_pages)
        self.dataHandler_similarity.signal_populate_page0.connect(self.gallery_similarity.populate_page0)

        self.gallery.signal_similaritySearch.connect(self.dataHandler_similarity.populate_similarity_search)
        self.gallery.signal_switchTab.connect(self.switch_tab)

        self.gallery_similarity.signal_similaritySearch.connect(self.dataHandler_similarity.populate_similarity_search)

    def switch_tab(self):
        self.gallery_tab.setCurrentIndex(1)

    def closeEvent(self, event):
        self.dataHandler.save_annotations()
        event.accept()

###########################################################################################
#####################################  The Program  #######################################
###########################################################################################
if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    # tw = TableWidget()
    # tw.populate(None,None)
    # tw.show()
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())