from PyQt5.QtWidgets import * 
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import QtCore
import pyqtgraph.dockarea as dock
import pyqtgraph as pg
import os
from datetime import datetime
import numpy as np
import pandas as pd
import cv2
import glob
import torch
import functools
import operator
from sklearn.neighbors import KNeighborsClassifier
import models
import utils

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from mpl_interactions import ioff, panhandler, zoom_factory
from utils_plot import *

##########################################################
################  Default configurations  ################
##########################################################
NUM_ROWS = 6
MAX_NUM_ROWS_DISPLAYED_PER_PAGE = 6
num_cols = 10
SCALE_FACTOR = 8

K_SIMILAR_DEFAULT = 20

model_spec = {'model':'resnet18','n_channels':4,'n_filters':64,'n_classes':1,'kernel_size':3,'stride':1,'padding':1}
batch_size_inference = 2048
KNN_METRIC = 'cosine'
COLOR_DICT = {0:QColor(150,200,250),1:QColor(250,200,200),9:QColor(250,250,200)} # 0: nonparasites, 1: parasites, 9: not sure
COLOR_DICT_PLOT = {-1:'#C8C8C8',0:'#96C8FA',1:'#FAC8C8',9:'#FAFAC8'}
ANNOTATIONS_DICT = {'Label as parasite':1,'Label as non-parasite':0,'Label as unsure':9,'Remove Annotation':-1}
ANNOTATIONS_REVERSE_DICT = {-1:'not labeled',0:'non-parasite',1:'parasite',9:'unsure'}
PLOTS = ['Labels','Annotation Progress','Prediction score','Similarity','UMAP']
DEV_MODE = True

# on mac
# NUM_ROWS = 2
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

# modified from https://stackoverflow.com/questions/45896291/how-to-show-image-and-text-at-same-cell-in-qtablewidget-in-pyqt
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
        self.id = None
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

    def populate(self, images, texts, annotations, image_id):
        self.id = image_id
        for i in range(self.rowCount()):
            for j in range(self.columnCount()):
                idx = i*self.num_cols + j
                if idx >= images.shape[0]:
                    self.setCellWidget(i, j, None)
                    # clear background - to improve the code
                    self.setItem(i, j, QTableWidgetItem())
                    palette = self.palette()
                    default_color = palette.color(QPalette.Window)
                    self.item(i,j).setBackground(default_color)
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

    def set_number_of_rows(self,rows):
        self.num_rows = rows
        self.setRowCount(rows)

    @pyqtSlot(int, int)
    def onCellClicked(self, row, column):
        w = self.cellWidget(row, column)
        if w:
            idx = row*self.num_cols + column
            print(str(self.id[idx]) + ' - ' + w.text)

###########################################################################################
#####################################  Gallery View  ######################################
###########################################################################################

class GalleryViewWidget(QFrame):

    signal_switchTab = pyqtSignal()
    signal_similaritySearch = pyqtSignal(np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray)
    signal_updatePage = pyqtSignal()

    def __init__(self, rows = 10, columns = 10, dataHandler=None, dataHandler2=None, is_main_gallery=False, parent=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataHandler = dataHandler
        self.dataHandler2 = dataHandler2 # secondary datahandler (in the similarity search gallery, this is linked to the full data)
        self.is_main_gallery = is_main_gallery
        self.image_id = None # for storing the ID of currently displayed images

        self.tableWidget = TableWidget(rows,columns,parent=self)

        # page navigation
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setMinimum(0)
        self.slider.setValue(0)
        self.slider.setSingleStep(1)

        self.entry = QSpinBox()
        self.entry.setMinimum(0) 
        self.entry.setValue(0)

        self.btn_search = QPushButton('Search Similar Images')
        self.dropdown_sort = QComboBox()
        self.dropdown_sort.addItems(['Sort by prediction score','Sort by labels'])
        if self.is_main_gallery == False:
            self.dropdown_sort.insertItem(0,'Sort by similarity')
            self.dropdown_sort.blockSignals(True)
            self.dropdown_sort.setCurrentIndex(0)
            self.dropdown_sort.blockSignals(False)

        self.btn_annotations = {}
        for key in ANNOTATIONS_DICT.keys():
            self.btn_annotations[key] = QPushButton(key)
        
        vbox = QVBoxLayout()
        vbox.addWidget(self.tableWidget)
        grid = QGridLayout()
        slider = QHBoxLayout()
        slider.addWidget(self.entry)
        slider.addWidget(self.slider)
        grid.addLayout(slider,0,0,1,len(ANNOTATIONS_DICT))
        # grid.addWidget(self.entry,0,0)
        # grid.addWidget(self.slider,0,1)
        # if self.is_main_gallery:
        grid.addWidget(self.btn_search,2,0,1,len(ANNOTATIONS_DICT))
        grid.addWidget(self.dropdown_sort,3,0,1,len(ANNOTATIONS_DICT))
        i = 0
        for key in ANNOTATIONS_DICT.keys():
            grid.addWidget(self.btn_annotations[key],4,i)
            i = i + 1  
        vbox.addLayout(grid)
        self.setLayout(vbox)
        
        # connections
        self.entry.valueChanged.connect(self.slider.setValue)
        self.slider.valueChanged.connect(self.entry.setValue)
        self.entry.valueChanged.connect(self.update_page)
        self.btn_search.clicked.connect(self.do_similarity_search)
        self.dropdown_sort.currentTextChanged.connect(self.dataHandler.sort)
        self.dataHandler.signal_sorting_method.connect(self.update_displayed_sorting_method)
        for key in ANNOTATIONS_DICT.keys():
            self.btn_annotations[key].clicked.connect(functools.partial(self.assign_annotations,ANNOTATIONS_DICT[key]))

    def update_page(self):
        # clear selections
        self.tableWidget.clearSelection()
        # self.tableWidget.populate_simulate(None,None)
        if self.dataHandler is not None:
            images,texts,self.image_id,annotations = self.dataHandler.get_page(self.entry.value())
            self.tableWidget.populate(images,texts,annotations,self.image_id)
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
        # ensure a model is present
        if self.is_main_gallery:
            model_loaded = self.dataHandler.model_loaded
        else:
            model_loaded = self.dataHandler2.model_loaded
        if model_loaded == False:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText("Load a model first")
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
        print( 'finding images similar to ' + str(selected_image) )

        # find similar images        
        if self.is_main_gallery:
            images, indices, scores, distances, annotations = self.dataHandler.find_similar_images(selected_image)
        else:
            images, indices, scores, distances, annotations = self.dataHandler2.find_similar_images(selected_image)
        # emit the results
        self.signal_similaritySearch.emit(images,indices,scores,distances,annotations)
        self.signal_switchTab.emit()

    def update_displayed_sorting_method(self,sorting_method):
        self.dropdown_sort.blockSignals(True)
        self.dropdown_sort.setCurrentText(sorting_method)
        self.dropdown_sort.blockSignals(False)

    def assign_annotations(self,annotation):
        selected_images = self.tableWidget.get_selected_cells() # index in the current page
        selected_images = [i for i in selected_images if i < len(self.image_id)] # filter it 
        # selected_images = self.image_id[selected_images] # global index of the images - this does not work for list, use the method below
        if len(selected_images) == 0:
            return
        selected_images = operator.itemgetter(*selected_images)(self.image_id) # selected_images = [self.image_id[i] for i in selected_images]
        print('lable ' + str(selected_images) + ' as ' + str(annotation))
        self.dataHandler.update_annotation(selected_images,annotation)
        self.update_page()
        if self.is_main_gallery == False:
            # update for the full dataset
            self.dataHandler2.update_annotation(selected_images,annotation)
            self.signal_updatePage.emit()

    def set_number_of_rows(self,rows):
        self.tableWidget.set_number_of_rows(rows)
        self.signal_updatePage.emit()


class GalleryViewSettingsWidget(QFrame):

    signal_numRowsPerPage = pyqtSignal(int)
    signal_numImagesPerPage = pyqtSignal(int)
    signal_k_similaritySearch = pyqtSignal(int)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFrameStyle(QFrame.Panel | QFrame.Raised)

        self.entry_num_rows_per_page = QSpinBox()
        self.entry_num_rows_per_page.setMinimum(NUM_ROWS) 
        self.entry_num_rows_per_page.setValue(NUM_ROWS)
        self.entry_k_similarity_search = QSpinBox()
        self.entry_k_similarity_search.setMinimum(1) 
        self.entry_k_similarity_search.setValue(K_SIMILAR_DEFAULT)
        
        grid_settings = QGridLayout()
        grid_settings.addWidget(QLabel('Number of rows per page'),0,0)
        grid_settings.addWidget(self.entry_num_rows_per_page,0,1)
        grid_settings.addWidget(QLabel('Number of images for similary search'),0,2)
        grid_settings.addWidget(self.entry_k_similarity_search,0,3)

        self.setLayout(grid_settings)

        # connections
        self.entry_num_rows_per_page.valueChanged.connect(self.update_num_rows_per_page)
        self.entry_k_similarity_search.valueChanged.connect(self.signal_k_similaritySearch.emit)

    def update_num_rows_per_page(self,rows):
        self.signal_numImagesPerPage.emit(rows*num_cols) # to change to NUM_COLS
        self.signal_numRowsPerPage.emit(rows)

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
##########################  Training and Visualization Widget  ############################
###########################################################################################

class TrainingAndVisualizationWidget(QFrame):

    def __init__(self):

        super().__init__()
        self.setFrameStyle(QFrame.Panel | QFrame.Raised)

        self.btn_open_model_training = QPushButton("Open Model Training Dialog")
        self.label_model = QLabel()
        self.label_model.setMinimumWidth(100)
        self.label_model.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.max_points_label = QLabel("Max number of points for UMAP")
        self.entry_max_n_for_umap = QSpinBox()
        self.entry_max_n_for_umap.setMinimum(1000)
        self.entry_max_n_for_umap.setMaximum(1000000)
        self.entry_max_n_for_umap.setSingleStep(1000)
        self.entry_max_n_for_umap.setValue(1000)
        self.btn_generate_umap_visualization = QPushButton("Generate UMAP Visualization")
        
        # Create the layout
        layout = QHBoxLayout()
        layout.addWidget(self.btn_open_model_training)
        # layout.addWidget(QLabel('Model'))
        layout.addWidget(self.label_model)
        layout.addStretch()
        layout.addWidget(QLabel('Max n for UMAP'))
        layout.addWidget(self.entry_max_n_for_umap)
        layout.addWidget(self.btn_generate_umap_visualization)
        
        self.setLayout(layout)


###########################################################################################
#####################################  Data Handaler  #####################################
###########################################################################################

class DataHandler(QObject):

    signal_populate_page0 = pyqtSignal()
    signal_set_total_page_count = pyqtSignal(int)
    signal_sorting_method = pyqtSignal(str)
    signal_annotation_stats = pyqtSignal(np.ndarray)
    signal_predictions = pyqtSignal(np.ndarray,np.ndarray)
    signal_distances = pyqtSignal(np.ndarray)

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

        self.k_similar = K_SIMILAR_DEFAULT

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
            self.run_model()
            self.signal_populate_page0.emit()

    def load_images(self,path):
        self.images = np.load(path)
        self.image_path = path
        
        if self.annotations_loaded:
            if self.images.shape[0] != self.data_pd.shape[0]:
                print('! dimension mismatch')
                return 1
        else:
            self.data_pd = pd.DataFrame({'annotation':-1},index=np.arange(self.images.shape[0]))
            self.data_pd.index.name = 'index'

        self.images_loaded = True
        print('images loaded')
        
        # run the model if the model has been loaded
        if self.model_loaded == True:
            self.run_model()
        else:
            self.data_pd['output'] = -1 # placeholder value
            if self.spot_idx_sorted is None:
                self.spot_idx_sorted = np.arange(self.images.shape[0]) # only populate the idx if it has not been populated

        # display the images
        self.signal_set_total_page_count.emit(int(np.ceil(self.get_number_of_rows()/self.n_images_per_page)))
        self.signal_populate_page0.emit()
        
        return 0

    def run_model(self):
        predictions, features = utils.generate_predictions_and_features(self.model,self.images,batch_size_inference)
        output_pd = pd.DataFrame({'output':predictions[:,0]},index=np.arange(self.images.shape[0]))
        if 'output' in self.data_pd:
            self.data_pd = self.data_pd.drop(columns=['output'])
        self.data_pd = self.data_pd.merge(output_pd,left_index=True,right_index=True)
        
        # sort the predictions
        self.data_pd = self.data_pd.sort_values('output',ascending=False)
        self.spot_idx_sorted = self.data_pd.index.to_numpy().astype(int)
        
        self.signal_set_total_page_count.emit(int(np.ceil(self.get_number_of_rows()/self.n_images_per_page)))
        self.signal_populate_page0.emit()
        self.signal_sorting_method.emit('Sort by prediction score')

        # embeddings
        self.embeddings = features
        self.embeddings_loaded = True
        self.neigh = KNeighborsClassifier(metric=KNN_METRIC)
        self.neigh.fit(self.embeddings, np.zeros(self.embeddings.shape[0]))

        # emit the results for display
        self.signal_predictions.emit(self.data_pd['output'].to_numpy(),self.data_pd['annotation'].to_numpy())

    def load_annotations(self,path):
        # load the annotation
        if self.data_pd is not None:
            annotation_pd = pd.read_csv(path,index_col='index')
            self.data_pd = self.data_pd.drop(columns=['annotation'])
            self.data_pd = self.data_pd.merge(annotation_pd,left_index=True, right_index=True, how='outer')
            self.signal_populate_page0.emit() # update the display
        else:
            self.data_pd = pd.read_csv(path,index_col='index')

        # size match check
        if self.images_loaded:
            if self.images.shape[0] != self.data_pd.shape[0]:
                print('! dimension mismatch')
                return 1
        self.annotations_loaded = True

        # update stats
        self.update_annotation_stats()

        # sort the annotations
        self.data_pd = self.data_pd.sort_values('annotation',ascending=False)
        self.spot_idx_sorted = self.data_pd.index.to_numpy().astype(int)
        self.signal_sorting_method.emit('Sort by labels')
        
        # update the display if images have been loaded already
        if self.images_loaded:
            self.signal_populate_page0.emit()
        
        print('annotations loaded')
        return 0
        
    def set_number_of_images_per_page(self,n):
        self.n_images_per_page = n

    def set_k_similar(self,k):
        self.k_similar = k

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
                texts.append( '[' + str(i) + ']  : ' + "{:.2f}".format(self.data_pd.iloc[i]['output']) + '\n{:.1e}'.format(self.data_pd.iloc[i]['distance'])) # .at would use the idx before sorting # + '{:.1e}'.format(self.data_pd.iloc[i]['distances'])

            else:
                texts.append( '[' + str(i) + ']  : ' + "{:.2f}".format(self.data_pd.iloc[i]['output'])) # .at would use the idx before sorting
            image_id.append(int(self.data_pd.index[i]))
            annotations.append(int(self.data_pd.iloc[i]['annotation']))
        return images, texts, image_id, annotations

    def sort(self,criterion):
        print(criterion)
        if criterion == 'Sort by prediction score':
            self.data_pd = self.data_pd.sort_values('output',ascending=False)
        elif criterion == 'Sort by labels':
            self.data_pd = self.data_pd.sort_values('annotation',ascending=False)
        elif criterion == 'Sort by similarity':
            self.data_pd = self.data_pd.sort_values('distance',ascending=True)
        
        # update the sorted spot idx
        if self.is_for_similarity_search:
            self.spot_idx_sorted = self.data_pd['idx_local'].to_numpy().astype(int)
        else:
            self.spot_idx_sorted = self.data_pd.index.to_numpy().astype(int)
        self.signal_populate_page0.emit()
        self.signal_sorting_method.emit(criterion)

    def find_similar_images(self,idx):
        self.neigh.set_params(n_neighbors=min(self.k_similar+1,len(self.embeddings)))
        distances, indices = self.neigh.kneighbors(self.embeddings[idx,].reshape(1, -1), return_distance=True)
        distances = distances.squeeze()
        self.signal_distances.emit(distances)
        indices = indices.squeeze()
        images = self.images[indices,]
        scores = self.data_pd.loc[indices]['output'].to_numpy() # use the presorting idx
        annotations = self.data_pd.loc[indices]['annotation'].to_numpy() # use the presorting idx
        # return images[1:,], indices[1:], scores[1:], distances[1:], annotations[1:]
        return images, indices, scores, distances, annotations

    def populate_similarity_search(self,images,indices,scores,distances,annotations):
        self.images = images
        self.data_pd = pd.DataFrame({'idx_global':indices,'idx_local':np.arange(self.images.shape[0]).astype(int),'output':scores, 'distance':distances, 'annotation':annotations},index=indices) # idx_local for indexing the spot images
        self.data_pd = self.data_pd.set_index('idx_global')
        print('populated data_pd of the similarity search data handler:')
        print(self.data_pd)
        self.images_loaded = True
        # self.spot_idx_sorted = self.data_pd['index'].to_numpy().astype(int)
        self.spot_idx_sorted = np.arange(self.images.shape[0]).astype(int)
        self.signal_set_total_page_count.emit(int(np.ceil(self.get_number_of_rows()/self.n_images_per_page)))
        self.signal_populate_page0.emit()

    def update_annotation(self,index,annotation):
        # condition = df['index'].isin(index), df.loc[condition, 'annotation'] = annotation 
        # to-do: support dealing with multiple datasets using multi-level indexing
        self.data_pd.loc[index,'annotation'] = annotation
        if self.is_for_similarity_search:
            print(self.data_pd)
        self.update_annotation_stats() # note - can also do it increamentally instead of going through the full df everytime

    def update_annotation_stats(self):
        # get annotation stats
        counts = []
        for label in ANNOTATIONS_REVERSE_DICT.keys():
            counts.append(sum(self.data_pd['annotation']==label))
        self.signal_annotation_stats.emit(np.array(counts))

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
#####################################  Matplotlib   #######################################
###########################################################################################
class PiePlotWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        #  create widgets
        self.view = FigureCanvas(Figure(figsize=(5, 3)))
        self.axes = self.view.figure.subplots()

        self.labels = list(ANNOTATIONS_REVERSE_DICT.values())
        self.counts = np.zeros(len(self.labels)) # self.counts = np.zeros(len(self.labels))
        self.counts[0] = 1 # to avoid divide by zero
        self.color = list(COLOR_DICT_PLOT.values())

        #  Create layout
        vlayout = QVBoxLayout()
        vlayout.addWidget(self.view)
        self.setLayout(vlayout)

        self._update_plot()

    def _update_plot(self):
        self.axes.clear()
        self.axes.pie(self.counts, explode = 0.1*np.ones(len(self.labels)), colors = self.color, shadow=False, startangle=0) # autopct='%1.1f%%', pctdistance=1.2
        self.axes.axis('equal')
        self.axes.legend(self.labels,loc='upper center',bbox_to_anchor=(0.5, 0.075),fancybox=True,ncol=int(len(self.labels)/2))
        self.view.draw()

    def update_plot(self,counts):
        self.counts = counts
        self._update_plot()

class BarPlotWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        #  create widgets
        self.view = FigureCanvas(Figure(figsize=(5, 3)))
        self.axes = self.view.figure.subplots()

        self.labels = list(ANNOTATIONS_REVERSE_DICT.values())
        self.counts = np.zeros(len(self.labels)) # self.counts = np.zeros(len(self.labels))
        self.color = list(COLOR_DICT_PLOT.values())   

        #  Create layout
        vlayout = QVBoxLayout()
        vlayout.addWidget(self.view)
        self.setLayout(vlayout)

        self._update_plot()

    def _update_plot(self):
        self.axes.clear()
        self.barh = self.axes.barh(self.labels, self.counts, label=self.labels, tick_label=['']*4, color=self.color)
        if max(self.counts==0):
            self.axes.set_xlim(0, 1)
        self.axes.get_xaxis().set_ticks([])
        self.axes.tick_params(left = False, bottom=False)
        # y_pos = np.arange(len(self.labels))
        # self.axes.set_yticks(y_pos, labels=self.labels)
        # self.axes.invert_yaxis()  # labels read top-to-bottom
        self.axes.bar_label(self.barh, fmt='%d')
        self.axes.spines['top'].set_visible(False)
        self.axes.spines['right'].set_visible(False)
        legend = self.axes.legend(self.labels,loc='upper center',bbox_to_anchor=(0.5, 0.075),fancybox=True,ncol=int(len(self.labels)/2))
        bbox = legend.get_window_extent()
        self.view.figure.set_size_inches(1.2*bbox.width/self.view.figure.dpi, self.view.figure.get_size_inches()[1]) 
        self.view.setMinimumSize(self.view.sizeHint())
        # self.view.setMinimumSize(self.view.size())       
        self.view.draw()

    def update_plot(self,counts):
        self.counts = counts
        self._update_plot()

class HistogramPlotWidget(QWidget):

    signal_bringToFront = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        #  create widgets
        self.view = FigureCanvas(Figure(figsize=(5, 3)))
        self.axes = self.view.figure.subplots()

        self.score = None
        self.labels = list(ANNOTATIONS_REVERSE_DICT.values())

        #  Create layout
        vlayout = QVBoxLayout()
        vlayout.addWidget(self.view)
        self.setLayout(vlayout)

        self._update_plot()

    def _update_plot(self):
        if self.score is not None:
            self.axes.clear()
            for label in COLOR_DICT_PLOT.keys():
                x = self.score[self.annotation==label]
                self.axes.hist(x,bins=50,range=(0,1),color=COLOR_DICT_PLOT[label])
            self.view.draw()
            self.signal_bringToFront.emit()

    def update_plot(self,score,annotation):
        self.score = score
        self.annotation = annotation
        self._update_plot()

class StemPlotWidget(QWidget):

    signal_bringToFront = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        #  create widgets
        self.view = FigureCanvas(Figure(figsize=(5, 3)))
        self.axes = self.view.figure.subplots()

        self.values = None

        #  Create layout
        vlayout = QVBoxLayout()
        vlayout.addWidget(self.view)
        self.setLayout(vlayout)

        self._update_plot()

    def _update_plot(self):
        if self.values is not None:
            self.axes.clear()
            self.axes.stem(self.values)
            self.view.draw()
            self.signal_bringToFront.emit()

    def update_plot(self,values):
        self.values = values
        self._update_plot()


class ScatterPlotWidget(QWidget):

    signal_bringToFront = pyqtSignal()
    signal_selected_points = pyqtSignal(np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        
        #  create widgets
        self.view = FigureCanvas(Figure(figsize=(5, 3)))
        with plt.ioff():
            self.axes = self.view.figure.subplots()

        self.x = None
        self.y = None
        self.annotation = None
        self.selector = None

        #  Create layout
        vlayout = QVBoxLayout()
        vlayout.addWidget(self.view)
        self.setLayout(vlayout)

        # self.x = np.random.rand(5000, 1)
        # self.y = np.random.rand(5000, 1)
        # self.annotation = np.array([np.random.choice(list(COLOR_DICT_PLOT.keys())) for _ in range(5000)])
        # self._update_plot()

    def _update_plot(self):
        if self.x is not None:
            # get color
            # c = [COLOR_DICT_PLOT[label] for label in self.annotation.tolist()]
            lookup = np.vectorize(COLOR_DICT_PLOT.get)
            c = lookup(self.annotation)
            self.axes.clear()
            pts = self.axes.scatter(self.x,self.y,c=c,s=5)
            zoom_factory(self.axes)
            self.selector = SelectFromCollection(self.axes,pts,alpha_other=0.1)
            self.selector.set_callback(self.on_select)
            self.view.draw()
            self.signal_bringToFront.emit()

    def update_plot(self,x,y,annotations):
        self.x = x
        self.y = y
        self.annotation = annotation
        self._update_plot()

    def on_select(self,selector):
        selected_points = selector.get_selection()
        print(selected_points)
        self.signal_selected_points.emit(selected_points)

###########################################################################################
#####################################  Main Window  #######################################
###########################################################################################

class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # core
        self.dataHandler = DataHandler()
        self.dataHandler.set_number_of_images_per_page(NUM_ROWS*num_cols)

        self.dataHandler_similarity = DataHandler(is_for_similarity_search=True)
        self.dataHandler_similarity.set_number_of_images_per_page(NUM_ROWS*num_cols)

        # widgets
        self.dataLoaderWidget = DataLoaderWidget(self.dataHandler)
        self.gallery = GalleryViewWidget(NUM_ROWS,num_cols,self.dataHandler,is_main_gallery=True)
        self.gallery_similarity = GalleryViewWidget(NUM_ROWS,num_cols,self.dataHandler_similarity,dataHandler2=self.dataHandler)
        self.gallerySettings = GalleryViewSettingsWidget()
        self.trainingAndVisualizationWidget = TrainingAndVisualizationWidget()

        self.plots = {}
        self.plots['Labels'] = PiePlotWidget()
        self.plots['Annotation Progress'] = BarPlotWidget()
        self.plots['Inference Result'] = HistogramPlotWidget()
        self.plots['Similarity'] = StemPlotWidget()
        self.plots['UMAP'] = ScatterPlotWidget()

        # tab widget
        self.gallery_tab = QTabWidget()
        self.gallery_tab.addTab(self.gallery,'Full Dataset')
        self.gallery_tab.addTab(self.gallery_similarity,'Similarity Search')

        layout = QVBoxLayout()
        layout.addWidget(self.dataLoaderWidget)
        layout.addWidget(self.gallerySettings)
        layout.addWidget(self.gallery_tab)
        layout.addWidget(self.trainingAndVisualizationWidget)

        # # plots
        centralWidget = QWidget()
        centralWidget.setLayout(layout)
        dock_annotations = dock.Dock('Interactive Annotations', autoOrientation = False)
        dock_annotations.addWidget(centralWidget)

        dock_plots = {}
        for plot in self.plots.keys():
            dock_plots[plot] = dock.Dock(plot, autoOrientation = False)
            dock_plots[plot].addWidget(self.plots[plot])
        
        main_dockArea = dock.DockArea()
        main_dockArea.addDock(dock_annotations)
        main_dockArea.addDock(dock_plots['Labels'],'right')
        main_dockArea.addDock(dock_plots['Annotation Progress'],'below',relativeTo=dock_plots['Labels'])
        dock_plots['Labels'].raiseDock()
        main_dockArea.addDock(dock_plots['Inference Result'],'bottom',relativeTo=dock_plots['Labels'])
        main_dockArea.addDock(dock_plots['Similarity'],'below',relativeTo=dock_plots['Inference Result'])
        main_dockArea.addDock(dock_plots['UMAP'],'below',relativeTo=dock_plots['Similarity'])
        dock_plots['Inference Result'].raiseDock() # bring some to the front
        
        self.setCentralWidget(main_dockArea)

        # self.centralWidget = QWidget()
        # self.centralWidget.setLayout(layout)
        # # self.centralWidget.setFixedWidth(self.centralWidget.minimumSizeHint().width())
        # self.setCentralWidget(self.centralWidget)

        # connect
        self.dataHandler.signal_set_total_page_count.connect(self.gallery.set_total_pages)
        self.dataHandler.signal_populate_page0.connect(self.gallery.populate_page0)
        
        self.dataHandler_similarity.signal_set_total_page_count.connect(self.gallery_similarity.set_total_pages)
        self.dataHandler_similarity.signal_populate_page0.connect(self.gallery_similarity.populate_page0)

        self.gallery.signal_similaritySearch.connect(self.dataHandler_similarity.populate_similarity_search)
        self.gallery.signal_switchTab.connect(self.switch_tab)

        self.gallery_similarity.signal_similaritySearch.connect(self.dataHandler_similarity.populate_similarity_search)
        self.gallery_similarity.signal_updatePage.connect(self.gallery.update_page)

        self.gallerySettings.signal_numRowsPerPage.connect(self.gallery.set_number_of_rows)
        self.gallerySettings.signal_numImagesPerPage.connect(self.dataHandler.set_number_of_images_per_page)
        self.gallerySettings.signal_k_similaritySearch.connect(self.dataHandler.set_k_similar)
        self.gallerySettings.signal_numImagesPerPage.connect(self.dataHandler_similarity.set_number_of_images_per_page)
        self.gallerySettings.signal_numRowsPerPage.connect(self.gallery_similarity.set_number_of_rows)
        self.gallerySettings.signal_k_similaritySearch.connect(self.dataHandler_similarity.set_k_similar)

        self.dataHandler.signal_annotation_stats.connect(self.plots['Labels'].update_plot)
        self.dataHandler.signal_annotation_stats.connect(self.plots['Annotation Progress'].update_plot)
        self.dataHandler.signal_predictions.connect(self.plots['Inference Result'].update_plot)
        self.dataHandler.signal_distances.connect(self.plots['Similarity'].update_plot)

        self.plots['Similarity'].signal_bringToFront.connect(dock_plots['Similarity'].raiseDock)
        self.plots['Inference Result'].signal_bringToFront.connect(dock_plots['Inference Result'].raiseDock)

        # dev mode
        if DEV_MODE:
            self.dataHandler.load_model('model.pt')
            self.dataHandler.load_images('test.npy')
            self.dataHandler.load_annotations('test.csv')

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