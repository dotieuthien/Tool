import argparse
import os.path
from pathlib import Path
import re
import sys
import subprocess
import glob
from natsort import natsorted
import cv2
from skimage import measure
import numpy as np
from functools import partial
from collections import defaultdict
from constants import class_names
import matplotlib.pyplot as plt
from PIL import Image
import logging
logging.basicConfig(level=logging.INFO)

try:
    # Graphical user interface components
    from PyQt5.QtGui import *
    # Core non-GUI classes used by other modules
    from PyQt5.QtCore import *
    from PyQt5.QtWidgets import *
    from PyQt5.QtPrintSupport import QPrinter
    PYQT5 = True
except ImportError:
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *
    PYQT5 = False

from labelme.lib import struct, newAction, newIcon, addActions, fmtShortcut
from labelme.shape import Shape, DEFAULT_LINE_COLOR, DEFAULT_FILL_COLOR
from labelme.zoomWidget import ZoomWidget
from labelme.labelDialog import LabelDialog
from labelme.choiceDialog import ChoiceDialog
from labelme.colorDialog import ColorDialog
from labelme.labelFile import LabelFile, LabelFileError
from labelme.toolBar import ToolBar
from labelme.canvas2 import Canvas2
from labelme.matching import predict_matching, ucn_matching
from labelme.components import extract_component_from_mask, extract_component_from_image, extract_component_from_sketch


__appname__ = 'LabelComponent'


class WindowMixin(object):
    # Bad implementation
    # The attributes menuBar() and addToolBar are not in WindowMixin
    # inherit from self.__dict__ --> QMainWindow
    def menu(self, title, actions=None):
        menu = self.menuBar().addMenu(title)
        if actions:
            addActions(menu, actions)
        return menu

    def toolbar(self, title, actions=None):
        toolbar = ToolBar(title)
        toolbar.setObjectName('%sToolBar' % title)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        if actions:
            addActions(toolbar, actions)
        self.addToolBar(Qt.LeftToolBarArea, toolbar)
        return toolbar


class MainWindow(QMainWindow, WindowMixin):
    # The QMainWindow class provides a main application window.
    # This enables to create a classic application skeleton with a status bar, toolbars, and a menu bar
    FIT_WINDOW, FIT_WIDTH, MANUAL_ZOOM = 0, 1, 2

    def __init__(self, filename=None, output=None):
        super(MainWindow, self).__init__()
        self.setWindowTitle(__appname__)

        # Whether we need to save or not
        self.dirty = False
        self._noSelectionSlot = False
        self._beginner = True

        # Help
        self.screencastViewer = "firefox"
        self.screencast = "screencast.ogv"

        # Dialog
        self.labelDialog = LabelDialog(parent=self)
        self.choiceDialog = ChoiceDialog(parent=self)
        self.itemsToShapes = []

        # QListWidget is a convenience class that provides a list view
        self.imageList = QListWidget()
        self.labelList = QListWidget()

        self.labelList.itemSelectionChanged.connect(self.clickLabel)
        self.imageList.itemSelectionChanged.connect(self.clickImage)

        self.viewButton = QToolButton()
        # Style of button, icon and text, with text beside the icon
        self.viewButton.setToolButtonStyle(Qt.ToolButtonFollowStyle)
        self.viewButton.setText("All")

        # Linear layout
        listLayout = QVBoxLayout()
        imageLayout = QVBoxLayout()
        # The width of the outer border on each side of the widget
        listLayout.setContentsMargins(0, 0, 0, 0)
        imageLayout.setContentsMargins(0, 0, 0, 0)

        self.labelListContainer = QWidget()
        self.imageListContainer = QWidget()

        # Sets the layout manager for this widget to layout
        self.labelListContainer.setLayout(listLayout)
        self.imageListContainer.setLayout(imageLayout)

        listLayout.addWidget(self.viewButton)
        listLayout.addWidget(self.labelList)
        imageLayout.addWidget(self.imageList)

        # Dock to show labeled pairs
        self.label_dock = QDockWidget('Labels', self)
        self.image_dock = QDockWidget('Image Pairs', self)

        # Sets the widget for the dock widget
        self.label_dock.setWidget(self.labelListContainer)
        self.addDockWidget(Qt.RightDockWidgetArea, self.label_dock)

        self.image_dock.setWidget(self.imageListContainer)
        self.addDockWidget(Qt.RightDockWidgetArea, self.image_dock)

        # The dock widget can be closed and detached from the main window
        self.dockFeatures = QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetFloatable
        self.label_dock.setFeatures(self.label_dock.features() | self.dockFeatures)
        self.image_dock.setFeatures(self.image_dock.features() | self.dockFeatures)

        self.zoomWidget = ZoomWidget()
        self.colorDialog = ColorDialog(parent=self)

        # Canvas init
        self.canvas2 = Canvas2(window=self)
        self.setCentralWidget(self.canvas2.centralWidget)

        # Actions
        action = partial(newAction, self)

        createMode = action('Create Mode',
                            self.setCreateMode,
                            'C',
                            'objects',
                            'Mode create new matching pairs of component', enabled=False)

        viewMode = action('&View Mode',
                          self.setViewMode,
                          'V',
                          'view',
                          'View labels', enabled=False)

        sub_ViewMode = action('&View Mode',
                              self.sub_ViewMode,
                              'V',
                              'view',
                              'Mode view pairs of component', enabled=False)

        self.viewButton.setDefaultAction(sub_ViewMode)

        deleteMode = action('&Delete Mode',
                            self.setDeleteMode,
                            'D',
                            'delete',
                            'Mode remove labels', enabled=False)

        advancedMode = action('&Advanced Mode',
                              self.toggleAdvancedMode,
                              'Ctrl+Shift+A',
                              'expert',
                              'Switch to advanced mode', checkable=True)

        quit = action('&Quit',
                      self.close,
                      'Ctrl+Q',
                      icon='quit',
                      tip='Quit application')

        open_file = action('&Open',
                           self.openFile,
                           'Ctrl+O',
                           'open',
                           'Open folder of images')

        save_next = action('&Save',
                           self.saveLabels,
                           'Ctrl+S',
                           'save',
                           'Save labels',
                           enabled=False)

        next_image = action('&Next Image',
                            self.nextImage,
                            'Ctrl+D',
                            'next',
                            'Next pair image',
                            enabled=False)

        prev_image = action('&Prev Image',
                            self.prevImage,
                            'Ctrl+A',
                            'prev',
                            'Prev pair image',
                            enabled=False)

        confirm_delete = action('&Confirm Delete',
                                self.delete_pair,
                                'Space',
                                'done',
                                'Confirm to delete',
                                enabled=False)

        confirm_create = action('&Confirm Create',
                                self.create_pair,
                                'Space',
                                'done',
                                'Confirm to create',
                                enabled=False)

        help = action('&Tutorial',
                      self.tutorial,
                      'Ctrl+T',
                      'help',
                      'Show screencast of introductory tutorial')

        zoom = QWidgetAction(self)
        zoom.setDefaultWidget(self.zoomWidget)

        self.zoomWidget.setWhatsThis("Zoom in or out of the image. Also accessible with %s and %s from the canvas." %
                                     (fmtShortcut("Ctrl+[-+]"), fmtShortcut("Ctrl+Wheel")))

        self.zoomWidget.setEnabled(False)

        zoomIn = action('Zoom &In',
                        partial(self.addZoom, 0.2),
                        'Ctrl+Z',
                        'zoom-in',
                        'Increase zoom level',
                        enabled=False)

        zoomOut = action('&Zoom Out',
                         partial(self.addZoom, -0.2),
                         'Ctrl+Shift+Z',
                         'zoom-out',
                         'Decrease zoom level',
                         enabled=False)

        zoomOrg = action('&Original size',
                         self.setOrgSize,
                         'Ctrl+=',
                         'zoom',
                         'Zoom to original size',
                         enabled=False)

        fitWindow = action('&Fit Window',
                           self.setFitWindow,
                           'Ctrl+F',
                           'fit-window',
                           'Zoom follows window size',
                           checkable=True,
                           enabled=False)

        self.scaleFactor = 1

        # Group zoom controls into a list for easier toggling
        zoomActions = (self.zoomWidget, zoomIn, zoomOut, zoomOrg)
        self.zoomMode = self.MANUAL_ZOOM

        self.scalers = {self.FIT_WINDOW: self.scaleFitWindow,
                        # Set to one to scale to 100% when loading files
                        self.MANUAL_ZOOM: lambda: 1,
                        }

        labels = self.label_dock.toggleViewAction()
        labels.setText('Show/Hide Labels Panel')
        labels.setShortcut('Ctrl+Shift+L')

        # Label list context menu
        labelMenu = QMenu()
        imageMenu = QMenu()

        # How the widget shows a context menu
        self.labelList.setContextMenuPolicy(Qt.CustomContextMenu)
        self.imageList.setContextMenuPolicy(Qt.CustomContextMenu)

        # The widget emits the signal from popLabelListMenu
        self.labelList.customContextMenuRequested.connect(self.popLabelListMenu)
        self.imageList.customContextMenuRequested.connect(self.popImageListMenu)

        # Store actions for further handling
        self.actions = struct(open=open_file,
                              save_next=save_next,
                              confirm_create=confirm_create,
                              confirm_delete=confirm_delete,
                              createMode=createMode,
                              viewMode=viewMode,
                              sub_ViewMode=sub_ViewMode,
                              advancedMode=advancedMode,
                              deleteMode=deleteMode,
                              next_image=next_image,
                              prev_image=prev_image,
                              zoom=zoom,
                              zoomIn=zoomIn,
                              zoomOut=zoomOut,
                              zoomOrg=zoomOrg,
                              # fitWindow=fitWindow,
                              zoomActions=zoomActions,
                              fileMenuActions=(open_file, quit),
                              beginner=(), advanced=(),
                              editMenu=(None, None, None, None, None),
                              beginnerContext=(createMode, deleteMode, None),
                              advancedContext=(createMode, viewMode, deleteMode, None, None, None),
                              onLoadActive=(save_next, createMode, viewMode))

        self.menus = struct(file=self.menu('&File'),
                            edit=self.menu('&Edit'),
                            view=self.menu('&View'),
                            help=self.menu('&Help'),
                            recentFiles=QMenu('Open &Recent'),
                            labelList=labelMenu,
                            imageList=imageMenu)

        addActions(self.menus.file, (open_file, quit))
        addActions(self.menus.view, (labels, zoomIn, zoomOut, zoomOrg))
        addActions(self.menus.help, (help,))

        self.menus.file.aboutToShow.connect(self.updateFileMenu)

        self.tools = self.toolbar('Tools')

        self.actions.beginner = (open_file, createMode, deleteMode, zoomIn, zoom, zoomOut, fitWindow)

        self.actions.advanced = (save_next, confirm_create, confirm_delete, None,
                                 createMode, deleteMode, viewMode, None,
                                 next_image, prev_image)

        self.statusBar().showMessage('%s started' % __appname__)
        self.statusBar().show()

        # Application state
        self.image = QImage()
        self.filename = filename
        self.labeling_once = output is not None
        self.output = output
        self.labelListClick = False

        # Remove
        self.recentFiles = []
        self.maxRecent = 7
        self.lineColor = None
        self.fillColor = None
        self.zoom_level = 100
        self.fit_window = False

        # Restore application settings
        self.settings = {}
        self.recentFiles = self.settings.get('recentFiles', [])
        size = self.settings.get('window/size', QSize(600, 500))
        position = self.settings.get('window/position', QPoint(0, 0))
        self.resize(size)
        self.move(position)
        self.restoreState(self.settings.get('window/state', QByteArray()))
        self.lineColor = QColor(self.settings.get('line/color', Shape.line_color))
        self.fillColor = QColor(self.settings.get('fill/color', Shape.fill_color))
        Shape.line_color = self.lineColor
        Shape.fill_color = self.fillColor

        # Toggle Advanced Mode
        if self.settings.get('advanced', QVariant()):
            self.actions.advancedMode.setChecked(True)
            self.toggleAdvancedMode()

        # Signal for zoom
        self.zoomWidget.valueChanged.connect(self.paintCanvas)

        self.InitActions()

    # def noShapes(self):
    #     return not self.itemsToShapes

    def resetState(self):
        logging.info('Reset Main')
        self.itemsToShapes = []
        # Clear label list in dock
        self.labelList.clear()

        self.filename = None
        self.imageData = None
        self.labelFile = None
        self.canvas2.resetState()

    def toggleAdvancedMode(self, value=True):
        pass

    def InitActions(self):
        logging.info('Init Action')
        tool, menu = self.actions.advanced, self.actions.advancedContext

        self.tools.clear()
        addActions(self.tools, tool)

        self.canvas2.menus[0].clear()
        addActions(self.canvas2.menus[0], menu)

        self.menus.edit.clear()
        actions = (self.actions.createMode, self.actions.viewMode, self.actions.deleteMode)
        addActions(self.menus.edit, actions + self.actions.editMenu)

    def setBeginner(self):
        logging.info('Set Beginner')
        self.tools.clear()
        addActions(self.tools, self.actions.beginner)

    def setAdvanced(self):
        logging.info('Set Advanced')
        self.tools.clear()
        addActions(self.tools, self.actions.advanced)

    # def setDirty(self):
    #     self.dirty = True
    #     self.actions.save.setEnabled(True)
    #
    def setClean(self):
        self.dirty = False
        # self.actions.save.setEnabled(False)
        self.actions.create.setEnabled(True)

    def toggleActions(self, value=True):
        """
        Enable/Disable widgets which depend on an opened image
        """
        for z in self.actions.zoomActions:
            z.setEnabled(value)

        for action in self.actions.onLoadActive:
            action.setEnabled(value)

    # def queueEvent(self, function):
    #     QTimer.singleShot(0, function)

    def status(self, message, delay=5000):
        self.statusBar().showMessage(message, delay)

    def nextFileWithoutSave(self):
        logging.info('Next pair')
        self.imgCnt += 1

        if self.imgCnt >= len(self.list_pair):
            self.imgCnt -= 1
            return

        self.canvas2.imgCnt = self.imgCnt
        self.imageList.setCurrentRow(self.imgCnt)
        self.loadFile(self.list_pair[self.imgCnt])
        self.setCreateMode()
        self.canvas2.update()

    def currentItem(self):
        """
        Return list current label
        """
        # List of all selected items in the list widget
        items = self.labelList.selectedItems()

        if items:
            return items[0]
        return None

    def addRecentFile(self, filename):
        logging.info('Add %s to Recent Files' % filename)
        if filename in self.recentFiles:
            self.recentFiles.remove(filename)

        elif len(self.recentFiles) >= self.maxRecent:
            self.recentFiles.pop()
        self.recentFiles.insert(0, filename)

    # def beginner(self):
    #     return self._beginner
    #
    # def advanced(self):
    #     return not self.beginner()

    def tutorial(self):
        subprocess.Popen([self.screencastViewer, self.screencast])

    # def createShape(self):
    #     assert self.beginner()
    #     self.canvas.setEditing(False)
    #     self.actions.create.setEnabled(False)

    # def toggleDrawingSensitive(self, drawing=True):
    #     """
    #     In the middle of drawing, toggling between modes should be disabled
    #     """
    #     self.actions.editMode.setEnabled(not drawing)
    #     if not drawing and self.beginner():
    #         # Cancel creation
    #         self.canvas.setEditing(True)
    #         self.canvas.restoreCursor()
    #         self.actions.create.setEnabled(True)

    def setCreateMode(self):
        logging.info('Create Mode')
        self.actions.createMode.setEnabled(False)
        self.actions.viewMode.setEnabled(True)
        self.actions.deleteMode.setEnabled(True)

        self.actions.save_next.setEnabled(True)

        # self.actions.next_image.setEnabled(True)
        # self.actions.prev_image.setEnabled(True)

        self.actions.zoomIn.setEnabled(True)
        self.actions.zoomOut.setEnabled(True)
        self.actions.zoomOrg.setEnabled(True)
        # self.actions.fitWindow.setEnabled(True)

        self.canvas2.mode = 0
        self.actions.confirm_create.setEnabled(False)
        self.actions.confirm_delete.setEnabled(False)
        self.actions.sub_ViewMode.setEnabled(False)
        self.canvas2.setEditing(False)
        self.addZoom(0)

    def setViewMode(self):
        logging.info('View Mode')
        self.actions.createMode.setEnabled(True)
        self.actions.viewMode.setEnabled(False)
        self.actions.deleteMode.setEnabled(True)
        self.actions.save_next.setEnabled(False)

        self.actions.zoomIn.setEnabled(True)
        self.actions.zoomOut.setEnabled(True)
        self.actions.zoomOrg.setEnabled(True)
        # self.actions.fitWindow.setEnabled(True)

        self.canvas2.mode = 2
        self.actions.confirm_create.setEnabled(False)
        self.actions.confirm_delete.setEnabled(False)
        self.actions.sub_ViewMode.setEnabled(True)
        self.canvas2.setEditing(True)
        self.addZoom(0)

    def sub_ViewMode(self):
        self.canvas2.setEditing(True)
        self.addZoom(0)

    def setDeleteMode(self):
        logging.info('Delete Mode')
        self.actions.createMode.setEnabled(True)
        self.actions.viewMode.setEnabled(True)
        self.actions.deleteMode.setEnabled(False)
        self.actions.save_next.setEnabled(True)

        self.actions.zoomIn.setEnabled(True)
        self.actions.zoomOut.setEnabled(True)
        self.actions.zoomOrg.setEnabled(True)
        # self.actions.fitWindow.setEnabled(True)

        self.canvas2.mode = 1
        self.actions.confirm_create.setEnabled(False)
        self.actions.confirm_delete.setEnabled(False)
        self.actions.sub_ViewMode.setEnabled(False)
        self.canvas2.setEditing(False)
        self.addZoom(0)

    def updateFileMenu(self):
        logging.info('Update File Menu')
        current = self.filename

        def exists(filename):
            return os.path.exists(str(filename))

        menu = self.menus.recentFiles
        # menu.clear()
        files = [f for f in self.recentFiles if f != current and exists(f)]

        for i, f in enumerate(files):
            icon = newIcon('labels')
            action = QAction(icon, '&%d %s' % (i+1, QFileInfo(f).fileName()), self)
            action.triggered.connect(partial(self.loadRecent, f))
            menu.addAction(action)

    def popLabelListMenu(self, point):
        self.menus.labelList.exec_(self.labelList.mapToGlobal(point))

    def popImageListMenu(self, point):
        self.menus.imagelist.exec_(self.imageList.mapToGlobal(point))

    def addLabel(self, value, pair):
        item = QListWidgetItem(value)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        self.itemsToShapes.append((item, pair))
        self.labelList.addItem(item)

    def addImage(self, value):
        item = QListWidgetItem(value)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        self.imageList.addItem(item)

    def remLabel(self, shape):
        for index, (item, shape_) in enumerate(self.itemsToShapes):
            if shape_ == shape:
                break
        self.itemsToShapes.pop(index)
        self.labelList.takeItem(self.labelList.row(item))
        self.labelDict['person_' + shape.label.split('_')[1]][shape.label.split('_')[0]] = False

        for item in self.canvas.shapes:
            print(item.label)

    def loadLabels(self, pairs):
        for key, pair in pairs.items():
            self.addLabel(key, pair)

    def clickImage(self):
        logging.info('Click Image List')
        if self.canvas2.mode != 2:
            item_index = self.imageList.currentRow()
            self.imgCnt = item_index
            self.canvas2.imgCnt = self.imgCnt
            self.imageList.setCurrentRow(self.imgCnt)
            logging.info('Load File from Click Image')
            self.loadFile(self.list_pair[self.imgCnt])
            self.setCreateMode()
            self.canvas2.update()

    def clickLabel(self):
        logging.info('Click Label List')
        # Delete mode
        if self.canvas2.mode == 1:
            logging.info('Click Label in mode 1 (delete mode)')
            item = self.currentItem()

            for item_, pair in self.itemsToShapes:
                if item_ == item:
                    self.canvas2.selectedComponent1 = pair['left']
                    self.canvas2.selectedComponent2 = pair['right']

                    self.canvas2.pressedLeft = True
                    self.canvas2.coord1 = self.component1[self.canvas2.selectedComponent1]['coords']
                    self.canvas2.bbox1 = self.component1[self.canvas2.selectedComponent1]['bbox']

                    self.canvas2.coord2 = self.component2[self.canvas2.selectedComponent2]['coords']
                    self.canvas2.bbox2 = self.component2[self.canvas2.selectedComponent2]['bbox']

                    self.canvas2.currentIDLeft = [pair['left']]
                    self.canvas2.currentIDRight = [pair['right']]

                    self.canvas2.imageLabelLeft.repaint()
                    self.canvas2.imageLabelRight.repaint()

                    self.canvas2.pressedLeft = False
                    self.actions.confirm_delete.setEnabled(True)

        # View mode
        if self.canvas2.mode == 2:
            item = self.currentItem()
            for item_, pair in self.itemsToShapes:
                if item_ == item:
                    self.canvas2.selectedComponent1 = pair['left']
                    self.canvas2.selectedComponent2 = pair['right']
                    self.labelListClick = True
                    self.canvas2.pressedLeft = True
                    self.canvas2.imageLabelLeft.repaint()
                    self.canvas2.imageLabelRight.repaint()
                    self.labelListClick = False
                    self.canvas2.pressedLeft = False

    def _saveLabels(self, outputFile):
        logging.info('Save Labels function')
        lf = LabelFile()
        pairs = self.canvas2.pairs
        img_name_1, img_name_2 = self.list_pair[self.imgCnt]

        try:
            lf.save(outputFile, pairs, img_name_1, img_name_2, self.imageData, self.lineColor.getRgb(), self.fillColor.getRgb())
            self.labelFile = lf
            return True

        except LabelFileError as e:
            self.errorMessage('Error saving label data', '<b>%s</b>' % e)
            return False

    # def newShape(self):
    #     """
    #     Pop-up and give focus to the label editor.
    #     position MUST be in global coordinates.
    #     """
    #     text = self.choiceDialog.popUp()
    #
    #     if text is not None and not self.labelDict['person_' + str(self.person_id)][text]:
    #         new_label = text + '_' + str(self.person_id)
    #         self.addLabel(self.canvas.setLastLabel(new_label), text)
    #         if self.beginner():
    #             self.canvas.setEditing(True)
    #             self.actions.create.setEnabled(True)
    #         else:
    #             self.actions.editMode.setEnabled(True)
    #         self.setDirty()
    #
    #     elif text is not None:
    #         self.labelDialog.popUp('Already label the "'+text+'" person '+str(self.person_id)+'!', notice=True)
    #         self.canvas.undoWrongLabel()
    #     else:
    #         self.canvas.undoWrongLabel()
    #
    #     self.updateProgress()

    def addZoom(self, factor):
        logging.info('Add Zoom with factor %f' % factor)
        if self.scaleFactor + factor <= 0:
            return

        self.scaleFactor += factor

        self.canvas2.scrollAreaLeft.setWidgetResizable(False)
        self.canvas2.scrollAreaRight.setWidgetResizable(False)

        h = self.scaleFactor * self.canvas2.imageLabelLeft.pixmap().height()
        w = self.scaleFactor * self.canvas2.imageLabelLeft.pixmap().width()

        self.canvas2.imageLabelLeft.resize(self.scaleFactor * self.canvas2.imageLabelLeft.pixmap().size())
        self.canvas2.imageLabelRight.resize(self.scaleFactor * self.canvas2.imageLabelRight.pixmap().size())

        self.canvas2.pixmapLeft = self.canvas2.pixmapLeft.scaled(w, h)
        self.canvas2.pixmapRight = self.canvas2.pixmapRight.scaled(w, h)

    def zoomRequest(self, delta):
        logging.info('Zoom Request')
        units = delta * 0.1
        self.addZoom(units)

    def setFitWindow(self):
        logging.info('Set Fit Window')
        self.canvas2.scrollAreaLeft.setWidgetResizable(True)
        self.canvas2.scrollAreaRight.setWidgetResizable(True)

    def setOrgSize(self):
        logging.info('Set Origin Size')
        self.scaleFactor = 1

        self.canvas2.scrollAreaLeft.setWidgetResizable(False)
        self.canvas2.scrollAreaRight.setWidgetResizable(False)

        # Load origin pixmap
        image1 = QImage(self.imageData1.data, self.imageData1.shape[1], self.imageData1.shape[0],
                        self.imageData1.strides[0], QImage.Format_RGB888)

        image2 = QImage(self.imageData2.data, self.imageData2.shape[1], self.imageData2.shape[0],
                        self.imageData2.strides[0], QImage.Format_RGB888)

        self.canvas2.loadPixmapLeft(QPixmap.fromImage(image1))
        self.canvas2.loadPixmapRight(QPixmap.fromImage(image2))

        # Scale canvas
        h, w = self.imageData1.shape[0], self.imageData1.shape[1]
        self.canvas2.pixmapLeft = self.canvas2.pixmapLeft.scaled(w, h)
        self.canvas2.pixmapRight = self.canvas2.pixmapRight.scaled(w, h)

        self.canvas2.imageLabelLeft.adjustSize()
        self.canvas2.imageLabelRight.adjustSize()

    def create_pair(self):
        # Save pair into pairs
        value = str(self.canvas2.selectedComponent1) + '_' + str(self.canvas2.selectedComponent2)
        self.canvas2.pairs[value] = self.canvas2.pair
        self.labelList.clear()
        self.loadLabels(self.canvas2.pairs)
        values = list(self.canvas2.pairs.keys())
        index = values.index(value)
        self.labelList.setCurrentRow(index)

        # Reset params and repaint canvas
        self.canvas2._selectedComponent1 = False
        self.canvas2._selectedComponent2 = False
        self.canvas2.pair = {}
        self.canvas2.confirm_create = True

        self.canvas2.pressedLeft = True
        self.canvas2.pressedRight = True

        self.canvas2.imageLabelLeft.repaint()
        self.canvas2.imageLabelRight.repaint()

        self.canvas2.pressedLeft = False
        self.canvas2.pressedRight = False

        self.actions.confirm_create.setEnabled(False)

    def delete_pair(self):
        # Remove pair from pairs
        try:
            self.canvas2.pairs.pop(str(self.canvas2.selectedComponent1) + '_' + str(self.canvas2.selectedComponent2))
        except:
            logging.warning('Have no label to delete in delete_pair function')
        self.labelList.clear()
        self.loadLabels(self.canvas2.pairs)
        self.canvas2._selectedComponent1 = False
        self.canvas2._selectedComponent2 = False
        self.canvas2.pair = {}

        self.canvas2.np_img1 = None
        self.canvas2.np_img2 = None

        self.actions.confirm_delete.setEnabled(False)

    def openFile(self, _value=False):
        logging.info('Open File and Load New Cut')
        dialog = QFileDialog()

        # # Option 1
        # # File Dialog for user to get the data directory
        # self.datadir = dialog.getExistingDirectory(self, 'Select an directory')
        #
        # # All cuts in data dir
        # list_cut = natsorted(glob.glob(os.path.join(self.datadir, 'color')))
        #
        # if len(list_cut) == 0:
        #     return
        #
        # list_pair = []
        # types = ['png', 'jpg', 'jpeg', 'tga', 'TGA']
        #
        # for cut in list_cut:
        #     list_img_in_cut = []
        #
        #     for type in types:
        #         list_img_in_cut.extend(glob.glob(os.path.join(cut, '*.' + type)))
        #     list_img_in_cut = natsorted(list_img_in_cut)
        #
        #     num_img = len(list_img_in_cut)
        #     num_pair = num_img - 1
        #     for i in range(num_pair):
        #         pair = [list_img_in_cut[i], list_img_in_cut[i + 1]]
        #         list_pair.append(pair)
        #
        # self.list_pair = list_pair

        # Option 2
        # User will choose 2 images instead of choosing a folder (cut)
        # Left is target sketch, right is reference
        img_path1 = dialog.getOpenFileName(self, 'Select left image - sketch image')[0]
        parts1 = Path(img_path1).parts
        cut_name_1 = parts1[-3]

        if parts1[-2] != 'color':
            self.errorMessage(
                'Error opening file',
                '<p>Make sure <i>{0}</i> is a valid image file.<br/>'
                'Left image is sketch image which in sketch folder'
                .format(parts1[-1]))

            self.status("Error reading sketch %s" % img_path1)
            return False

        img_path2 = dialog.getOpenFileName(self, 'Select right image - reference sketch image')[0]
        parts2 = Path(img_path2).parts
        cut_name_2 = parts2[-3]

        if cut_name_2 != cut_name_1:
            self.errorMessage(
                'Error opening file',
                '<p>Make sure <i>{0}</i> is a valid image file.<br/>'
                'Right image is sketch of reference which in the same cut with left image'
                .format(parts1[-1]))

            self.status("Error reading sketch %s" % img_path2)
            return False

        if parts2[-2] != 'color':
            self.errorMessage(
                'Error opening file',
                '<p>Make sure <i>{0}</i> is a valid image file.<br/>'
                'Right image is sketch of reference which in sketch folder'
                .format(parts1[-1]))

            self.status("Error reading sketch %s" % img_path2)
            return False

        self.list_pair = [[img_path1, img_path2]]

        # Add list pair to show in bottom-right dock
        self.imageList.clear()
        for img_path1, img_path2 in self.list_pair:
            name1 = os.path.basename(img_path1)
            name2 = os.path.basename(img_path2)
            value = name1 + '_' + name2
            self.addImage(value)

        self.imgCnt = 0
        self.imageList.setCurrentRow(self.imgCnt)
        self.canvas2.imgCnt = self.imgCnt

        # Load pair
        # self.loadFile(self.list_pair[self.imgCnt])

        # Init actions
        self.actions.next_image.setEnabled(True)
        self.actions.prev_image.setEnabled(True)

        self.canvas2.scrollAreaLeft.setVisible(True)
        self.canvas2.scrollAreaRight.setVisible(True)

        self.setCreateMode()
        self.canvas2.update()

    def loadFile(self, filename=None):
        """
        Load the specified file, or the last opened file if None
        """
        logging.info('Load File')
        self.resetState()
        self.canvas2.setEnabled(False)

        if filename is None:
            filename = self.settings.get('filename', '')

        filename1, filename2 = filename

        if QFile.exists(filename1) and QFile.exists(filename2):
            self.imageData1 = np.array(Image.open(filename1).convert('RGB'))
            self.imageData2 = np.array(Image.open(filename2).convert('RGB'))

            image1 = QImage(self.imageData1.data, self.imageData1.shape[1], self.imageData1.shape[0],
                            self.imageData1.strides[0], QImage.Format_RGB888)

            image2 = QImage(self.imageData2.data, self.imageData2.shape[1], self.imageData2.shape[0],
                            self.imageData2.strides[0], QImage.Format_RGB888)

            if image1.isNull():
                formats = ['*.{}'.format(fmt.data().decode()) for fmt in QImageReader.supportedImageFormats()]

                self.errorMessage(
                    'Error opening file',
                    '<p>Make sure <i>{0}</i> is a valid image file.<br/>'
                    'Supported image formats: {1}</p>'
                    .format(filename1, ','.join(formats)))

                self.status("Error reading %s" % filename1)
                return False

            if image2.isNull():
                formats = ['*.{}'.format(fmt.data().decode()) for fmt in QImageReader.supportedImageFormats()]

                self.errorMessage(
                    'Error opening file',
                    '<p>Make sure <i>{0}</i> is a valid image file.<br/>'
                    'Supported image formats: {1}</p>'
                    .format(filename2, ','.join(formats)))

                self.status("Error reading %s" % filename2)
                return False

            self.status("Loading %s and %s ..." % (os.path.basename(str(filename1)), os.path.basename(str(filename2))))

            np_image1 = self.imageData1
            np_image2 = self.imageData2

            cut_dir = ''
            for p in Path(filename1).parts[0:-2]:
                cut_dir = os.path.join(cut_dir, p)

            img_name1 = os.path.basename(filename1)[:-4]
            img_name2 = os.path.basename(filename2)[:-4]

            save_path1 = os.path.join(cut_dir, 'annotations', 'sketch_%s.png' % img_name1)
            save_path2 = os.path.join(cut_dir, 'annotations', 'sketch_%s.png' % img_name2)

            if not os.path.exists(os.path.split(save_path1)[0]):
                os.makedirs(os.path.split(save_path1)[0])

            # Load components and mask
            if not os.path.exists(save_path1):
                print('Extracting components 1 and mask 1 ...')
                # self.component1, self.mask1 = extract_component_from_image(np_image1)
                # self.component1, self.mask1, label_mask1 = extract_component_from_sketch(np_image1)
                self.component1, self.mask1, label_mask1 = extract_component_from_image(np_image1)
                cv2.imwrite(save_path1, label_mask1)
            else:
                print('Loading components 1 and mask 1 ...')
                label_mask1 = cv2.imread(save_path1, cv2.IMREAD_GRAYSCALE)
                self.component1, self.mask1 = extract_component_from_mask(label_mask1)

            if not os.path.exists(save_path2):
                print('Extracting components 2 and mask 2 ...')
                # self.component2, self.mask2, label_mask2 = extract_component_from_sketch(np_image2)
                self.component2, self.mask2, label_mask2 = extract_component_from_image(np_image2)
                cv2.imwrite(save_path2, label_mask2)
            else:
                print('Loading components 2 and mask 2 ...')
                label_mask2 = cv2.imread(save_path2, cv2.IMREAD_GRAYSCALE)
                self.component2, self.mask2 = extract_component_from_mask(label_mask2)

            # Load labeled pairs of components
            full_output_dir = os.path.join(cut_dir, 'annotations', "%s.json" % ('sketch_' + img_name1 + '__' + 'sketch_' + img_name2))
            if os.path.exists(full_output_dir):
                print('Loading available json ...')
                lf = LabelFile()
                try:
                    self.canvas2.pairs = lf.load(full_output_dir)
                except LabelFileError as e:
                    self.errorMessage('Error saving label data', '<b>%s</b>' % e)
            else:
                # print('Estimate pairs ...')
                # self.canvas2.pairs = ucn_matching(np_image1,
                #                                   np_image2,
                #                                   label_mask1,
                #                                   self.component1,
                #                                   label_mask2,
                #                                   self.component2)
                self.canvas2.pairs = {}

            self.image1 = image1
            self.filename1 = filename1

            # Load two images
            self.canvas2.loadPixmapLeft(QPixmap.fromImage(image1))
            self.canvas2.loadPixmapRight(QPixmap.fromImage(image2))
            self.scaleFactor = 1.0

            if self.labelFile:
                self.loadLabels(self.labelFile.pairs)
            else:
                self.loadLabels(self.canvas2.pairs)

            # self.setClean()
            self.canvas2.setEnabled(True)
            self.setOrgSize()
            # self.paintCanvas()
            self.addRecentFile(self.filename1)
            # self.toggleActions(True)

            return True

        return False

    # def resizeEvent(self, event):
    #     if self.canvas2 and not self.image.isNull() and self.zoomMode != self.MANUAL_ZOOM:
    #         self.adjustScale()
    #     super(MainWindow, self).resizeEvent(event)

    def paintCanvas(self):
        assert not self.image1.isNull(), "Cannot paint null image"
        self.canvas2.scale = 0.01 * self.zoomWidget.value()
        self.canvas2.adjustSize()
        self.canvas2.adjustSize()
        self.canvas2.update()

    def adjustScale(self, initial=False):
        logging.info('Adjust Scale')
        value = self.scalers[self.FIT_WINDOW if initial else self.zoomMode]()
        self.zoomWidget.setValue(int(100 * value))

    def scaleFitWindow(self):
        # Scale to fit the window left and right
        self.canvas2.scrollAreaLeft.setWidgetResizable(True)
        self.canvas2.scrollAreaRight.setWidgetResizable(True)

        # So that no scrollbars are generated
        e = 2.0
        w1 = (self.centralWidget().width() - e) / 2
        h1 = (self.centralWidget().height() - e) / 2
        a1 = w1 / h1

        # Calculate a new scale value based on the pixmap's aspect ratio.
        w2 = self.canvas2.pixmapLeft.width() - 0.0
        h2 = self.canvas2.pixmapLeft.height() - 0.0
        a2 = w2 / h2
        return w1 / w2 if a2 >= a1 else h1 / h2

    def loadRecent(self, filename):
        logging.info('Load Recent File')
        if self.mayContinue():
            self.loadFile(filename)

    def saveLabels(self):
        logging.info('Save Labels')
        img_name_1, img_name_2 = self.list_pair[self.imgCnt]
        cut_dir = ''
        for p in Path(img_name_1).parts[0:-2]:
            cut_dir = os.path.join(cut_dir, p)

        img_name_1, img_name_2 = os.path.basename(img_name_1).split('.')[0], os.path.basename(img_name_2).split('.')[0]
        full_output_dir = os.path.join(cut_dir, 'annotations')

        if not os.path.exists(full_output_dir):
            os.makedirs(full_output_dir)
        out_fn = os.path.join(full_output_dir, "%s.json" % ('sketch_' + img_name_1 + '__' + 'sketch_' + img_name_2))

        # Save the result from the previous pair image
        self._saveLabels(out_fn)

    def nextImage(self):
        print(10 * '-' + ' Next pair of image ' + 10 * '-')
        self.imgCnt += 1

        if self.imgCnt >= len(self.list_pair):
            self.imgCnt -= 1
            return

        self.canvas2.imgCnt = self.imgCnt
        self.imageList.setCurrentRow(self.imgCnt)
        self.loadFile(self.list_pair[self.imgCnt])
        self.setCreateMode()
        self.canvas2.update()

    def prevImage(self):
        print(10 * '-' + ' Previous pair of image ' + 10 * '-')
        self.imgCnt -= 1

        if self.imgCnt < 0:
            self.imgCnt += 1
            return

        self.canvas2.imgCnt = self.imgCnt
        self.imageList.setCurrentRow(self.imgCnt)
        self.loadFile(self.list_pair[self.imgCnt])
        self.setCreateMode()
        self.canvas2.update()

    def saveFile(self, _value=False):
        print(10 * '-' + ' Save file ' + 10 * '-')
        assert not self.image.isNull(), "cannot save empty image"
        assert not self.labelFile and not self.output, 'not label'
        
        if self.hasLabels():
            imgid = self.list_pair[self.imgCnt].split('.')[0]
            self._saveFile(os.path.join(self.dirname, 'annotations', imgid+'.json'))
        with open(os.path.join(self.dirname, 'annotations', 'current.txt'), 'w') as cnt_file:
            cnt_file.write(str(self.imgCnt))

    def saveFileAs(self, _value=False):
        print(10 * '-' + ' Save file as ' + 10 * '-')
        assert not self.image.isNull(), "cannot save empty image"
        if self.hasLabels():
            self._saveFile(self.saveFileDialog())

    def saveFileDialog(self):
        caption = '%s - Choose File' % __appname__
        filters = 'Label files (*%s)' % LabelFile.suffix
        dlg = QFileDialog(self, caption, self.currentPath(), filters)
        dlg.setDefaultSuffix(LabelFile.suffix[1:])
        dlg.setAcceptMode(QFileDialog.AcceptSave)
        dlg.setOption(QFileDialog.DontConfirmOverwrite, False)
        dlg.setOption(QFileDialog.DontUseNativeDialog, False)
        basename = os.path.splitext(self.filename)[0]
        default_labelfile_name = os.path.join(self.currentPath(),
                                              basename + LabelFile.suffix)
        filename = dlg.getSaveFileName(self, 'Choose File', default_labelfile_name, 'Label files (*%s)' % LabelFile.suffix)
        if PYQT5:
            filename, _ = filename
        filename = str(filename)
        return filename

    def _saveFile(self, filename):
        if filename and self.saveLabels(filename):
            self.addRecentFile(filename)
            self.setClean()
            if self.labeling_once:
                self.close()

    def closeFile(self, _value=False):
        logging.info('Close file')
        if not self.mayContinue():
            return
        self.resetState()
        self.setClean()
        self.toggleActions(False)
        self.canvas2.setEnabled(False)
        self.actions.saveAs.setEnabled(False)

    # Message Dialogs
    def hasLabels(self):
        if not self.itemsToShapes:
            self.errorMessage('No objects labeled', 'You must label at least one object to save the file.')
            return False
        return True

    def mayContinue(self):
        return not (self.dirty and not self.discardChangesDialog())

    def discardChangesDialog(self):
        yes, no = QMessageBox.Yes, QMessageBox.No
        msg = 'You have unsaved changes, proceed anyway?'
        return yes == QMessageBox.warning(self, 'Attention', msg, yes|no)

    def errorMessage(self, title, message):
        return QMessageBox.critical(self, title, '<p><b>%s</b></p>%s' % (title, message))

    def currentPath(self):
        return os.path.dirname(str(self.filename)) if self.filename else '.'

    def chooseColor1(self):
        color = self.colorDialog.getColor(self.lineColor, 'Choose line color', default=DEFAULT_LINE_COLOR)
        if color:
            self.lineColor = color
            # Change the color for all shape lines:
            Shape.line_color = self.lineColor
            self.canvas.update()
            self.setDirty()

    def chooseColor2(self):
       color = self.colorDialog.getColor(self.fillColor, 'Choose fill color', default=DEFAULT_FILL_COLOR)
       if color:
            self.fillColor = color
            Shape.fill_color = self.fillColor
            self.canvas.update()
            self.setDirty()

    def deleteSelectedShape(self):
        yes, no = QMessageBox.Yes, QMessageBox.No
        msg = 'You are about to permanently delete this polygon, proceed anyway?'
        if yes == QMessageBox.warning(self, 'Attention', msg, yes|no):
            self.remLabel(self.canvas.deleteSelected())
            self.setDirty()
            if self.noShapes():
                for action in self.actions.onShapesPresent:
                    action.setEnabled(False)
        self.updateProgress()

    def chshapeLineColor(self):
        color = self.colorDialog.getColor(self.lineColor, 'Choose line color',
                default=DEFAULT_LINE_COLOR)
        if color:
            self.canvas.selectedShape.line_color = color
            self.canvas.update()
            self.setDirty()

    def chshapeFillColor(self):
        color = self.colorDialog.getColor(self.fillColor, 'Choose fill color',
                                          default=DEFAULT_FILL_COLOR)
        if color:
            self.canvas.selectedShape.fill_color = color
            self.canvas.update()
            self.setDirty()

    def copyShape(self):
        self.canvas.endMove(copy=True)
        self.addLabel(self.canvas.selectedShape)
        self.setDirty()

    def moveShape(self):
        self.canvas.endMove(copy=False)
        self.setDirty()


class Settings(object):
    """
    Convenience dict-like wrapper around QSettings
    """
    def __init__(self, types=None):
        self.data = QSettings()
        self.types = defaultdict(lambda: QVariant, types if types else {})

    def __setitem__(self, key, value):
        t = self.types[key]
        self.data.setValue(key, t(value) if not isinstance(value, t) else value)

    def __getitem__(self, key):
        return self._cast(key, self.data.value(key))

    def get(self, key, default=None):
        return self._cast(key, self.data.value(key, default))

    def _cast(self, key, value):
        # XXX: Very nasty way of converting types to QVariant methods :P
        t = self.types[key]
        if t != QVariant:
            method = getattr(QVariant, re.sub('^Q', 'to', t.__name__, count=1))
            return method(value)
        return value


def inverted(color):
    return QColor(*[255 - v for v in color.getRgb()])


def read(filename, default=None):
    print('Reading ...')
    try:
        with open(filename, 'rb') as f:
            return f.read()
    except:
        return default


def main():
    """
    Standard boilerplate Qt application code.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', nargs='?', help='image or label filename')
    parser.add_argument('-O', '--output', help='output label name')
    args = parser.parse_args()

    filename = ['./icon/hades.png',
                './icon/hades.png']
    output = "."

    app = QApplication(sys.argv)
    app.setApplicationName(__appname__)
    app.setWindowIcon(newIcon("app"))

    win = MainWindow(filename, output)
    win.show()
    win.raise_()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
