from __future__ import print_function
import sys
import numpy as np
import cv2
from skimage import measure

try:
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
    from PyQt5.QtWidgets import *
    PYQT5 = True
except ImportError:
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *
    PYQT5 = False

from labelme.utils.shape import Shape
import matplotlib.pyplot as plt


class Canvas2(QWidget):
    CompletePair = pyqtSignal(bool)

    CREATE, DELETE, VIEW = 0, 1, 2

    def __init__(self, window=None):
        super(Canvas2, self).__init__()
        # Initialise local state
        self.mode = self.CREATE

        # The result for save
        self.pairs = {}
        self.pair = {}

        self.np_img1 = None
        self.np_img2 = None

        self.currentIDLeft = []
        self.currentIDRight = []

        self.currentBoxLeft = []
        self.currentBoxRight = []

        self.currentCoordLeft = []
        self.currentCoordRight = []

        self.confirm_create = False
        self.confirm_delete = False

        # 1 for left, 2 for right
        self.selectedComponent1 = None
        self.selectedComponent2 = None
        self._selectedComponent1 = False
        self._selectedComponent2 = False

        self.lineColor = QColor(0, 0, 255)
        self.line = Shape(line_color=self.lineColor)
        self.prevPoint = QPointF()
        self.offsets = QPointF(), QPointF()

        self.pixmapLeft = QPixmap()
        self.pixmapRight = QPixmap()

        self.bbox1 = None
        self.bbox2 = None

        self.coord1 = None
        self.coord2 = None

        self.visible = {}
        self._hideBackround = False
        self.hideBackround = False
        self.hShape1 = None
        self.hShape2 = None
        self.hVertex = None
        # self._cursor = CURSOR_DEFAULT
        self.paint_info = False

        # Menus:
        # self.menusLeft = (QMenu(), QMenu())
        # self.menusRight = (QMenu(), QMenu())
        self.menus = (QMenu(), QMenu())

        # Set widget options
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.WheelFocus)

        # self.window is the object inherited from MainWindow
        self.window = window

        # Scroll area left
        self.imageLabelLeft = QLabel()
        self.imageLabelLeft.setBackgroundRole(QPalette.Base)
        self.imageLabelLeft.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.imageLabelLeft.setScaledContents(True)

        self.scrollAreaLeft = QScrollArea()
        self.scrollAreaLeft.setBackgroundRole(QPalette.Dark)
        self.scrollAreaLeft.setWidget(self.imageLabelLeft)
        self.scrollAreaLeft.setVisible(False)

        # Scroll area right
        self.imageLabelRight = QLabel()
        self.imageLabelRight.setBackgroundRole(QPalette.Base)
        self.imageLabelRight.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.imageLabelRight.setScaledContents(True)

        self.scrollAreaRight = QScrollArea()
        self.scrollAreaRight.setBackgroundRole(QPalette.Dark)
        self.scrollAreaRight.setWidget(self.imageLabelRight)
        self.scrollAreaRight.setVisible(False)

        # Layout of the canvas
        self.centralWidget = QWidget()
        # Horizontal Layout
        self.layout = QHBoxLayout(self.centralWidget)
        self.layout.addWidget(self.scrollAreaLeft)
        self.layout.addWidget(self.scrollAreaRight)

        # Scroll
        self.scrollAreaLeft.verticalScrollBar().valueChanged.connect(
            self.scrollAreaRight.verticalScrollBar().setValue)
        self.scrollAreaLeft.horizontalScrollBar().valueChanged.connect(
            self.scrollAreaRight.horizontalScrollBar().setValue)

        self.scrollAreaRight.verticalScrollBar().valueChanged.connect(
            self.scrollAreaLeft.verticalScrollBar().setValue)
        self.scrollAreaRight.horizontalScrollBar().valueChanged.connect(
            self.scrollAreaLeft.horizontalScrollBar().setValue)

        # Mouse (Cursor)
        # self.scrollAreaLeft.setMouseTracking(True)
        self.scrollAreaLeft.mouseMoveEvent = self.mouseMoveEventLeft
        self.scrollAreaLeft.mousePressEvent = self.mousePressEventLeft
        self.scrollAreaLeft.mouseReleaseEvent = self.mouseReleaseEventLeft
        self.pressedLeft = False

        # self.scrollAreaRight.setMouseTracking(True)
        self.scrollAreaRight.mouseMoveEvent = self.mouseMoveEventRight
        self.scrollAreaRight.mousePressEvent = self.mousePressEventRight
        self.scrollAreaRight.mouseReleaseEvent = self.mouseReleaseEventRight
        self.pressedRight = False

        # Painter
        self.imageLabelLeft.paintEvent = self.paintEventLeft
        self.imageLabelRight.paintEvent = self.paintEventRight

        # Cursor style
        self.imageLabelLeft.setCursor(Qt.ArrowCursor)
        self.imageLabelRight.setCursor(Qt.ArrowCursor)


    def checkMouseinComponentLeft(self, pos, offset=1):
        # Coordinate of the canvas
        y, x = pos

        # Convert to coordinate of image
        y, x = int(y / self.window.scaleFactor), int(x / self.window.scaleFactor)

        # Check ID of component of cursor
        cursor_block = self.window.mask1[x - offset:x + offset, y - offset:y + offset]
        is_empty = cursor_block.size == 0
        if is_empty:
            return None

        values, counts = np.unique(cursor_block, return_counts=True)
        com_id = values[np.argmax(counts)]

        if com_id != -1:
            return int(com_id)
        else:
            return None


    def mouseMoveEventLeft(self, event):
        print(10 * '-' + ' Move Cursor Left ' + 10 * '-')
        if self.pressedLeft:
            self.scrollAreaLeft.horizontalScrollBar().setValue(self.initialPosX - event.pos().x())
            self.scrollAreaLeft.verticalScrollBar().setValue(self.initialPosY - event.pos().y())


    def mousePressEventLeft(self, event):
        print(10 * '-' + ' Press Cursor Left ' + 10 * '-')
        self.pressedLeft = True
        self.currentIDRight = []
        self.currentBoxRight = []
        self.currentCoordRight = []
        self.confirm_create = False

        # Style of mouse
        self.imageLabelLeft.setCursor(Qt.ArrowCursor)

        # Coordinate of cursor after press
        self.initialPosX = self.scrollAreaLeft.horizontalScrollBar().value() + event.pos().x()
        self.initialPosY = self.scrollAreaLeft.verticalScrollBar().value() + event.pos().y()
        self.selectedComponent1 = self.checkMouseinComponentLeft((self.initialPosX, self.initialPosY))

        if self.selectedComponent1 is None:
            return

        print('Selected component in left is ', self.selectedComponent1)
        self.bbox1 = self.window.component1[self.selectedComponent1]['bbox']
        self.coord1 = self.window.component1[self.selectedComponent1]['coords']

        if self.mode == self.CREATE:
            self.pair['left'] = self.selectedComponent1

            # Get all labeled pairs with self.selectedComponent1
            list_labeled_com1 = map(lambda x: x['left'], self.pairs.values())
            if self.selectedComponent1 in list_labeled_com1:
                for dict_pair in self.pairs.values():
                    if self.selectedComponent1 == dict_pair['left']:
                        com_id_Right = dict_pair['right']
                        self.currentIDRight.append(com_id_Right)
                        self.currentBoxRight.append(self.window.component2[com_id_Right]['bbox'])
                        self.currentCoordRight.append(self.window.component2[com_id_Right]['coords'])

            # Show all component in Right map with
            self.imageLabelLeft.repaint()
            self.imageLabelRight.repaint()
            self._selectedComponent1 = True

            if self._selectedComponent1 and self._selectedComponent2:
                self.window.actions.confirm_create.setEnabled(True)

        elif self.mode == self.DELETE:
            # Get all labeled pairs with self.selectedComponent1
            list_labeled_com1 = map(lambda x: x['left'], self.pairs.values())
            if self.selectedComponent1 in list_labeled_com1:
                for dict_pair in self.pairs.values():
                    if self.selectedComponent1 == dict_pair['left']:
                        com_id_Right = dict_pair['right']
                        self.currentIDRight.append(com_id_Right)
                        self.currentBoxRight.append(self.window.component2[com_id_Right]['bbox'])
                        self.currentCoordRight.append(self.window.component2[com_id_Right]['coords'])

            # Show all component in Right map with
            self.imageLabelLeft.repaint()
            self.imageLabelRight.repaint()
            self._selectedComponent1 = True

            if self._selectedComponent1 and self._selectedComponent2:
                self.window.actions.confirm_delete.setEnabled(True)


    def mouseReleaseEventLeft(self, event):
        print(10 * '-' + ' Release Cursor Left ' + 10 * '-')
        self.pressedLeft = False
        # Style of mouse
        self.imageLabelLeft.setCursor(Qt.ArrowCursor)
        # Coordinate of cursor after release
        self.initialPosX = self.scrollAreaLeft.horizontalScrollBar().value()
        self.initialPosY = self.scrollAreaLeft.verticalScrollBar().value()


    def paintEventLeft(self, event):
        painter = QPainter()
        painter.begin(self.imageLabelLeft)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.HighQualityAntialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.drawPixmap(0, 0, self.pixmapLeft)

        if self.mode != self.VIEW:
            if self.pressedLeft or self.pressedRight:
                np_image1 = self.window.imageData1.copy()

                if len(self.currentIDLeft) != 0:
                    for coord1 in self.currentCoordLeft:
                        np_image1[coord1[:, 0], coord1[:, 1]] = [200, 200, 0]

                if self.selectedComponent1 is not None:
                    # Draw the chosen component
                    if self.confirm_create:
                        np_image1[self.coord1[:, 0], self.coord1[:, 1]] = [0, 255, 255]
                    elif self.selectedComponent1 in self.currentIDLeft:
                        np_image1[self.coord1[:, 0], self.coord1[:, 1]] = [0, 255, 255]
                    else:
                        np_image1[self.coord1[:, 0], self.coord1[:, 1]] = [0, 200, 0]

                    mask1 = np.zeros((np_image1.shape[0], np_image1.shape[1]), dtype=np.uint8)
                    mask1[self.coord1[:, 0], self.coord1[:, 1]] = 255

                    _, contours, hierarchy = cv2.findContours(mask1, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
                    cv2.drawContours(np_image1, contours, -1, (0, 255, 255), 5)

                    x_min, y_min, x_max, y_max = self.bbox1
                    p1 = (int(y_min), int(x_min)); p2 = (int(y_max), int(x_max))
                    cv2.rectangle(np_image1, p1, p2, (0, 255, 255), 3)

                    cv2.putText(np_image1, str(self.selectedComponent1), p1, cv2.FONT_HERSHEY_SIMPLEX, 1, (209, 80, 0, 255), 3)

                image1 = QImage(np_image1.data, np_image1.shape[1], np_image1.shape[0],
                                np_image1.strides[0], QImage.Format_RGB888)

                self.loadPixmapLeft(QPixmap.fromImage(image1))

                h = self.window.scaleFactor * self.imageLabelLeft.pixmap().height()
                w = self.window.scaleFactor * self.imageLabelLeft.pixmap().width()
                self.pixmapLeft = self.pixmapLeft.scaled(w, h)

                self.np_img1 = np_image1
            else:
                if self.np_img1 is None:
                    self.np_img1 = self.window.imageData1.copy()

                image1 = QImage(self.np_img1.data, self.np_img1.shape[1], self.np_img1.shape[0],
                                self.np_img1.strides[0], QImage.Format_RGB888)

                self.loadPixmapLeft(QPixmap.fromImage(image1))

                h = self.window.scaleFactor * self.imageLabelLeft.pixmap().height()
                w = self.window.scaleFactor * self.imageLabelLeft.pixmap().width()
                self.pixmapLeft = self.pixmapLeft.scaled(w, h)

        elif self.mode == self.VIEW and self.selectedComponent1 is not None and self.window.labelListClick:
            np_image1 = self.window.imageData1.copy()
            bbox1 = self.window.component1[self.selectedComponent1]['bbox']
            coord1 = self.window.component1[self.selectedComponent1]['coords']
            np_image1[coord1[:, 0], coord1[:, 1]] = [0, 255, 255]
            x_min, y_min, x_max, y_max = bbox1
            p1 = (int(y_min), int(x_min)); p2 = (int(y_max), int(x_max))
            cv2.rectangle(np_image1, p1, p2, (0, 255, 255), 2)
            cv2.putText(np_image1, str(self.selectedComponent1), p1, cv2.FONT_HERSHEY_SIMPLEX, 1, (209, 80, 0, 255), 3)

            image1 = QImage(np_image1.data, np_image1.shape[1], np_image1.shape[0],
                            np_image1.strides[0], QImage.Format_RGB888)

            self.loadPixmapLeft(QPixmap.fromImage(image1))

            h = self.window.scaleFactor * self.imageLabelLeft.pixmap().height()
            w = self.window.scaleFactor * self.imageLabelLeft.pixmap().width()
            self.pixmapLeft = self.pixmapLeft.scaled(w, h)

        painter.end()

    def checkMouseinComponentRight(self, pos, offset=1):
        # Coordinate of the canvas
        y, x = pos
        # Convert to coordinate of image
        y, x = int(y / self.window.scaleFactor), int(x / self.window.scaleFactor)
        # Check ID of component of cursor
        cursor_block = self.window.mask2[x - offset:x + offset, y - offset:y + offset]
        is_empty = cursor_block.size == 0
        if is_empty:
            return None

        values, counts = np.unique(cursor_block, return_counts=True)
        com_id = values[np.argmax(counts)]

        if com_id != -1:
            return int(com_id)
        else:
            return None

    def mouseMoveEventRight(self, event):
        print(10 * '-' + ' Move Cursor Right ' + 10 * '-')
        if self.pressedRight:
            self.scrollAreaRight.horizontalScrollBar().setValue(self.initialPosX - event.pos().x())
            self.scrollAreaRight.verticalScrollBar().setValue(self.initialPosY - event.pos().y())

    def mousePressEventRight(self, event):
        print(10 * '-' + ' Press Cursor Right ' + 10 * '-')
        self.pressedRight = True
        self.currentIDLeft = []
        self.currentBoxLeft = []
        self.currentCoordLeft = []
        self.confirm_create = False

        self.imageLabelRight.setCursor(Qt.ArrowCursor)
        self.initialPosX = self.scrollAreaRight.horizontalScrollBar().value() + event.pos().x()
        self.initialPosY = self.scrollAreaRight.verticalScrollBar().value() + event.pos().y()
        self.selectedComponent2 = self.checkMouseinComponentRight((self.initialPosX, self.initialPosY))

        if self.selectedComponent2 is None:
            return

        print('Selected component in right is ', self.selectedComponent2)
        self.bbox2 = self.window.component2[self.selectedComponent2]['bbox']
        self.coord2 = self.window.component2[self.selectedComponent2]['coords']

        if self.mode == self.CREATE:
            self.pair['right'] = self.selectedComponent2

            # Get all labeled pairs with self.selectedComponent2
            list_labeled_com2 = map(lambda x: x['right'], self.pairs.values())
            if self.selectedComponent2 in list_labeled_com2:
                for dict_pair in self.pairs.values():
                    if self.selectedComponent2 == dict_pair['right']:
                        com_id_Left = dict_pair['left']
                        self.currentIDLeft.append(com_id_Left)
                        self.currentBoxLeft.append(self.window.component1[com_id_Left]['bbox'])
                        self.currentCoordLeft.append(self.window.component1[com_id_Left]['coords'])

            # Show all component in Left map with
            self.imageLabelLeft.repaint()
            self.imageLabelRight.repaint()
            self._selectedComponent2 = True

            if self._selectedComponent1 and self._selectedComponent2:
                self.window.actions.confirm_create.setEnabled(True)

        if self.mode == self.DELETE:
            # Get all labeled pairs with self.selectedComponent2
            list_labeled_com2 = map(lambda x: x['right'], self.pairs.values())
            if self.selectedComponent2 in list_labeled_com2:
                for dict_pair in self.pairs.values():
                    if self.selectedComponent2 == dict_pair['right']:
                        com_id_Left = dict_pair['left']
                        self.currentIDLeft.append(com_id_Left)
                        self.currentBoxLeft.append(self.window.component1[com_id_Left]['bbox'])
                        self.currentCoordLeft.append(self.window.component1[com_id_Left]['coords'])

            # Show all component in Left map with
            self.imageLabelLeft.repaint()
            self.imageLabelRight.repaint()
            self._selectedComponent2 = True

            if self._selectedComponent1 and self._selectedComponent2:
                self.window.actions.confirm_delete.setEnabled(True)


    def mouseReleaseEventRight(self, event):
        print(10 * '-' + ' Release Cursor Right ' + 10 * '-')
        self.pressedRight = False
        self.imageLabelRight.setCursor(Qt.ArrowCursor)
        self.initialPosX = self.scrollAreaRight.horizontalScrollBar().value()
        self.initialPosY = self.scrollAreaRight.verticalScrollBar().value()


    def paintEventRight(self, event):
        painter = QPainter()
        painter.begin(self.imageLabelRight)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.HighQualityAntialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.drawPixmap(0, 0, self.pixmapRight)

        if self.mode != self.VIEW:
            if self.pressedLeft or self.pressedRight:
                np_image2 = self.window.imageData2.copy()

                if len(self.currentIDRight) != 0:
                    for coord2 in self.currentCoordRight:
                        np_image2[coord2[:, 0], coord2[:, 1]] = [0, 200, 0]

                if self.selectedComponent2 is not None:
                    # Draw the chosen component
                    if self.confirm_create:
                        np_image2[self.coord2[:, 0], self.coord2[:, 1]] = [0, 255, 255]
                    elif self.selectedComponent2 in self.currentIDRight:
                        np_image2[self.coord2[:, 0], self.coord2[:, 1]] = [0, 255, 255]
                    else:
                        np_image2[self.coord2[:, 0], self.coord2[:, 1]] = [200, 200, 0]

                    mask2 = np.zeros((np_image2.shape[0], np_image2.shape[1]), dtype=np.uint8)
                    mask2[self.coord2[:, 0], self.coord2[:, 1]] = 255

                    _, contours, hierarchy = cv2.findContours(mask2, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
                    cv2.drawContours(np_image2, contours, -1, (0, 255, 255), 5)

                    x_min, y_min, x_max, y_max = self.bbox2
                    p1 = (int(y_min), int(x_min)); p2 = (int(y_max), int(x_max))
                    cv2.rectangle(np_image2, p1, p2, (0, 255, 255), 3)

                    cv2.putText(np_image2, str(self.selectedComponent2), p1, cv2.FONT_HERSHEY_SIMPLEX, 1, (209, 80, 0, 255), 3)

                image2 = QImage(np_image2.data, np_image2.shape[1], np_image2.shape[0],
                                np_image2.strides[0], QImage.Format_RGB888)
                self.loadPixmapRight(QPixmap.fromImage(image2))

                h = self.window.scaleFactor * self.imageLabelRight.pixmap().height()
                w = self.window.scaleFactor * self.imageLabelRight.pixmap().width()
                self.pixmapRight = self.pixmapRight.scaled(w, h)

                self.np_img2 = np_image2

            else:
                if self.np_img2 is None:
                    self.np_img2 = self.window.imageData2.copy()

                image2 = QImage(self.np_img2.data, self.np_img2.shape[1], self.np_img2.shape[0],
                                self.np_img2.strides[0], QImage.Format_RGB888)

                self.loadPixmapRight(QPixmap.fromImage(image2))

                h = self.window.scaleFactor * self.imageLabelRight.pixmap().height()
                w = self.window.scaleFactor * self.imageLabelRight.pixmap().width()
                self.pixmapRight = self.pixmapRight.scaled(w, h)

        elif self.mode == self.VIEW and self.selectedComponent2 is not None and self.window.labelListClick:
            np_image2 = self.window.imageData2.copy()
            bbox2 = self.window.component2[self.selectedComponent2]['bbox']
            coord2 = self.window.component2[self.selectedComponent2]['coords']
            np_image2[coord2[:, 0], coord2[:, 1]] = [0, 255, 255]
            x_min, y_min, x_max, y_max = bbox2
            p1 = (int(y_min), int(x_min)); p2 = (int(y_max), int(x_max))
            cv2.rectangle(np_image2, p1, p2, (0, 255, 255), 2)
            cv2.putText(np_image2, str(self.selectedComponent2), p1, cv2.FONT_HERSHEY_SIMPLEX, 1, (209, 80, 0, 255), 3)

            image2 = QImage(np_image2.data, np_image2.shape[1], np_image2.shape[0],
                            np_image2.strides[0], QImage.Format_RGB888)
            self.loadPixmapRight(QPixmap.fromImage(image2))

            h = self.window.scaleFactor * self.imageLabelRight.pixmap().height()
            w = self.window.scaleFactor * self.imageLabelRight.pixmap().width()
            self.pixmapRight = self.pixmapRight.scaled(w, h)

        painter.end()


    def restoreCursor(self):
        QApplication.restoreOverrideCursor()


    def loadPixmapLeft(self, pixmapLeft):
        self.pixmapLeft = pixmapLeft
        self.imageLabelLeft.setPixmap(pixmapLeft)
        self.shapes = []


    def loadPixmapRight(self, pixmapRight):
        self.pixmapRight = pixmapRight
        self.imageLabelRight.setPixmap(pixmapRight)
        self.shapes = []


    def paintVisualMap(self):
        np_image1 = self.window.imageData1.copy()
        component1 = self.window.component1
        np_image2 = self.window.imageData2.copy()
        component2 = self.window.component2

        # Get the current result
        pairs = self.pairs
        left_id = []
        right_id = []

        for index, pair in pairs.items():
            left_id.append(pair['left'])
            right_id.append(pair['right'])

        left_id = list(set(left_id))
        right_id = list(set(right_id))

        for id in left_id:
            coords = component1[id]['coords']
            np_image1[coords[:, 0], coords[:, 1]] = [0, 255, 255]

        for id in right_id:
            coords = component2[id]['coords']
            np_image2[coords[:, 0], coords[:, 1]] = [0, 255, 255]

        image1 = QImage(np_image1.data, np_image1.shape[1], np_image1.shape[0],
                        np_image1.strides[0], QImage.Format_RGB888)

        image2 = QImage(np_image2.data, np_image2.shape[1], np_image2.shape[0],
                        np_image2.strides[0], QImage.Format_RGB888)

        self.pressedLeft = True
        self.loadPixmapLeft(QPixmap.fromImage(image1))
        self.loadPixmapRight(QPixmap.fromImage(image2))
        self.pressedLeft = False
        self.update()


    def setEditing(self, value=True):
        print('Receive signal from UI for Create mode (False) or View mode (True) :', value)
        self.pair = {}

        self.selectedComponent1 = None
        self.selectedComponent2 = None
        self._selectedComponent1 = False
        self._selectedComponent2 = False

        self.bbox1 = None
        self.bbox2 = None

        self.coord1 = None
        self.coord2 = None

        self.currentIDLeft = []
        self.currentIDRight = []

        self.currentBoxLeft = []
        self.currentBoxRight = []

        self.currentCoordLeft = []
        self.currentCoordRight = []

        # Show the mask
        if value:
            self.paintVisualMap()
        else:
            np_image1 = self.window.imageData1
            np_image2 = self.window.imageData2

            image1 = QImage(np_image1.data, np_image1.shape[1], np_image1.shape[0],
                            np_image1.strides[0], QImage.Format_RGB888)

            image2 = QImage(np_image2.data, np_image2.shape[1], np_image2.shape[0],
                            np_image2.strides[0], QImage.Format_RGB888)

            self.pressedLeft = True
            self.loadPixmapLeft(QPixmap.fromImage(image1))
            self.loadPixmapRight(QPixmap.fromImage(image2))
            self.pressedLeft = False

    def resetState(self):
        print('RESET CANVAS WINDOW')
        # Reset result
        self.pairs = {}
        self.pair = {}

        self.np_img1 = None
        self.np_img2 = None

        # Reset ID of component
        self.selectedComponent1 = None
        self.selectedComponent2 = None

        self._selectedComponent1 = False
        self._selectedComponent2 = False

        self.bbox1 = None
        self.bbox2 = None

        self.coord1 = None
        self.coord2 = None

        self.currentIDLeft = []
        self.currentIDRight = []

        self.currentBoxLeft = []
        self.currentBoxRight = []

        self.restoreCursor()
        self.update()
