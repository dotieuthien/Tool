from copy import deepcopy
try:
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
    from PyQt5.QtWidgets import *
    PYQT5 = True
except ImportError:
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *
    PYQT5 = False

from .lib import newIcon, labelValidator

BB = QDialogButtonBox


class LabelDialog(QDialog):
    def __init__(self, text="Enter object label", parent=None):
        super(LabelDialog, self).__init__(parent)

        self.edit = QLineEdit()
        self.edit.setText(text)
        self.edit.setValidator(labelValidator())
        self.edit.editingFinished.connect(self.postProcess)
        layout = QVBoxLayout()
        layout.addWidget(self.edit)
        self.buttonBox = bb = BB(BB.Ok | BB.Cancel, Qt.Horizontal, self)
        bb.button(BB.Ok).setIcon(newIcon('done'))
        bb.button(BB.Cancel).setIcon(newIcon('undo'))
        bb.accepted.connect(self.validate)
        bb.rejected.connect(self.reject)
        layout.addWidget(bb)
        self.setLayout(layout)

    def validate(self):
        if PYQT5:
            if self.edit.text().strip():
                self.accept()
        else:
            if self.edit.text().trimmed():
                self.accept()

    def postProcess(self):
        if PYQT5:
            self.edit.setText(self.edit.text().strip())
        else:
            self.edit.setText(self.edit.text().trimmed())

    def popUp(self, text='', move=True, notice=False):
        self.edit.setText(text)
        self.edit.setSelection(0, len(text))
        self.edit.setFocus(Qt.PopupFocusReason)

        if notice:
            self.edit.setText(text)
        if move:
            self.move(QCursor.pos())
        return self.edit.text() if self.exec_() else None

