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
from constants import class_names

BB = QDialogButtonBox


class ChoiceDialog(QDialog):

    def __init__(self, text="Choose object label", parent=None):
        super(ChoiceDialog, self).__init__(parent)
        self.edit = QLineEdit()
        self.edit.setText(text)
        self.edit.setEnabled(False)
        self.edit.setValidator(labelValidator())
        self.edit.editingFinished.connect(self.postProcess)
        layout = QVBoxLayout()
        layout.addWidget(self.edit)

        self.buttonBox = bb = BB(BB.Abort | BB.Ok | BB.SaveAll | BB.Discard | BB.Apply | BB.Retry |
                                 BB.Cancel | BB.Reset | BB.Save | BB.Yes | BB.Ignore | BB.Help | BB.RestoreDefaults |
                                 BB.No | BB.NoToAll | BB.Open, Qt.Vertical, self)

        #
        # -------------------------------------------------
        #

        bb.button(BB.Yes).setText(class_names[0])
        bb.button(BB.No).setText(class_names[1])
        bb.button(BB.NoToAll).setText(class_names[2])
        bb.button(BB.Ok).setText(class_names[3])
        bb.button(BB.Abort).setText(class_names[4])
        bb.button(BB.Cancel).setText(class_names[5])
        bb.button(BB.Save).setText(class_names[6])
        bb.button(BB.SaveAll).setText(class_names[7])
        bb.button(BB.Open).setText(class_names[8])
        bb.button(BB.Retry).setText(class_names[9])
        bb.button(BB.Ignore).setText(class_names[10])
        bb.button(BB.Discard).setText(class_names[11])
        bb.button(BB.Apply).setText(class_names[12])
        bb.button(BB.Reset).setText(class_names[13])
        bb.button(BB.RestoreDefaults).setText(class_names[14])
        bb.button(BB.Help).setText(class_names[15])

        #
        # -------------------------------------------------
        #

        bb.button(BB.Yes).clicked.connect(self.p1)
        bb.button(BB.No).clicked.connect(self.p2)
        bb.button(BB.NoToAll).clicked.connect(self.p3)
        bb.button(BB.Ok).clicked.connect(self.p4)
        bb.button(BB.Abort).clicked.connect(self.p5)
        bb.button(BB.Cancel).clicked.connect(self.p6)
        bb.button(BB.Save).clicked.connect(self.p7)
        bb.button(BB.SaveAll).clicked.connect(self.p8)
        bb.button(BB.Open).clicked.connect(self.p9)
        bb.button(BB.Retry).clicked.connect(self.p10)
        bb.button(BB.Ignore).clicked.connect(self.p11)
        bb.button(BB.Discard).clicked.connect(self.p12)
        bb.button(BB.Apply).clicked.connect(self.p13)
        bb.button(BB.Reset).clicked.connect(self.p14)
        bb.button(BB.RestoreDefaults).clicked.connect(self.p15)
        bb.button(BB.Help).clicked.connect(self.p16)

        bb.accepted.connect(self.validate)
        bb.rejected.connect(self.reject)
        layout.addWidget(bb)
        self.setLayout(layout)

    def p1(self):
        self.edit.setText(class_names[0])
        self.accept()

    def p2(self):
        self.edit.setText(class_names[1])
        self.accept()

    def p3(self):
        self.edit.setText(class_names[2])
        self.accept()

    def p4(self):
        self.edit.setText(class_names[3])
        self.accept()

    def p5(self):
        self.edit.setText(class_names[4])
        self.accept()

    def p6(self):
        self.edit.setText(class_names[5])
        self.accept()

    def p7(self):
        self.edit.setText(class_names[6])
        self.accept()

    def p8(self):
        self.edit.setText(class_names[7])
        self.accept()

    def p9(self):
        self.edit.setText(class_names[8])
        self.accept()

    def p10(self):
        self.edit.setText(class_names[9])
        self.accept()

    def p11(self):
        self.edit.setText(class_names[10])
        self.accept()

    def p12(self):
        self.edit.setText(class_names[11])
        self.accept()

    def p13(self):
        self.edit.setText(class_names[12])
        self.accept()

    def p14(self):
        self.edit.setText(class_names[13])
        self.accept()

    def p15(self):
        self.edit.setText(class_names[14])
        self.accept()

    def p16(self):
        self.edit.setText(class_names[15])
        self.accept()

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

    def popUp(self, text='', move=True):
        self.edit.setText(text)
        self.edit.setSelection(0, len(text))
        self.edit.setFocus(Qt.PopupFocusReason)
        if move:
            self.move(QCursor.pos())
        return self.edit.text() if self.exec_() else None
