import os.path
import sys
import pandas as pd
# from qdarkstyle import load_stylesheet_pyqt5
from PyQt5.QtWidgets import QApplication, QTableView
from PyQt5.QtCore import QAbstractTableModel, Qt
from PyQt5 import uic, QtWidgets
from PyQt5.QtWidgets import *
from feature_selection import feat_sel, draw_model_roc, train_model
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import scikitplot as skplt


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        # self.init_ui()
        self.ui = uic.loadUi('appUI.ui')
        # print(self.ui.__dict__)  # 查看ui文件中有哪些控件
        # 提取要操作的控件
        self.dataView = self.ui.DataView

        self.figView = self.ui.ShowView
        self.graphicscene = QtWidgets.QGraphicsScene()
        self.fig = plt.figure(figsize=(9, 6))
        self.canvas = FigureCanvas(self.fig)

        self.loadDataBtn = self.ui.loadDataBtn
        self.featSelBtn = self.ui.featSelBtn
        self.ModelTrainBtn = self.ui.ModelTrainBtn
        self.ResultsBtn = self.ui.ResultsBtn

        # 绑定信号与槽函数
        self.loadDataBtn.clicked.connect(self.loadData)
        self.featSelBtn.clicked.connect(self.show_featSel)
        self.ModelTrainBtn.clicked.connect(self.model_train)
        self.ResultsBtn.clicked.connect(self.show_results)

    def loadData(self):
        data_path, filetype = QtWidgets.QFileDialog.getOpenFileName(self, "选取文件", "./",
                                                                    "*.*")
        if os.path.exists(data_path):
            # data_path = 'Data/task5_train1_temp.csv'
            data = pd.read_csv(data_path)
            model = QtTable(data)
            # app.setStyleSheet(load_stylesheet_pyqt5())
            fnt = self.dataView.font()
            fnt.setPointSize(9)
            self.dataView.setFont(fnt)
            self.dataView.setModel(model)
            self.dataView.setWindowTitle('viewer')
            # self.dataView.resize(1080, 400)
            self.dataView.show()
        else:
            pass

    def show_featSel(self):
        data_path, filetype = QtWidgets.QFileDialog.getOpenFileName(self, "选取文件", "./",
                                                                    "*.*")
        if os.path.exists(data_path):
            # data_path = 'Data/task5_train.csv'
            plt.clf()
            self.figView.close()
            ax = self.fig.add_subplot(111)
            feat_names, scores = feat_sel(data_path)
            ax.bar(feat_names, scores, color='#275092')
            plt.xticks(rotation=45,
                       ha='right',
                       va='top')
            plt.tick_params(labelsize=9)
            plt.ylabel('Mutual Information Score')
            plt.rcParams['font.sans-serif'] = ['Times New Roman']
            plt.rcParams['axes.unicode_minus'] = False
            plt.subplots_adjust(left=0.1, bottom=0.3)
            # width, height = self.figView.width(), self.figView.height()
            # self.fig.resize(width, height)
            self.canvas.draw()
            self.graphicscene.addWidget(self.canvas)
            self.figView.setScene(self.graphicscene)
            self.figView.show()
        else:
            pass

    def model_train(self):
        filePath, filetype = QtWidgets.QFileDialog.getOpenFileName(self, "选取文件", "./",
                                                                   "*.*")
        if os.path.exists(filePath):
            # filePath = 'Data/5. 术前眼球内陷、复视组合（697人）-不含v2.0内容1xlsx.xlsx'
            plt.clf()
            self.figView.close()
            ax = self.fig.add_subplot(111)
            y, pred = train_model(filePath)
            plt.plot(pred, y, marker='.', label='LR Curve')
            plt.plot([0, 1], [0, 1], linestyle='--', label='Ideal Curve')
            plt.xlabel('Predicted Probability')
            plt.ylabel('True Probability in each Bin')
            plt.legend()
            # plt.show()
            self.canvas.draw()
            self.graphicscene.addWidget(self.canvas)
            self.figView.setScene(self.graphicscene)
            self.figView.show()
        else:
            pass

    def show_results(self):
        # clear scene
        plt.clf()
        self.figView.close()
        ax = self.fig.add_subplot(111)
        model_names, roc_results, fprs, tprs = draw_model_roc()
        for model, auc_value, fpr, tpr in zip(model_names, roc_results, fprs, tprs):
            plt.plot(fpr, tpr, linewidth=2, label="{model} AUC = ".format(model=model.upper()) + '%.4f' % auc_value)
        x = [0.0, 1.0]
        plt.plot(x, x, linestyle='dashed', color='red', linewidth=2, label='Random')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc=4)  # 图例的位置
        plt.title('ROC Curve')
        self.canvas.draw()
        self.graphicscene.addWidget(self.canvas)
        self.figView.setScene(self.graphicscene)
        self.figView.show()


class QtTable(QAbstractTableModel):
    def __init__(self, data):
        QAbstractTableModel.__init__(self)
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parent=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self._data.columns[col]
        return None


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MainWindow()
    # 展示窗口
    w.ui.show()
    app.exec()
