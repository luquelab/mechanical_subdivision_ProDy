# import pyqtgraph as pg
# plt = pg.plot()
# scatter = pg.ScatterPlotItem(size=10)


# # importing Qt widgets
# from PyQt5.QtWidgets import *
#
# # importing system
# import sys
#
# # importing numpy as np
# import numpy as np
#
# # importing pyqtgraph as pg
# import pyqtgraph as pg
# from PyQt5.QtGui import *
#
#
# class Window(QMainWindow):
#
#     def __init__(self):
#         super().__init__()
#
#         # setting title
#         self.setWindowTitle("PyQtGraph")
#
#         # setting geometry
#         self.setGeometry(100, 100, 600, 500)
#
#         # icon
#         icon = QIcon("skin.png")
#
#         # setting icon to the window
#         self.setWindowIcon(icon)
#
#         # calling method
#         self.UiComponents()
#
#         # showing all the widgets
#         self.show()
#
#     # method for components
#     def UiComponents(self):
#         # creating a widget object
#         widget = QWidget()
#
#         # creating a label
#         label = QLabel("Geeksforgeeks Scatter Plot")
#
#         # making label do word wrap
#         label.setWordWrap(True)
#
#         # creating a plot window
#         plot = pg.plot()
#
#         # number of points
#         n = 300
#
#         # creating a scatter plot item
#         # of size = 10
#         # using brush to enlarge the of white color with transparency is 50%
#         scatter = pg.ScatterPlotItem(
#             size=10, brush=pg.mkBrush(255, 255, 255, 120))
#
#         # getting random position
#         pos = np.random.normal(size=(2, n), scale=1e-5)
#
#         # creating spots using the random position
#         spots = [{'pos': pos[:, i], 'data': 1}
#                  for i in range(n)] + [{'pos': [0, 0], 'data': 1}]
#
#         # adding points to the scatter plot
#         scatter.addPoints(spots)
#
#         # add item to plot window
#         # adding scatter plot item to the plot window
#         plot.addItem(scatter)
#
#         # Creating a grid layout
#         layout = QGridLayout()
#
#         # minimum width value of the label
#         label.setMinimumWidth(130)
#
#         # setting this layout to the widget
#         widget.setLayout(layout)
#
#         # adding label in the layout
#         layout.addWidget(label, 1, 0)
#
#         # plot window goes on right side, spanning 3 rows
#         layout.addWidget(plot, 0, 1, 3, 1)
#
#         # setting this widget as central widget of the main window
#         self.setCentralWidget(widget)
#
#
# # create pyqt5 app
# App = QApplication(sys.argv)
#
# # create the instance of our Window
# window = Window()
#
# # start the app
# sys.exit(App.exec())


from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

app = QtGui.QApplication([])
w = gl.GLViewWidget()
w.show()
g = gl.GLGridItem()
w.addItem(g)

from prody import *
import numpy as np
pdb = '6b0x'
nc = 210

results = np.load('./' + pdb + '/' + pdb + '_' + str(nc) + '_results.npz')
labels = results['labels']
calphas = loadAtoms('../models/' + 'calphas_' + pdb + '.ag.npz')
pos = calphas.getCoords()
print(pos)

sp2 = gl.GLScatterPlotItem(pos=pos)
w.addItem(sp2)

#generate a color opacity gradient
norm = mpl.colors.Normalize(vmin=np.min(labels), vmax=np.max(labels))
color = np.zeros((pos.shape[0],4), dtype=np.float32)
from cmaps import generate_colormap
cmap = generate_colormap(int(np.max(labels)))
#cmap = plt.get_cmap('Accent')
rgba = cmap(norm(labels))
rgba[:,3] = np.ones_like(labels)*0.9
print(rgba[:,3])
size = np.ones_like(labels)*20
sp2.setData(color=rgba, size=size)
sp2.setGLOptions('opaque')

# def update():
#     ## update volume colors
#     global color
#     color = np.roll(color,1, axis=0)
#     sp2.setData(color=color)
#
# t = QtCore.QTimer()
# t.timeout.connect(update)
# t.start(50)


## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, PYQT_VERSION):
        QtGui.QApplication.instance().exec_()