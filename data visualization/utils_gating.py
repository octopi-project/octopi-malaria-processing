import numpy as np
from matplotlib.widgets import LassoSelector
from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path

# modified from https://matplotlib.org/stable/gallery/widgets/lasso_selector_demo_sgskip.html, https://matplotlib.org/stable/gallery/widgets/polygon_selector_demo.html


class SelectFromCollection:
    """
    Select indices from a matplotlib collection using `LassoSelector`.

    Selected indices are saved in the `ind` attribute. This tool fades out the
    points that are not part of the selection (i.e., reduces their alpha
    values). If your collection has alpha < 1, this tool will permanently
    alter the alpha values.

    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        Axes to interact with.
    collection : `matplotlib.collections.Collection` subclass
        Collection you want to select from.
    alpha_other : 0 <= float <= 1
        To highlight a selection, this tool sets all selected points to an
        alpha value of 1 and non-selected points to *alpha_other*.
    """

    def __init__(self, ax, collection, select_method='Lasso', alpha_other=0.3):
        self.ax = ax
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.alpha_other = alpha_other

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        self.path = None
        self.verts = None

        # Ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))

        if select_method == 'Lasso':
            self.selector = LassoSelector(ax, onselect=self.onselect)
        elif select_method == 'Polygon':
            self.selector = PolygonSelector(ax, self.onselect)
        else:
            print('Please use a valid select method')
        self.ind = []

    def onselect(self, verts):
        self.verts = verts
        path = Path(verts)
        self.path = path
        self.ind = np.nonzero(path.contains_points(self.xys))[0]
        self.ax.set_title('number of selected spots: ' + str(len(self.ind)) + '. Enter to save')
        self.fc[:, -1] = self.alpha_other
        # set color
        self.fc[:, 0] = 0.25
        self.fc[:, 1] = 0.25
        self.fc[:, 2] = 0.25
        self.fc[self.ind, -1] = 1
        # set to red
        self.fc[self.ind, 0] = 0.75
        self.fc[self.ind, 1] = 0
        self.fc[self.ind, 2] = 0
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

    def disconnect(self):
        self.lasso.disconnect_events()
        self.fc[:, -1] = 1
        # set to black
        self.fc[:, 0] = 0.25
        self.fc[:, 1] = 0.25
        self.fc[:, 2] = 0.25
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()