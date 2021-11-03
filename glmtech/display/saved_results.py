import pickle
import matplotlib.pyplot as plt
import numpy as np

from ..utils import eval_norm2

CACHE_PATH = "..\\cached\\"
CMAP = "hot"


class SavedResult:
    def __init__(self, domain=[], image=[], comment="",
                 filename="", cache_folder=CACHE_PATH):
        self.domain = domain
        self.image = image
        self.comment = comment
        self.cache_folder = cache_folder
        self.filename = filename
    
    def dump(self, cache_folder=None, filename=None):
        if cache_folder is None: cache_folder = self.cache_folder
        if filename is None: filename = self.filename
        file = cache_folder + filename
        with open(file, "wb") as f:
            pickle.dump(self, f)


class SavedFieldResult(SavedResult):
    def __init__(self, *args, field_obj=None, **kwargs):
        self.field = field_obj
        super().__init__(*args, **kwargs)


class SavedFieldResult2D(SavedFieldResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.extent = (
            np.min(self.domain[0]), np.max(self.domain[0]),
            np.min(self.domain[1]), np.max(self.domain[1])
        )
    
    def make_fig(self, cmap=CMAP, func=lambda x: x, figsize=(3.5, 3.5),
            	interpolation="gaussian", **ax_params):
        fig = plt.Figure(figsize=figsize)
        ax = fig.add_subplot(**ax_params)
        im = ax.imshow(func(self.image), cmap=cmap, extent=self.extent,
            interpolation=interpolation)
        fig.tight_layout()
        return fig, ax
