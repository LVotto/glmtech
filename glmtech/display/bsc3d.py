import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MaxNLocator

from ..utils import eval_norm, eval_norm2, bscs_at_m

FIGURE_PATH = "D:\\USP\\masters\\python\\fig_storage"

COLOR_CYCLE_3D = list("kkrrbbgg")

def plot_bscs_3d(sph, ms=[], m_range=None, min_n=1, max_n=100, mode="tm",
                 apply_func=eval_norm2,
                 zlabel=r"$\left|g_{{n, \mathrm{{TM}}}}^{{m}}\right|^2$",
                 file_dir=FIGURE_PATH, extension=".pdf",
                 file_name="foo", show_legend=True,
                 legend_ms=None, save=True):
    ns = np.arange(min_n, max_n + 1)
    if len(ms) == 0: ms = sorted(list(sph.degrees))
    m_range = m_range or (min(ms), max(ms))
    m_range = np.arange(m_range[0], m_range[1] + 1)
    if legend_ms is None:
        legend_ms = ms
    bscs = {}
    proc = {}
    scaled_bscs = {}
    max_bscs = {}
    max_bsc = 0
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    for m in ms:
        bscs[m] = bscs_at_m(sph, m=m, min_n=min_n, max_n=max_n, mode=mode)
        proc[m] = apply_func(bscs[m])
        max_bscs[m] = np.max(eval_norm(proc[m]))
        if max_bsc < max_bscs[m]: max_bsc = max_bscs[m]
    for m in ms:
        n = 1.0
        while (n * max_bscs[m] < max_bsc / 10 and max_bsc != 0 and
               max_bscs[m] != 0 and np.log10(max_bscs[m]) - np.log10(max_bsc) > -50):
            n *= 10
        mag_text = ""
        if n != 1:
            mag_text = " $(\\times 10^{{{:d}}})$".format(int(np.log10(float(n))))
        nn, gg = ns[abs(m):], n * proc[m][abs(m):]
        scaled_bscs[m] = gg
        color = COLOR_CYCLE_3D[abs(m) % len(COLOR_CYCLE_3D)]
        label = "" if m not in legend_ms else "$m = {}${}".format(m, mag_text)
        ax.plot(
            nn, gg, zs=m, zdir="y", label=label,
            linestyle='-', linewidth=1, color=color,
        )
    for m in sorted(ms)[::-1]:
        nn, gg = ns[abs(m):], scaled_bscs[m]
        color = COLOR_CYCLE_3D[abs(m) % len(COLOR_CYCLE_3D)]
        ax.plot(
            nn, gg, zs=m, zdir="y",
            linestyle='-', linewidth=1, color=color,
        )
        ax.add_collection3d(
            ax.fill_between(nn, gg, color=color, alpha=0.3), zs=m, zdir='y'
        )
    
    for m in m_range:
        if m not in ms:
            ax.plot(ns, 0 * ns, zs=m, zdir='y', linestyle=":",
                color="k", linewidth=1)
    
    font_props = FontProperties()
    
    ax.set_proj_type("ortho")
    ax.ticklabel_format(style="sci", useOffset=True, scilimits=(0, 3))
    ax.view_init(elev=30, azim=-50)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("$n$")
    ax.set_ylabel("$m$")
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel(" ")
    ax.text(
        0, (np.max(ms) + np.min(ms)) / 2, 1.2 * max_bsc, zlabel,
        fontsize="xx-large", ha="center"
    )

    if show_legend: 
        ax.legend(loc="upper right", prop=font_props)
    ax.set_frame_on(False)
    fig.tight_layout()
    if save:
        filename = file_dir + file_name + extension
        fig.savefig(filename, bbox_inches='tight')
    plt.show()
    return fig, ax