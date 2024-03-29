import matplotlib as mpl

"""Making figures for articles."""
TEST_STYLE = r".\test.mplstyle"
PUBLISH_STYLE = r".\publish.mplstyle"

def configure_new_style(style_name, base="ieee", params={}):
    base_params = mpl.style.library[base]
    base_params.update(params)
    with open(".\\" + style_name + ".mplstyle", "w") as f:
        for key, val in base_params.items():
            f.write("{:s}: {}\n".format(key, val))

def setup_style(base=TEST_STYLE, params={}):
    base_params = mpl.style.library[base].copy()
    base_params.update(params)
    mpl.rcParams.update(base_params)
    return base_params

if __name__ == "__main__":
    # configure_new_style("ieee_with_100dpi", params={"figure.dpi": 100})
    # mpl.style.use(".\\ieee_with_100dpi.mplstyle")
    params = {
        "text.usetex": True,
        "lines.linewidth": 2,
        "text.usetex": True,
        "font.family": "serif",
        "axes.titlesize": "x-large",
        "axes.labelsize": "x-large",
        "axes.titlepad": 24,
        "xtick.labelsize": "x-large",
        "ytick.labelsize": "x-large",
        "legend.fontsize": "x-large",
        "figure.figsize": [12, 8],
        "figure.dpi": 200,
    }
    configure_new_style("article", base="ieee", params=params)