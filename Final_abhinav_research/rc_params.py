import matplotlib as mpl

fontsize = 9
black = 'k'

mpl.rcdefaults()  # Set to defaults

mpl.rc('text', usetex=True)
mpl.rc('text.latex', preamble=r'\usepackage{amssymb}\usepackage{fdsymbol}')
mpl.rcParams['font.size'] = fontsize
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'

mpl.rcParams['axes.labelsize'] = fontsize
mpl.rcParams['axes.edgecolor'] = black
# mpl.rcParams['axes.xmargin'] = 0
mpl.rcParams['axes.labelcolor'] = black
mpl.rcParams['axes.titlesize'] = fontsize

mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['xtick.labelsize'] = fontsize
mpl.rcParams['ytick.labelsize'] = fontsize
mpl.rcParams['xtick.color'] = black
mpl.rcParams['ytick.color'] = black
# Make the ticks thin enough to not be visible at the limits of the plot (over the axes border)
mpl.rcParams['xtick.major.width'] = mpl.rcParams['axes.linewidth'] * 0.95
mpl.rcParams['ytick.major.width'] = mpl.rcParams['axes.linewidth'] * 0.95
# The minor ticks are little too small, make them both bigger.
mpl.rcParams['xtick.minor.size'] = 2.4  # Default 2.0
mpl.rcParams['ytick.minor.size'] = 2.4
mpl.rcParams['xtick.major.size'] = 3.9  # Default 3.5
mpl.rcParams['ytick.major.size'] = 3.9
mpl.rcParams["xtick.minor.visible"] =  True
mpl.rcParams["ytick.minor.visible"] =  True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['xtick.top'] = True

ppi = 72  # points per inch
# dpi = 150
mpl.rcParams['figure.titlesize'] = fontsize
mpl.rcParams['figure.dpi'] = 150  # To show up reasonably in notebooks
mpl.rcParams['figure.constrained_layout.use'] = True
# 0.02 and 3 points are the defaults:
# can be changed on a plot-by-plot basis using fig.set_constrained_layout_pads()
mpl.rcParams['figure.constrained_layout.wspace'] = 0.0
mpl.rcParams['figure.constrained_layout.hspace'] = 0.0
mpl.rcParams['figure.constrained_layout.h_pad'] = 3. / ppi  # 3 points
mpl.rcParams['figure.constrained_layout.w_pad'] = 3. / ppi

mpl.rcParams['legend.title_fontsize'] = fontsize
mpl.rcParams['legend.fontsize'] = fontsize
mpl.rcParams['legend.edgecolor'] = 'inherit'  # inherits from axes.edgecolor, to match
mpl.rcParams['legend.facecolor'] = (1, 1, 1, 0.6)  # Set facecolor with its own alpha, so edgecolor is unaffected
mpl.rcParams['legend.fancybox'] = True
mpl.rcParams['legend.borderaxespad'] = 0.8
mpl.rcParams['legend.framealpha'] = None  # Do not set overall alpha (affects edgecolor). Handled by facecolor above
mpl.rcParams['patch.linewidth'] = 0.8  # This is for legend edgewidth, since it does not have its own option

mpl.rcParams['hatch.linewidth'] = 0.5
mpl.rcParams["axes.axisbelow"] = False
