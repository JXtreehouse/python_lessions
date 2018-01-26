from bokeh.plotting import figure
from bokeh.io import output_notebook, show
output_notebook()
from numpy import cos, linspace

x = linspace(-6, 6, 100)
y = cos(x)
p = figure(width=500, height=500)
p.circle(x, y, size=7, color="firebrick", alpha=0.5)
show(p)