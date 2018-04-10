import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
students = pd.read_csv("data/ucdavis.csv")
g = sns.FacetGrid(students, palette="Set10",size=7)
g.map(plt.scatter, "momheight", "height",s=140, linewidth=.7,edgecolor = "#ffad40",color="#ff8000")
g.set_axis_labels("Mothers Heilsght", "Students Height")
plt.show()