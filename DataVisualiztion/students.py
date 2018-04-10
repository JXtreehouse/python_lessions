# -*- coding:utf-8 -*-
# usr/bin/python 3.5+
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

students = pd.read_csv("data/ucdavis.csv")

g = sns.FacetGrid(students, hue="gender", palette="Set1", size=6)
g.map(plt.scatter, "gpa", "computer", s=250, linewidth= 0.65, edgecolor="white")

g.add_legend()
plt.show()