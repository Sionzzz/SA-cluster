import pandas as pd
from numpy import *
import numpy as np
import csv
import os
import matplotlib.pyplot as plt
import collections
from bokeh.transform import dodge
from bokeh.core.properties import value
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource


def result_pic(path_SA, path_S):
    # mod
    nmi_value = collections.defaultdict(list)
    mod_value = collections.defaultdict(list)
    pic_index = []
    for file in os.listdir(path_SA):
        z = file.find("mod.csv")
        x = file.find("nmi.csv")
        if z > -1 or x > -1:
            if z > -1:
                print("z")
                path_mod = path_SA + "/mod.csv"
                reader = csv.reader(open(path_mod))
                for i, line in enumerate(reader):
                    if i != 0:
                        mod_value["SA"].append(line[1])
                        pic_index.append(str(i))
            else:
                path_nmi = path_SA + "/nmi.csv"
                reader = csv.reader(open(path_nmi))
                for i, line in enumerate(reader):
                    if i != 0:
                        nmi_value["SA"].append(line[1])

    # nmi
    for file in os.listdir(path_S):
        z = file.find("mod.csv")
        x = file.find("nmi.csv")
        if z > -1 or x > -1:
            if z > -1:
                path_mod = path_S + "/mod.csv"
                reader = csv.reader(open(path_mod))
                for i, line in enumerate(reader):
                    if i != 0:
                        mod_value["S"].append(line[1])
            else:
                path_nmi = path_S + "/nmi.csv"
                reader = csv.reader(open(path_nmi))
                for i, line in enumerate(reader):
                    if i != 0:
                        nmi_value["S"].append(line[1])

    print(mod_value, len(mod_value))
    print(nmi_value, len(nmi_value))
    df_mod = pd.DataFrame(mod_value, index=pic_index)
    df_nmi = pd.DataFrame(nmi_value, index=pic_index)

    col_mod = ColumnDataSource(df_mod)
    col_nmi = ColumnDataSource(df_nmi)

    p_mod = figure(x_range=pic_index, y_range=(0, 1), plot_height=350, title="mod", tools="")
    p_nmi = figure(x_range=pic_index, y_range=(0, 1), plot_height=350, title="nmi", tools="")

    p_mod.vbar(x=dodge('index', -0.25, range=p_mod.x_range),
               top='SA', width=0.2, source=col_mod, color="#c9d9d3",
               legend=value("SA"))
    p_mod.vbar(x=dodge('index', 0.0, range=p_mod.x_range),
               top='S', width=0.2, source=col_mod, color="#718dbf",
               legend=value("S"))
    p_nmi.vbar(x=dodge('index', -0.25, range=p_nmi.x_range),
               top='SA', width=0.2, source=col_nmi, color="#c9d9d3",
               legend=value("SA"))
    p_nmi.vbar(x=dodge('index', 0.0, range=p_nmi.x_range),
               top='S', width=0.2, source=col_nmi, color="#718dbf",
               legend=value("S"))

    show(p_nmi)
    show(p_mod)


if __name__ == "__main__":
    print("this is result_pic")
    path_SA = "C:/Users/zzz/Desktop/SA-cluster_data"
    path_S = "C:/Users/zzz/Desktop/S-cluster_data"
    result_pic(path_SA, path_S)