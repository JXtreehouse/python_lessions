from .shared_imports import *


def plt_configure(ax=None, xlabel=None, ylabel=None, title='', legend=False, tight=False, figsize=False, no_axis=False):
    if ax == None:
        ax=plt.gca() # 返回当前ａｘｅｓ对象的句柄值
        plt.suptitle(title)
    else:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if legend
        if isinstance(legend, dict):
            ax.legend(**legend)
        else:
            ax.legend()
    if tight:

