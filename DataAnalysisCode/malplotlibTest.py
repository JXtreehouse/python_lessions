#encoding uft-8

import numpy as np

def main():

    # draw a line
    import matplotlib.pyplot as plt
    x = np.linspace(-np.pi, np.pi,256,endpoint= True)
    c,s = np.cos(x),np.sin(x)
    plt.figure(1)

    plt.plot(x,c, color="pink",linewidth=1.0,linestyle="-",label="cos")
    plt.plot(x,s,"r.",label="sin",linewidth=1.0)
    plt.title("cos & sin")


    ax = plt.gca()
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    ax.spines["left"].set_position(("data",0))
    ax.spines["bottom"].set_position(("data",0))

    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")

    plt.xticks([],[])
    plt.yticks(np.linspace(-1,1,5,endpoint=True))

    for label in ax.get_xticklabels()+ax.get_yticklabels():
        label.set_fontsize(16)
        label.set_bbox(dict(facecolor="white",edgecolor="none",alpha=0.2))
    plt.legend(loc="upper left")
    plt.grid()
    plt.show()

if __name__ == '__main__':
    main()