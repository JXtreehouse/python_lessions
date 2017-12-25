#encoding=utf-8
import numpy as np

def main():
    #1--Integral
    from scipy.integrate import quad,dblquad
    print(quad(lambda x:np.exp(-x),0,np.inf))

if__name__ == "__main__":
    main()


