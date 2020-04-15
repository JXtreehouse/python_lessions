#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/4/9 上午9:46
# @Author : AlexZ33
# @Site : 
# @File : number.py
# @Software: PyCharm

import sys, os, numpy

def isfloat(x):
    """
    Check if arguments is float
    """
    try:
        a = float(x)
    except ValueError:
        return False
    else
        return True
