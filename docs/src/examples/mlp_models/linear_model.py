#!/usr/bin/env python
# coding: utf-8

"""
MLP with Linear Model
=====================
"""
# %%
#


import numpy as np
import matplotlib.pyplot as plt

# %%
# Explanation

X = np.random.rand(10000)*400
Y = 0.1*X+10.1
Y_p = Y + np.random.randn(10000)*1.5
