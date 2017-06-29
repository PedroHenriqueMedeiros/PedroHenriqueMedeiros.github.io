#!/usr/bin/env python2

import numpy as np
import pprint
from scipy import signal

gauss = np.array([[1,2,1],[2,4,2],[1,2,1]])
lapl = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
laplgassu = signal.convolve2d(gauss,lapl)

pprint.pprint(laplgassu)
