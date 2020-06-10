"""Test Init file for BGNetwork module."""

from batch_BGNetwork import BGnetwork
from stimulation_functions import *
from SMC_Timespike_creator import *
import numpy as np

# Time Settings

# maximum time (ms)
tmax = 1000

# timestep (ms)
dt = 0.01
t = np.arange(0, tmax, dt)

# BG Network parameters
pd = 0
wstim = 0


# number of neurons in each nucleus (TH, STN, GPe, GPi)
n = 10

# Signals Parameters
SMCT_freq = 14
cv = 0.4
deltasm = 5
ism = 3.5
pulse_width = 0.3
iDBS = 300
DBS_freq = 130

# initial membrane voltages for all cells
v1 = -62 + np.random.normal(loc=0.0, scale=5.0, size=((n, 1)))
v2 = -62 + np.random.normal(loc=0.0, scale=5.0, size=((n, 1)))
v3 = -62 + np.random.normal(loc=0.0, scale=5.0, size=((n, 1)))
v4 = -62 + np.random.normal(loc=0.0, scale=5.0, size=((n, 1)))

# initial array
vv0 = np.array([v1, v2, v3, v4])
r = np.random.normal(loc=0.0, scale=2.0, size=((n, 1)))

# generates stimulation trains
timespike = createSMCTimespike(tmax, dt, t, SMCT_freq, cv)
Istim = createIstim(tmax, dt, t, timespike, ism, deltasm)

if wstim == 1:
    Idbs = createIDBS(tmax, dt, t, DBS_freq, iDBS, pulse_width)
else:
    Idbs = np.zeros(t.size)

[v1, v2, v3, v4] = BGnetwork(n, pd, dt, t, tmax, vv0, r, Istim, Idbs)
