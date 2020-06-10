from numpy import zeros
from numpy.random import gamma 

def createSMCTimespike(tmax, dt, t, freq, cv):
    i = 1
    A = 1 / cv ** 2
    B = freq / A
    if cv == 0:
        instfreq = freq
    else:
        instfreq = gamma(A, B)
    ipi = 1000 / freq
    i = i + round(ipi / dt)
    timespike = []

    while i < t.size:
        timespike.append(t[i])
        A = 1 / cv ** 2
        B = freq / A
        if cv == 0:
            instfreq = freq
        else:
            instfreq = gamma(A, B)
        ipi = 1000 / instfreq
        i = i + round(ipi / dt)
    return timespike

def createIstim(tmax,dt,t,timespike,ism = 3.5,deltasm = 5):
    Istim = zeros(t.size)
    i=0; j=0
    while (i < t.size - int(deltasm/dt)) and j < len(timespike):
        if t[i-1] <= timespike[j] <= t[i]:
            #print("Spike at t: "+str(t[i]))
            for k in range(int(deltasm/dt)-1):
                Istim[i+k]=ism
            j=j+1
        i=i+1
    return Istim