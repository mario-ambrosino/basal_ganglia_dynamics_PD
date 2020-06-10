from numpy import zeros

def createIDBS(tmax, dt, t, DBS_freq, iDBS, pulse_width=0.3):
    """ Creates DBS train of frequency f, of length tmax (msec),
    with time step dt (msec)"""
    IDBS=zeros(t.size)
    i=0
    while (i < t.size):
        for k in range(int(pulse_width/dt)):
            IDBS[k] = iD
        instfreq=freq
        isi=1000/instfreq
        i=i+round(isi*1/dt)
    return IDBS