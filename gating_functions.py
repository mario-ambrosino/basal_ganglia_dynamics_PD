from numpy import exp

def gpe_ainf(V):
    return 1/(1+exp(-(V+57)/2))
def gpe_hinf(V):
    return 1/(1+exp((V+58)/12))
def gpe_minf(V):
    return 1/(1+exp(-(V+37)/10))
def gpe_ninf(V):
    return 1/(1+exp(-(V+50)/14))
def gpe_rinf(V):
    return 1/(1+exp((V+70)/2))
def gpe_sinf(V):
    return 1/(1+exp(-(V+35)/2))
def gpe_tauh(V):
    return 0.05+0.27/(1+exp(-(V+40)/(-12))) #da controllare
def gpe_taun(V):
    return 0.05+0.27/(1+exp(-(V+40)/(-12))) #da controllare

# Heaviside function approximation
def Hinf(V):
    return 1/(1+exp(-(V+57)/2))

# Subthalamic Nucleus Functions
def stn_ainf(V):
    return 1/(1+exp(-(V+63)/7.8))
def stn_binf(R): #da controllare
    return 1/(1+exp(-(R-0.4)/0.1))-1/(1+exp(4))
def stn_cinf(V):
    return 1/(1+exp(-(V+20)/8))
def stn_hinf(V):
    return 1/(1+exp((V+39)/3.1))
def stn_minf(V):
    return 1/(1+exp(-(V+30)/15))
def stn_ninf(V):
    return 1/(1+exp(-(V+32)/8.0))
def stn_rinf(V):
    return 1/(1+exp((V+67)/2))
def stn_sinf(V):
    return 1/(1+exp(-(V+39)/8))
def stn_tauc(V):
    return 1+10/(1+exp((V+80)/26))
def stn_tauh(V):
    return 1+500/(1+exp(-(V+57)/(-3)))
def stn_taur(V):
    return 7.1+17.5/(1+exp(-(V+68)/(-2.2)))
def stn_taun(V):
    return 1+100/(1+exp(-(V+80)/(-26)))

# Thalamus Functions
def th_hinf(V):
    return 1/(1+exp((V+41)/4))
def th_minf(V):
    return 1/(1+exp(-(V+37)/7))
def th_pinf(V):
    return 1/(1+exp(-(V+60)/6.2))
def th_rinf(V):
    return 1/(1+exp((V+84)/4))
def th_tauh(V):
    return 1/(0.128*exp(-(V+46)/18)+4/(1+exp(-(V+23)/5)))
def th_taur(V):
    return 0.15*(28+exp(-(V+25)/10.5))