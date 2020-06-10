## Import Libraries

import numpy as np
from stimulation_functions import *

from gating_functions import *

# Basal Ganglia Markov model
def BGnetwork(n, pd, dt, t, tmax, initial_v, r, Istim, Idbs):
    """
    n : (int) - Number of Neurons
    pd : (int) - PD on/off
    initial_v : array[float] - initial value for membrane voltage

    """
    Cm = 1
    # "l" - correnti spiking dispersione (leakage)
    gl = np.array([0.05, 2.25, 0.1])
    El = np.array([-70, -60, -65])
    # "na" - correnti spiking - sodio
    gna = np.array([3, 37, 120])
    Ena = np.array([50, 55, 55])
    # "k" - correnti spiking - potassio
    gk = np.array([5, 45, 30])
    Ek = np.array([-75, -80, -80])
    # "t" - canale ionico a bassa soglia - calcio
    gt = np.array([5, 0.5, 0.5])
    Et = 0
    # "ca" - canale ionico ad alta soglia - calcio
    gca = np.array([0, 2, 0.15])
    Eca = np.array([0, 140, 120])
    # "hp" - canale ionico di potassio indotto da presenza ioni calcio
    # post iperpolarizzazione
    gahp = np.array([0, 20, 10])
    k1 = np.array([0, 15, 10])
    kca = np.array([0, 22.5, 15])

    # TODO
    A = np.array([0, 3, 2, 2])
    B = np.array([0, 0.1, 0.04, 0.04])
    the = np.array([0, 30, 20, 20])

    # Parametri sinaptici
    gsyn = np.array([1, 0.3, 1, 0.3, 1, 0.08])  # conducibilità sinaptica
    Esyn = np.array([-85, 0, -85, 0, -85, -85])
    tau = 5
    gpeak = 0.43
    gpeak1 = 0.3

    # Definisce array vuoti per le tensioni
    vth = np.zeros((n, t.size))  # thalamic membrane voltage
    vsn = np.zeros((n, t.size))  # STN membrane voltage
    vge = np.zeros((n, t.size))  # GPe membrane voltage
    vgi = np.zeros((n, t.size))  # GPi membrane voltage
    S2 = np.zeros((n, 1))
    S21 = np.zeros((n, 1))
    S3 = np.zeros((n, 1))
    S31 = np.zeros((n, 1))
    S32 = np.zeros((n, 1))
    S4 = np.zeros((n, 1))
    Z2 = np.zeros((n, 1))
    Z4 = np.zeros((n, 1))

    # initial conditions
    vth[:, 0] = np.array(initial_v[0]).reshape(n)
    vsn[:, 0] = np.array(initial_v[1]).reshape(n)
    vge[:, 0] = np.array(initial_v[2]).reshape(n)
    vgi[:, 0] = np.array(initial_v[3]).reshape(n)

    # variabili iniziali di gating
    N2 = stn_ninf(vsn[:, 0]).reshape((n, 1))
    N3 = gpe_ninf(vge[:, 0]).reshape((n, 1))
    N4 = gpe_ninf(vgi[:, 0]).reshape((n, 1))
    H1 = th_hinf(vth[:, 0]).reshape((n, 1))
    H2 = stn_hinf(vsn[:, 0]).reshape((n, 1))
    H3 = gpe_hinf(vge[:, 0]).reshape((n, 1))
    H4 = gpe_hinf(vgi[:, 0]).reshape((n, 1))
    R1 = th_rinf(vth[:, 0]).reshape((n, 1))
    R2 = stn_rinf(vsn[:, 0]).reshape((n, 1))
    R3 = gpe_rinf(vge[:, 0]).reshape((n, 1))
    R4 = gpe_rinf(vgi[:, 0]).reshape((n, 1))

    # concentrazioni uguali in ogni nucleo come condizione iniziale
    CA2 = 0.1
    CA3 = CA2
    CA4 = CA2
    # DA CAPIRE
    C2 = stn_cinf(vsn[:, 0]).reshape((n, 1))

    ## time-loop
    for i in range(1, t.size):
        print("#timestep {} - {}".format(i, i*dt))
        # 1: TH - 2: STN - 3: GPe - 4 GPi
        # Membrane potential initial parameter
        V1 = vth[:, i - 1].reshape((n, 1))
        V2 = vsn[:, i - 1].reshape((n, 1))
        V3 = vge[:, i - 1].reshape((n, 1))
        V4 = vgi[:, i - 1].reshape((n, 1))

        # Synapse parameters
        S21[1:n] = S2[0:n-1]
        S21[0] = S2[n-1]
        S31[:n-1] = S3[1:]
        S31[n-1] = S3[1]
        S32[2:] = S3[:n-2]
        S32[0:1] = S3[n-1:n]

        # membrane paremeters
        m1 = th_minf(V1).reshape((n, 1))
        m2 = stn_minf(V2).reshape((n, 1))
        m3 = gpe_minf(V3).reshape((n, 1))
        m4 = gpe_minf(V4).reshape((n, 1))
        n2 = stn_ninf(V2).reshape((n, 1))
        n3 = gpe_ninf(V3).reshape((n, 1))
        n4 = gpe_ninf(V4).reshape((n, 1))
        h1 = th_hinf(V1).reshape((n, 1))
        h2 = stn_hinf(V2).reshape((n, 1))
        h3 = gpe_hinf(V3).reshape((n, 1))
        h4 = gpe_hinf(V4).reshape((n, 1))
        p1 = th_pinf(V1).reshape((n, 1))
        a2 = stn_ainf(V2).reshape((n, 1))
        a3 = gpe_ainf(V3).reshape((n, 1))
        a4 = gpe_ainf(V4).reshape((n, 1))
        b2 = stn_binf(R2).reshape((n, 1))
        s3 = gpe_sinf(V3).reshape((n, 1))
        s4 = gpe_sinf(V4).reshape((n, 1))
        r1 = th_rinf(V1).reshape((n, 1))
        r2 = stn_rinf(V2).reshape((n, 1))
        r3 = gpe_rinf(V3).reshape((n, 1))
        r4 = gpe_rinf(V4).reshape((n, 1))
        c2 = stn_cinf(V2).reshape((n, 1))
        tn2 = stn_taun(V2).reshape((n, 1))
        tn3 = gpe_taun(V3).reshape((n, 1))
        tn4 = gpe_taun(V4).reshape((n, 1))
        th1 = th_tauh(V1).reshape((n, 1))
        th2 = stn_tauh(V2).reshape((n, 1))
        th3 = gpe_tauh(V3).reshape((n, 1))
        th4 = gpe_tauh(V4).reshape((n, 1))
        tr1 = th_taur(V1).reshape((n, 1))
        tr2 = stn_taur(V2).reshape((n, 1))
        tr3 = 30
        tr4 = 30
        tc2 = stn_tauc(V2).reshape((n, 1))

        # thalamic cell currents
        Il1 = gl[0] * (V1 - El[0])

        Ina1 = gna[0] * (m1 ** 3) * H1 * (V1 - Ena[0])
        Ik1 = gk[0] * 0.75 * np.multiply((1 - H1), (V1 - Ek[0]))
        It1 = gt[0] * (p1 ** 2) * R1 * (V1 - Et)
        Igith = 1.4 * gsyn[5] * np.multiply(V1 - Esyn[5], S4)

        # STN cell currents
        Il2 = gl[1] * (V2 - El[1])
        Ik2 = gk[1] * np.multiply(N2 ** 4, V2 - Ek[1])
        Ina2 = gna[1] * (m2 ** 3) * np.multiply(H2, V2 - Ena[1])
        It2 = gt[1] * np.multiply(np.multiply(a2 ** 3, b2 ** 2), V2 - Eca[1])
        Ica2 = gca[1] * np.multiply(C2 ** 2, (V2 - Eca[1]))
        Iahp2 = gahp[1] * (V2 - Ek[2]) * (CA2 / (CA2 + k1[1]))
        Igesn = 0.5 * (gsyn[0] * np.multiply(V2 - Esyn[0], S3 + S31))
        Iappstn = 33 - pd * 10

        # GPe cell currents
        Il3 = gl[2] * (V3 - El[2])
        Ik3 = gk[2] * np.multiply(N3 ** 4, V3 - Ek[2])
        Ina3 = gna[2] * np.multiply(m3 ** 3, V3 - Ena[2])
        It3 = gt[2] * np.multiply(np.multiply(a3 ** 3, R3), (V3 - Eca[2]))
        Ica3 = gca[2] * np.multiply(s3 ** 2, V3 - Eca[2])
        Iahp3 = gahp[2] * (V3 - Ek[2]) * (CA3 / (CA3 + k1[2]))
        Isnge = 0.5 * gsyn[1] * np.multiply(V3 - Esyn[1], S2 + S21)
        Igege = 0.5 * gsyn[2] * np.multiply(V3 - Esyn[2], S31 + S32)
        Iappgpe = 21 - 13 * pd + r

        # GPi cell currents
        Il4 = gl[2] * (V4 - El[2])
        Ik4 = gk[2] * np.multiply(N4 ** 4, V4 - Ek[2])
        Ina4 = gna[2] * np.multiply(np.multiply(m4 ** 3, H4), V4 - Ena[2])
        It4 = gt[2] * np.multiply(np.multiply(a4 ** 3, R4), V4 - Eca[2])
        Ica4 = gca[2] * np.multiply(s4 ** 2, V4 - Eca[2])
        Iahp4 = gahp[2] * (V4 - Ek[2]) * (CA4 / (CA4 + k1[2]))
        Isngi = 0.5 * (gsyn[3] * np.multiply(V4 - Esyn[3], S2 + S21))
        Igigi = 0.5 * (gsyn[4] * np.multiply(V4 - Esyn[4], S31 + S32))
        Iappgpi = 22 - pd * 6

        # Differential Equations for cells

        # thalamus
        vth[:, i] = (V1 + dt * (1 / Cm * (-Il1 - Ik1 - Ina1 - It1 - Igith
                                          + Istim[i]))).reshape(n)
        H1 = H1 + dt * ((h1 - H1) / th1)
        R1 = R1 + dt * ((r1 - R1) / tr1)
        #pdb.set_trace()
        # STN
        vsn[:, i] = (V2 + dt * (1 / Cm * (-Il2 - Ik2 - Ina2 - It2 - Ica2
                                          - Iahp2 - Igesn + Iappstn
                                          + Idbs[i]))).reshape(n)
        N2 = N2 + dt * (0.75 * (n2 - N2) / tn2)
        H2 = H2 + dt * (0.75 * (h2 - H2) / th2)
        R2 = R2 + dt * (0.2 * (r2 - R2) / tr2)
        CA2 = CA2 + dt * (3.75 * 10 ** (-5) * (-Ica2 - It2 - kca[2] * CA2))
        C2 = C2 + dt * (0.08 * (c2 - C2) / tc2)

        # cerca in quale neurone si ha la presenza di uno spike
        stim_index = np.where(
            vsn[:, i - 1] < -10) and np.where(vsn[:, i] > -10)
        # assegna a quei neuroni che hanno sparato un valore pari alla ???
        # DA CAPIRE! immagino alla corrente "impulsiva" fornita al STN
        stim_bringer = np.zeros((n, 1))
        stim_bringer[stim_index] = gpeak / (tau * np.exp(-1)) / dt
        S2 = S2 + dt * Z2
        zdot = stim_bringer - 2 / tau * Z2 - 1 / (tau ^ 2) * S2
        Z2 = Z2 + dt * zdot

        # GPe
        vge[:, i] = (V3 + dt *
                     (1 / Cm *(-Il3 - Ik3 - Ina3 - It3 - Ica3 - Iahp3
                               - Isnge - Igege + Iappgpe))).reshape(n)
        N3 = N3 + dt * (0.1 * (n3 - N3) / tn3)
        H3 = H3 + dt * (0.05 * (h3 - H3) / th3)
        R3 = R3 + dt * (1 * (r3 - R3) / tr3)
        CA3 = CA3 + dt * (1 * 10 ** (-4) * (-Ica3 - It3 - kca[2] * CA3))
        S3 = S3 - B[2] * S3 + dt * A[2] * np.multiply((1 - S3),(Hinf(V3 - the[2])))

        # GPi
        vgi[:, i] = (V4 + dt * (1 / Cm * (-Il4 - Ik4 - Ina4 -It4 - Ica4 - Iahp4 - Isngi - Igigi + Iappgpi))).reshape(n)
        N4 = N4 + dt * (0.1 * (n4 - N4) / tn4)
        H4 = H4 + dt * (0.05 * (h4 - H4) / th4)
        R4 = R4 + dt * (1 * (r4 - R4) / tr4)
        CA4 = CA4 + dt * (1 * 10 ** (-4) * (-Ica4 - It4 - kca[2] * CA4))
        stim_index = np.where(vgi[:, i - 1] < -10) and np.where(vgi[:, i] > -10)
        stim_bringer = np.zeros((n, 1))
        stim_bringer[stim_index] = gpeak1 / (tau * np.exp(-1)) / dt
        S4 = S4 + dt * Z4
        zdot = stim_bringer - 2 / tau * Z4 - 1 / (tau ^ 2) * S4
        Z4 = Z4 + dt * zdot

    return [vth, vsn, vge, vgi]
