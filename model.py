#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 09:50:05 2022

@author: aubouinb
"""
import numpy as np
import control


def PropoModel(model: str, age: int, sex: bool, weight: float, height: float):
    """PK model for propofol, with BIS and MAP effect sites.

    Parameters
    ----------
    model : str
        Specify the computation of the coefficient can be 'Schnider - Minto', 'Marsh - Minto' or 'Eleveld'.
    age : int
        Age of the patient in year.
    sex : Bool
        1=male, 0=female.
    weight : float
        weight in kg.
    height : float
        height in cm.

    Returns
    -------
    v1 : float
        Volume of the blood compartment.
    A : np.array
        matrix of the linear system x(k+1) = A x(k) + B u(k),
        where x includes Cblood, Cfat, Cmuscle, Cbis, Cmap1, Cmap2 in this order.

    """
    if model == 'Schnider - Minto':

        if sex == 1:  # homme
            lbm = 1.1 * weight - 128 * (weight / height) ** 2
        else:  # femme
            lbm = 1.07 * weight - 148 * (weight / height) ** 2

        # Clearance Rates [l/min]
        cl1 = 1.89 + 0.0456 * (weight - 77) - 0.0681 * (lbm - 59) + 0.0264 * (height - 177)
        cl2 = 1.29 - 0.024 * (age - 53)
        cl3 = 0.836
        # Volume of the compartments [l]
        v1 = 4.27
        v2 = 18.9 - 0.391 * (age - 53)
        v3 = 238
        # drug amount transfer rates [1/min]
        ke0 = 0.456
        k1e = ke0
    elif model == 'Marsh - Minto':

        # Volume of the compartments [l]
        v1 = 0.228 * weight
        v2 = 0.463 * weight
        v3 = 2.893 * weight
        # Clearance Rates [l/min]
        cl1 = 0.119 * v1
        cl2 = 0.112 * v1
        cl3 = 0.042 * v1
        # drug amount transfer rates [1/min]
        ke0 = 1.2
        k1e = ke0

    elif model == 'Eleveld':

        # see D. J. Eleveld, P. Colin, A. R. Absalom, and M. M. R. F. Struys,
        # “Pharmacokinetic–pharmacodynamic model for propofol for broad application in anaesthesia and sedation”
        # British Journal of Anaesthesia, vol. 120, no. 5, pp. 942–959, mai 2018, doi:10.1016/j.bja.2018.01.018.

        # reference patient
        AGE_ref = 35
        WGT_ref = 70
        HGT_ref = 1.7
        PMA_ref = (40+AGE_ref*52)/52  # not born prematurely and now 35 yo
        BMI_ref = WGT_ref/HGT_ref**2
        GDR_ref = 1  # 1 male, 0 female

        opiate = True  # consider opiate or not
        measurement = "arterial"  # can be "arterial" or "venous"
        theta = [None,                    # just to get same index than in the paper
                 6.2830780766822,       # V1ref [l]
                 25.5013145036879,      # V2ref [l]
                 272.8166615043603,     # V3ref [l]
                 1.7895836588902,       # Clref [l/min]
                 1.7500983738779,       # Q2ref [l/min]
                 1.1085424008536,       # Q3ref [l/min]
                 0.191307,              # Typical residual error
                 42.2760190602615,      # CL maturation E50
                 9.0548452392807,       # CL maturation slope [weeks]
                 -0.015633,             # Smaller V2 with age
                 -0.00285709,           # Lower CL with age
                 33.5531248778544,      # Weight for 50 % of maximal V1 [kg]
                 -0.0138166,            # Smaller V3 with age
                 68.2767978846832,      # Maturation of Q3 [weeks]
                 2.1002218877899,       # CLref (female) [l/min]
                 1.3042680471360,       # Higher Q2 for maturation of Q3
                 1.4189043652084,       # V1 venous samples (children)
                 0.6805003109141]       # Higer Q2 venous samples

        # function used in the model
        def faging(x): return np.exp(x * (age - AGE_ref))
        def fsig(x, C50, gam): return x**gam/(C50**gam + x**gam)
        def fcentral(x): return fsig(x, theta[12], 1)

        def fal_sallami(sexX, weightX, ageX, bmiX):
            if sexX:
                return (0.88 + (1-0.88)/(1+(ageX/13.4)**(-12.7)))*(9270*weightX)/(6680+216*bmiX)
            else:
                return (1.11 + (1 - 1.11)/(1+(ageX/7.1)**(-1.1)))*(9270*weightX)/(8780+244*bmiX)

        PMA = age + 40/52
        BMI = weight/(height/100)**2

        fCLmat = fsig(PMA * 52, theta[8], theta[9])
        fCLmat_ref = fsig(PMA_ref*52, theta[8], theta[9])
        fQ3mat = fsig(PMA * 52, theta[14], 1)
        fQ3mat_ref = fsig(PMA_ref * 52, theta[14], 1)
        fsal = fal_sallami(sex, weight, age, BMI)
        fsal_ref = fal_sallami(GDR_ref, WGT_ref, AGE_ref, BMI_ref)

        if opiate:
            def fopiate(x): return np.exp(x*age)
        else:
            def fopiate(x): return 1

        # reference: male, 70kg, 35 years and 170cm

        v1 = theta[1] * fcentral(weight)/fcentral(WGT_ref)
        if measurement == "venous":
            v1 = v1 * (1 + theta[17] * (1 - fcentral(weight)))
        v2 = theta[2] * weight/WGT_ref * faging(theta[10])
        v2ref = theta[2]
        v3 = theta[3] * fsal/fsal_ref * fopiate(theta[13])
        v3ref = theta[3]
        cl1 = (sex*theta[4] + (1-sex)*theta[15]) * (weight/WGT_ref)**0.75 * \
            fCLmat/fCLmat_ref * fopiate(theta[11])

        cl2 = theta[5]*(v2/v2ref)**0.75 * (1 + theta[16] * (1 - fQ3mat))
        if measurement == "venous":
            cl2 = cl2*theta[18]

        cl3 = theta[6] * (v3/v3ref)**0.75 * fQ3mat/fQ3mat_ref

        ke0 = 0.146*(weight/70)**(-0.25)
        k1e = ke0

    # MAP effect site transfert rates
    ke0_1 = 0.0540
    ke0_2 = 0.0695
    # drug amount transfer rates [1/min]
    k10 = cl1 / v1
    k12 = cl2 / v1
    k13 = cl3 / v1
    k21 = cl2 / v2
    k31 = cl3 / v3

    # Matrices system definition
    A = np.array([[-(k10 + k12 + k13), k21, k31, 0, 0, 0],
                  [k12, -k21, 0, 0, 0, 0],
                  [k13, 0, -k31, 0, 0, 0],
                  [k1e, 0, 0, -ke0, 0, 0],
                  [ke0_1, 0, 0, 0, -ke0_1, 0],
                  [ke0_2, 0, 0, 0, 0, -ke0_2]])

    return v1, A


def RemiModel(model: str, age: int, sex: bool, weight: float, height: float):
    """PK model for remifentanil, with BIS and MAP effect sites.

    Parameters
    ----------
    model : str
        Specify the computation of the coefficient can be 'Schnider - Minto', 'Marsh - Minto' or 'Eleveld'.
    age : int
        Age of the patient in year.
    sex : Bool
        1=male, 0=female.
    weight : float
        weight in kg.
    height : float
        height in cm.

    Returns
    -------
    v1 : float
        Volume of the blood compartment.
    A : np.array
        matrix of the linear system x(k+1) = A x(k) + B u(k),
        where x includes Cblood, Cfat, Cmuscle, Cbis, Cmap in this order.

    """
    if model == 'Marsh - Minto' or model == 'Schnider - Minto':
        if sex == 1:  # homme
            lbm = 1.1 * weight - 128 * (weight / height) ** 2
        else:  # femme
            lbm = 1.07 * weight - 148 * (weight / height) ** 2

        # Clearance Rates [l/min]
        cl1 = 2.6 + 0.0162 * (age - 40) + 0.0191 * (lbm - 55)
        cl2 = 2.05 - 0.0301 * (age - 40)
        cl3 = 0.076 - 0.00113 * (age - 40)
        # Volume of the compartments [l]
        v1 = 5.1 - 0.0201 * (age - 40) + 0.072 * (lbm - 55)
        v2 = 9.82 - 0.0811 * (age - 40) + 0.108 * (lbm - 55)
        v3 = 5.42
        # drug amount transfer rates [1/min]
        ke0 = 0.595 - 0.007 * (age - 40)
        k1e = ke0  # 0.456
    if model == 'Eleveld':
        # see D. J. Eleveld et al., “An Allometric Model of Remifentanil Pharmacokinetics and Pharmacodynamics,”
        # Anesthesiology, vol. 126, no. 6, pp. 1005–1018, juin 2017, doi: 10.1097/ALN.0000000000001634.

        # function used in the model
        def faging(x): return np.exp(x * (age - 35))
        def fsig(x, C50, gam): return x**gam/(C50**gam + x**gam)

        def fal_sallami(sexX, weightX, ageX, bmiX):
            if sexX:
                return (0.88 + (1-0.88)/(1+(ageX/13.4)**(-12.7)))*(9270*weightX)/(6680+216*bmiX)
            else:
                return (1.11 + (1 - 1.11)/(1+(ageX/7.1)**(-1.1)))*(9270*weightX)/(8780+244*bmiX)

        # reference patient
        AGE_ref = 35
        WGT_ref = 70
        HGT_ref = 1.7
        BMI_ref = WGT_ref/HGT_ref**2
        GDR_ref = 1  # 1 male, 0 female

        BMI = weight/(height/100)**2

        SIZE = (fal_sallami(sex, weight, age, BMI)/fal_sallami(GDR_ref, WGT_ref, AGE_ref, BMI_ref))

        theta = [None,      # Juste to have the same index as in the paper
                 2.88,
                 -0.00554,
                 -0.00327,
                 -0.0315,
                 0.470,
                 -0.0260]

        KMAT = fsig(weight, theta[1], 2)
        KMATref = fsig(WGT_ref, theta[1], 2)
        if sex:
            KSEX = 1
        else:
            KSEX = 1+theta[5]*fsig(age, 12, 6)*(1-fsig(age, 45, 6))

        v1ref = 5.81
        v1 = v1ref * SIZE * faging(theta[2])
        V2ref = 8.882
        v2 = V2ref * SIZE * faging(theta[3])
        V3ref = 5.03
        v3 = V3ref * SIZE * faging(theta[4])*np.exp(theta[6]*(weight - WGT_ref))
        cl1ref = 2.58
        cl2ref = 1.72
        cl3ref = 0.124
        cl1 = cl1ref * SIZE**0.75 * (KMAT/KMATref)*KSEX*faging(theta[3])
        cl2 = cl2ref * (v2/V2ref)**0.75 * faging(theta[2]) * KSEX
        cl3 = cl3ref * (v3/V3ref)**0.75 * faging(theta[2])
        ke0 = 1.09 * faging(-0.0289)
        k1e = ke0

    ke0_MAP = 0.81
    # drug amount transfer rates [1/min]
    k10 = cl1 / v1
    k12 = cl2 / v1
    k13 = cl3 / v1
    k21 = cl2 / v2
    k31 = cl3 / v3

    # Matrices system definition
    A = np.array([[-(k10 + k12 + k13), k21, k31, 0, 0],
                  [k12, -k21, 0, 0, 0],
                  [k13, 0, -k31, 0, 0],
                  [k1e, 0, 0, -ke0, 0],
                  [ke0_MAP, 0, 0, 0, -ke0_MAP]])

    return v1, A


def surface_model(x_p: list, x_r: list, base_MAP: float):
    """
    Surface model for Propofol-Remifentanil interaction on BIS and MAP.

    Coefficients come from:
        - "Refining Target-Controlled Infusion: An Assessment of Pharmacodynamic Target-Controlled Infusion of
            Propofol and Remifentanil Using a Response Surface Model of Their Combined Effects on Bispectral Index"
            by Short et al.
        - "Pharmacokinetic–pharmacodynamic modeling of the hypotensive effect of remifentanil in infants undergoing
            cranioplasty" by Standing et al.
        - "Pharmacodynamic response modelling of arterial blood pressure in adult volunteers during propofol
            anaesthesia" by C. Jeleazcov et al.

    Parameters
    ----------
    x_p : list
        state vector of Propofol PK model.
    x_r : list
        state vector of remifentanil Pk model.
    base_MAP : float
        Initial MAP.

    Returns
    -------
    bis : float
        in (%).
    Map : float
        in mmHg.

    """
    # BIS computation
    c50p_bis = 4.47
    C50r_bis = 19.3
    beta = 0
    gamma = 1.43
    E0 = 97.4

    cep_BIS = x_p[3]
    cer_BIS = x_r[3]
    up = cep_BIS / c50p_bis
    ur = cer_BIS / C50r_bis
    if (up+ur) != 0:
        Phi = up/(up + ur)
    else:
        Phi = 0
    U_50 = 1 - beta * (Phi - Phi**2)
    i = (up + ur)/U_50
    bis = E0 - E0 * i ** gamma / (1 + i ** gamma)

    # MAP computation
    # propofol influence
    Emax_SAP = 54.8
    Emax_DAP = 18.1
    EC50_1 = 1.96
    gamma_1 = 4.77
    EC50_2 = 2.20
    gamma_2 = 8.49

    U = (x_p[4]/EC50_1)**gamma_1 + (x_p[5]/EC50_2)**gamma_2
    Effect_Propo = - (Emax_DAP + (Emax_SAP+Emax_DAP)/3) * U/(1+U)

    # Remifenatnil influence
    EC50 = 17.1
    gamma = 4.56
    Emax = 69.7

    Effect_remi = - Emax * (x_p[4]**gamma/(x_p[4]**gamma + EC50**gamma))

    Map = base_MAP + Effect_Propo + Effect_remi

    return bis, Map


def discretize(A: list, B: list, Ts: float = 1):
    """Discretize LTI system.

    Parameters
    ----------
    A : list
        Dynamic matrix of the system dx/dt = Ax + Bu.
    B : list
        Input matrix of the system dx/dt = Ax + Bu.
    Ts : float
        Sampling time (s). Default is 1s.

    Returns
    -------
    Ad : list
        Dynamic matrix of the system x(t+Te) = Ad x(t) + Bd u(t).
    Bd : list
        Input matrix of the system x(t+Te) = Ad x(t) + Bd u(t).

    """
    """Discretize the system dx/dt = Ax + Bu and return Ad and Bd such that x(t+Te) = Ad x(t) + Bd u(t),
        Te is given in (s)"""
    C = np.zeros((1, len(A)))
    D = 0
    model = control.ss(np.array(A)/60, B, C, D)  # A divided by 60 because it was in 1/min
    model = control.sample_system(model, Ts)
    Ad = model.A
    Bd = model.B
    return Ad, Bd
