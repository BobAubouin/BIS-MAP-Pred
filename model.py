#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 09:50:05 2022

@author: aubouinb
"""
import numpy as np
import control

def PropoModel(model, age, sex, weight, height):
    """PK model for propofol, with BIS and MAP effect sites.
    Inputs:         -  model: string to specify the computation of the coefficient can be 'Schnider - Minto', 'Marsh - Minto' or 'Eleveld'
                    - age: age of the patient in year
                    - sex: 1=male, 0=female
                    - weight in kg
                    - height in cm
                    
    Outputs:        - v1: volume of the blood compartment
                    - A matrix of the linear system x(k+1) = A x(k) + B u(k), where x includes Cblood, Cfat, Cmuscle, Cbis, Cmap1, Cmap2 in this order
                    """
    if model == 'Schnider - Minto':      
        
        if sex == 1: # homme
            lbm = 1.1 * weight - 128 * (weight / height) ** 2
        else : #femme
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
        
        faging = lambda x: np.exp(x * (age - 35))
        fsig = lambda x,C50,gam :  x**gam/(C50**gam + x**gam)
        fcentral = lambda x: fsig(x, 33.6, 1)
        PMA = age + 40/52
        
        fCLmat = fsig(PMA * 52, 42.3, 9.06)
        fQ3mat = fsig(PMA * 52, 68.3, 1)
        fopiate = lambda x: np.exp(x*age)
        
        BMI = weight/(height/100)**2
        BMIref = 70/1.7**2
        #reference: male, 70kg, 35 years and 170cm
        if sex:
            fal_sallami = lambda weightX,ageX,bmiX:  (0.88 + (1-0.88)/(1+(ageX/13.4)**(-12.7)))*(9270*weightX)/(6680+216*bmiX)
        else:
            fal_sallami = lambda weightX,ageX,bmiX:  (1.11 + (1 - 1.11)/(1+(ageX/7.1)**(-1.1)))*(9270*weightX)/(8780+244*bmiX)
        # Volume of the compartments [l]
        v1 = 6.28 * fcentral(weight)/fcentral(35)
        #self.v1 = self.v1 * (1 + 1.42 * (1 - fcentral(weight)))
        v2 = 25.5 * weight/70 * faging(-0.0156)
        v2ref = 25.5
        v3 = 273 * fal_sallami(weight,age,BMI)/fal_sallami(70,35,BMIref) * fopiate(-0.0138)
        v3ref = 273*fopiate(-0.0138)
        # Clearance Rates [l/min]
        cl1 = (sex*1.79 + (1-sex)*2.1) * (weight/70)**0.75 * fCLmat/fsig(35*52+40, 42.3, 9.06) * fopiate(-0.00286)
        cl2 = 1.75*(v2/v2ref)**0.75 * (1 + 1.30 * (1 - fQ3mat ))
        #cl2 = cl2*0.68
        cl3 = 1.11 * (v3/v3ref)**0.75 * fQ3mat/fsig(35*52+40, 68.3, 1)
        
        # drug amount transfer rates [1/min]
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

def RemiModel(model, age, sex, weight, height):
    """PK model for remifentanil, with BIS and MAP effect sites.
    Inputs:         -  model: string to specify the computation of the coefficient can be 'Schnider - Minto', 'Marsh - Minto' or 'Eleveld'
                    - age: age of the patient in year
                    - sex: 1=male, 0=female
                    - weight in kg
                    - height in cm
                    
    Outputs:        - v1: volume of the blood compartment
                    - A matrix of the linear system x(k+1) = A x(k) + B u(k), where x includes Cblood, Cfat, Cmuscle, Cbis, Cmap in this order
                    """
                    
    if model == 'Marsh - Minto' or model == 'Schnider - Minto':
        if sex == 1: # homme
            lbm = 1.1 * weight - 128 * (weight / height) ** 2
        else : #femme
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
        k1e = ke0 #0.456
    if model=='Eleveld':
        faging = lambda x: np.exp(x * (age - 35))
        fsig = lambda x,C50,gam :  x**gam/(C50**gam + x**gam)
        
        BMI = weight/(height/100)**2
        BMIref = 70/1.7**2
        
        if sex:
            FFM = (0.88 + (1-0.88)/(1+(age/13.4)**(-12.7))) * (9270 * weight)/(6680 + 216*BMI)
            KSEX = 1
        else:
            FFM = (1.11 + (1 - 1.11)/(1+(age/7.1)**(-1.1))) * (9270 * weight)/(8780 + 244*BMI)
            KSEX = 1 + 0.47*fsig(age,12,6)*(1 - fsig(age,45,6))
            
        FFMref = (0.88 + (1-0.88)/(1+(35/13.4)**(-12.7))) * (9270 * 70)/(6680 + 216*BMIref)
        SIZE = (FFM/FFMref)
        KMAT = fsig(weight, 2.88,2)
        KMATref = fsig(70, 2.88,2)
        v1 = 5.81 * SIZE * faging(-0.00554) 
        v2 = 8.882 * SIZE * faging(-0.00327)
        V2ref = 8.882
        v3 = 5.03 * SIZE * faging(-0.0315)*np.exp(-0.026*(weight - 70))
        V3ref = 5.03 
        cl1 = 2.58 * SIZE**0.75 * (KMAT/KMATref)*KSEX*faging(-0.00327)
        cl2 = 1.72 * (v2/V2ref)**0.75 * faging(-0.00554) * KSEX
        cl3 = 0.124 * (v3/V3ref)**0.75 * faging(-0.00554)
        ke0 = 1.09  * faging(-0.0289)
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

def surface_model(x_p, x_r, base_MAP):
    """Surface model for Propofol-Remifentanil interaction on BIS and MAP. Coefficients come from:
        - "Refining Target-Controlled Infusion: An Assessment of Pharmacodynamic Target-Controlled Infusion of
            Propofol and Remifentanil Using a Response Surface Model of Their Combined Effects on Bispectral Index" by Short et al.
        - "Pharmacokinetic–pharmacodynamic modeling of the hypotensive effect of remifentanil in infants undergoing cranioplasty" by Standing et al.
        - "Pharmacodynamic response modelling of arterial blood pressure in adult volunteers during propofol anaesthesia" by C. Jeleazcov et al.
        
        Inputs:     - x_p: state vector of Propofol PK model
                    - x_r: state vector of remifentanil Pk model
                    - base_MAP: Initial MAP
                    
        Outputs:    - bis and MAP respectively in (%) and (mmHg)
                    """
    #BIS computation
    c50p_bis = 4.47
    C50r_bis = 19.3
    beta = 0
    gamma = 1.43
    E0 = 97.4
    
    cep_BIS = x_p[3]
    cer_BIS = x_r[3]
    up = cep_BIS / c50p_bis
    ur = cer_BIS / C50r_bis
    if (up+ur)!=0:
        Phi = up/(up + ur)
    else:   
        Phi = 0
    U_50 = 1 - beta * (Phi - Phi**2)
    i = (up + ur)/U_50
    bis = E0 - E0 * i ** gamma / (1 + i ** gamma)
    
    # MAP computation
    #propofol influence
    Emax_SAP = 54.8
    Emax_DAP = 18.1
    EC50_1 = 1.96
    gamma_1 = 4.77
    EC50_2 = 2.20
    gamma_2 = 8.49
    
    
    U = (x_p[4]/EC50_1)**gamma_1 + (x_p[5]/EC50_2)**gamma_2
    Effect_Propo =  - (Emax_DAP + (Emax_SAP+Emax_DAP)/3) * U/(1+U)
    
    #Remifenatnil influence
    EC50 = 17.1
    gamma = 4.56
    Emax = 69.7
    
    Effect_remi = - Emax * (x_p[4]**gamma/(x_p[4]**gamma + EC50**gamma))
    
    Map = base_MAP + Effect_Propo + Effect_remi
    
    return bis, Map


def discretize(A,B,Te):
    """Discretize the system dx/dt = Ax + Bu and return Ad and Bd such that x(t+Te) = Ad x(t) + Bd u(t),
        Te is given in (s)"""
    C = np.zeros((1,len(A)))
    D = 0
    model = control.ss(np.array(A)/60, B, C, D) # A divided by 60 because it was in 1/min
    model = control.sample_system(model, Te)
    Ad = model.A
    Bd = model.B
    return Ad, Bd