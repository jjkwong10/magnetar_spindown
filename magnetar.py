import numpy as np
from numpy import pi,exp
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import astropy.units as u
import astropy.constants as const

#Defining gauss in cgs base units
cgsgauss = (u.g)**0.5*(u.s)**(-1)*(u.cm)**(-0.5)

#Universal Constants with units
c = (const.c).cgs
G = (const.G).cgs
sigma = (const.sigma_sb).cgs
k_B = (const.k_B).to(u.keV/u.K)

def I_fun(M_NS,R_NS):
    return 2*M_NS*(R_NS**2)/5

def B_fun(t,B_i,B_decay=True):
    if B_decay==False:
        return B_i
    else: 
        B_min = np.minimum(B_i/2,2e+13)
        B = B_i*exp(-(t/3.154e+07)/1e+06)/(1+((1e+06/(1e+19/B_i))*(1 - exp(-(t/3.154e+07)/1e+06))))
        return np.maximum(B,B_min)

def mu_fun(t,B_i,R_NS,B_decay=True):
    return B_fun(t,B_i,B_decay)*(R_NS**3)

def Rm_fun(t,Omega,delta,B_i,rho_0,v_NS,M_NS,R_NS,grav_correction=True,B_decay=True):
    Rm = ((1/(8*pi))*(mu_fun(t,B_i,R_NS,B_decay)**2)*(v_NS**(delta-2))*(Omega**(-delta))*(1/rho_0))**(1/(delta+6))
    
    if grav_correction == False or Rm>(2*G.to_value(u.cm**3/(u.g*u.s**2))*M_NS/(v_NS**2)):
        return Rm
    
    else:
        Rm_grav = (1/8*pi)*((2*G.to_value(u.cm**3/(u.g*u.s**2)*M_NS)**((delta-5)/2))*(mu_fun(t,B_i,R_NS,B_decay)**2)*(v_NS**3)*(Omega**(-delta))*(1/rho_0))**(2/(3*delta+7))
        return Rm_grav
    
    #Dipole Phase spindown function
def dipole_spindown_fun(t,Omega,gamma,delta,B_i,v_NS,rho_0,M_NS,R_NS,grav_correction=True,B_decay=True):
    dOmega = -(2*(mu_fun(t,B_i,R_NS,B_decay)**2))/(3*((c.to_value(u.cm/u.s))**3)*I_fun(M_NS,R_NS))*(Omega**3)
    return dOmega

#Dipole Phase termination event
def dipole_prop_transition(t,Omega,gamma,delta,B_i,v_NS,rho_0,M_NS,R_NS,grav_correction=True, B_decay=True):
    return Rm_fun(t,Omega,delta,B_i,rho_0,v_NS,M_NS,R_NS,grav_correction,B_decay)-(c.to_value(u.cm/u.s)/Omega)

#Propeller Phase spindown function
def propeller_spindown_fun(t,Omega,gamma,delta, B_i, v_NS, rho_0, M_NS, R_NS, grav_correction=True, B_decay=True):
    
    if grav_correction == False or Rm_fun(t,Omega,delta,B_i,rho_0,v_NS,M_NS,R_NS,grav_correction,B_decay)>(2*G.to_value(u.cm**3/(u.g*u.s**2))*M_NS/(v_NS**2)):
        dOmega = -(mu_fun(t,B_i,R_NS,B_decay)**2)*((Omega/v_NS)**(gamma-delta))*(Rm_fun(t,Omega,delta,B_i,rho_0,v_NS,M_NS,R_NS,grav_correction,B_decay)**(gamma-delta-3))/(8*I_fun(M_NS,R_NS))
        return dOmega
    
    else:
        dOmega_grav = -((2*G.to_value(u.cm**3/(u.g*u.s**2))*M_NS)**((delta-gamma)/2))*(mu_fun(t,B_i,R_NS,B_decay)**2)*(Omega**(gamma-delta))*(Rm_fun(t,Omega,delta,B_i,rho_0,v_NS,M_NS,R_NS,grav_correction,B_decay)**((3*(gamma-delta)-6)/2))/(8*I_fun(M_NS,R_NS))
        return dOmega_grav
    
#Propeller Phase termination event
def prop_accretion_transition(t,Omega,gamma, delta, B_i, v_NS, rho_0, M_NS, R_NS, grav_correction=True, B_decay=True):    
    return Rm_fun(t,Omega,delta,B_i,rho_0,v_NS,M_NS,R_NS,grav_correction,B_decay)-((G.to_value(u.cm**3/(u.g*u.s**2))*M_NS/(Omega**2))**(1/3))

class magnetar:
    '''
    Defining the magnetar object which houses the necessary functions for running a single instance of spindown.
    
    Parameters:
    gamma: gamma exponent parameter (-1,0,1,2)
    delta: delta exponent parameter (0,1,2)
    B_i: initial value for the NS surface magnetic (in G)
    v_NS: translational NS velocity (in km/s)
    rho_0: density of material around the NS (in g/cm^3) [NOTE: want to change this to be in terms of molecules/cm^3]
    P0: initial spin period of the NS at t0
    t_obvs: age of the NS at the time of "observation" (in s) [NOTE: wnat to change this to be in terms of yrs?]
    M_NS: mass of the NS (in M_sun)
    R_NS: radius of the NS (in km)
    t0: time to begin integration
    grav_correction: toggle for enabling gravitational corrections (relevant for slower moving NSs)
    B_decay: toggle for enabling magnetic field decay
    dipole_only: toggle for dipole spindown only vs. dipole + propeller spindowns together
    '''

    def __init__(self, gamma, delta, B_i, v_NS, rho_0, P0, t_obvs, M_NS=1.4, R_NS=10, t0=0, 
                 grav_correction=True, B_decay=True, dipole_only=False):
        self.gamma = gamma
        self.delta = delta
        self.B_i = B_i*cgsgauss
        self.v_NS = v_NS*(u.km/u.s)
        self.rho_0 = rho_0*(u.g/(u.cm**3))
        self.P0 = P0*(u.s)
        self.Omega0 = 2*pi/self.P0
        self.t0 = t0*(u.s)
        self.t_obvs = t_obvs*(u.s)
        self.M_NS = M_NS*const.M_sun
        self.R_NS = R_NS*u.km
        self.grav_correction = grav_correction
        self.B_decay = B_decay
        self.dipole_only = dipole_only
        
        self.obvs_type = '0'

    def run(self):

        if self.dipole_only==True:
            dipole_prop_transition.terminal=False
            prop_accretion_transition.terminal=True
        else:
            dipole_prop_transition.terminal=True
            prop_accretion_transition.terminal=True
        
        dipole_integrator = solve_ivp(fun=dipole_spindown_fun,t_span=(self.t0.to_value(u.s),self.t_obvs.to_value(u.s)),y0=np.array([self.Omega0.to_value(1/u.s)]),
                                      args=(self.gamma,self.delta,self.B_i.to_value(cgsgauss),self.v_NS.to_value(u.cm/u.s),self.rho_0.to_value(u.g/(u.cm**3)),self.M_NS.to_value(u.g),self.R_NS.to_value(u.cm),self.grav_correction,self.B_decay),
                                      events=dipole_prop_transition,method='RK45',max_step=1E+12,rtol=1E-5,atol=1E-26)
        
        if self.dipole_only == True or dipole_integrator.t[-1]==self.t_obvs.to_value(u.s):
            self.obvs_type = '1'
            self.t_arr = dipole_integrator.t
            self.Omega_arr = dipole_integrator.y[0]
            self.dOmega_arr = dipole_spindown_fun(self.t_arr,self.Omega_arr,self.gamma,self.delta,self.B_i.to_value(cgsgauss),self.v_NS.to_value(u.cm/u.s),self.rho_0.to_value(u.g/(u.cm**3)),self.M_NS.to_value(u.g),self.R_NS.to_value(u.cm),self.grav_correction,self.B_decay)
            
        else:
            propeller_integrator = solve_ivp(fun=propeller_spindown_fun,t_span=(dipole_integrator.t[-1],self.t_obvs.to_value(u.s)),y0=np.array([dipole_integrator.y[0][-1]]),
                                             args=(self.gamma,self.delta,self.B_i.to_value(cgsgauss),self.v_NS.to_value(u.cm/u.s),self.rho_0.to_value(u.g/(u.cm**3)),self.M_NS.to_value(u.g),self.R_NS.to_value(u.cm),self.grav_correction,self.B_decay),
                                             events=prop_accretion_transition,method='RK45',max_step=1E+12,rtol=1E-5,atol=1E-26)
            
            self.t_arr = np.append(dipole_integrator.t,propeller_integrator.t[1:])
            self.Omega_arr = np.append(dipole_integrator.y[0],propeller_integrator.y[0][1:])
            dipole_dOmega_arr = dipole_spindown_fun(dipole_integrator.t,dipole_integrator.y[0],self.gamma,self.delta,self.B_i.to_value(cgsgauss),self.v_NS.to_value(u.cm/u.s),self.rho_0.to_value(u.g/(u.cm**3)),self.M_NS.to_value(u.g),self.R_NS.to_value(u.cm),self.grav_correction,self.B_decay)
            prop_dOmega_arr = propeller_spindown_fun(propeller_integrator.t[1:],propeller_integrator.y[0][1:],self.gamma,self.delta,self.B_i.to_value(cgsgauss),self.v_NS.to_value(u.cm/u.s),self.rho_0.to_value(u.g/(u.cm**3)),self.M_NS.to_value(u.g),self.R_NS.to_value(u.cm),self.grav_correction,self.B_decay)
            self.dOmega_arr = np.append(dipole_dOmega_arr,prop_dOmega_arr)

            if dipole_integrator.t[-1]==self.t_obvs.to_value(u.s):
                self.obvs_type = '2'

            else:
                self.obvs_type = '3'
        
        self.B_arr = (B_fun(self.t_arr,self.B_i.to_value(cgsgauss),B_decay=self.B_decay))*cgsgauss
        self.t_arr = self.t_arr*u.s
        self.Omega_arr = self.Omega_arr*(1/u.s)
        self.dOmega_arr = self.dOmega_arr*(1/(u.s**2))
        self.P_arr = 2*pi/(self.Omega_arr)
        self.P_dot_arr = -(1/(2*pi))*self.dOmega_arr*(self.P_arr**2)
        self.dP_arr = -2*pi*self.dOmega_arr/(self.Omega_arr**2)
        self.distance = (self.v_NS*self.t_arr[-1]).to(u.kpc)

        if self.obvs_type == '3':
            M_dot = (((2*G.to(u.cm**3/(u.g*u.s**2))*self.M_NS.to(u.g))**2)*pi*self.rho_0/(self.v_NS**3)).to(u.g/u.s)
            self.kT_eff = (k_B*(G.to(u.cm**3/(u.g*u.s**2))*self.M_NS*M_dot/(4*pi*sigma*(self.R_NS**3)))**(1/4)).to(u.keV)