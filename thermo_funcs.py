#Define functions
import numpy as np
import copy

import matplotlib.pyplot as plt
import warnings

from scipy import integrate
from scipy.optimize import minimize, fsolve
from inspect import signature

## GLOBAL CONSTANTS ##
# NOT USED #
# h = 1
# kb = 1
# e = 1
# N = 1


# Brute force root finder, decent enough for 1D problems with < 10 roots
def root_finder_guesses(func, x_low, x_high, tol = 1e-8):
    roots = []
    xs = np.linspace(x_low, x_high, 100000)
    signed = np.sign(func(xs))

    x_inits = xs[np.argwhere(signed[1:] - signed[:-1] != 0).flatten()]
    for x in x_inits:
        res = fsolve(func, x, factor = 0.1)[0]
        if any(np.abs(roots - res) < tol):
            continue
        roots.append(res)
    
    return roots

class two_terminals:

    def __init__(self, E_low, E_high, transf = lambda E: 1, cond_in = lambda E: 1, occupf_L = None,occupf_R = None,  muL = 0, TL = 1, muR = 0, TR = 1,
                 coeff_avg = None, coeff_noise = None, coeff_con = None, subdivide = False, debug = False):
        '''
        Class containing the functions needed for quantum transport between two terminals.
        Initialization:
            E_low: The lower energy limit. Integrals are at most calculated down to the limit.
            E_high: The higher energy limit. Integrals are at most calculated up to the limit. 
            transf (function of E): The transmission function as a function of energy
            cond_in (function of E): Condition used by integral calculations for better accuracy. The same condition that determines transf. 
            occupf_L (function of E): Distribution function for left reservoir
            occupf_R (function of E): Distribution function for right reservoir
            muL: Electrochemical potential of left reservoir
            TL: Temperature of the left reservoir
            muR: Electrochemical potential of right reservoir
            TR: temperature of right reservoir
            coeff_avg: Current coefficient used for the resource current
            coeff_noise: Current coefficient for the noise. Should likely be the same as for the output current
            coeff_con: Current coefficient for the output/fixed current
            subdivide: If True, all integrals will use cond_in to divide the integration into pieces, such that an integration over several boxcars is done only where they are 1 and not 0. 
                       If the regions where the integrated function is 0 are large, then the integral output will be inaccurate, often just 0. Especially for optimal transmission functions for 
                       fixed currents much smaller than maximum, this is needed.
            debug: If True, various debugging quantities are printed. 
        '''
        self.z = muR

        self.E_low = E_low
        self.E_high = E_high
        self.muL = muL
        self.muR = muR
        self.TL = TL
        self.TR = TR
        self.transf = transf
        self.cond_in = cond_in

        self.subdivide = subdivide
        self.debug = debug
        # Standard case is two thermal functions
        if occupf_L == None:
            self.set_fermi_dist_left()
        else:
            self.set_occupf_L(occupf_L)

        
        if occupf_R == None:
            self.set_fermi_dist_right()
        else:
            self.occupf_R = occupf_R

        # Set standard currents we want to compare. We choose the efficiency -J_R/P with J_R fixed as standard, and the output noise considered
        if coeff_avg == None:
            self.coeff_avg = lambda E: -self.power_coeff(E)
        else:
            self.coeff_avg = coeff_avg
        if coeff_con == None:
            self.coeff_con = self.right_heat_coeff
        else:
            self.coeff_con = coeff_con
        if coeff_noise == None:
            self.coeff_noise = self.right_heat_coeff
        else:
            self.coeff_noise = coeff_noise
        self.set_occup_roots()

    ########## Thermodynamic quantities ######################
    def carnot(self):
        return (1-self.TR/self.TL)

    def cop(self):
        return self.TR/(self.TL-self.TR)

    def pmax(self):
        A = 0.0321
        p = A * np.pi**2 *(self.TL-self.TR)**2
        return p
    ###########################################################
    ############## Current Coefficients ####################
    def entropy_coeff(E, occupf):
        coeff = np.log(occupf(E)/(1-occupf(E)))
        return coeff

    def left_entropy_coeff(self,E):
        return -two_terminals.entropy_coeff(E, self.occupf_L)

    def right_entropy_coeff(self,E):
        return  two_terminals.entropy_coeff(E, self.occupf_R)

    def left_heat_coeff(self,E):
        return E-self.muL

    def right_heat_coeff(self,E):
        return -E+self.muR

    def left_particle_coeff(self,E):
        return 1

    def right_particle_coeff(self,E):
        return -1

    def left_electric_coeff(self,E):
        return self.muL

    def right_electric_coeff(self,E):
        return -self.muR
    
    def power_coeff(self,E):
        return (self.muR-self.muL)

    def left_energy_coeff(self,E):
        return E

    def right_energy_coeff(self, E):
        return -E
    
    def left_noneq_free_coeff(self,E):
        return -E - self.TR*self.left_entropy_coeff(E)
    ##########################################################
    ############# Setting functions for transmissions and distributions #############
    def set_full_transmission(self):
        self.transf = lambda E: 1
        self.cond_in = lambda E: 1

    def set_fermi_dist_left(self):
        self.occupf_L = lambda E: distributions.fermi_dist(E, self.muL, self.TL)
    
    def set_fermi_dist_right(self):
        self.occupf_R = lambda E: distributions.fermi_dist(E, self.muR, self.TR)

    def set_transmission_noise_opt(self, C):
        self.transf = self._transmission_noise(C)
        self.cond_in = self._noise_condition(C)

    def set_transmission_avg_opt(self, C, coeff_x = None, coeff_y = None):
        self.transf = self._transmission_avg(C, coeff_x, coeff_y)
        self.cond_in = self._avg_condition(C, coeff_x, coeff_y)

    def set_transmission_product_opt(self, C, alpha = 0.5, start_avg = None, start_noise = None):
        calc_avg, calc_noise, err = self.calc_for_product_determined(C,alpha, start_avg, start_noise)
        self.transf = self._transmission_product([calc_avg,calc_noise, C], alpha)
        self.cond_in = self._product_condition([calc_avg,calc_noise, C], alpha)
        return calc_avg, calc_noise, err

    def set_ready_transmission_product(self, thetas, alpha = 0.5):
        self.transf = self._transmission_product(thetas, alpha)
        self.cond_in = self._product_condition(thetas, alpha)
        
    def set_occupf_L(self, occupf_L, z = 0):
        self.occupf_L = occupf_L

    def get_efficiency(self):
        return self.current_integral(self.coeff_con)/self.current_integral(self.coeff_avg)

    ############# DIFFERENTIATIONS FOR SECONDARY OPT #############
    def dfR_dmuR(self):
        return lambda E: self.occupf_R(E)**2 * np.exp((E-self.muR)/self.TR)/self.TR

    def dfR_dTR(self):
        return lambda E: self.occupf_R(E)**2 * np.exp((E-self.muR)/self.TR)* (E - self.muR)/(self.TR**2)

    def dSL_dmuR(self, transf):
        return lambda E: -self.coeff_avg(E)*transf(E)*self.dfR_dmuR()(E)
                
    def dFL_dmuR(self,transf):
        return lambda E: -self.coeff_avg(E)*transf(E)*self.dfR_dmuR()(E)

    def dSR_dmuR(self, transf):
        return lambda E: transf(E)*((self.occupf_L(E)- self.occupf_R(E))/self.TR - self.coeff_con(E)*self.dfR_dmuR()(E))

    def dSL_dTR(self, transf):
        return lambda E: -self.coeff_avg(E)*transf(E)*self.dfR_dTR()(E)

    def dSR_dTR(self, transf):
        return lambda E: -transf(E)*self.coeff_con(E)*((self.occupf_L(E)- self.occupf_R(E))/self.TR+self.dfR_dTR()(E))

    def dFL_dTR(self, transf):
        return lambda E: -self.coeff_avg(E)*transf(E)*self.dfR_dTR()(E)

    def dnoiseSR_dmuR(self, transf):
        thermal = lambda E: self.occupf_L(E)*(1-self.occupf_L(E)) + self.occupf_R(E)*(1-self.occupf_R(E))
        return lambda E: transf(E)*self.coeff_con(E)*(-2/self.TR*thermal(E))+ self.coeff_con(E)*self.dfR_dmuR()(E)*(1-2*self.occupf_R(E))

    def dnoiseSR_dTR(self, transf):
        thermal = lambda E: self.occupf_L(E)*(1-self.occupf_L(E)) + self.occupf_R(E)*(1-self.occupf_R(E))
        return lambda E: transf(E)*self.coeff_con(E)**2*(-2/self.TR*thermal(E))+ self.dfR_dTR()(E)*(1-2*self.occupf_R(E))
    ##############################################################
    

    ############# Practical functions for finding integration windows #############
    def find_occup_roots(self, tol = 1e-8):
        occupdiff = lambda E: (self.occupf_L(E) - self.occupf_R(E))        
        roots = root_finder_guesses(occupdiff, self.E_low, self.E_high,tol)
        if len(roots) == 0:
            warnings.warn("No occup roots found!", RuntimeWarning)
        return roots
    
    def set_occup_roots(self, tol = 1e-8):
        self.occuproots = self.find_occup_roots(tol)

    def constrained_current_max(self, tol = 1e-8, return_roots_con = False):
        func = lambda E:self.coeff_con(E)*(self.occupf_L(E) - self.occupf_R(E))

        roots_con = root_finder_guesses(self.coeff_con, self.E_low, self.E_high,tol)        
        if len(roots_con) == 0:
            warnings.warn("No roots found for con coefficient!", RuntimeWarning)

        self.set_occup_roots()
        roots = roots_con + self.occuproots
        transf = lambda E: np.heaviside(func(E),0)
        
        jmax = self.current_integral(self.coeff_con, transf_in = transf, cond_in = None, ignore_sub=True)
        
        if return_roots_con:
            return jmax, transf, roots, roots_con    
        return jmax, transf, roots

    def adjust_limits(self, factor = 0.1, E_low_start = -5, E_high_start = 5):
        self.E_low = E_low_start
        self.E_high = E_high_start
        jmax, transf, roots = self.constrained_current_max()
        if len(roots) == 0:
            raise Exception("Limits could not be adjusted to roots. Try wider starting values")
        lowest = np.min(roots)
        highest = np.max(roots)
        self.E_low = lowest - factor*np.abs(lowest)

        self.E_high = highest + factor*np.abs(highest)
    
        if self.debug:
            print("Roots: ", roots)
            print("New limits: ",self.E_low, self.E_high)
    
    #######################################
    
    ########## Integral functions #########
    def current_integral(self, coeff, transf_in = None, cond_in = None, ignore_sub = False):
        if transf_in == None:
            transf = self.transf
        else:
            transf = transf_in 

        if cond_in == None:
            cond_in = self.cond_in    
        else:
            cond_in = cond_in

        integrand = lambda E: coeff(E)*transf(E)*(self.occupf_L(E)- self.occupf_R(E))
        
        if self.subdivide and not ignore_sub:

            roots = root_finder_guesses(cond_in,self.E_low, self.E_high, tol = 1e-8)
            roots = np.sort(roots)
            current = 0
            
            # Specifically for the entropy currents... check that there are roots between the first two. 
            if len(roots) == 0 or len(roots) % 2 != 0:
                warnings.warn("No roots, reverting to regular integration", RuntimeWarning)
                current, err = integrate.quad(integrand, self.E_low, self.E_high, args=(), points=self.occuproots, limit = 1000)
                return current
            for i in range(0,len(roots),2):
                tmp_curr = integrate.quad(integrand, roots[i], roots[i+1], args=(), points=self.occuproots, limit = 1000)[0]           
                current +=  tmp_curr

        else:
            current, err = integrate.quad(integrand, self.E_low, self.E_high, args=(), points=self.occuproots, limit = 1000)
        return current

    def noise_cont(self, coeff, transf = None, only_thermal = True, cond_in = None):
        if transf == None:
            transf = self.transf
        thermal = lambda E: coeff(E)**2*transf(E)*(self.occupf_L(E)*(1-self.occupf_L(E))+ self.occupf_R(E)*(1-self.occupf_R(E)))
        if not only_thermal:
            shot = lambda E: coeff(E)**2 * self.N**2*transf(E)*(1-transf(E))*(self.occupf_L(E)+self.occupf_R(E))**2
            integrand = lambda E: thermal(E) + shot(E)
        else:
            integrand = thermal
        
        if cond_in == None:
            cond_in = self.cond_in    
        else:
            cond_in = cond_in
        
        if self.subdivide:
                
            roots = root_finder_guesses(cond_in,self.E_low, self.E_high)
            current = 0
            roots = np.sort(roots)

            if len(roots) == 0 or len(roots) % 2 != 0:
                warnings.warn("No roots, reverting to regular integration", RuntimeWarning)
                current, err = integrate.quad(integrand, self.E_low, self.E_high, args=(), points = self.occuproots, limit = 100)
                return current
            for i in range(0,len(roots),2):
                tmp_curr = integrate.quad(integrand, roots[i], roots[i+1], args=(), points=self.occuproots, limit = 100)[0] 

                current += tmp_curr

        else:
            current, err = integrate.quad(integrand, self.E_low, self.E_high, args=(), points = self.occuproots, limit = 100)
        return current
    
    def general_integral(self, func, cond_in = None):
        if cond_in == None:
            cond_in = self.cond_in    
        else:
            cond_in = cond_in
        if self.subdivide:
            res = 0
            roots = root_finder_guesses(cond_in,self.E_low, self.E_high, tol = 1e-8)
            roots = np.sort(roots)
            
            if len(roots) == 0 or len(roots) % 2 != 0:
                warnings.warn("No roots, reverting to regular integration", RuntimeWarning)
                res, err = integrate.quad(func, self.E_low, self.E_high, args=(), points=self.occuproots, limit = 100)
            
            else:
                for i in range(0,len(roots),2):
                    tmp_res = integrate.quad(func, roots[i], roots[i+1], args=(), points=self.occuproots, limit = 100)[0]           
                    
                    res +=  tmp_res           
        else:
            res, err = integrate.quad(func,self.E_low, self.E_high, args=(), points=self.occuproots, limit = 100)
        
        return res

    #####################################

    ###### Functions for minimizing resource (avg) current ##########

    def _avg_condition(self, C, coeff_x = None, coeff_y= None):
        if coeff_x ==  None and coeff_y == None:
            coeff_x = self.coeff_con
            coeff_y = self.coeff_avg
        return lambda E: -(coeff_y(E) - C*coeff_x(E))*(self.occupf_L(E) - self.occupf_R(E))

    def _transmission_avg(self, C, coeff_x = None, coeff_y = None):
        transf = lambda E: np.heaviside(self._avg_condition(C, coeff_x, coeff_y)(E), 0)
        return transf

    def optimize_for_avg(self,target, C_init = None, fixed = "nominator", C_limit = 1):
        '''
        Make sure coeffs are defined such that positive contributions to currents are desirable and negative suppressed
        '''
        if fixed == "nominator":
            coeff_nom = self.coeff_con
            coeff_denom = self.coeff_avg
            
        else:
            coeff_denom = self.coeff_con
            coeff_nom = self.coeff_avg

        transf = lambda C: self._transmission_avg(C, coeff_nom, coeff_denom)
        def fixed_current_eq(C):
            
            if C < C_limit:
                return 1000
            current = self.current_integral(self.coeff_con,transf(C), self._avg_condition(C, coeff_nom, coeff_denom)) - target
            if self.debug:
                print("C: ",C)
                print("current: ", current)
            return current
        res = fsolve(fixed_current_eq,C_init, factor = 0.1, xtol = 1e-6)
        self.transf = self._transmission_avg(res[0], coeff_nom, coeff_denom)
        err = fixed_current_eq(res[0])
        return res[0], err

    def optimize_for_best_avg(self,C_init = 1, fixed = "nominator", secondary_prop = "muR", left_current = "entropy", C_limit = 1):
        if fixed == "nominator":
            coeff_nom = self.coeff_con
            coeff_denom = self.coeff_avg
            
        else:
            coeff_denom = self.coeff_con
            coeff_nom = self.coeff_avg
        
        if secondary_prop == "muR":
            if left_current == "entropy":
                d_right = self.dSR_dmuR
                d_left = self.dSL_dmuR
            elif left_current == "free":
                d_right = self.dSR_dmuR
                d_left = self.dFL_dmuR
            else:
                print("Invalid left current")
                return -1

        elif secondary_prop == "TR":
            if left_current == "entropy":
                d_right = self.dSR_dTR
                d_left = self.dSL_dTR
            elif left_current == "free":
                pass
            else:
                print("Invalid left current")
                return -1
        elif secondary_prop == "z":
            if left_current == "entropy":
                pass
            elif left_current == "free":
                pass
            else:
                print("Invalid left current")
                return -1
        else:
            print("Invalid secondary prop")
            return -1
    
        transf = lambda C: self._transmission_avg(C, coeff_nom, coeff_denom)
        def func(C):
            if C < C_limit:
                return 1000

            dL_integ = self.general_integral(d_left(transf(C)), self._avg_condition(C, self.coeff_con, self.coeff_avg))
            dR_integ = self.general_integral(d_right(transf(C)), self._avg_condition(C, self.coeff_con, self.coeff_avg))

            if self.debug:
                print("dL_integ: ", dL_integ)
                print("dR_integ: ", dR_integ)
                print("d frac: ", dL_integ/dR_integ if dR_integ != 0.0 else 0)
                print("C: ", C)
            return dL_integ/dR_integ - C if dR_integ != 0.0 else 1000
        res = fsolve(func, C_init, factor = 0.1, xtol = 1e-6)
        self.transf = transf(res[0])
        diff = func(res[0])
        return res[0], diff

    ########################################################

    ######## Functions for minimizing noise ######################

    def _noise_condition(self,C):
        div = lambda E: self.coeff_noise(E)**2*(self.occupf_L(E) *(1-self.occupf_L(E)) + self.occupf_R(E)*(1-self.occupf_R(E)))
        return lambda E: C*self.coeff_con(E)*(self.occupf_L(E) - self.occupf_R(E)) - div(E)

    def _transmission_noise(self, C):
        #integrands = lambda E: self.coeff_con(E)*(self.occupf_L(E)- self.occupf_R(E))
        comp = self._noise_condition(C)
        transf = lambda E: np.heaviside(comp(E), 0)#*np.heaviside(integrands(E), 0) \
                #+np.heaviside(- (comp - C), 0)*np.heaviside(-integrands, 0)
        if self.debug:
            print("C",C)       
            # print("Con current", self.current_integral(self.coeff_con))
        
        return transf
        
    def optimize_for_noise(self, target, C_init = None):
        if C_init == None:
            C_init = 10
        transf = lambda C: self._transmission_noise(C)        
        fixed_current_eq = lambda C: self.current_integral(self.coeff_noise, transf(C), cond_in=self._noise_condition(C)) - target
        res = fsolve(fixed_current_eq,C_init, factor = 0.1, xtol=1e-6)
        self.transf = self._transmission_noise(res[0])
        return res[0], fixed_current_eq(res[0])

    def optimize_for_best_noise(self,C_init = 1, secondary_prop = "muR"):
        transf = lambda C: self._transmission_noise(C)
        
        C_limit = 0
        if secondary_prop == "muR":
            d_noise = self.dnoiseSR_dmuR
            d_right = self.dSR_dmuR

        elif secondary_prop == "TR":
            d_right = self.dSR_dTR
            d_noise = self.dnoiseSR_dTR
        else:
            print("Invalid secondary prop")
            return -1
    

        def func(C):
            if C < C_limit:
                return 1000

            dnoise_integ = self.general_integral(d_noise(transf(C)), cond_in=self._noise_condition(C))
            dR_integ = self.general_integral(d_right(transf(C)), cond_in=self._noise_condition(C))

            if self.debug:
                print("dnoise_integ: ", dnoise_integ)
                print("dR_integ: ", dR_integ)
                print("d frac: ", dnoise_integ/dR_integ if dR_integ != 0.0 else 1000)
                print("C: ", C)
            return dnoise_integ/dR_integ - C if dR_integ != 0.0 else 1000
        res = fsolve(func, C_init, factor = 0.1, xtol = 1e-6)
        self.transf = transf(res[0])

        print(func(res[0]))
        return res[0], func(res[0])

  ################################################################
  
  ############### Functions for minimizing trade-off (product) ###############
    def _product_condition(self,thetas, alpha = 0.5):
        con_avg = thetas[0]
        con_noise = thetas[1]
        C = thetas[2]
        div = lambda E: self.coeff_con(E)*(self.occupf_L(E)-self.occupf_R(E))
        term_one = lambda E: -2*alpha*con_avg*self.coeff_noise(E)**2*(self.occupf_L(E)*(1-self.occupf_L(E))+self.occupf_R(E)*(1-self.occupf_R(E)))#/div(E)
        term_two = lambda E: -2*(1-alpha)*con_noise*self.coeff_avg(E)*(self.occupf_L(E)-self.occupf_R(E))#/div(E)
        term_three = lambda E: C*div(E)
        opt_func = lambda E: term_one(E)+term_two(E)+term_three(E)
        return opt_func

    def _transmission_product(self,thetas, alpha = 0.5):
        transf = lambda E: np.heaviside(self._product_condition(thetas, alpha)(E), 0)
        return transf
        
    def optimize_for_product(self, target, thetas_init = None, alpha = 0.5, secondary = False):
        if thetas_init == None:
            max_transf = lambda E: np.heaviside(self.coeff_con(E)*(self.occupf_L(E) - self.occupf_R(E)),0)
            self.transf = max_transf
            thetas_init = [target, target, 3]
            
            if self.debug:
                print("Thetas init: ", thetas_init)

        def opt_func(thetas):
            if any(thetas < 0):
                return 1000,1000,1000
            transf = self._transmission_product(thetas)
            nois = self.noise_cont(self.coeff_noise, transf, cond_in = self._product_condition(thetas, alpha))
            avg = self.current_integral(self.coeff_avg, transf, cond_in = self._product_condition(thetas, alpha))
            con = self.current_integral(self.coeff_con, transf, cond_in = self._product_condition(thetas, alpha))
            if self.debug: 
                print("Thetas: ", thetas)
                print("opt func: ", avg - thetas[0], nois - thetas[1], con-target)
            return avg - thetas[0], nois - thetas[1], con-target
        
        res = fsolve(opt_func, thetas_init, factor=1, xtol=1e-6)
        err = opt_func(res)
        # print(err)
        self.transf = self._transmission_product(res)
        return res, err

    def calc_for_product_determined(self, C, alpha = 0.5, start_avg = None, start_noise = None):
             
        def opt_func(thetas):
            con_noise = thetas[1]
            con_avg = thetas[0]
            if any(thetas < 0):
                return 1000,1000
            transf = self._transmission_product([con_avg, con_noise, C], alpha)
            nois = self.noise_cont(self.coeff_noise, transf, cond_in = self._product_condition([con_avg, con_noise, C], alpha))
            avg = self.current_integral(self.coeff_avg, transf, cond_in = self._product_condition([con_avg, con_noise, C], alpha))
            if self.debug: 
                print("Thetas: ", thetas)
                print("Noise: ", nois)
                print("Avg: ", avg)
                print("opt func: ", avg - con_avg, nois - con_noise)
            return avg - con_avg, nois - con_noise

        
        if start_avg == None or start_noise == None:
            start_avg = np.random.uniform(0,0.005)
            start_noise = np.random.uniform(0,0.005)

        calc_avg, calc_noise = fsolve(opt_func, [start_avg,start_noise], factor = 0.1)
        err = np.max(np.abs(opt_func(np.array([calc_avg, calc_noise, float(C)]))))
        return calc_avg,calc_noise, err

    def optimize_for_best_product(self,C_init = 1, alpha = 0.5, secondary_prop = "muR"):
        C_limit = 0
        if secondary_prop == "muR":
            d_noise = self.dnoiseSR_dmuR
            d_right = self.dSR_dmuR
            d_left = self.dSL_dmuR
        elif secondary_prop == "TR":
            d_right = self.dSR_dTR
            d_noise = self.dnoiseSR_dTR
            d_left = self.dSL_dTR
        else:
            print("Invalid secondary prop")
            return -1
        def func(C):
            if C < C_limit:
                return 1000
            calc_avg, calc_noise,err = self.calc_for_product_determined(C, alpha)
            transf = self._transmission_product([calc_avg, calc_noise, C], alpha)
            dnoise_integ = self.general_integral(d_noise(transf), self._product_condition([calc_avg, calc_noise, C], alpha))
            dR_integ = self.general_integral(d_right(transf), self._product_condition([calc_avg, calc_noise, C], alpha))
            dL_integ = self.general_integral(d_left(transf), self._product_condition([calc_avg, calc_noise, C], alpha))

            frac = (calc_avg*dnoise_integ + calc_noise*dL_integ)/dR_integ if dR_integ != 0.0 else 1000
            if self.debug:
                print("dnoise_integ: ", dnoise_integ)
                print("dR_integ: ", dR_integ)
                print("dL_integ: ", dL_integ)
                print("d frac: ", frac)
                print("C: ", C)
            return frac - C
        res = fsolve(func, C_init, factor = 0.1, xtol = 1e-6)
        C = res[0]
        calc_avg, calc_noise, err = self.calc_for_product_determined(C, alpha)
        self.transf = self._transmission_product([calc_avg, calc_noise, C], alpha)

        return [calc_avg, calc_noise, C],func(C)
    #################################################################



### Class defining thremal and nonthermal distributions and the buttiker probe
class distributions:
    
    def fermi_dist(E, mu, T):
        f_dist = 1/(1+np.exp((E-mu)/(T)))
        return f_dist
    
    def thermal_with_lorentz(mu, T, width, height, position):
        lorentz_max = 2/(np.pi * width)
        height_factor = height/lorentz_max
        rel_position = position

        lorentzian = lambda E: height_factor * (width/((E-position-mu)**2 + width**2))
        reflect = lambda E: -height_factor * (width/((E+position-mu)**2 + width**2))
        fermi = lambda E: distributions.fermi_dist(E,mu,T)
        dist = lambda E: fermi(E) + lorentzian(E) + reflect(E)
        # if dist(position) < 0 or dist(position) > 1:
        #     print("Warning! Invalid occupation function", dist(position))
        
        return dist

    def two_thermals(mu1, T1, mu2, T2):
        dist1 = lambda E: distributions.fermi_dist(E,mu1,T1)
        dist2 = lambda E: distributions.fermi_dist(E,mu2,T2)
        return lambda E: 0.5*(dist1(E) + dist2(E))

    def buttiker_probe(system:two_terminals, set_right = True):
        mu_init = system.muR
        T_init = system.TR
        sys_copy = copy.deepcopy(system)
        mus = np.linspace(-1,1,10)
        def find_mu_T(params):
            mu = params[0]
            #mu = mus[np.random.randint(0,9)]
            T = params[1]
            sys_copy.muR = mu
            sys_copy.TR = T
            
            # Make sure to update the fermi distribution in the system
            sys_copy.set_fermi_dist_right()
            I_E = sys_copy.current_integral(sys_copy.left_energy_coeff)
            I_N = sys_copy.current_integral(sys_copy.left_particle_coeff)
            return I_E, I_N
        params_init = [mu_init, T_init]
        res = fsolve(find_mu_T, params_init, factor=0.1)

        system.muR = res[0]
        system.TR = res[1]
        system.set_fermi_dist_right()
        return find_mu_T(res)

