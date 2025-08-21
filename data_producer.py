### MATPLOT SETUP ###
import numpy as np
import copy

from thermo_funcs import two_terminals
from thermo_funcs import distributions as dists
from thermo_funcs import root_finder_guesses

class OptimData:
    def __init__(self, system_th:two_terminals, system_nth:two_terminals, verbose = False, n_targets = 10):
        self.system_th = copy.deepcopy(system_th)
        self.system_nth = copy.deepcopy(system_nth)
        self.n_targets = n_targets

        self.verbose = verbose

    def produce_eff_data(self, system:two_terminals):
        if self.verbose:
            print("Producing eff data")
        JR_list = []
        avg_list = []
        C_min = 1

        jmax,_,_ = system.constrained_current_max()
        C_max, err = system.optimize_for_avg(0.99*jmax, 10)

        C_list = np.linspace(C_min, C_max, self.n_targets)
        
        for C in C_list:
            system.set_transmission_avg_opt(C)
            JR = system.current_integral(system.coeff_con)
            avg = system.current_integral(system.coeff_avg)
            JR_list.append(JR)
            avg_list.append(avg)
        
        JR_arr = np.array(JR_list)
        avg_arr = np.array(avg_list)

        C_arr = np.array(C_list)
        eff_arr = JR_arr/avg_arr
        return JR_arr, eff_arr, C_arr
    
    def produce_noise_data(self, system:two_terminals):
        if self.verbose:
            print("Producing noise data")

        JR_list = []
        noise_list = []
        C_min = 0.005
        jmax,_,_ = system.constrained_current_max()
        C_max, err = system.optimize_for_noise(0.99*jmax, 10)
        C_list = np.linspace(C_min, C_max, self.n_targets)
        for C in C_list:
            system.set_transmission_noise_opt(C)
            nois = system.noise_cont(system.coeff_noise)
            JR = system.current_integral(system.coeff_con)
            JR_list.append(JR)
            noise_list.append(nois)
            
        JR_arr = np.array(JR_list)
        noise_arr = np.array(noise_list)
        C_arr = np.array(C_list)
        return JR_arr, noise_arr, C_arr

    def produce_product_data(self, system:two_terminals):
        if self.verbose:
            print("Producing product data")
        JR_list = []
        product_list = []
        theta_list = []
        C_min = 0.001
        jmax,_,_ = system.constrained_current_max()
        C_list = []
        targets = np.linspace(0,jmax, self.n_targets)
        err_list = []

        [_,_,C_max], err = system.optimize_for_product(0.99*jmax)
        print("C_min, C_max ", C_min,C_max)

        # The function of JR to C is highly non-linear, and JR increases very quickly with C, so we sample more points at low Cs to try to even out the spectrum
        base = 0.1
        C_list = np.logspace(np.emath.logn(base, C_min), np.emath.logn(base,C_max), self.n_targets, base = base)

        start_avg = 0.01
        start_noise = 0.0001

        k = 0
        for C in C_list:
            if self.verbose:
                print("On target nr ", k, "with C = ", C)
            avg, nois,err = system.set_transmission_product_opt(C, start_avg = start_avg, start_noise = start_noise)
            err_list.append(err)
            JR = system.current_integral(system.coeff_con)
            product = nois*avg
            
            JR_list.append(JR)
            product_list.append(product)
            theta_list.append([avg, nois, C])
        
            if self.verbose:
                print("Error: ", err)
                print("JR: ", JR_list[k])
            k += 1

        JR_arr = np.array(JR_list)
        product_arr = np.array(product_list)
        C_arr = np.array(theta_list)
        eff_arr = product_arr/JR_arr
        err_arr = np.array(err_list)

        return JR_arr, eff_arr, C_arr, err_arr

    def save_eff(self, system, filename):
        JR_arr, eff_arr, C_arr = self.produce_eff_data(system)
        np.savez(filename, JR_arr = JR_arr, eff_arr = eff_arr, C_arr = C_arr)
    
    def save_noise(self, system, filename):
        JR_arr, noise_arr, C_arr = self.produce_noise_data(system)
        np.savez(filename, JR_arr = JR_arr, noise_arr = noise_arr, C_arr = C_arr)

    def save_product(self, system, filename):
        JR_arr, eff_arr, C_arr, err_arr = self.produce_product_data(system)
        np.savez(filename, JR_arr = JR_arr, eff_arr = eff_arr, C_arr = C_arr, err_arr = err_arr)

    def save_example(self, target_factor, filename, exact_target = False, nth = True):        
        jmax, _,_, = self.system_nth.constrained_current_max()
        target = target_factor*jmax
        if exact_target:
            target = target_factor
        if self.verbose:
            print("Producing example for target ", target)
        if nth:
            C_avg, avg_err = self.system_nth.optimize_for_avg(target,10)
            C_noise, noise_err = self.system_nth.optimize_for_noise(target, 10)
            [prod_avg, prod_nois, C_prod], prod_err = self.system_nth.optimize_for_product(target)
        else:
            C_avg, avg_err = self.system_th.optimize_for_avg(target,10)
            C_noise, noise_err = self.system_th.optimize_for_noise(target, 10)
            [prod_avg, prod_nois, C_prod], prod_err = self.system_th.optimize_for_product(target, thetas_init=[target, target/10,0.1])
        if self.verbose:
            print("Average error: ", avg_err)
            print("Noise error: ", noise_err)
            print("Product error: ", prod_err)
        prod_vector = np.array([prod_avg, prod_nois, C_prod])
        np.savez(filename, C_avg = C_avg, C_noise = C_noise, prod_vector = prod_vector)

class SecondaryData:
    def __init__(self, system_th:two_terminals, system_nth:two_terminals, s_min, s_max, secondary_prop, n_points = 20, verbose = False):
        self.system_th = system_th
        self.system_nth = system_nth
        self.verbose = verbose
        self.secondary_prop = secondary_prop
        self.s_min = s_min
        self.s_max = s_max
        self.s_arr = np.linspace(s_min, s_max, n_points)

    def updater(self, s, system:two_terminals, start_override = None):
        if self.secondary_prop == "muR":
            system.muR = s
            system.set_fermi_dist_right()
            if start_override:
                system.adjust_limits(factor = 0.5, E_high_start=start_override)
            else:
                system.adjust_limits(factor = 0.5, E_high_start=1.5*system.E_high)
        elif self.secondary_prop == "TR":
            system.TR = s
            system.set_fermi_dist_right()
            system.adjust_limits()
        else:
            pass

    def produce_data_wrapper(self, system:two_terminals, opt_func, set_func, cond_in):
        JR_list = []
        avg_list = []
        C_list = []
        err_list = []
        noise_list = []
        k = 0
        for s in self.s_arr:
            if self.verbose:
                print("On point ", k)
            self.updater(s, system, start_override=30)
            C, err = opt_func(10, secondary_prop=self.secondary_prop)
            set_func(C)
            C_list.append(C)
            JR_list.append(system.current_integral(system.coeff_con,cond_in = cond_in(C)))
            avg_list.append(system.current_integral(system.coeff_avg, cond_in = cond_in(C)))
            noise_list.append(system.current_integral(system.coeff_noise, cond_in = cond_in(C)))
            err_list.append(err)
            
            if self.verbose:
                print("Error: ", err)
                print("JR: ", JR_list[k])
                print("Avg: ", avg_list[k])
                print("C: ", C)
 
            k += 1

        JR_arr = np.array(JR_list)
        avg_arr = np.array(avg_list)
        C_arr = np.array(C_list)
        err_arr = np.array(err_list)
        noise_arr = np.array(noise_list)
        return JR_arr, avg_arr, noise_arr, C_arr, err_arr

    def produce_eff_data(self, system:two_terminals):
        if self.verbose:
            print("Producing eff data")

        return self.produce_data_wrapper(system, system.optimize_for_best_avg, lambda C: system.set_transmission_avg_opt(C, system.coeff_con, system.coeff_avg), lambda C: system._avg_condition(C, system.coeff_con, system.coeff_avg))
  
    def produce_noise_data(self, system:two_terminals):
        if self.verbose:
            print("Producing noise data")
        return self.produce_data_wrapper(system, system.optimize_for_best_noise, system.set_transmission_noise_opt, system._noise_condition)

    def produce_product_data(self, system:two_terminals):
        if self.verbose:
            print("Producing product data")
        return self.produce_data_wrapper(system, system.optimize_for_best_product, system.set_ready_transmission_product, system._product_condition)

    def produce_example(self, system:two_terminals, target):
        JR_list = []
        avg_list = []
        C_list = []
        err_list = []
        noise_list = []
        d_avg_list = []
        d_con_list = []
        k = 0
        cond_in = system._avg_condition
        for s in self.s_arr:
            if self.verbose:
                print("On point ", k)
            self.updater(s, system)
            C, err = system.optimize_for_avg(target,10, C_limit = 1)
            system.set_transmission_avg_opt(C)
            C_list.append(C)
            JR_list.append(system.current_integral(system.coeff_con,cond_in = cond_in(C)))
            avg_list.append(system.current_integral(system.coeff_avg, cond_in = cond_in(C)))
            noise_list.append(system.current_integral(system.coeff_noise, cond_in = cond_in(C)))
            err_list.append(err)
            d_avg_list.append(system.general_integral(system.dSL_dmuR(system.transf), system._avg_condition(C)))
            d_con_list.append(system.general_integral(system.dSR_dmuR(system.transf), system._avg_condition(C)))

            if self.verbose:
                print("Error: ", err)
                print("JR: ", JR_list[k])
                print("Avg: ", avg_list[k])

            k += 1

        JR_arr = np.array(JR_list)
        avg_arr = np.array(avg_list)
        C_arr = np.array(C_list)
        err_arr = np.array(err_list)
        noise_arr = np.array(noise_list)
        d_avg_arr = np.array(d_avg_list)
        d_con_arr = np.array(d_con_list)
        return JR_arr, avg_arr, noise_arr, C_arr, err_arr, d_avg_arr, d_con_arr
    
    def save_eff(self, system, filename):
        JR_arr, avg_arr, noise_arr, C_arr, err_arr = self.produce_eff_data(system)
        np.savez(filename, JR_arr = JR_arr, avg_arr = avg_arr, C_arr = C_arr, s_arr = self.s_arr, err_arr = err_arr, noise_arr = noise_arr)
    
    def save_noise(self, system, filename):
        JR_arr, avg_arr, noise_arr, C_arr, err_arr = self.produce_noise_data(system)
        np.savez(filename, JR_arr = JR_arr, noise_arr = noise_arr, C_arr = C_arr, s_arr = self.s_arr, err_arr = err_arr, avg_arr = avg_arr)

    def save_product(self, system, filename):
        JR_arr, avg_arr, noise_arr, C_arr, err_arr = self.produce_product_data(system)
        np.savez(filename, JR_arr = JR_arr, avg_arr = avg_arr, C_arr = C_arr, s_arr = self.s_arr, err_arr = err_arr, noise_arr = noise_arr)

    def save_example(self, system, filename, target):
        JR_arr, avg_arr, noise_arr, C_arr, err_arr, d_avg_arr, d_con_arr = self.produce_example(system, target)
        np.savez(filename, JR_arr = JR_arr, avg_arr = avg_arr, noise_arr = noise_arr, C_arr = C_arr, s_arr = self.s_arr, err_arr = err_arr, d_avg_arr = d_avg_arr, d_con_arr = d_con_arr) 

class RealTransfData:

    def lorentzian(self, gamma, position):
        return lambda E: gamma**2 / ((E-position)**2 + gamma**2)

    def real_transf(self,gammas, positions, Es = [None]):
        lorentz_list = []
        for gamma, position in zip(gammas, positions):
            lorentz_list.append(self.lorentzian(gamma, position))
        
        if any(E is None for E in Es):
            transf = lambda E: min(sum(lorentz(E) for lorentz in lorentz_list),1)#if sum(lorentz(E) for lorentz in lorentz_list) <= 1 else 1
        else:
            transf = sum(lorentz(Es) for lorentz in lorentz_list)
            transf = np.clip(transf, 0,1)

        return transf    

    def produce_eff_data(self,system:two_terminals, C_list):
        
        print("Producing eff data")
        JR_list = []
        avg_list = []
        k = 0
        jmax, transf_max, roots_max = system.constrained_current_max()   
        print(roots_max)
        for C in C_list:

            cond_avg = system._avg_condition(C)    
            positions = []
            gammas = []
            for j in range(0,len(roots_max)-1,2):

                roots = root_finder_guesses(cond_avg, 0.8*roots_max[j], 1.2*roots_max[j+1])

                for i in range(0,len(roots),2):
                    positions.append((roots[i] + roots[i+1])/2)
                    gammas.append(width*np.abs(roots[i+1]-roots[i])/2)
                
            transf = self.real_transf(gammas, positions)
            system.transf = transf
            JR = system.current_integral(system.coeff_con, cond_in= system._avg_condition(C))
            avg = system.current_integral(system.coeff_avg, cond_in=system._avg_condition(C))
            JR_list.append(JR)
            avg_list.append(avg)

            k += 1

        JR_arr = np.array(JR_list)
        avg_arr = np.array(avg_list)
        C_arr = np.array(C_list)
        eff_arr = JR_arr/avg_arr
        return JR_arr, eff_arr, C_arr

    def save_eff_data(self,system, in_filename, out_filename):
        file = np.load(in_filename)
        C_list = file["C_arr"]
        JR_arr, eff_arr, C_arr = self.produce_eff_data(system, C_list)
        np.savez(out_filename, JR_arr = JR_arr, eff_arr = eff_arr, C_arr = C_arr)

if __name__ == "__main__":
    midT = 1
    deltaT = 2
    deltamu = -1.5
    muR = 0
    TR = midT
    muL = muR + deltamu
    TL = midT+deltaT
    width = 0.1

    E_low = -5
    E_high = 5

    load_params = True
    dist_type = "dippeak"
    folder = "data/"+dist_type+"/"
    # dist_type = "mixed"
    
    if load_params:
        th_dist_params = np.load(folder+"th_params_"+dist_type+".npz")['arr_0']
        muR = th_dist_params[0]
        TR = th_dist_params[1]
        nth_dist_params = np.load(folder+"nth_params_"+dist_type+".npz")['arr_0']
    else:
        th_dist_params = np.array([muR, TR])
        nth_dist_params = np.array([muL, TL, 0.1 ,0.3, 1])
        # nth_dist_params = np.array([-2, 5, -0.5, 1.2])

    dip_peak = False

    if "dippeak" in dist_type:
        occupf_L_nth = dists.thermal_with_lorentz(*nth_dist_params)
        example_target = 0.008
        dip_peak = True
    else:
        example_target = 0.0015
        occupf_L_nth = dists.two_thermals(*nth_dist_params)
        
    left_virtual = two_terminals(-20, 20, occupf_L = occupf_L_nth, muL=muL, muR=1.2, TL=TL, TR=TR)

    I_E, I_N = dists.buttiker_probe(left_virtual)

    print("Energy and particle current between thermal probe and nonthermal distribution: ", I_E, I_N)
    print("New muL and TL: ", left_virtual.muR, left_virtual.TR)

    thermal_left = two_terminals(E_low, E_high, muL=left_virtual.muR, TL = left_virtual.TR, muR = muR, TR = TR)
    nonthermal_left = two_terminals(E_low, E_high, occupf_L= occupf_L_nth, muL=muL, TL = TL, muR = muR, TR = TR)

    thermal_left.coeff_con = lambda E: thermal_left.right_entropy_coeff(E)
    thermal_left.coeff_avg = lambda E: -thermal_left.left_entropy_coeff(E)
    thermal_left.coeff_noise = lambda E: thermal_left.right_entropy_coeff(E)

    nonthermal_left.coeff_con = lambda E: nonthermal_left.right_entropy_coeff(E)
    nonthermal_left.coeff_avg = lambda E: -nonthermal_left.left_entropy_coeff(E)
    nonthermal_left.coeff_noise = lambda E: nonthermal_left.right_entropy_coeff(E)

    thermal_left.debug = False
    nonthermal_left.debug = False

    thermal_left.adjust_limits(0.5)
    nonthermal_left.adjust_limits(0.5)

    thermal_left.subdivide = True
    nonthermal_left.subdivide = True  
           
    optimData = OptimData(thermal_left, nonthermal_left, True, n_targets=10)    

    filenames = [folder+"th_"+dist_type+"_eff.npz",folder+"nth_"+dist_type+"_eff.npz",folder+"th_"+dist_type+"_noise.npz",
                folder+"nth_"+dist_type+"_noise.npz",folder+"th_"+dist_type+"_product.npz",folder+"nth_"+dist_type+"_product.npz"]
    examplefiles = [folder+"th_"+dist_type+"_example.npz", folder+"nth_"+dist_type+"_example.npz"]


    if not load_params:            
        np.savez(folder+"th_params_"+dist_type, th_dist_params)
        np.savez(folder+"nth_params_"+dist_type, nth_dist_params)
    optimData.save_eff(thermal_left, filenames[0])
    optimData.save_eff(nonthermal_left, filenames[1])
    optimData.save_noise(thermal_left, filenames[2])
    optimData.save_noise(nonthermal_left, filenames[3])            
    optimData.save_product(thermal_left, filenames[4])
    optimData.save_product(nonthermal_left, filenames[5])
    
    optimData.save_example(example_target, examplefiles[0], exact_target=True, nth = False)
    optimData.save_example(example_target, examplefiles[1], exact_target=True, nth = True)
    
    eff_file_nth = folder+"nth_"+dist_type+"_realtransf_eff.npz"
    eff_file_th = folder+"th_"+dist_type+"_realtransf_eff.npz"
        
    thermal_left.adjust_limits(0.5)
    nonthermal_left.adjust_limits(0.5)
    
    realTransfData = RealTransfData()

    realTransfData.save_eff_data(thermal_left, filenames[0],eff_file_th)
    realTransfData.save_eff_data(nonthermal_left, filenames[1],eff_file_nth)

    secondary_prop = "muR"
    filenames = [folder+"th_"+dist_type+"_eff_"+secondary_prop + ".npz",folder+"nth_"+dist_type+"_eff_"+secondary_prop + ".npz",folder+"th_"+dist_type+"_noise_"+secondary_prop + ".npz",
                folder+"nth_"+dist_type+"_noise_"+secondary_prop + ".npz",folder+"th_"+dist_type+"_product_"+secondary_prop + ".npz",folder+"nth_"+dist_type+"_product_"+secondary_prop + ".npz"]
    
    examplefile = folder+"nth_"+dist_type+"_eff_"+secondary_prop + "example.npz"
    
    secondaryData = SecondaryData(thermal_left, nonthermal_left, 0, 10, secondary_prop, n_points=10, verbose=True)

    if not load_params:            
        np.savez(folder+"th_params_"+dist_type, th_dist_params)
        np.savez(folder+"nth_params_"+dist_type, nth_dist_params)
    secondaryData.save_eff(thermal_left, filenames[0])
    secondaryData.save_eff(nonthermal_left, filenames[1])
    # secondaryData.save_noise(thermal_left, filenames[2])
    # secondaryData.save_noise(nonthermal_left, filenames[3])            
    # secondaryData.save_product(thermal_left, filenames[4])
    # secondaryData.save_product(nonthermal_left, filenames[5])
    nonthermal_left.E_high = 50
    secondaryData.save_example(nonthermal_left, examplefile, 0.015)