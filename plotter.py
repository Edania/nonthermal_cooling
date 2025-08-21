### MATPLOT SETUP ###
import matplotlib.pyplot as plt
import numpy as np
import copy

from thermo_funcs import two_terminals
from thermo_funcs import distributions as dists
from thermo_funcs import root_finder_guesses
from matplotlib.lines import Line2D
from matplotlib.collections import PolyCollection
from scipy.signal import find_peaks

#cold = "#007EF5"
cold = "#009BFF"
#hot = "#FF4400"
hot = "#FF4C00"
nonthermal = "#A400FF"
nocool = "#A5ABB0"
nocooldark = "#4A5157"
negative = "#BC7B7D"

nth_one = "#DB70FF"
nth_two = "#9D52CC"
nth_three = "#6F20A0"

th_one = "#EC7846"
th_two = "#DA622F"
th_three = "#B34C20"

plt.style.use("nonthermal_style.mplstyle")



pt = 1/72
col = 246*pt

class OptimPlot:
    def __init__(self, system_th:two_terminals, system_nth:two_terminals, verbose = False):
        self.system_th = copy.deepcopy(system_th)
        self.system_nth = copy.deepcopy(system_nth)
        self.verbose = verbose

    def make_figure(self,examplefile, make_eff = True, make_noise = True, make_product = True,
                    filenames = [None]*6, dip_peak = False):
        file = np.load(examplefile)
        C_avg = file["C_avg"]
        C_noise = file["C_noise"]
        C_prod = file["prod_vector"]
        Es = np.linspace(-5, 5,10000)

        fig = plt.figure(figsize = (col,col), layout = "constrained")
        figs = fig.subfigures(2, 2,width_ratios=[1, 1], height_ratios=[1, 1])
        axs = [[subfig.subplots() for subfig in subfigs] for subfigs in figs]

        axs[0][0].plot(Es - self.system_th.muR, self.system_nth.occupf_L(Es), label = "Nonthermal", color = nonthermal, zorder = 2)
        axs[0][0].plot(Es- self.system_th.muR, self.system_th.occupf_L(Es), "--",label = "Thermal", color = hot, zorder = 3)
        axs[0][0].plot(Es- self.system_th.muR, self.system_th.occupf_R(Es), label = "Cold", color = cold, zorder = 1)
        figs[0][0].set_facecolor("0.85")
        axs[0][0].set_xlabel(r"$\varepsilon$ [$k_\mathrm{B} T\,$]")
        axs[0][0].set_ylabel("Occupation prob.")

        if dip_peak:
            axs[0][0].annotate("(i)", (0.75,0.85), xycoords = "axes fraction")
        else:
            axs[0][0].annotate("(a)", (0.75,0.85), xycoords = "axes fraction")
        axs[0][0].set_yticks([0.00,0.50,1.00],["0.00","0.50","1.00"])

        if make_eff:
            self.eff_plot(system = self.system_nth, color = nonthermal, axs = axs[0][1], label = "nonthermal", filename=filenames[1], C_avg=C_avg)
            self.eff_plot(axs[0][1], color = hot, label = "thermal",system = self.system_th,  filename = filenames[0], striped = True)
            
            if dip_peak:
                axs[0][1].annotate("(j)", (0.75,0.85), xycoords = "axes fraction")
            else:
                axs[0][1].annotate("(b)", (0.75,0.85), xycoords = "axes fraction")
            axs[0][1].locator_params(axis = "x", nbins = 3)

            axs[0][1].set_yticks([0.50, 0.75, 1.00],["0.50", "0.75", "1.00"])
            axs[0][1].set_box_aspect(1)

        if make_noise:
            self.noise_plot(system=self.system_nth, color = nonthermal, axs=axs[1][0], label = "nonthermal", filename=filenames[3], C_noise = C_noise)
            self.noise_plot(system=self.system_th, color = hot, axs=axs[1][0], label = "thermal", filename=filenames[2], striped = True)
            
            if dip_peak:
                axs[1][0].annotate("(k)", (0.05,0.85), xycoords = "axes fraction")
            else:
                axs[1][0].annotate("(c)", (0.05,0.85), xycoords = "axes fraction")
            axs[1][0].locator_params(axis = "x", nbins = 3)
            #axs[1][0].set_yticks([0.00, 0.02,0.04,0.06],["0.00", "0.02","0.04","0.06"])
            if dip_peak:
                axs[1][0].set_yticks([0.00, 2.00,4.00],["0.00", "2.00","4.00"])
            else:
                axs[1][0].set_yticks([0.00, 2.00,4.00, 6.00],["0.00", "2.00","4.00","6.00"])
            axs[1][0].sharex(axs[0][1])
            axs[1][0].set_box_aspect(1)
  
        if make_product:
            self.product_plot(system=self.system_nth, color = nonthermal, axs=axs[1][1], label = "nonthermal", filename=filenames[5], C_prod=C_prod)
            self.product_plot(system=self.system_th, color = hot, axs=axs[1][1], label = "thermal", filename=filenames[4], striped = True)
            
            if dip_peak:
                axs[1][1].annotate("(l)", (0.05,0.85), xycoords = "axes fraction")
            else:
                axs[1][1].annotate("(d)", (0.05,0.85), xycoords = "axes fraction")
            axs[1][1].locator_params(axis = "x", nbins = 3)
            axs[1][1].locator_params(axis = "y", nbins = 5)
            axs[1][1].set_yticks([2.5, 7.5, 12.5],["2.5", "7.5", "12.5"])
            axs[1][1].sharex(axs[0][1])
            axs[1][1].set_box_aspect(1)
  
        lines, labels = axs[0][0].get_legend_handles_labels()
        if dip_peak:
            labels = ["Dip-peak", "Probe", "Cold"]
        else:
            labels = ["Mixed", "Probe", "Cold"]
        fig.legend(lines, labels, loc= "outside lower center", ncols = 3)#, bbox_to_anchor = (0.55,-0.1))
        file.close()
        return fig

    def make_example_figure(self, example_file, make_eff = False, make_noise = False, make_product = False, striped = False):
        jmax, transf_max, _ = self.system_nth.constrained_current_max()
        self.system_nth.adjust_limits(0.5)
        Es = np.linspace(-5, 5,10000)
        
        fig = plt.figure(figsize = (col,col), layout = "constrained")
        figs = fig.subfigures(2, 1)
        axs = [subfig.subplots() for subfig in figs]
        
        axs[0].plot(Es- self.system_th.muR, self.system_nth.occupf_L(Es), label = "Nonthermal", color = nonthermal, zorder = 3)
        axs[0].plot(Es- self.system_th.muR, self.system_th.occupf_R(Es), label = "Cold", color = cold, zorder = 2)
        axs[0].vlines(self.system_nth.muR- self.system_th.muR, 0,1, colors = cold, alpha = 0.7, linestyles = "dashed")
        axs[0].annotate(r"$\mu$", (self.system_nth.muR- self.system_th.muR, 0.7), textcoords = "offset points", xytext = (2,0), color = cold, alpha =0.7)
        axs[0].fill_between(Es- self.system_th.muR, 1, where = transf_max(Es) == 0, facecolor = nocool, zorder = 1, label = "No cooling")
        axs[0].annotate("(a)", (0.75,0.85), xycoords = "axes fraction")    
        axs[0].set_xlabel(r"$\varepsilon$ [$k_\mathrm{B} T\,$]")
        axs[0].set_ylabel("Occupation prob.")
        file = np.load(example_file)
                
        if make_eff:
            C_avg = file["C_avg"]
            transf_avg = self.system_nth._transmission_avg(float(C_avg), self.system_nth.coeff_con, self.system_nth.coeff_avg)
            self.system_nth.transf = transf_avg
            axs[1].plot(Es- self.system_th.muR, transf_avg(Es), color = "#DB70FF", label = "Best eff.")

        if make_noise:
            C_noise = file["C_noise"]
            transf_noise = self.system_nth._transmission_noise(float(C_noise))
            self.system_nth.transf = transf_noise
            if striped:
                axs[1].plot(Es- self.system_th.muR, transf_noise(Es),"--",color = "#58008F", label = "Best precis.", zorder = 2)
            else:
                axs[1].plot(Es- self.system_th.muR, transf_noise(Es),color = "#58008F", label = "Best precis.", zorder = 2)
        
        if make_product:
            C_prod = file["prod_vector"]
            transf_prod = self.system_nth._transmission_product(C_prod)
            self.system_nth.transf = transf_prod
            axs[1].plot(Es- self.system_th.muR, transf_prod(Es),color = "#843FAF", label = "Best trade.", zorder = 1)

        axs[1].annotate("(b)", (0.75,0.85), xycoords = "axes fraction")
        axs[1].set_xlabel(r"$\varepsilon$ [$k_\mathrm{B} T\,$]")
        axs[1].set_ylabel("Transmission func.")
        
        lines, labels = axs[0].get_legend_handles_labels()
        line1 = Line2D([],[])
        line1.update_from(lines[0])
        line2 = Line2D([],[])
        line2.update_from(lines[1])
        line3 = PolyCollection([])
        line3.update_from(lines[2])

        fig.legend([line1, line2,line3], labels, loc = "outside right upper", prop = {"size":10}, frameon = False, labelspacing = 1)
        
        lines, labels = axs[1].get_legend_handles_labels()
        fig.legend(lines, labels, loc = "outside center right", prop = {"size":10}, frameon = False, labelspacing = 1, bbox_to_anchor = (1,0.4))

        file.close()

        return fig 

    def make_big_example_figure(self, example_files, dip_peak = False):
        jmax, transf_max, _ = self.system_nth.constrained_current_max()
        self.system_nth.adjust_limits(0.5)
        Es = np.linspace(-1, 1,10000)
        
        fig = plt.figure(figsize = (col,col), layout = "constrained")
        figs = fig.subfigures(2, 2)
        axs = [[subfig.subplots() for subfig in subfigs] for subfigs in figs]
        figs[0][0].set_facecolor("0.85")
        axs[0][0].plot(Es- self.system_th.muR, self.system_nth.occupf_L(Es), label = "Nonthermal", color = nonthermal, zorder = 3)
        axs[0][0].plot(Es- self.system_th.muR, self.system_th.occupf_L(Es), "--",label = "Nonthermal", color = hot, zorder = 4)
        axs[0][0].plot(Es- self.system_th.muR, self.system_th.occupf_R(Es), label = "Cold", color = cold, zorder = 2)
        axs[0][0].vlines(self.system_nth.muR- self.system_th.muR, 0,1, colors = cold, alpha = 0.7, linestyles = "dashed")
        axs[0][0].annotate(r"$\mu$", (self.system_nth.muR- self.system_th.muR, 0.7), textcoords = "offset points", xytext = (2,0), color = cold, alpha =0.7)
        axs[0][0].fill_between(Es- self.system_th.muR, 1, where = transf_max(Es) == 0, facecolor = nocool, zorder = 1, label = "No cooling")
        axs[0][0].set_xlabel(r"$\varepsilon$ [$k_\mathrm{B} T\,$]")
        axs[0][0].set_ylabel("Occupation prob.")

        if dip_peak:
            axs[0][0].annotate("(m)", (0.03,0.85), xycoords = "axes fraction")    
        else:
            axs[0][0].annotate("(e)", (0.03,0.85), xycoords = "axes fraction")    

        color_one = [th_one, nth_one]
        color_two = [th_two, nth_two]
        color_three = [th_three, nth_three]
        systems = [self.system_th, self.system_nth]
        zorders = [2,1]
        linestyles = ["--", "-"]
        for i, example_file in enumerate(example_files):
            file = np.load(example_file)    
            
            C_avg = file["C_avg"]
            transf_avg = systems[i]._transmission_avg(float(C_avg), systems[i].coeff_con, systems[i].coeff_avg)
            systems[i].set_transmission_avg_opt(C_avg)
            axs[0][1].plot(Es, transf_avg(Es), color = color_one[i], label = "Best eff.", zorder = zorders[i], linestyle = linestyles[i])
            axs[0][1].scatter(-0.95, 0.8, marker = "P", color = color_one[i])
            axs[0][1].set_xlabel(r"$\varepsilon$ [$k_\mathrm{B} T\,$]")
            axs[0][1].set_ylabel("Transmission func.")
            
            if dip_peak:
                axs[0][1].annotate("(n)", (0.03,0.85), xycoords = "axes fraction")    
            else:
                axs[0][1].annotate("(f)", (0.03,0.85), xycoords = "axes fraction")    
            
            C_noise = file["C_noise"]
            transf_noise = systems[i]._transmission_noise(float(C_noise))
            systems[i].set_transmission_noise_opt(C_noise)
            
            axs[1][0].plot(Es- self.system_th.muR, transf_noise(Es),color = color_two[i], label = "Best precis.", zorder = zorders[i], linestyle = linestyles[i])
            axs[1][0].scatter(-0.95, 0.8, marker = "X", color = color_two[i])
            axs[1][0].set_xlabel(r"$\varepsilon$ [$k_\mathrm{B} T\,$]")
            axs[1][0].set_ylabel("Transmission func.")    
            if dip_peak:
                axs[1][0].annotate("(o)", (0.03,0.85), xycoords = "axes fraction")    
            else:
                axs[1][0].annotate("(g)", (0.03,0.85), xycoords = "axes fraction")

            C_prod = file["prod_vector"]
            transf_prod = systems[i]._transmission_product(C_prod)
            systems[i].set_ready_transmission_product(C_prod)
            axs[1][1].plot(Es- self.system_th.muR, transf_prod(Es),color = color_three[i], label = "Best trade.", zorder = zorders[i], linestyle = linestyles[i])
            axs[1][1].scatter(-0.95, 0.8, marker = "*", color = color_three[i])
            axs[1][1].set_xlabel(r"$\varepsilon$ [$k_\mathrm{B} T\,$]")
            axs[1][1].set_ylabel("Transmission func.")

            if dip_peak:
                axs[1][1].annotate("(p)", (0.03,0.85), xycoords = "axes fraction")  
            else:
                axs[1][1].annotate("(h)", (0.03,0.85), xycoords = "axes fraction") 
            
            file.close()

        return fig 

    def make_crossing_figure(self, filename, system:two_terminals):
        occupf_L = system.occupf_L
        occupf_R = system.occupf_R
        file = np.load(filename)
        C_avg = file["C_avg"]
        
        mosaic = """
            AABB
            .CC.
        """
        y = system.coeff_avg
        x = system.coeff_con
        factor_two = lambda E: y(E)-C_avg*x(E)
        factor_one = lambda E: occupf_L(E)-occupf_R(E)
        
        Es = np.linspace(system.E_low, system.E_high, 10000)
        fig = plt.figure(figsize = (col,0.8*col), layout = "constrained")
        ax_dict = fig.subplot_mosaic(mosaic)
        axs = [ax for ax in ax_dict.values()]
        
        axs[0].plot(Es - system.muR, occupf_L(Es), label = "Left", color = nonthermal)
        axs[0].plot(Es - system.muR, occupf_R(Es), label = "Right", color = cold)
        #axs[0].plot(Es-system.muR, system._transmission_avg(C_arr[10], x, y)(Es), label = "transf")
        axs[0].fill_between(Es- system.muR, 1, where = factor_one(Es) > 0, facecolor = negative, alpha = 0.5, zorder = 1)
        axs[0].annotate(r"$f(\varepsilon)$", (0.2, 0.55), color = cold)
        axs[0].annotate(r"$g(\varepsilon)$", (0.1, 0.2), color = nonthermal)
        axs[0].set_xlabel(r"$\varepsilon$ [$k_\mathrm{B} T\,$]")
        axs[0].set_ylabel("Occupation prob.")
        axs[0].annotate("(a)", (0.75,0.85), xycoords = "axes fraction")

        #### NOTE: THE SIGNS ARE FLIPPED BECAUSE THE COLD BATH IS PLACED TO LEFT IN THE PAPER, BUT THIS IS CANCELLED OUT BY THE EFFECTIVE SIGN CHANGE IN THE DIFFERENCE OF THE DISTRIBUTION FUNCTIONS
        axs[1].plot(Es-system.muR, -y(Es), label = "Left", color = nonthermal)
        axs[1].plot(Es-system.muR, -C_avg*x(Es), label = "Right", color = cold)
        axs[1].fill_between(Es- system.muR, 5, -5, where = factor_two(Es) < 0, facecolor = negative, alpha = 0.5, zorder = 1)        
        axs[1].set_xlabel(r"$\varepsilon$ [$k_\mathrm{B} T\,$]")
        axs[1].set_ylabel("Coeff. spectrum")
        axs[1].annotate("(b)", (0.05,0.85), xycoords = "axes fraction")
        axs[1].annotate(r"$y(\varepsilon)$", (0.55, -1.5), color = nonthermal)
        axs[1].annotate(r"$\lambda x(\varepsilon)$", (0.3, 3.5), color = cold)
        
        axs[2].fill_between(Es- system.muR, 0, 1, where = factor_two(Es) < 0, facecolor = negative, zorder = 1, alpha = 0.5)
        axs[2].fill_between(Es- system.muR, 0, 1, where = factor_one(Es) > 0, facecolor = negative, zorder = 1, alpha = 0.5)
        axs[2].annotate("(c)", (0.75,0.85), xycoords = "axes fraction")
        axs[2].set_xlabel(r"$\varepsilon$ [$k_\mathrm{B} T\,$]")
        axs[2].set_ylabel("Condition sign")

        file.close()

        return fig
        
    def make_all_crossing_figure(self, filenames, system:two_terminals, example_file = None):
        mosaic = """
            AB
            C.
        """
        fig = plt.figure(figsize = (col,0.8*col), layout = "constrained")
        ax_dict = fig.subplot_mosaic(mosaic)
        axs = [ax for ax in ax_dict.values()]
        
        JR_arr, data_arr, C_eff_arr = self.get_eff_data(filename=filenames[0])
        JR_arr, data_arr, C_noise_arr = self.get_noise_data(filename=filenames[1])
        JR_arr, data_arr, C_prod_arr, err_arr = self.get_product_data(filename=filenames[2])
        
        if example_file:
            file = np.load(example_file)
            C_eff = file["C_avg"]
            C_noise = file["C_noise"]
            C_prod = file["prod_vector"]

        else:
            C_pick = int(len(C_eff_arr)/2)
            C_eff = C_eff_arr[C_pick]
            C_noise = C_noise_arr[C_pick]
            C_prod = C_prod_arr[C_pick]
        
        
        system.adjust_limits(0.2)
        Es = np.linspace(system.E_low, system.E_high, 10000)
        cond_eff = system._avg_condition(C_eff)
        cond_noise = system._noise_condition(C_noise)
        cond_prod = system._product_condition(C_prod.flatten())

        max_cool = system.coeff_con(Es)*(system.occupf_L(Es)- system.occupf_R(Es))
        max_cool = max_cool/np.max(np.abs(max_cool))
        
        axs[0].plot(Es, cond_eff(Es)/np.max(np.abs(cond_eff(Es))), color = nonthermal)        
        axs[0].grid()
        axs[0].set_yticks([0],[0])
        axs[0].annotate("(a)", (0.75,0.85), xycoords = "axes fraction")
        axs[0].set_xlabel(r"$\varepsilon$ [$k_\mathrm{B} T\,$]")
        axs[0].plot(Es, max_cool, color = nonthermal, alpha = 0.5)
        
        axs[1].plot(Es, cond_noise(Es)/np.max(np.abs(cond_noise(Es))), color = nonthermal)
        axs[1].set_yticks([0],[0])
        axs[1].grid()
        axs[1].annotate("(b)", (0.75,0.85), xycoords = "axes fraction")
        axs[1].plot(Es, max_cool, color = nonthermal, alpha = 0.5)
        axs[1].set_xlabel(r"$\varepsilon$ [$k_\mathrm{B} T\,$]")
        
        axs[2].plot(Es, cond_prod(Es)/np.max(np.abs(cond_prod(Es))), color = nonthermal)
        axs[2].set_xlabel(r"$\varepsilon$ [$k_\mathrm{B} T\,$]")
        axs[2].set_yticks([0],[0])
        axs[2].plot(Es, max_cool, color = nonthermal, alpha = 0.5)
        axs[2].grid()
        axs[2].annotate("(c)", (0.75,0.85), xycoords = "axes fraction")

        file.close()

        return fig

    def make_char_eff_figure(self, filename, system:two_terminals):
        occupf_L = system.occupf_L
        occupf_R = system.occupf_R
        occupdiff = lambda E: occupf_L(E) - occupf_R(E)

        file = np.load(filename)
        C = file["C_avg"]

        y = system.coeff_avg
        x = system.coeff_con
        Es = np.linspace(system.E_low, system.E_high, 10000)
        fig, axs = plt.subplots(1, 1, figsize = (col, 150*pt), layout = "constrained")
        axs = [axs]
        axs[0].plot(Es-system.muR, x(Es)/y(Es)*np.heaviside(x(Es)*occupdiff(Es),0), label = "frac", color = nonthermal, zorder = 3)
        axs[0].fill_between(Es- system.muR, 0, 1, where = x(Es)*occupdiff(Es) < 0, facecolor = nocool, zorder = 2)
        axs[0].fill_between(Es- system.muR, 0, 1, where = x(Es)/y(Es)  < 1/C, facecolor = negative, zorder = 1, alpha = 0.5)
        axs[0].hlines(1/C, Es[0], Es[-1], colors = "black", zorder = 4)
        axs[0].annotate(r"$\eta_{\mathrm{min}}^{\mathrm{char}}$", (0.5, 1/C*1.1))
        axs[0].annotate(r"$\eta^{\mathrm{char}}$", (0.29, 0.75), color = nonthermal)
        axs[0].annotate("No\n  cooling", (-0.36, 0.75), color = nocooldark)
        axs[0].set_xlabel(r"$\varepsilon$ [$k_\mathrm{B} T\,$]")
        axs[0].set_ylabel("Characteristic efficiency")
        file.close()
        return fig

    def make_local_eff_figure(self, filename, system:two_terminals, dip_peak = False):
        JR_arr, eff_arr, C_arr = self.get_eff_data(filename)
        fig, axs = plt.subplots(1,2,figsize = (col,0.7*col), layout = "constrained")
       
        axs[0].plot(JR_arr, eff_arr, label = "Total efficiency", color = "#DB70FF")
        axs[0].plot(JR_arr, 1/C_arr, label = "Characteristic efficiency", color = "#58008F")

        if dip_peak:
            axs[0].annotate("(c)", (0.75,0.85), xycoords = "axes fraction")
        else:
            axs[0].annotate("(a)", (0.75,0.85), xycoords = "axes fraction")
        axs[0].set_xlabel(r"$I^{\Sigma,\mathrm{L}}$ [$k_\mathrm{B}^2 T/h$]")
        axs[0].set_ylabel("Efficiency")

        axs[1].plot(JR_arr, eff_arr - 1/C_arr, color = "#843FAF")
        axs[1].set_xlabel(r"$I^{\Sigma,\mathrm{L}}$ [$k_\mathrm{B}^2 T/h$]")
        axs[1].set_ylabel(r"$\Delta \eta$")

        if dip_peak:
            axs[1].annotate("(d)", (0.05,0.85), xycoords = "axes fraction")
        else:
            axs[1].annotate("(b)", (0.05,0.85), xycoords = "axes fraction")
        
        lines, labels = axs[0].get_legend_handles_labels()
        
        fig.legend(lines, labels, loc = "outside lower left")
        return fig

    def eff_plot(self, axs, color, label,system:two_terminals=None,filename = None, C_avg = None, striped = False):
        if self.verbose:
            print("Making efficiency plot for ", label)
        if C_avg != None:
            system.set_transmission_avg_opt(C_avg)
            ex_eff = system.get_efficiency()
            ex_JR = system.current_integral(system.coeff_con, cond_in = system._avg_condition(C_avg))
            print("Example JR in eff plot: ", ex_JR)
            axs.scatter(ex_JR, ex_eff, marker = "P", color = nth_one,zorder = 2)

        JR_arr, eff_arr, C_arr = self.get_eff_data(filename)
        if striped:
            axs.plot(JR_arr, eff_arr, "--",label=label, color = color, zorder = 1)        
        else:
            axs.plot(JR_arr, eff_arr, label=label, color = color, zorder = 1)        
        axs.set_xlabel(r"$I^{\Sigma,\mathrm{L}}$ [$k_\mathrm{B}^2 T/h$]")
        axs.set_ylabel(r"$\eta$")

    def noise_plot(self,  axs, color, label,system:two_terminals=None,filename = None, C_noise = None, striped = False):
        if self.verbose:
            print("Making noise plot for ", label)
        if C_noise != None:
            system.set_transmission_noise_opt(C_noise)
            ex_noise = system.noise_cont(system.coeff_noise)
            ex_JR = system.current_integral(system.coeff_con)
            axs.scatter(ex_JR, ex_noise/ex_JR, marker = "X", color = nth_two, zorder = 2)
        JR_arr, noise_arr, C_arr = self.get_noise_data(filename)
        if striped:
            axs.plot(JR_arr, (noise_arr/(JR_arr)),"--", label = label, color = color, zorder = 1)
        else:
            axs.plot(JR_arr, (noise_arr/(JR_arr)), label = label, color = color, zorder = 1)
        axs.set_xlabel(r"$I^{\Sigma,\mathrm{L}}$ [$k_\mathrm{B}^2 T/h$]")
        axs.set_ylabel(r"NSR")

    def product_plot(self,  axs, color, label,system:two_terminals=None,filename = None, C_prod = [None], striped = False):
        if self.verbose:
            print("Making product plot for ", label)
        JR_arr, eff_arr, C_arr, err_arr = self.get_product_data(filename)
        keep_index = np.argwhere(np.abs(err_arr) < 1e-7)
        JR_arr = JR_arr[keep_index]
        eff_arr = eff_arr[keep_index]
        eff_arr = eff_arr/JR_arr
        if not None in C_prod:
            system.set_ready_transmission_product(C_prod)
            ex_prod = system.current_integral(system.coeff_avg)*system.noise_cont(system.coeff_noise)
            ex_JR = system.current_integral(system.coeff_con)
            axs.scatter(ex_JR, ex_prod/ex_JR**2, marker = "*", color = nth_three, zorder = 2)
        
        if striped:
            axs.plot(JR_arr, eff_arr, "--",label=label, color = color, zorder = 1)
        else:
            axs.plot(JR_arr, eff_arr, label=label, color = color, zorder = 1)
        axs.set_xlabel(r"$I^{\Sigma,\mathrm{L}}$ [$k_\mathrm{B}^2 T/h$]")
        axs.set_ylabel(r"$A^{(\Sigma, \mathrm{L}),(\Sigma, \mathrm{R})}$")

    def get_eff_data(self,filename):
        file = np.load(filename)
        JR_arr = file["JR_arr"]
        eff_arr = file["eff_arr"]
        C_arr = file["C_arr"]
        file.close()
        return JR_arr, eff_arr, C_arr

    def get_noise_data(self, filename):
        file = np.load(filename)
        JR_arr = file["JR_arr"]
        noise_arr = file["noise_arr"]
        C_arr = file["C_arr"]
        file.close()
        return JR_arr, noise_arr, C_arr

    def get_product_data(self, filename):
        file = np.load(filename)
        JR_arr = file["JR_arr"]
        eff_arr = file["eff_arr"]
        C_arr = file["C_arr"]
        err_arr = file["err_arr"]
        file.close()
        return JR_arr, eff_arr, C_arr, err_arr

class SecondaryPlot:
    def __init__(self, system_th:two_terminals, system_nth:two_terminals, secondary_prop, verbose = False):
        self.system_th = system_th
        self.system_nth = system_nth
        self.verbose = verbose
        self.secondary_prop = secondary_prop

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

    def make_eff_figure(self, filenames):
        mosaic = """
            AB
            C.
        """
        fig = plt.figure(figsize = (col,col), layout = "constrained")
        ax_dict = fig.subplot_mosaic(mosaic)
        axs = [ax for ax in ax_dict.values()]
        
        idx_list_th = [8,58,83]
        idx_list_nth = [8,55,80]

        ex_Es = np.linspace(-1, 11, 1000000)
        self.eff_plot(axs, color = hot, label = "thermal", color2 = th_one, ex_Es= ex_Es,system = self.system_th,  filename = filenames[0], style = "--", zorder = 2, idx_list = idx_list_th)
        self.eff_plot(axs,  color = nonthermal, color2 = nth_one, label = "nonthermal", ex_Es= ex_Es,system = self.system_nth, filename=filenames[1], zorder = 1, idx_list = idx_list_nth)

        lines, labels = axs[0].get_legend_handles_labels() 
        fig.legend(lines, labels, loc= "center", ncols = 1,bbox_to_anchor=(0.8, 0.3))          

    def eff_plot(self, axs, color,color2,label,ex_Es, system=None,filename = None, style = '-', zorder = 1, idx_list = [100]):
        if self.verbose:
            print("Making efficiency plot for ", label)
        JR_arr, avg_arr, noise_arr, C_arr, s_arr, err_arr = self.get_eff_data(filename)
        eff_arr = JR_arr/avg_arr
        keep_index = np.argwhere(np.abs(err_arr) < 1e-7)
        JR_arr = JR_arr[keep_index]
        eff_arr = eff_arr[keep_index]
        s_arr = s_arr[keep_index]
        C_arr = C_arr[keep_index]
        
        axs[0].plot(JR_arr, eff_arr, style,label=label, color = color, zorder = zorder)
        axs[0].annotate("(a)", (0.75,0.85), xycoords = "axes fraction")
        axs[0].locator_params(axis = "x", nbins = 3)
        axs[0].set_xlabel(r"$I^{\Sigma,\mathrm{L}}$ [$k_\mathrm{B}^2 T/h$]")
        axs[0].set_ylabel(r"$\eta$ [$I^{\Sigma,\mathrm{L}}/I^{\Sigma,R}$]")
        axs[0].set_ylabel(r"$\eta$")


        axs[1].plot(s_arr, eff_arr, style,label=label, color = color, zorder = zorder)
        axs[1].annotate("(b)", (0.75,0.85), xycoords = "axes fraction")
        axs[1].set_xlabel(r"$\mu$ [$k_\mathrm{B} T\,$]")
        axs[1].set_ylabel(r"$\eta$")
        axs[1].locator_params(axis = "x", nbins = 3)

        axs[2].plot(s_arr, JR_arr, style,label=label, color = color, zorder = zorder)
        axs[2].locator_params(axis = "x", nbins = 3)
        axs[2].annotate("(c)", (0.05,0.85), xycoords = "axes fraction")
        axs[2].set_xlabel(r"$\mu$ [$k_\mathrm{B} T\,$]")
        axs[2].set_ylabel(r"$I^{\Sigma,\mathrm{L}}$ [$k_\mathrm{B}^2 T/h$]")

        for i,idx in enumerate(idx_list):
            self.updater(s_arr[idx], self.system_nth, 30)
            self.updater(s_arr[idx], self.system_th, 30)
            C = C_arr[idx]
            if label == "nonthermal":        
                axs[1].scatter(s_arr[idx], eff_arr[idx], marker = "P", color = color2, zorder = zorder +2)

    def make_transition_figure(self, filenames):  
        idx_list_th = [10,73,98]
        idx_list_nth = [8,55,80]
        ex_Es = np.linspace(-2,4, 1000000)
        fig = plt.figure(figsize=(col,1.2*col), layout = "constrained")
        gs = fig.add_gridspec(3,3, wspace = 0)
        axs = gs.subplots(sharex=True, sharey=True)

        def plotter(row, idx_list, system:two_terminals, filename, color2, label = "nonthermal", style = "-", zorder = 1):    
            JR_arr, avg_arr, noise_arr, C_arr, s_arr, err_arr = self.get_eff_data(filename)
            eff_arr = JR_arr/avg_arr
            keep_index = np.argwhere(np.abs(err_arr) < 1e-7)
            JR_arr = JR_arr[keep_index]
            eff_arr = eff_arr[keep_index]
            s_arr = s_arr[keep_index]
            C_arr = C_arr[keep_index]
            for i,idx in enumerate(idx_list):
                ex_transf = system._transmission_avg(C_arr[idx])
                self.updater(s_arr[idx], self.system_nth, 30)
                self.updater(s_arr[idx], self.system_th, 30)
                C = C_arr[idx]
                d_right = system.dSR_dmuR
                d_left = system.dSL_dmuR
                dL_integ = system.general_integral(d_left(system._transmission_avg(C)), system._avg_condition(C))
                dR_integ = system.general_integral(d_right(system._transmission_avg(C)), system._avg_condition(C))
                print("Example at mu = ", s_arr[idx])
                print("Example at JR: ", JR_arr[idx])
                print("Calc JR: ", system.current_integral(system.coeff_con, cond_in=system._avg_condition(C_arr[idx])))
                print("Calc fraction: ", dL_integ/dR_integ)
                print("Diff: ", dL_integ/dR_integ - C_arr[idx])
                C = C_arr[idx]
                if label == "nonthermal":        
                    axs[0][i].plot(ex_Es, self.system_nth.occupf_L(ex_Es), label = "Nonthermal", color = nonthermal, zorder = 2, alpha = 1)
                    axs[0][i].plot(ex_Es, self.system_th.occupf_L(ex_Es), "--",label = "Thermal", color = hot, zorder = 3)
                    axs[0][i].plot(ex_Es, self.system_th.occupf_R(ex_Es), label = "Cold", color = cold, zorder = 1, alpha = 1)
                    axs[0][i].tick_params(direction="in")
                axs[row][i].plot(ex_Es, ex_transf(ex_Es), style,color = color2, zorder = zorder+3)
                axs[row][i].tick_params(direction="in")
                
            axs[0][0].set_ylabel("Occupation prob.")
            axs[1][0].set_ylabel("Transmission func.")
            axs[2][0].set_ylabel("Transmission func.")
            axs[2][1].set_xlabel(r"$\varepsilon$ [$k_\mathrm{B} T\,$]")

        fig.get_layout_engine().set(w_pad=0 / 72, wspace=0)

        plotter(1, idx_list_th, self.system_th, filenames[0], th_one, label="thermal", style = "--", zorder = 2)
        plotter(2, idx_list_nth, self.system_nth, filenames[1], nth_one, label="nonthermal", style = "-")
        return fig

    def make_example_figure(self, filename):
        JR_arr, avg_arr, noise_arr, C_arr, err_arr, d_avg_arr, d_con_arr, s_arr = self.get_example(filename)
        eff_arr = JR_arr/avg_arr
        deriv_arr = d_avg_arr/d_con_arr

        mu_cross = s_arr[np.argwhere(np.abs(deriv_arr-C_arr) == np.min(np.abs(deriv_arr-C_arr))).flatten()][0]
        mu_max = s_arr[np.argwhere(eff_arr == np.max(eff_arr)).flatten()][0]
        
        fig = plt.figure(figsize = (col,col), layout = "constrained")
        mosaic = """
                AB
                CD
                """
        ax_dict = fig.subplot_mosaic(mosaic)
        axs = [ax for ax in ax_dict.values()]
        axs[0].plot(s_arr, eff_arr, color = nonthermal)
        axs[0].vlines(mu_max, np.min(eff_arr)*np.ones_like(mu_max), np.max(eff_arr)*np.ones_like(mu_max)+0.005, colors = "black")
        axs[0].set_xlabel(r"$\mu$ [$k_\mathrm{B} T\,$]")
        axs[0].set_ylabel(r"$\eta$")
        
        axs[0].annotate("max\n"+rf"$\mu = {mu_max:.2f}$", (mu_max*1.05, 0.66))
        axs[0].annotate("(a)", (0.8,0.08), xycoords = "axes fraction")

        axs[1].plot(s_arr, 1/C_arr, label = r"$1/\lambda$", color = "#DB70FF")
        axs[1].plot(s_arr, 1/deriv_arr, "--",label = r"$\frac{(\partial I^{\Sigma, L} / \partial \mu)_{\mathcal{D}(\varepsilon)}}{(\partial I^{\Sigma, R} / \partial \mu)_{\mathcal{D}(\varepsilon)}}$", color = "#58008F")
        axs[1].vlines(mu_cross, np.min(1/C_arr)*np.ones_like(mu_cross), np.max(1/C_arr)*np.ones_like(mu_cross), colors = "black")
        axs[1].set_xlabel(r"$\mu$ [$k_\mathrm{B} T\,$]")
        axs[1].annotate("crossing\n"+rf"$\mu = {mu_cross:.2f}$", (mu_cross*1.05, 0.50))
        axs[1].annotate("(b)", (0.8,0.08), xycoords = "axes fraction")

        
        mu_cross = s_arr[find_peaks(-np.abs(deriv_arr-C_arr))[0]]
        mu_max = s_arr[find_peaks(eff_arr)[0]]
        mu_min = s_arr[find_peaks(-eff_arr)[0]]

        axs[2].plot(s_arr, eff_arr, color = nonthermal)
        axs[2].vlines(mu_max, np.min(eff_arr)*np.ones_like(mu_max), np.max(eff_arr)*np.ones_like(mu_max)+0.005, colors = "black")
        axs[2].vlines(mu_min, np.min(eff_arr)*np.ones_like(mu_min), np.max(eff_arr)*np.ones_like(mu_min)+0.005, colors = "black")
        axs[2].set_xlabel(r"$\mu$ [$k_\mathrm{B} T\,$]")
        axs[2].set_ylabel(r"$\eta$")
        axs[2].set_xlim([1,2])
        axs[2].set_ylim([0.8,0.9])
        axs[2].annotate("(c)", (0.8,0.08), xycoords = "axes fraction")

        axs[3].plot(s_arr, 1/C_arr, label = r"$1/\lambda$", color = "#DB70FF")
        axs[3].plot(s_arr, 1/deriv_arr, "--",label = r"$\frac{(\partial I^{\Sigma, L} / \partial \mu)_{\mathcal{D}(\varepsilon)}}{(\partial I^{\Sigma, R} / \partial \mu)_{\mathcal{D}(\varepsilon)}}$", color = "#58008F")
        axs[3].vlines(mu_cross, np.min(1/C_arr)*np.ones_like(mu_cross), np.max(1/C_arr)*np.ones_like(mu_cross), colors = "black")
        axs[3].set_xlabel(r"$\mu$ [$k_\mathrm{B} T\,$]")
        axs[3].set_xlim([1,2])
        axs[3].set_ylim([0.7,0.85])
        axs[3].annotate("(d)", (0.8,0.08), xycoords = "axes fraction")

        lines, labels = axs[1].get_legend_handles_labels() 
        fig.legend(lines, labels, loc= "outside lower right", ncols = 2) 
        return fig

    def get_eff_data(self,filename = None):
        file = np.load(filename)
        JR_arr = file["JR_arr"]
        avg_arr = file["avg_arr"]
        noise_arr = file["noise_arr"]
        C_arr = file["C_arr"]
        s_arr = file["s_arr"]
        err_arr = file["err_arr"]
        file.close()
        return JR_arr, avg_arr, noise_arr, C_arr, s_arr, err_arr

    def get_noise_data(self, filename = None):
        file = np.load(filename)
        JR_arr = file["JR_arr"]
        noise_arr = file["noise_arr"]
        avg_arr = file["avg_arr"]
        C_arr = file["C_arr"]
        s_arr = file["s_arr"]
        err_arr = file["err_arr"]
        file.close()
        return JR_arr, avg_arr, noise_arr, C_arr, s_arr, err_arr

    def get_product_data(self, filename = None):
        file = np.load(filename)
        JR_arr = file["JR_arr"]
        avg_arr = file["avg_arr"]
        C_arr = file["C_arr"]
        noise_arr = file["noise_arr"]
        s_arr = file["s_arr"]
        err_arr = file["err_arr"]
        file.close()
        return JR_arr, avg_arr, noise_arr, C_arr, s_arr, err_arr

    def get_example(self, filename):
        file = np.load(filename)
        JR_arr = file["JR_arr"]
        avg_arr = file["avg_arr"]
        C_arr = file["C_arr"]
        noise_arr = file["noise_arr"]
        s_arr = file["s_arr"]
        err_arr = file["err_arr"]
        d_avg_arr = file["d_avg_arr"]
        d_con_arr = file["d_con_arr"]
        file.close()
        return JR_arr, avg_arr, noise_arr, C_arr, err_arr, d_avg_arr, d_con_arr, s_arr
    
class RealTransfPlot():
    def __init__(self, width):
        self.width = width
    def lorentzian(self, gamma, position):
        return lambda E: gamma**2 / ((E-position)**2 + gamma**2)

    def real_transf(self, gammas, positions, Es = [None]):
        lorentz_list = []
        for gamma, position in zip(gammas, positions):
            lorentz_list.append(self.lorentzian(gamma, position))
        
        if any(E is None for E in Es):
            transf = lambda E: min(sum(lorentz(E) for lorentz in lorentz_list),1)#if sum(lorentz(E) for lorentz in lorentz_list) <= 1 else 1
        else:
            transf = sum(lorentz(Es) for lorentz in lorentz_list)
            transf = np.clip(transf, 0,1)

        return transf    

    def eff_plot(self, axs, color, label,linestyle = "-", system=None,filename = None, alpha = 1, C_pick = None):
        file = np.load(filename)
        JR_arr = file["JR_arr"]
        eff_arr = file["eff_arr"]
        C_arr = file["C_arr"]
        if C_pick != None:
            ex_JR = JR_arr[C_pick]
            ex_eff = eff_arr[C_pick]
            axs.scatter(ex_JR, ex_eff, marker = "P", color = nth_one,zorder = 2)
        axs.plot(JR_arr, eff_arr, linestyle, label=label, color = color, alpha = alpha, zorder = 1)
        axs.set_xlabel(r"$I^{\Sigma,L}$ [$k_B^2 T/h$]")
        axs.set_ylabel(r"$\eta$")
        file.close()
    def make_transf_figure(self, filename):
        Es = np.linspace(nonthermal_left.E_low, nonthermal_left.E_high, 10000)
        file = np.load(filename)

        C_arr = file["C_arr"]
        C_pick = 200
        
        fig = plt.figure(figsize = (col,0.8*col), layout = "constrained")
        figs = fig.subfigures(2, 1)
        axs = [subfig.subplots() for subfig in figs]  
   
        self.eff_plot(axs[0],system = nonthermal_left, color = nonthermal,  label = "nonthermal real.", filename = eff_file_nth, C_pick=C_pick)
        self.eff_plot(axs[0],linestyle="--",system = thermal_left, color = hot, label = "thermal real.",  filename = eff_file_th)
        axs[0].locator_params(axis = "x", nbins = 3)
        axs[0].locator_params(axis = "y", nbins = 3)
        axs[0].annotate("(a)", (0.8,0.85), xycoords = "axes fraction")
        Es = np.linspace(nonthermal_left.E_low, nonthermal_left.E_high, 10000)
        axs[1].plot(Es, nonthermal_left._transmission_avg(C_arr[C_pick])(Es), color = nth_one, label = "Optimal")

        jmax, transf_max, roots_max = nonthermal_left.constrained_current_max()
        
        cond_avg = nonthermal_left._avg_condition(C_arr[C_pick])

        positions = []
        gammas = []
        
        for j in range(0,len(roots_max),2):
            roots = root_finder_guesses(cond_avg, 0.8*roots_max[j], 1.2*roots_max[j+1])
            for i in range(0,len(roots),2):
                positions.append((roots[i] + roots[i+1])/2)
                gammas.append(self.width*np.abs(roots[i+1]-roots[i])/2)

        transf = self.real_transf(gammas, positions, Es=Es)
        
        axs[1].plot(Es, transf, color = nth_three, label = "Realistic")
        axs[1].set_xlabel(r"$\varepsilon$ [$k_B T$]")
        axs[1].set_ylabel("Transmission func.")
        axs[1].annotate("(b)", (0.8,0.85), xycoords = "axes fraction")

        lines, labels = axs[0].get_legend_handles_labels()
        
        figs[0].legend(lines, labels, loc = "outside right upper")

        lines, labels = axs[1].get_legend_handles_labels()
        figs[1].legend(lines, labels, loc = "outside right upper")

        file.close()

        return fig

def make_discrete_illus():
    fig, axs = plt.subplots(1,2,figsize = (col,0.6*col), layout = "constrained")
    ds = np.linspace(0,1,1000)
    ks = np.array([-0.5,0.4]).reshape(-1,1)
    ms = np.array([0.7,0.3]).reshape(-1,1)

    Is = ds*ks + ms

    axs[0].plot(ds, Is.T[:,0],"tab:blue")
    axs[0].plot(ds, Is.T[:,1],"tab:green")
    axs[0].scatter(1, float(Is.T[-1,0]), color="tab:blue")
    axs[0].scatter(0, float(Is.T[0,1]), color="tab:green")
    axs[0].set_xlabel(r"$d_\gamma$")
    axs[0].set_ylabel(r"$I^y$")
    axs[0].set_yticks([])
    axs[0].text(0.8, Is.T[-1,0] + 0.13, r"$d_i$", color = "tab:blue")
    axs[0].text(0.1, 0.41, r"$d_j$", color = "tab:green")
    axs[0].text(0.25, 0.25, r"$\left.\frac{\partial I^y}{\partial d_i}\right|_{I^x_\mathrm{fix}} < 0$", color = "tab:blue", size = 12)
    axs[0].text(0.25, 0.65, r"$\left.\frac{\partial I^y}{\partial d_j}\right|_{I^x_\mathrm{fix}} > 0$", color = "tab:green", size = 12)
    axs[0].annotate("(a)", (0.05,0.05), xycoords = "axes fraction")

    ds = np.linspace(0,1,1000)
    ks = np.array([4,5]).reshape(-1,1)
    ms = np.array([0.7,0.3]).reshape(-1,1)
    bs = np.array([-5,-3]).reshape(-1,1)

    Is = bs*ds**2 + ds*ks + ms

    axs[1].plot(ds, Is.T[:,0],"tab:blue")
    axs[1].plot(ds, Is.T[:,1],"tab:green")
    axs[1].scatter(1, float(Is.T[-1,0]), color="tab:blue")
    axs[1].scatter(0, float(Is.T[0,1]), color="tab:green")
    axs[1].set_xlabel(r"$d_\gamma$")
    axs[1].set_ylabel(r"$S^x$")
    axs[1].set_yticks([])
    axs[1].text(0.8, Is.T[-1,0] + 0.1, r"$d_i$", color = "tab:blue")
    axs[1].text(0.1, 0.4, r"$d_j$", color = "tab:green")
    axs[1].annotate("(b)", (0.05,0.05), xycoords = "axes fraction")

    return fig

if __name__ == "__main__":
    width = 0.1


    dist_type = "dippeak_test"
    # dist_type = "mixed"

    fig_type = "png"

    # folder = "data/"+dist_type+"/"
    folder = "data/"

    file = np.load(folder+"th_params_"+dist_type+".npz")
    th_dist_params = file['arr_0']
    file.close()
    muR = th_dist_params[0]
    TR = th_dist_params[1]
    
    file = np.load(folder+"nth_params_"+dist_type+".npz")
    nth_dist_params = file['arr_0']
    file.close()

    dip_peak = False
    if "dippeak" in dist_type:
        occupf_L_nth = dists.thermal_with_lorentz(*nth_dist_params)
        example_target = 0.008
        dip_peak = True
    else:
        example_target = 0.0015
        occupf_L_nth = dists.two_thermals(*nth_dist_params)

    left_virtual = two_terminals(-20, 20, occupf_L = occupf_L_nth, muR=1.2, TR=TR)

    I_E, I_N = dists.buttiker_probe(left_virtual)

    print("Energy and particle current between thermal probe and nonthermal distribution: ", I_E, I_N)
    print("New muL and TL: ", left_virtual.muR, left_virtual.TR)

    ########## Setup for the two systems ##############################
    E_low = -5
    E_high = 5

    thermal_left = two_terminals(E_low, E_high, muL=left_virtual.muR, TL = left_virtual.TR, muR = muR, TR = TR)
    nonthermal_left = two_terminals(E_low, E_high, occupf_L= occupf_L_nth, muR = muR, TR = TR)

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

    ################################################################

    fig = make_discrete_illus()
    plt.savefig("figs/"+fig_type+"/method_illus."+fig_type, dpi = 1000)

    ######## Plot the regular optimization results ############

    optimPlot = OptimPlot(thermal_left, nonthermal_left, True)    

    filenames = [folder+"th_"+dist_type+"_eff.npz",folder+"nth_"+dist_type+"_eff.npz",folder+"th_"+dist_type+"_noise.npz",
                folder+"nth_"+dist_type+"_noise.npz",folder+"th_"+dist_type+"_product.npz",folder+"nth_"+dist_type+"_product.npz"]
    examplefiles = [folder+"th_"+dist_type+"_example.npz", folder+"nth_"+dist_type+"_example.npz"]
    

    fig = optimPlot.make_example_figure(examplefiles[1], make_eff=True, make_noise = True, make_product=False, striped = True)
    plt.savefig("figs/"+fig_type+"/"+dist_type+"_example."+fig_type, dpi = 1000)

    fig = optimPlot.make_big_example_figure(examplefiles, dip_peak=dip_peak)
    plt.savefig("figs/"+fig_type+"/"+dist_type+"_big_example."+fig_type, dpi = 1000)

    fig = optimPlot.make_figure(examplefiles[1],make_eff=True, make_noise=True, make_product=True, filenames=filenames, dip_peak=dip_peak)
    plt.savefig("figs/"+fig_type+"/"+dist_type+"_opts."+fig_type, dpi = 1000)

    fig = optimPlot.make_crossing_figure(examplefiles[1], nonthermal_left)
    plt.savefig("figs/"+fig_type+"/"+dist_type+"_crossing."+fig_type, dpi = 1000)
    
    fig = optimPlot.make_all_crossing_figure([filenames[1],filenames[3],filenames[5]], nonthermal_left, example_file=examplefiles[1])
    plt.savefig("figs/"+fig_type+"/"+dist_type+"_crossing_all."+fig_type, dpi = 1000)

    fig = optimPlot.make_char_eff_figure(examplefiles[1], nonthermal_left)
    plt.savefig("figs/"+fig_type+"/"+dist_type+"_char_eff."+fig_type, dpi = 1000)

    fig = optimPlot.make_local_eff_figure(filenames[1], nonthermal_left, dip_peak=dip_peak)
    plt.savefig("figs/"+fig_type+"/"+dist_type+"_local_eff."+fig_type, dpi = 1000)

    ###################################################################

    if dip_peak:
        ################# Plot for realistic transmission ########################

        eff_file_nth = folder+"nth_"+dist_type+"_realtransf_eff.npz"
        eff_file_th = folder+"th_"+dist_type+"_realtransf_eff.npz"
        realTransfPlot = RealTransfPlot(width)
        
        fig = realTransfPlot.make_transf_figure(filenames[1])
        plt.savefig("figs/"+fig_type+"/"+dist_type+"_real_transf."+fig_type, dpi = 1000)

        ##############################################################################
        ################## Plot for optimization with second variable ################

        secondary_prop = "muR"
        filenames_second = [folder+"th_"+dist_type+"_eff_"+secondary_prop + ".npz",folder+"nth_"+dist_type+"_eff_"+secondary_prop + ".npz",folder+"th_"+dist_type+"_noise_"+secondary_prop + ".npz",
                    folder+"nth_"+dist_type+"_noise_"+secondary_prop + ".npz",folder+"th_"+dist_type+"_product_"+secondary_prop + ".npz",folder+"nth_"+dist_type+"_product_"+secondary_prop + ".npz"]
        max_filenames = [folder+"th_max_"+dist_type+secondary_prop+".npz",folder+"nth_max_"+dist_type+secondary_prop+".npz"]
        examplefile = folder+"nth_"+dist_type+"_eff_"+secondary_prop + "example.npz"
        
        secondaryPlot = SecondaryPlot(thermal_left, nonthermal_left,secondary_prop, verbose=True)

        fig = secondaryPlot.make_eff_figure(filenames_second)
        plt.savefig("figs/"+fig_type+"/"+dist_type+"_"+secondary_prop+"_only_eff_."+fig_type, dpi = 1000)

        fig = secondaryPlot.make_example_figure(examplefile)
        plt.savefig("figs/"+fig_type+"/"+dist_type+"_"+secondary_prop+"_example."+fig_type, dpi = 1000)

        fig = secondaryPlot.make_transition_figure(filenames_second)
        plt.savefig("figs/"+fig_type+"/"+dist_type+"_"+secondary_prop+"_transition."+fig_type, dpi = 1000)


