# README
This repository contains data, code and figures for "Optimizing energy conversion with nonthermal resources in steady-state quantum
devices" 
(Reference to be provided)

## Organization and setup
The environment.yaml file specifies the Python environment that was used during development. Highlights are:
- python 3.12.7
- numpy 1.26.4
- scipy 1.13.1

All data used in the article is provided under the data folder. The code reads and writes .npz files, but .csv files are provided for compatibility outside Python. 

Under the figs folder all figures used in the paper are found.

### Python files
- The file thermo_funcs.py contains all functions needed for the numerical optimizations and calculations. In particular, the class two_terminals gives and object that describes a coherent conductor between two terminals. 
- In data_producer.py the data files are produced using the functions provided by thermo_funcs.py
- The data is plotted by plotter.py
- The file data_converter.py converts .npz files to .csv.

## Numerical procedure
The general approach to the calculations is described in Appendix D in the paper. Some further details are brought up here, along with some caveats and notes to the numerical methods.

First, note that the right reservoir is considered the cold one which we want to cool, such that the left reservoir is nonthermal or its thermal equivalent. In the paper, it is the other way around. This does not change the results, but note that the output current is  denoted "JR". Currents are still defined as positive out of reservoirs.

The most important and difficult part of the calculations is the evaluation of integrals, reliant on scipy's integrate.quad. Because the integrands contain boxcar functions, there are potentially large energy regions where the integrand is zero. If too much of these regions are within the integration bounds, the integral evaluation is erroneous, simply returning zero. Due to this, the integral evaluation is divided into a sum of integrals over the areas where the transmission function is set to one. This requires that one finds the zero points of the condition functions that determines the transmission functions (the functions inside the Heaviside functions giving the optimal transmission functions). To find the roots, a brute-force search is done by finding the closest root for 100 000 points within a set minimum and maximum energy. This method works well even for small output currents, where the boxcars are very narrow. 

However, there is an additional calculational cost and the search area must be reasonable narrow. The latter is addressed by adjusting the energy limits close to the maximum cooling window. This cooling window also needs to be searched for, so note that if one changes a distribution function drastically in a two_terminals objects, it is prudent to call adjust_limits with some wide starting values. If the integration methods are called and the maximum cooling window is not properly encompassed by two_terminal's energy limits, there will likely be errors. 

There is a caveat regarding the integration methods, relating to dividing the integration into different parts. Since all examples in the paper address cooling, there is always a finite window of maximum cooling, for finite temperatures and electrochemical potentials. On the other hand, a heat engine has a power production window between the distribution crossing point and infinity, for thermal distributions. This creates open-ended boxcars when finding the optimal transmission functions. This is presently not taking into account in the integrations methods, which assume that there are a number of finite boxcars. For reproducing the results of the paper, this is not an issue, but it is a limitation to be aware of. 

Finally, note that the equation solvers to some optimization methods do not always converge within error tolerance. This is especially the case for methods involving trade-off (or product) optimization, where there are more variables to solve for, and for very low output currents. These solutions are saved by data_producer.py, but plotter.py removes data points with errors > 10^(-7). 
