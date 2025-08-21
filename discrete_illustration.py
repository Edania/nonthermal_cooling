import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'NewComputerModernSans08'
plt.rcParams["font.size"] = 10

plt.rcParams["axes.titlesize"] = "medium"
plt.rcParams["figure.titlesize"] = "medium"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams['pdf.fonttype'] = 42
pt = 1/72 
col = 246*pt

fig_type = "png"
fig, axs = plt.subplots(1,2,figsize = (col,0.6*col), layout = "constrained")


ds = np.linspace(0,1,1000)
ks = np.array([-0.5,0.4]).reshape(-1,1)
ms = np.array([0.7,0.3]).reshape(-1,1)

Is = ds*ks + ms



axs[0].plot(ds, Is.T[:,0],"tab:blue")
axs[0].plot(ds, Is.T[:,1],"tab:cyan")
axs[0].scatter(1, float(Is.T[-1,0]), color="tab:blue")
axs[0].scatter(0, float(Is.T[0,1]), color="tab:cyan")
#axs[0].title("Illustration of how average current \n changes with transmission height")#, fontname = "Segoe UI")
# axs[0].set_title("Illustration of how average current \n changes with transmission height")
axs[0].set_xlabel(r"$d_\gamma$")
#axs[0].ylabel(r"$I_y$", size = 12)
axs[0].set_ylabel(r"$I^y$")
axs[0].set_yticks([])
axs[0].text(0.8, Is.T[-1,0] + 0.13, r"$d_i$", color = "tab:blue")
axs[0].text(0.1, 0.41, r"$d_j$", color = "tab:cyan")
axs[0].text(0.25, 0.25, r"$\left.\frac{\partial I^y}{\partial d_i}\right|_{I^x_\mathrm{fix}} < 0$", color = "tab:blue", size = 12)
axs[0].text(0.25, 0.65, r"$\left.\frac{\partial I^y}{\partial d_j}\right|_{I^x_\mathrm{fix}} > 0$", color = "tab:cyan", size = 12)
axs[0].annotate("(a)", (0.05,0.05), xycoords = "axes fraction")
#axs[0].legend()

ds = np.linspace(0,1,1000)
ks = np.array([4,5]).reshape(-1,1)
ms = np.array([0.7,0.3]).reshape(-1,1)
bs = np.array([-5,-3]).reshape(-1,1)

Is = bs*ds**2 + ds*ks + ms


axs[1].plot(ds, Is.T[:,0],"tab:blue")
axs[1].plot(ds, Is.T[:,1],"tab:cyan")
axs[1].scatter(1, float(Is.T[-1,0]), color="tab:blue")
axs[1].scatter(0, float(Is.T[0,1]), color="tab:cyan")
#axs[1].title("Illustration of how average current \n changes with transmission height")#, fontname = "Segoe UI")
# axs[1].set_title("Illustration of how noise current \n changes with transmission height")
axs[1].set_xlabel(r"$d_\gamma$")
#axs[1].ylabel(r"$I_y$", size = 12)
axs[1].set_ylabel(r"$S^x$")
axs[1].set_yticks([])
axs[1].text(0.8, Is.T[-1,0] + 0.1, r"$d_i$", color = "tab:blue")
axs[1].text(0.1, 0.4, r"$d_j$", color = "tab:cyan")
axs[1].annotate("(b)", (0.05,0.05), xycoords = "axes fraction")
# axs[1].text(0.7, 0.4, r"$\left.\frac{\partial I_y}{\partial d_1}\right|_{I_x} < 0$", color = "tab:blue", size = 12)
# axs[1].text(0.25, 0.3, r"$\left.\frac{\partial I_y}{\partial d_2}\right|_{I_x} > 0$", color = "tab:cyan", size = 12)
#axs[1].legend()

# axs[1].savefig("figs/linear_illus.eps")
plt.savefig("figs/"+fig_type+"/method_illus."+fig_type, dpi = 1000)