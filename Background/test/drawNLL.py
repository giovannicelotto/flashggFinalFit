# %%
import uproot
import matplotlib.pyplot as plt
import numpy as np
import mplhep
mplhep.style.use("CMS")
category = 2
number_of_pdfs = 1
functions={}
labels = ["Bern3", "Bern4", "Pow1"]
for i in range(number_of_pdfs):
    i=i+1
    functions["F%d"%i] = ["/t3home/gcelotto/ggHbb/WSFit/datacards/higgsCombinescan1D.MultiDimFit.mH125.root", labels[i]] 
files = [uproot.open(f[0]) for f in list(functions.values())]
# %%
curves = []

#labels = ["Bern4", "Bern5", "Expo3"]
fig, ax = plt.subplots(1, 1)
minValue = np.inf
for idx, (key, file) in enumerate(zip(functions.keys(), files)):
    tree = file["limit"]
    branches = tree.arrays()
    deltaNLL = branches["deltaNLL"]
    r = branches["r"]
    nll = branches["nll"]
    nll_zero = branches["nll0"]

    # First value of r is minimum!
    # Get sorting indices
    order = np.argsort(r)
    
    # Apply sorting consistently
    r = r[order]
    deltaNLL = deltaNLL[order]
    nll = nll[order]
    nll_zero = nll_zero[order]
    print(nll_zero)
    if key =="Envelope":
        color='black'
        markersize=4
        label = functions[key][1]
    else:
        color="C%d"%(int(key[1]))
        markersize=2
        label = functions[key][1]

    ax.plot(r, 2*(deltaNLL+nll+nll_zero) , color=color, linestyle='--', marker='o', markersize=markersize, label=label)
    if minValue<np.min(2*(deltaNLL+nll+nll_zero)):
        pass
    else:
        minValue = np.min(2*(deltaNLL+nll+nll_zero))
#ax.set_ylim(np.min(curves), np.min(curves)+10)
ax.set_xlabel('r')
ax.set_xlim(ax.get_xlim())
ax.set_ylabel("-2ln(L) + c")
ax.hlines(y=minValue+1, xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1], linestyle='dashed', color='grey')
ax.hlines(y=minValue+4, xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1], linestyle='dashed', color='grey')
ax.legend()
fig.show()


# %%
curves = np.array(curves)   # shape = (n_curves, n_points)
import numpy as np

# --- Envelope already computed as before ---
envelope = np.min(curves, axis=0)
r_values = r  # from before

# Minimum point
ymin = np.min(envelope)
rmin = r_values[np.argmin(envelope)]

# Target line = minimum + 1
target = ymin + 1

# Function to find crossings between curve and target
def find_crossings(x, y, target):
    crossings = []
    for i in range(len(x)-1):
        if (y[i]-target)*(y[i+1]-target) <= 0:  # sign change or exact crossing
            # Linear interpolation
            slope = (y[i+1]-y[i]) / (x[i+1]-x[i])
            if slope != 0:
                xc = x[i] + (target - y[i]) / slope
                crossings.append(xc)
    return crossings

crossings = find_crossings(r_values, envelope, target)

print("Minimum at r =", rmin, "with -2lnL =", ymin)
print("Crossings at r =", crossings)




#ax.set_xlim(-4,5)










# %%

labels=["Bern2", "Bern3", "Expo1"]
toy=True
if (toy):
    envelope = uproot.open("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/ws_dt_fit/Aug20/higgsCombineEnvelope.MultiDimFit.mH125.root")
    f0 = uproot.open("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/ws_dt_fit/Aug20/higgsCombinefixed_pdfCat%d_%d.MultiDimFit.mH125.root"%(category, 0))
    f1 = uproot.open("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/ws_dt_fit/Aug20/higgsCombinefixed_pdfCat%d_%d.MultiDimFit.mH125.root"%(category, 1))
    f2 = uproot.open("/pnfs/psi.ch/cms/trivcat/store/user/gcelotto/ws_dt_fit/Aug20/higgsCombinefixed_pdfCat%d_%d.MultiDimFit.mH125.root"%(category, 2))
    fig, ax = plt.subplots(1, 1)
    for idx, file in enumerate([envelope, f0, f1, f2]):
        tree = file["limit"]
        branches = tree.arrays()
        deltaNLL = branches["deltaNLL"]
        r = branches["r"]
        nll = branches["nll"]
        nll_zero = branches["nll0"]
        if idx ==0:
            color='black'
            markersize=4
            label = 'envelope'
        else:
            color="C%d"%(idx-1)
            markersize=2
            label = labels[idx-1]
        ax.plot(r, 2*(deltaNLL+nll+nll_zero), color=color, linestyle='none', marker='o', markersize=markersize, label=label)
    ax.legend()
    #ax.set_ylim(8595, 8610)
    #ax.set_yscale('log')
# %%
