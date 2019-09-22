# %%
import numpy as np
from scipy.optimize import brute
import pickle
from scipy.constants import mu_0
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
from numba import jit

verbose = False
if "-V" in sys.argv:
    verbose = True
refresh = False
if "-r" in sys.argv:
    refresh = True
draw = False
if "-g" in sys.argv:
    draw = True
noCheck = False
if "-nc" in sys.argv:
    noCheck = True

# Relative constants
# Things that aren't easily changed
BR_Mag = 1 #1.2
SIGMA_WIRE = 3.46e6
CP_WIRE = 296
RHO_WIRE = 6440 # kg/m3
REL_PERM_STEEL = 2000
REL_PERM_NEO = 1.05
REL_PERM_AIR = 1
VISCO_WIRE = 0.0024 #Pa.s
RHO_TUBE = 1100 # kg/m3
RHO_IRON = 7850 # kg/m3
RHO_MAG = 7010 # kg/m3

BIG_NUM = 1e12

moe = 1e-8

thickWireWall = 0.5 # mm
rWireIn = 1 # mm

dTravel = 15 # 9mm
fMotor = 10 # 9N
freq = 5 # 1Hz

deltaT = 30

maxLayers = 14

# %%
# rWireIn lMag rMag lCore
# r_rWireIn = (0.05,2.5) # mm - reasonable sizes for silicone tube radius
r_lMag = (5,95) # mm - try not to exceed 10 cm size
r_rMag = (1,20) # mm - try not to exceed 40 mm diameter - health and safety hazard
r_lCore = (1,25) # mm - try to be decently sized
r_lWireVessel = (5,95)
grid = (r_lMag,r_rMag,r_lCore,r_lWireVessel)
outSelect = 1 # optimising for min power * mass

# %%
@jit(nopython=True)
def motorOpt(x, *args):

    rho = args[0]
    sigma = args[1]
    cp = args[2]
    br = args[3]
    dTravel = args[4]
    fMotor = args[5]
    freq = args[6]
    thickWireWall = args[7]
    layers = args[8]
    deltaT = args[9]
    visco = args[10]
    outSelect = args[11]
    rWireIn = args[12]/1000

    # print("rho:{}, sigma:{}, cp:{}, br:{}, dTravel:{}, fMotor:{}, freq:{}, thickWireWall:{}, layers:{}, deltaT:{}, outSelect:{}".format(rho,sigma,cp,br,dTravel,fMotor,freq,thickWireWall,layers,deltaT,outSelect))

    # print("Valid! layers {}, x = {}".format(layers,x))

    # print("motorOpt: {}".format(x))
    
    lMag = x[0] / 1000
    rMag = x[1] / 1000
    lCore = x[2] / 1000
    lWireVessel = x[3] / 1000
    dTravel = dTravel / 1000
    thickWireWall = thickWireWall / 1000
    
    rWireOut = rWireIn + thickWireWall # Wire outer radius
    
    rShellIn = rMag+rWireOut*layers*2+1.5e-3 # Shell inner radius
    # if verbose:
    #     print("rShellIn: {} mm".format(rShellIn*1e3))

    aWire = rWireIn**2*np.pi
    #lWireTotal = lWireVessel*np.pi*(layers+1)*layers + lWireVessel*rMag*np.pi/rWireOut # Total length of wire
    lWireTotal = 0
    wireCycle = lWireVessel/(rWireOut*2)
    for i in range(1,layers+1):
        rLayer = (i*rWireOut*2)+rMag
        lWireTotal += 2*rLayer*np.pi * wireCycle
    # if verbose:
    #     print("lWireTotal: {} m".format(lWireTotal))

    # 5. Minimum travel distance
    if dTravel > (lWireVessel-lCore):
        # if verbose:
        #     print("Not enough travel distance, {} mm v {} mm, x: {}, layer {}".format(dTravel,lCore-lWireVessel,x,layers))
        return BIG_NUM

    # 6. Wire Vessel needs to be shorter than Core+Mag
    if lWireVessel > lCore+lMag:
        # if verbose:
        #     print("Wire Vessel too long, {} mm, x: {}, layer {}".format(lWireVessel,x,layers))
        return BIG_NUM

    # 4. Maximum shell dimensions ($ r_{shellIn} $)
    maxShell = 80 #mm
    if rShellIn*2 > maxShell:
        # if verbose:
        #     print("Shell Radius too large, {} mm, x: {}, layer {}".format(rShellIn,x,layers))
        return BIG_NUM

    # 3. Maximum volume of liquid metal in circuit ($ A_{wire} l_{wireTotal} $)
    maxVolume = 15 #ml
    aWire = rWireIn**2*np.pi
    volume = aWire*lWireTotal
    if volume/1e3 > maxVolume:
        # if verbose:
        #     print("Volume too large, {} ml, x: {}, layer {}".format(volume/1e3,x,layers))
        return BIG_NUM

    resTotal = lWireTotal/aWire/sigma/4 # Total wire resistance
    # if verbose:
    #     print("resTotal: {} Ohm".format(resTotal))
    
    rg = np.log(rShellIn/rMag)/(2*np.pi*lCore) * REL_PERM_AIR# gap reluctance
    # if verbose:
    #     print("RG:{}".format(rg))
    rm = lMag / (np.pi*rMag**2) * REL_PERM_NEO
    # if verbose:
    #     print("RM:{}".format(rm))

    rmTtl = rg + rm # Total reluctance - approx. gap reluctance + magnet reluctance
    
    emmf = br*lMag*REL_PERM_NEO # electromagneticmotive force
    
    flux = emmf / rmTtl # Magnetic flux through circuit

    AA = lCore*2*np.pi*(rMag+layers*rWireOut*2-rWireOut) # approx the largest area layer

    if lWireVessel > lCore:
        lWireInField = lCore*np.pi*(layers+1)*layers + lCore*rMag*np.pi/rWireOut
    else:
        lWireInField = lWireVessel*np.pi*(layers+1)*layers + lWireVessel*rMag*np.pi/rWireOut
    
    # LAA = 0 # Length of wire in field/AreaAction
    # for layer in range(1,layers+1):
    #     LAA += (2*layer-1)/(2*rMag+4*layer*rWireOut-2*rWireOut)
    
    B = flux / AA

    if B > 2:
        # if verbose:
        #     print("B too large: {} H".format(B))
        B = 2

    maxI = 18
    I = fMotor / B / lWireInField
    # if verbose:
    #     print("I: {} A".format(I))
    if I > maxI:
        return BIG_NUM

    # V = I * resTotal
    # if verbose:
    #     print("V: {} V".format(V))

    pIn = I**2 * resTotal
    # if verbose:
    #     print("Pin: {} W".format(pIn))

    pMech = 2*dTravel*fMotor*freq # Mechanical power output

    pHeat = pIn - pMech

    Q = pHeat / (RHO_WIRE*deltaT*CP_WIRE)
    # if verbose:
    #     print("Q: {} ml/s".format(Q*1e6))

    # pressure = 8*visco*lWireTotal*Q/(np.pi*rWireIn**4)
    # if verbose:
    #     print("Pressure: {} Pa".format(pressure))

    if Q <= 0 or pIn <= 0:
        # if verbose:
        #     print("Invalid Output! layers {}, x = {}, Q = {}, pIn = {}".format(layers,x,Q,pIn))
        return BIG_NUM # return some really large number
    
    volWire = lWireTotal * np.pi * rWireIn**2
    volTube = lWireTotal * np.pi * (rWireOut**2 - rWireIn**2)
    volMag = lMag * np.pi * rMag**2
    volCore = lCore * np.pi * rMag**2
    thickShell = 5e-3
    thickBase = 5e-3
    volShell = (lCore+lMag) * np.pi * ((rShellIn+thickShell)**2 - rShellIn**2) + thickBase * np.pi * (rShellIn+thickShell)**2

    massTotal = volWire*RHO_WIRE+volTube*RHO_TUBE+(volCore+volShell)*RHO_IRON+volMag*RHO_MAG

    # Choose which output variable to optimise for
    # Can optimise power * mass or power * volume to maximise size efficiency
    if outSelect == 0:
        return Q
    elif outSelect == 1:
        return pIn * massTotal
    # elif outSelect== -1:
    #     return Q,pIn,I,V,pressure


def motorMain(x, *args):
    rho = args[0]
    sigma = args[1]
    cp = args[2]
    br = args[3]
    dTravel = args[4]
    fMotor = args[5]
    freq = args[6]
    thickWireWall = args[7]
    layers = args[8]
    deltaT = args[9]
    visco = args[10]
    rWireIn = args[11]/1000
    
    lMag = x[0] / 1000
    rMag = x[1] / 1000
    lCore = x[2] / 1000
    lWireVessel = x[3] / 1000
    dTravel = dTravel / 1000
    thickWireWall = thickWireWall / 1000
    
    rWireOut = rWireIn + thickWireWall # Wire outer radius
    
    rShellIn = rMag+rWireOut*layers*2+1.5e-3 # Shell inner radius
    if verbose:
        print("rShellIn, {} mm".format(rShellIn*1e3))

    aWire = rWireIn**2*np.pi
    #lWireTotal = lWireVessel*np.pi*(layers+1)*layers + lWireVessel*rMag*np.pi/rWireOut # Total length of wire
    lWireTotal = 0
    wireCycle = lWireVessel/(rWireOut*2)
    for i in range(1,layers+1):
        rLayer = (i*rWireOut*2)+rMag+rWireOut
        lWireTotal += 2*rLayer*np.pi * wireCycle
#        print(lWireTotal)

    if verbose:
        print("lWireTotal, {} m".format(lWireTotal))

    resTotal = lWireTotal/aWire/sigma/4 # Total wire resistance
    if verbose:
        print("resTotal, {} Ohm".format(resTotal))
    
    rg = np.log(rShellIn/rMag)/(2*np.pi*lCore) * REL_PERM_AIR# gap reluctance
    if verbose:
        print("RG,{}".format(rg))
    rm = lMag / (np.pi*rMag**2) * REL_PERM_NEO
    if verbose:
        print("RM,{}".format(rm))

    rmTtl = rg + rm # Total reluctance - approx. gap reluctance + magnet reluctance
    
    emmf = br*lMag*REL_PERM_NEO # electromagneticmotive force
    
    flux = emmf / rmTtl # Magnetic flux through circuit

    AA = lCore*2*np.pi*(rMag+layers*rWireOut*2-rWireOut) # approx the largest area layer

    if lWireVessel > lCore:
        lWireInField = lWireTotal / lWireVessel * lCore
    else:
        lWireInField = lWireTotal
    
    # LAA = 0 # Length of wire in field/AreaAction
    # for layer in range(1,layers+1):
    #     LAA += (2*layer-1)/(2*rMag+4*layer*rWireOut-2*rWireOut)
    
    B = flux / AA

    if B > 2:
        if verbose:
            print("B too large: {} H".format(B))
        B = 2
    if verbose:
        print("B, {} H".format(B))

    I = fMotor / B / lWireInField
    if verbose:
        print("I, {} A".format(I))

    V = I * resTotal
    if verbose:
        print("V, {} V".format(V))

    pIn = I**2 * resTotal
    if verbose:
        print("Pin, {} W".format(pIn))

    pMech = 2*dTravel*fMotor*freq # Mechanical power output

    pHeat = pIn - pMech

    Q = pHeat / (RHO_WIRE*deltaT*CP_WIRE)
    if verbose:
        print("Q, {} ml/s".format(Q*1e6))

    pressure = 8*visco*lWireTotal*Q/(np.pi*rWireIn**4)
    if verbose:
        print("Pressure, {} Pa".format(pressure))

    if Q <= 0 or pIn <= 0:
        if verbose:
            print("Invalid Output! layers {}, x = {}, Q = {}, pIn = {}".format(layers,x,Q,pIn))
        return BIG_NUM # return some really large number
    
    volWire = lWireTotal * np.pi * rWireIn**2
    volTube = lWireTotal * np.pi * (rWireOut**2 - rWireIn**2)
    volMag = lMag * np.pi * rMag**2
    volCore = lCore * np.pi * rMag**2
    thickShell = 3.5e-3
    thickBase = 10e-3
    volShell = (lCore+lMag) * np.pi * ((rShellIn+thickShell)**2 - rShellIn**2) + thickBase * np.pi * (rShellIn+thickShell)**2

    massTotal = volWire*RHO_WIRE+volTube*RHO_TUBE+(volCore+volShell)*RHO_IRON+volMag*RHO_MAG

    return Q,pIn,massTotal,pIn * massTotal,I,V,pressure

# %% DON'T RUN ON JUPYTER
# def motorOpt(x,rho,sigma,cp,br,dTravel,fMotor,freq,thickWireWall,layers,deltaT,outSelect)
optOut = []
if refresh:
    for layers in range(2,maxLayers+1,2):
        print("Layer {}".format(layers))
        arglist = (RHO_WIRE,SIGMA_WIRE,CP_WIRE,BR_Mag,dTravel,fMotor,freq,thickWireWall,layers,deltaT,VISCO_WIRE,outSelect,rWireIn)
        optOut.append(brute(motorOpt,ranges=grid,args=arglist,Ns=25,full_output=True,disp=True,finish=None))

    with open('optOut.pkl','wb') as f:
        pickle.dump(optOut, f)
    print("Done")

# %%
with open('optOut.pkl','rb') as f:
    optOut = pickle.load(f)

# DIAGRAMMING
def diagramDraw(x,layers,thickWireWall,rWireIn,name):
    lMag = x[0]
    rMag = x[1]
    lCore = x[2]
    lWireVessel = x[3]
    thickShell = 3.5
    thickBase = 10
    thickbobbin = 1.5

    realLayers = (layers+1)*2
    rWireOut = rWireIn + thickWireWall # Wire outer radius
    rShellIn = rMag+rWireOut*realLayers*2+thickbobbin # Shell inner radius
    print(rMag)
    rShellOut = rShellIn + thickShell
    isize = (100,100)
    lTtl = lCore+lMag+thickBase

    xblank = (isize[0] - lTtl)/2
    yblank = (isize[1] - 2*rShellOut)/2

    # Create figure and axes
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xlim([0, isize[0]])
    plt.ylim([0, isize[1]])
    plt.title("{} Layer Optimised Motor Schematic".format(realLayers))
    plt.xlabel("mm")
    plt.ylabel("mm")
    plt.grid()

    shellPatch = patches.Patch(color='grey', label='Shell')
    magPatch = patches.Patch(color='green', label='Magnet')
    corePatch = patches.Patch(color='red', label='Core')
    wirePatch = patches.Patch(color='blue', label='Wires')
    bobbinPatch = patches.Patch(color='yellow', label='Wire Bobbin')
    plt.legend(handles=[shellPatch,magPatch,corePatch,wirePatch,bobbinPatch])

    # Shell -grey
    shellTop0 = (xblank,yblank+thickShell+2*rShellIn)
    shellTopWidth = lTtl
    shellTopHeight = thickShell
    shellTop = patches.Rectangle(shellTop0,shellTopWidth,shellTopHeight,linewidth=0,color='grey')
    ax.add_patch(shellTop)

    shellBottom0 = (xblank,yblank)
    shellBottomWidth = lTtl
    shellBottomHeight = thickShell
    shellBottom = patches.Rectangle(shellBottom0,shellBottomWidth,shellBottomHeight,linewidth=0,color='grey')
    ax.add_patch(shellBottom)

    shellLeft0 = (xblank,yblank)
    shellLeftWidth = thickBase
    shellLeftHeight = 2*rShellOut
    shellLeft = patches.Rectangle(shellLeft0,shellLeftWidth,shellLeftHeight,linewidth=0,color='grey')
    ax.add_patch(shellLeft)

    # Mag - green
    Mag0 = (xblank+thickBase,yblank+rShellOut-rMag)
    MagWidth = lMag
    MagHeight = 2*rMag
    # Create a Rectangle patch
    Mag = patches.Rectangle(Mag0,MagWidth,MagHeight,linewidth=0,color='g')
    # Add the patch to the Axes
    ax.add_patch(Mag)

    # Core - red
    Core0 = (xblank+thickBase+lMag,yblank+rShellOut-rMag)
    CoreWidth = lCore
    CoreHeight = 2*rMag
    # Create a Rectangle patch
    Core = patches.Rectangle(Core0,CoreWidth,CoreHeight,linewidth=0,color='r')
    # Add the patch to the Axes
    ax.add_patch(Core)

    # Wires - blue
    wire0 = (xblank+lTtl,yblank+2*rShellOut-thickShell)
    wireWidth = -lWireVessel
    wireHeight = -realLayers*rWireOut*2
    # Create a Rectangle patch
    wire = patches.Rectangle(wire0,wireWidth,wireHeight,linewidth=0,color='b')
    # Add the patch to the Axes
    ax.add_patch(wire)
    wire2_0 = (xblank+lTtl,yblank+thickShell)
    wire2Width = -lWireVessel
    wire2Height = realLayers*rWireOut*2
    # Create a Rectangle patch
    wire = patches.Rectangle(wire2_0,wire2Width,wire2Height,linewidth=0,color='b')
    # Add the patch to the Axes
    ax.add_patch(wire)

    # bobbin - yellow
    bobbintop0 = (xblank+lTtl,yblank+rShellOut+rMag)
    bobbintopWidth = -lWireVessel
    bobbintopHeight = thickbobbin
    # Create a Rectangle patch
    bobbintop = patches.Rectangle(bobbintop0,bobbintopWidth,bobbintopHeight,linewidth=0,color='y')
    # Add the patch to the Axes
    ax.add_patch(bobbintop)

    bobbinbottom0 = (xblank+lTtl,yblank+rShellOut-rMag)
    bobbinbottomWidth = -lWireVessel
    bobbinbottomHeight = -thickbobbin
    # Create a Rectangle patch
    bobbinbottom = patches.Rectangle(bobbinbottom0,bobbinbottomWidth,bobbinbottomHeight,linewidth=0,color='y')
    # Add the patch to the Axes
    ax.add_patch(bobbinbottom)

    bobbinright0 = (xblank+lTtl,yblank+thickShell)
    bobbinrightWidth = thickbobbin
    bobbinrightHeight = 2*rShellIn
    # Create a Rectangle patch
    bobbinright = patches.Rectangle(bobbinright0,bobbinrightWidth,bobbinrightHeight,linewidth=0,color='y')
    # Add the patch to the Axes
    ax.add_patch(bobbinright)

    bobbinflantop0 = (xblank+lTtl-lWireVessel,yblank+2*rShellOut-thickShell)
    bobbinflantopWidth = -thickbobbin
    bobbinflantopHeight = -realLayers*rWireOut*2-thickbobbin
    # Create a Rectangle patch
    bobbinflantop = patches.Rectangle(bobbinflantop0,bobbinflantopWidth,bobbinflantopHeight,linewidth=0,color='y')
    # Add the patch to the Axes
    ax.add_patch(bobbinflantop)

    bobbinflanbottom0 = (xblank+lTtl-lWireVessel,yblank+thickShell)
    bobbinflanbottomWidth = -thickbobbin
    bobbinflanbottomHeight = realLayers*rWireOut*2+thickbobbin
    # Create a Rectangle patch
    bobbinflanbottom = patches.Rectangle(bobbinflanbottom0,bobbinflanbottomWidth,bobbinflanbottomHeight,linewidth=0,color='y')
    # Add the patch to the Axes
    ax.add_patch(bobbinflanbottom)

    if verbose:
        print("blank:{}".format((xblank,yblank)))
        print("shell:{}".format((shellTop0,shellTopWidth,shellTopHeight)))
        print("shell:{}".format((shellBottom0,shellBottomWidth,shellBottomHeight)))
        print("shell:{}".format((shellLeft0,shellLeftWidth,shellLeftHeight)))
        print("Mag:{}".format((Mag0,MagWidth,MagHeight)))
        print("Core:{}".format((Core0,CoreWidth,CoreHeight)))
        print("wires1:{}".format((wire0,wireWidth,wireHeight)))
        print("wires2:{}".format((wire2_0,wire2Width,wire2Height)))

    fig.savefig("generatedImages/{}_layers.png".format(name), dpi=300)

    return ax,fig

#%%
for layers in range(int(maxLayers/2)): 
    print("\n********LAYER: {}********".format((layers+1)*2))
    # Don't optimise for wire size
    print("lMag, {} mm".format(optOut[layers][0][0]))
    print("rMag, {} mm".format(optOut[layers][0][1]))
    print("lCore, {} mm".format(optOut[layers][0][2]))
    print("lWireVessel, {} mm".format(optOut[layers][0][3]))
    Q,pIn,massTotal,pInMassTotal,I,V,pressure = motorMain(optOut[layers][0],RHO_WIRE,SIGMA_WIRE,CP_WIRE,BR_Mag,dTravel,fMotor,freq,thickWireWall,(layers+1)*2,deltaT,VISCO_WIRE,rWireIn)
    print("Q, {} ml/s".format(Q*1e6))
    print("Pin, {} W".format(pIn))
    print("massTotal, {} kg".format(massTotal))
    print("Pin*massTotal, {} W*kg".format(pInMassTotal))
    print("I, {} A".format(I))
    print("V, {} V".format(V))
    print("Pressure, {} kPa".format(pressure*1e-3))
    print("rWireIn, {} mm".format(rWireIn))
    print("thickWireWall, {} mm".format(thickWireWall))
    
    with open('graphingLayer{}.pkl'.format(layers),'wb') as f:
        pickle.dump((optOut[layers][0],Q,pIn,I,V,pressure), f)
    
    if draw:
        ax,fig = diagramDraw(optOut[layers][0], layers, thickWireWall,rWireIn,str((layers+1)*2))
        if verbose:
            plt.show()

#%%
layers = 6
print("\nManual")
print("\n********LAYER: {}********".format(layers))
x = (40,20,8,24)
Q,pIn,massTotal,pInMassTotal,I,V,pressure = motorMain(x,RHO_WIRE,SIGMA_WIRE,CP_WIRE,BR_Mag,dTravel,fMotor,freq,thickWireWall,layers,deltaT,VISCO_WIRE,rWireIn)

lMag = x[0]
rMag = x[1]
lCore = x[2]
lWireVessel = x[3]

print("lMag, {} mm".format(lMag))
print("rMag, {} mm".format(rMag))
print("lCore, {} mm".format(lCore))
print("lWireVessel, {} mm".format(lWireVessel))
print("Q, {} ml/s".format(Q*1e6))
print("Pin, {} W".format(pIn))
print("I, {} A".format(I))
print("V, {} V".format(V))
print("Pressure, {} kPa".format(pressure*1e-3))
print("Pin*massTotal, {} W*kg".format(pInMassTotal))
print("total length, {} mm".format(lCore+lMag))
print("rWireIn, {} mm".format(rWireIn))
print("thickWireWall, {} mm".format(thickWireWall))

if draw:
        ax,fig = diagramDraw(x, 2, thickWireWall,rWireIn,'finalDesign')
        if verbose:
            plt.show()