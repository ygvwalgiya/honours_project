'''
Jason Guan 892594255
Part IV Project
This script finds the most optimal (minimum mass*power use) design of motor.
The script uses gridsearch, and produces a seperate result for all even number of layers of wire in the bobbin up to a limit.
The script can be used to generate cross-sectional diagrams of motor designs corresponding to all output results.
'''

# %% Import statements
import numpy as np
from scipy.optimize import brute
import pickle
from scipy.constants import mu_0
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
from numba import jit

# verbose command for debugging
verbose = False
if "-V" in sys.argv:
    verbose = True

# refresh command for re-running optimisation algorithm
# used on the Mac as it takes a few minutes to run
# used for debugging and getting quick results/drawings
refresh = False
if "-r" in sys.argv:
    refresh = True

# draw command for producting cross-section diagrams of the motor corresponding to output parameters
# used for sanity checks
draw = False
if "-g" in sys.argv:
    draw = True

# Relative constants
# Things that aren't easily changed
BR_Mag = 1.35 #1.35-1.4 Remanence magnetic field density of magnet
RES_WIRE = 29.4e-8 # Resistivity of eGaIn
CP_WIRE = 296 # Specific heat of eGaIn
RHO_WIRE = 6440 # kg/m3 density of eGaIn
REL_PERM_STEEL = 2000 # relative magnetic permiability of steel
REL_PERM_NEO = 1.05 # relative magnetic permiability of Nd magnet
REL_PERM_AIR = 1 # relative magnetic permiability of air
VISCO_WIRE = 0.0024 #Pa.s viscosity of eGaIn
RHO_TUBE = 1100 # kg/m3 density of silicone tubing
RHO_IRON = 7850 # kg/m3 density of shell
RHO_MAG = 7010 # kg/m3 density of magnet

# Used for optimisation - things that are invalid yeild a really large output, so they're discarded in the minimisation.
BIG_NUM = 1e12

# Margin of error
moe = 1e-8

# Wire parameters - there are only so many combinations for off-the-shelf solutions. 
# This particular one has the best fill factor.
thickWireWall = 0.5 # mm
rWireIn = 1 # mm

# Design requirements re: pump.
dTravel = 15 # 9mm
fMotor = 9 # 9N
freq = 1 # 1Hz

# Maximum temperature change - health and safety constraint
deltaT = 30

# Maximum number of layers optimised for.
maxLayers = 14

# %%
# rWireIn lMag rMag lCore
# r_rWireIn = (0.05,2.5) # mm - reasonable sizes for silicone tube radius
r_lMag = (5,95) # mm - try not to exceed 10 cm size - magnet length
r_rMag = (1,20) # mm - try not to exceed 40 mm diameter - health and safety hazard - magnet radius
r_lCore = (1,25) # mm - try to be decently sized - core length. Core radius is the same as rMag
r_lWireVessel = (5,95) # Wire vessel length - the part of the bobbin that the wire goes around.
grid = (r_lMag,r_rMag,r_lCore,r_lWireVessel) # Gridsearch init
outSelect = 1 # optimising for min power * mass

# %% This is the numba-optimised version of motorOpt
# motorOpt takes the input variables (x) and the optimisation constants (*args), 
# and outputs the optimisation variable (mass*power)
# Numba is a high-performance Python to LLVM bytecode compiler.
# Numba doesn't allow print statements, hence the need for another version of this function below, 
# and the extensive commenting-out in this function.
@jit(nopython=True)
def motorOpt(x, *args):

    # Unpack input variables
    rho = args[0]
    RES_WIRE = args[1]
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

    # Convert from mm to m
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

    aWire = rWireIn**2*np.pi # Area of wire in field
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

    resTotal = lWireTotal/aWire*RES_WIRE/4 # Total wire resistance
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

    # If wire vessel is longer than core, length of wire in field 
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

    maxI = 10
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

# This is the non-numba optimised version of motorOpt.
# Used for debugging mostly. Does the same thing, just with print statements everywhere.
# Outputs everything instead of just the optimisation variable, for the graphing function to use.
def motorMain(x, *args):
    rho = args[0]
    RES_WIRE = args[1]
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
        print("rShellIn, {}, mm".format(rShellIn*1e3))

    aWire = rWireIn**2*np.pi
    #lWireTotal = lWireVessel*np.pi*(layers+1)*layers + lWireVessel*rMag*np.pi/rWireOut # Total length of wire
    lWireTotal = 0
    wireCycle = lWireVessel/(rWireOut*2)
    for i in range(1,layers+1):
        rLayer = (i*rWireOut*2)+rMag+rWireOut
        lWireTotal += 2*rLayer*np.pi * wireCycle
#        print(lWireTotal)

    if verbose:
        print("lWireTotal, {}, m".format(lWireTotal))

    resTotal = lWireTotal/aWire * RES_WIRE/4 # Total wire resistance
    if verbose:
        print("resTotal, {}, Ohm".format(resTotal))

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
        print("B, {}, H".format(B))

    I = fMotor / B / lWireInField * 2
    if verbose:
        print("I, {}, A".format(I))

    V = I * resTotal
    if verbose:
        print("V, {}, V".format(V))

    pIn = I**2 * resTotal
    if verbose:
        print("Pin, {}, W".format(pIn))

    pMech = 2*dTravel*fMotor*freq # Mechanical power output

    pHeat = pIn - pMech

    Q = pHeat / (RHO_WIRE*deltaT*CP_WIRE)
    if verbose:
        print("Q, {}, ml/s".format(Q*1e6))

    pressure = 8*visco*lWireTotal*Q/(np.pi*rWireIn**4)
    if verbose:
        print("Pressure, {}, Pa".format(pressure))

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

    return Q,pIn,massTotal,pIn * massTotal,I,V,pressure, massTotal

# %% DON'T RUN ON JUPYTER - will crash the notebook
# def motorOpt(x,rho,sigma,cp,br,dTravel,fMotor,freq,thickWireWall,layers,deltaT,outSelect)
# This is the block of code that does the actual optimisation
optOut = []
if refresh:
    # For every even number of wire layers, determine optimal design.
    for layers in range(2,maxLayers+1,2):
        print("Layer {}".format(layers))
        arglist = (RHO_WIRE,RES_WIRE,CP_WIRE,BR_Mag,dTravel,fMotor,freq,thickWireWall,layers,deltaT,VISCO_WIRE,outSelect,rWireIn)
        optOut.append(brute(motorOpt,ranges=grid,args=arglist,Ns=25,full_output=True,disp=True,finish=None))

    # Save outputs into pickle file for later
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

    fig.savefig("images/{}_layers.png".format(name), dpi=300)

    return ax,fig

#%%
# This is the print section to see the numerical optimisation outputs.
for layers in range(int(maxLayers/2)):
    print("\n********LAYER: {}********".format((layers+1)*2))
    # Don't optimise for wire size
    print("lMag, {}, mm".format(optOut[layers][0][0]))
    print("rMag, {}, mm".format(optOut[layers][0][1]))
    print("lCore, {}, mm".format(optOut[layers][0][2]))
    print("lWireVessel, {}, mm".format(optOut[layers][0][3]))
    Q,pIn,massTotal,pInMassTotal,I,V,pressure,massTotal = motorMain(optOut[layers][0],RHO_WIRE,RES_WIRE,CP_WIRE,BR_Mag,dTravel,fMotor,freq,thickWireWall,(layers+1)*2,deltaT,VISCO_WIRE,rWireIn)
    print("Q, {}, ml/s".format(Q*1e6))
    print("Pin, {}, W".format(pIn))
    print("massTotal, {}, kg".format(massTotal))
    print("Pin*massTotal, {}, W*kg".format(pInMassTotal))
    print("I, {}, A".format(I))
    print("V, {}, V".format(V))
    print("Pressure, {}, kPa".format(pressure*1e-3))
    print("rWireIn, {}, mm".format(rWireIn))
    print("thickWireWall, {}, mm".format(thickWireWall))
    print("massTotal, {}, kg".format(massTotal))

    with open('graphingLayer{}.pkl'.format(layers),'wb') as f:
        pickle.dump((optOut[layers][0],Q,pIn,I,V,pressure), f)

    if draw:
        ax,fig = diagramDraw(optOut[layers][0], layers, thickWireWall,rWireIn,str((layers+1)*2))
        if verbose:
            plt.show()

#%%
# The manual input/output section to verify slight changes to the optimal output for practical reasons 
# e.g. magnets only come in 10 mm long increments so 37.5 mm length doesn't work
# This function would be used to test the 40 mm design and see if it still fits the requirements
layers = 6
print("\nManual")
print("\n********LAYER: {}********".format(layers))
x = (40,20,8,24)
Q,pIn,massTotal,pInMassTotal,I,V,pressure,massTotal = motorMain(x,RHO_WIRE,RES_WIRE,CP_WIRE,BR_Mag,dTravel,fMotor,freq,thickWireWall,layers,deltaT,VISCO_WIRE,rWireIn)

lMag = x[0]
rMag = x[1]
lCore = x[2]
lWireVessel = x[3]

print("lMag, {}, mm".format(lMag))
print("rMag, {}, mm".format(rMag))
print("lCore, {}, mm".format(lCore))
print("lWireVessel, {}, mm".format(lWireVessel))
print("Q, {}, ml/s".format(Q*1e6))
print("Pin, {}, W".format(pIn))
print("I, {}, A".format(I))
print("V, {}, V".format(V))
print("Pressure, {}, kPa".format(pressure*1e-3))
print("Pin*massTotal, {}, W*kg".format(pInMassTotal))
print("total length, {}, mm".format(lCore+lMag))
print("rWireIn, {}, mm".format(rWireIn))
print("thickWireWall, {}, mm".format(thickWireWall))
print("massTotal, {}, kg".format(massTotal))

if draw:
        ax,fig = diagramDraw(x, 2, thickWireWall,rWireIn,'finalDesign')
        if verbose:
            plt.show()