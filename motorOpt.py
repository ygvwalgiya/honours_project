# %%
import numpy as np
from scipy.optimize import brute
import pickle
from scipy.constants import mu_0
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys

verbose = False
if "-V" in sys.argv:
    verbose = True
refresh = False
if "-r" in sys.argv:
    refresh = True
draw = False
if "-g" in sys.argv:
    draw = True

# Relative constants
# Things that aren't easily changed
BR_CORE = 1.2
SIGMA_WIRE = 3.46e6
CP_WIRE = 296
RHO_WIRE = 6440
REL_PERM_STEEL = 2000
REL_PERM_NEO = 1.05
REL_PERM_AIR = 1
thickWireWall = 0.75 # mm
VISCO_WIRE = 0.0024 #Pa.s

moe = 1e-8

dDrive = 9 # 9mm
fMotor = 9 # 9N
freq = 1 # 1Hz

# %%
def constraintsCheck(x, layers, rho, cp, thickWireWall,dDrive):
    if verbose:
        print("constraintsCheck, layers {}, x = {}".format(layers,x))
    
    rWireIn = x[0]
    lCore = x[1]
    rCore = x[2]
    lConnector = x[3]
    lWireVessal = x[4]
    thickWireWall = thickWireWall
    dDrive = dDrive
    
    rWireOut = rWireIn + thickWireWall # Wire outer radius
    
    rShellIn = rCore+rWireOut*layers # Shell inner radius
    
    lWireTotal = 2*np.pi*(rCore+2*layers*rWireOut-rWireOut)*((lCore+lConnector)/(2*rWireOut)) # Total length of wire

    # 5. Minimum drive distance
    if dDrive > (lWireVessal-lConnector):
        if verbose:
            print("Not enough drive distance, {} mm v {} mm, x: {}, layer {}".format(dDrive,lConnector-lWireVessal,x,layers))
        return False

    # 6. Wire Vessal needs to be shorter than connector+core
    if lWireVessal > lConnector+lCore:
        if verbose:
            print("Wire Vessal too long, {} mm, x: {}, layer {}".format(lWireVessal,x,layers))
        return False

    # 4. Maximum shell dimensions ($ r_{shellIn} $)
    maxShell = 100 #mm
    if rShellIn*2 > maxShell:
        if verbose:
            print("Shell Radius too large, {} mm, x: {}, layer {}".format(rShellIn,x,layers))
        return False

    # 3. Maximum volume of liquid metal in circuit ($ A_{wire} l_{wireTotal} $)
    maxVolume = 15 #ml
    aWire = rWireIn**2*np.pi
    volume = aWire*lWireTotal
    if volume/1e3 > maxVolume:
        if verbose:
            print("Volume too large, {} ml, x: {}, layer {}".format(volume/1e3,x,layers))
        return False

    return True

# %%
def motorMain(x, *args):

    rho = args[0]
    sigma = args[1]
    cp = args[2]
    br = args[3]
    dDrive = args[4]
    fMotor = args[5]
    freq = args[6]
    thickWireWall = args[7]
    layers = args[8]
    deltaT = args[9]
    visco = args[10]
    outSelect = args[11]

    # print("rho:{}, sigma:{}, cp:{}, br:{}, dDrive:{}, fMotor:{}, freq:{}, thickWireWall:{}, layers:{}, deltaT:{}, outSelect:{}".format(rho,sigma,cp,br,dDrive,fMotor,freq,thickWireWall,layers,deltaT,outSelect))
    
    if not constraintsCheck(x, layers, rho, cp, thickWireWall,dDrive): # Check constraints
        return np.Inf # return some really large number
    # print("Valid! layers {}, x = {}".format(layers,x))

    # print("motorMain: {}".format(x))
    
    rWireIn = x[0] / 1000
    lCore = x[1] / 1000
    rCore = x[2] / 1000
    lConnector = x[3] / 1000
    lWireVessal = x[4] / 1000
    dDrive = dDrive / 1000
    thickWireWall = thickWireWall / 1000
    
    rWireOut = rWireIn + thickWireWall # Wire outer radius
    
    rShellIn = rCore+rWireOut*layers*2 # Shell inner radius

    aWire = rWireIn**2*np.pi
    lWireTotal = lWireVessal*np.pi*(layers+1)*layers + lWireVessal*rCore*np.pi/rWireOut # Total length of wire
    
    resTotal = lWireTotal/aWire/sigma # Total wire resistance
    
    rg = np.log(rShellIn/rCore)/(2*np.pi*lConnector) * REL_PERM_AIR# gap reluctance

    rm = lCore / (np.pi*rCore**2) * REL_PERM_NEO

    rmTtl = rg + rm # Total reluctance - approx. gap reluctance + magnet reluctance
    
    emmf = br*lCore*REL_PERM_NEO # electromagneticmotive force
    
    flux = emmf / rmTtl # Magnetic flux through circuit

    AA = lConnector*2*np.pi*(rCore+layers*rWireOut*2-rWireOut) # approx the largest area layer

    if lWireVessal > lConnector:
        lWireInField = lConnector*np.pi*(layers+1)*layers + lConnector*rCore*np.pi/rWireOut
    else:
        lWireInField = lWireVessal*np.pi*(layers+1)*layers + lWireVessal*rCore*np.pi/rWireOut
    
    # LAA = 0 # Length of wire in field/AreaAction
    # for layer in range(1,layers+1):
    #     LAA += (2*layer-1)/(2*rCore+4*layer*rWireOut-2*rWireOut)
    
    B = flux / AA

    if B > 2:
        if verbose:
            print("B too large: {} H".format(B))
        B = 2

    I = fMotor / B / lWireInField
    if verbose:
        print("I: {} A".format(I))

    V = I * resTotal
    if verbose:
        print("V: {} V".format(V))

    pIn = I**2 * resTotal
    if verbose:
        print("Pin: {} W".format(pIn))

    pMech = 2*dDrive*fMotor*freq # Mechanical power output

    pHeat = pIn - pMech

    Q = pHeat / (RHO_WIRE*deltaT*CP_WIRE)
    if verbose:
        print("Q: {} ml/s".format(Q*1e6))

    pressure = 8*visco*lWireTotal*Q/(np.pi*rCore**4)
    if verbose:
        print("Pressure: {} Pa".format(pressure))

    if Q <= 0 or pIn <= 0:
        if verbose:
            print("Invalid Output! layers {}, x = {}, Q = {}, pIn = {}".format(layers,x,Q,pIn))
        return np.Inf # return some really large number

    # Choose which output variable to optimise for
    if outSelect == 0:
        return Q
    else:
        return pIn

    

# %%
# rWireIn lCore rCore lConnector
r_rWireIn = (0.5,3.5) # mm - reasonable sizes for silicone tube radius
r_lCore = (5,100) # mm - try not to exceed 10 cm size
r_rCore = (5,40) # mm - try not to exceed 10 cm size
r_lConnector = (1,25) # mm - try to be decently sized
r_lWireVessal = (10,100)
grid = (r_rWireIn,r_lCore,r_rCore,r_lConnector,r_lWireVessal)

thickWireWall = 0.75
layers = 1
deltaT = 60
outSelect = 0 # optimising for min Flow

# %%
# def motorMain(x,rho,sigma,cp,br,dDrive,fMotor,freq,thickWireWall,layers,deltaT,outSelect)
optOut = []
maxLayers = 5

#%% DON'T RUN ON JUPYTER
if refresh:
    for layers in range(1,maxLayers+1):
        print("Layer {}".format(layers))
        arglist = (RHO_WIRE,SIGMA_WIRE,CP_WIRE,BR_CORE,dDrive,fMotor,freq,thickWireWall,layers,deltaT,VISCO_WIRE,outSelect)
        optOut.append(brute(motorMain,ranges=grid,args=arglist,Ns=20,full_output=True,disp=True,finish=None))

    with open('optOut.pkl','wb') as f:
        pickle.dump(optOut, f)
    print("Done")

# %%
with open('optOut.pkl','rb') as f:
    optOut = pickle.load(f)

# DIAGRAMMING
def diagramDraw(x,layers,thickWireWall):
    rWireIn = x[0]
    lCore = x[1]
    rCore = x[2]
    lConnector = x[3]
    lWireVessal = x[4]
    thickShell = 5

    rWireOut = rWireIn + thickWireWall # Wire outer radius
    rShellIn = rCore+rWireOut*layers*2 # Shell inner radius
    rShellOut = rShellIn + thickShell
    isize = (200,200)
    lTtl = lConnector+lCore+thickShell

    xblank = (isize[0] - lTtl)/2
    yblank = (isize[1] - 2*rShellOut)/2

    # Create figure and axes
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xlim([0, isize[0]])
    plt.ylim([0, isize[1]])

    # Shell - grey
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
    shellLeftWidth = thickShell
    shellLeftHeight = 2*rShellOut
    shellLeft = patches.Rectangle(shellLeft0,shellLeftWidth,shellLeftHeight,linewidth=0,color='grey')
    ax.add_patch(shellLeft)

    # core - green
    core0 = (xblank+thickShell,yblank+rShellOut-rCore)
    coreWidth = lCore
    coreHeight = 2*rCore
    # Create a Rectangle patch
    core = patches.Rectangle(core0,coreWidth,coreHeight,linewidth=0,color='g')
    # Add the patch to the Axes
    ax.add_patch(core)

    # connector - red
    connector0 = (xblank+thickShell+lCore,yblank+rShellOut-rCore)
    connectorWidth = lConnector
    connectorHeight = 2*rCore
    # Create a Rectangle patch
    connector = patches.Rectangle(connector0,connectorWidth,connectorHeight,color='r')
    # Add the patch to the Axes
    ax.add_patch(connector)

    # Wires - blue
    wire0 = (xblank+thickShell,yblank+rShellOut-rCore-layers*rWireOut*2)
    wireWidth = lWireVessal
    wireHeight = layers*rWireOut*2
    # Create a Rectangle patch
    wire = patches.Rectangle(wire0,wireWidth,wireHeight,color='b')
    # Add the patch to the Axes
    ax.add_patch(wire)

    wire2_0 = (xblank+thickShell,yblank+rShellOut+rCore)
    wire2Width = lWireVessal
    wire2Height = layers*rWireOut*2
    # Create a Rectangle patch
    wire = patches.Rectangle(wire2_0,wire2Width,wire2Height,color='b')
    # Add the patch to the Axes
    ax.add_patch(wire)

    if verbose:
        print("blank:{}".format((xblank,yblank)))
        print("shell:{}".format((shellTop0,shellTopWidth,shellTopHeight)))
        print("shell:{}".format((shellBottom0,shellBottomWidth,shellBottomHeight)))
        print("shell:{}".format((shellLeft0,shellLeftWidth,shellLeftHeight)))
        print("core:{}".format((core0,coreWidth,coreHeight)))
        print("connector:{}".format((connector0,connectorWidth,connectorHeight)))
        print("wires1:{}".format((wire0,wireWidth,wireHeight)))
        print("wires2:{}".format((wire2_0,wire2Width,wire2Height)))

    fig.savefig("generatedImages/{}_layers.png".format(layers+1), dpi=300)

    return ax,fig

#%%
for layers in range(maxLayers): 
    print("\n********LAYER: {}********".format(layers+1))
    print("rWireIn:{} mm".format(optOut[layers][0][0]))
    print("lCore:{} mm".format(optOut[layers][0][1]))
    print("rCore:{} mm".format(optOut[layers][0][2]))
    print("lConnector:{} mm".format(optOut[layers][0][3]))
    print("lWireVessal:{} mm".format(optOut[layers][0][4]))
    print(motorMain(optOut[layers][0],RHO_WIRE,SIGMA_WIRE,CP_WIRE,BR_CORE,dDrive,fMotor,freq,thickWireWall,layers+1,deltaT,VISCO_WIRE,outSelect))
    if draw:
        ax,fig = diagramDraw(optOut[layers][0], layers, thickWireWall)
        plt.show()