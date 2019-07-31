# %%
import numpy as np
from scipy.optimize import brute
import pickle
from scipy.constants import mu_0

# Relative constants
# Things that aren't easily changed
BR_CORE = 1.2
SIGMA_WIRE = 2.3e6
CP_WIRE = 296
RHO_WIRE = 6440
REL_PERM_STEEL = 2000
REL_PERM_NEO = 1.05
REL_PERM_AIR = 1
thickWireWall = 0.75 # mm

moe = 1e-8

dDrive = 9 # 9mm
fMotor = 9 # 9N
freq = 1 # 1Hz

# %%
def constraintsCheck(x, layers, rho, cp, thickWireWall):
    # print("constraintsCheck, layers {}, x = {}".format(layers,x))
    
    rWireIn = x[0] / 1000
    lCore = x[1] / 1000
    rCore = x[2] / 1000
    lConnector = x[3] / 1000
    thickWireWall = thickWireWall / 1000
    
    rWireOut = rWireIn + thickWireWall # Wire outer radius
    
    rShellIn = rCore+rWireOut*layers # Shell inner radius
    
    lWireTotal = 2*np.pi*(rCore+2*layers*rWireOut-rWireOut)*((lCore+lConnector)/(2*rWireOut)) # Total length of wire

    # 3. Maximum volume of liquid metal in circuit ($ A_{wire} l_{wireTotal} $)
    maxVolume = 15 #ml
    aWire = rWireIn**2*np.pi
    volume = aWire*lWireTotal
    if volume*1e6 > maxVolume:
        print("Volume too large, {} ml, x: {}, layer {}".format(volume*1e6,x,layers))
        return False
    
    # 4. Maximum shell dimensions ($ r_{shellIn} $)
    maxShell = 100 #mm
    if rShellIn*1e3 > maxShell:
        print("Shell Radius too large, {} mm, x: {}, layer {}".format(rShellIn*1e3,x,layers))
        return False
    
    # 5. Maximum temperature change ($ \Delta T $)
    # Needs to be larger than value
    maxTempChange = 80 #degC
    if deltaT > maxTempChange:
        print("Too hot, {} degC, x: {}, layer {}".format(maxTempChange,x,layers))
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
    outSelect = args[10]

    # print("rho:{}, sigma:{}, cp:{}, br:{}, dDrive:{}, fMotor:{}, freq:{}, thickWireWall:{}, layers:{}, deltaT:{}, outSelect:{}".format(rho,sigma,cp,br,dDrive,fMotor,freq,thickWireWall,layers,deltaT,outSelect))
    
    if not constraintsCheck(x, layers, rho, cp, thickWireWall): # Check constraints
        return np.Inf # return some really large number
    # print("Valid! layers {}, x = {}".format(layers,x))

    # print("motorMain: {}".format(x))
    
    rWireIn = x[0] / 1000
    lCore = x[1] / 1000
    rCore = x[2] / 1000
    lConnector = x[3] / 1000
    dDrive = dDrive / 1000
    thickWireWall = thickWireWall / 1000
    
    rWireOut = rWireIn + thickWireWall # Wire outer radius
    
    rShellIn = rCore+rWireOut*layers*2 # Shell inner radius

    aWire = rWireIn**2*np.pi
    lWireTotal = (lConnector+lCore)*np.pi*(layers+1)*layers + (lConnector+lCore)*rCore*np.pi/rWireOut # Total length of wire
    volume = aWire*lWireTotal
    
    resTotal = lWireTotal/np.pi/rWireIn**2/sigma # Total wire resistance
    
    rg = np.log(rShellIn/rCore)/(2*np.pi*lConnector) * REL_PERM_AIR# gap reluctance

    rm = lCore / (np.pi*rCore**2) * REL_PERM_NEO

    rmTtl = rg + rm # Total reluctance - approx. gap reluctance + magnet reluctance
    
    emmf = br*lCore*REL_PERM_NEO # electromagneticmotive force
    
    flux = emmf / rmTtl # Magnetic flux through circuit

    AA = 2*np.pi*lConnector*(rCore-rWireOut+(layers+1)*layers*rWireOut)

    lWireInField = lConnector*np.pi*(layers+1)*layers + lConnector*rCore*np.pi/rWireOut
    
    LAA = 0 # Length of wire in field/AreaAction
    for layer in range(1,layers+1):
        LAA += (2*layer-1)/(2*rCore+4*layer*rWireOut-2*rWireOut)
    
    B = flux / AA

    if B > 2:
        print("B too large: {} H".format(B))
        B = 2

    I = fMotor / B / lWireInField
    print("I: {} A".format(I))

    V = I * resTotal
    print("V: {} V".format(V))

    pIn = I**2 * resTotal
    print("Pin: {} W".format(pIn))

    pMech = 2*dDrive*fMotor*freq # Mechanical power output

    pHeat = pIn - pMech

    Q = pHeat / (RHO_WIRE*deltaT*CP_WIRE)

    if Q <= 0 or pIn <= 0:
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
r_lCore = (5,75) # mm - try not to exceed 10 cm size
r_rCore = (5,80) # mm - try not to exceed 10 cm size
r_lConnector = (1,25) # mm - try to be decently sized
grid = (r_rWireIn,r_lCore,r_rCore,r_lConnector)

thickWireWall = 0.75
layers = 1
deltaT = 60
outSelect = 1 # optimising for min Power

# %%
# def motorMain(x,rho,sigma,cp,br,dDrive,fMotor,freq,thickWireWall,layers,deltaT,outSelect)
optOut = []
maxLayers = 5

for layers in range(1,maxLayers+1):
    print("Layer {}".format(layers))
    arglist = (RHO_WIRE,SIGMA_WIRE,CP_WIRE,BR_CORE,dDrive,fMotor,freq,thickWireWall,layers,deltaT,outSelect)
    optOut.append(brute(motorMain,ranges=grid,args=arglist,Ns=30,full_output=True,disp=True,finish=None))

with open('optOut.pkl','wb') as f:
    pickle.dump(optOut, f)
print("Done")

# with open('optOut.pkl','rb') as f:
#     optOut = pickle.load(f)

for layers in range(maxLayers): 
    print(layers+1)
    print(optOut[layers][0]) 
    print(optOut[layers][1]*1e6)
    print(motorMain(optOut[layers][0],RHO_WIRE,SIGMA_WIRE,CP_WIRE,BR_CORE,dDrive,fMotor,freq,thickWireWall,layers+1,deltaT,outSelect))