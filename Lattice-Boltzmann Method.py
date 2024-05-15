import numpy as np
from matplotlib import pyplot

plot_every = 100


def distance(x1,y1,x2,y2):
    return np.sqrt((x2-x1)**2 +(y2-y1)**2)

def main():
    
    pic=1
    
    #defining Constants
    
    
    Nx=400                    #Amount of cells in x and y directions
    Ny=100
    
    tau=0.53                  #Kinematic Viscosity
    Nt=30000                  #Number of iterations
    
    
    
    #lattice speeds / weights
    
    
    NL=9                                                                  #9 possible velocity
    
    cxs=np.array([0, 0, 1, 1, 1, 0, -1, -1, -1])                          #discrete velocity
    cys=np.array([0, 1, 1, 0, -1, -1, -1, 0, 1])
    
    
    
    weights=np.array([4/9, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36])   #Weights associated to each lattice site
    
    
    #initial Conditions
    
    
    F=np.ones([Ny,Nx,NL]) + 0.01 * np.random.randn(Ny,Nx,NL)     #assigning velocities with some randomness
    
    F[:, :, 3] = 2.3                                             #assigning velocity to the lattice site in east side
    
    
     
    cylinder = np.full((Ny,Nx), False)                           #defining empty flow domain
    
    
        
    for y in range(0,Ny):                                        #defining position and size of the obstruction
        for x in range(0,Nx):  
            if(distance(Nx//4, Ny//2, x, y)<13):
                cylinder[y][x] = True
                  
               
    
    #main loop
    
    
    for it in range(Nt):                                                 #Iterating through time
        print(it)
        
        
        F[:, -1, [6, 7, 8]] = F[:, -2, [6, 7, 8]]                        #zou-he boundary condition
        F[:, 0, [2, 3, 4]] = F[:, 1, [2, 3, 4]]
    
        
        
        for i, cx, cy in zip(range(NL), cxs, cys):                       #Drift
            F[:, :, i] = np.roll(F[: , :, i], cx, axis=1)
            F[:, :, i] = np.roll(F[: , :, i], cy, axis=0)
            
            
        
        bndryF = F[cylinder, :]                                         # Set reflective boundaries
        bndryF = bndryF[:, [0, 5, 6, 7, 8, 1, 2, 3, 4]]
        
        
        
        # Fluid Varibles
        
        
        rho = np.sum(F,2)
        ux = np.sum(F * cxs, 2) / rho
        uy = np.sum(F * cys, 2) / rho
        
        F[cylinder, :] = bndryF                                       #velocities inside cylinder to 0
        ux[cylinder] = 0
        uy[cylinder] = 0
        
          
        
        #collision
        
        Feq = np.zeros(F.shape)                                       #equillibrium equation
        for i, cx, cy, w in zip(range(NL), cxs, cys, weights):
            
            Feq[:, :, i]= rho * w * ( 1 +3 * (cx*ux + cy*uy) + 9 * (cx*ux + cy*uy)**2 / 2 - 3 * (ux**2 + uy**2)/2)
            
        F = F + -(1/tau) * (F-Feq)
            
                       
        if(it%plot_every == 0):                                       #plotting only every n iteration
            
            dfydx = ux[2:, 1:-1] - ux[0:-2, 1:-1]
            dfxdy = uy[1:-1, 2:] - uy[1:-1, 0:-2]
            curl = dfydx - dfxdy
            
           
            pyplot.imshow(curl, cmap="bwr")
            pyplot.gca().add_patch(pyplot.Circle((Nx//4,Ny//2),13, color="black"))
            #pyplot.colorbar().set_label("Vorticity Magnitude")
            #pyplot.imshow(np.sqrt(ux**2 + uy**2))
            pyplot.savefig( "Lattice-Boltzmann-" + str(pic))           #saves pictures in the same directory
            pyplot.pause(0.01)
            pyplot.cla()
            pic=pic+1
            
                   
    
if __name__ == "__main__":
    main()
