#   Script to compute a G(r) from a LAMMPS trajectory
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

lammps_trj = "trajectory.lammpstrj"    #   Path to LAMMPS dump file
Nframes = 2000                  #   Number of frames to read from the dump file
Nbins = 200                      #   Number of bins to use in the histogram
pbc = True                      #   Set wether to implement periodic boundary conditions

frame_spacing = 1              #   Set spacing between frames we use for computing g(r)
output_path = "./Gr.dat"        #   Set the path to the output file for the raw data
output_image = "./Gr.png"       #   Set path to the output file for the plotted g(r) (using Matplotlib)

def read_box(box_string):
    #   Function to convert the box line in a LAMMPS file into a float for the box length

    BoxL = box_string.replace("\n","")
    BoxL = BoxL.split(" ")
    BoxL = float(BoxL[1]) - float(BoxL[0])

    return BoxL

def read_frame(file):
    #   Function that reads a single frame from a LAMMPS dump file

    #   Pass through the header, extract the box dimension (assumed cubic box)
    file.readline()
    timestep = file.readline().replace("\n","")
    file.readline()
    NPart = int(file.readline())
    file.readline()
    BoxLX = file.readline()
    BoxLY = file.readline()
    BoxLZ = file.readline()
    file.readline()

    BoxLX = read_box(BoxLX)
    BoxLY = read_box(BoxLY)
    BoxLZ = read_box(BoxLZ)

    data = np.loadtxt(file, max_rows = NPart, usecols=(0,2,3,4))

    return data, [BoxLX,BoxLY,BoxLZ], timestep, NPart

def main(lammps_trj, Nframes, Nbins, pbc, frame_spacing, output_path, output_image):

    #   Open the input file so we can parse through it
    with open(lammps_trj, "r") as infile:
        #   Iterate over every frame in Nframes
        for _ in range(Nframes):

            #   Read a snapshot from the trajectory file
            pos, boxL, timestep, NPart = read_frame(infile)

            #   Skip the next X frames so that each snapshot is independant
            for _ in range(frame_spacing-1):
                read_frame(infile)

            #   if the histograms bin edges haven't been defined already, compute them
            try:
                bins
            except NameError:
                #   Define Nbins bins for the histogram, going from a seperation of 0 to half the smallest box dimension
                bins = np.linspace(0, (min(boxL)/2), num = Nbins+1)
                #   Also initialise the g(r) array
                g_r = np.zeros(Nbins)
            
            #   Sort positions by particle ID to make analysis simpler
            pos = pos[pos[:,0].argsort()]

            #   Iterate over every particle in the system
            for i in range(NPart):
                #   First delete the ith position (to prevent finding distaces of 0)
                #   Then find the x,y,z distance of every jth particle from the ith particle
                distances = np.delete(pos[:,1:], i, 0) - pos[i,1:]

                #   Apply the minimum image convention
                if pbc:
                    distances[:,0] = distances[:,0] - (np.round(distances[:,0]/boxL[0])*boxL[0])
                    distances[:,1] = distances[:,1] - (np.round(distances[:,1]/boxL[1])*boxL[1])
                    distances[:,2] = distances[:,2] - (np.round(distances[:,2]/boxL[2])*boxL[2])
                
                #   Compute the Euclidean distance (i.e. apply Pythagoras)
                distances = np.sqrt(np.sum(np.square(distances), axis = 1))

                #   Bin the distances to a histogram
                hist, bins = np.histogram(distances, bins=bins)
                g_r += hist
            
            #   Divide through by NPart to account for the average over every particle
            g_r = g_r / NPart

            #   Normalise the G(r) by the ideal gas G(r)
            density = NPart / (boxL[0]*boxL[1]*boxL[2])
            volume = (4/3)*math.pi * (np.power(bins[1:],3) - np.power(bins[:-1],3))
             
            #   Initialise the averaged g(r) array
            try:
                average_g_r
            except NameError:
                average_g_r = np.zeros(len(g_r))
            #   Add the g(r) from the current frame to the running average
            average_g_r += g_r / (volume * density)

        #   Normalise the final g(r) by the number of frames considered
        average_g_r = average_g_r / Nframes

        #   Use matplotlib to plot the g(r) and save as a png
        matplotlib.rcParams.update({'font.size': 22})
        px = 1/plt.rcParams['figure.dpi']  # pixel in inches
        fig, ax = plt.subplots(figsize=(1338*px, 761*px))
        ax.plot((bins[1:]-((bins[1]-bins[0])/2)), average_g_r)
        ax.set_ylabel("$g$($r$)")
        ax.set_xlabel(u"$r$ /$\u03C3$")
        data = np.empty((len(average_g_r),2))
        data[:,0] = (bins[1:]-((bins[1]-bins[0])/2))
        data[:,1] = average_g_r

        #   Save the g(r) data to a text file
        np.savetxt(output_path, data)
        #   Save the g(r) plot as a png
        plt.savefig(output_image)

        plt.show()


                    

if __name__ == '__main__':
    main(lammps_trj, Nframes, Nbins, pbc, frame_spacing, output_path, output_image)