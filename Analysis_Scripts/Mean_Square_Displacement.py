#   Script to compute the MSD from a LAMMPS trajectory file
#   NB: Assumes constant volume ensemble, for constant pressure using LAMMPS's in-built displacement function is reccommended
#   The script also assumes a constant dump frequency for the particle positions
import numpy as np


#   Set parameters
lammps_trj = "trajectory.lammpstrj"     #   Path to the LAMMPS trajectory file
Ntau = 1000            #   Maximum lag time to compute (in frames)
md_timestep = 0.005   #   Frequency of dumped frames (in MD timesteps)
Nt0 = 100           #   Number of time origins to average over

tau_spacing = 1             #   Lag time resolution of one dumped frame
t0_spacing = 10              #   Set spacing of time origins to 1, this means they are likely to be correlated but we still use all the data available


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

    data = np.loadtxt(file, max_rows = NPart, usecols=(0,2,3,4,5,6,7))

    return data, [BoxLX,BoxLY,BoxLZ], timestep, NPart

def calc_msd (dataset, tau, time_origins):
    
    #   Calculate the mean squared displacement for a given tau
    #   tau is the number of data points between the two points
    #   time origins should be an array of ints, corresponding to the indicies of the data points we're using as time origins

    #   Calculate the MSD for each dimension
    displacements = dataset[(time_origins+tau),:,:] - dataset[time_origins,:,:]

    #   Calculate overall displacement per particle
    #       Can leave as squared displacements for calculating MSD
    displacements_squared = np.sum(displacements**2, axis=2)


    MSD = np.mean(displacements_squared)

    return MSD

def main(lammps_trj, Ntau, md_timestep, Nt0, tau_spacing, t0_spacing):

    Nframes =  (Ntau * tau_spacing) + (Nt0 * t0_spacing)    #   Calculate the total number of dump frames needed to use the specified max lag time and number of time origins
    
    #   Quickly check to see if the trajectory file actually contains the required number of frames
    with open(lammps_trj, "r") as infile:
        pos, boxL, timestep, NPart = read_frame(infile)
    with open(lammps_trj, "rb") as infile:
        num_lines = sum(1 for _ in infile)
    if num_lines < ((NPart+9)*Nframes):
        raise Exception("Trajectory file is not long enough, check your input numbers make sense!")
    

    output_path = "./MSD.dat"   #   Path to output file

    print("Starting calculation!")
    #   Open the input file so we can parse through it
    with open(lammps_trj, "r") as infile:

        #   initialise timestep array
        #   we use this to determine the spacing between samples
        timesteps = np.empty(Ntau)

        #   Iterate over every frame in Nframes
        for i in range(Nframes):

            #   Read a snapshot from the trajectory file
            pos, boxL, timestep, NPart = read_frame(infile)
            
            #   Sort positions by particle ID to make analysis simpler
            pos = pos[pos[:,0].argsort()]
            #   Unwrap coordinates using the image flags
            flags = pos[:,4:]
            pos = pos[:,1:4]
            pos = pos + (flags * boxL)

            #   Create trajectory array if it doesn't already exist
            try:
                traj
            except NameError:
                traj = np.empty((Nframes,NPart,3))

            #   Add current frame to the trajectory
            traj[i,:,:] = pos

            if i < Ntau: 
                #   Add timestep information to the timesteps array
                timesteps[i] = timestep
        
        #   Initialise the MSD array
        MSD = np.empty(((Ntau-1),2))
        #   Calculate the lag times and add them to the first column of the MSD array
        #   NB: We use [1:] to skip the lag time of 0
        MSD[:,0] = (timesteps[1:] - timesteps[0]) * md_timestep
        #   Calculate the array indexes for the time orignins
        time_orgins = np.arange(0, (Nt0 * t0_spacing), t0_spacing)

        #   Iterate over every lag time, and calculate the MSD
        for i in range(Ntau-1):
            #   Note the i+1 here to avoid a lag time of zero, this will by definition have a MSD of 0 and causes problems with log-log analysis
            MSD[i,1] = calc_msd(traj, ((i+1)*tau_spacing), time_orgins)
    
    #   Write the lag times and mean squared displacements to a text file
    np.savetxt(output_path, MSD)


                    

if __name__ == '__main__':
    main(lammps_trj, Ntau, md_timestep, Nt0, tau_spacing, t0_spacing)