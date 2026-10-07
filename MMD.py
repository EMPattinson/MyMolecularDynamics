#   My Molecular Dynamics is a script that performs basic molecular dynamics simulations in 2D/3D

import numpy as np
import math
import random

global boltzmann_constant, mass, sigma, epsilon, cutoff, cutoff_squared, Npart, ndim, box, timestep, timestep2, tail_correction, thermostat_frequency, init_file, temperature, num_equilibration, num_production, dump_frequency, seed, outfile, log_file

##################################################
#   Key parameters - change these!
##################################################

#   System parameters
Npart = 500                                     #   number of particles
cutoff = 3.0                                    #   cut off for the LJ potential
density = 0.776                                 #   number density
ndim = 3                                        #   dimensionality of the system
timestep = 0.005                                #   MD timestep
temperature = 0.850                             #   initial temperature of the system

box_length = (Npart / density) ** (1/ndim)      #   determine the box lengths corresponding to input parameters

#   Simulation parameters
num_equilibration = 100                         #   number of equilibration (NVT) steps to perform
num_production = 200                            #   number of production (NVE) steps to perform
dump_frequency = 10                             #   how often to print configurations, in MD timesteps
thermo_frequency = 1                            #   how often to print thermodynamic quantities, in MD timesteps

thermostat_frequency = 1                        #   how often to rescale velocities during euqilibration

lattice = False                                 #   if True, generate a configuration using an FCC lattice.
                                                #   if False, read an input configuration from a file
init_file = "initial_config.conf"               #   name of the file containing an input configuration

seed = 618131849621                             #   seed to use for the random number generator, used in debugging
print("Seed for simulation is : ", seed)

##################################################
#   Other parameters - Be careful when changing these!
##################################################

#   Define scientific constants
boltzmann_constant = 1  #   We set kB = 1 to be working in reduced units of temperature (which makes the numbers much more managable)

#   Define potential parameters
mass = 1
sigma = 1
epsilon = 1
tail_correction = True

#   Define the seed for the RNG and the names of the output files
outfile = "trajectory.lammpstrj"
log_file = "MMD.csv"

# The command that runs the simulation is a function called "md_simulation()" and is called at the end of this file
#   This is because we need to define all of the functions used to run the simulation before we can run it.

##################################################
#   Functions
##################################################

#   A single MD timestep
def progress_MD(positions, velocities, accelerations):

    #   Update positions using:
    #       dr = v*dt + (1/2)*a*(dt^2)
    positions += (velocities*timestep) + ((1/2) * accelerations * timestep2)

    #   Update the velocities using the formula:
    #       dv(t + dt) = (1/2)*a(t)*dt + (1/2)*a(t+dt)*dt
    
    #   Add the first term: (1/2)*a(t)*dt
    velocities += (1/2) * accelerations * timestep

    #   Update accelerations using the updated positions to get a(t+dt)
    accelerations = compute_accelerations(positions)

    #   Add the second term: (1/2)*a(t+dt)*dt
    velocities += (1/2) * accelerations * timestep

    return positions, velocities, accelerations

#   Take a set of vectors and find the minimum image equivalent of those vectors
#       Wikipedia has a really good article covering this! https://en.wikipedia.org/wiki/Periodic_boundary_conditions
def minimum_image(vectors, box):

    #   In each dimension the minimum image convention is:
    #       dx = dx - (nearbyint(dx / boxL_x) * boxL_x)

    minimum_image_vectors = vectors - (np.rint(vectors / box[None,:]) * box[None,:])

    return minimum_image_vectors

#   Compute the truncated Lennard-Jones energy for a particle from a set of interparticle distances
def lennard_jones_energy(distances):

    #   Compute the magnitude^2 of the distance vectors
    # r2 = (distances[:,0] ** 2) + (distances[:,1] ** 2) + (distances[:,2] ** 2)
    #   A more compact version of this line is
    r2 = np.sum(distances ** 2, axis = 1)

    #   Check which of these distances fall outside the distance cut-off and ignore them
    cutoff_mask = np.less_equal(r2, cutoff_squared)
    r2 = r2[cutoff_mask]

    #   Compute (1/r)^6 as a faster way of computing the LJ energy
    r_neg2 = 1 / r2
    r_neg6 = r_neg2 * r_neg2 * r_neg2

    #   The Lennard-Jones potential has the form:
    #       U = (4*epsilon) * [(sigma/r)^12 - (sigma/r)^6]
    #   [WIP] for now we work in reduced units so epsilon = 1 and sigma = 1
    energy = 4 * ((r_neg6 * r_neg6) - r_neg6)

    total_energy = np.sum(energy)
    
    return total_energy

#   Compute the truncated Lennard-Jones forces for a particle, from a set of distance vectors
#       we return a vector of x,y,z forces to be used in acceleration calculations
def lennard_jones_force(distances): # !!! WIP - check if this is right, I believe we need to seperate magnitude and direction. but I'm not 100% sure what the best way to handle that is 
    
    #   Compute the magnitude^2 of the distance vectors
    # r2 = (distances[:,0] ** 2) + (distances[:,1] ** 2) + (distances[:,2] ** 2)
    #   A more compact version of this line is
    r2 = np.sum(distances ** 2, axis = 1)

    #   Check which of these distances fall outside the distance cut-off and ignore them
    cutoff_mask = np.less_equal(r2, cutoff_squared)
    r2 = r2[cutoff_mask]
    distances = distances[cutoff_mask,:]

    #   Compute (1/r)^6 as a faster way of computing the LJ energy
    r_neg2 = 1 / r2
    r_neg6 = r_neg2 * r_neg2 * r_neg2

    #   The force enacted on a particle is equal to the gradient of the energy potential
    #   So, the force for a Lennard-Jones potential is given by:
    #       F = (-48*epsilon) * [(sigma/r)^14 - (1/2)(sigma/r)^8]
    forces = 48 * ((r_neg6 * r_neg6 * r_neg2) - (0.5)*(r_neg6 * r_neg2))

    #   Multiply the magnitude of the forces by normalised distance vectors to get a force vector with appropriate direction
    forces = forces[:,None] * distances

    #   Sum the forces in the x,y and z directions
    forces = np.sum(forces, axis=0)

    return forces


#   Function that computes the Maxwell-Boltzmann distribution for a set of given points at a defined temperature T
def maxwell_boltzmann(x, T):
    return (4*math.pi) * ((mass / (2 * math.pi * boltzmann_constant * T)) ** (3/2)) * (x ** 2) * np.exp(-((mass * (x**2))/(2 * boltzmann_constant * T)))

#   Initialise a face centred cubic lattice
def init_positions_fcc(Npart, box):

    #   Ensure the box is a numpy array
    box = np.array(box)

    #   Determine the smallest cubic number greater than Npart/4
    #       This is used as the unit cell of an fcc lattice contains 4 particles
    num_unit_cells = math.ceil(abs(Npart/4) ** (1/3))

    #   Define the atomic positions of the unit cell, using an arbitrary lattice vector of 1
    unit_cell = np.array([[0  , 0  , 0  ],
                          [1/2, 1/2, 0  ],
                          [1/2, 0  , 1/2],
                          [0  , 1/2, 1/2] ])

    pos = np.empty((((num_unit_cells*4) ** 3), 3))
    #   Fill in the lattice using a loop over box dimensions
    if np.shape(box) == (3,):
        i = 0
        for nX in range(num_unit_cells):
            for nY in range(num_unit_cells):
                for nZ in range(num_unit_cells):
                    pos[(i*4):((i+1)*4),:] =  unit_cell + np.array([[nX,nY,nZ]])
                    i += 1
    else:
        print("Error encountered with box dimensions in ffc lattice initalisation. Only 3D systems allowed.")
        exit()

    # Convert these to "real" coordinates by scalling the lattice coordinates by the box dimensions
    pos = pos[:Npart] * (box / np.array([num_unit_cells])[:,None])

    return pos

#   Initialise random velocities, drawn from a Boltzmann distribution and then scaled to a temperature T
def init_velocities_random(target_T):

    # Initialise an array to store the velocities, with random numbers between 0 and 1 in each entry

    velocities = rng.random((Npart, ndim))
    #   Velocities are drawn from values 0 to 1, we subtract 0.5 in each dimension so they can now vary from -0.5 to 0.5
    velocities = velocities - (np.ones(ndim) * 0.5)

    # Normalise these velocities so they have a magnitude of 1
    # These can then be used as directional vectors that we add a magnitude to drawn from the desired distribution
    velocities = velocities / np.linalg.norm(velocities, axis=1)[:,None]
    
    # Draw the velocity magnitudes from a Boltzmann distribution
    dist = np.arange(0,1,0.0001, dtype=float)
    probabilities = maxwell_boltzmann(dist, target_T)
    probabilities = probabilities / np.sum(probabilities)
    vel_magnitudes = rng.choice(dist, size= Npart, p = probabilities)

    #   Multiply the velocity directions by their magnitudes
    velocities = velocities * vel_magnitudes[:,None]

    #   Make sure the overall momentum of the system is 0 (this avoids the flying ice cube effect)
    #   Below is a condensed version of the following, repeated in each dimension
    #       overall_vel_x = np.sum(velocities[:,0])
    #       velocities[:,0] = velocities[:,0] - (overall_vel_x / Npart)
    velocities -= np.sum(velocities, axis = 0) / Npart

    #   Rescale the velocities to a target temperature
    velocities = velocity_rescaling(velocities, target_T)

    return velocities

#   Load a configuration from a LAMMPS format dump file
def load_LAMMPS(input_file):

    with open(input_file, "r") as file:
        file.readline()
        file.readline()
        Num_particles = int(file.readline().split()[0])
        if Num_particles != Npart:
            print("Error, number of particles specified does not match number of particles in the input file")
            exit()
        file.readline()
        file.readline()
        box_x_str = file.readline().split()[0:2]
        box_y_str = file.readline().split()[0:2]
        box_z_str = file.readline().split()[0:2]
        box = np.array([ (float(box_x_str[1]) - float(box_x_str[0])) , (float(box_y_str[1]) - float(box_y_str[0])) , (float(box_z_str[1]) - float(box_z_str[0])) ])
        file.readline()
        file.readline()
        file.readline()
        mass = float(file.readline().split()[0])
        file.readline()
        file.readline()
        file.readline()
        #   Read in atomic positions and image flags
        pos_and_flags = np.loadtxt(file, max_rows= Npart, usecols=(2,3,4,5,6,7))
        positions = pos_and_flags[:,0:3]
        image_flags = pos_and_flags[:,3:].astype(int)
        file.readline()
        file.readline()
        file.readline()
        #   Read in velocities
        velocities = np.loadtxt(file, max_rows= Npart, usecols=(1,2,3))

    return positions, image_flags, velocities

#   Function to rescale velocities from a computed instantaneous temperature to a target temperature
def velocity_rescaling(velocities, target_T):
    #   Compute the instantaneous temperature of the sample
    #   Compute the total kinetic energies using the equation:
    #       Ek = (1/2) * m * (v^2)
    kinetic_energy = (1/2) * mass * np.sum(np.sum(np.square(velocities), axis=1))
    #   Compute the instantaneous temperature using the equipartition theorem given by the equation:
    #       <Ek> = (3N/2)*kB*T
    temperature = (2*kinetic_energy) / (3 * boltzmann_constant * Npart)

    #   Scale the velocities based on the ratio (computed T/ target T)
    velocities = velocities * math.sqrt(target_T / temperature)

    return velocities

#   Compute forces using a Lennard-Jones potential and from this compute accelerations
def compute_accelerations(positions):

    accelerations = np.empty(np.shape(positions))

    #   Iterate over every particle
    for particle_i in range(Npart):
        #   Determine the distance vectors between i and all other particles in the system j
        distances = positions[particle_i,:] - np.delete(positions,particle_i, axis=0) #   WIP - rewrite to use masked arrays which run faster)
        #   Apply the minimum image convention
        distances = minimum_image(distances, box)
        #   Compute the forces
        force = lennard_jones_force(distances)

        accelerations[particle_i,:] = force / mass

    return accelerations

#   Compute various thermodynamic properties including: Energy and Temperature
#       Prints output to a log file
def compute_thermodynamics(positions, velocities, current_timestep, log_file):

    #   Compute potential energy
    potential_energy = 0
    for particle_i in range(Npart):
        distances = positions[particle_i,:] - positions[(particle_i+1):,:]      # by only looking at pairs of particles i and j where j > i we avoid double counting!
        distances = minimum_image(distances, box)
        potential_energy += lennard_jones_energy(distances)

    #   Normalise by number of particles
    Ep_per_particle = potential_energy / Npart

    #   Adding a tail correction
    #   Compute the tail correction for the truncated Lennard-Jones potential and add this to the overall energy
    if tail_correction == True:
        long_range_correction = (8/3) * math.pi * (Npart/np.prod(box)) * epsilon * (sigma**3) * (((1/3) * ((sigma / cutoff) ** 9)) - ((sigma / cutoff) ** 3) )
        Ep_per_particle += (long_range_correction)

    #   Compute the total kinetic energies using the equation:
    #       Ek = (1/2) * m * (v^2)
    #   Note that we use reduced units so the mass of the particles is 1m
    kinetic_energy = (1/2) * mass * np.sum(np.sum(np.square(velocities), axis=1))

    #   Normalise by the number of particles
    Ek_per_particle = kinetic_energy / Npart

    #   Compute the instantaneous temperature using the equipartition theorem given by the equation:
    #       <Ek> = (3N/2)*kB*T
    temperature = (2*Ek_per_particle) / (3 * boltzmann_constant)

    #   Compute total energy
    E_total = Ek_per_particle + Ep_per_particle

    #   Compute the system density (this should always be constant!!!)
    density = Npart / np.prod(box)

    with open(log_file, "a") as f:
        f.write(str(current_timestep)+", "+str(temperature)+", "+str(density)+", "+str(E_total)+", "+str(Ek_per_particle)+", "+str(Ep_per_particle)+"\n")



#   Function that prints the current positions and velocities in a LAMMPS style dump format
def dump_frame(positions, image_flags, velocities, out_file, frame, Npart, box_bounds):

    with open(out_file, "a") as f:
        f.write("ITEM: TIMESTEP\n")
        f.write(str(frame)+"\n")
        f.write("ITEM: NUMBER OF ATOMS\n")
        f.write(str(Npart)+"\n")
        f.write("ITEM: BOX BOUNDS pp pp pp\n")
        np.savetxt(f, box_bounds)
        f.write("ITEM: ATOMS id type x y z ix iy iz vx vy vz\n") 

        for i in range(Npart):
            f.write(str(i)+" 1 "+str(positions[i,0])+" "+str(positions[i,1])+" "+str(positions[i,2])+" "
                +str(image_flags[i,0])+" "+str(image_flags[i,1])+" "+str(image_flags[i,2])+" "
                +str(velocities[i,0])+" "+str(velocities[i,1])+" "+str(velocities[i,2])+"\n")

    print("frame sucessfully dumped at timestep: ", frame)

def dump_restart(positions, image_flags, velocities, out_file, frame, Npart, box_bounds):
    with open(out_file, "w") as f:
            f.write("LAMMPS data file via write_data, version 23 Jun 2022, timestep = 10000000\n"+
                    "\n"+
                    str(Npart)+" atoms\n"+
                    "1 atom types\n"+
                    "\n"+
                    str(box_bounds[0,0])+" "+str(box_bounds[0,1])+" xlo xhi\n"+
                    str(box_bounds[1,0])+" "+str(box_bounds[1,1])+" ylo yhi\n"+
                    str(box_bounds[2,0])+" "+str(box_bounds[2,1])+" zlo zhi\n"+
                    "\n"+
                    "Masses\n"+
                    "\n"+
                    "1 1\n"+
                    "\n"+
                    "Atoms # atomic\n"+
                    "\n")
            
            for i in range(Npart):
                f.write(str(i+1)+" 1 "+str(positions[i,0])+" "+str(positions[i,1])+" "+str(positions[i,2])+" "
                    +str(int(image_flags[i,0]))+" "+str(int(image_flags[i,1]))+" "+str(int(image_flags[i,2]))+"\n")

            f.write("\nVelocities\n\n")

            for i in range(Npart):
                f.write(str(i+1)+" "+str(velocities[i,0])+" "+str(velocities[i,1])+" "+str(velocities[i,2])+"\n")


#   Main Script
def md_simulation():
    print("Setting up simulation")

    #   Set up a variable to keep track of the current timestep
    current_timestep = 0

    #   Define the bounds of the simulation box
    box_bounds = np.zeros((len(box),2))
    box_bounds[:,1] = box

    #   Initialise the random number generator for this run
    global rng 
    rng = np.random.default_rng(seed)

    #   Clear the output and log files if they exists
    with open(outfile, "w+") as f:
        f.write("")
    with open(log_file, "w+") as f:
            f.write("TIMESTEP TEMPERATURE DENSITY E_total E_kinetic E_potential # seed = "+str(seed)+"\n")

    #   Initialise starting positions
    if lattice == True:
        positions = init_positions_fcc(Npart, box)

        #   Prepare image flags, these keep track of how many boxes an atom has moved through
        image_flags = np.zeros(np.shape(positions))
        image_flags = np.zeros((Npart,ndim))
    
        #   Give each atom a starting velocity
        velocities = init_velocities_random(temperature)
        
    else:
        positions, image_flags, velocities = load_LAMMPS(init_file)

    
    #   Initialise an array to store accelerations
    accelerations = np.zeros(np.shape(positions))

    #   Write the starting configuration to the output file
    compute_thermodynamics(positions, velocities, current_timestep, log_file)
    dump_frame(positions, image_flags, velocities, outfile, current_timestep, Npart, box_bounds)
    dump_restart(positions, image_flags, velocities, "initial_config.dat", current_timestep, Npart, box_bounds)

    #   Equilibration steps
    for current_timestep in np.arange(1,(num_equilibration+1)):

            #   If on a velocity-rescaling timestep
            if current_timestep % thermostat_frequency == 0:
                #   Rescale the velocities to the target temperature
                velocities = velocity_rescaling(velocities, temperature)

            positions, velocities, accelerations = progress_MD(positions, velocities, accelerations) 
    
            #   Print configurations every thermo_frequency timesteps
            if current_timestep % thermo_frequency == 0:
                compute_thermodynamics(positions, velocities, current_timestep, log_file)
    
            #   Print configurations every dump_frequency timesteps
            if current_timestep % dump_frequency == 0:
                dump_frame(positions, image_flags, velocities, outfile, current_timestep, Npart, box_bounds)

    #   Production steps
    for current_timestep in np.arange(1,(num_production+1)):

        #   Add on the quilibration timesteps
        current_timestep = current_timestep + num_equilibration

        positions, velocities, accelerations = progress_MD(positions, velocities, accelerations)

        #   Print configurations every N timesteps
        if current_timestep % thermo_frequency == 0:
            compute_thermodynamics(positions, velocities, current_timestep, log_file)

        #   Print configurations every N timesteps
        if current_timestep % dump_frequency == 0:
            dump_frame(positions, image_flags, velocities, outfile, current_timestep, Npart, box_bounds)

    dump_restart(positions, image_flags, velocities, "final_config.conf", current_timestep, Npart, box_bounds)
    print("Simulation finished successfully")

##################################################
#   Setting up system constants
##################################################

cutoff_squared = cutoff ** 2
timestep2 = timestep ** 2

#   Note that non-cubic boxes are "allowed" but may cause unexpected behaviour when generating initial configurations
box = np.array([box_length,box_length,box_length])

md_simulation()
