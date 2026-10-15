#   My Molecular Dynamics is a script that performs basic molecular dynamics simulations in 2D/3D

import numpy as np
import math
import random
import pathlib
import MMD_functions as mmd

global boltzmann_constant, mass, sigma, epsilon, cutoff, cutoff_squared, Npart, ndim, box, timestep, timestep2, tail_correction, thermostat_frequency, init_file, temperature, num_equilibration, num_production, dump_frequency, seed, outfile, log_file, rng, sim_params

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
num_equilibration = 0                         #   number of equilibration (NVT) steps to perform
num_production = 100                            #   number of production (NVE) steps to perform
dump_frequency = 10                             #   how often to print configurations, in MD timesteps
thermo_frequency = 1                            #   how often to print thermodynamic quantities, in MD timesteps

thermostat_frequency = 1                        #   how often to rescale velocities during euqilibration

lattice = True                                 #   if True, generate a configuration using an FCC lattice.
                                                #   if False, read an input configuration from a file
init_file = "config.conf"               #   name of the file containing an input configuration


tail_correction = False

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

#   Define the seed for the RNG and the names of the output files
outfile = "trajectory.lammpstrj"
log_file = "MMD.csv"

# The command that runs the simulation is a function called "md_simulation()" and is called at the end of this file
#   This is because we need to define all of the functions used to run the simulation before we can run it.

##################################################
#   Functions
##################################################

#   A single MD timestep
def progress_MD(positions, velocities, accelerations, image_flags):

    #   Update positions using:
    #       dr = v*dt + (1/2)*a*(dt^2)
    positions += (velocities*timestep) + ((1/2) * accelerations * timestep2)

    #   Wrap positions back inside the simulation box (asumes a box with a corner at 0,0,0)
    image_flags += np.floor(positions / box[None,:]).astype(int)
    positions = positions - (np.floor(positions / box[None,:]) * box[None,:])

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
    r_neg2 = (sigma**2) / r2
    r_neg6 = r_neg2 * r_neg2 * r_neg2

    #   The Lennard-Jones potential has the form:
    #       U = (4*epsilon) * [(sigma/r)^12 - (sigma/r)^6]
    energy = 4 * epsilon * ((r_neg6 * r_neg6) - r_neg6)

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
    r_neg2 = (sigma**2) / r2
    r_neg6 = r_neg2 * r_neg2 * r_neg2

    #   The force enacted on a particle is equal to the gradient of the energy potential
    #   So, the force for a Lennard-Jones potential is given by:
    #       F = (-48*epsilon) * [(sigma/r)^14 - (1/2)(sigma/r)^8]
    forces = (48 * epsilon) * ((r_neg6 * r_neg6 * r_neg2) - (0.5)*(r_neg6 * r_neg2))

    #   Multiply the magnitude of the forces by normalised distance vectors to get a force vector with appropriate direction
    forces = forces[:,None] * distances

    #   Sum the forces in the x,y and z directions
    forces = np.sum(forces, axis=0)

    return forces

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

#   Main Script
def md_simulation():
    print("Setting up simulation")

    #   Set up a variable to keep track of the current timestep
    current_timestep = 0

    #   Define the bounds of the simulation box
    box_bounds = np.zeros((len(box),2))
    box_bounds[:,1] = box

    #   Initialise the random number generator for this run 
    rng = np.random.default_rng(seed)

    #   Clear the output and log files if they exists
    with open(outfile, "w+") as f:
        f.write("")
    with open(log_file, "w+") as f:
        f.write("TIMESTEP, TEMPERATURE, DENSITY, E_total, E_kinetic, E_potential # seed = "+str(seed)+"\n")

    #   Initialise starting positions
    if lattice == True:
        positions = mmd.init_positions_fcc(Npart, box)

        #   Prepare image flags, these keep track of how many boxes an atom has moved through
        image_flags = np.zeros(np.shape(positions))
        image_flags = np.zeros((Npart,ndim))
    
        #   Give each atom a starting velocity
        velocities = mmd.init_velocities_random(temperature, Npart, ndim, rng, mass, boltzmann_constant)   
    else:
        positions, image_flags, velocities = mmd.load_LAMMPS(init_file, Npart)

    
    #   Initialise an array to store accelerations
    accelerations = compute_accelerations(positions)

    #   Write the starting configuration to the output file
    mmd.compute_thermodynamics(positions, velocities, current_timestep, log_file, **sim_params)
    mmd.dump_frame(positions, image_flags, velocities, outfile, current_timestep, Npart, box_bounds)
    mmd.dump_restart(positions, image_flags, velocities, "initial_config.conf", current_timestep, Npart, box_bounds)

    #   Equilibration steps
    for current_timestep in np.arange(1,(num_equilibration+1)):

        #   If on a velocity-rescaling timestep
        if current_timestep % thermostat_frequency == 0:
            #   Rescale the velocities to the target temperature
            velocities = mmd.velocity_rescaling(velocities, temperature, mass, boltzmann_constant, Npart)

        positions, velocities, accelerations = progress_MD(positions, velocities, accelerations, image_flags) 

        #   Print configurations every thermo_frequency timesteps
        if current_timestep % thermo_frequency == 0:
            mmd.compute_thermodynamics(positions, velocities, current_timestep, log_file, **sim_params)

        #   Print configurations every dump_frequency timesteps
        if current_timestep % dump_frequency == 0:
            mmd.dump_frame(positions, image_flags, velocities, outfile, current_timestep, Npart, box_bounds)

    #   Production steps
    for current_timestep in np.arange(1,(num_production+1)):

        #   Add on the quilibration timesteps
        current_timestep = current_timestep + num_equilibration

        positions, velocities, accelerations = progress_MD(positions, velocities, accelerations, image_flags)

        #   Print configurations every N timesteps
        if current_timestep % thermo_frequency == 0:
            mmd.compute_thermodynamics(positions, velocities, current_timestep, log_file, **sim_params)

        #   Print configurations every N timesteps
        if current_timestep % dump_frequency == 0:
            mmd.dump_frame(positions, image_flags, velocities, outfile, current_timestep, Npart, box_bounds)

    mmd.dump_restart(positions, image_flags, velocities, "final_config.conf", current_timestep, Npart, box_bounds)
    print("Simulation finished successfully")

##################################################
#   Setting up system constants
##################################################

cutoff_squared = cutoff ** 2
timestep2 = timestep ** 2

#   Note that non-cubic boxes are "allowed" but may cause unexpected behaviour when generating initial configurations
box = np.array([box_length,box_length,box_length])

#   Organise all the simulation parameters into a dictionary for easy calling later
sim_params = {
    "Npart" : Npart,
    "box" : box,
    "sigma" : sigma,
    "epsilon" : epsilon,
    "cutoff" : cutoff,
    "cutoff_squared" : cutoff_squared, 
    "mass" : mass, 
    "boltzmann_constant" : boltzmann_constant,
    "tail_correction" : tail_correction
}

md_simulation()
