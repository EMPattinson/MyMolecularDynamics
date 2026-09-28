#   My Molecular Dynamics is a script that performs basic molecular dynamics simulations in 2D/3D

import numpy as np
import math

global boltzmann_constant, mass, sigma, epsilon, cutoff, cutoff_squared, Npart, box, timestep, timestep2
##################################################
#   Key parameters - change these!
##################################################

#   System parameters
Npart = 500
cutoff = 3
box_length = 8.6371
timestep = 0.005
temperature = 0.851

#   Simulation parameters
num_equilibration = 1000
num_production = 2000
dump_frequency = 10
thermo_frequency = 10

##################################################
#   Other paramters - Be careful when changin these!
##################################################

#   Define scientific constants
boltzmann_constant = 1  #   We set kB = 1 to be working in reduced units of temperature (which makes the numbers much more managable)

#   Define potential parameters
mass = 1
sigma = 1
epsilon = 1

#   Define the seed for the RNG and the names of the output files
seed = 2398008135762
outfile_name = "trajectory.lammpstrj"
log_file_name = "MMD.log"

# The command that runs the simulation is a function called "md_simulation()" and is called at the end of this file
#   This is because we need to define all of the functions used to run the simulation before we can run it.

##################################################
#   Functions
##################################################

#   Generally useful functions

#   Take a set of vectors and find the minimum image equivalent of those vectors
def minimum_image(vectors, box):

    minimum_image_vectors = vectors - (np.rint(vectors / box[None,:]) * box[None,:])

    return minimum_image_vectors

#   Function that computes the Maxwell-Boltzmann distribution for a set of given points at a defined temperature T
def maxwell_boltzmann(x, T):
    return (4*math.pi) * ((mass / (2 * math.pi * boltzmann_constant * T)) ** (3/2)) * (x ** 2) * np.exp(-((mass * (x**2))/(2 * boltzmann_constant * T)))

#   Function to rescale velocities from a computed instantaneous temperature to a target temperature
def velocity_rescaling(velocities, target_T):
    #   Compute the instantaneous temperature of the sample
    kinetic_energy = np.sum((1/2) * np.sum(np.square(velocities), axis=1))
    Ek_per_particle = kinetic_energy / Npart
    temperature = (2*Ek_per_particle) / (3 * boltzmann_constant)

    #   Scale the velocities based on the ratio (computed T/ target T)
    velocities = velocities * math.sqrt(target_T / temperature)

    return velocities

#   Functions that define the system potential

#   Compute the truncated Lennard-Jones energy for a particle from a set of interparticle distances
#       [Note] Unlike with forces, energy is a scalar quantity and so we return a scalar
def lennard_jones_energy(distances):

    #   Compute the magnitude of the distance vectors
    r2 = np.sum( (distances ** 2) , axis = 1)

    #   Check which of these distances fall outside the distance cut-off and ignore them
    cutoff_mask = np.less_equal(r2, cutoff_squared)
    r2 = r2[cutoff_mask]

    #   Compute the actual distance
    r = np.sqrt(r2)

    #   The Lennard-Jones potential has the form:
    #       U = (4*epsilon) * [(sigma/r)^12 - (sigma/r)^6]
    #   [WIP] for now we work in reduced units so epsilon = 1 and sigma = 1
    energy = 4 * (((1/r) ** 12) - ((1/r) ** 6))

    total_energy = np.sum(energy)
    return total_energy

#   Compute the truncated Lennard-Jones forces for a particle, from a set of distance vectors
#       we return a vector of x,y,z forces to be used in acceleration calculations
def lennard_jones_force(distances): # !!! WIP - check if this is right, I believe we need to seperate magnitude and direction. but I'm not 100% sure what the best way to handle that is 
    
    #   Compute the magnitude^2 of the distance vectors
    r2 = np.sum(distances ** 2, axis = 1)

    #   Check which of these distances fall outside the distance cut-off and ignore them
    cutoff_mask = np.less_equal(r2, cutoff_squared)
    r2 = r2[cutoff_mask]
    distances = distances[cutoff_mask,:]

    #   Compute the actual distance
    r = np.sqrt(r2)

    #   The force enacted on a particle is equal to the gradient of the energy potential
    #   So, the force for a Lennard-Jones potential is given by:
    #       F = (-48*epsilon) * [(sigma/r)^14 - (1/2)(sigma/r)^8]
    forces = 48 * (((1/r) ** 14) - (0.5)*((1/r) ** 8))

    #   Multiply the magnitude of the forces by the distance vectors to get the final force vectors
    forces = forces[:,None] * distances

    #   Sum the forces in the x,y and z directions
    forces = np.sum(forces, axis=0)

    return forces


#   Functions that initialise the system

#   Initialise a face centred cubic lattice
def init_positions_fcc(Npart, box):

    #   Ensure the box is a numpy array
    box = np.array(box)

    #   Determine the smallest cubic number greater than Npart/4
    #       This is used as the unit cell of an fcc lattice contains 4 particles
    num_unit_cells = math.ceil(abs(Npart/4) ** (1/3))

    #   Define the positions of the unit cell, using an arbitraty lattice vector of 1
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
        print("Error encountered with box dimensions in ffc lattice initalisation. Only 3D allowed.")
        exit()

    # Convert these to "real" coordinates by scalling the lattice coordinates by the box dimensions
    pos = pos[:Npart] * (box / np.array([num_unit_cells])[:,None])

    return pos

#   Initialise random velocities, drawn from a Boltzmann distribution and then scaled to a temperature T
#   We use a position array as input to determine the number of particles and dimensionality of the system
def init_velocities_random(positions, target_T):

    # Initialise an array to store the velocities, with random numbers between 0 and 1 in each entry

    velocities = rng.random(np.shape(positions))
    velocities = velocities - np.array([0.5,0.5,0.5])

    # Normalise these velocities so they have a magnitude of 1
    # These can then be used as directional vectors that we add a magnitude to drawn from the desired distribution
    velocities = velocities / np.linalg.norm(velocities, axis=1)[:,None]

    #   Make sure the overall momentum of the system is 0 (this avoids centre of mass drift in the simulation)
    velocities -= np.sum(velocities, axis = 0) / Npart
    
    # Draw the velocity magnitudes
    dist = np.arange(0,1,0.0001, dtype=float)
    probabilities = maxwell_boltzmann(dist, target_T)
    probabilities = probabilities / np.sum(probabilities)
    vel_magnitudes = np.random.choice(dist, size= Npart, p = probabilities)


    #   Multiply the velocity directions by their magnitudes
    velocities = velocities * vel_magnitudes[:,None]

    #   Rescale the velocities to a target temperature
    velocity_rescaling(velocities, target_T)

    return velocities


#   Functions that run the simulation

#   A single MD timestep
def progress_MD(positions, velocities, accelerations):

    #   Update positions using:
    #       dr = v*dt + (1/2)*a*(dt^2)
    positions += (velocities*timestep) + ((1/2) * accelerations * timestep2)

    #   Update the velocities using the formula:
    #       dv(t + dt) = (1/2)*a(t)*dt + (1/2)*a(t+dt)*dt
    velocities += (1/2) * accelerations * timestep

    #   Update accelerations
    accelerations = compute_accelerations(positions)

    velocities += (1/2) * accelerations * timestep

    return positions, velocities, accelerations

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


#   Functions that print the results of the simulations

#   Compute various thermodynamic properties including: Energy and Temperature
#       Prints output to a log file
def compute_thermodynamics(positions, velocities, current_timestep, log_file):

    #   Compute potential energy - WIP
    #       [Note] -    Energies are almost computed when calculating the accelerations,
    #                   it's an easy speed-up to calculate energies there
    potential_energy = 0
    for particle_i in range(Npart):
        distances = positions[particle_i,:] - np.delete(positions,particle_i, axis=0)
        distances = minimum_image(distances, box)
        potential_energy += lennard_jones_energy(distances)
    potential_energy = potential_energy / 2

    #   Normalise by number of particles
    Ep_per_particle = potential_energy / Npart

    #   Compute the total kinetic energies using the equation:
    #       Ek = (1/2) * m * (v^2)
    #   Note that we use reduced units so the mass of the particles is 1m
    kinetic_energy = np.sum((1/2) * np.sum(np.square(velocities), axis=1))

    #   Normalise by the number of particles
    Ek_per_particle = kinetic_energy / Npart

    #   Compute total energy
    E_total = Ek_per_particle + Ep_per_particle

    #   Compute the instantaneous temperature using the equipartition theorem given by the equation:
    #       <Ek> = (3N/2)*kB*T
    temperature = (2*Ek_per_particle) / (3 * boltzmann_constant)

    density = Npart / np.prod(box)

    with open(log_file, "a") as f:
        f.write(str(current_timestep)+", "+str(temperature)+", "+str(density)+", "+str(E_total)+", "+str(Ek_per_particle)+", "+str(Ep_per_particle)+"\n")

#   Function that prints the current positions and velocities in a LAMMPS style dump format
def dump_frame(positions, velocities, out_file, frame, Npart, box_bounds):

    with open(out_file, "a") as f:
        f.write("ITEM: TIMESTEP\n")
        f.write(str(frame)+"\n")
        f.write("ITEM: NUMBER OF ATOMS\n")
        f.write(str(Npart)+"\n")
        f.write("ITEM: BOX BOUNDS pp pp pp\n")
        np.savetxt(f, box_bounds)
        f.write("ITEM: ATOMS id type x y zz vx vy vz\n") # TO-DO: Need to adjust this to preserve dimensionality of the system

        #   Stack the data into one array for quick writing
        for i in range(Npart):
            f.write(str(i)+" 1 "+str(positions[i,0])+" "+str(positions[i,1])+" "+str(positions[i,2])+" "
                +str(velocities[i,0])+" "+str(velocities[i,1])+" "+str(velocities[i,2])+"\n")

    print("frame sucessfully dumped at timestep: ", frame)


#   The main MD script
def md_simulation(Npart, temperature, box, num_equilibration, num_timesteps, dump_frequency, seed, outfile, log_file):
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
            f.write("TIMESTEP TEMPERATURE DENSITY E_total E_kinetic E_potential\n")

    #   Initialise the starting positions of the atoms
    positions = init_positions_fcc(Npart, box)

    #   Give each atom a starting velocity
    velocities = init_velocities_random(positions, temperature)

    #   Compute the starting accelerations
    accelerations = np.zeros(np.shape(positions))

    #   Write the starting configuration to the output file
    compute_thermodynamics(positions, velocities, current_timestep, log_file)
    dump_frame(positions, velocities, outfile, current_timestep, Npart, box_bounds)

    #   Equilibration steps
    for current_timestep in np.arange(1,(num_equilibration+1)):
            positions, velocities, accelerations = progress_MD(positions, velocities, accelerations)

            #   Rescale the velocities to the target temperature
            velocities = velocity_rescaling(velocities, temperature)
    
            #   Print configurations every N timesteps
            if current_timestep % thermo_frequency == 0:
                compute_thermodynamics(positions, velocities, current_timestep, log_file)
    
            #   Print configurations every N timesteps
            if current_timestep % dump_frequency == 0:
                dump_frame(positions, velocities, outfile, current_timestep, Npart, box_bounds)


    #   Production steps
    for current_timestep in np.arange(1,(num_timesteps+1)):

        #   Add on the quilibration timesteps
        current_timestep = current_timestep + num_equilibration

        positions, velocities, accelerations = progress_MD(positions, velocities, accelerations)

        #   Print configurations every N timesteps
        if current_timestep % thermo_frequency == 0:
            compute_thermodynamics(positions, velocities, current_timestep, log_file)

        #   Print configurations every N timesteps
        if current_timestep % dump_frequency == 0:
            dump_frame(positions, velocities, outfile, current_timestep, Npart, box_bounds)


##################################################
#   Setting up system constants
##################################################

cutoff_squared = cutoff ** 2
timestep2 = timestep ** 2

#   Note that non-cubic boxes are "allowed" but may cause unexpected behaviour when generating initial configurations
box = np.array([box_length,box_length,box_length])

md_simulation(Npart, temperature, box, num_equilibration, num_production, dump_frequency, seed, outfile_name, log_file_name)
