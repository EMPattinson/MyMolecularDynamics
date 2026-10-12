# MyMolecularDynamics
A very simple molecular dynamics script used for educational purposes.

Can perform NVE simulations of a Lennard-Jones fluid. And output basic properties and a LAMMPS-style trajectory file for visualisation.

It's certainly not the quickest, but the aim is to maximise readability so students can investigate each part that makes up a Molecular Dynamics script seperately and develop their understanding.

The code has been benchmarked against both LAMMPS (https://www.lammps.org) by comparing the result of each time integration between MMD and LAMMPS. With the two only showing some divergence after thousands of timesteps.

It has also been benchmarked against the NIST LJ-MD dataset hosted here : https://mmlapps.nist.gov/srs/LJ_PURE/md.html
And shows fair agreement, aside from at a reduced density of 0.900 in which we encounter issues when equilibrating straight from a crystal to the desired temperature.

All benchmarking has been provided in the relevant folders.
