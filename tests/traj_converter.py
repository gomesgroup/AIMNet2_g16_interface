from ase.io import read, write

# Replace 'CCOC.trj' with the path to your ASE trajectory file
trajectory_file = 'CCOC.traj'

# The output .xyz filename that will contain all frames
output_file = 'CCOC_all_frames.xyz'

# Read the trajectory
atoms = read(trajectory_file, index=':')

# Convert and save all frames to a single .xyz file
write(output_file, atoms)

