A PHYSICALLY CONSISTENT SINGLE-PHOTON LIDAR SIMULATION DATASET
FOR MOVING TARGET SENSING
=============================================================

This repository provides a physically consistent simulation framework
and baseline algorithms for single-photon LiDAR-based moving target
sensing and 3D imaging.

The project focuses on modeling photon-level detection processes,
background noise, and target motion effects using SPAD-based LiDAR
systems.


1. Repository Contents
----------------------

This repository includes:

- Simulation source code (C++ / CUDA)
- Parameter configuration files
- Baseline processing and evaluation scripts
- Documentation and usage instructions

Due to file size limitations, large-scale simulation datasets are not
stored directly in this GitHub repository.


2. Parameter Configuration
--------------------------

Simulation parameters can be modified in the file:

    params.h

This file controls system configuration, background photon rate,
target motion parameters, and other physical modeling settings.


3. Output Directory Configuration
---------------------------------

The directory used to save generated simulation data can be specified
in the file:

    main.cpp

Modify the following code segment to define the output path:

    std::string save_dir = "D:\\cu_file\\20260101\\rbg_400k_data";
    EnsureDirectory(save_dir);

The output directory will be created automatically if it does not
already exist. All generated simulation results will be saved to this
folder.


4. Running the Simulation
-------------------------

After configuring the parameters and output directory:

1) Compile the project using CMake or your preferred build system.
2) Run the executable to generate simulation data.
3) Generated data will be written to the specified output directory.


5. Dataset Availability
-----------------------

Due to GitHub file size limitations, the original generated datasets
are not included directly in this repository.

The complete datasets used in this project are publicly available at
the following Google Drive link:

https://drive.google.com/drive/folders/1ckOeZJovUHEUg4uEyC9QghNfRKeHnyBj?usp=drive_link

Alternatively, all datasets can be regenerated using the provided
simulation code by configuring the parameters and output directory
as described above.


6. Reproducibility
------------------

This repository is designed to support full reproducibility.
All datasets referenced in the project can be reproduced using the
released simulation framework and parameter configurations.


7. Contact
----------

For questions, suggestions, or collaboration inquiries, please contact
the repository maintainer via GitHub.


8. Citation
-----------

If you use this code or dataset in your research, please cite the
associated publication or reference this repository accordingly.
