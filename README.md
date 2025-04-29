# cervenka_tma 

This repository contains input files for molecular dynamics (MD) simulations of tetramethylammonium chloride (TMACl) interacting with pyridine or phenol in aqueous solution. The simulations are performed using either the CHARMM36 force field or the ProsECCo75 force field (which incorporates an ECC scaling factor to account for electronic polarization effects in charged moieties). Additionally, this repository includes **neutron scattering data analysis** for comparing the MD results with experimental data.

## Repository Structure

The repository is organized into seven directories:

### 1. `TMA_Pyridine/`
- Contains all necessary input files (structures, topology files, and MD parameter files) for simulating **40 molecules of TMACl, pyridine, and 1110 TIP3P water molecules** using the **CHARMM36 force field**.

### 2. `TMA_Pyridine_ECC/`
- Identical to `TMA_Pyridine`, but using the **ProsECCo75 force field**, which applies ECC scaling to better capture cation-π interactions.

### 3. `TMA_Phenol/`
- Contains input files for simulating **40 molecules of TMACl, phenol, and 1110 TIP3P water molecules** using the **CHARMM36 force field**.

### 4. `TMA_Phenol_ECC/`
- Identical to `TMA_Phenol`, but using the **ProsECCo75 force field** with ECC scaling.

### 5. `TMA_Pyridine_TIP4P/`
- Contains all necessary input files (structures, topology files, and MD parameter files) for simulating **40 molecules of TMACl, pyridine, and 1110 TIP4P water molecules** using the **CHARMM36 force field**.

### 6. `TMA_Phenol_TIP4P/`
- Contains input files for simulating **40 molecules of TMACl, phenol, and 1110 TIP4P water molecules** using the **CHARMM36 force field**.

### 7. `neutron_scattering_data_and_analysis/`
- Contains **experimental neutron scattering data** and **Gaussian Process Regression (GPR) analysis** scripts for evaluating the accuracy of different force field models.
