# cervenka_tma 

This repository contains input files for molecular dynamics (MD) simulations of tetramethylammonium chloride (TMACl) interacting with pyridine or phenol in aqueous solution. The simulations are performed using either the CHARMM36 force field or the ProsECCo75 force field (which incorporates an ECC scaling factor to account for electronic polarization effects). 

## Repository Structure

The repository is organized into four directories, each corresponding to a separate simulation setup:

### 1. TMA_Pyridine
- Contains all necessary input files (structures, topology files, and MD parameter files) for simulating **40 molecules of TMACl, pyridine, and 1110 TIP3P water molecules** using the **CHARMM36 force field**.

### 2. TMA_Pyridine_ECC
- Identical to `TMA_Pyridine`, but using the **ProsECCo75 force field**, which applies ECC scaling to better capture cation-π interactions.

### 3. TMA_Phenol
- Contains input files for simulating **40 molecules of TMACl, phenol, and 1110 TIP3P water molecules** using the **CHARMM36 force field**.

### 4. TMA_Phenol_ECC
- Identical to `TMA_Phenol`, but using the **ProsECCo75 force field** with ECC scaling.

---

This structure ensures reproducibility and allows for direct comparisons between standard and ECC-scaled force fields in modeling cation-π interactions.