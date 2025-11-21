#!/bin/bash
### Job request info
#SBATCH --job-name=FDC1EmuMCMC
#SBATCH --output=/xdisk/timeifler/jiachuanxu/job_logs/RomanFDC1EmuMCMC-%A_%a.out
#SBATCH --nodes=1
#SBATCH --array=1-20
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=5GB
#SBATCH --time=48:00:00
#SBATCH --export=none
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jiachuanxu@arizona.edu

### Set up your job partition here
#SBATCH --partition=high_priority
#SBATCH --qos=user_qos_timeifler
#SBATCH --account=timeifler

### List all the yaml files
declare -a YAML_LIST=(
"ROMAN_FOURIER_CHALLENGE1_MCMC_LES_LCDM_3x2pt.yaml"
"ROMAN_FOURIER_CHALLENGE1_MCMC_LES_LCDM_cosmic_shear.yaml"
"ROMAN_FOURIER_CHALLENGE1_MCMC_LES_nuCDM_3x2pt.yaml"
"ROMAN_FOURIER_CHALLENGE1_MCMC_LES_nuCDM_cosmic_shear.yaml"
"ROMAN_FOURIER_CHALLENGE1_MCMC_LES_w0waCDM_3x2pt.yaml"
"ROMAN_FOURIER_CHALLENGE1_MCMC_LES_w0waCDM_cosmic_shear.yaml"
"ROMAN_FOURIER_CHALLENGE1_MCMC_LES_w0wanuCDM_3x2pt.yaml"
"ROMAN_FOURIER_CHALLENGE1_MCMC_LES_w0wanuCDM_cosmic_shear.yaml"
"ROMAN_FOURIER_CHALLENGE2_MCMC_LES_w0waCDM_3x2pt.yaml"
"ROMAN_FOURIER_CHALLENGE2_MCMC_LES_w0waCDM_cosmic_shear.yaml"
"ROMAN_FOURIER_CHALLENGE2_MCMC_LES_LCDM_3x2pt.yaml"
"ROMAN_FOURIER_CHALLENGE2_MCMC_LES_LCDM_cosmic_shear.yaml"
"ROMAN_FOURIER_CHALLENGE3_MCMC_LES_w0waCDM_3x2pt.yaml"
"ROMAN_FOURIER_CHALLENGE3_MCMC_LES_w0waCDM_cosmic_shear.yaml"
"ROMAN_FOURIER_CHALLENGE3_MCMC_LES_LCDM_3x2pt.yaml"
"ROMAN_FOURIER_CHALLENGE3_MCMC_LES_LCDM_cosmic_shear.yaml"
"ROMAN_FOURIER_CHALLENGE4_MCMC_LES_LCDM_3x2pt_Halofit.yaml"
"ROMAN_FOURIER_CHALLENGE4_MCMC_LES_LCDM_cosmic_shear_Halofit.yaml"
"ROMAN_FOURIER_CHALLENGE5_MCMC_LES_LCDM_3x2pt_DMO.yaml"
"ROMAN_FOURIER_CHALLENGE5_MCMC_LES_LCDM_cosmic_shear_DMO.yaml"
"ROMAN_FOURIER_CHALLENGE6_MCMC_LES_LCDM_3x2pt_DMO.yaml"
"ROMAN_FOURIER_CHALLENGE6_MCMC_LES_LCDM_cosmic_shear_DMO.yaml"
"ROMAN_FOURIER_CHALLENGE7_MCMC_LES_LCDM_3x2pt_NLA.yaml"
"ROMAN_FOURIER_CHALLENGE7_MCMC_LES_LCDM_cosmic_shear_NLA.yaml"
"ROMAN_FOURIER_CHALLENGE9_MCMC_LES_LCDM_3x2pt_nostretch.yaml"
"ROMAN_FOURIER_CHALLENGE9_MCMC_LES_LCDM_cosmic_shear_nostretch.yaml"
"ROMAN_FOURIER_CHALLENGE10_MCMC_LES_LCDM_3x2pt_narrow.yaml"
"ROMAN_FOURIER_CHALLENGE10_MCMC_LES_LCDM_cosmic_shear_narrow.yaml"
"ROMAN_FOURIER_CHALLENGE11_MCMC_LES_LCDM_3x2pt_nomag.yaml"
"ROMAN_FOURIER_CHALLENGE11_MCMC_LES_LCDM_cosmic_shear_nomag.yaml"
)
declare -a LCDM_YAML_LIST=(
"ROMAN_FOURIER_CHALLENGE1_MCMC_LES_LCDM_3x2pt.yaml"
"ROMAN_FOURIER_CHALLENGE1_MCMC_LES_LCDM_cosmic_shear.yaml"
"ROMAN_FOURIER_CHALLENGE2_MCMC_LES_LCDM_3x2pt.yaml"
"ROMAN_FOURIER_CHALLENGE2_MCMC_LES_LCDM_cosmic_shear.yaml"
"ROMAN_FOURIER_CHALLENGE3_MCMC_LES_LCDM_3x2pt.yaml"
"ROMAN_FOURIER_CHALLENGE3_MCMC_LES_LCDM_cosmic_shear.yaml"
"ROMAN_FOURIER_CHALLENGE4_MCMC_LES_LCDM_3x2pt_Halofit.yaml"
"ROMAN_FOURIER_CHALLENGE4_MCMC_LES_LCDM_cosmic_shear_Halofit.yaml"
"ROMAN_FOURIER_CHALLENGE5_MCMC_LES_LCDM_3x2pt_DMO.yaml"
"ROMAN_FOURIER_CHALLENGE5_MCMC_LES_LCDM_cosmic_shear_DMO.yaml"
"ROMAN_FOURIER_CHALLENGE6_MCMC_LES_LCDM_3x2pt_DMO.yaml"
"ROMAN_FOURIER_CHALLENGE6_MCMC_LES_LCDM_cosmic_shear_DMO.yaml"
"ROMAN_FOURIER_CHALLENGE7_MCMC_LES_LCDM_3x2pt_NLA.yaml"
"ROMAN_FOURIER_CHALLENGE7_MCMC_LES_LCDM_cosmic_shear_NLA.yaml"
"ROMAN_FOURIER_CHALLENGE9_MCMC_LES_LCDM_3x2pt_nostretch.yaml"
"ROMAN_FOURIER_CHALLENGE9_MCMC_LES_LCDM_cosmic_shear_nostretch.yaml"
"ROMAN_FOURIER_CHALLENGE10_MCMC_LES_LCDM_3x2pt_narrow.yaml"
"ROMAN_FOURIER_CHALLENGE10_MCMC_LES_LCDM_cosmic_shear_narrow.yaml"
"ROMAN_FOURIER_CHALLENGE11_MCMC_LES_LCDM_3x2pt_nomag.yaml"
"ROMAN_FOURIER_CHALLENGE11_MCMC_LES_LCDM_cosmic_shear_nomag.yaml"
)
### Customize the job
export MY_COCOA_ENV=cocoa_v4
export MY_MCMC_YAML=./projects/roman_fourier/emu_yaml/${LCDM_YAML_LIST[$((${SLURM_ARRAY_TASK_ID} - 1))]}
echo ${MY_MCMC_YAML}
cd $SLURM_SUBMIT_DIR
### Run the job

module purge > /dev/null 2>&1
module load anaconda
#module load openmpi5
conda init bash
source ~/.bashrc 
conda activate ${MY_COCOA_ENV}
source start_cocoa.sh

export OMP_PROC_BIND=close
if [ -n "$SLURM_CPUS_PER_TASK" ]; then
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
else
  export OMP_NUM_THREADS=1
fi

export OPTION_MPI_PML="--mca pml ob1"
export OPTION_MPI_BTL="--mca btl vader,tcp,self"

echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo Slurm job NAME is $SLURM_JOB_NAME
echo Slurm job ID is $SLURM_JOBID
echo "$(ls external_modules/data/roman_fourier/*)"

### without CPU binding
mpirun -n ${SLURM_NTASKS} ${OPTION_MPI_BTL} ${OPTION_MPI_PML} cobaya-run ${MY_MCMC_YAML} -r

### Note: if the cocoa code is installed in /group or /xdisk directory, DO NOT run stop_cocoa.sh
### Becasue /group and /xdisk directories are not cached during SLURM run, and running stop_cocoa.sh
### will terminate other running jobs.
#source stop_cocoa.sh
