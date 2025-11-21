#!/bin/bash
### Job request info
#SBATCH --job-name=FDC1EmuMCMC
#SBATCH --output=/xdisk/timeifler/jiachuanxu/job_logs/RomanFDC1EmuMCMC-%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks-per-socket=2
#SBATCH --cpus-per-task=10
###SBATCH --mem-per-cpu=5GB
#SBATCH --time=120:00:00
#SBATCH --export=none
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jiachuanxu@arizona.edu

### Set up your job partition here
#SBATCH --partition=high_priority
#SBATCH --qos=user_qos_timeifler
#SBATCH --account=timeifler

### Customize the job
export MY_COCOA_ENV=cocoa_v4
export MY_MCMC_YAML=./projects/roman_fourier/ROMAN_FOURIER_CHALLENGE1_EMUMCMC.yaml
cd $SLURM_SUBMIT_DIR
### Run the job
echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo Slurm job NAME is $SLURM_JOB_NAME
echo Slurm job ID is $SLURM_JOBID

module purge > /dev/null 2>&1
module load anaconda
conda init bash
source ~/.bashrc 
conda activate ${MY_COCOA_ENV}
source start_cocoa.sh

export OMP_PROC_BIND=close
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPTION_MPI_OS="--oversubscribe"
export OPTION_MPI_PML="--mca pml ob1"
export OPTION_MPI_BTL="--mca btl vader,tcp,self"
export OPTION_MPI_BIND="--bind-to core:overload-allowed"
export OPTION_MPI_RANK="--rank-by core"
export OPTION_MPI_MAP="--map-by numa:pe=${OMP_NUM_THREADS}"

### with CPU binding
#mpirun -n ${SLURM_NTASKS} ${OPTION_MPI_BTL} ${OPTION_MPI_PML} ${OPTION_MPI_OS} ${OPTION_MPI_BIND} ${OPTION_MPI_RANK} ${OPTION_MPI_MAP} cobaya-run ${MY_MCMC_YAML} -r

### without CPU binding
mpirun -n ${SLURM_NTASKS} ${OPTION_MPI_BTL} ${OPTION_MPI_PML} cobaya-run ${MY_MCMC_YAML} -r

source stop_cocoa.sh
