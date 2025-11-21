#!/bin/bash
### Job request info
#SBATCH --job-name=FDC1EmuMCMC
#SBATCH --output=/xdisk/timeifler/jiachuanxu/job_logs/RomanFDC1EmuMCMC-%A_%a.out
#SBATCH --nodes=1
###SBATCH --array=1-100
#SBATCH --array=26-28,50-56,87
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=5GB
#SBATCH --time=36:00:00
#SBATCH --export=none
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jiachuanxu@arizona.edu

### Set up your job partition here
#SBATCH --partition=high_priority
#SBATCH --qos=user_qos_timeifler
#SBATCH --account=timeifler

### Customize the job
export MY_COCOA_ENV=cocoa_v4
export MY_MCMC_YAML=./projects/roman_fourier/emu_yaml/3x2pt_scalecuts/MCMC${SLURM_ARRAY_TASK_ID}.yaml
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
