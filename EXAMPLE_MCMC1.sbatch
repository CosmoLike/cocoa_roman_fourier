#!/bin/bash
### Job request info
#SBATCH --job-name=RomanMCMC
#SBATCH --output=/path/to/job/log/output/RomanFDC1MCMC-%a.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
### --ntasks-per-socket is important for CPU binding
#SBATCH --ntasks-per-socket=2
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=5GB
#SBATCH --time=120:00:00
#SBATCH --export=none
###SBATCH --mail-type=ALL
###SBATCH --mail-user=your_email_here

### Set up your job partition here
#SBATCH --partition=high_priority
#SBATCH --qos=user_qos_xxx
#SBATCH --account=xxx

### Customize the job
export MY_COCOA_ENV=cocoa_v4
export MY_MCMC_YAML=./projects/roman_fourier/EXAMPLE_MCMC1.yaml

### Run the job
echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo Slurm job NAME is $SLURM_JOB_NAME
echo Slurm job ID is $SLURM_JOBID

cd $SLURM_SUBMIT_DIR
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

mpirun -n ${SLURM_NTASKS} ${OPTION_MPI_BTL} ${OPTION_MPI_PML} ${OPTION_MPI_OS} ${OPTION_MPI_BIND} ${OPTION_MPI_RANK} ${OPTION_MPI_MAP} cobaya-run ${MY_MCMC_YAML} -r

