#!/bin/bash

#SBATCH --job-name=EMC_LSSTF
#SBATCH --output=emc-%a.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks-per-socket=2
#SBATCH --cpus-per-task=10
#SBATCH --partition=standard
#SBATCH --qos=user_qos_timeifler
#SBATCH --account=cosmo
#SBATCH --mem-per-cpu=5GB
#SBATCH --time=99:00:00
#SBATCH --export=none

echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo Slurm job NAME is $SLURM_JOB_NAME
echo Slurm job ID is $SLURM_JOBID

cd $SLURM_SUBMIT_DIR
source start_cocoa

export OMP_PROC_BIND=close
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
mpirun -n ${SLURM_NTASKS} --bind-to core --rank-by core --map-by numa:pe=${OMP_NUM_THREADS} cobaya-run ./projects/lsst_fourier/EXAMPLE_MCMC${SLURM_ARRAY_TASK_ID}.yaml -r