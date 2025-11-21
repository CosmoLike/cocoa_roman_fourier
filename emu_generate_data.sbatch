#!/bin/bash

### Calculate training and validation dataset using cocoa

#SBATCH --job-name=EMUGenData
#SBATCH --output=/xdisk/timeifler/jiachuanxu/job_logs/emu_gen_data_%A_%a.out

#SBATCH --nodes=10
#SBATCH --ntasks-per-node=28
#SBATCH --exclusive

###SBATCH --partition=high_priority
###SBATCH --qos=user_qos_timeifler
#SBATCH --partition=standard
#SBATCH --qos=qual_qos_timeifler

#SBATCH --account=timeifler
#SBATCH --time=96:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jiachuanxu@arizona.edu

# Clear the environment from any previously loaded modules
module purge > /dev/null 2>&1
module load openmpi3
conda init bash
source ~/.bashrc

ROOTDIR="/groups/timeifler/jiachuanxu/cocoa_v4/Cocoa"
cd ${ROOTDIR}

echo Running on host `hostname`
###$PYTHON3 ${TRAINEMU} $CONFIGFILE 0 --temper --save_emu --debug
echo Time is `date`
echo Directory is `pwd`
echo Slurm job NAME is $SLURM_JOB_NAME
echo Slurm job ID is $SLURM_JOBID

conda activate cocoa_v4
source start_cocoa.sh

MPIRUN="${CONDA_PREFIX}/bin/mpirun"
PYTHON="${ROOTDIR}/.local/bin/python3"
#CONFIGFILE=./projects/roman_fourier/emulator/train_emu_resnet_HF_NLA.yaml
#CONFIGFILE=./projects/roman_fourier/emulator/train_emu_resmlp512_HF_NLA_reduced.yaml
CONFIGFILE=./projects/roman_fourier/emulator/train_emu_restrf512_HF_NLA_minimal.yaml

CALDV=./projects/roman_fourier/emulator/calculate_dv.py

export OMP_NUM_THREADS=1

${MPIRUN} -n ${SLURM_NTASKS} --mca btl tcp,self ${PYTHON} ${CALDV} $CONFIGFILE

#source stop_cocoa.sh
