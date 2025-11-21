#!/bin/bash

### Valid emulator with ResNet model (Evan's version, no TRF, 1 iter)

#SBATCH --job-name=TRFvalid
#SBATCH --output=/xdisk/timeifler/jiachuanxu/job_logs/emuvalid_restrf_HF_TATT_%A_%a.out

#SBATCH --account=timeifler

### Ocelote
###SBATCH --partition=gpu_standard
###SBATCH --qos=qual_qos_timeifler
###SBATCH --mem-per-gpu=16G

### Puma
#SBATCH --partition=gpu_standard
#SBATCH --mem-per-gpu=32G

#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12

#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jiachuanxu@arizona.edu

# Clear the environment from any previously loaded modules
module purge > /dev/null 2>&1
module load openmpi5
conda init bash
source ~/.bashrc

cd /groups/timeifler/jiachuanxu/cocoa_v4/Cocoa
echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo Slurm job NAME is $SLURM_JOB_NAME
echo Slurm job ID is $SLURM_JOBID

conda activate cocoa_v4
source start_cocoa.sh

MPIRUN="${CONDA_PREFIX}/bin/mpirun"
PYTHON="${CONDA_PREFIX}/bin/python3"

CONFIGFILE=./projects/roman_fourier/emulator/emulator/train_emu_restrf_HF_NLA.yaml
EMU_DCHI2=./projects/roman_fourier/emulator/chi2_examination.py

export OMP_NUM_THREADS=1

mpirun -n 1 --mca btl tcp,self --mca pml ob1 python -u ${EMU_DCHI2} $CONFIGFILE -thin=1

source stop_cocoa.sh
