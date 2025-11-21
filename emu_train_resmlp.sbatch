#!/bin/bash

### Train emulator with ResNet model (Evan's version, no TRF, 1 iter)

#SBATCH --job-name=mlp512
#SBATCH --output=/xdisk/timeifler/jiachuanxu/job_logs/emutrain_resmlp_HF_NLA_RomanFourier_%A_%a.out

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
###SBATCH --mem-per-cpu=5gb


#SBATCH --time=48:00:00
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
echo $LD_LIBRARY_PATH
MPIRUN="${CONDA_PREFIX}/bin/mpirun"
PYTHON="${CONDA_PREFIX}/bin/python3"

#CONFIGFILE=./projects/roman_fourier/emulator/train_emu_resnet_HF_TATT.yaml
#CONFIGFILE=./projects/roman_fourier/emulator/train_emu_resmlp_HF_NLA.yaml
#CONFIGFILE=./projects/roman_fourier/emulator/train_emu_resmlp512_HF_NLA.yaml
#CONFIGFILE=./projects/roman_fourier/emulator/train_emu_resmlp512_HF_NLA_reduced.yaml
CONFIGFILE=./projects/roman_fourier/emulator/train_emu_resmlp512_7_HF_NLA_minimal.yaml

TRAINEMU=./projects/roman_fourier/emulator/train_emulator.py

export OMP_NUM_THREADS=1

mpirun -n 1 --mca btl tcp,self --mca pml ob1 python -u ${TRAINEMU} $CONFIGFILE --overwrite

#source stop_cocoa.sh
