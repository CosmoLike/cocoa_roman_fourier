## Running Cosmolike projects (Basic instructions) <a name="running_cosmolike_projects"></a> 

From `Cocoa/Readme` instructions:

> [!Note]
> We provide several cosmolike projects that can be loaded and compiled using `setup_cocoa.sh` and `compile_cocoa.sh` scripts. To activate them, comment the following lines on `set_installation_options.sh` 
> 
>     [Adapted from Cocoa/set_installation_options.sh shell script] 
>     # The keys below control which cosmolike projects will be installed and compiled 
>     # inset # symbol in the lines below (i.e., unset these environmental keys)
>     #export IGNORE_COSMOLIKE_LSSTY1_CODE=1
>     export IGNORE_COSMOLIKE_DES_Y3_CODE=1
>     export IGNORE_COSMOLIKE_ROMAN_FOURIER_CODE=1
>     export IGNORE_COSMOLIKE_ROMAN_REAL_CODE=1
>
> If users comment these lines (unsetting the corresponding IGNORE keys) after running `setup_cocoa.sh` and `compile_cocoa.sh`, there is no need to rerun these general scripts, which would reinstall many packages (slow). Instead, run the following three commands:
>
>      source start_cocoa.sh
>
> and
> 
>      source ./installation_scripts/setup_cosmolike_projects.sh
>
> and
> 
>       source ./installation_scripts/compile_all_projects.sh

To run the example

 **Step :one:**: activate the cocoa Conda environment,  and the private Python environment 
    
      conda activate cocoa

and

      source start_cocoa.sh
 
 **Step :two:**: Select the number of OpenMP cores (below, we set it to 8).
    
    export OMP_PROC_BIND=close; export OMP_NUM_THREADS=8
      
 **Step :three:**: The folder `projects/roman_fourier` contains examples. So, run the `cobaya-run` on the first example following the commands below.

One model evaluation:
      
       mpirun -n 1 --oversubscribe --mca pml ^ucx --mca btl vader,tcp,self --bind-to core:overload-allowed --rank-by slot --map-by numa:pe=${OMP_NUM_THREADS} cobaya-run ./projects/roman_fourier/EXAMPLE_EVALUATE1.yaml -f
