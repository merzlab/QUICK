Installation Guide
========================

1. Single CPU Installation 
^^^^^^^^^^^^^^^^^^^^^^^^^^

You can use gnu or intel compilers for this purpose. Go to QUICK main folder and run the following
commands.  If you are using gnu compiler (version 4.6 or later)

	cp ./makein/make.in.gnu ./make.in
	
If you are using intel compiler (version 2011 or later)

	cp ./makein/make.in.intel ./make.in

Then, run the following command. 

        make quick
     
This will compile a single CPU version of quick. 

2. MPI Installation
^^^^^^^^^^^^^^^^^^^

If you have intel mpi (version 2011 or later) installed, you can compile the MPI version by running 
following commands from quick main folder. 

	cp ./makein/make.in.MPI ./make.in
	
	make quick.MPI

3. CUDA Version Installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you want to install the GPU version, NVIDIA CUDA COMPILER is required. You may check your CUDA 
compiler version using 'nvcc --version'. 
a) Run the following command.

	cp ./makein/make.in.gnu.cuda ./make.in

b) Open up the make.in file and set CUDA_HOME. This is essential to link or compile CUBLAS and other 
	libraries.

	CUDA_HOME=(your cuda home) 

c) You may have to change the "-gencode arch=compute_70,code=sm_70" options in CUDA_FLAGS 
variable depending on the type of your GPU. The default value (70) is for a Volta gpu. Use 30, 50, 60 
and 75 for Kepler, Maxwell, Pascal and Turing GPUs respectively. 

	-gencode arch=compute_(your gpu),code=sm_(your gpu)

d) Then run
     
     make quick.cuda

in ./bin directory, you can find executable files. 
