/*
 *  gpu_type.h
 *  new_quick
 *
 *  Created by Yipu Miao on 6/1/11.
 *  Copyright 2011 University of Florida. All rights reserved.
 *
 */

/*
 * this head file includes following type define
    a. common variables             : see details below
    b. gpu type and buffer type     : for communication between GPU and CPU
 */

#include <stdio.h>
#include "gpu_common.h"

// CUDA-C includes
#include <cuda.h>
//#include <cuda_runtime_api.h>

/*
 ****************************************************************
 *  gpu type and buffer type
 ****************************************************************
 */
template <typename T> struct cuda_buffer_type;
struct gpu_calculated_type {
    int                             natom;  // number of atom
    int                             nbasis; // number of basis sets
    cuda_buffer_type<QUICKDouble>*  o;      // O matrix
    cuda_buffer_type<QUICKDouble>*  dense;  // Density Matrix
    cuda_buffer_type<QUICKULL>*     oULL;   // Unsigned long long int type O matrix
    cuda_buffer_type<QUICKDouble>*  distance; // distance matrix
};

struct gpu_cutoff_type {
    int                             natom;
    int                             nbasis;
    int                             nshell;
    
    // the following are for pre-sorting cutoff
    int                             sqrQshell;
    cuda_buffer_type<int2>*         sorted_YCutoffIJ;
    
    // Cutoff matrix
    cuda_buffer_type<QUICKDouble>*  cutMatrix;
    cuda_buffer_type<QUICKDouble>*  YCutoff;
    cuda_buffer_type<QUICKDouble>*  cutPrim;
    
    // Cutoff criteria
    QUICKDouble                     integralCutoff;
    QUICKDouble                     primLimit;
    QUICKDouble                     DMCutoff;
    QUICKDouble                     gradCutoff;
    
};

struct DFT_calculated_type {
    QUICKULL                     Eelxc;      // exchange correction energy
    QUICKULL                     aelec;      // alpha electron
    QUICKULL                     belec;      // beta electron
};

/*Madu Manathunga 11/21/2019*/
struct XC_quadrature_type{
	int npoints;								//Total number of packed grid points
	int nbins;									//Total number of bins
	int ntotbf;									//Total number of basis functions
	int ntotpf;									//Total number of primitive functions
	int bin_size;                               //Size of an octree bin

	cuda_buffer_type<QUICKDouble>* gridx;		//X coordinate of a grid point
	cuda_buffer_type<QUICKDouble>* gridy;		//Y coordinate of a grid point
	cuda_buffer_type<QUICKDouble>* gridz;		//Z coordinate of a grid point
	cuda_buffer_type<QUICKDouble>* sswt;		//A version of weight required for gradients
	cuda_buffer_type<QUICKDouble>* weight;		//Scuzeria weight of a grid point
	cuda_buffer_type<int>*	gatm;			//To which atom does a given grid point belongs to?
	cuda_buffer_type<int>*	dweight;		//Dummy weight of grid points
	cuda_buffer_type<int>*  dweight_ssd;            //Dummy weight of grid points for sswder 
	cuda_buffer_type<int>*	basf;			//Basis function indices of all grid points
	cuda_buffer_type<int>*	primf;			//Primitive function inidices of all grid points
        cuda_buffer_type<int>*  primfpbin;                 //Number of primitive functions per bin
	cuda_buffer_type<int>*	basf_locator;		//Helps accessing b.f. indices of a grid point
	cuda_buffer_type<int>*	primf_locator;		//Helps accessing p.f. indices of a b.f.

	//Temporary variables
	cuda_buffer_type<QUICKDouble>* densa;
	cuda_buffer_type<QUICKDouble>* densb;
	cuda_buffer_type<QUICKDouble>* gax;
	cuda_buffer_type<QUICKDouble>* gbx;
	cuda_buffer_type<QUICKDouble>* gay;
	cuda_buffer_type<QUICKDouble>* gby;
	cuda_buffer_type<QUICKDouble>* gaz;
	cuda_buffer_type<QUICKDouble>* gbz;
	cuda_buffer_type<QUICKDouble>* exc;
	cuda_buffer_type<QUICKDouble>* xc_grad;
	cuda_buffer_type<QUICKDouble>* gxc_grad;        // a global xc gradient vector of size number_of_blocks * number_of_threads_per_block

        //Variables for ssw derivative calculation
        int npoints_ssd; //Total number of input points for ssd

        cuda_buffer_type<QUICKDouble>* gridx_ssd;       //X coordinate of a grid point
        cuda_buffer_type<QUICKDouble>* gridy_ssd;       //Y coordinate of a grid point
        cuda_buffer_type<QUICKDouble>* gridz_ssd;       //Z coordinate of a grid point
        cuda_buffer_type<QUICKDouble>* exc_ssd;
        cuda_buffer_type<QUICKDouble>* quadwt;          //quadrature weight
        cuda_buffer_type<int>*  gatm_ssd;               //To which atom does a given grid point belongs to?
	cuda_buffer_type<QUICKDouble>* uw_ssd;          //Holds unnormalized weights during ssd calculation
	
	//Variables for grid weight calculation
	cuda_buffer_type<QUICKDouble>* wtang;
	cuda_buffer_type<QUICKDouble>* rwt;
	cuda_buffer_type<QUICKDouble>* rad3;

	//Variables for obtaining octree info 
	cuda_buffer_type<unsigned char>* gpweight;     //keeps track of significant grid points for octree pruning
	cuda_buffer_type<unsigned int>*  cfweight;     //keeps track of significant b.f. for octree pruning 
	cuda_buffer_type<unsigned int>*  pfweight;     //keeps track of significant p.f. for octree pruning

        // mpi variables
        cuda_buffer_type<char>*          mpi_bxccompute;

        // shared memory size
        int smem_size;                                 //size of shared memory buffer in xc kernels 
};


struct gpu_simulation_type {
    
    // basic molecule information and method information
    QUICK_METHOD                    method;
    DFT_calculated_type*            DFT_calculated;
    XC_quadrature_type*             xcq;
    QUICKDouble                     hyb_coeff;   
 
    // used for DFT
    int                             isg;        // isg algrothm
    QUICKDouble*                    sigrad2;    // basis set range
    
    int                             natom;
    int                             nbasis;
    int                             nshell;
    int                             nprim;
    int                             jshell;
    int                             jbasis;
    int                             nElec;
    int                             imult;
    int                             molchg;
    int                             iAtomType;
    int                             maxcontract;
    int                             Qshell;
    int                             fStart;
    int                             ffStart;
    int                             maxL;

	//New XC implementation
    int npoints;                                //Total number of packed grid points
    int nbins;                                  //Total number of bins
    int ntotbf;                                 //Total number of basis functions
    int ntotpf;                                 //Total number of primitive functions
    int bin_size;				//Size of an octree bin
    int npoints_ssd;                  //Total number of input points for ssd

    QUICKDouble* gridx;       //X coordinate of a grid point
    QUICKDouble* gridy;       //Y coordinate of a grid point
    QUICKDouble* gridz;       //Z coordinate of a grid point
    QUICKDouble* gridx_ssd;
    QUICKDouble* gridy_ssd;
    QUICKDouble* gridz_ssd;
    QUICKDouble* sswt;        //A version of weight required for gradients
    QUICKDouble* weight;      //Scuzeria weight of a grid point
    QUICKDouble* quadwt;      //quadrature weight for sswder
    QUICKDouble* uw_ssd;      //unnormalized weight for sswder
    QUICKDouble* densa;       //Alpha electron density
    QUICKDouble* densb;       //Beta electron density
    QUICKDouble* gax;         //Gradient of densities
    QUICKDouble* gbx;
    QUICKDouble* gay;
    QUICKDouble* gby;
    QUICKDouble* gaz;
    QUICKDouble* gbz;
    QUICKDouble* exc;         //Exchange correlation energy
    QUICKDouble* exc_ssd;     //Exchange correlation energy for sswder calculation
    QUICKDouble* xc_grad;     //Exchange correlation energy gradient
    QUICKDouble* gxc_grad;
    QUICKDouble* wtang;
    QUICKDouble* rwt;
    QUICKDouble* rad3;

    int*  gatm;               //To which atom does a given grid point belongs to?
    int*  gatm_ssd;           //Parent atom index for sswder calculation
    int*  dweight;            //Dummy weight of grid points
    int*  dweight_ssd;        //Dummy weight of grid points for sswder 
    int*  basf;               //Basis function indices of all grid points
    int*  primf;              //Primitive function inidices of all grid points
    int*  primfpbin;             //Number of primitive functions per bin
    int*  basf_locator;       //Helps accessing b.f. indices of a grid point
    int*  primf_locator;      //Helps accessing p.f. indices of a b.f.   
    unsigned char*gpweight;   //keeps track of significant grid points for octree pruning
    unsigned int* cfweight;   //keeps track of significant b.f. for octree pruning
    unsigned int* pfweight;   //keeps track of significant p.f. for octree pruning

    int maxpfpbin;            //maximum number of primitive function per bin xc kernels
    int maxbfpbin;            //maximum number of basis function per bin in xc kernels

    // Gaussian Type function
    
    int*                            ncontract;
    int*                            itype;
    QUICKDouble*                    aexp;
    QUICKDouble*                    dcoeff;
    
    
    //charge and atom type
    int*                            iattype;
    QUICKDouble*                    chg;
    
    // Some more infos about basis function
    QUICKDouble*                    xyz;
/*
    int*                            first_basis_function;
    int*                            last_basis_function;
    int*                            first_shell_basis_function;
    int*                            last_shell_basis_function;
  */
    int*                            ncenter;
  
    int*                            kstart;
    int*                            katom;
//    int*                            ktype;
    int*                            kprim;
//    int*                            kshell;
    int*                            Ksumtype;
    int*                            Qnumber;
    int*                            Qstart;
    int*                            Qfinal;
    int*                            Qsbasis;
    int*                            Qfbasis;
    int*                            sorted_Qnumber;
    int*                            sorted_Q;
    QUICKDouble*                    gccoeff;
    QUICKDouble*                    cons;
    QUICKDouble*                    gcexpo;
    int*                            KLMN;
    int                             prim_total;
    int*                            prim_start;

    // Some more infos about pre-calculated values
    QUICKDouble*                    o;
    QUICKULL*                       oULL;
    QUICKDouble*                    dense;
    
    QUICKDouble*                    distance;
    QUICKDouble*                    Xcoeff;
    QUICKDouble*                    expoSum;
    QUICKDouble*                    weightedCenterX;
    QUICKDouble*                    weightedCenterY;
    QUICKDouble*                    weightedCenterZ;
    
    // cutoff
    int                             sqrQshell;
    int2*                           sorted_YCutoffIJ;
    QUICKDouble*                    cutMatrix;
    QUICKDouble*                    YCutoff;
    QUICKDouble*                    cutPrim;
    QUICKDouble                     integralCutoff;
    QUICKDouble                     primLimit;
    QUICKDouble                     DMCutoff;
    QUICKDouble                     gradCutoff;
    
    
    // for ERI generator
    ERI_entry**                     aoint_buffer;
    
    QUICKDouble                     maxIntegralCutoff;
    QUICKDouble                     leastIntegralCutoff;
    int                             iBatchSize;
    QUICKULL*                       intCount;
    
    // For Grad
    QUICKDouble*                    grad;
    QUICKULL*                       gradULL;
  
    // mpi variable definitions
    int                             mpirank;
    int                             mpisize;

    // multi-GPU variables
    char*                           mpi_bcompute;
    char*                           mpi_bxccompute;

    int                             mpi_xcstart;
    int                             mpi_xcend;
};

struct gpu_basis_type {
    int                             natom;
    int                             nbasis;
    int                             nshell;
    int                             nprim;
    int                             jshell;
    int                             jbasis;
    int                             Qshell;
    int                             maxcontract;
    int                             prim_total;
    
    int                             fStart;
    int                             ffStart;
    
    // Gaussian Type function

    cuda_buffer_type<int>*          ncontract;
    cuda_buffer_type<int>*          itype;
    cuda_buffer_type<QUICKDouble>*  aexp;
    cuda_buffer_type<QUICKDouble>*  dcoeff;
  
    // Some more infos about basis function
/*
    cuda_buffer_type<QUICKDouble>*  xyz;
    cuda_buffer_type<int>*          first_basis_function;
    cuda_buffer_type<int>*          last_basis_function;
    cuda_buffer_type<int>*          first_shell_basis_function;
    cuda_buffer_type<int>*          last_shell_basis_function;
*/
    cuda_buffer_type<int>*          ncenter;
    /*
    cuda_buffer_type<int>*          ktype;
    cuda_buffer_type<int>*          kshell;
  */
    cuda_buffer_type<QUICKDouble>*  sigrad2;
    cuda_buffer_type<int>*          kstart;
    cuda_buffer_type<int>*          katom;  
    cuda_buffer_type<int>*          kprim;
    cuda_buffer_type<int>*          Ksumtype;
    cuda_buffer_type<int>*          Qnumber;
    cuda_buffer_type<int>*          Qstart;
    cuda_buffer_type<int>*          Qfinal;
    cuda_buffer_type<int>*          Qsbasis;
    cuda_buffer_type<int>*          Qfbasis;
    cuda_buffer_type<int>*          sorted_Qnumber;
    cuda_buffer_type<int>*          sorted_Q;
    cuda_buffer_type<QUICKDouble>*  gccoeff;
    cuda_buffer_type<QUICKDouble>*  Xcoeff;                     // 4-dimension one
    cuda_buffer_type<QUICKDouble>*  expoSum;                    // 4-dimension one
    cuda_buffer_type<QUICKDouble>*  weightedCenterX;            // 4-dimension one
    cuda_buffer_type<QUICKDouble>*  weightedCenterY;            // 4-dimension one
    cuda_buffer_type<QUICKDouble>*  weightedCenterZ;            // 4-dimension one
    cuda_buffer_type<QUICKDouble>*  cons;
    cuda_buffer_type<QUICKDouble>*  gcexpo;
    cuda_buffer_type<int>*          KLMN;
    cuda_buffer_type<QUICKDouble>*  Apri;
    cuda_buffer_type<QUICKDouble>*  Kpri;
    cuda_buffer_type<QUICKDouble>*  PpriX;
    cuda_buffer_type<QUICKDouble>*  PpriY;
    cuda_buffer_type<QUICKDouble>*  PpriZ;
    cuda_buffer_type<int>*          prim_start;

    // For multi GPU version
    cuda_buffer_type<char>*           mpi_bcompute;

    void upload_all();
    
};


// a type to define a graphic card
struct gpu_type {

#if defined DEBUG || defined DEBUGTIME
    FILE                            *debugFile;
#endif

    SM_VERSION                      sm_version;
    
    // Memory parameters
    long long int                   totalCPUMemory; // total CPU memory allocated by CUDA part
    long long int                   totalGPUMemory; // total GPU memory allocated by CUDA part
    
    // Launch parameters
    int                             gpu_dev_id;  // set 0 for master GPU
    unsigned int                    blocks;
    unsigned int                    threadsPerBlock;
    unsigned int                    twoEThreadsPerBlock;
    unsigned int                    XCThreadsPerBlock;
    unsigned int                    gradThreadsPerBlock;
    unsigned int                    xc_blocks;	//Num of blocks for octree based dft implementation
    unsigned int                    xc_threadsPerBlock; //Num of threads/block for octree based dft implementation   

    // mpi variable definitions
    int                             mpirank;
    int                             mpisize;    

    // Molecule specification part
    int                             natom;
    int                             nbasis;
    int                             nElec;
    int                             imult;
    int                             molchg;
    int                             iAtomType;
    
    int                             nshell;
    int                             nprim;
    int                             jshell;
    int                             jbasis;
    int                             maxL;
    
    cuda_buffer_type<int>*          iattype;
    cuda_buffer_type<QUICKDouble>*  xyz;
    cuda_buffer_type<QUICKDouble>*  chg;
    cuda_buffer_type<DFT_calculated_type>*
                                    DFT_calculated;
    
    // For gradient
    cuda_buffer_type<QUICKDouble>*  grad;
    cuda_buffer_type<QUICKULL>*     gradULL;

    gpu_calculated_type*            gpu_calculated;
    gpu_basis_type*                 gpu_basis;
    gpu_cutoff_type*                gpu_cutoff;
    gpu_simulation_type             gpu_sim;
    XC_quadrature_type*             gpu_xcq;

    cuda_buffer_type<ERI_entry>**   aoint_buffer;
    
    cuda_buffer_type<QUICKULL>*     intCount;

    
/*    
    // Method
    cuda_gpu_type();
    ~cuda_gpu_type();
 */
};

typedef struct gpu_type *_gpu_type;
static _gpu_type gpu = NULL;

// template to pack buffered data for GPU-CPU communication
template <typename T>
struct cuda_buffer_type {
    bool            _bPinned;    // if pinned mem
    unsigned int    _length;     // length of the data
    unsigned int    _length2;    // length 2 is the row, and if it's not equals to 0, the data is a matrix
    T*              _hostData;   // data on host (CPU)
    T*              _devData;    // data on device (GPU)
    T*              _f90Data;    // if constructed from f90 array, it is the pointer
    
    // constructor
    cuda_buffer_type(int length);
    cuda_buffer_type(int length, bool bPinned);
    cuda_buffer_type(unsigned int length);
    cuda_buffer_type(int length, int length2);
    cuda_buffer_type(unsigned int length, unsigned int length2);
    cuda_buffer_type(T* f90data, int length, int length2);
    cuda_buffer_type(T* f90Data, unsigned int length, unsigned int length2);
    cuda_buffer_type(T* f90data, int length);
    cuda_buffer_type(T* f90Data, unsigned int length);
    
    // destructor
    virtual ~cuda_buffer_type();    
    
    // allocate and deallocate data
    void Allocate();
    void Deallocate();
    
    // use pinned data communication method. Upload and Download from host to device
    void Upload();      
    void Download();
    void Download(T* f90Data);
    
    void DeleteCPU();
    void DeleteGPU();
};


template <typename T>
cuda_buffer_type<T> :: cuda_buffer_type(int length) :
_length(length), _length2(1), _hostData(NULL), _devData(NULL), _f90Data(NULL), _bPinned(false)
{
    Allocate();
}

template <typename T>
cuda_buffer_type<T> :: cuda_buffer_type(int length, bool bPinned) :
_length(length), _length2(1), _hostData(NULL), _devData(NULL), _f90Data(NULL), _bPinned(bPinned)
{
    Allocate();
}

template <typename T>
cuda_buffer_type<T> :: cuda_buffer_type(unsigned int length) :
_length(length), _length2(1), _hostData(NULL), _devData(NULL), _f90Data(NULL), _bPinned(false)
{
    Allocate();
}

template <typename T>
cuda_buffer_type<T> :: cuda_buffer_type(int length, int length2) :
_length(length), _length2(length2), _hostData(NULL), _devData(NULL), _f90Data(NULL), _bPinned(false)
{
    Allocate();
}

template <typename T>
cuda_buffer_type<T> :: cuda_buffer_type(unsigned int length, unsigned int length2) :
_length(length), _length2(length2), _hostData(NULL), _devData(NULL), _f90Data(NULL), _bPinned(false)
{
    Allocate();
}

template <typename T>
cuda_buffer_type<T> :: cuda_buffer_type(T* f90data, unsigned int length, unsigned int length2) :
_length(length), _length2(length2), _hostData(NULL), _devData(NULL), _f90Data(f90data), _bPinned(false)
{
    Allocate();
}

template <typename T>
cuda_buffer_type<T> :: cuda_buffer_type(T* f90data, int length, int length2) :
_length(length), _length2(length2), _hostData(NULL), _devData(NULL), _f90Data(f90data), _bPinned(false)
{
    Allocate();
}

template <typename T>
cuda_buffer_type<T> :: cuda_buffer_type(T* f90data, unsigned int length) :
_length(length), _length2(1), _hostData(NULL), _devData(NULL), _f90Data(f90data), _bPinned(false)
{
    Allocate();
}

template <typename T>
cuda_buffer_type<T> :: cuda_buffer_type(T* f90data, int length) :
_length(length), _length2(1), _hostData(NULL), _devData(NULL), _f90Data(f90data), _bPinned(false)
{
    Allocate();
}


template <typename T>
cuda_buffer_type<T> :: ~cuda_buffer_type()
{
    Deallocate();
}

template <typename T>
void cuda_buffer_type<T> :: Allocate()
{
    
    PRINTDEBUG(">>BEGIN TO ALLOCATE TEMPLATE")

    if (! _f90Data) // if not constructed from f90 array
    {
        cudaError_t status;
        if (!_bPinned) {
            //Allocate GPU memeory
            status = cudaMalloc((void**)&_devData,_length*_length2*sizeof(T));
            PRINTERROR(status, " cudaMalloc cuda_buffer_type :: Allocate failed!");
            gpu->totalGPUMemory   += _length*_length2*sizeof(T);
            gpu->totalCPUMemory   += _length*_length2*sizeof(T);
            
            //Allocate CPU emembory
            _hostData = new T[_length*_length2];
            memset(_hostData, 0, _length*_length2*sizeof(T));
        }else{
            //Allocate GPU memeory
            status = cudaHostAlloc((void**)&_hostData, _length*_length2*sizeof(T),cudaHostAllocMapped);
            PRINTERROR(status, " cudaMalloc cuda_buffer_type :: Allocate failed!");
            gpu->totalGPUMemory   += _length*_length2*sizeof(T);
            gpu->totalCPUMemory   += _length*_length2*sizeof(T);
            
            //Allocate CPU emembory
            status = cudaHostGetDevicePointer((void **)&_devData, (void *)_hostData, 0);
            PRINTERROR(status, " cudaGetDevicePointer cuda_buffer_type :: Allocate failed!");
            memset(_hostData, 0, _length*_length2*sizeof(T));
        }
    }else {
        cudaError_t status;
        if (!_bPinned) {
            //Allocate GPU memeory
            status = cudaMalloc((void**)&_devData,_length*_length2*sizeof(T));
            PRINTERROR(status, " cudaMalloc cuda_buffer_type :: Allocate failed!");
            gpu->totalGPUMemory   += _length*_length2*sizeof(T);
            gpu->totalCPUMemory   += _length*_length2*sizeof(T);
            
            //Allocate CPU emembory
            _hostData = new T[_length*_length2];
            memset(_hostData, 0, _length*_length2*sizeof(T));
        }else{
            //Allocate GPU memeory
            status = cudaHostAlloc((void**)&_hostData, _length*_length2*sizeof(T),cudaHostAllocMapped);
            PRINTERROR(status, " cudaMalloc cuda_buffer_type :: Allocate failed!");
            gpu->totalGPUMemory   += _length*_length2*sizeof(T);
            gpu->totalCPUMemory   += _length*_length2*sizeof(T);
            
            //Allocate CPU emembory
            status = cudaHostGetDevicePointer((void **)&_devData, (void *)_hostData, 0);
            PRINTERROR(status, " cudaGetDevicePointer cuda_buffer_type :: Allocate failed!");
            memset(_hostData, 0, _length*_length2*sizeof(T));
        }
        
        // copy f90 data to _hostData
        size_t index_c = 0;
        size_t index_f = 0;
        for (size_t j=0; j<_length2; j++) {
            for (size_t i=0; i<_length; i++) {
                index_c = j * _length + i;
                _hostData[index_c] = _f90Data[index_f++];
            }
        }
        
    }
    PRINTMEM("ALLOCATE GPU MEMORY",(unsigned long long int)_length*_length2*sizeof(T))
    PRINTMEM("GPU++",gpu->totalGPUMemory);
    PRINTMEM("CPU++",gpu->totalCPUMemory);
    PRINTDEBUG("<<FINISH ALLOCATION TEMPLATE")
}

template <typename T>
void cuda_buffer_type<T> :: Deallocate()
{

    PRINTDEBUG(">>BEGIN TO DEALLOCATE TEMPLATE")
    if (!_bPinned) {
        
        if (_devData != NULL) {
            cudaError_t status;
            status = cudaFree(_devData);
            //	status = cudaFreeHost(_hostData);
            PRINTERROR(status, " cudaFree cuda_buffer_type :: Deallocate failed!");
            gpu->totalGPUMemory -= _length*_length2*sizeof(T);
        }
        
        if (_hostData != NULL) {
//            free(_hostData);
            delete [] _hostData;
            gpu->totalCPUMemory -= _length*_length2*sizeof(T);
        }
    }else{
        cudaError_t status;
        status = cudaFreeHost(_hostData);
        PRINTERROR(status, " cudaFree cuda_buffer_type :: Deallocate failed!");
        gpu->totalGPUMemory -= _length*_length2*sizeof(T);
        gpu->totalCPUMemory -= _length*_length2*sizeof(T);
    }
    _hostData = NULL;
    _devData = NULL;
    _f90Data = NULL;
#ifdef DEBUG
    
    PRINTMEM("GPU--",gpu->totalGPUMemory);
    PRINTMEM("CPU--",gpu->totalCPUMemory);
    
#endif
    PRINTDEBUG("<<FINSH DEALLOCATION TEMPLATE")

}

template <typename T>
void cuda_buffer_type<T> :: Upload()
{

    PRINTDEBUG(">>BEGIN TO UPLOAD TEMPLATE")

    cudaError_t status;
    status = cudaMemcpy(_devData,_hostData,_length*_length2*sizeof(T),cudaMemcpyHostToDevice);
    PRINTERROR(status, " cudaMemcpy cuda_buffer_type :: Upload failed!");
    PRINTDEBUG("<<FINISH UPLOADING TEMPLATE")

}

template <typename T>
void cuda_buffer_type<T> :: Download()
{

    PRINTDEBUG(">>BEGIN TO DOWNLOAD TEMPLATE")
    cudaError_t status;
    status = cudaMemcpy(_hostData,_devData,_length*_length2*sizeof(T),cudaMemcpyDeviceToHost);
    PRINTERROR(status, " cudaMemcpy cuda_buffer_type :: Download failed!");
    PRINTDEBUG("<<FINISH DOWNLOADING TEMPLATE")
}

template <typename T>
void cuda_buffer_type<T> :: Download(T* f90Data)
{
    PRINTDEBUG(">>BEGIN TO DOWNLOAD TEMPLATE TO FORTRAN ARRAY")
    size_t index_c = 0;
    size_t index_f;
    for (size_t i = 0; i < _length; i++) {
        for (size_t j = 0; j <_length2; j++) {
            index_f = j*_length+i;
            f90Data[index_f] = _hostData[index_c++];
        }
    }
    PRINTDEBUG("<<FINISH DOWNLOADING TEMPLATE TO FORTRAN ARRAY")
}

template <typename T>
void cuda_buffer_type<T> :: DeleteCPU()
{
    
    PRINTDEBUG(">>BEGIN TO DELETE CPU")
    
    if (_hostData != NULL) {
//        free(_hostData);
        delete [] _hostData;
        _hostData = NULL;
#ifdef DEBUG
        gpu->totalCPUMemory -= _length*_length2*sizeof(T);
        
        PRINTMEM("GPU  ",gpu->totalGPUMemory);
        PRINTMEM("CPU--",gpu->totalCPUMemory);
        
#endif
    }
    PRINTDEBUG("<<FINSH DELETE CPU")
}

template <typename T>
void cuda_buffer_type<T> :: DeleteGPU()
{
    
    PRINTDEBUG(">>BEGIN TO DELETE GPU")
    
    if (_devData != NULL) { 
        cudaFree(_devData);
        _devData = NULL;
#ifdef DEBUG
        gpu->totalGPUMemory -= _length*_length2*sizeof(T);
    
        PRINTMEM("GPU--",gpu->totalGPUMemory);
    	PRINTMEM("CPU  ",gpu->totalCPUMemory);
    
#endif
    }
    PRINTDEBUG("<<FINSH DELETE CPU")
}
