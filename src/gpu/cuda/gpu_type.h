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
#include "../gpu_common.h"
#include "gpu_utils.h"
#include "gpu_libxc_type.h"

#include <cuda.h>
//#include <cuda_runtime_api.h>

/*
 ****************************************************************
 *  gpu type and buffer type
 ****************************************************************
 */
template <typename T> struct gpu_buffer_type;
struct gpu_calculated_type {
    int natom;  // number of atom
    int nbasis; // number of basis sets
    gpu_buffer_type<QUICKDouble>* o;      // O matrix
    gpu_buffer_type<QUICKDouble>* ob;     // beta O matrix
    gpu_buffer_type<QUICKDouble>* dense;  // Density Matrix
    gpu_buffer_type<QUICKDouble>* denseb; // Beta Density Matrix
#if defined(USE_LEGACY_ATOMICS)
    gpu_buffer_type<QUICKULL>*     oULL;   // Unsigned long long int type O matrix
    gpu_buffer_type<QUICKULL>*     obULL;  // Unsigned long long int type Ob matrix
#endif
    gpu_buffer_type<QUICKDouble>* distance; // distance matrix
};

// struct to hold large temporary device arrays
struct gpu_scratch {
    gpu_buffer_type<QUICKDouble>* store;     // holds temporary primitive integrals in OEI and ERI algorithms
    gpu_buffer_type<QUICKDouble>* store2;    // holds temporary primitive integrals in OEI and ERI algorithms
    gpu_buffer_type<QUICKDouble>* storeAA;   // holds weighted temporary primitive integrals in OEI and ERI gradient algorithms
    gpu_buffer_type<QUICKDouble>* storeBB;   // holds weighted temporary primitive integrals in OEI and ERI gradient algorithms
    gpu_buffer_type<QUICKDouble>* storeCC;   // holds weighted temporary primitive integrals in OEI and ERI gradient algorithms
    gpu_buffer_type<QUICKDouble>* YVerticalTemp;  // holds boys function values
};

struct gpu_timer_type {
   double t_2elb; // time for eri load balancing in mgpu version
   double t_xclb; // time for xc load balancing in mgpu version
   double t_xcrb; // time for xc load re-balancing in mgpu version
   double t_xcpg; // grid pruning time
};

struct gpu_cutoff_type {
    int natom;
    int nbasis;
    int nshell;
    
    // the following are for pre-sorting cutoff
    int sqrQshell;
    gpu_buffer_type<int2>* sorted_YCutoffIJ;
    
    // Cutoff matrix
    gpu_buffer_type<QUICKDouble>* cutMatrix;
    gpu_buffer_type<QUICKDouble>* YCutoff;
    gpu_buffer_type<QUICKDouble>* cutPrim;
    
    // Cutoff criteria
    QUICKDouble integralCutoff;
    QUICKDouble coreIntegralCutoff;
    QUICKDouble primLimit;
    QUICKDouble DMCutoff;
    QUICKDouble XCCutoff;
    QUICKDouble gradCutoff;

    // One electron pre-sorting cutoff
    gpu_buffer_type<int2>* sorted_OEICutoffIJ;
};

struct DFT_calculated_type {
#if defined(USE_LEGACY_ATOMICS)
    QUICKULL Eelxc;      // exchange correction energy
    QUICKULL aelec;      // alpha electron
    QUICKULL belec;      // beta electron
#else
    QUICKDouble Eelxc;      // exchange correction energy
    QUICKDouble aelec;      // alpha electron
    QUICKDouble belec;      // beta electron
#endif
};

/*Madu Manathunga 11/21/2019*/
struct XC_quadrature_type {
	int npoints;								//Total number of packed grid points
	int nbins;									//Total number of bins
	int ntotbf;									//Total number of basis functions
	int ntotpf;									//Total number of primitive functions
	int bin_size;                               //Size of an octree bin

	gpu_buffer_type<QUICKDouble>* gridx;		//X coordinate of a grid point
	gpu_buffer_type<QUICKDouble>* gridy;		//Y coordinate of a grid point
	gpu_buffer_type<QUICKDouble>* gridz;		//Z coordinate of a grid point
	gpu_buffer_type<QUICKDouble>* sswt;		//A version of weight required for gradients
	gpu_buffer_type<QUICKDouble>* weight;		//Scuzeria weight of a grid point
	gpu_buffer_type<int>* gatm;			//To which atom does a given grid point belongs to?
        gpu_buffer_type<int>* bin_counter;            //Keeps track of bin borders 
	gpu_buffer_type<int>* dweight_ssd;            //Dummy weight of grid points for sswder 
	gpu_buffer_type<int>* basf;			//Basis function indices of all grid points
	gpu_buffer_type<int>* primf;			//Primitive function inidices of all grid points
        gpu_buffer_type<int>* primfpbin;                 //Number of primitive functions per bin
	gpu_buffer_type<int>* basf_locator;		//Helps accessing b.f. indices of a grid point
	gpu_buffer_type<int>* primf_locator;		//Helps accessing p.f. indices of a b.f.
        gpu_buffer_type<int>* bin_locator;            //Helps accessing bin of a grid point

	//Temporary variables
	gpu_buffer_type<QUICKDouble>* densa;
	gpu_buffer_type<QUICKDouble>* densb;
	gpu_buffer_type<QUICKDouble>* gax;
	gpu_buffer_type<QUICKDouble>* gbx;
	gpu_buffer_type<QUICKDouble>* gay;
	gpu_buffer_type<QUICKDouble>* gby;
	gpu_buffer_type<QUICKDouble>* gaz;
	gpu_buffer_type<QUICKDouble>* gbz;
	gpu_buffer_type<QUICKDouble>* exc;
	gpu_buffer_type<QUICKDouble>* xc_grad;
	gpu_buffer_type<QUICKDouble>* gxc_grad;        // a global xc gradient vector of size number_of_blocks * number_of_threads_per_block
        gpu_buffer_type<QUICKDouble>* phi;             // value of a basis function at a grid point 
        gpu_buffer_type<QUICKDouble>* dphidx;          // x gradient of a basis function at a grid point 
        gpu_buffer_type<QUICKDouble>* dphidy;          // y gradient of a basis function at a grid point
        gpu_buffer_type<QUICKDouble>* dphidz;          // z gradient of a basis function at a grid point  
        gpu_buffer_type<unsigned int>* phi_loc;       // stores locations of phi array for each grid point

        //Variables for ssw derivative calculation
        int npoints_ssd; //Total number of input points for ssd

        gpu_buffer_type<QUICKDouble>* gridx_ssd;       //X coordinate of a grid point
        gpu_buffer_type<QUICKDouble>* gridy_ssd;       //Y coordinate of a grid point
        gpu_buffer_type<QUICKDouble>* gridz_ssd;       //Z coordinate of a grid point
        gpu_buffer_type<QUICKDouble>* exc_ssd;
        gpu_buffer_type<QUICKDouble>* quadwt;          //quadrature weight
        gpu_buffer_type<int>* gatm_ssd;               //To which atom does a given grid point belongs to?
	gpu_buffer_type<QUICKDouble>* uw_ssd;          //Holds unnormalized weights during ssd calculation
	
	//Variables for grid weight calculation
	gpu_buffer_type<QUICKDouble>* wtang;
	gpu_buffer_type<QUICKDouble>* rwt;
	gpu_buffer_type<QUICKDouble>* rad3;

	//Variables for obtaining octree info 
	gpu_buffer_type<unsigned char>* gpweight;     //keeps track of significant grid points for octree pruning
	gpu_buffer_type<unsigned int>* cfweight;     //keeps track of significant b.f. for octree pruning 
	gpu_buffer_type<unsigned int>* pfweight;     //keeps track of significant p.f. for octree pruning

        // mpi variables
        gpu_buffer_type<char>* mpi_bxccompute;

        // shared memory size
        int smem_size;                                 //size of shared memory buffer in xc kernels 
};

struct lri_data_type {
    int zeta;
    gpu_buffer_type<QUICKDouble>* cc;
    gpu_buffer_type<QUICKDouble>* vrecip;
};

struct gpu_simulation_type {
    // basic molecule information and method information
    QUICK_METHOD                    method;
    DFT_calculated_type*            DFT_calculated;
    XC_quadrature_type*             xcq;
    QUICKDouble                     hyb_coeff;   
    bool                            is_oshell;
 
    // used for DFT
    int                             isg;        // isg algrothm
    bool                            prePtevl;   // precompute and store values and gradients of basis functions at grid points
    QUICKDouble*                    sigrad2;    // basis set range
    
    int                             natom;
    int                             nextatom;
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
    int                             Qshell_OEI; // number of Qshell pairs after OEI prescreening

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
    QUICKDouble* phi;             // value of a basis function at a grid point 
    QUICKDouble* dphidx;          // x gradient of a basis function at a grid point 
    QUICKDouble* dphidy;          // y gradient of a basis function at a grid point
    QUICKDouble* dphidz;          // z gradient of a basis function at a grid point
    unsigned int* phi_loc;        // stores locations of phi array for each grid point      

    int*  gatm;               //To which atom does a given grid point belongs to?
    int*  gatm_ssd;           //Parent atom index for sswder calculation
    int*  dweight;            //Dummy weight of grid points
    int*  dweight_ssd;        //Dummy weight of grid points for sswder 
    int*  basf;               //Basis function indices of all grid points
    int*  primf;              //Primitive function inidices of all grid points
    int*  primfpbin;             //Number of primitive functions per bin
    int*  basf_locator;       //Helps accessing b.f. indices of a grid point
    int*  primf_locator;      //Helps accessing p.f. indices of a b.f.   
    int*  bin_locator;        //Helps accessing bin of a grid point
    unsigned char*gpweight;   //keeps track of significant grid points for octree pruning
    unsigned int* cfweight;   //keeps track of significant b.f. for octree pruning
    unsigned int* pfweight;   //keeps track of significant p.f. for octree pruning

    int maxpfpbin;            //maximum number of primitive function per bin xc kernels
    int maxbfpbin;            //maximum number of basis function per bin in xc kernels

    // libxc data
    gpu_libxc_info** glinfo;        // pointer to an array hosting gpu_libxc_info type pointers
    int nauxfunc;                   // number of auxilary functions, equal to the length of array pointed by glinfo     

    // Gaussian Type function
    
    int*                            ncontract;
    int*                            itype;
    QUICKDouble*                    aexp;
    QUICKDouble*                    dcoeff;
    
    
    //charge and atom type
    int*                            iattype;
    QUICKDouble*                    chg;
    QUICKDouble*                    allchg; // charges of nuclei and external charges for oei
    
    // Some more infos about basis function
    QUICKDouble*                    xyz;
    QUICKDouble*                    allxyz; // coordinates of nuclei and external charges for oei
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
    unsigned char*                  KLMN;
    int                             prim_total;
    int*                            prim_start;

    // Some more infos about pre-calculated values
    QUICKDouble*                    o;
    QUICKDouble*                    ob;
    QUICKULL*                       oULL;
    QUICKULL*                       obULL;
    QUICKDouble*                    dense;
    QUICKDouble*                    denseb;
    
    QUICKDouble*                    distance;
    QUICKDouble*                    Xcoeff;
    QUICKDouble*                    Xcoeff_oei; // precomputed overlap prefactor for oei
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
    QUICKDouble                     coreIntegralCutoff;
    QUICKDouble                     primLimit;
    QUICKDouble                     DMCutoff;
    QUICKDouble                     XCCutoff;
    QUICKDouble                     gradCutoff;
    int2*                           sorted_OEICutoffIJ;
    
    
    // for ERI generator
    ERI_entry**                     aoint_buffer;
    
    QUICKDouble                     maxIntegralCutoff;
    QUICKDouble                     leastIntegralCutoff;
    int                             iBatchSize;
    QUICKULL*                       intCount;
    
    // For Grad
    QUICKDouble*                    grad;
    QUICKDouble*                    ptchg_grad;
    QUICKULL*                       gradULL;
    QUICKULL*                       ptchg_gradULL;
  
    // mpi variable definitions
    int                             mpirank;
    int                             mpisize;

    // multi-GPU variables
    unsigned char*                           mpi_bcompute;
    char*                           mpi_bxccompute;
    unsigned char*                           mpi_boeicompute;

    int                             mpi_xcstart;
    int                             mpi_xcend;

    // pointers to temporary data structures
    QUICKDouble*                    store;
    QUICKDouble*                    store2;
    QUICKDouble*                    storeAA;
    QUICKDouble*                    storeBB;
    QUICKDouble*                    storeCC;
    QUICKDouble*                    YVerticalTemp;

    // for long range integrals
    QUICKDouble                     lri_zeta;
    QUICKDouble*                    lri_cc;
    QUICKDouble*                    cew_vrecip;
    bool                            use_cew;

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

    gpu_buffer_type<int>*          ncontract;
    gpu_buffer_type<int>*          itype;
    gpu_buffer_type<QUICKDouble>*  aexp;
    gpu_buffer_type<QUICKDouble>*  dcoeff;
  
    // Some more infos about basis function
/*
    gpu_buffer_type<QUICKDouble>*  xyz;
    gpu_buffer_type<int>*          first_basis_function;
    gpu_buffer_type<int>*          last_basis_function;
    gpu_buffer_type<int>*          first_shell_basis_function;
    gpu_buffer_type<int>*          last_shell_basis_function;
*/
    gpu_buffer_type<int>*          ncenter;
    /*
    gpu_buffer_type<int>*          ktype;
    gpu_buffer_type<int>*          kshell;
  */
    gpu_buffer_type<QUICKDouble>*  sigrad2;
    gpu_buffer_type<int>*          kstart;
    gpu_buffer_type<int>*          katom;  
    gpu_buffer_type<int>*          kprim;
    gpu_buffer_type<int>*          Ksumtype;
    gpu_buffer_type<int>*          Qnumber;
    gpu_buffer_type<int>*          Qstart;
    gpu_buffer_type<int>*          Qfinal;
    gpu_buffer_type<int>*          Qsbasis;
    gpu_buffer_type<int>*          Qfbasis;
    gpu_buffer_type<int>*          sorted_Qnumber;
    gpu_buffer_type<int>*          sorted_Q;
    gpu_buffer_type<QUICKDouble>*  gccoeff;
    gpu_buffer_type<QUICKDouble>*  Xcoeff;                     // 4-dimension one
    gpu_buffer_type<QUICKDouble>*  Xcoeff_oei;                 // 4-dimension one, precomputed overlap prefactor for oei
    gpu_buffer_type<QUICKDouble>*  expoSum;                    // 4-dimension one
    gpu_buffer_type<QUICKDouble>*  weightedCenterX;            // 4-dimension one
    gpu_buffer_type<QUICKDouble>*  weightedCenterY;            // 4-dimension one
    gpu_buffer_type<QUICKDouble>*  weightedCenterZ;            // 4-dimension one
    gpu_buffer_type<QUICKDouble>*  cons;
    gpu_buffer_type<QUICKDouble>*  gcexpo;
    gpu_buffer_type<unsigned char>* KLMN;
    gpu_buffer_type<QUICKDouble>*  Apri;
    gpu_buffer_type<QUICKDouble>*  Kpri;
    gpu_buffer_type<QUICKDouble>*  PpriX;
    gpu_buffer_type<QUICKDouble>*  PpriY;
    gpu_buffer_type<QUICKDouble>*  PpriZ;
    gpu_buffer_type<int>*          prim_start;

    // For multi GPU version
    gpu_buffer_type<unsigned char>*           mpi_bcompute;
    gpu_buffer_type<unsigned char>*           mpi_boeicompute;

    void upload_all();
    
};


// a type to define a graphic card
struct gpu_type {

#if defined DEBUG || defined DEBUGTIME
    FILE                            *debugFile;
#endif

    SM_VERSION                      sm_version;
    
    // Memory parameters
    long long int                   totalCPUMemory; // total CPU memory allocated
    long long int                   totalGPUMemory; // total GPU memory allocated
    
    // Launch parameters
    int                             gpu_dev_id;  // set 0 for master GPU
    unsigned int                    blocks;
    unsigned int                    threadsPerBlock;
    unsigned int                    twoEThreadsPerBlock;
    unsigned int                    XCThreadsPerBlock;
    unsigned int                    gradThreadsPerBlock;
    unsigned int                    xc_blocks;	//Num of blocks for octree based dft implementation
    unsigned int                    xc_threadsPerBlock; //Num of threads/block for octree based dft implementation   
    unsigned int                    sswGradThreadsPerBlock;

    // mpi variable definitions
    int                             mpirank;
    int                             mpisize;    

    // timer 
    gpu_timer_type*                 timer;

    // Molecule specification part
    int                             natom;
    int                             nextatom;
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
    
    gpu_buffer_type<int>*          iattype;
    gpu_buffer_type<QUICKDouble>*  xyz;
    gpu_buffer_type<QUICKDouble>*  allxyz; // coordinates of nuclei and external point charges
    gpu_buffer_type<QUICKDouble>*  chg;
    gpu_buffer_type<QUICKDouble>*  allchg; // charges of nuclei and external point charges
    gpu_buffer_type<DFT_calculated_type>*
                                    DFT_calculated;
    
    // For gradient
    gpu_buffer_type<QUICKDouble>*  grad;
    gpu_buffer_type<QUICKDouble>*  ptchg_grad;
    gpu_buffer_type<QUICKULL>*     gradULL;
    gpu_buffer_type<QUICKULL>*     ptchg_gradULL;
    gpu_buffer_type<QUICKDouble>*  cew_grad;

    gpu_calculated_type*            gpu_calculated;
    gpu_basis_type*                 gpu_basis;
    gpu_cutoff_type*                gpu_cutoff;
    gpu_simulation_type             gpu_sim;
    XC_quadrature_type*             gpu_xcq;

    gpu_buffer_type<ERI_entry>**   aoint_buffer;
    
    gpu_buffer_type<QUICKULL>*     intCount;

    gpu_scratch*                    scratch;
    
    lri_data_type*                  lri_data;
};

typedef struct gpu_type *_gpu_type;
static _gpu_type gpu = NULL;

// template to pack buffered data for GPU-CPU communication
template <typename T>
struct gpu_buffer_type {
    bool            _bPinned;    // if pinned mem
    unsigned int    _length;     // length of the data
    unsigned int    _length2;    // length 2 is the row, and if it's not equals to 0, the data is a matrix
    T*              _hostData;   // data on host (CPU)
    T*              _devData;    // data on device (GPU)
    T*              _f90Data;    // if constructed from f90 array, it is the pointer
    
    // constructor
    gpu_buffer_type(int length);
    gpu_buffer_type(int length, bool bPinned);
    gpu_buffer_type(unsigned int length);
    gpu_buffer_type(int length, int length2);
    gpu_buffer_type(unsigned int length, unsigned int length2);
    gpu_buffer_type(T* f90data, int length, int length2);
    gpu_buffer_type(T* f90Data, unsigned int length, unsigned int length2);
    gpu_buffer_type(T* f90data, int length);
    gpu_buffer_type(T* f90Data, unsigned int length);
    
    // destructor
    virtual ~gpu_buffer_type();    
    
    // allocate and deallocate data
    void Allocate();
    void Deallocate();

    // reallocate device memory for an existing template
    void ReallocateGPU();
    
    // use pinned data communication method. Upload and Download from host to device
    void Upload();      
    void UploadAsync();
    void Download();
    void Download(T* f90Data);
    void DownloadSum(T* f90Data);
    
    void DeleteCPU();
    void DeleteGPU();
};


template <typename T>
gpu_buffer_type<T> :: gpu_buffer_type(int length) :
_length(length), _length2(1), _hostData(NULL), _devData(NULL), _f90Data(NULL), _bPinned(false)
{
    Allocate();
}

template <typename T>
gpu_buffer_type<T> :: gpu_buffer_type(int length, bool bPinned) :
_length(length), _length2(1), _hostData(NULL), _devData(NULL), _f90Data(NULL), _bPinned(bPinned)
{
    Allocate();
}

template <typename T>
gpu_buffer_type<T> :: gpu_buffer_type(unsigned int length) :
_length(length), _length2(1), _hostData(NULL), _devData(NULL), _f90Data(NULL), _bPinned(false)
{
    Allocate();
}

template <typename T>
gpu_buffer_type<T> :: gpu_buffer_type(int length, int length2) :
_length(length), _length2(length2), _hostData(NULL), _devData(NULL), _f90Data(NULL), _bPinned(false)
{
    Allocate();
}

template <typename T>
gpu_buffer_type<T> :: gpu_buffer_type(unsigned int length, unsigned int length2) :
_length(length), _length2(length2), _hostData(NULL), _devData(NULL), _f90Data(NULL), _bPinned(false)
{
    Allocate();
}

template <typename T>
gpu_buffer_type<T> :: gpu_buffer_type(T* f90data, unsigned int length, unsigned int length2) :
_length(length), _length2(length2), _hostData(NULL), _devData(NULL), _f90Data(f90data), _bPinned(false)
{
    Allocate();
}

template <typename T>
gpu_buffer_type<T> :: gpu_buffer_type(T* f90data, int length, int length2) :
_length(length), _length2(length2), _hostData(NULL), _devData(NULL), _f90Data(f90data), _bPinned(false)
{
    Allocate();
}

template <typename T>
gpu_buffer_type<T> :: gpu_buffer_type(T* f90data, unsigned int length) :
_length(length), _length2(1), _hostData(NULL), _devData(NULL), _f90Data(f90data), _bPinned(false)
{
    Allocate();
}

template <typename T>
gpu_buffer_type<T> :: gpu_buffer_type(T* f90data, int length) :
_length(length), _length2(1), _hostData(NULL), _devData(NULL), _f90Data(f90data), _bPinned(false)
{
    Allocate();
}


template <typename T>
gpu_buffer_type<T> :: ~gpu_buffer_type()
{
    Deallocate();
}

template <typename T>
void gpu_buffer_type<T> :: Allocate()
{
    
    PRINTDEBUG(">>BEGIN TO ALLOCATE TEMPLATE")

    if (! _f90Data) // if not constructed from f90 array
    {
        cudaError_t status;
        if (!_bPinned) {
            //Allocate GPU memeory
            status = cudaMalloc((void**)&_devData,_length*_length2*sizeof(T));
            PRINTERROR(status, " cudaMalloc gpu_buffer_type :: Allocate failed!");
            gpu->totalGPUMemory   += _length*_length2*sizeof(T);
            gpu->totalCPUMemory   += _length*_length2*sizeof(T);
            
            //Allocate CPU emembory
            _hostData = new T[_length*_length2];
            memset(_hostData, 0, _length*_length2*sizeof(T));
        }else{
            //Allocate GPU memeory
            status = cudaHostAlloc((void**)&_hostData, _length*_length2*sizeof(T),cudaHostAllocMapped);
            PRINTERROR(status, " cudaMalloc gpu_buffer_type :: Allocate failed!");
            gpu->totalGPUMemory   += _length*_length2*sizeof(T);
            gpu->totalCPUMemory   += _length*_length2*sizeof(T);
            
            //Allocate CPU emembory
            status = cudaHostGetDevicePointer((void **)&_devData, (void *)_hostData, 0);
            PRINTERROR(status, " cudaHostGetDevicePointer gpu_buffer_type :: Allocate failed!");
            memset(_hostData, 0, _length*_length2*sizeof(T));
        }
    }else {
        cudaError_t status;
        if (!_bPinned) {
            //Allocate GPU memeory
            status = cudaMalloc((void**)&_devData,_length*_length2*sizeof(T));
            PRINTERROR(status, " cudaMalloc gpu_buffer_type :: Allocate failed!");
            gpu->totalGPUMemory   += _length*_length2*sizeof(T);
            gpu->totalCPUMemory   += _length*_length2*sizeof(T);
            
            //Allocate CPU emembory
            _hostData = new T[_length*_length2];
            memset(_hostData, 0, _length*_length2*sizeof(T));
        }else{
            //Allocate GPU memeory
            status = cudaHostAlloc((void**)&_hostData, _length*_length2*sizeof(T),cudaHostAllocMapped);
            PRINTERROR(status, " cudaMalloc gpu_buffer_type :: Allocate failed!");
            gpu->totalGPUMemory   += _length*_length2*sizeof(T);
            gpu->totalCPUMemory   += _length*_length2*sizeof(T);
            
            //Allocate CPU emembory
            status = cudaHostGetDevicePointer((void **)&_devData, (void *)_hostData, 0);
            PRINTERROR(status, " cudaHostGetDevicePointer gpu_buffer_type :: Allocate failed!");
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
void gpu_buffer_type<T> :: ReallocateGPU()
{
  PRINTDEBUG(">>BEGIN TO REALLOCATE GPU")

   cudaError_t status;

   if (_devData == NULL) // if memory has not been allocated
    {
        if (!_bPinned) {
            //Allocate GPU memeory
            status = cudaMalloc((void**)&_devData,_length*_length2*sizeof(T));
            PRINTERROR(status, " cudaMalloc gpu_buffer_type :: Allocate failed!");
            gpu->totalGPUMemory   += _length*_length2*sizeof(T);

        }else{
            //Allocate GPU memeory
            status = cudaHostAlloc((void**)&_hostData, _length*_length2*sizeof(T),cudaHostAllocMapped);
            PRINTERROR(status, " cudaMalloc gpu_buffer_type :: Allocate failed!");
            gpu->totalGPUMemory   += _length*_length2*sizeof(T);

        }
    }else{
#ifndef CUDART_VERSION
       status=cudaErrorInvalidValue;
#elif (CUDART_VERSION < 10010)
       status=cudaErrorInvalidDevicePointer;
#else
       status=cudaErrorNotMappedAsPointer;
#endif
        PRINTERROR(status, " cudaMalloc gpu_buffer_type :: Reallocation failed!");
    }

    PRINTMEM("ALLOCATE GPU MEMORY",(unsigned long long int)_length*_length2*sizeof(T))
    PRINTMEM("GPU++",gpu->totalGPUMemory);
    PRINTMEM("CPU  ",gpu->totalCPUMemory);
    PRINTDEBUG("<<FINISHED ALLOCATING GPU")
}


template <typename T>
void gpu_buffer_type<T> :: Deallocate()
{

    PRINTDEBUG(">>BEGIN TO DEALLOCATE TEMPLATE")
    if (!_bPinned) {
        
        if (_devData != NULL) {
            cudaError_t status;
            status = cudaFree(_devData);
            //	status = cudaFreeHost(_hostData);
            PRINTERROR(status, " cudaFree gpu_buffer_type :: Deallocate failed!");
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
        PRINTERROR(status, " cudaFree gpu_buffer_type :: Deallocate failed!");
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
void gpu_buffer_type<T> :: Upload()
{

    PRINTDEBUG(">>BEGIN TO UPLOAD TEMPLATE")

    cudaError_t status;
    status = cudaMemcpy(_devData,_hostData,_length*_length2*sizeof(T),cudaMemcpyHostToDevice);
    PRINTERROR(status, " cudaMemcpy gpu_buffer_type :: Upload failed!");
    PRINTDEBUG("<<FINISH UPLOADING TEMPLATE")

}

template <typename T>
void gpu_buffer_type<T> :: UploadAsync()
{

    PRINTDEBUG(">>BEGIN TO UPLOAD TEMPLATE")

    cudaError_t status;
    status = cudaMemcpyAsync(_devData,_hostData,_length*_length2*sizeof(T),cudaMemcpyHostToDevice);
    PRINTERROR(status, " cudaMemcpy gpu_buffer_type :: Upload failed!");
    PRINTDEBUG("<<FINISH UPLOADING TEMPLATE")

}

template <typename T>
void gpu_buffer_type<T> :: Download()
{

    PRINTDEBUG(">>BEGIN TO DOWNLOAD TEMPLATE")
    cudaError_t status;
    status = cudaMemcpy(_hostData,_devData,_length*_length2*sizeof(T),cudaMemcpyDeviceToHost);
    PRINTERROR(status, " cudaMemcpy gpu_buffer_type :: Download failed!");
    PRINTDEBUG("<<FINISH DOWNLOADING TEMPLATE")
}

template <typename T>
void gpu_buffer_type<T> :: Download(T* f90Data)
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
void gpu_buffer_type<T> :: DownloadSum(T* f90Data)
{
    PRINTDEBUG(">>BEGIN TO DOWNLOAD TEMPLATE TO FORTRAN ARRAY")
    size_t index_c = 0;
    size_t index_f;
    for (size_t i = 0; i < _length; i++) {
        for (size_t j = 0; j <_length2; j++) {
            index_f = j*_length+i;
            f90Data[index_f] += _hostData[index_c++];
        }
    }
    PRINTDEBUG("<<FINISH DOWNLOADING TEMPLATE TO FORTRAN ARRAY")
}

template <typename T>
void gpu_buffer_type<T> :: DeleteCPU()
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
void gpu_buffer_type<T> :: DeleteGPU()
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
