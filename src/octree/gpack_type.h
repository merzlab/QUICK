#ifndef QUICK_GPACK_COMMON_H
#define QUICK_GPACK_COMMON_H

#include "../config.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <iostream>
using namespace std;

// setup a debug file
#ifdef DEBUG
static FILE *gpackDebugFile = NULL;
#endif

//Some useful macros
#ifdef DEBUG
#define PRINTOCTDEBUG(s) \
{\
    fprintf(gpackDebugFile,"FILE:%15s, LINE:%5d DATE: %s TIME:%s DEBUG : %s. \n", __FILE__,__LINE__,__DATE__,__TIME__,s );\
}

#define PRINTOCTTIME(s,time)\
{\
    fprintf(gpackDebugFile,"TIME:%15s, LINE:%5d DATE: %s TIME:%s TIMING:%20s ======= %f s =======.\n", __FILE__, __LINE__, __DATE__,__TIME__,s,time);\
}

#define PRINTOCTMEM(s,a) \
{\
        fprintf(gpackDebugFile,"MEM :%15s, LINE:%5d DATE: %s TIME:%s MEM   : %10s %lli BYTES\n", __FILE__,__LINE__,__DATE__,__TIME__,s,a);\
}

#define PRINTOCTMEMCOUNT(s,a,b) \
{\
        fprintf(gpackDebugFile,"MEM :%15s, LINE:%5d DATE: %s TIME:%s %10s : %5d ELEMENTS, %lli BYTES\n", __FILE__,__LINE__,__DATE__,__TIME__,s,a,b);\
}
#else
#define PRINTOCTDEBUG(s)
#define PRINTOCTTIME(s,time)
#define PRINTOCTMEM(s,a)
#define PRINTOCTMEMCOUNT(s,a,b)
#endif


template <typename T> struct gpack_buffer_type;

struct gpack_type{

   long long int totalGPACKMemory;   // keeps track of total memory allocated by octree code

   gpack_buffer_type<double>* gridx; // gridx_in, gridy_in, gridz_in: xyz coordinates of grid points
   gpack_buffer_type<double>* gridy;
   gpack_buffer_type<double>* gridz;
   gpack_buffer_type<double>* sswt;  // sswt_in, ss_weight_in: ss weights of grid points 
   gpack_buffer_type<double>* ss_weight;
   gpack_buffer_type<int>* grid_atm; // a set of atomic indices, helps to identify which grid point belong to which atom  

   int arr_size; // size of the grid arrays
   int natoms;   // number of atoms
   int nbasis;   // total number of basis functions
   int maxcontract; // maximum number of contractions
   double DMCutoff; // Density matrix cut off
   gpack_buffer_type<double>* sigrad2; // square of the radius of sigificance
   gpack_buffer_type<int>* ncontract; // number of contraction functions
   gpack_buffer_type<double>* aexp; // alpha values of the gaussian primivite function exponents
   gpack_buffer_type<double>* dcoeff; // Contraction coefficients
   gpack_buffer_type<double>* xyz; // xyz coordinates of atomic positions
   gpack_buffer_type<int>* ncenter; // centers of the basis functions
   gpack_buffer_type<int>* itype;

   gpack_buffer_type<double>* gridxb; // gridxb_out, gridyb_out, gridzb_out: binned grid x, y and z grid points
   gpack_buffer_type<double>* gridyb;
   gpack_buffer_type<double>* gridzb;
   gpack_buffer_type<double>* gridb_sswt; // sswt_out, ss_weight_out: binned ss weights
   gpack_buffer_type<double>* gridb_weight;
   gpack_buffer_type<int>* gridb_atm;
   gpack_buffer_type<int>* dweight; // an array indicating if a binned grid point is true or a dummy grid point
   gpack_buffer_type<int>* basf;    // array of basis functions belonging to each bin
   gpack_buffer_type<int>* primf;   // array of primitive functions beloning to binned basis functions
   gpack_buffer_type<int>* basf_counter;  // a counter to keep track of which basis functions belong to which bin
   gpack_buffer_type<int>* primf_counter; // a counter to keep track of which primitive functions belong to which basis function
   gpack_buffer_type<int>* bin_counter;   // a counter to keep track of bins with different number of points in cpu implementation
   int gridb_count; // length of binned grid arrays
   int nbins;       // number of bins
   int nbtotbf;     // total number of basis functions
   int nbtotpf;     // total number of primitive functions
   int ntgpts;  // total number of true grid points(i.e. excluding dummy points)

   double time_octree; // time for running octree algorithm
   double time_bfpf_prescreen; // time for prescreening basis and primitive functions

#ifdef DEBUG
   FILE *gpackDebugFile;
#endif

};

typedef struct gpack_type *_gpack_type;
static _gpack_type gps = NULL;


//template to pack data for octree run
template <typename T>
struct gpack_buffer_type{

    unsigned int    _length;     // length of the data
    unsigned int    _length2;    // length 2 is the row, and if it's not equals to 0, the data is a matrix
    T*              _cppData;   // data on host (CPU)
    T*              _f90Data;    // if constructed from f90 array, it is the pointer


    //constructor
    gpack_buffer_type(int length);
    gpack_buffer_type(unsigned int length);
    gpack_buffer_type(int length, int length2);
    gpack_buffer_type(unsigned int length, unsigned int length2);
    gpack_buffer_type(T* f90data, int length, int length2);
    gpack_buffer_type(T* f90Data, unsigned int length, unsigned int length2);
    gpack_buffer_type(T* f90data, int length);
    gpack_buffer_type(T* f90Data, unsigned int length);

    // destructor
    virtual ~gpack_buffer_type();

    // allocate and deallocate data
    void Allocate();
    void Deallocate();
    void Transfer(T* f90Data);

    void DeleteCPU();
    
};


template <typename T>
gpack_buffer_type<T> :: gpack_buffer_type(int length) :
_length(length), _length2(1), _cppData(NULL), _f90Data(NULL)
{
    Allocate();
}

template <typename T>
gpack_buffer_type<T> :: gpack_buffer_type(unsigned int length) :
_length(length), _length2(1), _cppData(NULL), _f90Data(NULL)
{
    Allocate();
}

template <typename T>
gpack_buffer_type<T> :: gpack_buffer_type(int length, int length2) :
_length(length), _length2(length2), _cppData(NULL), _f90Data(NULL)
{
    Allocate();
}

template <typename T>
gpack_buffer_type<T> :: gpack_buffer_type(unsigned int length, unsigned int length2) :
_length(length), _length2(length2), _cppData(NULL), _f90Data(NULL)
{
    Allocate();
}

//create arrays based on f90 data

template <typename T>
gpack_buffer_type<T> :: gpack_buffer_type(T* f90data, unsigned int length, unsigned int length2) :
_length(length), _length2(length2), _cppData(NULL), _f90Data(f90data)
{
    Allocate();
}

template <typename T>
gpack_buffer_type<T> :: gpack_buffer_type(T* f90data, int length, int length2) :
_length(length), _length2(length2), _cppData(NULL), _f90Data(f90data)
{
    Allocate();
}

template <typename T>
gpack_buffer_type<T> :: gpack_buffer_type(T* f90data, unsigned int length) :
_length(length), _length2(1), _cppData(NULL), _f90Data(f90data)
{
    Allocate();
}

template <typename T>
gpack_buffer_type<T> :: gpack_buffer_type(T* f90data, int length) :
_length(length), _length2(1), _cppData(NULL), _f90Data(f90data)
{
    Allocate();
}

template <typename T>
gpack_buffer_type<T> :: ~gpack_buffer_type()
{
    Deallocate();
}

//Template for allocating memory
template <typename T>
void gpack_buffer_type<T> :: Allocate()
{

    gps->totalGPACKMemory += _length*_length2*sizeof(T);
    _cppData = new T[_length*_length2];
    memset(_cppData, 0, _length*_length2*sizeof(T));

    if (_f90Data) // if not constructed from f90 array
    {     
        // copy f90 data to _cppData
        size_t index_c = 0;
        size_t index_f = 0;
        for (size_t j=0; j<_length2; j++) {
            for (size_t i=0; i<_length; i++) {
                index_c = j * _length + i;
                _cppData[index_c] = _f90Data[index_f++];
            }
        }         
    }

    PRINTOCTMEMCOUNT("ALLOCATING",_length*_length2,(unsigned long long int)_length*_length2*sizeof(T))
    PRINTOCTMEM("TOTAL OCTREE MEMORY++", gps->totalGPACKMemory)

#ifdef DEBUG
//    printf(">> ALLOCATING OCTREE MEMORY %i, NUMBER OF ELEMENTS %i \n", (unsigned long long int)_length*_length2*sizeof(T), _length*_length2);
//    printf("TOTAL OCTREE MEMORY++ %i \n",gps->totalGPACKMemory);
#endif

}

template <typename T>
void gpack_buffer_type<T> :: Deallocate()
{
    if (_cppData != NULL) {
        delete [] _cppData;
        gps->totalGPACKMemory -= _length*_length2*sizeof(T);
    }

    _cppData = NULL;
    _f90Data = NULL;

    PRINTOCTMEMCOUNT("DEALLOCATING",_length*_length2,(unsigned long long int)_length*_length2*sizeof(T))
    PRINTOCTMEM("TOTAL OCTREE MEMORY--", gps->totalGPACKMemory)

#ifdef DEBUG
//    printf(">> DEALLOCATING OCTREE MEMORY %i, NUMBER OF ELEMENTS %i \n", (unsigned long long int)_length*_length2*sizeof(T), _length*_length2);
//    printf(">> TOTAL OCTREE MEMORY-- %i \n",gps->totalGPACKMemory);
#endif
}

// Load data back into f90 array
template <typename T>
void gpack_buffer_type<T> :: Transfer(T* f90Data)
{
    size_t index_c = 0;
    size_t index_f;
    for (size_t i = 0; i < _length; i++) {
        for (size_t j = 0; j <_length2; j++) {
            index_f = j*_length+i;
            f90Data[index_f] = _cppData[index_c++];
        }
    }

    PRINTOCTMEMCOUNT("TRANSFERRING TO F90 SIDE",_length*_length2,(unsigned long long int)_length*_length2*sizeof(T))

#ifdef DEBUG
//    printf(">> TRANSFERRING %i BYTES TO F90 SIDE, NUMBER OF ELEMENTS %i \n", (unsigned long long int)_length*_length2*sizeof(T), _length*_length2);
#endif
}


#endif
