#include "hip/hip_runtime.h"
//
//  gpu_get2e_subs.h
//  new_quick 2
//
//  Created by Yipu Miao on 6/18/13.
//
//
#include "gpu_common.h"

#undef STOREDIM

#ifdef int_spd
#define STOREDIM STOREDIM_S
#else
#define STOREDIM STOREDIM_L
#endif


/*
To understand the following comments better, please refer to Figure 2(b) and 2(d) in Miao and Merz 2015 paper. 

 In the following kernel, we treat f orbital into 5 parts.
 
 type:   ss sp ps sd ds pp dd sf pf | df ff |
 ss                                 |       |
 sp                                 |       |
 ps                                 | zone  |
 sd                                 |  2    |
 ds         zone 0                  |       |
 pp                                 |       |
 dd                                 |       |
 sf                                 |       |
 pf                                 |       |
 -------------------------------------------
 df         zone 1                  | z | z |
 ff                                 | 3 | 4 |
 -------------------------------------------
 
 
 because the single f orbital kernel is impossible to compile completely, we treat VRR as:
 
 
 I+J  0 1 2 3 4 | 5 | 6 |
 0 ----------------------
 1|             |       |
 2|   Kernel    |  K2   |
 3|     0       |       |
 4|             |       |
 -----------------------|
 5|   Kernel    | K | K |
 6|     1       | 3 | 4 |
 ------------------------
 
 Their responses for
              I+J          K+L
 Kernel 0:   0-4           0-4
 Kernel 1:   0-4           5,6
 Kernel 2:   5,6           0-4
 Kernel 3:   5             5,6
 Kernel 4:   6             5,6
 
 Integrals in zone need kernel:
 zone 0: kernel 0
 zone 1: kernel 0,1
 zone 2: kernel 0,2
 zone 3: kernel 0,1,2,3
 zone 4: kernel 0,1,2,3,4
 
 so first, kernel 0: zone 0,1,2,3,4 (get2e_kernel()), if no f, then that's it.
 second,   kernel 1: zone 1,3,4(get2e_kernel_spdf())
 then,     kernel 2: zone 2,3,4(get2e_kernel_spdf2())
 then,     kernel 3: zone 3,4(get2e_kernel_spdf3())
 finally,  kernel 4: zone 4(get2e_kernel_spdf4())

 */
#ifdef OSHELL
#ifdef int_spd
__global__ void
__launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) get_oshell_eri_kernel()
#elif defined int_spdf
__global__ void
__launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) get_oshell_eri_kernel_spdf()
#elif defined int_spdf2
__global__ void
__launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) get_oshell_eri_kernel_spdf2()
#elif defined int_spdf3
__global__ void
__launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) get_oshell_eri_kernel_spdf3()
#elif defined int_spdf4
__global__ void
__launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) get_oshell_eri_kernel_spdf4()
#elif defined int_spdf5
__global__ void
__launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) get_oshell_eri_kernel_spdf5()
#elif defined int_spdf6
__global__ void
__launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) get_oshell_eri_kernel_spdf6()
#elif defined int_spdf7
__global__ void
__launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) get_oshell_eri_kernel_spdf7()
#elif defined int_spdf8
__global__ void
__launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) get_oshell_eri_kernel_spdf8()
#elif defined int_spdf9
__global__ void
__launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) get_oshell_eri_kernel_spdf9()
#elif defined int_spdf10
__global__ void
__launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) get_oshell_eri_kernel_spdf10()
#endif
#else
#ifdef int_spd
__global__ void
__launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) get2e_kernel()
#elif defined int_spdf
__global__ void
__launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) get2e_kernel_spdf()
#elif defined int_spdf2
__global__ void
__launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) get2e_kernel_spdf2()
#elif defined int_spdf3
__global__ void
__launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) get2e_kernel_spdf3()
#elif defined int_spdf4
__global__ void
__launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) get2e_kernel_spdf4()
#elif defined int_spdf5
__global__ void
__launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) get2e_kernel_spdf5()
#elif defined int_spdf6
__global__ void
__launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) get2e_kernel_spdf6()
#elif defined int_spdf7
__global__ void
__launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) get2e_kernel_spdf7()
#elif defined int_spdf8
__global__ void
__launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) get2e_kernel_spdf8()
#elif defined int_spdf9
__global__ void
__launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) get2e_kernel_spdf9()
#elif defined int_spdf10
__global__ void
__launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) get2e_kernel_spdf10()
#endif
#endif
{
    unsigned int offside = blockIdx.x*blockDim.x+threadIdx.x;
    int totalThreads = blockDim.x*gridDim.x;
    
    // jshell and jshell2 defines the regions in i+j and k+l axes respectively.    
    // sqrQshell= Qshell x Qshell; where Qshell is the number of sorted shells (see gpu_upload_basis_ in gpu.cu)
    // for details on sorting. 
 
#ifdef int_spd
/*
 Here we walk through full cutoff matrix.

 --sqrQshell --
 _______________ 
 |             |  |
 |             |  |
 |             | sqrQshell
 |             |  |
 |             |  |
 |_____________|  |

*/

    QUICKULL jshell = (QUICKULL) devSim.sqrQshell;
    QUICKULL jshell2 = (QUICKULL) devSim.sqrQshell;

#elif defined int_spdf

/*  
 Here we walk through following region of the cutoff matrix.

 --sqrQshell --
 _______________ 
 |             |  
 |             |  
 |             |  
 |_____________|  
 |             |  | sqrQshell - fStart
 |_____________|  |

*/
  
    QUICKULL jshell = (QUICKULL) devSim.sqrQshell;
    QUICKULL jshell2 = (QUICKULL) devSim.sqrQshell - devSim.fStart;
    
#elif defined int_spdf2

    QUICKULL jshell = (QUICKULL) devSim.sqrQshell;
    QUICKULL jshell2 = (QUICKULL) devSim.sqrQshell - devSim.fStart;

#elif defined int_spdf3

    QUICKULL jshell0 = (QUICKULL) devSim.fStart;
    QUICKULL jshell = (QUICKULL) devSim.sqrQshell - jshell0;
    QUICKULL jshell2 = (QUICKULL) devSim.sqrQshell - jshell0;
    
#elif defined int_spdf4
    
    QUICKULL jshell0 = (QUICKULL) devSim.fStart;
    QUICKULL jshell00 = (QUICKULL) devSim.ffStart;
    QUICKULL jshell = (QUICKULL) devSim.sqrQshell - jshell00;
    QUICKULL jshell2 = (QUICKULL) devSim.sqrQshell - jshell0;

#elif defined int_spdf5
    
    QUICKULL jshell = (QUICKULL) devSim.sqrQshell;
    QUICKULL jshell2 = (QUICKULL) devSim.sqrQshell - devSim.ffStart;

#elif defined int_spdf6
    
    QUICKULL jshell = (QUICKULL) devSim.sqrQshell;
    QUICKULL jshell2 = (QUICKULL) devSim.sqrQshell - devSim.ffStart;
    
#elif defined int_spdf7
    
    QUICKULL jshell0 = (QUICKULL) devSim.fStart;
    QUICKULL jshell00 = (QUICKULL) devSim.ffStart;
    QUICKULL jshell = (QUICKULL) devSim.sqrQshell - jshell0;
    QUICKULL jshell2 = (QUICKULL) devSim.sqrQshell - jshell00;

#elif defined int_spdf8
    
    QUICKULL jshell0 = (QUICKULL) devSim.ffStart;
    QUICKULL jshell00 = (QUICKULL) devSim.ffStart;
    QUICKULL jshell = (QUICKULL) devSim.sqrQshell - jshell00;
    QUICKULL jshell2 = (QUICKULL) devSim.sqrQshell - jshell0;

#elif defined int_spdf9
    
    QUICKULL jshell0 = (QUICKULL) devSim.ffStart;
    QUICKULL jshell00 = (QUICKULL) devSim.ffStart;
    QUICKULL jshell = (QUICKULL) devSim.sqrQshell - jshell00;
    QUICKULL jshell2 = (QUICKULL) devSim.sqrQshell - jshell0;

#elif defined int_spdf10
    
    QUICKULL jshell0 = (QUICKULL) devSim.ffStart;
    QUICKULL jshell00 = (QUICKULL) devSim.ffStart;
    QUICKULL jshell = (QUICKULL) devSim.sqrQshell - jshell00;
    QUICKULL jshell2 = (QUICKULL) devSim.sqrQshell - jshell0;

#endif

    for (QUICKULL i = offside; i < jshell * jshell2; i+= totalThreads) {
        
#ifdef int_spd
       /* 
        QUICKULL a, b;
        
        double aa = (double)((i+1)*1E-4);
        QUICKULL t = (QUICKULL)(sqrt(aa)*1E2);
        if ((i+1)==t*t) {
            t--;
        }
        
        QUICKULL k = i-t*t;
        if (k<=t) {
            a = k;
            b = t;
        }else {
            a = t;
            b = 2*t-k;
        }
        */
        // Zone 0

        QUICKULL a = (QUICKULL) i/jshell;
        QUICKULL b = (QUICKULL) (i - a*jshell);

#elif defined int_spdf
        
        
        // Zone 1
        QUICKULL b = (QUICKULL) i/jshell;
        QUICKULL a = (QUICKULL) (i - b*jshell);
        b = b + devSim.fStart;
                
#elif defined int_spdf2
        
        // Zone 2
        QUICKULL a = (QUICKULL) i/jshell;
        QUICKULL b = (QUICKULL) (i - a*jshell);
        a = a + devSim.fStart;

#elif defined int_spdf3
        
        // Zone 3
        QUICKULL a, b;
        if (jshell != 0 ) {
            a = (QUICKULL) i/jshell;
            b = (QUICKULL) (i - a*jshell);
            a = a + jshell0;
            b = b + jshell0;
        }else{
            a = 0;
            b = 0;
        }

#elif defined int_spdf4
        
        // Zone 4
        QUICKULL a, b;
        if (jshell2 != 0 ) {
            a = (QUICKULL) i/jshell2;
            b = (QUICKULL) (i - a*jshell2);
            a = a + jshell00;
            b = b + jshell0;
        }else{
            a = 0;
            b = 0;
        }

#elif defined int_spdf5
        
        // Zone 5
        QUICKULL b = (QUICKULL) i/jshell;
        QUICKULL a = (QUICKULL) (i - b*jshell);
        b = b + devSim.ffStart;
        
#elif defined int_spdf6
        
        // Zone 2
        QUICKULL a = (QUICKULL) i/jshell;
        QUICKULL b = (QUICKULL) (i - a*jshell);
        a = a + devSim.ffStart;

#elif defined int_spdf7
        
        // Zone 3
        QUICKULL a, b;
        if (jshell != 0 ) {
            a = (QUICKULL) i/jshell;
            b = (QUICKULL) (i - a*jshell);
            a = a + jshell0;
            b = b + jshell00;
        }else{
            a = 0;
            b = 0;
        }

#elif defined int_spdf8
        
        // Zone 4
        QUICKULL a, b;
        if (jshell2 != 0 ) {
            a = (QUICKULL) i/jshell2;
            b = (QUICKULL) (i - a*jshell2);
            a = a + jshell00;
            b = b + jshell0;
        }else{
            a = 0;
            b = 0;
        }
      
#elif defined int_spdf9
        
        // Zone 4
        QUICKULL a, b;
        if (jshell2 != 0 ) {
            a = (QUICKULL) i/jshell2;
            b = (QUICKULL) (i - a*jshell2);
            a = a + jshell00;
            b = b + jshell0;
        }else{
            a = 0;
            b = 0;
        }

#elif defined int_spdf10
        
        // Zone 4
        QUICKULL a, b;
        if (jshell2 != 0 ) {
            a = (QUICKULL) i/jshell2;
            b = (QUICKULL) (i - a*jshell2);
            a = a + jshell00;
            b = b + jshell0;
        }else{
            a = 0;
            b = 0;
        }

#endif

#ifdef HIP_MPIV
        if(devSim.mpi_bcompute[a] > 0){
#endif 

        int II = devSim.sorted_YCutoffIJ[a].x;
        int KK = devSim.sorted_YCutoffIJ[b].x;        

        int ii = devSim.sorted_Q[II];
        int kk = devSim.sorted_Q[KK];
        
        if (ii<=kk){
            
            
            int JJ = devSim.sorted_YCutoffIJ[a].y;
            int LL = devSim.sorted_YCutoffIJ[b].y;
            
            int jj = devSim.sorted_Q[JJ];
            int ll = devSim.sorted_Q[LL];
            
            
            
            int nshell = devSim.nshell;
            QUICKDouble DNMax = MAX(MAX(4.0*LOC2(devSim.cutMatrix, ii, jj, nshell, nshell), 4.0*LOC2(devSim.cutMatrix, kk, ll, nshell, nshell)),
                                    MAX(MAX(LOC2(devSim.cutMatrix, ii, ll, nshell, nshell),     LOC2(devSim.cutMatrix, ii, kk, nshell, nshell)),
                                        MAX(LOC2(devSim.cutMatrix, jj, kk, nshell, nshell),     LOC2(devSim.cutMatrix, jj, ll, nshell, nshell))));
            
            if ((LOC2(devSim.YCutoff, kk, ll, nshell, nshell) * LOC2(devSim.YCutoff, ii, jj, nshell, nshell))> devSim.integralCutoff && \
                (LOC2(devSim.YCutoff, kk, ll, nshell, nshell) * LOC2(devSim.YCutoff, ii, jj, nshell, nshell) * DNMax) > devSim.integralCutoff) {
                
                int iii = devSim.sorted_Qnumber[II];
                int jjj = devSim.sorted_Qnumber[JJ];
                int kkk = devSim.sorted_Qnumber[KK];
                int lll = devSim.sorted_Qnumber[LL];

                
      
#ifdef OSHELL
#ifdef int_spd
                    iclass_oshell(iii, jjj, kkk, lll, ii, jj, kk, ll, DNMax, devSim.YVerticalTemp+offside, devSim.store+offside);

#elif defined int_spdf
                if ( (kkk + lll) <= 6 && (kkk + lll) > 4) {
                    iclass_oshell_spdf(iii, jjj, kkk, lll, ii, jj, kk, ll, DNMax, devSim.YVerticalTemp+offside, devSim.store+offside);
                }


#elif defined int_spdf2
                if ( (iii + jjj) > 4 && (iii + jjj) <= 6 ) {
                    iclass_oshell_spdf2(iii, jjj, kkk, lll, ii, jj, kk, ll, DNMax, devSim.YVerticalTemp+offside, devSim.store+offside);
                }

#elif defined int_spdf3


                if ( (iii + jjj) >= 5 && (iii + jjj) <= 6 && (kkk + lll) <= 6 && (kkk + lll) >= 5) {
                    iclass_oshell_spdf3(iii, jjj, kkk, lll, ii, jj, kk, ll, DNMax, devSim.YVerticalTemp+offside, devSim.store+offside);
                }

#elif defined int_spdf4


                if ( (iii + jjj) == 6 && (kkk + lll) <= 6 && (kkk + lll) >= 5) {
                    iclass_oshell_spdf4(iii, jjj, kkk, lll, ii, jj, kk, ll, DNMax, devSim.YVerticalTemp+offside, devSim.store+offside);
                }

#elif defined int_spdf5

                if ( (kkk + lll) == 6 && (iii + jjj) >= 4 && (iii + jjj) <= 6) {
                    iclass_oshell_spdf5(iii, jjj, kkk, lll, ii, jj, kk, ll, DNMax, devSim.YVerticalTemp+offside, devSim.store+offside);
                }
#elif defined int_spdf6
                if ( (iii + jjj) == 6 && (kkk + lll) <= 6 && (kkk + lll) >= 4) {
                    iclass_oshell_spdf6(iii, jjj, kkk, lll, ii, jj, kk, ll, DNMax, devSim.YVerticalTemp+offside, devSim.store+offside);
                }
#elif defined int_spdf7


                if ( (iii + jjj) >=5 && (iii + jjj) <= 6 && (kkk + lll) == 6) {
                    iclass_oshell_spdf7(iii, jjj, kkk, lll, ii, jj, kk, ll, DNMax, devSim.YVerticalTemp+offside, devSim.store+offside);
                }

#elif defined int_spdf8


                if ( (iii + jjj) == 6 && (kkk + lll) == 6) {
                    iclass_oshell_spdf8(iii, jjj, kkk, lll, ii, jj, kk, ll, DNMax, devSim.YVerticalTemp+offside, devSim.store+offside);
                }
#elif defined int_spdf9


                if ( (iii + jjj) == 6 && (kkk + lll) == 6) {
                    iclass_oshell_spdf9(iii, jjj, kkk, lll, ii, jj, kk, ll, DNMax, devSim.YVerticalTemp+offside, devSim.store+offside);
                }
#elif defined int_spdf10


                if ( (iii + jjj) == 6 && (kkk + lll) == 6) {
                    iclass_oshell_spdf10(iii, jjj, kkk, lll, ii, jj, kk, ll, DNMax, devSim.YVerticalTemp+offside, devSim.store+offside);
                }
#endif
#else          
#ifdef int_spd
                    iclass(iii, jjj, kkk, lll, ii, jj, kk, ll, DNMax, devSim.YVerticalTemp+offside, devSim.store+offside);
                
#elif defined int_spdf
                if ( (kkk + lll) <= 6 && (kkk + lll) > 4) {
                    iclass_spdf(iii, jjj, kkk, lll, ii, jj, kk, ll, DNMax, devSim.YVerticalTemp+offside, devSim.store+offside);
                }
                
                
#elif defined int_spdf2
                if ( (iii + jjj) > 4 && (iii + jjj) <= 6 ) {
                    iclass_spdf2(iii, jjj, kkk, lll, ii, jj, kk, ll, DNMax, devSim.YVerticalTemp+offside, devSim.store+offside);
                }
                
#elif defined int_spdf3
                
                
                if ( (iii + jjj) >= 5 && (iii + jjj) <= 6 && (kkk + lll) <= 6 && (kkk + lll) >= 5) {
                    iclass_spdf3(iii, jjj, kkk, lll, ii, jj, kk, ll, DNMax, devSim.YVerticalTemp+offside, devSim.store+offside);
                }
                
#elif defined int_spdf4
                
                
                if ( (iii + jjj) == 6 && (kkk + lll) <= 6 && (kkk + lll) >= 5) {
                    iclass_spdf4(iii, jjj, kkk, lll, ii, jj, kk, ll, DNMax, devSim.YVerticalTemp+offside, devSim.store+offside);
                }
                
#elif defined int_spdf5
                
                if ( (kkk + lll) == 6 && (iii + jjj) >= 4 && (iii + jjj) <= 6) {
                    iclass_spdf5(iii, jjj, kkk, lll, ii, jj, kk, ll, DNMax, devSim.YVerticalTemp+offside, devSim.store+offside);
                }
                
                
#elif defined int_spdf6
                if ( (iii + jjj) == 6 && (kkk + lll) <= 6 && (kkk + lll) >= 4) {
                    iclass_spdf6(iii, jjj, kkk, lll, ii, jj, kk, ll, DNMax, devSim.YVerticalTemp+offside, devSim.store+offside);
                }
                
#elif defined int_spdf7
                
                
                if ( (iii + jjj) >=5 && (iii + jjj) <= 6 && (kkk + lll) == 6) {
                    iclass_spdf7(iii, jjj, kkk, lll, ii, jj, kk, ll, DNMax, devSim.YVerticalTemp+offside, devSim.store+offside);
                }
                
#elif defined int_spdf8
                
                
                if ( (iii + jjj) == 6 && (kkk + lll) == 6) {
                    iclass_spdf8(iii, jjj, kkk, lll, ii, jj, kk, ll, DNMax, devSim.YVerticalTemp+offside, devSim.store+offside);
                }
#elif defined int_spdf9
                
                
                if ( (iii + jjj) == 6 && (kkk + lll) == 6) {
                    iclass_spdf9(iii, jjj, kkk, lll, ii, jj, kk, ll, DNMax, devSim.YVerticalTemp+offside, devSim.store+offside);
                }
#elif defined int_spdf10
                
                
                if ( (iii + jjj) == 6 && (kkk + lll) == 6) {
                    iclass_spdf10(iii, jjj, kkk, lll, ii, jj, kk, ll, DNMax, devSim.YVerticalTemp+offside, devSim.store+offside);
                }
#endif
#endif
                
            }
        }

#ifdef HIP_MPIV
        }      
#endif        
    }
}

/*
 iclass subroutine is to generate 2-electron intergral using HRR and VRR method, which is the most
 performance algrithem for electron intergral evaluation. See description below for details
 */
#ifdef OSHELL
#ifdef int_spd
__device__ __forceinline__ void iclass_oshell
#elif defined int_spdf
__device__ __forceinline__ void iclass_oshell_spdf
#elif defined int_spdf2
__device__ __forceinline__ void iclass_oshell_spdf2
#elif defined int_spdf3
__device__ __forceinline__ void iclass_oshell_spdf3
#elif defined int_spdf4
__device__ __forceinline__ void iclass_oshell_spdf4
#elif defined int_spdf5
__device__ __forceinline__ void iclass_oshell_spdf5
#elif defined int_spdf6
__device__ __forceinline__ void iclass_oshell_spdf6
#elif defined int_spdf7
__device__ __forceinline__ void iclass_oshell_spdf7
#elif defined int_spdf8
__device__ __forceinline__ void iclass_oshell_spdf8
#elif defined int_spdf9
__device__ __forceinline__ void iclass_oshell_spdf9
#elif defined int_spdf10
__device__ __forceinline__ void iclass_oshell_spdf10
#endif
#else
#ifdef int_spd
__device__ __forceinline__ void iclass
#elif defined int_spdf
__device__ __forceinline__ void iclass_spdf
#elif defined int_spdf2
__device__ __forceinline__ void iclass_spdf2
#elif defined int_spdf3
__device__ __forceinline__ void iclass_spdf3
#elif defined int_spdf4
__device__ __forceinline__ void iclass_spdf4
#elif defined int_spdf5
__device__ __forceinline__ void iclass_spdf5
#elif defined int_spdf6
__device__ __forceinline__ void iclass_spdf6
#elif defined int_spdf7
__device__ __forceinline__ void iclass_spdf7
#elif defined int_spdf8
__device__ __forceinline__ void iclass_spdf8
#elif defined int_spdf9
__device__ __forceinline__ void iclass_spdf9
#elif defined int_spdf10
__device__ __forceinline__ void iclass_spdf10
#endif
#endif
                                      (int I, int J, int K, int L, unsigned int II, unsigned int JJ, unsigned int KK, unsigned int LL, QUICKDouble DNMax, \
                                      QUICKDouble* YVerticalTemp, QUICKDouble* store)
{
    
    /*
     kAtom A, B, C ,D is the coresponding atom for shell ii, jj, kk, ll
     and be careful with the index difference between Fortran and C++,
     Fortran starts array index with 1 and C++ starts 0.
     
     
     RA, RB, RC, and RD are the coordinates for atom katomA, katomB, katomC and katomD,
     which means they are corrosponding coorinates for shell II, JJ, KK, and LL.
     And we don't need the coordinates now, so we will not retrieve the data now.
     */
    QUICKDouble RAx = LOC2(devSim.xyz, 0 , devSim.katom[II]-1, 3, devSim.natom);
    QUICKDouble RAy = LOC2(devSim.xyz, 1 , devSim.katom[II]-1, 3, devSim.natom);
    QUICKDouble RAz = LOC2(devSim.xyz, 2 , devSim.katom[II]-1, 3, devSim.natom);
    
    QUICKDouble RCx = LOC2(devSim.xyz, 0 , devSim.katom[KK]-1, 3, devSim.natom);
    QUICKDouble RCy = LOC2(devSim.xyz, 1 , devSim.katom[KK]-1, 3, devSim.natom);
    QUICKDouble RCz = LOC2(devSim.xyz, 2 , devSim.katom[KK]-1, 3, devSim.natom);
    
    /*
     kPrimI, J, K and L indicates the primtive gaussian function number
     kStartI, J, K, and L indicates the starting guassian function for shell I, J, K, and L.
     We retrieve from global memory and save them to register to avoid multiple retrieve.
     */
    int kPrimI = devSim.kprim[II];
    int kPrimJ = devSim.kprim[JJ];
    int kPrimK = devSim.kprim[KK];
    int kPrimL = devSim.kprim[LL];
    
    int kStartI = devSim.kstart[II]-1;
    int kStartJ = devSim.kstart[JJ]-1;
    int kStartK = devSim.kstart[KK]-1;
    int kStartL = devSim.kstart[LL]-1;
    
    
    /*
     store saves temp contracted integral as [as|bs] type. the dimension should be allocatable but because
     of cuda limitation, we can not do that now.
     
     See M.Head-Gordon and J.A.Pople, Jchem.Phys., 89, No.9 (1988) for VRR algrithem details.
     */
    //QUICKDouble store[STOREDIM*STOREDIM];
    
    /*
     Initial the neccessary element for
     */
    
#ifdef int_spd
    for (int i = Sumindex[K+1]+1; i<= Sumindex[K+L+2]; i++) {
        for (int j = Sumindex[I+1]+1; j<= Sumindex[I+J+2]; j++) {
            if ( i <= STOREDIM && j <= STOREDIM) {
                LOCSTORE(store, j-1, i-1, STOREDIM, STOREDIM) = 0;
            }
        }
    }
#elif defined int_spdf
    
    for (int i = Sumindex[K+1]+1; i<= Sumindex[K+L+2]; i++) {
        for (int j = Sumindex[I+1]+1; j<= Sumindex[I+J+2]; j++) {
            if ( i <= STOREDIM && j <= STOREDIM) {
                LOCSTORE(store, j-1, i-1, STOREDIM, STOREDIM) = 0;
            }
        }
    }
#elif defined int_spdf2
    
    for (int i = Sumindex[K+1]+1; i<= Sumindex[K+L+2]; i++) {
        for (int j = Sumindex[I+1]+1; j<= Sumindex[I+J+2]; j++) {
            if ( i <= STOREDIM && j <= STOREDIM) {
                LOCSTORE(store, j-1, i-1, STOREDIM, STOREDIM) = 0;
            }
        }
    }
#elif defined int_spdf3
    
    for (int i = Sumindex[K+1]+1; i<= Sumindex[K+L+2]; i++) {
        for (int j = Sumindex[I+1]+1; j<= Sumindex[I+J+2]; j++) {
            if ( i <= STOREDIM && j <= STOREDIM) {
                LOCSTORE(store, j-1, i-1, STOREDIM, STOREDIM) = 0;
            }
        }
    }
#elif defined int_spdf4
    
    for (int i = Sumindex[K+1]+1; i<= Sumindex[K+L+2]; i++) {
        for (int j = Sumindex[I+1]+1; j<= Sumindex[I+J+2]; j++) {
            if ( i <= STOREDIM && j <= STOREDIM) {
                LOCSTORE(store, j-1, i-1, STOREDIM, STOREDIM) = 0;
            }
        }
    }
#elif defined int_spdf5
    
    for (int i = Sumindex[K+1]+1; i<= Sumindex[K+L+2]; i++) {
        for (int j = Sumindex[I+1]+1; j<= Sumindex[I+J+2]; j++) {
            if ( i <= STOREDIM && j <= STOREDIM) {
                LOCSTORE(store, j-1, i-1, STOREDIM, STOREDIM) = 0;
            }
        }
    }
#elif defined int_spdf6
    
    for (int i = Sumindex[K+1]+1; i<= Sumindex[K+L+2]; i++) {
        for (int j = Sumindex[I+1]+1; j<= Sumindex[I+J+2]; j++) {
            if ( i <= STOREDIM && j <= STOREDIM) {
                LOCSTORE(store, j-1, i-1, STOREDIM, STOREDIM) = 0;
            }
        }
    }
#elif defined int_spdf7
    
    for (int i = Sumindex[K+1]+1; i<= Sumindex[K+L+2]; i++) {
        for (int j = Sumindex[I+1]+1; j<= Sumindex[I+J+2]; j++) {
            if ( i <= STOREDIM && j <= STOREDIM) {
                LOCSTORE(store, j-1, i-1, STOREDIM, STOREDIM) = 0;
            }
        }
    }
#elif defined int_spdf8
    
    for (int i = Sumindex[K+1]+1; i<= Sumindex[K+L+2]; i++) {
        for (int j = Sumindex[I+1]+1; j<= Sumindex[I+J+2]; j++) {
            if ( i <= STOREDIM && j <= STOREDIM) {
                LOCSTORE(store, j-1, i-1, STOREDIM, STOREDIM) = 0;
            }
        }
    }
#elif defined int_spdf9
    
    for (int i = Sumindex[K+1]+1; i<= Sumindex[K+L+2]; i++) {
        for (int j = Sumindex[I+1]+1; j<= Sumindex[I+J+2]; j++) {
            if ( i <= STOREDIM && j <= STOREDIM) {
                LOCSTORE(store, j-1, i-1, STOREDIM, STOREDIM) = 0;
            }
        }
    }
#elif defined int_spdf10
    
    for (int i = Sumindex[K+1]+1; i<= Sumindex[K+L+2]; i++) {
        for (int j = Sumindex[I+1]+1; j<= Sumindex[I+J+2]; j++) {
            if ( i <= STOREDIM && j <= STOREDIM) {
                LOCSTORE(store, j-1, i-1, STOREDIM, STOREDIM) = 0;
            }
        }
    }
#endif
    
    
    for (int i = 0; i<kPrimI*kPrimJ;i++){
        int JJJ = (int) i/kPrimI;
        int III = (int) i-kPrimI*JJJ;
        /*
         In the following comments, we have I, J, K, L denote the primitive gaussian function we use, and
         for example, expo(III, ksumtype(II)) stands for the expo for the IIIth primitive guassian function for II shell,
         we use I to express the corresponding index.
         AB = expo(I)+expo(J)
         --->                --->
         ->     expo(I) * xyz (I) + expo(J) * xyz(J)
         P  = ---------------------------------------
         expo(I) + expo(J)
         Those two are pre-calculated in CPU stage.
         
         */
        int ii_start = devSim.prim_start[II];
        int jj_start = devSim.prim_start[JJ];
        
        QUICKDouble AB = LOC2(devSim.expoSum, ii_start+III, jj_start+JJJ, devSim.prim_total, devSim.prim_total);
        QUICKDouble Px = LOC2(devSim.weightedCenterX, ii_start+III, jj_start+JJJ, devSim.prim_total, devSim.prim_total);
        QUICKDouble Py = LOC2(devSim.weightedCenterY, ii_start+III, jj_start+JJJ, devSim.prim_total, devSim.prim_total);
        QUICKDouble Pz = LOC2(devSim.weightedCenterZ, ii_start+III, jj_start+JJJ, devSim.prim_total, devSim.prim_total);
        
        /*
         X1 is the contracted coeffecient, which is pre-calcuated in CPU stage as well.
         cutoffprim is used to cut too small prim gaussian function when bring density matrix into consideration.
         */
        QUICKDouble cutoffPrim = DNMax * LOC2(devSim.cutPrim, kStartI+III, kStartJ+JJJ, devSim.jbasis, devSim.jbasis);
        QUICKDouble X1 = LOC4(devSim.Xcoeff, kStartI+III, kStartJ+JJJ, I - devSim.Qstart[II], J - devSim.Qstart[JJ], devSim.jbasis, devSim.jbasis, 2, 2);
        
        for (int j = 0; j<kPrimK*kPrimL; j++){
            int LLL = (int)j/kPrimK;
            int KKK = (int) j-kPrimK*LLL;
            
            if (cutoffPrim * LOC2(devSim.cutPrim, kStartK+KKK, kStartL+LLL, devSim.jbasis, devSim.jbasis) > devSim.primLimit) {
                /*
                 CD = expo(L)+expo(K)
                 ABCD = 1/ (AB + CD) = 1 / (expo(I)+expo(J)+expo(K)+expo(L))
                 AB * CD      (expo(I)+expo(J))*(expo(K)+expo(L))
                 Rou(Greek Letter) =   ----------- = ------------------------------------
                 AB + CD         expo(I)+expo(J)+expo(K)+expo(L)
                 
                 expo(I)+expo(J)                        expo(K)+expo(L)
                 ABcom = --------------------------------  CDcom = --------------------------------
                 expo(I)+expo(J)+expo(K)+expo(L)           expo(I)+expo(J)+expo(K)+expo(L)
                 
                 ABCDtemp = 1/2(expo(I)+expo(J)+expo(K)+expo(L))
                 */
                
                int kk_start = devSim.prim_start[KK];
                int ll_start = devSim.prim_start[LL];
                
                QUICKDouble CD = LOC2(devSim.expoSum, kk_start+KKK, ll_start+LLL, devSim.prim_total, devSim.prim_total);
                
                QUICKDouble ABCD = 1/(AB+CD);
                
                /*
                 X2 is the multiplication of four indices normalized coeffecient
                 */
                QUICKDouble X2 = sqrt(ABCD) * X1 * LOC4(devSim.Xcoeff, kStartK+KKK, kStartL+LLL, K - devSim.Qstart[KK], L - devSim.Qstart[LL], devSim.jbasis, devSim.jbasis, 2, 2);
                
                /*
                 Q' is the weighting center of K and L
                 --->           --->
                 ->  ------>       expo(K)*xyz(K)+expo(L)*xyz(L)
                 Q = P'(K,L)  = ------------------------------
                 expo(K) + expo(L)
                 
                 W' is the weight center for I, J, K, L
                 
                 --->             --->             --->            --->
                 ->     expo(I)*xyz(I) + expo(J)*xyz(J) + expo(K)*xyz(K) +expo(L)*xyz(L)
                 W = -------------------------------------------------------------------
                 expo(I) + expo(J) + expo(K) + expo(L)
                 ->  ->  2
                 RPQ =| P - Q |
                 
                 ->  -> 2
                 T = ROU * | P - Q|
                 */
                
                QUICKDouble Qx = LOC2(devSim.weightedCenterX, kk_start+KKK, ll_start+LLL, devSim.prim_total, devSim.prim_total);
                QUICKDouble Qy = LOC2(devSim.weightedCenterY, kk_start+KKK, ll_start+LLL, devSim.prim_total, devSim.prim_total);
                QUICKDouble Qz = LOC2(devSim.weightedCenterZ, kk_start+KKK, ll_start+LLL, devSim.prim_total, devSim.prim_total);
                
                //QUICKDouble T = AB * CD * ABCD * ( quick_dsqr(Px-Qx) + quick_dsqr(Py-Qy) + quick_dsqr(Pz-Qz));
                
                //QUICKDouble YVerticalTemp[VDIM1*VDIM2*VDIM3];
                FmT(I+J+K+L, AB * CD * ABCD * ( quick_dsqr(Px-Qx) + quick_dsqr(Py-Qy) + quick_dsqr(Pz-Qz)), YVerticalTemp);
                for (int i = 0; i<=I+J+K+L; i++) {
                    VY(0, 0, i) = VY(0, 0, i) * X2;
                }
#ifdef int_spd
                vertical(I, J, K, L, YVerticalTemp, store, \
                         Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, \
                         Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, \
                         0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD);
#elif defined int_spdf
                
                vertical_spdf(I, J, K, L, YVerticalTemp, store, \
                         Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, \
                         Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, \
                         0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD);
#elif defined int_spdf2
                
                vertical_spdf2(I, J, K, L, YVerticalTemp, store, \
                              Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, \
                              Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, \
                              0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD);
#elif defined int_spdf3
                
                vertical_spdf3(I, J, K, L, YVerticalTemp, store, \
                              Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, \
                              Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, \
                              0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD);
#elif defined int_spdf4
                
                vertical_spdf4(I, J, K, L, YVerticalTemp, store, \
                               Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, \
                               Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, \
                               0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD);
#elif defined int_spdf5
                
                vertical_spdf5(I, J, K, L, YVerticalTemp, store, \
                              Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, \
                              Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, \
                              0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD);
#elif defined int_spdf6
                
                vertical_spdf6(I, J, K, L, YVerticalTemp, store, \
                               Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, \
                               Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, \
                               0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD);
#elif defined int_spdf7
                
                vertical_spdf7(I, J, K, L, YVerticalTemp, store, \
                               Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, \
                               Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, \
                               0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD);
#elif defined int_spdf8
                
                vertical_spdf8(I, J, K, L, YVerticalTemp, store, \
                               Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, \
                               Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, \
                               0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD);
#elif defined int_spdf9
                
                vertical_spdf9(I, J, K, L, YVerticalTemp, store, \
                               Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, \
                               Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, \
                               0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD);
#elif defined int_spdf10
                
                vertical_spdf10(I, J, K, L, YVerticalTemp, store, \
                               Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, \
                               Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, \
                               0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD);
#endif
                
            }
        }
    }
    
    
    // IJKLTYPE is the I, J, K,L type
    int IJKLTYPE = (int) (1000 * I + 100 *J + 10 * K + L);
    
    QUICKDouble RBx, RBy, RBz;
    QUICKDouble RDx, RDy, RDz;
    
    RBx = LOC2(devSim.xyz, 0 , devSim.katom[JJ]-1, 3, devSim.natom);
    RBy = LOC2(devSim.xyz, 1 , devSim.katom[JJ]-1, 3, devSim.natom);
    RBz = LOC2(devSim.xyz, 2 , devSim.katom[JJ]-1, 3, devSim.natom);
    
    
    RDx = LOC2(devSim.xyz, 0 , devSim.katom[LL]-1, 3, devSim.natom);
    RDy = LOC2(devSim.xyz, 1 , devSim.katom[LL]-1, 3, devSim.natom);
    RDz = LOC2(devSim.xyz, 2 , devSim.katom[LL]-1, 3, devSim.natom);
    
    int III1 = LOC2(devSim.Qsbasis, II, I, devSim.nshell, 4);
    int III2 = LOC2(devSim.Qfbasis, II, I, devSim.nshell, 4);
    int JJJ1 = LOC2(devSim.Qsbasis, JJ, J, devSim.nshell, 4);
    int JJJ2 = LOC2(devSim.Qfbasis, JJ, J, devSim.nshell, 4);
    int KKK1 = LOC2(devSim.Qsbasis, KK, K, devSim.nshell, 4);
    int KKK2 = LOC2(devSim.Qfbasis, KK, K, devSim.nshell, 4);
    int LLL1 = LOC2(devSim.Qsbasis, LL, L, devSim.nshell, 4);
    int LLL2 = LOC2(devSim.Qfbasis, LL, L, devSim.nshell, 4);
    
    
    // maxIJKL is the max of I,J,K,L
    int maxIJKL = (int)MAX(MAX(I,J),MAX(K,L));
    
    if (((maxIJKL == 2)&&(J != 0 || L!=0)) || (maxIJKL >= 3)) {
        IJKLTYPE = 999;
    }
    
    /*QUICKDouble hybrid_coeff = 0.0;
    if (devSim.method == HF){
        hybrid_coeff = 1.0;
    }else if (devSim.method == B3LYP){
        hybrid_coeff = 0.2;
    }else if (devSim.method == DFT){
        hybrid_coeff = 0.0;
    }else if(devSim.method == LIBXC){
        hybrid_coeff = devSim.hyb_coeff;                        
    }*/
    
    
    for (int III = III1; III <= III2; III++) {
        for (int JJJ = MAX(III,JJJ1); JJJ <= JJJ2; JJJ++) {
            for (int KKK = MAX(III,KKK1); KKK <= KKK2; KKK++) {
                for (int LLL = MAX(KKK,LLL1); LLL <= LLL2; LLL++) {
                    
                    if (III < KKK ||
                        ((III == JJJ) && (III == LLL)) ||
                        ((III == JJJ) && (III  < LLL)) ||
                        ((JJJ == LLL) && (III  < JJJ)) ||
                        ((III == KKK) && (III  < JJJ)  && (JJJ < LLL))) {
                        
#ifdef int_spd
                        QUICKDouble Y = (QUICKDouble) hrrwhole
#elif defined int_spdf1
                        QUICKDouble Y = (QUICKDouble) hrrwhole2_1
#elif defined int_spdf2
                        QUICKDouble Y = (QUICKDouble) hrrwhole2_2
#elif defined int_spdf3
                        QUICKDouble Y = (QUICKDouble) hrrwhole2_3
#elif defined int_spdf4
                        QUICKDouble Y = (QUICKDouble) hrrwhole2_4
#elif defined int_spdf5
                        QUICKDouble Y = (QUICKDouble) hrrwhole2_5
#elif defined int_spdf6
                        QUICKDouble Y = (QUICKDouble) hrrwhole2_6
#elif defined int_spdf7
                        QUICKDouble Y = (QUICKDouble) hrrwhole2_7
#elif defined int_spdf8
                        QUICKDouble Y = (QUICKDouble) hrrwhole2_8
#elif defined int_spdf9
                        QUICKDouble Y = (QUICKDouble) hrrwhole2_9
#elif defined int_spdf10
                        QUICKDouble Y = (QUICKDouble) hrrwhole2_10
#else
                        
                        QUICKDouble Y = (QUICKDouble) hrrwhole2
                        

#endif
                                                               (I, J, K, L,\
                                                               III, JJJ, KKK, LLL, IJKLTYPE, store, \
                                                               RAx, RAy, RAz, RBx, RBy, RBz, \
                                                               RCx, RCy, RCz, RDx, RDy, RDz);
#ifdef int_spd
                        if (abs(Y) > 0.0e0)
#else
                        if (abs(Y) > devSim.integralCutoff)
#endif
                        {
#ifdef OSHELL
                            addint_oshell(devSim.oULL,devSim.obULL, Y, III, JJJ, KKK, LLL, devSim.hyb_coeff, devSim.dense, devSim.denseb, devSim.nbasis);
#else
                            addint(devSim.oULL, Y, III, JJJ, KKK, LLL, devSim.hyb_coeff, devSim.dense, devSim.nbasis);
#endif
                        }
                        
                    }
                }
            }
        }
    }
    return;
}



#ifndef OSHELL
#ifdef int_spd
__global__ void 
__launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) getAOInt_kernel(QUICKULL intStart, QUICKULL intEnd, ERI_entry* aoint_buffer, int streamID)
#elif defined int_spdf
__global__ void 
__launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) getAOInt_kernel_spdf(QUICKULL intStart, QUICKULL intEnd, ERI_entry* aoint_buffer, int streamID)
#elif defined int_spdf2
__global__ void 
__launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) getAOInt_kernel_spdf2(QUICKULL intStart, QUICKULL intEnd, ERI_entry* aoint_buffer, int streamID)
#elif defined int_spdf3
__global__ void 
__launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) getAOInt_kernel_spdf3(QUICKULL intStart, QUICKULL intEnd, ERI_entry* aoint_buffer, int streamID)
#elif defined int_spdf4
__global__ void 
__launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) getAOInt_kernel_spdf4(QUICKULL intStart, QUICKULL intEnd, ERI_entry* aoint_buffer, int streamID)
#elif defined int_spdf5
__global__ void 
__launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) getAOInt_kernel_spdf5(QUICKULL intStart, QUICKULL intEnd, ERI_entry* aoint_buffer, int streamID)
#elif defined int_spdf6
__global__ void 
__launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) getAOInt_kernel_spdf6(QUICKULL intStart, QUICKULL intEnd, ERI_entry* aoint_buffer, int streamID)
#elif defined int_spdf7
__global__ void 
__launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) getAOInt_kernel_spdf7(QUICKULL intStart, QUICKULL intEnd, ERI_entry* aoint_buffer, int streamID)
#elif defined int_spdf8
__global__ void 
__launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) getAOInt_kernel_spdf8(QUICKULL intStart, QUICKULL intEnd, ERI_entry* aoint_buffer, int streamID)
#elif defined int_spdf9
__global__ void 
__launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) getAOInt_kernel_spdf9(QUICKULL intStart, QUICKULL intEnd, ERI_entry* aoint_buffer, int streamID)
#elif defined int_spdf10
__global__ void 
__launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) getAOInt_kernel_spdf10(QUICKULL intStart, QUICKULL intEnd, ERI_entry* aoint_buffer, int streamID)
#endif
{
    
    unsigned int offside = blockIdx.x*blockDim.x+threadIdx.x;
    int totalThreads = blockDim.x*gridDim.x;
    
    QUICKULL jshell         = (QUICKULL) devSim.sqrQshell;
    QUICKULL myInt          = (QUICKULL) (intEnd - intStart + 1) / totalThreads;
    
    
    
    if ((intEnd - intStart + 1 - myInt*totalThreads)> offside) myInt++;
    
    for (QUICKULL i = 1; i<=myInt; i++) {
        QUICKULL currentInt = totalThreads * (i-1) + offside + intStart;
        QUICKULL a = (QUICKULL) currentInt/jshell;
        QUICKULL b = (QUICKULL) (currentInt - a*jshell);
        
        int II = devSim.sorted_YCutoffIJ[a].x;
        int JJ = devSim.sorted_YCutoffIJ[a].y;
        int KK = devSim.sorted_YCutoffIJ[b].x;
        int LL = devSim.sorted_YCutoffIJ[b].y;
        
        int ii = devSim.sorted_Q[II];
        int jj = devSim.sorted_Q[JJ];
        int kk = devSim.sorted_Q[KK];
        int ll = devSim.sorted_Q[LL];
        
        if (ii<=kk) {
            int nshell = devSim.nshell;
            
            if ((LOC2(devSim.YCutoff, kk, ll, nshell, nshell) * LOC2(devSim.YCutoff, ii, jj, nshell, nshell))> devSim.leastIntegralCutoff) {
                
                int iii = devSim.sorted_Qnumber[II];
                int jjj = devSim.sorted_Qnumber[JJ];
                int kkk = devSim.sorted_Qnumber[KK];
                int lll = devSim.sorted_Qnumber[LL];
#ifdef int_spd
        //        if (!((iii + jjj) > 4 || (kkk + lll) > 4)) {
                    iclass_AOInt(iii, jjj, kkk, lll, ii, jj, kk, ll, 1.0, aoint_buffer, streamID, devSim.YVerticalTemp+offside, devSim.store+offside);
        //        }
#elif defined int_spdf
                if ((iii + jjj) > 4 || (kkk + lll) > 4) {
                    iclass_AOInt_spdf(iii, jjj, kkk, lll, ii, jj, kk, ll, 1.0, aoint_buffer, streamID, devSim.YVerticalTemp+offside, devSim.store+offside);
                }
#elif defined int_spdf2
                if ((iii + jjj) > 4 || (kkk + lll) > 4) {
                    iclass_AOInt_spdf2(iii, jjj, kkk, lll, ii, jj, kk, ll, 1.0, aoint_buffer, streamID, devSim.YVerticalTemp+offside, devSim.store+offside);
                }
#elif defined int_spdf3
                if ((iii + jjj) > 4 || (kkk + lll) > 4) {
                    iclass_AOInt_spdf3(iii, jjj, kkk, lll, ii, jj, kk, ll, 1.0, aoint_buffer, streamID, devSim.YVerticalTemp+offside, devSim.store+offside);
                }
#elif defined int_spdf4
                if ((iii + jjj) > 4 || (kkk + lll) > 4) {
                    iclass_AOInt_spdf4(iii, jjj, kkk, lll, ii, jj, kk, ll, 1.0, aoint_buffer, streamID, devSim.YVerticalTemp+offside, devSim.store+offside);
                }
                
#elif defined int_spdf5
                if ((iii + jjj) > 4 || (kkk + lll) > 4) {
                    iclass_AOInt_spdf5(iii, jjj, kkk, lll, ii, jj, kk, ll, 1.0, aoint_buffer, streamID, devSim.YVerticalTemp+offside, devSim.store+offside);
                }
#elif defined int_spdf6
                if ((iii + jjj) > 4 || (kkk + lll) > 4) {
                    iclass_AOInt_spdf6(iii, jjj, kkk, lll, ii, jj, kk, ll, 1.0, aoint_buffer, streamID, devSim.YVerticalTemp+offside, devSim.store+offside);
                }
#elif defined int_spdf7
                if ((iii + jjj) > 4 || (kkk + lll) > 4) {
                    iclass_AOInt_spdf7(iii, jjj, kkk, lll, ii, jj, kk, ll, 1.0, aoint_buffer, streamID, devSim.YVerticalTemp+offside, devSim.store+offside);
                }
#elif defined int_spdf8
                if ((iii + jjj) > 4 || (kkk + lll) > 4) {
                    iclass_AOInt_spdf8(iii, jjj, kkk, lll, ii, jj, kk, ll, 1.0, aoint_buffer, streamID, devSim.YVerticalTemp+offside, devSim.store+offside);
                }
#elif defined int_spdf9
                if ((iii + jjj) > 4 || (kkk + lll) > 4) {
                    iclass_AOInt_spdf9(iii, jjj, kkk, lll, ii, jj, kk, ll, 1.0, aoint_buffer, streamID, devSim.YVerticalTemp+offside, devSim.store+offside);
                }
#elif defined int_spdf10
                if ((iii + jjj) > 4 || (kkk + lll) > 4) {
                    iclass_AOInt_spdf10(iii, jjj, kkk, lll, ii, jj, kk, ll, 1.0, aoint_buffer, streamID, devSim.YVerticalTemp+offside, devSim.store+offside);
                }
#endif
            }
        }
    }
}







/*
 iclass subroutine is to generate 2-electron intergral using HRR and VRR method, which is the most
 performance algrithem for electron intergral evaluation. See description below for details
 */
#ifdef int_spd
__device__ __forceinline__ void iclass_AOInt
#elif defined int_spdf
__device__ __forceinline__ void iclass_AOInt_spdf
#elif defined int_spdf2
__device__ __forceinline__ void iclass_AOInt_spdf2
#elif defined int_spdf3
__device__ __forceinline__ void iclass_AOInt_spdf3
#elif defined int_spdf4
__device__ __forceinline__ void iclass_AOInt_spdf4
#elif defined int_spdf5
__device__ __forceinline__ void iclass_AOInt_spdf5
#elif defined int_spdf6
__device__ __forceinline__ void iclass_AOInt_spdf6
#elif defined int_spdf7
__device__ __forceinline__ void iclass_AOInt_spdf7
#elif defined int_spdf8
__device__ __forceinline__ void iclass_AOInt_spdf8
#elif defined int_spdf9
__device__ __forceinline__ void iclass_AOInt_spdf9
#elif defined int_spdf10
__device__ __forceinline__ void iclass_AOInt_spdf10
#endif
(int I, int J, int K, int L, unsigned int II, unsigned int JJ, unsigned int KK, unsigned int LL, QUICKDouble DNMax, ERI_entry* aoint_buffer, int streamID, \
QUICKDouble* YVerticalTemp, QUICKDouble* store)
{
    
    /*
     kAtom A, B, C ,D is the coresponding atom for shell ii, jj, kk, ll
     and be careful with the index difference between Fortran and C++,
     Fortran starts array index with 1 and C++ starts 0.
     
     
     RA, RB, RC, and RD are the coordinates for atom katomA, katomB, katomC and katomD,
     which means they are corrosponding coorinates for shell II, JJ, KK, and LL.
     And we don't need the coordinates now, so we will not retrieve the data now.
     */
    QUICKDouble RAx = LOC2(devSim.xyz, 0 , devSim.katom[II]-1, 3, devSim.natom);
    QUICKDouble RAy = LOC2(devSim.xyz, 1 , devSim.katom[II]-1, 3, devSim.natom);
    QUICKDouble RAz = LOC2(devSim.xyz, 2 , devSim.katom[II]-1, 3, devSim.natom);
    
    QUICKDouble RCx = LOC2(devSim.xyz, 0 , devSim.katom[KK]-1, 3, devSim.natom);
    QUICKDouble RCy = LOC2(devSim.xyz, 1 , devSim.katom[KK]-1, 3, devSim.natom);
    QUICKDouble RCz = LOC2(devSim.xyz, 2 , devSim.katom[KK]-1, 3, devSim.natom);
    
    /*
     kPrimI, J, K and L indicates the primtive gaussian function number
     kStartI, J, K, and L indicates the starting guassian function for shell I, J, K, and L.
     We retrieve from global memory and save them to register to avoid multiple retrieve.
     */
    int kPrimI = devSim.kprim[II];
    int kPrimJ = devSim.kprim[JJ];
    int kPrimK = devSim.kprim[KK];
    int kPrimL = devSim.kprim[LL];
    
    int kStartI = devSim.kstart[II]-1;
    int kStartJ = devSim.kstart[JJ]-1;
    int kStartK = devSim.kstart[KK]-1;
    int kStartL = devSim.kstart[LL]-1;
    
    
    /*
     store saves temp contracted integral as [as|bs] type. the dimension should be allocatable but because
     of cuda limitation, we can not do that now.
     
     See M.Head-Gordon and J.A.Pople, Jchem.Phys., 89, No.9 (1988) for VRR algrithem details.
     */
    //QUICKDouble store[STOREDIM*STOREDIM];
    
    /*
     Initial the neccessary element for
     */
    for (int i = Sumindex[K+1]+1; i<= Sumindex[K+L+2]; i++) {
        for (int j = Sumindex[I+1]+1; j<= Sumindex[I+J+2]; j++) {
            if ( i <= STOREDIM && j <= STOREDIM) {
                LOCSTORE(store, j-1, i-1, STOREDIM, STOREDIM) = 0;
            }
        }
    }
    
    for (int i = 0; i<kPrimI*kPrimJ;i++){
        int JJJ = (int) i/kPrimI;
        int III = (int) i-kPrimI*JJJ;
        /*
         In the following comments, we have I, J, K, L denote the primitive gaussian function we use, and
         for example, expo(III, ksumtype(II)) stands for the expo for the IIIth primitive guassian function for II shell,
         we use I to express the corresponding index.
         AB = expo(I)+expo(J)
         --->                --->
         ->     expo(I) * xyz (I) + expo(J) * xyz(J)
         P  = ---------------------------------------
         expo(I) + expo(J)
         Those two are pre-calculated in CPU stage.
         
         */
        int ii_start = devSim.prim_start[II];
        int jj_start = devSim.prim_start[JJ];
        
        QUICKDouble AB = LOC2(devSim.expoSum, ii_start+III, jj_start+JJJ, devSim.prim_total, devSim.prim_total);
        QUICKDouble Px = LOC2(devSim.weightedCenterX, ii_start+III, jj_start+JJJ, devSim.prim_total, devSim.prim_total);
        QUICKDouble Py = LOC2(devSim.weightedCenterY, ii_start+III, jj_start+JJJ, devSim.prim_total, devSim.prim_total);
        QUICKDouble Pz = LOC2(devSim.weightedCenterZ, ii_start+III, jj_start+JJJ, devSim.prim_total, devSim.prim_total);
        
        /*
         X1 is the contracted coeffecient, which is pre-calcuated in CPU stage as well.
         cutoffprim is used to cut too small prim gaussian function when bring density matrix into consideration.
         */
        QUICKDouble cutoffPrim = DNMax * LOC2(devSim.cutPrim, kStartI+III, kStartJ+JJJ, devSim.jbasis, devSim.jbasis);
        QUICKDouble X1 = LOC4(devSim.Xcoeff, kStartI+III, kStartJ+JJJ, I - devSim.Qstart[II], J - devSim.Qstart[JJ], devSim.jbasis, devSim.jbasis, 2, 2);
        
        for (int j = 0; j<kPrimK*kPrimL; j++){
            int LLL = (int) j/kPrimK;
            int KKK = (int) j-kPrimK*LLL;
            
            if (cutoffPrim * LOC2(devSim.cutPrim, kStartK+KKK, kStartL+LLL, devSim.jbasis, devSim.jbasis) > devSim.primLimit) {
                /*
                 CD = expo(L)+expo(K)
                 ABCD = 1/ (AB + CD) = 1 / (expo(I)+expo(J)+expo(K)+expo(L))
                 
                 `````````````````````````AB * CD      (expo(I)+expo(J))*(expo(K)+expo(L))
                 Rou(Greek Letter) =   ----------- = ------------------------------------
                 `````````````````````````AB + CD         expo(I)+expo(J)+expo(K)+expo(L)
                 
                 ```````````````````expo(I)+expo(J)                        expo(K)+expo(L)
                 ABcom = --------------------------------  CDcom = --------------------------------
                 `````````expo(I)+expo(J)+expo(K)+expo(L)           expo(I)+expo(J)+expo(K)+expo(L)
                 
                 ABCDtemp = 1/2(expo(I)+expo(J)+expo(K)+expo(L))
                 */
                
                int kk_start = devSim.prim_start[KK];
                int ll_start = devSim.prim_start[LL];
                
                QUICKDouble CD = LOC2(devSim.expoSum, kk_start+KKK, ll_start+LLL, devSim.prim_total, devSim.prim_total);
                
                QUICKDouble ABCD = 1/(AB+CD);
                
                /*
                 X2 is the multiplication of four indices normalized coeffecient
                 */
                QUICKDouble X2 = sqrt(ABCD) * X1 * LOC4(devSim.Xcoeff, kStartK+KKK, kStartL+LLL, K - devSim.Qstart[KK], L - devSim.Qstart[LL], devSim.jbasis, devSim.jbasis, 2, 2);
                
                /*
                 Q' is the weighting center of K and L
                 ```````````````````````````--->           --->
                 ->  ------>       expo(K)*xyz(K)+expo(L)*xyz(L)
                 Q = P'(K,L)  = ------------------------------
                 `````````````````````````expo(K) + expo(L)
                 
                 W' is the weight center for I, J, K, L
                 
                 ```````````````--->             --->             --->            --->
                 ->     expo(I)*xyz(I) + expo(J)*xyz(J) + expo(K)*xyz(K) +expo(L)*xyz(L)
                 W = -------------------------------------------------------------------
                 `````````````````````````expo(I) + expo(J) + expo(K) + expo(L)
                 ``````->  ->  2
                 RPQ =| P - Q |
                 
                 ```````````->  -> 2
                 T = ROU * | P - Q|
                 */
                
                QUICKDouble Qx = LOC2(devSim.weightedCenterX, kk_start+KKK, ll_start+LLL, devSim.prim_total, devSim.prim_total);
                QUICKDouble Qy = LOC2(devSim.weightedCenterY, kk_start+KKK, ll_start+LLL, devSim.prim_total, devSim.prim_total);
                QUICKDouble Qz = LOC2(devSim.weightedCenterZ, kk_start+KKK, ll_start+LLL, devSim.prim_total, devSim.prim_total);
                
                QUICKDouble T = AB * CD * ABCD * ( quick_dsqr(Px-Qx) + quick_dsqr(Py-Qy) + quick_dsqr(Pz-Qz));
                
                //QUICKDouble YVerticalTemp[VDIM1*VDIM2*VDIM3];
                FmT(I+J+K+L, T, YVerticalTemp);
                for (int i = 0; i<=I+J+K+L; i++) {
                    VY(0, 0, i) = VY(0, 0, i) * X2;
                }
                
#ifdef int_spd
                vertical(I, J, K, L, YVerticalTemp, store, \
                         Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, \
                         Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, \
                         0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD);
#elif defined int_spdf
                
                vertical_spdf(I, J, K, L, YVerticalTemp, store, \
                              Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, \
                              Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, \
                              0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD);
#elif defined int_spdf2
                
                vertical_spdf2(I, J, K, L, YVerticalTemp, store, \
                               Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, \
                               Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, \
                               0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD);
#elif defined int_spdf3
                
                vertical_spdf3(I, J, K, L, YVerticalTemp, store, \
                               Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, \
                               Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, \
                               0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD);
#elif defined int_spdf4
                
                vertical_spdf4(I, J, K, L, YVerticalTemp, store, \
                               Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, \
                               Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, \
                               0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD);
#elif defined int_spdf5
                
                vertical_spdf5(I, J, K, L, YVerticalTemp, store, \
                               Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, \
                               Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, \
                               0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD);
#elif defined int_spdf6
                
                vertical_spdf6(I, J, K, L, YVerticalTemp, store, \
                               Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, \
                               Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, \
                               0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD);
#elif defined int_spdf7
                
                vertical_spdf7(I, J, K, L, YVerticalTemp, store, \
                               Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, \
                               Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, \
                               0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD);
#elif defined int_spdf8
                
                vertical_spdf8(I, J, K, L, YVerticalTemp, store, \
                               Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, \
                               Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, \
                               0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD);
#elif defined int_spdf9
                
                vertical_spdf9(I, J, K, L, YVerticalTemp, store, \
                               Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, \
                               Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, \
                               0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD);
#elif defined int_spdf10
                
                vertical_spdf10(I, J, K, L, YVerticalTemp, store, \
                               Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, \
                               Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, \
                               0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD);
#endif
            }
        }
    }
    
    
    // IJKLTYPE is the I, J, K,L type
    int IJKLTYPE = (int) (1000 * I + 100 *J + 10 * K + L);
    
    QUICKDouble RBx, RBy, RBz;
    QUICKDouble RDx, RDy, RDz;
    
    RBx = LOC2(devSim.xyz, 0 , devSim.katom[JJ]-1, 3, devSim.natom);
    RBy = LOC2(devSim.xyz, 1 , devSim.katom[JJ]-1, 3, devSim.natom);
    RBz = LOC2(devSim.xyz, 2 , devSim.katom[JJ]-1, 3, devSim.natom);
    
    
    RDx = LOC2(devSim.xyz, 0 , devSim.katom[LL]-1, 3, devSim.natom);
    RDy = LOC2(devSim.xyz, 1 , devSim.katom[LL]-1, 3, devSim.natom);
    RDz = LOC2(devSim.xyz, 2 , devSim.katom[LL]-1, 3, devSim.natom);
    
    int III1 = LOC2(devSim.Qsbasis, II, I, devSim.nshell, 4);
    int III2 = LOC2(devSim.Qfbasis, II, I, devSim.nshell, 4);
    int JJJ1 = LOC2(devSim.Qsbasis, JJ, J, devSim.nshell, 4);
    int JJJ2 = LOC2(devSim.Qfbasis, JJ, J, devSim.nshell, 4);
    int KKK1 = LOC2(devSim.Qsbasis, KK, K, devSim.nshell, 4);
    int KKK2 = LOC2(devSim.Qfbasis, KK, K, devSim.nshell, 4);
    int LLL1 = LOC2(devSim.Qsbasis, LL, L, devSim.nshell, 4);
    int LLL2 = LOC2(devSim.Qfbasis, LL, L, devSim.nshell, 4);
    
    
    // maxIJKL is the max of I,J,K,L
    int maxIJKL = (int)MAX(MAX(I,J),MAX(K,L));
    
    if (((maxIJKL == 2)&&(J != 0 || L!=0)) || (maxIJKL >= 3)) {
        IJKLTYPE = 999;
    }
    
    
    // Store generated ERI to buffer
    for (int III = III1; III <= III2; III++) {
        for (int JJJ = MAX(III,JJJ1); JJJ <= JJJ2; JJJ++) {
            for (int KKK = MAX(III,KKK1); KKK <= KKK2; KKK++) {
                for (int LLL = MAX(KKK,LLL1); LLL <= LLL2; LLL++) {
                    if( (III < JJJ && III < KKK && KKK < LLL) ||
                       (III < KKK || JJJ <= LLL)){
                        
#ifdef int_spd
                        QUICKDouble Y = (QUICKDouble) hrrwhole
#else
                        
                        QUICKDouble Y = (QUICKDouble) hrrwhole2
#endif
                        
                        ( I, J, K, L,\
                         III, JJJ, KKK, LLL, IJKLTYPE, store, \
                         RAx, RAy, RAz, RBx, RBy, RBz, \
                         RCx, RCy, RCz, RDx, RDy, RDz);
                        
                        if (abs(Y) > devSim.maxIntegralCutoff){
                            ERI_entry a;
                            a.value = Y;
                            a.IJ = (III - 1) * devSim.nbasis + JJJ - 1;
                            a.KL = (KKK - 1) * devSim.nbasis + LLL - 1;
                            
                            aoint_buffer[QUICKADD(devSim.intCount[streamID], 1)] = a;
                        }
                    }
                }
            }
        }
    }
}
#endif




#ifndef new_quick_2_gpu_get2e_subs_h
#define new_quick_2_gpu_get2e_subs_h


#ifdef OSHELL
__device__ __forceinline__ void addint_oshell(QUICKULL* oULL, QUICKULL* obULL,QUICKDouble Y, int III, int JJJ, int KKK, int LLL,QUICKDouble hybrid_coeff,  QUICKDouble* dense, QUICKDouble* denseb,int nbasis)
#else
__device__ __forceinline__ void addint(QUICKULL* oULL, QUICKDouble Y, int III, int JJJ, int KKK, int LLL,QUICKDouble hybrid_coeff,  QUICKDouble* dense, int nbasis)
#endif
{
    
#ifdef OSHELL
    QUICKDouble DENSELK = (QUICKDouble) (LOC2(dense, LLL-1, KKK-1, nbasis, nbasis)+LOC2(denseb, LLL-1, KKK-1, nbasis, nbasis));
    QUICKDouble DENSEJI = (QUICKDouble) (LOC2(dense, JJJ-1, III-1, nbasis, nbasis)+LOC2(denseb, JJJ-1, III-1, nbasis, nbasis));

    QUICKDouble DENSEKIA = (QUICKDouble) LOC2(dense, KKK-1, III-1, nbasis, nbasis);
    QUICKDouble DENSEKJA = (QUICKDouble) LOC2(dense, KKK-1, JJJ-1, nbasis, nbasis);
    QUICKDouble DENSELJA = (QUICKDouble) LOC2(dense, LLL-1, JJJ-1, nbasis, nbasis);
    QUICKDouble DENSELIA = (QUICKDouble) LOC2(dense, LLL-1, III-1, nbasis, nbasis);

    QUICKDouble DENSEKIB = (QUICKDouble) LOC2(denseb, KKK-1, III-1, nbasis, nbasis);
    QUICKDouble DENSEKJB = (QUICKDouble) LOC2(denseb, KKK-1, JJJ-1, nbasis, nbasis);
    QUICKDouble DENSELJB = (QUICKDouble) LOC2(denseb, LLL-1, JJJ-1, nbasis, nbasis);
    QUICKDouble DENSELIB = (QUICKDouble) LOC2(denseb, LLL-1, III-1, nbasis, nbasis);


    // ATOMIC ADD VALUE 1
    QUICKDouble _tmp = 2.0;
    if (KKK==LLL) {
        _tmp = 1.0;
    }

    QUICKDouble val1d = _tmp*DENSELK*Y;
    QUICKULL val1 = (QUICKULL) (fabs(val1d*OSCALE) + (QUICKDouble)0.5);
    if ( val1d < (QUICKDouble)0.0) val1 = 0ull - val1;
    QUICKADD(LOC2(oULL, JJJ-1, III-1, nbasis, nbasis), val1);
    QUICKADD(LOC2(obULL, JJJ-1, III-1, nbasis, nbasis), val1);

    // ATOMIC ADD VALUE 2
    if ((LLL != JJJ) || (III!=KKK)) {
        _tmp = 2.0;
        if (III==JJJ) {
            _tmp = 1.0;
        }

        QUICKDouble val2d = _tmp*DENSEJI*Y;
        QUICKULL val2 = (QUICKULL) (fabs(val2d*OSCALE) + (QUICKDouble)0.5);
        if ( val2d < (QUICKDouble)0.0) val2 = 0ull - val2;
        QUICKADD(LOC2(oULL, LLL-1, KKK-1, nbasis, nbasis), val2);
        QUICKADD(LOC2(obULL, LLL-1, KKK-1, nbasis, nbasis), val2);
    }

    // ATOMIC ADD VALUE 3
    QUICKDouble val3da = hybrid_coeff*DENSELJA*Y;

    QUICKULL val3a = (QUICKULL) (fabs(val3da*OSCALE) + (QUICKDouble)0.5);
    if (((III == KKK) && (III <  JJJ) && (JJJ < LLL))) {
        val3a = (QUICKULL) (fabs(2*val3da*OSCALE) + (QUICKDouble)0.5);
    }
    if ( DENSELJA*Y < (QUICKDouble)0.0) val3a = 0ull - val3a;
    QUICKADD(LOC2(oULL, KKK-1, III-1, nbasis, nbasis), 0ull-val3a);


    QUICKDouble val3db = hybrid_coeff*DENSELJB*Y;

    QUICKULL val3b = (QUICKULL) (fabs(val3db*OSCALE) + (QUICKDouble)0.5);
    if (((III == KKK) && (III <  JJJ) && (JJJ < LLL))) {
        val3b = (QUICKULL) (fabs(2*val3db*OSCALE) + (QUICKDouble)0.5);
    }
    if ( DENSELJB*Y < (QUICKDouble)0.0) val3b = 0ull - val3b;
    QUICKADD(LOC2(obULL, KKK-1, III-1, nbasis, nbasis), 0ull-val3b);

    // ATOMIC ADD VALUE 4
    if (KKK != LLL) {
        QUICKDouble val4da = hybrid_coeff*DENSEKJA*Y;

        QUICKULL val4a = (QUICKULL) (fabs(val4da*OSCALE) + (QUICKDouble)0.5);
        if ( val4da < (QUICKDouble)0.0) val4a = 0ull - val4a;
        QUICKADD(LOC2(oULL, LLL-1, III-1, nbasis, nbasis), 0ull-val4a);
    }


    if (KKK != LLL) {
        QUICKDouble val4db = hybrid_coeff*DENSEKJB*Y;

        QUICKULL val4b = (QUICKULL) (fabs(val4db*OSCALE) + (QUICKDouble)0.5);
        if ( val4db < (QUICKDouble)0.0) val4b = 0ull - val4b;
        QUICKADD(LOC2(obULL, LLL-1, III-1, nbasis, nbasis), 0ull-val4b);
    }

    // ATOMIC ADD VALUE 5
    QUICKDouble val5da = hybrid_coeff*DENSELIA*Y;

    QUICKULL val5a = (QUICKULL) (fabs(val5da*OSCALE) + (QUICKDouble)0.5);
    if ( val5da < (QUICKDouble)0.0) val5a = 0ull - val5a;

    if ((III != JJJ && III<KKK) || ((III == JJJ) && (III == KKK) && (III < LLL)) || ((III == KKK) && (III <  JJJ) && (JJJ < LLL))) {
        QUICKADD(LOC2(oULL, MAX(JJJ,KKK)-1, MIN(JJJ,KKK)-1, nbasis, nbasis), 0ull-val5a);
    }
    // ATOMIC ADD VALUE 5 - 2
    if ( III != JJJ && JJJ == KKK) {
        QUICKADD(LOC2(oULL, JJJ-1, KKK-1, nbasis, nbasis), 0ull-val5a);
    }


    QUICKDouble val5db = hybrid_coeff*DENSELIB*Y;

    QUICKULL val5b = (QUICKULL) (fabs(val5db*OSCALE) + (QUICKDouble)0.5);
    if ( val5db < (QUICKDouble)0.0) val5b = 0ull - val5b;

    if ((III != JJJ && III<KKK) || ((III == JJJ) && (III == KKK) && (III < LLL)) || ((III == KKK) && (III <  JJJ) && (JJJ < LLL))) {
        QUICKADD(LOC2(obULL, MAX(JJJ,KKK)-1, MIN(JJJ,KKK)-1, nbasis, nbasis), 0ull-val5b);
    }
    // ATOMIC ADD VALUE 5 - 2
    if ( III != JJJ && JJJ == KKK) {
        QUICKADD(LOC2(obULL, JJJ-1, KKK-1, nbasis, nbasis), 0ull-val5b);
    }

    // ATOMIC ADD VALUE 6
    if (III != JJJ) {
        if (KKK != LLL) {
            QUICKDouble val6da = hybrid_coeff*DENSEKIA*Y;
            QUICKULL val6a = (QUICKULL) (fabs(val6da*OSCALE) + (QUICKDouble)0.5);
            if ( val6da < (QUICKDouble)0.0) val6a = 0ull - val6a;

            QUICKADD(LOC2(oULL, MAX(JJJ,LLL)-1, MIN(JJJ,LLL)-1, devSim.nbasis, devSim.nbasis), 0ull-val6a);

            // ATOMIC ADD VALUE 6 - 2
            if (JJJ == LLL && III!= KKK) {
                QUICKADD(LOC2(oULL, LLL-1, JJJ-1, nbasis, nbasis), 0ull-val6a);
            }
        }
    }

    if (III != JJJ) {
        if (KKK != LLL) {
            QUICKDouble val6db = hybrid_coeff*DENSEKIB*Y;
            QUICKULL val6b = (QUICKULL) (fabs(val6db*OSCALE) + (QUICKDouble)0.5);
            if ( val6db < (QUICKDouble)0.0) val6b = 0ull - val6b;

            QUICKADD(LOC2(obULL, MAX(JJJ,LLL)-1, MIN(JJJ,LLL)-1, devSim.nbasis, devSim.nbasis), 0ull-val6b);

            // ATOMIC ADD VALUE 6 - 2
            if (JJJ == LLL && III!= KKK) {
                QUICKADD(LOC2(obULL, LLL-1, JJJ-1, nbasis, nbasis), 0ull-val6b);
            }
        }
    }

#else

    QUICKDouble DENSEKI = (QUICKDouble) LOC2(dense, KKK-1, III-1, nbasis, nbasis);
    QUICKDouble DENSEKJ = (QUICKDouble) LOC2(dense, KKK-1, JJJ-1, nbasis, nbasis);
    QUICKDouble DENSELJ = (QUICKDouble) LOC2(dense, LLL-1, JJJ-1, nbasis, nbasis);
    QUICKDouble DENSELI = (QUICKDouble) LOC2(dense, LLL-1, III-1, nbasis, nbasis);
    QUICKDouble DENSELK = (QUICKDouble) LOC2(dense, LLL-1, KKK-1, nbasis, nbasis);
    QUICKDouble DENSEJI = (QUICKDouble) LOC2(dense, JJJ-1, III-1, nbasis, nbasis);
    
    
    // ATOMIC ADD VALUE 1
    QUICKDouble _tmp = 2.0;
    if (KKK==LLL) {
        _tmp = 1.0;
    }
    
    QUICKDouble val1d = _tmp*DENSELK*Y;
    QUICKULL val1 = (QUICKULL) (fabs(val1d*OSCALE) + (QUICKDouble)0.5);
    if ( val1d < (QUICKDouble)0.0) val1 = 0ull - val1;
    QUICKADD(LOC2(oULL, JJJ-1, III-1, nbasis, nbasis), val1);
    
    
    // ATOMIC ADD VALUE 2
    if ((LLL != JJJ) || (III!=KKK)) {
        _tmp = 2.0;
        if (III==JJJ) {
            _tmp = 1.0;
        }
        
        QUICKDouble val2d = _tmp*DENSEJI*Y;
        QUICKULL val2 = (QUICKULL) (fabs(val2d*OSCALE) + (QUICKDouble)0.5);
        if ( val2d < (QUICKDouble)0.0) val2 = 0ull - val2;
        QUICKADD(LOC2(oULL, LLL-1, KKK-1, nbasis, nbasis), val2);
    }
    
    
    // ATOMIC ADD VALUE 3
    QUICKDouble val3d = hybrid_coeff*0.5*DENSELJ*Y;
    
    QUICKULL val3 = (QUICKULL) (fabs(val3d*OSCALE) + (QUICKDouble)0.5);
    if (((III == KKK) && (III <  JJJ) && (JJJ < LLL))) {
        val3 = (QUICKULL) (fabs(2*val3d*OSCALE) + (QUICKDouble)0.5);
    }
    if ( DENSELJ*Y < (QUICKDouble)0.0) val3 = 0ull - val3;
    QUICKADD(LOC2(oULL, KKK-1, III-1, nbasis, nbasis), 0ull-val3);
    
    // ATOMIC ADD VALUE 4
    if (KKK != LLL) {
        QUICKDouble val4d = hybrid_coeff*0.5*DENSEKJ*Y;
        
        QUICKULL val4 = (QUICKULL) (fabs(val4d*OSCALE) + (QUICKDouble)0.5);
        if ( val4d < (QUICKDouble)0.0) val4 = 0ull - val4;
        QUICKADD(LOC2(oULL, LLL-1, III-1, nbasis, nbasis), 0ull-val4);
    }
    
    
    
    // ATOMIC ADD VALUE 5
    QUICKDouble val5d = hybrid_coeff*0.5*DENSELI*Y;
    
    QUICKULL val5 = (QUICKULL) (fabs(val5d*OSCALE) + (QUICKDouble)0.5);
    if ( val5d < (QUICKDouble)0.0) val5 = 0ull - val5;
    
    if ((III != JJJ && III<KKK) || ((III == JJJ) && (III == KKK) && (III < LLL)) || ((III == KKK) && (III <  JJJ) && (JJJ < LLL))) {
        QUICKADD(LOC2(oULL, MAX(JJJ,KKK)-1, MIN(JJJ,KKK)-1, nbasis, nbasis), 0ull-val5);
    }
    
    
    // ATOMIC ADD VALUE 5 - 2
    if ( III != JJJ && JJJ == KKK) {
        QUICKADD(LOC2(oULL, JJJ-1, KKK-1, nbasis, nbasis), 0ull-val5);
    }
    
    // ATOMIC ADD VALUE 6
    if (III != JJJ) {
        if (KKK != LLL) {
            QUICKDouble val6d = hybrid_coeff*0.5*DENSEKI*Y;
            QUICKULL val6 = (QUICKULL) (fabs(val6d*OSCALE) + (QUICKDouble)0.5);
            if ( val6d < (QUICKDouble)0.0) val6 = 0ull - val6;
            
            QUICKADD(LOC2(oULL, MAX(JJJ,LLL)-1, MIN(JJJ,LLL)-1, devSim.nbasis, devSim.nbasis), 0ull-val6);
            
            // ATOMIC ADD VALUE 6 - 2
            if (JJJ == LLL && III!= KKK) {
                QUICKADD(LOC2(oULL, LLL-1, JJJ-1, nbasis, nbasis), 0ull-val6);
            }
        }
    }
#endif
}

#ifndef OSHELL
#include "gpu_fmt.h"

/*
 sqr for double precision. there no internal function to do that in fast-math-lib of CUDA
 */
__device__ __forceinline__ QUICKDouble quick_dsqr(QUICKDouble a)
{
    return a*a;
}

//#endif
#endif
#endif

#undef STOREDIM
