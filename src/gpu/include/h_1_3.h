#include "../gpu_common.h"
#include "./h_all_files.h"
// Class for L =            1  B =            3
__device__ __inline__  f_1_3_t :: f_1_3_t ( f_0_3_t t_0_3_0, f_0_3_t t_0_3_1,  f_0_2_t t_0_2_1, QUICKDouble ABCDtemp, QUICKDouble Ptempx, QUICKDouble Ptempy, QUICKDouble Ptempz,  QUICKDouble WPtempx, QUICKDouble WPtempy, QUICKDouble WPtempz) {
    x_1_10 = Ptempx * t_0_3_0.x_0_10 + WPtempx * t_0_3_1.x_0_10 + ABCDtemp * t_0_2_1.x_0_5 ;
    x_1_11 = Ptempx * t_0_3_0.x_0_11 + WPtempx * t_0_3_1.x_0_11 + 2 * ABCDtemp * t_0_2_1.x_0_4 ;
    x_1_12 = Ptempx * t_0_3_0.x_0_12 + WPtempx * t_0_3_1.x_0_12 + ABCDtemp * t_0_2_1.x_0_8 ;
    x_1_13 = Ptempx * t_0_3_0.x_0_13 + WPtempx * t_0_3_1.x_0_13 + 2 * ABCDtemp * t_0_2_1.x_0_6 ;
    x_1_14 = Ptempx * t_0_3_0.x_0_14 + WPtempx * t_0_3_1.x_0_14 + ABCDtemp * t_0_2_1.x_0_9 ;
    x_1_15 = Ptempx * t_0_3_0.x_0_15 + WPtempx * t_0_3_1.x_0_15 ;
    x_1_16 = Ptempx * t_0_3_0.x_0_16 + WPtempx * t_0_3_1.x_0_16 ;
    x_1_17 = Ptempx * t_0_3_0.x_0_17 + WPtempx * t_0_3_1.x_0_17 + 3 * ABCDtemp * t_0_2_1.x_0_7 ;
    x_1_18 = Ptempx * t_0_3_0.x_0_18 + WPtempx * t_0_3_1.x_0_18 ;
    x_1_19 = Ptempx * t_0_3_0.x_0_19 + WPtempx * t_0_3_1.x_0_19 ;
    x_2_10 = Ptempy * t_0_3_0.x_0_10 + WPtempy * t_0_3_1.x_0_10 + ABCDtemp * t_0_2_1.x_0_6 ;
    x_2_11 = Ptempy * t_0_3_0.x_0_11 + WPtempy * t_0_3_1.x_0_11 + ABCDtemp * t_0_2_1.x_0_7 ;
    x_2_12 = Ptempy * t_0_3_0.x_0_12 + WPtempy * t_0_3_1.x_0_12 + 2 * ABCDtemp * t_0_2_1.x_0_4 ;
    x_2_13 = Ptempy * t_0_3_0.x_0_13 + WPtempy * t_0_3_1.x_0_13 ;
    x_2_14 = Ptempy * t_0_3_0.x_0_14 + WPtempy * t_0_3_1.x_0_14 ;
    x_2_15 = Ptempy * t_0_3_0.x_0_15 + WPtempy * t_0_3_1.x_0_15 + 2 * ABCDtemp * t_0_2_1.x_0_5 ;
    x_2_16 = Ptempy * t_0_3_0.x_0_16 + WPtempy * t_0_3_1.x_0_16 + ABCDtemp * t_0_2_1.x_0_9 ;
    x_2_17 = Ptempy * t_0_3_0.x_0_17 + WPtempy * t_0_3_1.x_0_17 ;
    x_2_18 = Ptempy * t_0_3_0.x_0_18 + WPtempy * t_0_3_1.x_0_18 + 3 * ABCDtemp * t_0_2_1.x_0_8 ;
    x_2_19 = Ptempy * t_0_3_0.x_0_19 + WPtempy * t_0_3_1.x_0_19 ;
    x_3_10 = Ptempz * t_0_3_0.x_0_10 + WPtempz * t_0_3_1.x_0_10 + ABCDtemp * t_0_2_1.x_0_4 ;
    x_3_11 = Ptempz * t_0_3_0.x_0_11 + WPtempz * t_0_3_1.x_0_11 ;
    x_3_12 = Ptempz * t_0_3_0.x_0_12 + WPtempz * t_0_3_1.x_0_12 ;
    x_3_13 = Ptempz * t_0_3_0.x_0_13 + WPtempz * t_0_3_1.x_0_13 + ABCDtemp * t_0_2_1.x_0_7 ;
    x_3_14 = Ptempz * t_0_3_0.x_0_14 + WPtempz * t_0_3_1.x_0_14 + 2 * ABCDtemp * t_0_2_1.x_0_6 ;
    x_3_15 = Ptempz * t_0_3_0.x_0_15 + WPtempz * t_0_3_1.x_0_15 + ABCDtemp * t_0_2_1.x_0_8 ;
    x_3_16 = Ptempz * t_0_3_0.x_0_16 + WPtempz * t_0_3_1.x_0_16 + 2 * ABCDtemp * t_0_2_1.x_0_5 ;
    x_3_17 = Ptempz * t_0_3_0.x_0_17 + WPtempz * t_0_3_1.x_0_17 ;
    x_3_18 = Ptempz * t_0_3_0.x_0_18 + WPtempz * t_0_3_1.x_0_18 ;
    x_3_19 = Ptempz * t_0_3_0.x_0_19 + WPtempz * t_0_3_1.x_0_19 + 3 * ABCDtemp * t_0_2_1.x_0_9 ;
}
