#include "../gpu_common.h"
#include "./h_all_files.h"
// Class for L =            1  B =            2
__device__ __inline__  f_1_2_t :: f_1_2_t ( f_0_2_t t_0_2_0, f_0_2_t t_0_2_1,  f_0_1_t t_0_1_1, QUICKDouble ABCDtemp, QUICKDouble Ptempx, QUICKDouble Ptempy, QUICKDouble Ptempz,  QUICKDouble WPtempx, QUICKDouble WPtempy, QUICKDouble WPtempz) {
    x_1_4 = Ptempx * t_0_2_0.x_0_4 + WPtempx * t_0_2_1.x_0_4 + ABCDtemp * t_0_1_1.x_0_2 ;
    x_1_5 = Ptempx * t_0_2_0.x_0_5 + WPtempx * t_0_2_1.x_0_5 ;
    x_1_6 = Ptempx * t_0_2_0.x_0_6 + WPtempx * t_0_2_1.x_0_6 + ABCDtemp * t_0_1_1.x_0_3 ;
    x_1_7 = Ptempx * t_0_2_0.x_0_7 + WPtempx * t_0_2_1.x_0_7 + 2 * ABCDtemp * t_0_1_1.x_0_1 ;
    x_1_8 = Ptempx * t_0_2_0.x_0_8 + WPtempx * t_0_2_1.x_0_8 ;
    x_1_9 = Ptempx * t_0_2_0.x_0_9 + WPtempx * t_0_2_1.x_0_9 ;
    x_2_4 = Ptempy * t_0_2_0.x_0_4 + WPtempy * t_0_2_1.x_0_4 + ABCDtemp * t_0_1_1.x_0_1 ;
    x_2_5 = Ptempy * t_0_2_0.x_0_5 + WPtempy * t_0_2_1.x_0_5 + ABCDtemp * t_0_1_1.x_0_3 ;
    x_2_6 = Ptempy * t_0_2_0.x_0_6 + WPtempy * t_0_2_1.x_0_6 ;
    x_2_7 = Ptempy * t_0_2_0.x_0_7 + WPtempy * t_0_2_1.x_0_7 ;
    x_2_8 = Ptempy * t_0_2_0.x_0_8 + WPtempy * t_0_2_1.x_0_8 + 2 * ABCDtemp * t_0_1_1.x_0_2 ;
    x_2_9 = Ptempy * t_0_2_0.x_0_9 + WPtempy * t_0_2_1.x_0_9 ;
    x_3_4 = Ptempz * t_0_2_0.x_0_4 + WPtempz * t_0_2_1.x_0_4 ;
    x_3_5 = Ptempz * t_0_2_0.x_0_5 + WPtempz * t_0_2_1.x_0_5 + ABCDtemp * t_0_1_1.x_0_2 ;
    x_3_6 = Ptempz * t_0_2_0.x_0_6 + WPtempz * t_0_2_1.x_0_6 + ABCDtemp * t_0_1_1.x_0_1 ;
    x_3_7 = Ptempz * t_0_2_0.x_0_7 + WPtempz * t_0_2_1.x_0_7 ;
    x_3_8 = Ptempz * t_0_2_0.x_0_8 + WPtempz * t_0_2_1.x_0_8 ;
    x_3_9 = Ptempz * t_0_2_0.x_0_9 + WPtempz * t_0_2_1.x_0_9 + 2 * ABCDtemp * t_0_1_1.x_0_3 ;
}
