#include "../gpu_common.h"
#include "./h_all_files.h"
// Class for L =            1  B =            1
__device__ __inline__ f_1_1_t :: f_1_1_t ( f_0_1_t t_0_1_0, f_0_1_t t_0_1_1,  QUICKDouble t_0_0_1, QUICKDouble ABCDtemp, QUICKDouble Ptempx, QUICKDouble Ptempy, QUICKDouble Ptempz,  QUICKDouble WPtempx, QUICKDouble WPtempy, QUICKDouble WPtempz) {
    x_1_1 = Ptempx * t_0_1_0.x_0_1 + WPtempx * t_0_1_1.x_0_1 + ABCDtemp * t_0_0_1 ;
    x_1_2 = Ptempx * t_0_1_0.x_0_2 + WPtempx * t_0_1_1.x_0_2 ;
    x_1_3 = Ptempx * t_0_1_0.x_0_3 + WPtempx * t_0_1_1.x_0_3 ;
    x_2_1 = Ptempy * t_0_1_0.x_0_1 + WPtempy * t_0_1_1.x_0_1 ;
    x_2_2 = Ptempy * t_0_1_0.x_0_2 + WPtempy * t_0_1_1.x_0_2 + ABCDtemp * t_0_0_1 ;
    x_2_3 = Ptempy * t_0_1_0.x_0_3 + WPtempy * t_0_1_1.x_0_3 ;
    x_3_1 = Ptempz * t_0_1_0.x_0_1 + WPtempz * t_0_1_1.x_0_1 ;
    x_3_2 = Ptempz * t_0_1_0.x_0_2 + WPtempz * t_0_1_1.x_0_2 ;
    x_3_3 = Ptempz * t_0_1_0.x_0_3 + WPtempz * t_0_1_1.x_0_3 + ABCDtemp * t_0_0_1 ;
}
