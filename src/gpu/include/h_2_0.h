#include "../gpu_common.h"
#include "./h_all_files.h"
// Class for L =            2  B =            0
__device__ __inline__  f_2_0_t :: f_2_0_t ( f_1_0_t t_1_0_0, f_1_0_t t_1_0_1, QUICKDouble t_0_0_0, QUICKDouble t_0_0_1, QUICKDouble ABtemp, QUICKDouble CDcom, QUICKDouble Ptempx, QUICKDouble Ptempy, QUICKDouble Ptempz,  QUICKDouble WPtempx, QUICKDouble WPtempy, QUICKDouble WPtempz) {
    x_4_0 = Ptempx * t_1_0_0.x_2_0 + WPtempx * t_1_0_1.x_2_0 ;
    x_5_0 = Ptempy * t_1_0_0.x_3_0 + WPtempy * t_1_0_1.x_3_0 ;
    x_6_0 = Ptempx * t_1_0_0.x_3_0 + WPtempx * t_1_0_1.x_3_0 ;
    x_7_0 = Ptempx * t_1_0_0.x_1_0 + WPtempx * t_1_0_1.x_1_0 + ABtemp * ( t_0_0_0 -  CDcom * t_0_0_1 ) ;
    x_8_0 = Ptempy * t_1_0_0.x_2_0 + WPtempy * t_1_0_1.x_2_0 + ABtemp * ( t_0_0_0 -  CDcom * t_0_0_1 ) ;
    x_9_0 = Ptempz * t_1_0_0.x_3_0 + WPtempz * t_1_0_1.x_3_0 + ABtemp * ( t_0_0_0 -  CDcom * t_0_0_1 ) ;
}

