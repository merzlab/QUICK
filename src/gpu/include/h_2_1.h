#include "../gpu_common.h"
#include "./h_all_files.h"
// Class for B =            2  L =            1
__device__ __inline__  f_2_1_t :: f_2_1_t ( f_2_0_t t_2_0_0, f_2_0_t t_2_0_1,  f_1_0_t t_1_0_1, QUICKDouble ABCDtemp, QUICKDouble Qtempx, QUICKDouble Qtempy, QUICKDouble Qtempz,  QUICKDouble WQtempx, QUICKDouble WQtempy, QUICKDouble WQtempz) {
    x_4_1 = Qtempx * t_2_0_0.x_4_0 + WQtempx * t_2_0_1.x_4_0 + ABCDtemp * t_1_0_1.x_2_0 ;
    x_5_1 = Qtempx * t_2_0_0.x_5_0 + WQtempx * t_2_0_1.x_5_0 ;
    x_6_1 = Qtempx * t_2_0_0.x_6_0 + WQtempx * t_2_0_1.x_6_0 + ABCDtemp * t_1_0_1.x_3_0 ;
    x_7_1 = Qtempx * t_2_0_0.x_7_0 + WQtempx * t_2_0_1.x_7_0 + 2 * ABCDtemp * t_1_0_1.x_1_0 ;
    x_8_1 = Qtempx * t_2_0_0.x_8_0 + WQtempx * t_2_0_1.x_8_0 ;
    x_9_1 = Qtempx * t_2_0_0.x_9_0 + WQtempx * t_2_0_1.x_9_0 ;
    x_4_2 = Qtempy * t_2_0_0.x_4_0 + WQtempy * t_2_0_1.x_4_0 + ABCDtemp * t_1_0_1.x_1_0 ;
    x_5_2 = Qtempy * t_2_0_0.x_5_0 + WQtempy * t_2_0_1.x_5_0 + ABCDtemp * t_1_0_1.x_3_0 ;
    x_6_2 = Qtempy * t_2_0_0.x_6_0 + WQtempy * t_2_0_1.x_6_0 ;
    x_7_2 = Qtempy * t_2_0_0.x_7_0 + WQtempy * t_2_0_1.x_7_0 ;
    x_8_2 = Qtempy * t_2_0_0.x_8_0 + WQtempy * t_2_0_1.x_8_0 + 2 * ABCDtemp * t_1_0_1.x_2_0 ;
    x_9_2 = Qtempy * t_2_0_0.x_9_0 + WQtempy * t_2_0_1.x_9_0 ;
    x_4_3 = Qtempz * t_2_0_0.x_4_0 + WQtempz * t_2_0_1.x_4_0 ;
    x_5_3 = Qtempz * t_2_0_0.x_5_0 + WQtempz * t_2_0_1.x_5_0 + ABCDtemp * t_1_0_1.x_2_0 ;
    x_6_3 = Qtempz * t_2_0_0.x_6_0 + WQtempz * t_2_0_1.x_6_0 + ABCDtemp * t_1_0_1.x_1_0 ;
    x_7_3 = Qtempz * t_2_0_0.x_7_0 + WQtempz * t_2_0_1.x_7_0 ;
    x_8_3 = Qtempz * t_2_0_0.x_8_0 + WQtempz * t_2_0_1.x_8_0 ;
    x_9_3 = Qtempz * t_2_0_0.x_9_0 + WQtempz * t_2_0_1.x_9_0 + 2 * ABCDtemp * t_1_0_1.x_3_0 ;
}
