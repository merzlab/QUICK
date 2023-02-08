#include "../gpu_common.h"
#include "./h_all_files.h"
// Class for B =            3  L =            1
__device__ __inline__  f_3_1_t :: f_3_1_t ( f_3_0_t t_3_0_0, f_3_0_t t_3_0_1,  f_2_0_t t_2_0_1, QUICKDouble ABCDtemp, QUICKDouble Qtempx, QUICKDouble Qtempy, QUICKDouble Qtempz,  QUICKDouble WQtempx, QUICKDouble WQtempy, QUICKDouble WQtempz) {
    x_10_1 = Qtempx * t_3_0_0.x_10_0 + WQtempx * t_3_0_1.x_10_0 + ABCDtemp * t_2_0_1.x_5_0 ;
    x_11_1 = Qtempx * t_3_0_0.x_11_0 + WQtempx * t_3_0_1.x_11_0 + 2 * ABCDtemp * t_2_0_1.x_4_0 ;
    x_12_1 = Qtempx * t_3_0_0.x_12_0 + WQtempx * t_3_0_1.x_12_0 + ABCDtemp * t_2_0_1.x_8_0 ;
    x_13_1 = Qtempx * t_3_0_0.x_13_0 + WQtempx * t_3_0_1.x_13_0 + 2 * ABCDtemp * t_2_0_1.x_6_0 ;
    x_14_1 = Qtempx * t_3_0_0.x_14_0 + WQtempx * t_3_0_1.x_14_0 + ABCDtemp * t_2_0_1.x_9_0 ;
    x_15_1 = Qtempx * t_3_0_0.x_15_0 + WQtempx * t_3_0_1.x_15_0 ;
    x_16_1 = Qtempx * t_3_0_0.x_16_0 + WQtempx * t_3_0_1.x_16_0 ;
    x_17_1 = Qtempx * t_3_0_0.x_17_0 + WQtempx * t_3_0_1.x_17_0 + 3 * ABCDtemp * t_2_0_1.x_7_0 ;
    x_18_1 = Qtempx * t_3_0_0.x_18_0 + WQtempx * t_3_0_1.x_18_0 ;
    x_19_1 = Qtempx * t_3_0_0.x_19_0 + WQtempx * t_3_0_1.x_19_0 ;
    x_10_2 = Qtempy * t_3_0_0.x_10_0 + WQtempy * t_3_0_1.x_10_0 + ABCDtemp * t_2_0_1.x_6_0 ;
    x_11_2 = Qtempy * t_3_0_0.x_11_0 + WQtempy * t_3_0_1.x_11_0 + ABCDtemp * t_2_0_1.x_7_0 ;
    x_12_2 = Qtempy * t_3_0_0.x_12_0 + WQtempy * t_3_0_1.x_12_0 + 2 * ABCDtemp * t_2_0_1.x_4_0 ;
    x_13_2 = Qtempy * t_3_0_0.x_13_0 + WQtempy * t_3_0_1.x_13_0 ;
    x_14_2 = Qtempy * t_3_0_0.x_14_0 + WQtempy * t_3_0_1.x_14_0 ;
    x_15_2 = Qtempy * t_3_0_0.x_15_0 + WQtempy * t_3_0_1.x_15_0 + 2 * ABCDtemp * t_2_0_1.x_5_0 ;
    x_16_2 = Qtempy * t_3_0_0.x_16_0 + WQtempy * t_3_0_1.x_16_0 + ABCDtemp * t_2_0_1.x_9_0 ;
    x_17_2 = Qtempy * t_3_0_0.x_17_0 + WQtempy * t_3_0_1.x_17_0 ;
    x_18_2 = Qtempy * t_3_0_0.x_18_0 + WQtempy * t_3_0_1.x_18_0 + 3 * ABCDtemp * t_2_0_1.x_8_0 ;
    x_19_2 = Qtempy * t_3_0_0.x_19_0 + WQtempy * t_3_0_1.x_19_0 ;
    x_10_3 = Qtempz * t_3_0_0.x_10_0 + WQtempz * t_3_0_1.x_10_0 + ABCDtemp * t_2_0_1.x_4_0 ;
    x_11_3 = Qtempz * t_3_0_0.x_11_0 + WQtempz * t_3_0_1.x_11_0 ;
    x_12_3 = Qtempz * t_3_0_0.x_12_0 + WQtempz * t_3_0_1.x_12_0 ;
    x_13_3 = Qtempz * t_3_0_0.x_13_0 + WQtempz * t_3_0_1.x_13_0 + ABCDtemp * t_2_0_1.x_7_0 ;
    x_14_3 = Qtempz * t_3_0_0.x_14_0 + WQtempz * t_3_0_1.x_14_0 + 2 * ABCDtemp * t_2_0_1.x_6_0 ;
    x_15_3 = Qtempz * t_3_0_0.x_15_0 + WQtempz * t_3_0_1.x_15_0 + ABCDtemp * t_2_0_1.x_8_0 ;
    x_16_3 = Qtempz * t_3_0_0.x_16_0 + WQtempz * t_3_0_1.x_16_0 + 2 * ABCDtemp * t_2_0_1.x_5_0 ;
    x_17_3 = Qtempz * t_3_0_0.x_17_0 + WQtempz * t_3_0_1.x_17_0 ;
    x_18_3 = Qtempz * t_3_0_0.x_18_0 + WQtempz * t_3_0_1.x_18_0 ;
    x_19_3 = Qtempz * t_3_0_0.x_19_0 + WQtempz * t_3_0_1.x_19_0 + 3 * ABCDtemp * t_2_0_1.x_9_0 ;
}
