#include "../gpu_common.h"
#include "./h_all_files.h"
// Class for B =            4  L =            1
__device__ __inline__  f_4_1_t :: f_4_1_t ( f_4_0_t t_4_0_0, f_4_0_t t_4_0_1,  f_3_0_t t_3_0_1, QUICKDouble ABCDtemp, QUICKDouble Qtempx, QUICKDouble Qtempy, QUICKDouble Qtempz,  QUICKDouble WQtempx, QUICKDouble WQtempy, QUICKDouble WQtempz) {
    x_20_1 = Qtempx * t_4_0_0.x_20_0 + WQtempx * t_4_0_1.x_20_0 + 2 * ABCDtemp * t_3_0_1.x_12_0 ;
    x_21_1 = Qtempx * t_4_0_0.x_21_0 + WQtempx * t_4_0_1.x_21_0 + 2 * ABCDtemp * t_3_0_1.x_14_0 ;
    x_22_1 = Qtempx * t_4_0_0.x_22_0 + WQtempx * t_4_0_1.x_22_0 ;
    x_23_1 = Qtempx * t_4_0_0.x_23_0 + WQtempx * t_4_0_1.x_23_0 + 2 * ABCDtemp * t_3_0_1.x_10_0 ;
    x_24_1 = Qtempx * t_4_0_0.x_24_0 + WQtempx * t_4_0_1.x_24_0 + ABCDtemp * t_3_0_1.x_15_0 ;
    x_25_1 = Qtempx * t_4_0_0.x_25_0 + WQtempx * t_4_0_1.x_25_0 + ABCDtemp * t_3_0_1.x_16_0 ;
    x_26_1 = Qtempx * t_4_0_0.x_26_0 + WQtempx * t_4_0_1.x_26_0 + 3 * ABCDtemp * t_3_0_1.x_13_0 ;
    x_27_1 = Qtempx * t_4_0_0.x_27_0 + WQtempx * t_4_0_1.x_27_0 + ABCDtemp * t_3_0_1.x_19_0 ;
    x_28_1 = Qtempx * t_4_0_0.x_28_0 + WQtempx * t_4_0_1.x_28_0 + 3 * ABCDtemp * t_3_0_1.x_11_0 ;
    x_29_1 = Qtempx * t_4_0_0.x_29_0 + WQtempx * t_4_0_1.x_29_0 + ABCDtemp * t_3_0_1.x_18_0 ;
    x_30_1 = Qtempx * t_4_0_0.x_30_0 + WQtempx * t_4_0_1.x_30_0 ;
    x_31_1 = Qtempx * t_4_0_0.x_31_0 + WQtempx * t_4_0_1.x_31_0 ;
    x_32_1 = Qtempx * t_4_0_0.x_32_0 + WQtempx * t_4_0_1.x_32_0 + 4 * ABCDtemp * t_3_0_1.x_17_0 ;
    x_33_1 = Qtempx * t_4_0_0.x_33_0 + WQtempx * t_4_0_1.x_33_0 ;
    x_34_1 = Qtempx * t_4_0_0.x_34_0 + WQtempx * t_4_0_1.x_34_0 ;
    x_20_2 = Qtempy * t_4_0_0.x_20_0 + WQtempy * t_4_0_1.x_20_0 + 2 * ABCDtemp * t_3_0_1.x_11_0 ;
    x_21_2 = Qtempy * t_4_0_0.x_21_0 + WQtempy * t_4_0_1.x_21_0 ;
    x_22_2 = Qtempy * t_4_0_0.x_22_0 + WQtempy * t_4_0_1.x_22_0 + 2 * ABCDtemp * t_3_0_1.x_16_0 ;
    x_23_2 = Qtempy * t_4_0_0.x_23_0 + WQtempy * t_4_0_1.x_23_0 + ABCDtemp * t_3_0_1.x_13_0 ;
    x_24_2 = Qtempy * t_4_0_0.x_24_0 + WQtempy * t_4_0_1.x_24_0 + 2 * ABCDtemp * t_3_0_1.x_10_0 ;
    x_25_2 = Qtempy * t_4_0_0.x_25_0 + WQtempy * t_4_0_1.x_25_0 + ABCDtemp * t_3_0_1.x_14_0 ;
    x_26_2 = Qtempy * t_4_0_0.x_26_0 + WQtempy * t_4_0_1.x_26_0 ;
    x_27_2 = Qtempy * t_4_0_0.x_27_0 + WQtempy * t_4_0_1.x_27_0 ;
    x_28_2 = Qtempy * t_4_0_0.x_28_0 + WQtempy * t_4_0_1.x_28_0 + ABCDtemp * t_3_0_1.x_17_0 ;
    x_29_2 = Qtempy * t_4_0_0.x_29_0 + WQtempy * t_4_0_1.x_29_0 + 3 * ABCDtemp * t_3_0_1.x_12_0 ;
    x_30_2 = Qtempy * t_4_0_0.x_30_0 + WQtempy * t_4_0_1.x_30_0 + 3 * ABCDtemp * t_3_0_1.x_15_0 ;
    x_31_2 = Qtempy * t_4_0_0.x_31_0 + WQtempy * t_4_0_1.x_31_0 + ABCDtemp * t_3_0_1.x_19_0 ;
    x_32_2 = Qtempy * t_4_0_0.x_32_0 + WQtempy * t_4_0_1.x_32_0 ;
    x_33_2 = Qtempy * t_4_0_0.x_33_0 + WQtempy * t_4_0_1.x_33_0 + 4 * ABCDtemp * t_3_0_1.x_18_0 ;
    x_34_2 = Qtempy * t_4_0_0.x_34_0 + WQtempy * t_4_0_1.x_34_0 ;
    x_20_3 = Qtempz * t_4_0_0.x_20_0 + WQtempz * t_4_0_1.x_20_0 ;
    x_21_3 = Qtempz * t_4_0_0.x_21_0 + WQtempz * t_4_0_1.x_21_0 + 2 * ABCDtemp * t_3_0_1.x_13_0 ;
    x_22_3 = Qtempz * t_4_0_0.x_22_0 + WQtempz * t_4_0_1.x_22_0 + 2 * ABCDtemp * t_3_0_1.x_15_0 ;
    x_23_3 = Qtempz * t_4_0_0.x_23_0 + WQtempz * t_4_0_1.x_23_0 + ABCDtemp * t_3_0_1.x_11_0 ;
    x_24_3 = Qtempz * t_4_0_0.x_24_0 + WQtempz * t_4_0_1.x_24_0 + ABCDtemp * t_3_0_1.x_12_0 ;
    x_25_3 = Qtempz * t_4_0_0.x_25_0 + WQtempz * t_4_0_1.x_25_0 + 2 * ABCDtemp * t_3_0_1.x_10_0 ;
    x_26_3 = Qtempz * t_4_0_0.x_26_0 + WQtempz * t_4_0_1.x_26_0 + ABCDtemp * t_3_0_1.x_17_0 ;
    x_27_3 = Qtempz * t_4_0_0.x_27_0 + WQtempz * t_4_0_1.x_27_0 + 3 * ABCDtemp * t_3_0_1.x_14_0 ;
    x_28_3 = Qtempz * t_4_0_0.x_28_0 + WQtempz * t_4_0_1.x_28_0 ;
    x_29_3 = Qtempz * t_4_0_0.x_29_0 + WQtempz * t_4_0_1.x_29_0 ;
    x_30_3 = Qtempz * t_4_0_0.x_30_0 + WQtempz * t_4_0_1.x_30_0 + ABCDtemp * t_3_0_1.x_18_0 ;
    x_31_3 = Qtempz * t_4_0_0.x_31_0 + WQtempz * t_4_0_1.x_31_0 + 3 * ABCDtemp * t_3_0_1.x_16_0 ;
    x_32_3 = Qtempz * t_4_0_0.x_32_0 + WQtempz * t_4_0_1.x_32_0 ;
    x_33_3 = Qtempz * t_4_0_0.x_33_0 + WQtempz * t_4_0_1.x_33_0 ;
    x_34_3 = Qtempz * t_4_0_0.x_34_0 + WQtempz * t_4_0_1.x_34_0 + 4 * ABCDtemp * t_3_0_1.x_19_0 ;
}
