#include "../gpu_common.h"
#include "./h_all_files.h"
// Class for L =            0  B =            3
__device__ __inline__  f_0_3_t :: f_0_3_t ( f_0_2_t t_0_2_0, f_0_2_t t_0_2_1, f_0_1_t t_0_1_0, f_0_1_t t_0_1_1, QUICKDouble CDtemp, QUICKDouble ABcom, QUICKDouble Qtempx, QUICKDouble Qtempy, QUICKDouble Qtempz,  QUICKDouble WQtempx, QUICKDouble WQtempy, QUICKDouble WQtempz) {
    x_0_10 = Qtempx * t_0_2_0.x_0_5 + WQtempx * t_0_2_1.x_0_5 ;
    x_0_11 = Qtempx * t_0_2_0.x_0_4 + WQtempx * t_0_2_1.x_0_4 + CDtemp * ( t_0_1_0.x_0_2 -  ABcom * t_0_1_1.x_0_2 ) ;
    x_0_12 = Qtempx * t_0_2_0.x_0_8 + WQtempx * t_0_2_1.x_0_8 ;
    x_0_13 = Qtempx * t_0_2_0.x_0_6 + WQtempx * t_0_2_1.x_0_6 + CDtemp * ( t_0_1_0.x_0_3 -  ABcom * t_0_1_1.x_0_3 ) ;
    x_0_14 = Qtempx * t_0_2_0.x_0_9 + WQtempx * t_0_2_1.x_0_9 ;
    x_0_15 = Qtempy * t_0_2_0.x_0_5 + WQtempy * t_0_2_1.x_0_5 + CDtemp * ( t_0_1_0.x_0_3 -  ABcom * t_0_1_1.x_0_3 ) ;
    x_0_16 = Qtempy * t_0_2_0.x_0_9 + WQtempy * t_0_2_1.x_0_9 ;
    x_0_17 = Qtempx * t_0_2_0.x_0_7 + WQtempx * t_0_2_1.x_0_7 + CDtemp * 2 * ( t_0_1_0.x_0_1 -  ABcom * t_0_1_1.x_0_1 ) ;
    x_0_18 = Qtempy * t_0_2_0.x_0_8 + WQtempy * t_0_2_1.x_0_8 + CDtemp * 2 * ( t_0_1_0.x_0_2 -  ABcom * t_0_1_1.x_0_2 ) ;
    x_0_19 = Qtempz * t_0_2_0.x_0_9 + WQtempz * t_0_2_1.x_0_9 + CDtemp * 2 * ( t_0_1_0.x_0_3 -  ABcom * t_0_1_1.x_0_3 ) ;
}
