#include "../gpu_common.h"
#include "./h_all_files.h"
// Class for L =            0  B =            4
__device__ __inline__ f_0_4_t :: f_0_4_t ( f_0_3_t t_0_3_0, f_0_3_t t_0_3_1, f_0_2_t t_0_2_0, f_0_2_t t_0_2_1, QUICKDouble CDtemp, QUICKDouble ABcom, QUICKDouble Qtempx, QUICKDouble Qtempy, QUICKDouble Qtempz,  QUICKDouble WQtempx, QUICKDouble WQtempy, QUICKDouble WQtempz) {
    x_0_20 = Qtempx * t_0_3_0.x_0_12 + WQtempx * t_0_3_1.x_0_12 + CDtemp * ( t_0_2_0.x_0_8 -  ABcom * t_0_2_1.x_0_8 ) ;
    x_0_21 = Qtempx * t_0_3_0.x_0_14 + WQtempx * t_0_3_1.x_0_14 + CDtemp * ( t_0_2_0.x_0_9 -  ABcom * t_0_2_1.x_0_9 ) ;
    x_0_22 = Qtempy * t_0_3_0.x_0_16 + WQtempy * t_0_3_1.x_0_16 + CDtemp * ( t_0_2_0.x_0_9 -  ABcom * t_0_2_1.x_0_9 ) ;
    x_0_23 = Qtempx * t_0_3_0.x_0_10 + WQtempx * t_0_3_1.x_0_10 + CDtemp * ( t_0_2_0.x_0_5 -  ABcom * t_0_2_1.x_0_5 ) ;
    x_0_24 = Qtempx * t_0_3_0.x_0_15 + WQtempx * t_0_3_1.x_0_15 ;
    x_0_25 = Qtempx * t_0_3_0.x_0_16 + WQtempx * t_0_3_1.x_0_16 ;
    x_0_26 = Qtempx * t_0_3_0.x_0_13 + WQtempx * t_0_3_1.x_0_13 + CDtemp * 2 * ( t_0_2_0.x_0_6 -  ABcom * t_0_2_1.x_0_6 ) ;
    x_0_27 = Qtempx * t_0_3_0.x_0_19 + WQtempx * t_0_3_1.x_0_19 ;
    x_0_28 = Qtempx * t_0_3_0.x_0_11 + WQtempx * t_0_3_1.x_0_11 + CDtemp * 2 * ( t_0_2_0.x_0_4 -  ABcom * t_0_2_1.x_0_4 ) ;
    x_0_29 = Qtempx * t_0_3_0.x_0_18 + WQtempx * t_0_3_1.x_0_18 ;
    x_0_30 = Qtempy * t_0_3_0.x_0_15 + WQtempy * t_0_3_1.x_0_15 + CDtemp * 2 * ( t_0_2_0.x_0_5 -  ABcom * t_0_2_1.x_0_5 ) ;
    x_0_31 = Qtempy * t_0_3_0.x_0_19 + WQtempy * t_0_3_1.x_0_19 ;
    x_0_32 = Qtempx * t_0_3_0.x_0_17 + WQtempx * t_0_3_1.x_0_17 + CDtemp * 3 * ( t_0_2_0.x_0_7 -  ABcom * t_0_2_1.x_0_7 ) ;
    x_0_33 = Qtempy * t_0_3_0.x_0_18 + WQtempy * t_0_3_1.x_0_18 + CDtemp * 3 * ( t_0_2_0.x_0_8 -  ABcom * t_0_2_1.x_0_8 ) ;
    x_0_34 = Qtempz * t_0_3_0.x_0_19 + WQtempz * t_0_3_1.x_0_19 + CDtemp * 3 * ( t_0_2_0.x_0_9 -  ABcom * t_0_2_1.x_0_9 ) ;
}
