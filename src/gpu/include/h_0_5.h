#include "../gpu_common.h"
#include "./h_all_files.h"
// Class for L =            0  B =            5
__device__ __inline__  f_0_5_t :: f_0_5_t ( f_0_4_t t_0_4_0, f_0_4_t t_0_4_1, f_0_3_t t_0_3_0, f_0_3_t t_0_3_1, QUICKDouble CDtemp, QUICKDouble ABcom, QUICKDouble Qtempx, QUICKDouble Qtempy, QUICKDouble Qtempz,  QUICKDouble WQtempx, QUICKDouble WQtempy, QUICKDouble WQtempz) {
    x_0_35 = Qtempx * t_0_4_0.x_0_22 + WQtempx * t_0_4_1.x_0_22 ;
    x_0_36 = Qtempx * t_0_4_0.x_0_25 + WQtempx * t_0_4_1.x_0_25 + CDtemp * ( t_0_3_0.x_0_16 -  ABcom * t_0_3_1.x_0_16 ) ;
    x_0_37 = Qtempx * t_0_4_0.x_0_24 + WQtempx * t_0_4_1.x_0_24 + CDtemp * ( t_0_3_0.x_0_15 -  ABcom * t_0_3_1.x_0_15 ) ;
    x_0_38 = Qtempx * t_0_4_0.x_0_23 + WQtempx * t_0_4_1.x_0_23 + CDtemp * 2 * ( t_0_3_0.x_0_10 -  ABcom * t_0_3_1.x_0_10 ) ;
    x_0_39 = Qtempx * t_0_4_0.x_0_30 + WQtempx * t_0_4_1.x_0_30 ;
    x_0_40 = Qtempx * t_0_4_0.x_0_31 + WQtempx * t_0_4_1.x_0_31 ;
    x_0_41 = Qtempy * t_0_4_0.x_0_31 + WQtempy * t_0_4_1.x_0_31 + CDtemp * ( t_0_3_0.x_0_19 -  ABcom * t_0_3_1.x_0_19 ) ;
    x_0_42 = Qtempy * t_0_4_0.x_0_22 + WQtempy * t_0_4_1.x_0_22 + CDtemp * 2 * ( t_0_3_0.x_0_16 -  ABcom * t_0_3_1.x_0_16 ) ;
    x_0_43 = Qtempx * t_0_4_0.x_0_27 + WQtempx * t_0_4_1.x_0_27 + CDtemp * ( t_0_3_0.x_0_19 -  ABcom * t_0_3_1.x_0_19 ) ;
    x_0_44 = Qtempx * t_0_4_0.x_0_21 + WQtempx * t_0_4_1.x_0_21 + CDtemp * 2 * ( t_0_3_0.x_0_14 -  ABcom * t_0_3_1.x_0_14 ) ;
    x_0_45 = Qtempx * t_0_4_0.x_0_29 + WQtempx * t_0_4_1.x_0_29 + CDtemp * ( t_0_3_0.x_0_18 -  ABcom * t_0_3_1.x_0_18 ) ;
    x_0_46 = Qtempx * t_0_4_0.x_0_20 + WQtempx * t_0_4_1.x_0_20 + CDtemp * 2 * ( t_0_3_0.x_0_12 -  ABcom * t_0_3_1.x_0_12 ) ;
    x_0_47 = Qtempy * t_0_4_0.x_0_34 + WQtempy * t_0_4_1.x_0_34 ;
    x_0_48 = Qtempy * t_0_4_0.x_0_30 + WQtempy * t_0_4_1.x_0_30 + CDtemp * 3 * ( t_0_3_0.x_0_15 -  ABcom * t_0_3_1.x_0_15 ) ;
    x_0_49 = Qtempx * t_0_4_0.x_0_34 + WQtempx * t_0_4_1.x_0_34 ;
    x_0_50 = Qtempx * t_0_4_0.x_0_26 + WQtempx * t_0_4_1.x_0_26 + CDtemp * 3 * ( t_0_3_0.x_0_13 -  ABcom * t_0_3_1.x_0_13 ) ;
    x_0_51 = Qtempx * t_0_4_0.x_0_33 + WQtempx * t_0_4_1.x_0_33 ;
    x_0_52 = Qtempx * t_0_4_0.x_0_28 + WQtempx * t_0_4_1.x_0_28 + CDtemp * 3 * ( t_0_3_0.x_0_11 -  ABcom * t_0_3_1.x_0_11 ) ;
    x_0_53 = Qtempx * t_0_4_0.x_0_32 + WQtempx * t_0_4_1.x_0_32 + CDtemp * 4 * ( t_0_3_0.x_0_17 -  ABcom * t_0_3_1.x_0_17 ) ;
    x_0_54 = Qtempy * t_0_4_0.x_0_33 + WQtempy * t_0_4_1.x_0_33 + CDtemp * 4 * ( t_0_3_0.x_0_18 -  ABcom * t_0_3_1.x_0_18 ) ;
    x_0_55 = Qtempz * t_0_4_0.x_0_34 + WQtempz * t_0_4_1.x_0_34 + CDtemp * 4 * ( t_0_3_0.x_0_19 -  ABcom * t_0_3_1.x_0_19 ) ;
}
