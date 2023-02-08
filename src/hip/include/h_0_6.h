#include "../gpu_common.h"
#include "./h_all_files.h"
// Class for L =            0  B =            6
__device__ __inline__ f_0_6_t :: f_0_6_t ( f_0_5_t t_0_5_0, f_0_5_t t_0_5_1, f_0_4_t t_0_4_0, f_0_4_t t_0_4_1, QUICKDouble CDtemp, QUICKDouble ABcom, QUICKDouble Qtempx, QUICKDouble Qtempy, QUICKDouble Qtempz,  QUICKDouble WQtempx, QUICKDouble WQtempy, QUICKDouble WQtempz) {
    x_0_56 = Qtempx * t_0_5_0.x_0_38 + WQtempx * t_0_5_1.x_0_38 + CDtemp * 3 * ( t_0_4_0.x_0_23 -  ABcom * t_0_4_1.x_0_23 ) ;
    x_0_57 = Qtempx * t_0_5_0.x_0_48 + WQtempx * t_0_5_1.x_0_48 ;
    x_0_58 = Qtempx * t_0_5_0.x_0_47 + WQtempx * t_0_5_1.x_0_47 ;
    x_0_59 = Qtempx * t_0_5_0.x_0_41 + WQtempx * t_0_5_1.x_0_41 ;
    x_0_60 = Qtempx * t_0_5_0.x_0_42 + WQtempx * t_0_5_1.x_0_42 ;
    x_0_61 = Qtempx * t_0_5_0.x_0_40 + WQtempx * t_0_5_1.x_0_40 + CDtemp * ( t_0_4_0.x_0_31 -  ABcom * t_0_4_1.x_0_31 ) ;
    x_0_62 = Qtempx * t_0_5_0.x_0_36 + WQtempx * t_0_5_1.x_0_36 + CDtemp * 2 * ( t_0_4_0.x_0_25 -  ABcom * t_0_4_1.x_0_25 ) ;
    x_0_63 = Qtempx * t_0_5_0.x_0_39 + WQtempx * t_0_5_1.x_0_39 + CDtemp * ( t_0_4_0.x_0_30 -  ABcom * t_0_4_1.x_0_30 ) ;
    x_0_64 = Qtempx * t_0_5_0.x_0_37 + WQtempx * t_0_5_1.x_0_37 + CDtemp * 2 * ( t_0_4_0.x_0_24 -  ABcom * t_0_4_1.x_0_24 ) ;
    x_0_65 = Qtempx * t_0_5_0.x_0_35 + WQtempx * t_0_5_1.x_0_35 + CDtemp * ( t_0_4_0.x_0_22 -  ABcom * t_0_4_1.x_0_22 ) ;
    x_0_66 = Qtempy * t_0_5_0.x_0_55 + WQtempy * t_0_5_1.x_0_55 ;
    x_0_67 = Qtempy * t_0_5_0.x_0_48 + WQtempy * t_0_5_1.x_0_48 + CDtemp * 4 * ( t_0_4_0.x_0_30 -  ABcom * t_0_4_1.x_0_30 ) ;
    x_0_68 = Qtempx * t_0_5_0.x_0_55 + WQtempx * t_0_5_1.x_0_55 ;
    x_0_69 = Qtempx * t_0_5_0.x_0_50 + WQtempx * t_0_5_1.x_0_50 + CDtemp * 4 * ( t_0_4_0.x_0_26 -  ABcom * t_0_4_1.x_0_26 ) ;
    x_0_70 = Qtempx * t_0_5_0.x_0_54 + WQtempx * t_0_5_1.x_0_54 ;
    x_0_71 = Qtempx * t_0_5_0.x_0_52 + WQtempx * t_0_5_1.x_0_52 + CDtemp * 4 * ( t_0_4_0.x_0_28 -  ABcom * t_0_4_1.x_0_28 ) ;
    x_0_72 = Qtempy * t_0_5_0.x_0_47 + WQtempy * t_0_5_1.x_0_47 + CDtemp * ( t_0_4_0.x_0_34 -  ABcom * t_0_4_1.x_0_34 ) ;
    x_0_73 = Qtempy * t_0_5_0.x_0_42 + WQtempy * t_0_5_1.x_0_42 + CDtemp * 3 * ( t_0_4_0.x_0_22 -  ABcom * t_0_4_1.x_0_22 ) ;
    x_0_74 = Qtempx * t_0_5_0.x_0_49 + WQtempx * t_0_5_1.x_0_49 + CDtemp * ( t_0_4_0.x_0_34 -  ABcom * t_0_4_1.x_0_34 ) ;
    x_0_75 = Qtempx * t_0_5_0.x_0_44 + WQtempx * t_0_5_1.x_0_44 + CDtemp * 3 * ( t_0_4_0.x_0_21 -  ABcom * t_0_4_1.x_0_21 ) ;
    x_0_76 = Qtempx * t_0_5_0.x_0_51 + WQtempx * t_0_5_1.x_0_51 + CDtemp * ( t_0_4_0.x_0_33 -  ABcom * t_0_4_1.x_0_33 ) ;
    x_0_77 = Qtempx * t_0_5_0.x_0_46 + WQtempx * t_0_5_1.x_0_46 + CDtemp * 3 * ( t_0_4_0.x_0_20 -  ABcom * t_0_4_1.x_0_20 ) ;
    x_0_78 = Qtempy * t_0_5_0.x_0_41 + WQtempy * t_0_5_1.x_0_41 + CDtemp * 2 * ( t_0_4_0.x_0_31 -  ABcom * t_0_4_1.x_0_31 ) ;
    x_0_79 = Qtempx * t_0_5_0.x_0_43 + WQtempx * t_0_5_1.x_0_43 + CDtemp * 2 * ( t_0_4_0.x_0_27 -  ABcom * t_0_4_1.x_0_27 ) ;
    x_0_80 = Qtempx * t_0_5_0.x_0_45 + WQtempx * t_0_5_1.x_0_45 + CDtemp * 2 * ( t_0_4_0.x_0_29 -  ABcom * t_0_4_1.x_0_29 ) ;
    x_0_81 = Qtempx * t_0_5_0.x_0_53 + WQtempx * t_0_5_1.x_0_53 + CDtemp * 5 * ( t_0_4_0.x_0_32 -  ABcom * t_0_4_1.x_0_32 ) ;
    x_0_82 = Qtempy * t_0_5_0.x_0_54 + WQtempy * t_0_5_1.x_0_54 + CDtemp * 5 * ( t_0_4_0.x_0_33 -  ABcom * t_0_4_1.x_0_33 ) ;
    x_0_83 = Qtempz * t_0_5_0.x_0_55 + WQtempz * t_0_5_1.x_0_55 + CDtemp * 5 * ( t_0_4_0.x_0_34 -  ABcom * t_0_4_1.x_0_34 ) ;
}
