#include "../gpu_common.h"
#include "./h_all_files.h"
// Class for L =            0  B =            7
__device__ __inline__ f_0_7_t :: f_0_7_t ( f_0_6_t t_0_6_0, f_0_6_t t_0_6_1, f_0_5_t t_0_5_0, f_0_5_t t_0_5_1, QUICKDouble CDtemp, QUICKDouble ABcom, QUICKDouble Qtempx, QUICKDouble Qtempy, QUICKDouble Qtempz,  QUICKDouble WQtempx, QUICKDouble WQtempy, QUICKDouble WQtempz) {
    x_0_84 = Qtempx * t_0_6_0.x_0_56 + WQtempx * t_0_6_1.x_0_56 + CDtemp * 4 * ( t_0_5_0.x_0_38 -  ABcom * t_0_5_1.x_0_38 ) ;
    x_0_85 = Qtempx * t_0_6_0.x_0_67 + WQtempx * t_0_6_1.x_0_67 ;
    x_0_86 = Qtempx * t_0_6_0.x_0_66 + WQtempx * t_0_6_1.x_0_66 ;
    x_0_87 = Qtempx * t_0_6_0.x_0_72 + WQtempx * t_0_6_1.x_0_72 ;
    x_0_88 = Qtempx * t_0_6_0.x_0_73 + WQtempx * t_0_6_1.x_0_73 ;
    x_0_89 = Qtempx * t_0_6_0.x_0_58 + WQtempx * t_0_6_1.x_0_58 + CDtemp * ( t_0_5_0.x_0_47 -  ABcom * t_0_5_1.x_0_47 ) ;
    x_0_90 = Qtempx * t_0_6_0.x_0_62 + WQtempx * t_0_6_1.x_0_62 + CDtemp * 3 * ( t_0_5_0.x_0_36 -  ABcom * t_0_5_1.x_0_36 ) ;
    x_0_91 = Qtempx * t_0_6_0.x_0_57 + WQtempx * t_0_6_1.x_0_57 + CDtemp * ( t_0_5_0.x_0_48 -  ABcom * t_0_5_1.x_0_48 ) ;
    x_0_92 = Qtempx * t_0_6_0.x_0_64 + WQtempx * t_0_6_1.x_0_64 + CDtemp * 3 * ( t_0_5_0.x_0_37 -  ABcom * t_0_5_1.x_0_37 ) ;
    x_0_93 = Qtempx * t_0_6_0.x_0_78 + WQtempx * t_0_6_1.x_0_78 ;
    x_0_94 = Qtempx * t_0_6_0.x_0_61 + WQtempx * t_0_6_1.x_0_61 + CDtemp * 2 * ( t_0_5_0.x_0_40 -  ABcom * t_0_5_1.x_0_40 ) ;
    x_0_95 = Qtempx * t_0_6_0.x_0_63 + WQtempx * t_0_6_1.x_0_63 + CDtemp * 2 * ( t_0_5_0.x_0_39 -  ABcom * t_0_5_1.x_0_39 ) ;
    x_0_96 = Qtempx * t_0_6_0.x_0_65 + WQtempx * t_0_6_1.x_0_65 + CDtemp * 2 * ( t_0_5_0.x_0_35 -  ABcom * t_0_5_1.x_0_35 ) ;
    x_0_97 = Qtempx * t_0_6_0.x_0_60 + WQtempx * t_0_6_1.x_0_60 + CDtemp * ( t_0_5_0.x_0_42 -  ABcom * t_0_5_1.x_0_42 ) ;
    x_0_98 = Qtempx * t_0_6_0.x_0_59 + WQtempx * t_0_6_1.x_0_59 + CDtemp * ( t_0_5_0.x_0_41 -  ABcom * t_0_5_1.x_0_41 ) ;
    x_0_99 = Qtempy * t_0_6_0.x_0_83 + WQtempy * t_0_6_1.x_0_83 ;
    x_0_100 = Qtempy * t_0_6_0.x_0_67 + WQtempy * t_0_6_1.x_0_67 + CDtemp * 5 * ( t_0_5_0.x_0_48 -  ABcom * t_0_5_1.x_0_48 ) ;
    x_0_101 = Qtempx * t_0_6_0.x_0_83 + WQtempx * t_0_6_1.x_0_83 ;
    x_0_102 = Qtempx * t_0_6_0.x_0_69 + WQtempx * t_0_6_1.x_0_69 + CDtemp * 5 * ( t_0_5_0.x_0_50 -  ABcom * t_0_5_1.x_0_50 ) ;
    x_0_103 = Qtempx * t_0_6_0.x_0_82 + WQtempx * t_0_6_1.x_0_82 ;
    x_0_104 = Qtempx * t_0_6_0.x_0_71 + WQtempx * t_0_6_1.x_0_71 + CDtemp * 5 * ( t_0_5_0.x_0_52 -  ABcom * t_0_5_1.x_0_52 ) ;
    x_0_105 = Qtempy * t_0_6_0.x_0_66 + WQtempy * t_0_6_1.x_0_66 + CDtemp * ( t_0_5_0.x_0_55 -  ABcom * t_0_5_1.x_0_55 ) ;
    x_0_106 = Qtempy * t_0_6_0.x_0_73 + WQtempy * t_0_6_1.x_0_73 + CDtemp * 4 * ( t_0_5_0.x_0_42 -  ABcom * t_0_5_1.x_0_42 ) ;
    x_0_107 = Qtempx * t_0_6_0.x_0_68 + WQtempx * t_0_6_1.x_0_68 + CDtemp * ( t_0_5_0.x_0_55 -  ABcom * t_0_5_1.x_0_55 ) ;
    x_0_108 = Qtempx * t_0_6_0.x_0_75 + WQtempx * t_0_6_1.x_0_75 + CDtemp * 4 * ( t_0_5_0.x_0_44 -  ABcom * t_0_5_1.x_0_44 ) ;
    x_0_109 = Qtempx * t_0_6_0.x_0_70 + WQtempx * t_0_6_1.x_0_70 + CDtemp * ( t_0_5_0.x_0_54 -  ABcom * t_0_5_1.x_0_54 ) ;
    x_0_110 = Qtempx * t_0_6_0.x_0_77 + WQtempx * t_0_6_1.x_0_77 + CDtemp * 4 * ( t_0_5_0.x_0_46 -  ABcom * t_0_5_1.x_0_46 ) ;
    x_0_111 = Qtempy * t_0_6_0.x_0_72 + WQtempy * t_0_6_1.x_0_72 + CDtemp * 2 * ( t_0_5_0.x_0_47 -  ABcom * t_0_5_1.x_0_47 ) ;
    x_0_112 = Qtempy * t_0_6_0.x_0_78 + WQtempy * t_0_6_1.x_0_78 + CDtemp * 3 * ( t_0_5_0.x_0_41 -  ABcom * t_0_5_1.x_0_41 ) ;
    x_0_113 = Qtempx * t_0_6_0.x_0_74 + WQtempx * t_0_6_1.x_0_74 + CDtemp * 2 * ( t_0_5_0.x_0_49 -  ABcom * t_0_5_1.x_0_49 ) ;
    x_0_114 = Qtempx * t_0_6_0.x_0_79 + WQtempx * t_0_6_1.x_0_79 + CDtemp * 3 * ( t_0_5_0.x_0_43 -  ABcom * t_0_5_1.x_0_43 ) ;
    x_0_115 = Qtempx * t_0_6_0.x_0_76 + WQtempx * t_0_6_1.x_0_76 + CDtemp * 2 * ( t_0_5_0.x_0_51 -  ABcom * t_0_5_1.x_0_51 ) ;
    x_0_116 = Qtempx * t_0_6_0.x_0_80 + WQtempx * t_0_6_1.x_0_80 + CDtemp * 3 * ( t_0_5_0.x_0_45 -  ABcom * t_0_5_1.x_0_45 ) ;
    x_0_117 = Qtempx * t_0_6_0.x_0_81 + WQtempx * t_0_6_1.x_0_81 + CDtemp * 6 * ( t_0_5_0.x_0_53 -  ABcom * t_0_5_1.x_0_53 ) ;
    x_0_118 = Qtempy * t_0_6_0.x_0_82 + WQtempy * t_0_6_1.x_0_82 + CDtemp * 6 * ( t_0_5_0.x_0_54 -  ABcom * t_0_5_1.x_0_54 ) ;
    x_0_119 = Qtempz * t_0_6_0.x_0_83 + WQtempz * t_0_6_1.x_0_83 + CDtemp * 6 * ( t_0_5_0.x_0_55 -  ABcom * t_0_5_1.x_0_55 ) ;
}
