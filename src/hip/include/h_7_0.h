#include "../gpu_common.h"
#include "./h_all_files.h"
// Class for B =            7  L =            0
__device__ __inline__ f_7_0_t :: f_7_0_t ( f_6_0_t t_6_0_0, f_6_0_t t_6_0_1, f_5_0_t t_5_0_0, f_5_0_t t_5_0_1, QUICKDouble ABtemp, QUICKDouble CDcom, QUICKDouble Ptempx, QUICKDouble Ptempy, QUICKDouble Ptempz,  QUICKDouble WPtempx, QUICKDouble WPtempy, QUICKDouble WPtempz) {
    x_84_0 = Ptempx * t_6_0_0.x_56_0 + WPtempx * t_6_0_1.x_56_0 + ABtemp * 4 * ( t_5_0_0.x_38_0 -  CDcom * t_5_0_1.x_38_0 ) ;
    x_85_0 = Ptempx * t_6_0_0.x_67_0 + WPtempx * t_6_0_1.x_67_0 ;
    x_86_0 = Ptempx * t_6_0_0.x_66_0 + WPtempx * t_6_0_1.x_66_0 ;
    x_87_0 = Ptempx * t_6_0_0.x_72_0 + WPtempx * t_6_0_1.x_72_0 ;
    x_88_0 = Ptempx * t_6_0_0.x_73_0 + WPtempx * t_6_0_1.x_73_0 ;
    x_89_0 = Ptempx * t_6_0_0.x_58_0 + WPtempx * t_6_0_1.x_58_0 + ABtemp * ( t_5_0_0.x_47_0 -  CDcom * t_5_0_1.x_47_0 ) ;
    x_90_0 = Ptempx * t_6_0_0.x_62_0 + WPtempx * t_6_0_1.x_62_0 + ABtemp * 3 * ( t_5_0_0.x_36_0 -  CDcom * t_5_0_1.x_36_0 ) ;
    x_91_0 = Ptempx * t_6_0_0.x_57_0 + WPtempx * t_6_0_1.x_57_0 + ABtemp * ( t_5_0_0.x_48_0 -  CDcom * t_5_0_1.x_48_0 ) ;
    x_92_0 = Ptempx * t_6_0_0.x_64_0 + WPtempx * t_6_0_1.x_64_0 + ABtemp * 3 * ( t_5_0_0.x_37_0 -  CDcom * t_5_0_1.x_37_0 ) ;
    x_93_0 = Ptempx * t_6_0_0.x_78_0 + WPtempx * t_6_0_1.x_78_0 ;
    x_94_0 = Ptempx * t_6_0_0.x_61_0 + WPtempx * t_6_0_1.x_61_0 + ABtemp * 2 * ( t_5_0_0.x_40_0 -  CDcom * t_5_0_1.x_40_0 ) ;
    x_95_0 = Ptempx * t_6_0_0.x_63_0 + WPtempx * t_6_0_1.x_63_0 + ABtemp * 2 * ( t_5_0_0.x_39_0 -  CDcom * t_5_0_1.x_39_0 ) ;
    x_96_0 = Ptempx * t_6_0_0.x_65_0 + WPtempx * t_6_0_1.x_65_0 + ABtemp * 2 * ( t_5_0_0.x_35_0 -  CDcom * t_5_0_1.x_35_0 ) ;
    x_97_0 = Ptempx * t_6_0_0.x_60_0 + WPtempx * t_6_0_1.x_60_0 + ABtemp * ( t_5_0_0.x_42_0 -  CDcom * t_5_0_1.x_42_0 ) ;
    x_98_0 = Ptempx * t_6_0_0.x_59_0 + WPtempx * t_6_0_1.x_59_0 + ABtemp * ( t_5_0_0.x_41_0 -  CDcom * t_5_0_1.x_41_0 ) ;
    x_99_0 = Ptempy * t_6_0_0.x_83_0 + WPtempy * t_6_0_1.x_83_0 ;
    x_100_0 = Ptempy * t_6_0_0.x_67_0 + WPtempy * t_6_0_1.x_67_0 + ABtemp * 5 * ( t_5_0_0.x_48_0 -  CDcom * t_5_0_1.x_48_0 ) ;
    x_101_0 = Ptempx * t_6_0_0.x_83_0 + WPtempx * t_6_0_1.x_83_0 ;
    x_102_0 = Ptempx * t_6_0_0.x_69_0 + WPtempx * t_6_0_1.x_69_0 + ABtemp * 5 * ( t_5_0_0.x_50_0 -  CDcom * t_5_0_1.x_50_0 ) ;
    x_103_0 = Ptempx * t_6_0_0.x_82_0 + WPtempx * t_6_0_1.x_82_0 ;
    x_104_0 = Ptempx * t_6_0_0.x_71_0 + WPtempx * t_6_0_1.x_71_0 + ABtemp * 5 * ( t_5_0_0.x_52_0 -  CDcom * t_5_0_1.x_52_0 ) ;
    x_105_0 = Ptempy * t_6_0_0.x_66_0 + WPtempy * t_6_0_1.x_66_0 + ABtemp * ( t_5_0_0.x_55_0 -  CDcom * t_5_0_1.x_55_0 ) ;
    x_106_0 = Ptempy * t_6_0_0.x_73_0 + WPtempy * t_6_0_1.x_73_0 + ABtemp * 4 * ( t_5_0_0.x_42_0 -  CDcom * t_5_0_1.x_42_0 ) ;
    x_107_0 = Ptempx * t_6_0_0.x_68_0 + WPtempx * t_6_0_1.x_68_0 + ABtemp * ( t_5_0_0.x_55_0 -  CDcom * t_5_0_1.x_55_0 ) ;
    x_108_0 = Ptempx * t_6_0_0.x_75_0 + WPtempx * t_6_0_1.x_75_0 + ABtemp * 4 * ( t_5_0_0.x_44_0 -  CDcom * t_5_0_1.x_44_0 ) ;
    x_109_0 = Ptempx * t_6_0_0.x_70_0 + WPtempx * t_6_0_1.x_70_0 + ABtemp * ( t_5_0_0.x_54_0 -  CDcom * t_5_0_1.x_54_0 ) ;
    x_110_0 = Ptempx * t_6_0_0.x_77_0 + WPtempx * t_6_0_1.x_77_0 + ABtemp * 4 * ( t_5_0_0.x_46_0 -  CDcom * t_5_0_1.x_46_0 ) ;
    x_111_0 = Ptempy * t_6_0_0.x_72_0 + WPtempy * t_6_0_1.x_72_0 + ABtemp * 2 * ( t_5_0_0.x_47_0 -  CDcom * t_5_0_1.x_47_0 ) ;
    x_112_0 = Ptempy * t_6_0_0.x_78_0 + WPtempy * t_6_0_1.x_78_0 + ABtemp * 3 * ( t_5_0_0.x_41_0 -  CDcom * t_5_0_1.x_41_0 ) ;
    x_113_0 = Ptempx * t_6_0_0.x_74_0 + WPtempx * t_6_0_1.x_74_0 + ABtemp * 2 * ( t_5_0_0.x_49_0 -  CDcom * t_5_0_1.x_49_0 ) ;
    x_114_0 = Ptempx * t_6_0_0.x_79_0 + WPtempx * t_6_0_1.x_79_0 + ABtemp * 3 * ( t_5_0_0.x_43_0 -  CDcom * t_5_0_1.x_43_0 ) ;
    x_115_0 = Ptempx * t_6_0_0.x_76_0 + WPtempx * t_6_0_1.x_76_0 + ABtemp * 2 * ( t_5_0_0.x_51_0 -  CDcom * t_5_0_1.x_51_0 ) ;
    x_116_0 = Ptempx * t_6_0_0.x_80_0 + WPtempx * t_6_0_1.x_80_0 + ABtemp * 3 * ( t_5_0_0.x_45_0 -  CDcom * t_5_0_1.x_45_0 ) ;
    x_117_0 = Ptempx * t_6_0_0.x_81_0 + WPtempx * t_6_0_1.x_81_0 + ABtemp * 6 * ( t_5_0_0.x_53_0 -  CDcom * t_5_0_1.x_53_0 ) ;
    x_118_0 = Ptempy * t_6_0_0.x_82_0 + WPtempy * t_6_0_1.x_82_0 + ABtemp * 6 * ( t_5_0_0.x_54_0 -  CDcom * t_5_0_1.x_54_0 ) ;
    x_119_0 = Ptempz * t_6_0_0.x_83_0 + WPtempz * t_6_0_1.x_83_0 + ABtemp * 6 * ( t_5_0_0.x_55_0 -  CDcom * t_5_0_1.x_55_0 ) ;
}
