#include "../gpu_common.h"
#include "./h_all_files.h"
// Class for L =            2  B =            2
__device__ __inline__ f_2_2_t :: f_2_2_t ( f_1_2_t t_1_2_0, f_1_2_t t_1_2_1, f_0_2_t t_0_2_0, f_0_2_t t_0_2_1, QUICKDouble ABtemp, QUICKDouble CDcom, f_1_1_t t_1_1_1, QUICKDouble ABCDtemp, QUICKDouble Ptempx, QUICKDouble Ptempy, QUICKDouble Ptempz,  QUICKDouble WPtempx, QUICKDouble WPtempy, QUICKDouble WPtempz) {
    x_4_4 = Ptempx * t_1_2_0.x_2_4 + WPtempx * t_1_2_1.x_2_4 + ABCDtemp * t_1_1_1.x_2_2 ;
    x_4_5 = Ptempx * t_1_2_0.x_2_5 + WPtempx * t_1_2_1.x_2_5 ;
    x_4_6 = Ptempx * t_1_2_0.x_2_6 + WPtempx * t_1_2_1.x_2_6 + ABCDtemp * t_1_1_1.x_2_3 ;
    x_4_7 = Ptempx * t_1_2_0.x_2_7 + WPtempx * t_1_2_1.x_2_7 + 2 * ABCDtemp * t_1_1_1.x_2_1 ;
    x_4_8 = Ptempx * t_1_2_0.x_2_8 + WPtempx * t_1_2_1.x_2_8 ;
    x_4_9 = Ptempx * t_1_2_0.x_2_9 + WPtempx * t_1_2_1.x_2_9 ;
    x_5_4 = Ptempy * t_1_2_0.x_3_4 + WPtempy * t_1_2_1.x_3_4 + ABCDtemp * t_1_1_1.x_3_1 ;
    x_5_5 = Ptempy * t_1_2_0.x_3_5 + WPtempy * t_1_2_1.x_3_5 + ABCDtemp * t_1_1_1.x_3_3 ;
    x_5_6 = Ptempy * t_1_2_0.x_3_6 + WPtempy * t_1_2_1.x_3_6 ;
    x_5_7 = Ptempy * t_1_2_0.x_3_7 + WPtempy * t_1_2_1.x_3_7 ;
    x_5_8 = Ptempy * t_1_2_0.x_3_8 + WPtempy * t_1_2_1.x_3_8 + 2 * ABCDtemp * t_1_1_1.x_3_2 ;
    x_5_9 = Ptempy * t_1_2_0.x_3_9 + WPtempy * t_1_2_1.x_3_9 ;
    x_6_4 = Ptempx * t_1_2_0.x_3_4 + WPtempx * t_1_2_1.x_3_4 + ABCDtemp * t_1_1_1.x_3_2 ;
    x_6_5 = Ptempx * t_1_2_0.x_3_5 + WPtempx * t_1_2_1.x_3_5 ;
    x_6_6 = Ptempx * t_1_2_0.x_3_6 + WPtempx * t_1_2_1.x_3_6 + ABCDtemp * t_1_1_1.x_3_3 ;
    x_6_7 = Ptempx * t_1_2_0.x_3_7 + WPtempx * t_1_2_1.x_3_7 + 2 * ABCDtemp * t_1_1_1.x_3_1 ;
    x_6_8 = Ptempx * t_1_2_0.x_3_8 + WPtempx * t_1_2_1.x_3_8 ;
    x_6_9 = Ptempx * t_1_2_0.x_3_9 + WPtempx * t_1_2_1.x_3_9 ;
    x_7_4 = Ptempx * t_1_2_0.x_1_4 + WPtempx * t_1_2_1.x_1_4 + ABtemp * ( t_0_2_0.x_0_4 -  CDcom * t_0_2_1.x_0_4 ) + ABCDtemp * t_1_1_1.x_1_2 ;
    x_7_5 = Ptempx * t_1_2_0.x_1_5 + WPtempx * t_1_2_1.x_1_5 + ABtemp * ( t_0_2_0.x_0_5 -  CDcom * t_0_2_1.x_0_5 ) ;
    x_7_6 = Ptempx * t_1_2_0.x_1_6 + WPtempx * t_1_2_1.x_1_6 + ABtemp * ( t_0_2_0.x_0_6 -  CDcom * t_0_2_1.x_0_6 ) + ABCDtemp * t_1_1_1.x_1_3 ;
    x_7_7 = Ptempx * t_1_2_0.x_1_7 + WPtempx * t_1_2_1.x_1_7 + ABtemp * ( t_0_2_0.x_0_7 -  CDcom * t_0_2_1.x_0_7 ) + 2 * ABCDtemp * t_1_1_1.x_1_1 ;
    x_7_8 = Ptempx * t_1_2_0.x_1_8 + WPtempx * t_1_2_1.x_1_8 + ABtemp * ( t_0_2_0.x_0_8 -  CDcom * t_0_2_1.x_0_8 ) ;
    x_7_9 = Ptempx * t_1_2_0.x_1_9 + WPtempx * t_1_2_1.x_1_9 + ABtemp * ( t_0_2_0.x_0_9 -  CDcom * t_0_2_1.x_0_9 ) ;
    x_8_4 = Ptempy * t_1_2_0.x_2_4 + WPtempy * t_1_2_1.x_2_4 + ABtemp * ( t_0_2_0.x_0_4 -  CDcom * t_0_2_1.x_0_4 ) + ABCDtemp * t_1_1_1.x_2_1 ;
    x_8_5 = Ptempy * t_1_2_0.x_2_5 + WPtempy * t_1_2_1.x_2_5 + ABtemp * ( t_0_2_0.x_0_5 -  CDcom * t_0_2_1.x_0_5 ) + ABCDtemp * t_1_1_1.x_2_3 ;
    x_8_6 = Ptempy * t_1_2_0.x_2_6 + WPtempy * t_1_2_1.x_2_6 + ABtemp * ( t_0_2_0.x_0_6 -  CDcom * t_0_2_1.x_0_6 ) ;
    x_8_7 = Ptempy * t_1_2_0.x_2_7 + WPtempy * t_1_2_1.x_2_7 + ABtemp * ( t_0_2_0.x_0_7 -  CDcom * t_0_2_1.x_0_7 ) ;
    x_8_8 = Ptempy * t_1_2_0.x_2_8 + WPtempy * t_1_2_1.x_2_8 + ABtemp * ( t_0_2_0.x_0_8 -  CDcom * t_0_2_1.x_0_8 ) + 2 * ABCDtemp * t_1_1_1.x_2_2 ;
    x_8_9 = Ptempy * t_1_2_0.x_2_9 + WPtempy * t_1_2_1.x_2_9 + ABtemp * ( t_0_2_0.x_0_9 -  CDcom * t_0_2_1.x_0_9 ) ;
    x_9_4 = Ptempz * t_1_2_0.x_3_4 + WPtempz * t_1_2_1.x_3_4 + ABtemp * ( t_0_2_0.x_0_4 -  CDcom * t_0_2_1.x_0_4 ) ;
    x_9_5 = Ptempz * t_1_2_0.x_3_5 + WPtempz * t_1_2_1.x_3_5 + ABtemp * ( t_0_2_0.x_0_5 -  CDcom * t_0_2_1.x_0_5 ) + ABCDtemp * t_1_1_1.x_3_2 ;
    x_9_6 = Ptempz * t_1_2_0.x_3_6 + WPtempz * t_1_2_1.x_3_6 + ABtemp * ( t_0_2_0.x_0_6 -  CDcom * t_0_2_1.x_0_6 ) + ABCDtemp * t_1_1_1.x_3_1 ;
    x_9_7 = Ptempz * t_1_2_0.x_3_7 + WPtempz * t_1_2_1.x_3_7 + ABtemp * ( t_0_2_0.x_0_7 -  CDcom * t_0_2_1.x_0_7 ) ;
    x_9_8 = Ptempz * t_1_2_0.x_3_8 + WPtempz * t_1_2_1.x_3_8 + ABtemp * ( t_0_2_0.x_0_8 -  CDcom * t_0_2_1.x_0_8 ) ;
    x_9_9 = Ptempz * t_1_2_0.x_3_9 + WPtempz * t_1_2_1.x_3_9 + ABtemp * ( t_0_2_0.x_0_9 -  CDcom * t_0_2_1.x_0_9 ) + 2 * ABCDtemp * t_1_1_1.x_3_3 ;
}
