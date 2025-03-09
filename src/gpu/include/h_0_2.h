#include "../gpu_common.h"
#include "./h_all_files.h"
// Class for L =            0  B =            2
__device__ __inline__ f_0_2_t :: f_0_2_t ( f_0_1_t t_0_1_0, f_0_1_t t_0_1_1, QUICKDouble t_0_0_0, QUICKDouble t_0_0_1, QUICKDouble CDtemp, QUICKDouble ABcom, QUICKDouble Qtempx, QUICKDouble Qtempy, QUICKDouble Qtempz,  QUICKDouble WQtempx, QUICKDouble WQtempy, QUICKDouble WQtempz) {
    x_0_4 = Qtempx * t_0_1_0.x_0_2 + WQtempx * t_0_1_1.x_0_2 ;
    x_0_5 = Qtempy * t_0_1_0.x_0_3 + WQtempy * t_0_1_1.x_0_3 ;
    x_0_6 = Qtempx * t_0_1_0.x_0_3 + WQtempx * t_0_1_1.x_0_3 ;
    x_0_7 = Qtempx * t_0_1_0.x_0_1 + WQtempx * t_0_1_1.x_0_1 + CDtemp * ( t_0_0_0 -  ABcom * t_0_0_1 ) ;
    x_0_8 = Qtempy * t_0_1_0.x_0_2 + WQtempy * t_0_1_1.x_0_2 + CDtemp * ( t_0_0_0 -  ABcom * t_0_0_1 ) ;
    x_0_9 = Qtempz * t_0_1_0.x_0_3 + WQtempz * t_0_1_1.x_0_3 + CDtemp * ( t_0_0_0 -  ABcom * t_0_0_1 ) ;
}
