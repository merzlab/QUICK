#include "../gpu_common.h"
#include "./h_all_files.h"
// Class for L =            0  B =            1
__device__ __inline__  f_0_1_t :: f_0_1_t ( QUICKDouble t_0_0_0, QUICKDouble t_0_0_1, QUICKDouble Qtempx, QUICKDouble Qtempy, QUICKDouble Qtempz,  QUICKDouble WQtempx, QUICKDouble WQtempy, QUICKDouble WQtempz) {
    x_0_1 = Qtempx * t_0_0_0 + WQtempx * t_0_0_1 ;
    x_0_2 = Qtempy * t_0_0_0 + WQtempy * t_0_0_1 ;
    x_0_3 = Qtempz * t_0_0_0 + WQtempz * t_0_0_1 ;
}
