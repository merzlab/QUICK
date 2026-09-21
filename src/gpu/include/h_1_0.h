#include "../gpu_common.h"
#include "./h_all_files.h"
// Class for L =            1  B =            0
__device__ __inline__  f_1_0_t :: f_1_0_t ( QUICKDouble t_0_0_0, QUICKDouble t_0_0_1, QUICKDouble Ptempx, QUICKDouble Ptempy, QUICKDouble Ptempz,  QUICKDouble WPtempx, QUICKDouble WPtempy, QUICKDouble WPtempz) {
    x_1_0 = Ptempx * t_0_0_0 + WPtempx * t_0_0_1 ;
    x_2_0 = Ptempy * t_0_0_0 + WPtempy * t_0_0_1 ;
    x_3_0 = Ptempz * t_0_0_0 + WPtempz * t_0_0_1 ;
}

