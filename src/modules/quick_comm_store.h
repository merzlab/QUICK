// quick_comm_store.h
#pragma once
#if defined(MPIV) || defined(MPIV_GPU)
#include <mpi.h>
#ifdef __cplusplus
extern "C" {
#endif
void     quick_set_comm_c(int *comm_f);
MPI_Comm quick_get_comm_c(void);
#ifdef __cplusplus
}
#endif
#endif
