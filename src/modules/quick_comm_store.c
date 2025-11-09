// quick_comm_store.c
#if defined(MPIV) ||  defined(MPIV_GPU)
#include <mpi.h>
static MPI_Comm quick_comm_c = MPI_COMM_WORLD;  // default

void quick_set_comm_c(int *comm_f) {
    quick_comm_c = MPI_Comm_f2c((MPI_Fint)(*comm_f));
}

int quick_get_comm_c(void) {
    int inited = 0;
    MPI_Initialized(&inited);
    if (!inited) return MPI_COMM_NULL;
    return quick_comm_c;
}
#endif
