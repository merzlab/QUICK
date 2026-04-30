#ifndef XC_REDISTRIBUTE
#define XC_REDISTRIBUTE
#if defined(MPIV_GPU)
#ifdef __cplusplus
extern "C" {
#endif

int getAdjustment(MPI_Comm, int, int, int);

void sswderRedistribute(MPI_Comm, int, int, int, int,
  double *, double *, double *, double *, double *, int *,
  double *, double *, double *, double *, double *, int *);

#ifdef __cplusplus
}
#endif
#endif
#endif
