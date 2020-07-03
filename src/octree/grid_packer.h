#include "octree.h"


struct bas_func{

        /*Basis function number*/
        int bas_id;

        /*A list for primitive functions*/
        vector<int> prim_list;
};

/* A struct to hold basis function lists*/
struct bflist{

        int node_id; /* For which node does this list belong to*/
        vector<bas_func> bfs;
};

/*Fortran interface to prune & pack grid points*/
extern "C" {

void gpack_initialize_();

void gpack_finalize_();

void pack_grid_pts();

/*Fortran interface to prune & pack grid points*/
void gpack_pack_pts_(double *grid_ptx, double *grid_pty, double *grid_ptz, int *grid_atm, double *grid_sswt, double *grid_weight, int *arr_size, int *natoms, int *nbasis, int *maxcontract, double *DMCutoff, double *sigrad2, int *ncontract, double *aexp, double *dcoeff, int *ncenter, int *itype, double *xyz, int *ngpts, int *ntgpts, int *nbins, int *nbtotbf, int *nbtotpf, double *toct, double *tprscrn);

/*interface to save packed info in fortran data structures*/
#ifdef CUDA
void get_gpu_grid_info_(double *gridx, double *gridy, double *gridz, double *ssw, double *weight, int *atm, int *dweight, int *basf, int *primf, int *basf_counter, int *primf_counter);

#else
void get_cpu_grid_info_(double *gridx, double *gridy, double *gridz, double *ssw, double *weight, int *atm, int *basf, int *primf, int *basf_counter, int *primf_counter, int *bin_counter);
#endif

}

// prunes grid based on ssw
void get_ssw_pruned_grid();

#ifdef CUDA
int gpu_get_pfbased_basis_function_lists_new_imp(vector<node> *octree, vector<node> *signodes, vector<bflist> *bflst);
#endif

void cpu_get_primf_contraf_lists_method_new_imp(double gridx, double gridy, double gridz, unsigned char *gpweight, unsigned int *cfweight, unsigned int *pfweight, unsigned int bin_id, unsigned int gid);

void cpu_get_pfbased_basis_function_lists_new_imp(vector<node> *octree);

//MPI setup for the grid operations
#ifdef MPIV

  int mpisize;
  int mpirank;

//Prescreening is parallelized by sharing bins among slaves, this array keeps track of that.
  unsigned int *mpi_binlst;

void setup_gpack_mpi_1();

void setup_gpack_mpi_2(unsigned int nbins, double *gridx, double *gridy, double *gridz, unsigned char *gpweight, unsigned char *tmp_gpweight, unsigned int *cfweight, unsigned int *tmp_cfweight, unsigned int *pfweight, unsigned int *tmp_pfweight, double *sswt, double *weight, int *iatm, unsigned int *bs_tracker);

void get_slave_primf_contraf_lists(unsigned int nbins, unsigned char *gpweight, unsigned char *tmp_gpweight, unsigned int *cfweight, unsigned int *tmp_cfweight, unsigned int *pfweight, unsigned int *tmp_pfweight, unsigned int *bs_tracker);

void delete_gpack_mpi();

#endif

