#include "octree.h"

grd_pck_strct gpst;

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
/*Fortran interface to prune & pack grid points*/
void pack_grid_pts_f90_(double *grid_ptx, double *grid_pty, double *grid_ptz, int *grid_atm, double *grid_sswt, double *grid_weight, int *arr_size, int *nbasis, int *maxcontract, double *DMCutoff, double *sigrad2, int *ncontract, double *aexp, double *dcoeff, int *ncenter, int *itype, double *xyz, int *ngpts, int *nbins, int *nbtotbf, int *nbtotpf, double *toct, double* tprscrn);

/*interface to save packed info in fortran data structures*/
void save_dft_grid_info_(double *gridx, double *gridy, double *gridz, double *ssw, double *weight, int *atm, int *dweight, int *basf, int *primf, int *basf_counter, int *primf_counter, int *bin_counter);
}

#ifdef CUDA
int gpu_get_pfbased_basis_function_lists_new_imp(vector<node> *octree, grd_pck_strct *gps, vector<node> *signodes, vector<bflist> *bflst);
#endif

void cpu_get_primf_contraf_lists_method_new_imp(double gridx, double gridy, double gridz, grd_pck_strct *gps, unsigned char *gpweight, unsigned int *cfweight, unsigned int *pfweight, unsigned int bin_id, unsigned int gid);

void cpu_get_pfbased_basis_function_lists_new_imp(vector<node> *octree, grd_pck_strct *gps);

//MPI setup for the grid operations
#ifdef MPIV

struct gpack_mpi{
	int mpisize;
	int mpirank;

//Prescreening is parallelized by sharing bins among slaves, this array keeps track of that.
	unsigned int *mpi_binlst;
};

gpack_mpi gmpi;

void setup_gpack_mpi_1(grd_pck_strct *gps);

void setup_gpack_mpi_2(unsigned int nbins, double *gridx, double *gridy, double *gridz, grd_pck_strct *gps, unsigned char *gpweight, unsigned char *tmp_gpweight, unsigned int *cfweight, unsigned int *tmp_cfweight, unsigned int *pfweight, unsigned int *tmp_pfweight, double *sswt, double *weight, int *iatm, unsigned int *bs_tracker);

void get_slave_primf_contraf_lists(unsigned int nbins, grd_pck_strct *gps, unsigned char *gpweight, unsigned char *tmp_gpweight, unsigned int *cfweight, unsigned int *tmp_cfweight, unsigned int *pfweight, unsigned int *tmp_pfweight, unsigned int *bs_tracker);

void delete_gpack_mpi();

#endif

