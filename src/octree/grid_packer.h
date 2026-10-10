#include "octree.h"
#include "gpack_common.h"

// Enable to generate coordinate files for the visualization
//#define WRITE_TCL_XYZ

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

#if defined(MPIV) && !defined(MPIV_GPU)
void gpack_initialize_(int *);
#else
void gpack_initialize_();
#endif

void gpack_finalize_();

void pack_grid_pts();

/*Fortran interface to prune & pack grid points*/
void gpack_pack_pts_(double *grid_ptx, double *grid_pty, double *grid_ptz, int *grid_atm,
		double *grid_sswt, double *grid_weight, int *arr_size, int *natoms,
		int *nbasis, int *maxcontract, double *DMCutoff, double *XCCutoff,
		double *sigrad2, int *ncontract, double *aexp, double *dcoeff,
		int *ncenter, int *itype, double *xyz, int *ngpts, int *nbins,
		int *nbtotbf, int *nbtotpf, double *toct, double *tprscrn);

/*interface to save packed info in fortran data structures*/
#if defined(GPU) || defined(MPIV_GPU)
void get_gpu_grid_info_(double *gridx, double *gridy, double *gridz, double *ssw,
		double *weight, int *atm, int *bin_locator, int *basf, int *primf,
		int *basf_counter, int *primf_counter, int *bin_counter);
#else
void get_cpu_grid_info_(double *gridx, double *gridy, double *gridz, double *ssw,
		double *weight, int *atm, int *basf, int *primf, int *basf_counter,
		int *primf_counter, int *bin_counter);
#endif

}

// prunes grid based on ssw
void get_ssw_pruned_grid();

#if defined(GPU) || defined(MPIV_GPU)
void gpu_get_pfbased_basis_function_lists_new_imp(vector<node> *octree, vector<node> *signodes, vector<bflist> *bflst);
#endif

void cpu_get_primf_contraf_lists_method_new_imp(double gridx, double gridy, double gridz, unsigned char *gpweight, unsigned int *cfweight, unsigned int *pfweight, unsigned int bin_id, unsigned int gid);

void cpu_get_pfbased_basis_function_lists_new_imp(vector<node> *octree);

//MPI setup for the grid operations
#if defined(MPIV) && !defined(MPIV_GPU)
static int mpi_size;
static int mpi_rank;
static MPI_Comm mpi_comm;
//Prescreening is parallelized by sharing bins among slaves, this array keeps track of that.
static unsigned int *mpi_binlst;


void setup_gpack_mpi_1();

void setup_gpack_mpi_2(unsigned int, double *, double *, double *,
        unsigned char *, unsigned char *, unsigned int *, unsigned int *, unsigned int *,
        unsigned int *, double *, double *, int *, unsigned int *);

void get_slave_primf_contraf_lists(unsigned int, unsigned char *, unsigned char *,
        unsigned int *, unsigned int *, unsigned int *, unsigned int *, unsigned int *);

void delete_gpack_mpi();
#endif
