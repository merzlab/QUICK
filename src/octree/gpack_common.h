
#include "../config.h"

//Constants for packing grid points

static const int MAX_POINTS_PER_CLUSTER = 256;
static const int OCTREE_DEPTH = 64;

/*struct grd_pck_strct{

        double *gridx; //gridx_in, gridy_in, gridz_in: xyz coordinates of grid points
        double *gridy;
        double *gridz;
        double *sswt; //sswt_in, ss_weight_in: ss weights of grid points
        double *ss_weight;	
	int *grid_atm; //a set of atom indices, required to calculate the derivatives of weights

        int arr_size; //size of the grid arrays
        int nbasis; //total number of basis functions
        int maxcontract; //maximum number of contractions
        double DMCutoff; //Density matrix cut off
        double *sigrad2; //square of the radius of sigificance
        int *ncontract; //number of contraction functions
        double *aexp; //alpha values of the gaussian primivite function exponents
	double *dcoeff; //Contraction coefficients
        int *ncenter; //centers of the basis functions
	int *itype;
        double *xyz; //xyz coordinates of atomic positions

        double *gridxb; //gridxb_out, gridyb_out, gridzb_out: binned grid x, y and z grid points 
        double *gridyb;
        double *gridzb;
        double *gridb_sswt; //sswt_out, ss_weight_out: binned ss weights
        double *gridb_weight;
	int *gridb_atm; 
        int *dweight; //an array indicating if a binned grid point is true or a dummy grid point
        int *basf; //array of basis functions belonging to each bin
        int *primf; //array of primitive functions beloning to binned basis functions
        int *basf_counter; //a counter to keep track of which basis functions belong to which bin
        int *primf_counter; //a counter to keep track of which primitive functions belong to which basis function
	int *bin_counter; //a counter to keep track of bins with different number of points in cpu implementation
        int gridb_count; //length of binned grid arrays
        int nbins; //number of bins 
        int nbtotbf; //total number of basis functions
        int nbtotpf; //total number of primitive functions

	double time_octree; //Time for running octree algorithm
        double time_bfpf_prescreen; //Time for prescreening basis and primitive functions
};
*/

void pack_grid_pts();
//void get_pruned_grid_ssw(grd_pck_strct *gps_in, grd_pck_strct *gps_out);
//extern "C" void gpu_get_octree_info(double *gridx, double *gridy, double *gridz, unsigned char *gpweight, unsigned char *cfweight, unsigned char *pfweight, int count);
extern "C" void gpu_get_octree_info_new_imp(double *gridx, double *gridy, double *gridz, double *sigrad2, unsigned char *gpweight, unsigned int *cfweight, unsigned int *pfweight, int count);
