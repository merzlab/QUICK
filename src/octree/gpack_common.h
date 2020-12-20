
//Constants for packing grid points
static const int MAX_POINTS_PER_CLUSTER = 256;
static const int OCTREE_DEPTH = 64;

//void pack_grid_pts();
extern "C" void gpu_get_octree_info(double *gridx, double *gridy, double *gridz, double *sigrad2, unsigned char *gpweight, unsigned int *cfweight, unsigned int *pfweight, int *bin_locator, int count, int nbins);
