/*
  !---------------------------------------------------------------------!
  ! Written by Madu Manathunga on 10/22/2019                            !
  !                                                                     ! 
  ! Copyright (C) 2020-2021 Merz lab                                    !
  ! Copyright (C) 2020-2021 Götz lab                                    !
  !                                                                     !
  ! This Source Code Form is subject to the terms of the Mozilla Public !
  ! License, v. 2.0. If a copy of the MPL was not distributed with this !
  ! file, You can obtain one at http://mozilla.org/MPL/2.0/.            !
  !_____________________________________________________________________!

  !---------------------------------------------------------------------!
  ! This source file contains functions required for numerical grid     !
  ! point pruning.                                                      !
  !                                                                     ! 
  !---------------------------------------------------------------------!
*/

#include "grid_packer.h"
#include "gpack_type.h"
#include <cmath>
#include <fstream>
#include <time.h>


// initialize data structure for grid partitioning algorithm
void gpack_initialize_(){

    gps = new gpack_type;
    gps->totalGPACKMemory = 0;

#if defined MPIV && !defined CUDA_MPIV
    MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
#endif

// setup debug file if necessary
#ifdef DEBUG

  #if defined MPIV && !defined CUDA_MPIV
    char fname[16];
    sprintf(fname, "debug.oct.%i", mpirank);
    gpackDebugFile = fopen(fname, "w+");
  #else
    gpackDebugFile = fopen("debug.oct", "w+");
  #endif

    gps->gpackDebugFile = gpackDebugFile;
#endif

}

// finalize data structure of grid partitioning algorithm
void gpack_finalize_(){

    delete gps->sigrad2;
    delete gps->ncontract;
    delete gps->aexp;
    delete gps->dcoeff;
    delete gps->xyz;
    delete gps->ncenter;
    delete gps->itype;

#if defined MPIV && !defined CUDA_MPIV
    if(mpirank == 0){
#endif
      delete gps->gridx;
      delete gps->gridy;
      delete gps->gridz;
      delete gps->sswt;
      delete gps->ss_weight;
      delete gps->grid_atm;

      delete gps->gridxb;
      delete gps->gridyb;
      delete gps->gridzb;
      delete gps->gridb_sswt;
      delete gps->gridb_weight;
      delete gps->gridb_atm;
      delete gps->basf;
      delete gps->primf;
      delete gps->basf_counter;
      delete gps->primf_counter;
#if defined MPIV && !defined CUDA_MPIV
    }
#endif

#if defined CUDA || defined CUDA_MPIV
    delete gps->dweight;    
#else
#if defined MPIV && !defined CUDA_MPIV
    if(mpirank == 0){
#endif
      delete gps->bin_counter;
#if defined MPIV && !defined CUDA_MPIV
    }
#endif
#endif

    delete gps;

#ifdef DEBUG
    fclose(gpackDebugFile);
#endif

}

#if defined CUDA || defined CUDA_MPIV
// loads packed grid information into f90 data structures
void get_gpu_grid_info_(double *gridx, double *gridy, double *gridz, double *ssw, double *weight, int *atm, int *dweight, int *basf, int *primf, int *basf_counter, int *primf_counter){

	gps->gridxb->Transfer(gridx);
	gps->gridyb->Transfer(gridy);
	gps->gridzb->Transfer(gridz);
        gps->gridb_sswt->Transfer(ssw);
        gps->gridb_weight->Transfer(weight);
        gps->gridb_atm->Transfer(atm);
	gps->dweight->Transfer(dweight);
	gps->basf->Transfer(basf);
	gps->primf->Transfer(primf);
	gps->basf_counter->Transfer(basf_counter);
        gps->primf_counter->Transfer(primf_counter);

}
#else
// loads packed grid information into f90 data structures
void get_cpu_grid_info_(double *gridx, double *gridy, double *gridz, double *ssw, double *weight, int *atm, int *basf, int *primf, int *basf_counter, int *primf_counter, int *bin_counter){

        gps->gridxb->Transfer(gridx);
        gps->gridyb->Transfer(gridy);
        gps->gridzb->Transfer(gridz);
        gps->gridb_sswt->Transfer(ssw);
        gps->gridb_weight->Transfer(weight);
        gps->gridb_atm->Transfer(atm);
        gps->basf->Transfer(basf);
        gps->primf->Transfer(primf);
        gps->basf_counter->Transfer(basf_counter);
        gps->primf_counter->Transfer(primf_counter);
	gps->bin_counter->Transfer(bin_counter);

}
#endif


/*Fortran accessible method to pack grid points*/
void gpack_pack_pts_(double *grid_ptx, double *grid_pty, double *grid_ptz, int *grid_atm, double *grid_sswt, double *grid_weight, int *arr_size, int *natoms, int *nbasis, int *maxcontract, double *DMCutoff, double *sigrad2, int *ncontract, double *aexp, double *dcoeff, int *ncenter, int *itype, double *xyz, int *ngpts, int *ntgpts, int *nbins, int *nbtotbf, int *nbtotpf, double *toct, double *tprscrn){
        
        gps->arr_size    = *arr_size;
        gps->natoms      = *natoms;
        gps->nbasis      = *nbasis;
        gps->maxcontract = *maxcontract;
        gps->DMCutoff    = *DMCutoff;

        gps->sigrad2     = new gpack_buffer_type<double>(sigrad2, gps->nbasis);
        gps->ncontract   = new gpack_buffer_type<int>(ncontract, gps->nbasis);
        gps->aexp        = new gpack_buffer_type<double>(aexp, gps->maxcontract, gps->nbasis);
        gps->dcoeff      = new gpack_buffer_type<double>(dcoeff, gps->maxcontract, gps->nbasis);
        gps->xyz         = new gpack_buffer_type<double>(xyz, 3, gps->natoms);
        gps->ncenter     = new gpack_buffer_type<int>(ncenter, gps->nbasis);
        gps->itype       = new gpack_buffer_type<int>(itype, 3, gps->nbasis);

#if defined MPIV && !defined CUDA_MPIV
    if(mpirank==0){
#endif
        gps->gridx       = new gpack_buffer_type<double>(grid_ptx, gps->arr_size);
        gps->gridy       = new gpack_buffer_type<double>(grid_pty, gps->arr_size);
        gps->gridz       = new gpack_buffer_type<double>(grid_ptz, gps->arr_size);
        gps->sswt        = new gpack_buffer_type<double>(grid_sswt, gps->arr_size);
        gps->ss_weight   = new gpack_buffer_type<double>(grid_weight, gps->arr_size);
        gps->grid_atm    = new gpack_buffer_type<int>(grid_atm, gps->arr_size);

        get_ssw_pruned_grid();

#if defined MPIV && !defined CUDA_MPIV
    }
#endif

        pack_grid_pts();

#if defined MPIV && !defined CUDA_MPIV
    if(mpirank==0){
#endif

	*ngpts   = gps->gridb_count;
        *ntgpts  = gps->ntgpts;
	*nbins   = gps->nbins;
	*nbtotbf = gps->nbtotbf;
	*nbtotpf = gps->nbtotpf;
	*toct    = gps->time_octree;
	*tprscrn = gps->time_bfpf_prescreen;

#if defined MPIV && !defined CUDA_MPIV
   }
#endif

}

//Prints the spatial grid used to generate the octree.
void write_vmd_grid(vector<node> octree, string filename){

	ofstream txtOut;

        //Convert the string file name into char array
        unsigned int fnlength = filename.length() + 1;
	char fname[fnlength];
        for(int i=0; i<fnlength;i++){
                fname[i] = filename[i];
        }

        txtOut.open(fname);

	//Set the cage color
	txtOut << "draw color black \n";

	//Count the total number of grid pts
	int tot_pts = 0;
	for(int i=0; i<octree.size();i++){

		node n = octree.at(i);
		
		/*
              p8------------------------- p7
		|\                      |\
 		| \                     | \
		|  \                    |  \
 		| p5\___________________|___\p6
              p4|___|___________________|p3 |
		 \  |			 \  |
		  \ |			  \ |
		   \|______________________\|
		    p1                      p2
   		*/

                txtOut << "draw line {" << n.xmin << " " << n.ymin << " " << n.zmin << "} {" << n.xmax << " " << n.ymin << " " << n.zmin << "} width 1 \n"; /*p1-p2*/
		txtOut << "draw line {" << n.xmin << " " << n.ymin << " " << n.zmin << "} {" << n.xmin << " " << n.ymax << " " << n.zmin << "} width 1 \n"; /*p1-p4*/
		txtOut << "draw line {" << n.xmin << " " << n.ymin << " " << n.zmin << "} {" << n.xmin << " " << n.ymin << " " << n.zmax << "} width 1 \n"; /*p1-p5*/
		txtOut << "draw line {" << n.xmax << " " << n.ymin << " " << n.zmin << "} {" << n.xmax << " " << n.ymax << " " << n.zmin << "} width 1 \n"; /*p2-p3*/
		txtOut << "draw line {" << n.xmax << " " << n.ymin << " " << n.zmin << "} {" << n.xmax << " " << n.ymin << " " << n.zmax << "} width 1 \n"; /*p2-p6*/
		txtOut << "draw line {" << n.xmax << " " << n.ymax << " " << n.zmin << "} {" << n.xmax << " " << n.ymax << " " << n.zmax << "} width 1 \n"; /*p3-p7*/
		txtOut << "draw line {" << n.xmax << " " << n.ymax << " " << n.zmin << "} {" << n.xmin << " " << n.ymax << " " << n.zmin << "} width 1 \n"; /*p3-p4*/
		txtOut << "draw line {" << n.xmin << " " << n.ymax << " " << n.zmin << "} {" << n.xmin << " " << n.ymax << " " << n.zmax << "} width 1 \n"; /*p4-p8*/
		txtOut << "draw line {" << n.xmin << " " << n.ymin << " " << n.zmax << "} {" << n.xmax << " " << n.ymin << " " << n.zmax << "} width 1 \n"; /*p5-p6*/
		txtOut << "draw line {" << n.xmax << " " << n.ymin << " " << n.zmax << "} {" << n.xmax << " " << n.ymax << " " << n.zmax << "} width 1 \n"; /*p6-p7*/ 
		txtOut << "draw line {" << n.xmax << " " << n.ymax << " " << n.zmax << "} {" << n.xmin << " " << n.ymax << " " << n.zmax << "} width 1 \n"; /*p7-p8*/
		txtOut << "draw line {" << n.xmin << " " << n.ymax << " " << n.zmax << "} {" << n.xmin << " " << n.ymin << " " << n.zmax << "} width 1 \n"; /*p8-p5*/

		tot_pts += n.ptlst.size();

	}

	txtOut.close();


}

//Write a .xyz file contanining grid points
void write_xyz(vector<node> *octree, vector<point> *ptlst, bool isptlst, string filename){

	ofstream txtOut;
	//Convert the string file name into char array
	unsigned int fnlength = filename.length() + 1;
	char fname[fnlength];
	for(int i=0; i<fnlength;i++){
		fname[i] = filename[i];
	}

	txtOut.open(fname);

	vector<point> allpts;

	if( isptlst == false ){
        	for(int i=0; i<octree->size();i++){
                	node n = octree->at(i);
			vector<point> tmp_ptlst = n.ptlst;

			for(int j=0; j<tmp_ptlst.size(); j++){
				point p = tmp_ptlst.at(j);			
				allpts.push_back(p);
			}
		}
	}else{
		allpts = *ptlst;
	}

	txtOut << " " << allpts.size() << " \n\n";

	for(int i=0; i<allpts.size();i++){
		point p = allpts.at(i);
		double *x = p.x;
		double *y = p.y;
		double *z = p.z;
		txtOut << " H     " << *x << "     " << *y << "     " << *z << " \n";
	}

	txtOut << "\n";

	txtOut.close();
}


/*Computes representive points for a bin*/
void get_rep_pts(node *n, vector<point> *rep_pts){

         /*
       p8------------------------- p7
         |\                      |\
         | \                     | \
         |  \                    |  \
         | p5\___________________|___\p6
       p4|___|___________________|p3 |
          \  |                    \  |
           \ |                     \ |
            \|______________________\|
             p1                      p2
         */	

	point p1, p2, p3, p4, p5, p6, p7, p8; 

	p1.x = &n->xmin;
	p1.y = &n->ymin;
	p1.z = &n->zmin;

        p2.x = &n->xmax;
        p2.y = &n->ymin;
        p2.z = &n->zmin;	
 
        p3.x = &n->xmax;
        p3.y = &n->ymax;
        p3.z = &n->zmin;

        p4.x = &n->xmin;
        p4.y = &n->ymax;
        p4.z = &n->zmin;

        p5.x = &n->xmin;
        p5.y = &n->ymin;
        p5.z = &n->zmax;

        p6.x = &n->xmax;
        p6.y = &n->ymin;
        p6.z = &n->zmax;
	
        p7.x = &n->xmax;
        p7.y = &n->ymax;
        p7.z = &n->zmax;

        p8.x = &n->xmin;
        p8.y = &n->ymax;
        p8.z = &n->zmax;

	rep_pts->push_back(p1);
	rep_pts->push_back(p2);
	rep_pts->push_back(p3);
	rep_pts->push_back(p4);
	rep_pts->push_back(p5);
	rep_pts->push_back(p6);
	rep_pts->push_back(p7);
	rep_pts->push_back(p8);

}


void pack_grid_pts(){


        clock_t start, end;
        double time_octree;

#if defined MPIV && !defined CUDA_MPIV
	setup_gpack_mpi_1();
#endif

	vector<node> octree;

#if defined MPIV && !defined CUDA_MPIV
    if(mpirank==0){
#endif

	start = clock();

	//Generate the octree
        octree = generate_octree(gps, MAX_POINTS_PER_CLUSTER, OCTREE_DEPTH);

	end = clock();

	time_octree = ((double) (end - start)) / CLOCKS_PER_SEC;

	PRINTOCTTIME("OCTREE ALGORITHM", time_octree)

	gps -> time_octree = time_octree;

#if defined CUDA || defined CUDA_MPIV

	vector<node> new_imp_signodes;

	vector<bflist> new_imp_bflst;

	gpu_get_pfbased_basis_function_lists_new_imp(&octree, &new_imp_signodes, &new_imp_bflst);

#else

#if defined MPIV && !defined CUDA_MPIV
    }

    MPI_Barrier(MPI_COMM_WORLD);
#endif

    cpu_get_pfbased_basis_function_lists_new_imp(&octree);

#if defined MPIV && !defined CUDA_MPIV
    delete_gpack_mpi();
#endif

#endif

}

/*Prune grid points based on ss weights*/
void get_ssw_pruned_grid(){

        // initialize temporary vectors
	vector<double> gridx_out;
	vector<double> gridy_out;
	vector<double> gridz_out;
	vector<int> grid_atm_out;
	vector<double> sswt_out;
	vector<double> weight_out;

        // screen data based on a threshold weight
        for(int i=0;i<gps->arr_size;i++){
                if(gps->ss_weight->_cppData[i] > gps->DMCutoff){

                        gridx_out.push_back(gps->gridx->_cppData[i]);
                        gridy_out.push_back(gps->gridy->_cppData[i]);
                        gridz_out.push_back(gps->gridz->_cppData[i]);
                        sswt_out.push_back(gps->sswt->_cppData[i]);
                        weight_out.push_back(gps->ss_weight->_cppData[i]);
                        grid_atm_out.push_back(gps->grid_atm->_cppData[i]);

                }
        }

        // delete original data
        delete gps->gridx;
        delete gps->gridy;
        delete gps->gridz;
        delete gps->sswt;
        delete gps->ss_weight;
        delete gps->grid_atm;

        // set new array size
        gps->arr_size = gridx_out.size();

        // create new arrays based on new size
        gps->gridx     = new gpack_buffer_type<double>(gps->arr_size);
        gps->gridy     = new gpack_buffer_type<double>(gps->arr_size);
        gps->gridz     = new gpack_buffer_type<double>(gps->arr_size);
        gps->sswt      = new gpack_buffer_type<double>(gps->arr_size);
        gps->ss_weight = new gpack_buffer_type<double>(gps->arr_size);
        gps->grid_atm  = new gpack_buffer_type<int>(gps->arr_size);

        // copy data from temporary vectors into arrays
        copy(gridx_out.begin(), gridx_out.end(), gps->gridx->_cppData);
        copy(gridy_out.begin(), gridy_out.end(), gps->gridy->_cppData);
        copy(gridz_out.begin(), gridz_out.end(), gps->gridz->_cppData);
        copy(sswt_out.begin(), sswt_out.end(), gps->sswt->_cppData);
        copy(weight_out.begin(), weight_out.end(), gps->ss_weight->_cppData);
        copy(grid_atm_out.begin(), grid_atm_out.end(), gps->grid_atm->_cppData);        

}


#if defined CUDA || defined CUDA_MPIV
void gpu_get_pfbased_basis_function_lists_new_imp(vector<node> *octree, vector<node> *signodes, vector<bflist> *bflst){

        double *gridx, *gridy, *gridz, *sswt, *weight;	                 			//Keeps all grid points
        unsigned int *cfweight, *pfweight;   //Holds 1 or 0 depending on the significance of each candidate
	unsigned char *gpweight;
	int *iatm;

	//get the number of octree leaves 
	unsigned int leaf_count = 0;

#ifdef DEBUG
	vector<node> dbg_leaf_nodes; //Store leaves for grid visialization
	vector<node> dbg_signodes;   //Store significant nodes for grid visualization
	vector<int>  dbg_signdidx;    //Keeps track of leaf node indices to remove
	vector<point> dbg_pts; 	     //Keeps all pruned grid points
#endif

        for(int i=0; i<octree -> size();i++){

                node n = octree->at(i);

                if(n.has_children == false || n.level == OCTREE_DEPTH-1){
			leaf_count++;
#ifdef DEBUG
			dbg_leaf_nodes.push_back(n);
#endif
		}
	}

#ifdef DEBUG
	fprintf(gps->gpackDebugFile,"FILE: %s, LINE: %d, FUNCTION: %s, Total number of leaf nodes: %i \n", __FILE__, __LINE__, __func__, leaf_count);
#endif
	
	unsigned int init_arr_size = leaf_count * MAX_POINTS_PER_CLUSTER;
        gridx = (double*) malloc(init_arr_size * sizeof(double));
        gridy = (double*) malloc(init_arr_size * sizeof(double));
        gridz = (double*) malloc(init_arr_size * sizeof(double));
	sswt  = (double*) malloc(init_arr_size * sizeof(double));
	weight= (double*) malloc(init_arr_size * sizeof(double));	

        //bin_counter = (unsigned int*) malloc((leaf_count + 1) * sizeof(unsigned int));
        gpweight = (unsigned char*) malloc(init_arr_size * sizeof(unsigned char));
        cfweight = (unsigned int*) malloc(leaf_count * gps->nbasis * sizeof(unsigned int));
        pfweight = (unsigned int*) malloc(leaf_count * gps->nbasis * gps->maxcontract * sizeof(unsigned int));
	iatm     = (int*) malloc(init_arr_size * sizeof(int));

        unsigned int cgp = 0; //current grid point

	clock_t start, end;
        double time_prep_gpu_input;
        double time_run_gpu;
        double time_proc_gpu_output;

        start = clock();

        for(int i=0; i<octree -> size();i++){
                node n = octree->at(i);

                if(n.has_children == false || n.level == OCTREE_DEPTH-1){
                        //Get all the points in current bin
                        vector<point> pts;

                        pts = n.ptlst;

                        //Go through all points in current bin
                        unsigned int ptofcount = pts.size();
                        for(int r=0;r<ptofcount;r++){
                                point rp = pts.at(r);
                                
                                gridx[cgp] = *rp.x;
                                gridy[cgp] = *rp.y;
                                gridz[cgp] = *rp.z;
				sswt[cgp]  = *rp.sswt;
				weight[cgp]= *rp.weight;
				iatm[cgp]  = *rp.iatm;

				gpweight[cgp] = 1;
                                cgp++;
                        }

			for(int r=ptofcount; r < MAX_POINTS_PER_CLUSTER; r++){
				gridx[cgp] = 0.0;
				gridy[cgp] = 0.0;
				gridz[cgp] = 0.0;
				sswt[cgp]  = 0.0;
				weight[cgp]= 0.0;
				iatm[cgp]  = 0;

				gpweight[cgp] = 0;
				cgp++;
			}

                }

        }

#ifdef DEBUG
	unsigned int init_true_gpcount=0;

	for(int i=0; i<leaf_count*MAX_POINTS_PER_CLUSTER; i++){
		if(gpweight[i]>0){
			init_true_gpcount++;
		}
	}

        fprintf(gps->gpackDebugFile,"FILE: %s, LINE: %d, FUNCTION: %s, Total number of true grid points before pruning: %i \n", __FILE__, __LINE__, __func__, init_true_gpcount);
#endif
	//Also set result arrays to zero
	for(int i=0; i<leaf_count * gps->nbasis;i++){
		cfweight[i]=0;
		for(int j=0; j<gps->maxcontract ; j++){
			pfweight[i*gps->maxcontract+j]=0;
		}
	}

	
        end = clock();

        time_prep_gpu_input = ((double) (end - start)) / CLOCKS_PER_SEC;

        PRINTOCTTIME("PRESCREEN BASIS & PRIMITIVE FUNCTIONS : PREP GPU INPUT", time_prep_gpu_input)

	start = clock();

        gpu_get_octree_info(gridx, gridy, gridz, gps->sigrad2->_cppData, gpweight, cfweight, pfweight, init_arr_size);

	end = clock();

        time_run_gpu = ((double) (end - start)) / CLOCKS_PER_SEC;

        PRINTOCTTIME("PRESCREEN BASIS & PRIMITIVE FUNCTIONS : GPU RUN", time_run_gpu)

	gps -> time_bfpf_prescreen = time_run_gpu;

	start = clock();

	//pruned grid info lists
	vector<int> pgpweight;
	vector<double> pgridx;
	vector<double> pgridy;
	vector<double> pgridz;
	vector<double> psswt;
	vector<double> pweight;
	vector<int> piatm;
	vector<int> pcfweight;
	vector<int> ppfweight;
	vector<int> pcf_counter;
	vector<int> ppf_counter;
	

#ifdef DEBUG
        int dbg_totncf = 0;
#endif

	unsigned int pcf_count=0;
	unsigned int ppf_count=0;

	pcf_counter.push_back(pcf_count);
	ppf_counter.push_back(ppf_count);

	//Get the pruned grid
	for(int i=0; i<leaf_count;i++){
		int cfcount = 0;
		for(int j=0; j<gps -> nbasis; j++){
			if(cfweight[(i * gps -> nbasis) + j] >0){
				cfcount++;
#ifdef DEBUG
				dbg_totncf++;
#endif
			}
		}
		//If there is at least one cf per bin, the bin is significant
		if(cfcount>0){
			for(int j=0; j< MAX_POINTS_PER_CLUSTER; j++){
				pgpweight.push_back(gpweight[i*MAX_POINTS_PER_CLUSTER+j]);
				pgridx.push_back(gridx[i*MAX_POINTS_PER_CLUSTER+j]);
				pgridy.push_back(gridy[i*MAX_POINTS_PER_CLUSTER+j]);
				pgridz.push_back(gridz[i*MAX_POINTS_PER_CLUSTER+j]);
				psswt.push_back(sswt[i*MAX_POINTS_PER_CLUSTER+j]);
				pweight.push_back(weight[i*MAX_POINTS_PER_CLUSTER+j]);
				piatm.push_back(iatm[i*MAX_POINTS_PER_CLUSTER+j]);
#ifdef DEBUG
				point db_p;
				db_p.x = &gridx[i*MAX_POINTS_PER_CLUSTER+j];
				db_p.y = &gridy[i*MAX_POINTS_PER_CLUSTER+j];
				db_p.z = &gridz[i*MAX_POINTS_PER_CLUSTER+j];
				dbg_pts.push_back(db_p);				
#endif
			}
		
			//Save the corresponding contraction function list
			for(int j=0; j<gps -> nbasis; j++){
				if(cfweight[(i * gps -> nbasis) + j] >0){
					pcfweight.push_back(j);
					pcf_count++;
					
					//Save the corresponding primitive list
					for(int k=0; k<gps -> maxcontract; k++){
						if(pfweight[(i * gps -> nbasis * gps -> maxcontract) + j*gps -> maxcontract + k]>0){
							ppfweight.push_back(k);
							ppf_count++;
						}
					}
					ppf_counter.push_back(ppf_count);

				}
			}
		
			pcf_counter.push_back(pcf_count);					
#ifdef DEBUG
			dbg_signodes.push_back(dbg_leaf_nodes.at(i));
#endif

		}
	}

#if defined DEBUG && defined WRITE_TCL_XYZ

        fprintf(gps->gpackDebugFile,"FILE: %s, LINE: %d, FUNCTION: %s, Total number of contracted functions: %i \n", __FILE__, __LINE__, __func__, pcfweight.size());
        fprintf(gps->gpackDebugFile,"FILE: %s, LINE: %d, FUNCTION: %s, Total number of primitive functions: %i \n", __FILE__, __LINE__, __func__, ppfweight.size());

	//print grid for vmd visualization
        write_vmd_grid(dbg_leaf_nodes, "initgrid.tcl");
        write_xyz(&dbg_leaf_nodes, NULL, false, "initgpts.xyz");

	//dbg_signodes = dbg_leaf_nodes;
	//write first 3 levels of the octree for vmd visualization
	vector<node> dbg_lvl0_nodes;
	vector<node> dbg_lvl1_nodes;
	vector<node> dbg_lvl2_nodes;

        for(int i=0; i<octree -> size();i++){

                node n = octree->at(i);

                if(n.level == 0){
                        dbg_lvl0_nodes.push_back(n);
                }else if(n.level == 1){
			dbg_lvl1_nodes.push_back(n);
		}else if(n.level == 2){
			dbg_lvl2_nodes.push_back(n);
		}    
        }
	
	write_vmd_grid(dbg_lvl0_nodes, "octgrid0.tcl");
	write_vmd_grid(dbg_lvl1_nodes, "octgrid1.tcl");
	write_vmd_grid(dbg_lvl2_nodes, "octgrid2.tcl");

	//Prints only the significant bins and points 
	write_vmd_grid(dbg_signodes, "pgrid.tcl");
	write_xyz(NULL, &dbg_pts, true, "bgpts.xyz");
#endif

	//Convert lists into arrays
         
        gps->gridb_count   = pgpweight.size();
        gps->nbins         = (gps->gridb_count)/MAX_POINTS_PER_CLUSTER;
        gps->nbtotbf       = pcfweight.size();
        gps->nbtotpf       = ppfweight.size();

        gps->gridxb        = new gpack_buffer_type<double>(gps->gridb_count);
        gps->gridyb        = new gpack_buffer_type<double>(gps->gridb_count);        
        gps->gridzb        = new gpack_buffer_type<double>(gps->gridb_count);
        gps->gridb_sswt    = new gpack_buffer_type<double>(gps->gridb_count);
        gps->gridb_weight  = new gpack_buffer_type<double>(gps->gridb_count);
        gps->gridb_atm     = new gpack_buffer_type<int>(gps->gridb_count);
        gps->dweight       = new gpack_buffer_type<int>(gps->gridb_count);
        gps->basf          = new gpack_buffer_type<int>(gps->nbtotbf);
        gps->primf         = new gpack_buffer_type<int>(gps->nbtotpf);
        gps->basf_counter  = new gpack_buffer_type<int>(gps->nbins + 1);
        gps->primf_counter = new gpack_buffer_type<int>(gps->nbtotbf + 1);

	copy(pgridx.begin(), pgridx.end(), gps->gridxb->_cppData);
	copy(pgridy.begin(), pgridy.end(), gps->gridyb->_cppData);
	copy(pgridz.begin(), pgridz.end(), gps->gridzb->_cppData);
	copy(psswt.begin(), psswt.end(), gps->gridb_sswt->_cppData);
	copy(pweight.begin(), pweight.end(), gps->gridb_weight->_cppData);
        copy(piatm.begin(), piatm.end(), gps->gridb_atm->_cppData);
	copy(pgpweight.begin(), pgpweight.end(), gps->dweight->_cppData);
	copy(pcfweight.begin(), pcfweight.end(), gps->basf->_cppData);
	copy(ppfweight.begin(), ppfweight.end(), gps->primf->_cppData);
	copy(pcf_counter.begin(), pcf_counter.end(), gps->basf_counter->_cppData);
	copy(ppf_counter.begin(), ppf_counter.end(), gps->primf_counter->_cppData);

	int ntgpts = 0;
	for(int i=0; i<gps->gridb_count; i++){
		if(gps->dweight->_cppData[i] > 0){

			ntgpts++;
		}
	}

        gps->ntgpts = ntgpts;

        free(gridx);
        free(gridy);
        free(gridz);
        free(gpweight);
        free(cfweight);
        free(pfweight);
        free(sswt);
        free(weight);
        free(iatm);

#ifdef DEBUG
        fprintf(gps->gpackDebugFile,"FILE: %s, LINE: %d, FUNCTION: %s, Total number of true bins & grid points after pruning: %i %i\n", __FILE__, __LINE__, __func__, gps->nbins, gps->ntgpts);
#endif

        end = clock();

        time_proc_gpu_output = ((double) (end - start)) / CLOCKS_PER_SEC;

        PRINTOCTTIME("PRESCREEN BASIS & PRIMITIVE FUNCTIONS : PROCESS GPU OUTPUT", time_proc_gpu_output)

}
#endif

//#define CBFPF_DEBUG


// This function prepares primitive and contracted function lists in serial and mpi versions
void cpu_get_pfbased_basis_function_lists_new_imp(vector<node> *octree){

        double *gridx, *gridy, *gridz, *sswt, *weight;                                          //Keeps all grid points
        unsigned int *cfweight, *pfweight;   //Holds 1 or 0 depending on the significance of each candidate
	unsigned int *bs_tracker;  //Keeps track of bin sizes 
        unsigned char *gpweight;
        int *iatm; 

	unsigned char *tmp_gpweight;
	unsigned int *tmp_cfweight, *tmp_pfweight;
        //get the number of octree leaves 
        unsigned int leaf_count = 0;

#ifdef CBFPF_DEBUG
        vector<node> dbg_leaf_nodes; //Store leaves for grid visualization
        vector<node> dbg_signodes;   //Store significant nodes for grid visualization
        vector<int>  dbg_signdidx;    //Keeps track of leaf node indices to remove
        vector<point> dbg_pts;       //Keeps all pruned grid points
#endif

        for(int i=0; i<octree -> size();i++){

                node n = octree->at(i);

                if(n.has_children == false || n.level == OCTREE_DEPTH-1){
                        leaf_count++;
#ifdef CBFPF_DEBUG
                        dbg_leaf_nodes.push_back(n);
#endif
                }
        }

#ifdef CBFPF_DEBUG
        fprintf(gps->gpackDebugFile,"FILE: %s, LINE: %d, FUNCTION: %s, Total number of leaf nodes: %i \n", __FILE__, __LINE__, __func__, leaf_count);
#endif

        unsigned int init_arr_size = gps->arr_size;

        gridx = (double*) malloc(init_arr_size * sizeof(double));
        gridy = (double*) malloc(init_arr_size * sizeof(double));
        gridz = (double*) malloc(init_arr_size * sizeof(double));
        sswt  = (double*) malloc(init_arr_size * sizeof(double));
        weight= (double*) malloc(init_arr_size * sizeof(double));

#if defined MPIV && !defined CUDA_MPIV
        MPI_Bcast(&leaf_count, 1, MPI_INT, 0, MPI_COMM_WORLD); 
#endif
	tmp_gpweight = (unsigned char*) malloc(init_arr_size * sizeof(unsigned char));
	tmp_cfweight = (unsigned int*) malloc(leaf_count * gps->nbasis * sizeof(unsigned int));
	tmp_pfweight = (unsigned int*) malloc(leaf_count * gps->nbasis * gps->maxcontract * sizeof(unsigned int));

        //bin_counter = (unsigned int*) malloc((leaf_count + 1) * sizeof(unsigned int));
        gpweight = (unsigned char*) malloc(init_arr_size * sizeof(unsigned char));
        cfweight = (unsigned int*) malloc(leaf_count * gps->nbasis * sizeof(unsigned int));
        pfweight = (unsigned int*) malloc(leaf_count * gps->nbasis * gps->maxcontract * sizeof(unsigned int));
        iatm     = (int*) malloc(init_arr_size * sizeof(int));
	bs_tracker = (unsigned int*) malloc((leaf_count+1) * sizeof(unsigned int));	

        unsigned int cgp = 0; //current grid point
	unsigned int cb = 0;

        clock_t start, end;
        double time_prep_input;
        double run_time;
        double time_proc_output;

#if defined MPIV && !defined CUDA_MPIV
	double mpi_prep_time;
	double mpi_run_time;
	double mpi_post_proc_time;
#endif

#if defined MPIV && !defined CUDA_MPIV
        if(mpirank == 0){
#endif

        start = clock();

	bs_tracker[cb] = 0;

        for(int i=0; i<octree -> size();i++){
                node n = octree->at(i);

                if(n.has_children == false || n.level == OCTREE_DEPTH-1){
                        //Get all the points in current bin
                        vector<point> pts;

                        pts = n.ptlst;

                        //Go through all points in current bin
                        unsigned int ptofcount = pts.size();
                        for(int r=0;r<ptofcount;r++){
                                point rp = pts.at(r);

                                gridx[cgp] = *rp.x;
                                gridy[cgp] = *rp.y;
                                gridz[cgp] = *rp.z;
                                sswt[cgp]  = *rp.sswt;
                                weight[cgp]= *rp.weight;
                                iatm[cgp]  = *rp.iatm;
                                gpweight[cgp] = 1;
				
                                cgp++;
                        }
			cb++;
			bs_tracker[cb] = cgp;

                }

        }

#ifdef CBFPF_DEBUG
        fprintf(gps->gpackDebugFile,"FILE: %s, LINE: %d, FUNCTION: %s, Total number of true grid points before pruning: %i \n", __FILE__, __LINE__, __func__, init_arr_size);
#endif
        //Also set result arrays to zero
        for(int i=0; i<leaf_count * gps->nbasis;i++){
                cfweight[i]=0;
		tmp_cfweight[i]=0;
                for(int j=0; j<gps->maxcontract ; j++){
                        pfweight[i*gps->maxcontract+j]=0;
			tmp_pfweight[i*gps->maxcontract+j]=0;
                }
        }


        end = clock();

        time_prep_input = ((double) (end - start)) / CLOCKS_PER_SEC;

        PRINTOCTTIME("PRESCREEN BASIS & PRIMITIVE FUNCTIONS : PREP INPUT", time_prep_input)

        start = clock();

#if defined MPIV && !defined CUDA_MPIV
	}

	setup_gpack_mpi_2(leaf_count, gridx, gridy, gridz, gpweight, tmp_gpweight, cfweight, tmp_cfweight, pfweight, tmp_pfweight, sswt, weight, iatm, bs_tracker);

#endif

#if defined MPIV && !defined CUDA_MPIV
	if(mpirank == 0){
	end = clock();

        mpi_prep_time = ((double) (end - start)) / CLOCKS_PER_SEC;

        PRINTOCTTIME("PRESCREEN BASIS & PRIMITIVE FUNCTIONS : BROADCAST INPUT", mpi_prep_time)

	start = clock();
	}

#endif

	int bstart, bend;

#if defined MPIV && !defined CUDA_MPIV
        bstart=mpi_binlst[mpirank];
        bend=mpi_binlst[mpirank+1];
#else
        bstart=0;
        bend=leaf_count;        
#endif

	for(unsigned int i=bstart; i< bend; i++){
		for(unsigned int j=bs_tracker[i]; j<bs_tracker[i+1]; j++){
			cpu_get_primf_contraf_lists_method_new_imp(gridx[j], gridy[j], gridz[j], gpweight, cfweight, pfweight, i, j);	
		}	
	}

#if defined MPIV && !defined CUDA_MPIV
        if(mpirank == 0){
        end = clock();

        mpi_run_time = ((double) (end - start)) / CLOCKS_PER_SEC;

        PRINTOCTTIME("PRESCREEN BASIS & PRIMITIVE FUNCTIONS : RUN TIME", mpi_run_time)
        
        start = clock();
        }

#endif



#if defined MPIV && !defined CUDA_MPIV
        
        get_slave_primf_contraf_lists(leaf_count, gpweight, tmp_gpweight, cfweight, tmp_cfweight, pfweight, tmp_pfweight, bs_tracker);

        if(mpirank == 0){

	  end = clock();

	  mpi_post_proc_time = ((double) (end - start)) / CLOCKS_PER_SEC;

          PRINTOCTTIME("PRESCREEN BASIS & PRIMITIVE FUNCTIONS : PROCESS OUTPUT", mpi_post_proc_time)

	  gps -> time_bfpf_prescreen = mpi_post_proc_time+mpi_run_time+mpi_prep_time;
#else

          end = clock();

          run_time = ((double) (end - start)) / CLOCKS_PER_SEC;

	  PRINTOCTTIME("PRESCREEN BASIS & PRIMITIVE FUNCTIONS : RUN TIME", run_time)        

          gps -> time_bfpf_prescreen = run_time;
#endif
          start = clock();

          //pruned grid info lists
          vector<int> pgpweight;
          vector<double> pgridx;
          vector<double> pgridy;
          vector<double> pgridz;
          vector<double> psswt;
          vector<double> pweight;
          vector<int> piatm;
          vector<int> pcfweight;
          vector<int> ppfweight;
          vector<int> pcf_counter;
          vector<int> ppf_counter;
	  vector<int> pbs_tracker;


#ifdef CBFPF_DEBUG
          int dbg_totncf = 0;
#endif

          unsigned int pcf_count=0;
          unsigned int ppf_count=0;
	  unsigned int ppt_count=0;

          pcf_counter.push_back(pcf_count);
          ppf_counter.push_back(ppf_count);
	  pbs_tracker.push_back(ppt_count);	

          //Get the pruned grid
          for(int i=0; i<leaf_count;i++){
                  int cfcount = 0;
                  for(int j=0; j<gps -> nbasis; j++){
                          if(cfweight[(i * gps -> nbasis) + j] >0){
                                  cfcount++;
#ifdef CBFPF_DEBUG
                                  dbg_totncf++;
#endif
                          }
                  }
                  //If there is at least one cf per bin, the bin is significant
                  if(cfcount>0){

                        for(int j=bs_tracker[i]; j< bs_tracker[i+1]; j++){
		  		if(gpweight[j]>0){				
                                  	pgridx.push_back(gridx[j]);
                                	pgridy.push_back(gridy[j]);
                                	pgridz.push_back(gridz[j]);
                                	psswt.push_back(sswt[j]);
                                	pweight.push_back(weight[j]);
                                	piatm.push_back(iatm[j]);
					ppt_count++;
#ifdef CBFPF_DEBUG
                                	point db_p;
                                	db_p.x = &gridx[j];
                                	db_p.y = &gridy[j];
                                	db_p.z = &gridz[j];
                                	dbg_pts.push_back(db_p);
#endif
				}
                        }

			pbs_tracker.push_back(ppt_count);

                        //Save the corresponding contraction function list
                        for(int j=0; j<gps -> nbasis; j++){
                                if(cfweight[(i * gps -> nbasis) + j] >0){
                                        pcfweight.push_back(j);
                                        pcf_count++;

                                        //Save the corresponding primitive list
                                        for(int k=0; k<gps -> maxcontract; k++){
                                                if(pfweight[(i * gps -> nbasis * gps -> maxcontract) + j*gps -> maxcontract + k]>0){
                                                        ppfweight.push_back(k);
                                                        ppf_count++;
                                                }
                                        }
                                        ppf_counter.push_back(ppf_count);

                                }
                        }

                        pcf_counter.push_back(pcf_count);
#ifdef CBFPF_DEBUG
                        dbg_signodes.push_back(dbg_leaf_nodes.at(i));

                        for(int j=0; j<gps -> nbasis; j++){

                                //Save the corresponding primitive list
                                for(int k=0; k<gps -> maxcontract; k++){
//					printf("Debugging cpu vs gpu Leaf: %i bfct: %i pfct: %i \n", i, cfweight[(i * gps -> nbasis) + j], pfweight[(i * gps -> nbasis * gps -> maxcontract) + j*gps -> maxcontract + k]);
                                }

                        }

#endif

                  }
          }

#if defined DEBUG && defined WRITE_TCL_XYZ

          fprintf(gps->gpackDebugFile,"FILE: %s, LINE: %d, FUNCTION: %s, Total number of contracted functions: %i \n", __FILE__, __LINE__, __func__, pcfweight.size());
          fprintf(gps->gpackDebugFile,"FILE: %s, LINE: %d, FUNCTION: %s, Total number of primitive functions: %i \n", __FILE__, __LINE__, __func__, ppfweight.size());


          //print grid for vmd visualization
          write_vmd_grid(dbg_leaf_nodes, "initgrid.tcl");
          write_xyz(&dbg_leaf_nodes, NULL, false, "initgpts.xyz");

          //dbg_signodes = dbg_leaf_nodes;
          //write first 3 levels of the octree for vmd visualization
          vector<node> dbg_lvl0_nodes;
          vector<node> dbg_lvl1_nodes;
          vector<node> dbg_lvl2_nodes;

          for(int i=0; i<octree -> size();i++){

                node n = octree->at(i);

                if(n.level == 0){
                        dbg_lvl0_nodes.push_back(n);
                }else if(n.level == 1){
                        dbg_lvl1_nodes.push_back(n);
                }else if(n.level == 2){
                        dbg_lvl2_nodes.push_back(n);
                }
          }

          write_vmd_grid(dbg_lvl0_nodes, "octgrid0.tcl");
          write_vmd_grid(dbg_lvl1_nodes, "octgrid1.tcl");
          write_vmd_grid(dbg_lvl2_nodes, "octgrid2.tcl");

          //Prints only the significant bins and points 
          write_vmd_grid(dbg_signodes, "pgrid.tcl");
          write_xyz(NULL, &dbg_pts, true, "bgpts.xyz");
#endif

          //Convert lists into arrays
//        int pgridinfo_arr_size = ppt_count;

          gps->gridb_count   = ppt_count;
          gps->nbins         = pbs_tracker.size() - 1;
          gps->nbtotbf       = pcfweight.size();
          gps->nbtotpf       = ppfweight.size();
          gps->ntgpts        = gps->gridb_count;

          gps->gridxb        = new gpack_buffer_type<double>(gps->gridb_count);
          gps->gridyb        = new gpack_buffer_type<double>(gps->gridb_count);
          gps->gridzb        = new gpack_buffer_type<double>(gps->gridb_count);
          gps->gridb_sswt    = new gpack_buffer_type<double>(gps->gridb_count);
          gps->gridb_weight  = new gpack_buffer_type<double>(gps->gridb_count);
          gps->gridb_atm     = new gpack_buffer_type<int>(gps->gridb_count);
          gps->basf          = new gpack_buffer_type<int>(gps->nbtotbf);
          gps->primf         = new gpack_buffer_type<int>(gps->nbtotpf);
          gps->basf_counter  = new gpack_buffer_type<int>(gps->nbins + 1);
          gps->primf_counter = new gpack_buffer_type<int>(gps->nbtotbf + 1);
          gps->bin_counter   = new gpack_buffer_type<int>(gps->nbins + 1);

          copy(pgridx.begin(), pgridx.end(), gps->gridxb->_cppData);
          copy(pgridy.begin(), pgridy.end(), gps->gridyb->_cppData);
          copy(pgridz.begin(), pgridz.end(), gps->gridzb->_cppData);
          copy(psswt.begin(), psswt.end(), gps->gridb_sswt->_cppData);
          copy(pweight.begin(), pweight.end(), gps->gridb_weight->_cppData);
          copy(piatm.begin(), piatm.end(), gps->gridb_atm->_cppData);
          copy(pcfweight.begin(), pcfweight.end(), gps->basf->_cppData);
          copy(ppfweight.begin(), ppfweight.end(), gps->primf->_cppData);
          copy(pcf_counter.begin(), pcf_counter.end(), gps->basf_counter->_cppData);
          copy(ppf_counter.begin(), ppf_counter.end(), gps->primf_counter->_cppData);
	  copy(pbs_tracker.begin(), pbs_tracker.end(), gps->bin_counter->_cppData);

#ifdef DEBUG
          fprintf(gps->gpackDebugFile,"FILE: %s, LINE: %d, FUNCTION: %s, Total number of true bins & grid points after pruning: %i %i\n", __FILE__, __LINE__, __func__, gps->nbins, gps->ntgpts);
#endif

          end = clock();

          time_proc_output = ((double) (end - start)) / CLOCKS_PER_SEC;

          PRINTOCTTIME("PRESCREEN BASIS & PRIMITIVE FUNCTIONS : PROCESS OUTPUT", time_proc_output)

#if defined MPIV && !defined CUDA_MPIV
	}
#endif

        free(gridx);
        free(gridy);
        free(gridz);
        free(gpweight);
        free(cfweight);
        free(pfweight);
        free(sswt);
        free(weight);
        free(bs_tracker);
        free(iatm);

	free(tmp_gpweight);
	free(tmp_cfweight);
	free(tmp_pfweight);

}



void cpu_get_primf_contraf_lists_method_new_imp(double gridx, double gridy, double gridz, unsigned char *gpweight, unsigned int *cfweight, unsigned int *pfweight, unsigned int bin_id, unsigned int gid){

        unsigned int sigcfcount=0;

        // relative coordinates between grid point and basis function I.

        for(int ibas=0; ibas<gps->nbasis;ibas++){

		unsigned int nc = (gps->ncenter->_cppData[ibas])-1;
                unsigned long cfwid = bin_id * gps->nbasis + ibas; //Change here
        	double x1 = gridx - gps->xyz->_cppData[0+nc*3];
        	double y1 = gridy - gps->xyz->_cppData[1+nc*3];
        	double z1 = gridz - gps->xyz->_cppData[2+nc*3];
			

                double x1i, y1i, z1i;
                double x1imin1, y1imin1, z1imin1;
                double x1iplus1, y1iplus1, z1iplus1;

                double phi = 0.0;
                double dphidx = 0.0;
                double dphidy = 0.0;
                double dphidz = 0.0;

		int itypex = gps->itype->_cppData[0+ibas*3];
                int itypey = gps->itype->_cppData[1+ibas*3];
                int itypez = gps->itype->_cppData[2+ibas*3];
                double dist = x1*x1+y1*y1+z1*z1;

                       if ( dist <= gps->sigrad2->_cppData[ibas]){

                               if ( itypex == 0) {
                                       x1imin1 = 0.0;
                                       x1i = 1.0;
                                       x1iplus1 = x1;
                               }else {
                                       x1imin1 = pow(x1, itypex-1);
                                       x1i = x1imin1 * x1;
                                       x1iplus1 = x1i * x1;
                               }

                               if ( itypey == 0) {
                                       y1imin1 = 0.0;
                                       y1i = 1.0;
                                       y1iplus1 = y1;
                               }else {
                                       y1imin1 = pow(y1, itypey-1);
                                       y1i = y1imin1 * y1;
                                       y1iplus1 = y1i * y1;
                               }

                               if ( itypez == 0) {
                                       z1imin1 = 0.0;
                                       z1i = 1.0;
                                       z1iplus1 = z1;
                               }else {
                                       z1imin1 = pow(z1, itypez-1);
                                       z1i = z1imin1 * z1;
                                       z1iplus1 = z1i * z1;
                               }
                               for(int kprim=0; kprim< gps->ncontract->_cppData[ibas]; kprim++){

                                       unsigned long pfwid = bin_id * gps->nbasis * gps->maxcontract + ibas * gps->maxcontract + kprim; //Change
				       double alpha = gps->aexp->_cppData[kprim + ibas * gps->maxcontract];
				       double tmp = (gps->dcoeff->_cppData[kprim + ibas * gps->maxcontract]) * exp( -alpha * dist);

					double tmpdx = tmp * ( -2.0 * alpha * x1iplus1 + (double)itypex * x1imin1);
					double tmpdy = tmp * ( -2.0 * alpha * y1iplus1 + (double)itypey * y1imin1);
					double tmpdz = tmp * ( -2.0 * alpha * z1iplus1 + (double)itypez * z1imin1);

                                       phi = phi + tmp;
                                       dphidx = dphidx + tmpdx;
                                       dphidy = dphidy + tmpdy;
                                       dphidz = dphidz + tmpdz;

                                       //Check the significance of the primitive
                                       if(abs(tmp+tmpdx+tmpdy+tmpdz) > gps->DMCutoff){
                                               pfweight[pfwid] += 1;
                                       }
                               }

                               phi = phi * x1i * y1i * z1i;
                               dphidx = dphidx * y1i * z1i;
                               dphidy = dphidy * x1i * z1i;
                               dphidz = dphidz * x1i * y1i;

                       }

                       if (abs(phi+dphidx+dphidy+dphidz)> gps->DMCutoff ){
                               cfweight[cfwid] += 1;
                               sigcfcount++;
                       }

               }
               if(sigcfcount < 1){
                       gpweight[gid] = 0;
               }
       
}

#if defined MPIV && !defined CUDA_MPIV

void setup_gpack_mpi_1(){

	MPI_Bcast(&gps->arr_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

}


void setup_gpack_mpi_2(unsigned int nbins, double *gridx, double *gridy, double *gridz, unsigned char *gpweight, unsigned char *tmp_gpweight, unsigned int *cfweight, unsigned int *tmp_cfweight, unsigned int *pfweight, unsigned int *tmp_pfweight, double *sswt, double *weight, int *iatm, unsigned int *bs_tracker){
	unsigned int tmp_arr[mpisize];
	unsigned int *tmp_mpi_binlst;

	tmp_mpi_binlst = (unsigned int*) malloc((mpisize+1)*sizeof(unsigned int));
	
        if(mpirank == 0){

	//Set array values to zero
	tmp_mpi_binlst[0]=0;
	for(unsigned int i=1;i<mpisize+1;i++){
		tmp_mpi_binlst[i]=0;
		tmp_arr[i-1]=0;
	}

	//Set bin count for each cpu
	unsigned int ndist = nbins;
	do{

		for(unsigned int j=0; j<mpisize; j++){

			if(ndist < 1 ){
                                break;
                        }else{

				tmp_arr[j] = tmp_arr[j] +1;
				ndist--;
			}

		}
	}while(ndist > 0);

	//Set bin ranges for each cpu
	tmp_mpi_binlst[0]=0;
	for(unsigned int i=1; i<mpisize+1; i++){
		tmp_mpi_binlst[i] = tmp_mpi_binlst[i-1] + tmp_arr[i-1];
	}


	}	

        mpi_binlst = tmp_mpi_binlst;

	MPI_Bcast(mpi_binlst, mpisize+1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(bs_tracker, nbins+1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(gridx, gps->arr_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(gridy, gps->arr_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(gridz, gps->arr_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(gpweight, gps->arr_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
	MPI_Bcast(cfweight, nbins*gps->nbasis, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(pfweight, nbins*gps->nbasis*gps->maxcontract, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(sswt, gps->arr_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(weight, gps->arr_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(iatm, gps->arr_size, MPI_INT, 0, MPI_COMM_WORLD);

	MPI_Barrier(MPI_COMM_WORLD);


}


void get_slave_primf_contraf_lists(unsigned int nbins, unsigned char *gpweight, unsigned char *tmp_gpweight, unsigned int *cfweight, unsigned int *tmp_cfweight, unsigned int *pfweight, unsigned int *tmp_pfweight, unsigned int *bs_tracker){

        MPI_Status status;
	clock_t start, end;

        if(mpirank != 0){

                        MPI_Send(gpweight, gps->arr_size, MPI_UNSIGNED_CHAR, 0, mpirank+600, MPI_COMM_WORLD);
                        MPI_Send(cfweight, nbins*gps->nbasis, MPI_INT, 0, mpirank+700, MPI_COMM_WORLD);
                        MPI_Send(pfweight, nbins*gps->nbasis*gps->maxcontract, MPI_INT, 0, mpirank+800, MPI_COMM_WORLD);

        }else{

		start=clock();

		for(unsigned int i=1; i< mpisize; i++){

			MPI_Recv(tmp_gpweight, gps->arr_size, MPI_UNSIGNED_CHAR, i, i+600, MPI_COMM_WORLD, &status);
			MPI_Recv(tmp_cfweight, nbins*gps->nbasis, MPI_INT, i, i+700, MPI_COMM_WORLD, &status);
			MPI_Recv(tmp_pfweight, nbins*gps->nbasis*gps->maxcontract, MPI_INT, i, i+800, MPI_COMM_WORLD, &status);

		        unsigned int bstart=mpi_binlst[i];
			unsigned int bend=mpi_binlst[i+1];	

			for(unsigned int j=bstart; j< bend; j++){

				for(unsigned int k=bs_tracker[j]; k<bs_tracker[j+1]; k++){
					gpweight[k] = tmp_gpweight[k];

				}
				for(unsigned int kbas=j*gps->nbasis;kbas<((j+1)*gps->nbasis);kbas++){
					cfweight[kbas] += tmp_cfweight[kbas];

					unsigned long init_pfid = kbas*gps->maxcontract;
					for(unsigned int kprim=0;kprim<gps->maxcontract;kprim++){
						unsigned long pfwid = init_pfid+kprim;
						pfweight[pfwid] += tmp_pfweight[pfwid]; 
					}

				}

				

			}

		}

		end = clock();

	//	printf("Time for running through data: %f \n",  ((double) (end - start)) / CLOCKS_PER_SEC);

        }

        MPI_Barrier(MPI_COMM_WORLD);

}

void delete_gpack_mpi(){

		free(mpi_binlst);
}

#endif

