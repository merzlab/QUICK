/* Written by Madu Manathunga on 10/22/2019. */
#include "grid_packer.h"
#include <cmath>
#include <fstream>
#include <time.h>


/*Deallocates memory of a grd_pck_strct*/
void dealloc_grd_pck_strct(grd_pck_strct *gp){

        free(gp->gridx);
        free(gp->gridy);
        free(gp->gridz);
        free(gp->sswt);
        free(gp->ss_weight);
        free(gp->grid_atm);
        free(gp->gridxb);
        free(gp->gridyb);
        free(gp->gridzb);
        free(gp->gridb_sswt);
        free(gp->gridb_weight);
        free(gp->gridb_atm);
        free(gp->basf);
        free(gp->basf_counter);
        free(gp->primf);
        free(gp->primf_counter);
#ifdef CUDA
        free(gp->dweight);
#else
        free(gp->bin_counter);
#endif
        free(gp);
}


void save_dft_grid_info_(double *gridx, double *gridy, double *gridz, double *ssw, double *weight, int *atm, int *dweight, int *basf, int *primf, int *basf_counter, int *primf_counter, int *bin_counter){

	for(int i=0; i< gpst.gridb_count;i++){
		gridx[i]=gpst.gridxb[i];
		gridy[i]=gpst.gridyb[i];
		gridz[i]=gpst.gridzb[i];
		ssw[i]=gpst.gridb_sswt[i];
		weight[i]=gpst.gridb_weight[i];
		atm[i]=gpst.gridb_atm[i];
#ifdef CUDA
		dweight[i]=gpst.dweight[i];
#endif
	}

	for(int i=0;i<gpst.nbtotbf;i++){
		basf[i]=gpst.basf[i];
	}

	for(int i=0; i<gpst.nbtotpf;i++){
		primf[i]=gpst.primf[i];
	}

	for(int i=0; i< (gpst.nbins +1); i++){
		basf_counter[i]=gpst.basf_counter[i];
#ifndef CUDA
		bin_counter[i]=gpst.bin_counter[i];	
#endif
	}

	for(int i=0; i< (gpst.nbtotbf +1);i++){
		primf_counter[i]=gpst.primf_counter[i];
	}
}

/*Fortran accessible method to pack grid points*/
void pack_grid_pts_f90_(double *grid_ptx, double *grid_pty, double *grid_ptz, int *grid_atm, double *grid_sswt, double *grid_weight, int *arr_size, int *nbasis, int *maxcontract, double *DMCutoff, double *sigrad2, int *ncontract, double *aexp, double *dcoeff, int *ncenter, int *itype, double *xyz, int *ngpts, int *nbins, int *nbtotbf, int *nbtotpf, double *toct, double *tprscrn){


	/*Prune grid points based on ssw. This is written with stupid array memory allocations
 	Must be cleaned up with lists.*/

        grd_pck_strct *gps_ssw, *gps;
        gps_ssw = (grd_pck_strct*) malloc(sizeof(grd_pck_strct));
	gps = (grd_pck_strct*) malloc(sizeof(grd_pck_strct));

        gps_ssw->gridx = grid_ptx;
        gps_ssw->gridy = grid_pty;
        gps_ssw->gridz = grid_ptz;
        gps_ssw->sswt = grid_sswt;
        gps_ssw->ss_weight = grid_weight;
        gps_ssw->grid_atm = grid_atm;
        gps_ssw->arr_size = *arr_size;
	gps_ssw->DMCutoff = *DMCutoff;

	get_pruned_grid_ssw(gps_ssw, gps);

/*	for(int i=0; i<gps_ssw->arr_size;i++){
		printf("test get_pruned_grid_ssw: %i x: %f y: %f z: %f sswt: %f weight: %f atm: %i \n ", i, gps_ssw->gridx[i], gps_ssw->gridy[i], gps_ssw->gridz[i], gps_ssw->sswt[i], gps_ssw->ss_weight[i], gps_ssw->grid_atm[i]);
	}
*/
        gps->nbasis = *nbasis;
        gps->maxcontract = *maxcontract;
        gps->DMCutoff = *DMCutoff;
        gps->sigrad2 = sigrad2;
        gps->ncontract = ncontract;
        gps->aexp = aexp;
        gps->dcoeff = dcoeff;
        gps->ncenter = ncenter;
        gps->itype = itype;
        gps->xyz = xyz;

	gpst = *gps;

        pack_grid_pts(&gpst);

	*ngpts = gpst.gridb_count;
	*nbins = gpst.nbins;
	*nbtotbf = gpst.nbtotbf;
	*nbtotpf = gpst.nbtotpf;
	*toct = gpst.time_octree;
	*tprscrn = gpst.time_bfpf_prescreen;
	
	free(gps_ssw);
//	free(gps);

}

//Prints the spatial grid used to generate the octree.
void write_vmd_grid(vector<node> octree, string filename){

	ofstream txtOut;

        //Convert the string file name into char array
        char fname[filename.length() + 1];
        for(int i=0; i<sizeof(fname);i++){
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
	char fname[filename.length() + 1];
	for(int i=0; i<sizeof(fname);i++){
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

//Calculates the value of a basis function at a given grid point
void pteval(grd_pck_strct *gps, double gridx, double gridy, double gridz, double* phi, double* dphidx, double* dphidy,  double* dphidz, int ibas, vector<int> *prim_lst){

	int nc = (gps->ncenter[ibas])-1;

	double x1 = gridx - gps->xyz[0+nc*3];
	double y1 = gridy - gps->xyz[1+nc*3];
	double z1 = gridz - gps->xyz[2+nc*3];

	double x1i, y1i, z1i;
	double x1imin1, y1imin1, z1imin1;
	double x1iplus1, y1iplus1, z1iplus1;
	
	*phi = 0.0;
	*dphidx = 0.0;
	*dphidy = 0.0;
	*dphidz = 0.0;

	int itypex = gps->itype[0+ibas*3];
	int itypey = gps->itype[1+ibas*3];
	int itypez = gps->itype[2+ibas*3];

	double dist = x1*x1+y1*y1+z1*z1;

	if ( dist <= gps->sigrad2[ibas]){
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

        for (int i = 0; i < gps->ncontract[ibas]; i++) {
		double alpha = gps->aexp[i + ibas * gps->maxcontract];
		double tmp = (gps->dcoeff[i + ibas * gps->maxcontract]) * exp( -alpha * dist);

		double tmpdx = tmp * ( -2.0 * alpha * x1iplus1 + (double)itypex * x1imin1);
		double tmpdy = tmp * ( -2.0 * alpha * y1iplus1 + (double)itypey * y1imin1);
		double tmpdz = tmp * ( -2.0 * alpha * z1iplus1 + (double)itypez * z1imin1);

		*phi = *phi + tmp;		
		*dphidx = *dphidx + tmpdx;
		*dphidy = *dphidy + tmpdy;
		*dphidz = *dphidz + tmpdz;

		/*Check the significance of the primitive and add the corresponding index to prim_lst*/
		if(abs(tmp+tmpdx+tmpdy+tmpdz) > gps->DMCutoff){
			prim_lst->push_back(i);	
		}
        }

        *phi = *phi * x1i * y1i * z1i;
        *dphidx = *dphidx * y1i * z1i;
        *dphidy = *dphidy * x1i * z1i;
        *dphidz = *dphidz * x1i * y1i;

	}
		
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

//This method goes though representative points for a bin and select the significant basis functions
//based on value the of contracted functions
vector<bflist> get_cfbased_basis_function_lists(vector<node> *octree, grd_pck_strct *gps){

        /*A vector to hold basis function lists for each node. Size of the vector equals to the number of nodes*/
        vector<bflist> bflst;

        /*Store all the basis function indices in a list*/
        vector<int> all_bfs;
        for(int j=0; j < gps->nbasis; j++){
                all_bfs.push_back(j);
        }

        for(int i=0;i<octree->size();i++){
                node n = octree->at(i);
                if(n.has_children == false || n.level == OCTREE_DEPTH-1){

                        /*Get the representative points*/
                        vector<point> rep_pts;
                        get_rep_pts(&n, &rep_pts);

                        /* define a bflist type variable to store basis functions of the node*/
                        bflist bflstn;

                        /*Set the corresponding node id*/
                        bflstn.node_id = n.id;

                        /*basis function list of node n*/
                        vector<bas_func> bfs;

                        /*Load all the basis functions into a temporary list*/
                        vector<int> tmp_bfs = all_bfs;

                        /*Go through rep_pts, pick significant basis functions from tmp_bfs and store them in bflstn */
                        for(int r=0;r<rep_pts.size();r++){

                                point rp = rep_pts.at(r);

                                /*We will use an iterator since we have to remove basis functions from tmp_bfs as we pick them*/
                                vector<int>:: iterator j = tmp_bfs.begin();

                                while( j != tmp_bfs.end()){
                                        /*Get the jth basis function*/
                                        int jbas = *j;

                                        /*Define a bas_func type variable to hold the primitives of jbas basis function*/
                                        bas_func bf;
                                        bf.bas_id = jbas;

                                        /*Define an integer vector to keep a list of primitive functions belonging to jth basis function*/
                                        vector<int> pl;
					
					//Also define a dummy primitive array to call pteval. We wont be using the pf indices it sends. 
					vector<int> dummy_pl;

                                        /*Evalute the value and the gradient of jbas at point rp*/
                                        double phi, dphidx, dphidy, dphidz;
                                        pteval(gps, *rp.x, *rp.y, *rp.z, &phi, &dphidx, &dphidy, &dphidz, jbas, &dummy_pl);

                                        if (abs(phi+dphidx+dphidy+dphidz)> gps->DMCutoff ){

                                                /*Go through primitives of jth basis function and list the significant ones*/
                                                for(int jprim=0; jprim < gps->ncontract[jbas]; jprim++){
                                                        pl.push_back(jprim);
                                                }

                                                /*Save the primitive list only if it is non-empty*/
                                                if(pl.size() > 0){
                                                        bf.prim_list = pl;
                                                        bfs.push_back(bf);
                                                }

                                                /*Remove the jth basis index from tmp_bfs list and update*/
                                                j = tmp_bfs.erase(j);
                                        }else{
                                                j++;
                                        }
                                }

                        }

                        /*Save the basis function list only if it is non-empty*/
                        if(bfs.size()>0){
                                bflstn.bfs = bfs;
                                bflst.push_back(bflstn);
                        }

                }
        }

        printf("octree->size(): %i bflst.size(): %i \n", octree->size(), bflst.size());

        return bflst;
}



//This method goes though each each grid point in a bin and select the significant basis functions
//based on value of the primitive functions
vector<bflist> get_pfbased_basis_function_lists(vector<node> *octree, grd_pck_strct *gps){

	/*A vector to hold basis function lists for each node. Size of the vector equals to the number of nodes*/
	vector<bflist> bflst;

	/*Store all the basis function indices in a list*/
	vector<int> all_bfs;
 	for(int j=0; j < gps->nbasis; j++){
		all_bfs.push_back(j);
	}

        for(int i=0;i<octree->size();i++){
                node n = octree->at(i);
                if(n.has_children == false || n.level == OCTREE_DEPTH-1){

			//Get the representative points, which in this case are all points
			vector<point> rep_pts;

			rep_pts = n.ptlst;
			
			/* define a bflist type variable to store basis functions of the node*/
			bflist bflstn;
			
			/*Set the corresponding node id*/
			bflstn.node_id = n.id;

			/*basis function list of node n*/
			vector<bas_func> bfs;

			/*We will use an iterator since we have to remove basis functions from tmp_bfs as we pick them*/
			vector<int>:: iterator j = all_bfs.begin();

			while( j != all_bfs.end()){

				/*Get the jth basis function*/
				int jbas = *j;
	
				/*Define a bas_func type variable to hold the primitives of jbas basis function*/
				bas_func bf;
				bf.bas_id = jbas;

				/*Define a list to hold primitive function lists from each rep_pt*/
				vector<int> pl;

				double func_tst_val = 0.0;

				/*Go through rep_pts, pick significant basis functions from tmp_bfs and store them in bflstn */
				for(int r=0;r<rep_pts.size();r++){

					point rp = rep_pts.at(r);

					/*Define an integer vector to keep a list of primitive functions belonging to jth basis function*/
					vector<int> tmp_pl;

					/*Evalute the value and the gradient of jbas at point rp*/
					double phi, dphidx, dphidy, dphidz;
					pteval(gps, *rp.x, *rp.y, *rp.z, &phi, &dphidx, &dphidy, &dphidz, jbas, &tmp_pl);

					double tmp_func_tst_val = abs(phi+dphidx+dphidy+dphidz);

					if( func_tst_val < tmp_func_tst_val ){
						func_tst_val = tmp_func_tst_val;
						/*Insert the primitive list from rth rep_pt to the end of list*/
						pl.insert(pl.end(), tmp_pl.begin(), tmp_pl.end());
					}					

				}

				if(func_tst_val > gps->DMCutoff){

					/*Remove duplicate prim indices */
				        sort( pl.begin(), pl.end() );
				        pl.erase( unique( pl.begin(), pl.end() ), pl.end() );	

					 /*Save the primitive list and basis functon*/
					bf.prim_list = pl;
					bfs.push_back(bf);
				}
				j++;
			}	

                        /*Save the basis function list only if it is non-empty*/
                        if(bfs.size()>0){
                                bflstn.bfs = bfs;
                                bflst.push_back(bflstn);
                        }

                }
        }

#ifdef OCT_DEBUG
	printf("octree->size(): %i bflst.size(): %i \n", octree->size(), bflst.size());	
#endif

	return bflst;
}

/*This method packs a given set of grid points by using an octree algorithm. The parameters are as follows*/

void pack_grid_pts(grd_pck_strct *gps){

//	printf("num grid points after pruning: %i DMCutoff: %.10e \n", gps->arr_size, gps->DMCutoff);

//	int mpisize, mpirank;

//	MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);

        clock_t start, end;
        double time_octree;
//        double time_bfpf_prescreen;
//        double time_pack_pts;

#ifdef MPIV
	setup_gpack_mpi_1(gps);
#endif

	vector<node> octree;

#ifdef MPIV
    if(gmpi.mpirank==0){
#endif

	start = clock();

	//Generate the octree
        octree = generate_octree(gps->gridx, gps->gridy, gps->gridz, gps->sswt, gps->ss_weight, gps->grid_atm, gps->arr_size, MAX_POINTS_PER_CLUSTER, OCTREE_DEPTH);

	end = clock();

	time_octree = ((double) (end - start)) / CLOCKS_PER_SEC;

	printf("Time for executing octree algorithm: %f s \n", time_octree);

	gps -> time_octree = time_octree;


//----------------- Uncomment for old primitve function approach ----------------
/*
	start = clock();

	//A vector to hold the basis function lists for each node
	vector<bflist> bflst = get_pfbased_basis_function_lists(&octree, gps);	
        //vector<bflist> bflst = get_cfbased_basis_function_lists(&octree, gps);

	end = clock();

	time_bfpf_prescreen = ((double) (end - start)) / CLOCKS_PER_SEC;

	printf("Time for prescreening basis and primitive functions: %f s \n", time_bfpf_prescreen);
*/
//----------------- End uncomment for old primitve function approach ----------------

#ifdef CUDA

//	start = clock();

	vector<node> new_imp_signodes;

	vector<bflist> new_imp_bflst;

//	gpu_get_pfbased_basis_function_lists(&octree, gps);
	int new_imp_total_pts = gpu_get_pfbased_basis_function_lists_new_imp(&octree, gps, &new_imp_signodes, &new_imp_bflst);

//	end = clock();

//	time_bfpf_prescreen = ((double) (end - start)) / CLOCKS_PER_SEC;

//	printf("Time for prescreening basis and primitive functions (new_imp): %f s \n", time_bfpf_prescreen);

//	gps -> time_bfpf_prescreen = time_bfpf_prescreen;

#else
/*
        start = clock();

        //A vector to hold the basis function lists for each node
        vector<bflist> bflst = get_pfbased_basis_function_lists(&octree, gps);  
        //vector<bflist> bflst = get_cfbased_basis_function_lists(&octree, gps);

        end = clock();

        time_bfpf_prescreen = ((double) (end - start)) / CLOCKS_PER_SEC;

        printf("Time for prescreening basis and primitive functions: %.10e s \n", time_bfpf_prescreen);

	gps -> time_bfpf_prescreen = time_bfpf_prescreen;

	start = clock();

	for(int i=0; i<bflst.size();i++){
		bflist bflstn = bflst.at(i);

		int node_id = bflstn.node_id;

		vector<bas_func> bfs = bflstn.bfs;	
		int bfs_count = bfs.size();

#ifdef OCT_DEBUG
		if(bfs_count >0){
			for(int j=0;j<bfs_count;j++){
 				bas_func bf = bfs.at(j);
				vector<int> pf = bf.prim_list;
				for(int k=0; k< pf.size(); k++){
					printf("Node: %i basis function number: %i pf: %i \n", node_id, bf.bas_id, pf.at(k));
				}
			}
			printf("Total number of basis functions for node: %i is %i. \n", node_id, bfs_count);
		}
#endif
	}

//----------------- Uncomment for old primitve function approach ----------------
	
	//At this point, we have binned all the grid points, listed out basis & primitive functions for each bin. 
 	//We shall now prepare them as appropriate for gpu uploading. In doing so, we would neglect bins with 0 basis
	//functions.

	//Pick nodes with a list of basis functions from octree.
	vector<node> signodes;

	int total_pts=0;
	for(int i=0; i<bflst.size();i++){
		int id = bflst.at(i).node_id;
		node n = octree.at(id);
		signodes.push_back(n);
	//	printf("Octree list id: %i Node id: %i Number of grid pts: %i Number of basis functions: %i \n", id, n.id, n.ptlst.size(), bflst.at(i).bfs.size());
		total_pts += n.ptlst.size();
	}

//----------------- End uncomment for old primitve function approach ----------------

	// Save the number of significant nodes
	gps->nbins = signodes.size();

	//printf("Total true grid points after pruning: %i \n", total_pts);	

#ifdef OCT_DEBUG
	// Write files for visualization
	write_vmd_grid(signodes, "pgrid.tcl");
	write_xyz(&signodes, NULL, false, "bgpts.xyz");

	vector<node> oct0;
	oct0.push_back(octree.at(0));
	write_xyz(&oct0, NULL, false, "initgpts.xyz");
#endif

//	Prepare grid points, basis and primitive function lists to send back to gpu.cu
//	gridxb, gridyb, gridzb are 1D arrays keeping binned grid points. Each bin is set to have cluster_size number of 
//	grid points by using padding. Then each array is cluster_size*number_of_nodes long. 
//	Ex. gridxb storing a node with 245 grid points. 246 to 255 will have a dummy value equal to the value of
//	245th location.  
//         *<------------------------ gridxb ---- -----------------
//          <--------------- First bin -------------------> <------
//	   0      1      2     .....                255    256 ...
//	   0      1      2     245     ..           255     0  ...
//        ________________________________________________________
//        |      |      |      |      |      |      |      |
//        | 0.1  | 0.2  | ...  | 0.8  |  0.8 | ...  | 0.8  |  ...
//        |______|______|______|______|______|______|______|______
//
//        Dummy and true values are tracked by dweight integer array. Each location contains 0 or 1 value which stands for
//	dummy and true respectively. 

	//For cpu version, the output arrays sizes should eqaul to total number of grid points
	int grid_out_size = total_pts;
	int grid_out_double_byte_size = sizeof(double)*grid_out_size;
	int grid_out_int_byte_size = sizeof(int)*grid_out_size;		

	printf("cluster_size: %i bflst.size(): %i signodes.size(): %i grid_out_size: %i byte size: %i \n", MAX_POINTS_PER_CLUSTER, bflst.size(), signodes.size(), grid_out_size, grid_out_double_byte_size );	


        double *gridxb, *gridyb, *gridzb, *ssw, *ss_weight;
        int *gridb_atm;

        gridxb = (double*) malloc(grid_out_double_byte_size);
        gridyb = (double*) malloc(grid_out_double_byte_size);
        gridzb = (double*) malloc(grid_out_double_byte_size);
        ssw = (double*) malloc(grid_out_double_byte_size);
        ss_weight = (double*) malloc(grid_out_double_byte_size);
        gridb_atm = (int*) malloc(grid_out_int_byte_size);
        int arr_loc=0;

	int *bin_counter;
	bin_counter = (int*) malloc(sizeof(int)* (signodes.size()+1));
	bin_counter[0] = 0;

	//load grid point info into output arrays
	for(int i=0; i<signodes.size();i++){
		node n = signodes.at(i);

		//Update the bin_counter array
		bin_counter[i+1] = bin_counter[i] + n.ptlst.size();

                for(int j=0;j<n.ptlst.size();j++){
			point p = n.ptlst.at(j);
                        double *x = p.x;
                        double *y = p.y;
                        double *z = p.z;
                        double *sswt = p.sswt;
                        double *weight = p.weight;
                        int *iatm = p.iatm;			

                        gridxb[arr_loc] = *x;
                        gridyb[arr_loc] = *y;
                        gridzb[arr_loc] = *z;
                        ssw[arr_loc] = *sswt;
                        ss_weight[arr_loc] = *weight;
                        gridb_atm[arr_loc] = *iatm;
			arr_loc++;
			
                }

	}

	gps->gridxb = gridxb;
	gps->gridyb = gridyb;
	gps->gridzb = gridzb;
        gps->gridb_sswt = ssw;
        gps->gridb_weight = ss_weight;
	gps->gridb_atm = gridb_atm;
        gps->gridb_count = grid_out_size;
	gps->bin_counter = bin_counter;

	//Get the total counts of basis and primitive function ids
        int tot_bf=0;
        int tot_pf=0;

	for(int i=0;i<bflst.size();i++){

		vector<bas_func> bfs = bflst.at(i).bfs;
		tot_bf += bfs.size();

		for(int j=0;j<bfs.size();j++){
			bas_func bf = bfs.at(j);
			tot_pf += bf.prim_list.size();
		}
	}

	//Save the total number of basis and primitive functions

	gps->nbtotbf = tot_bf;
	gps->nbtotpf = tot_pf;

	printf("octree.size(): %i bflst.size(): %i Total number of basis functions: %i and primitive functions: %i \n", octree.size(), bflst.size(), tot_bf, tot_pf);	

	//Arrays to store basis and primitive function ids 
	//basf and primf arrays store basis and primitive function indices.
	int *basf, *primf;

	basf = (int*) malloc(sizeof(int)*tot_bf);
	primf = (int*) malloc(sizeof(int)*tot_pf);

	//We use the following counters to keep track of basis and primitive functions.
	int *basf_counter, *primf_counter;
	basf_counter = (int*) malloc(sizeof(int)*(bflst.size()+1));
	primf_counter = (int*) malloc(sizeof(int)*(tot_bf+1));

	tot_bf=0;
	tot_pf=0;

	int basf_loc=0;	
	int primf_loc =0;

	//Load data into arrays
	basf_counter[0] = 0;
	primf_counter[0] = 0;
	for(int i=0; i<bflst.size();i++){
		vector<bas_func> bfs = bflst.at(i).bfs;
		tot_bf += bfs.size();
		basf_counter[i+1]=tot_bf;
//		printf("i: %i node: %i nbasis: %i \n", i, bflst.at(i).node_id, bflst.at(i).bfs.size());				
		for(int j=0;j<bfs.size();j++){
			
			bas_func bf = bfs.at(j);
			basf[basf_loc] = bf.bas_id;
			tot_pf += bf.prim_list.size();
			primf_counter[basf_loc+1] = tot_pf;			

//			printf("i: %i j: %i node: %i nbasis: %i bf.bas_id: %i \n", i, j, bflst.at(i).node_id, bfs.size(), bf.bas_id);			

			for(int k=0;k<bf.prim_list.size();k++){
				primf[primf_loc] = bf.prim_list.at(k);
				primf_loc += 1;					
			}

			basf_loc +=1;
		}
		
	}

#ifdef OCT_DEBUG
	for(int i=0; i< (bflst.size());i++){
		int nid = basf_counter[i+1] - basf_counter[i];
		for(int j=basf_counter[i]; j<basf_counter[i+1]; j++){
		//	printf("i: %i nbs: %i bf_cntr[i]: %i bf_cntr[i+1]: %i j: %i bf: %i \n", i, nid, basf_counter[i], basf_counter[i+1], j, basf[j]);
			for(int k=primf_counter[j]; k< primf_counter[j+1]; k++){
				printf("i: %i nbs: %i bf_cntr[i]: %i bf_cntr[i+1]: %i j: %i bf: %i k: %i primf: %i \n", i, nid, basf_counter[i], basf_counter[i+1], j, basf[j], k, primf[k]);
			}
		}

	}
#endif

	gps->basf = basf;
	gps->primf = primf;
	gps->basf_counter = basf_counter;
	gps->primf_counter = primf_counter;
	
//	for(int i=0; i< tot_bf; i++){
//		printf("i: %i basf: %i \n", i, basf[i]);
//	}

        end = clock();

        time_pack_pts = ((double) (end - start)) / CLOCKS_PER_SEC;

        printf("Time to pack points: %f s \n", time_pack_pts);
*/

#ifdef MPIV

//	start = clock();

    }
#endif
	cpu_get_pfbased_basis_function_lists_new_imp(&octree, gps);

#ifdef MPIV
	delete_gpack_mpi();

//	if(gmpi.mpirank==0){

//        end = clock();

//        time_bfpf_prescreen = ((double) (end - start)) / CLOCKS_PER_SEC;

//        printf("Time for prescreening basis and primitive functions (new_imp): %f s \n", time_bfpf_prescreen);

//        gps -> time_bfpf_prescreen = time_bfpf_prescreen;
	
//    }
#endif

#endif

}

/*Prune grid points based on ss weights*/
void get_pruned_grid_ssw(grd_pck_strct *gps_in, grd_pck_strct *gps_out){

	vector<double> gridx_out;
	vector<double> gridy_out;
	vector<double> gridz_out;
	vector<int> grid_atm_out;
	vector<double> sswt_out;
	vector<double> weight_out;

        for(int i=0;i<gps_in->arr_size;i++){
                if(gps_in->ss_weight[i] > gps_in->DMCutoff){

                        gridx_out.push_back(gps_in->gridx[i]);
                        gridy_out.push_back(gps_in->gridy[i]);
                        gridz_out.push_back(gps_in->gridz[i]);
                        grid_atm_out.push_back(gps_in->grid_atm[i]);
                        sswt_out.push_back(gps_in->sswt[i]);
                        weight_out.push_back(gps_in->ss_weight[i]);

                }
        }

	double *arr_gridx_out, *arr_gridy_out, *arr_gridz_out, *arr_sswt_out, *arr_weight_out;
	int *arr_grid_atm_out;

	int dbl_arr_byte_size = sizeof(double)*gridx_out.size();
	int int_arr_byte_size = sizeof(int)*grid_atm_out.size();

	arr_gridx_out = (double*)malloc(dbl_arr_byte_size);
	arr_gridy_out = (double*)malloc(dbl_arr_byte_size);
	arr_gridz_out = (double*)malloc(dbl_arr_byte_size);
	arr_sswt_out = (double*)malloc(dbl_arr_byte_size);
	arr_weight_out = (double*)malloc(dbl_arr_byte_size);
	arr_grid_atm_out = (int*)malloc(int_arr_byte_size);

	copy(gridx_out.begin(), gridx_out.end(), arr_gridx_out);
	copy(gridy_out.begin(), gridy_out.end(), arr_gridy_out);
	copy(gridz_out.begin(), gridz_out.end(), arr_gridz_out);
	copy(sswt_out.begin(), sswt_out.end(), arr_sswt_out);
	copy(weight_out.begin(), weight_out.end(), arr_weight_out);
	copy(grid_atm_out.begin(), grid_atm_out.end(), arr_grid_atm_out);

	gps_out->gridx = arr_gridx_out;
	gps_out->gridy = arr_gridy_out;
	gps_out->gridz = arr_gridz_out;
	gps_out->sswt = arr_sswt_out;
	gps_out->ss_weight = arr_weight_out;
	gps_out->grid_atm = arr_grid_atm_out;
	gps_out->arr_size = gridx_out.size();
}


//#define GPACK_DEBUG

#ifdef CUDA
int gpu_get_pfbased_basis_function_lists_new_imp(vector<node> *octree, grd_pck_strct *gps, vector<node> *signodes, vector<bflist> *bflst){

        double *gridx, *gridy, *gridz, *sswt, *weight;	                 			//Keeps all grid points
        unsigned int *cfweight, *pfweight;   //Holds 1 or 0 depending on the significance of each candidate
	unsigned char *gpweight;
	int *iatm; //**************** Has to be changed into unsigned int later ************

	//get the number of octree leaves 
	unsigned int leaf_count = 0;

#ifdef GPACK_DEBUG
	vector<node> dbg_leaf_nodes; //Store leaves for grid visialization
	vector<node> dbg_signodes;   //Store significant nodes for grid visualization
	vector<int>  dbg_signdidx;    //Keeps track of leaf node indices to remove
	vector<point> dbg_pts; 	     //Keeps all pruned grid points
#endif

        for(int i=0; i<octree -> size();i++){

                node n = octree->at(i);

                if(n.has_children == false || n.level == OCTREE_DEPTH-1){
			leaf_count++;
#ifdef GPACK_DEBUG
			dbg_leaf_nodes.push_back(n);
#endif
		}
	}

#ifdef GPACK_DEBUG
	printf("FILE: %s, LINE: %d, FUNCTION: %s, Total number of leaf nodes: %i \n", __FILE__, __LINE__, __func__, leaf_count);
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

#ifdef GPACK_DEBUG
	unsigned int init_true_gpcount=0;

	for(int i=0; i<leaf_count*MAX_POINTS_PER_CLUSTER; i++){
		if(gpweight[i]>0){
			init_true_gpcount++;
		}
	}

	printf("Total number of true grid points before pruning: %i \n", init_true_gpcount);
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

        printf("Time for prescreening basis and primitive functions (new_imp): Prep GPU input: %f s \n", time_prep_gpu_input);


	start = clock();

        gpu_get_octree_info_new_imp(gridx, gridy, gridz, gps->sigrad2, gpweight, cfweight, pfweight, init_arr_size);

	end = clock();

        time_run_gpu = ((double) (end - start)) / CLOCKS_PER_SEC;

        printf("Time for prescreening basis and primitive functions (new_imp): GPU run: %f s \n", time_run_gpu);

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
	

#ifdef GPACK_DEBUG
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
#ifdef GPACK_DEBUG
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
#ifdef GPACK_DEBUG
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
#ifdef GPACK_DEBUG
			dbg_signodes.push_back(dbg_leaf_nodes.at(i));
#endif

		}
	}

#ifdef GPACK_DEBUG
        printf("FILE: %s, LINE: %d, FUNCTION: %s, Total number of contracted functions from GPU: %i \n", __FILE__, __LINE__, __func__, pcfweight.size());
        printf("FILE: %s, LINE: %d, FUNCTION: %s, Total number of primitive functions from GPU: %i \n", __FILE__, __LINE__, __func__, ppfweight.size());


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
	int pgridinfo_arr_size = pgpweight.size();

	double *apgridx, *apgridy, *apgridz, *apsswt, *apweight;
	int *apgpweight, *apiatm, *apcfweight, *appfweight, *apcf_counter, *appf_counter; 

	apgridx    = (double*) malloc(pgridinfo_arr_size * sizeof(double));
        apgridy    = (double*) malloc(pgridinfo_arr_size * sizeof(double));	
        apgridz    = (double*) malloc(pgridinfo_arr_size * sizeof(double));
        apsswt     = (double*) malloc(pgridinfo_arr_size * sizeof(double));
        apweight   = (double*) malloc(pgridinfo_arr_size * sizeof(double));
	apgpweight = (int*) malloc(pgridinfo_arr_size * sizeof(int));
	apiatm     = (int*) malloc(pgridinfo_arr_size * sizeof(int));
	apcfweight = (int*) malloc(pcfweight.size() * sizeof(int));
	appfweight = (int*) malloc(ppfweight.size() * sizeof(int));
	apcf_counter = (int*) malloc((pcfweight.size() + 1) * sizeof(int));
	appf_counter = (int*) malloc((ppfweight.size() + 1) * sizeof(int));

	copy(pgridx.begin(), pgridx.end(), apgridx);
	copy(pgridy.begin(), pgridy.end(), apgridy);
	copy(pgridz.begin(), pgridz.end(), apgridz);
	copy(psswt.begin(), psswt.end(), apsswt);
	copy(pweight.begin(), pweight.end(), apweight);
	copy(pgpweight.begin(), pgpweight.end(), apgpweight);
	copy(piatm.begin(), piatm.end(), apiatm);
	copy(pcfweight.begin(), pcfweight.end(), apcfweight);
	copy(ppfweight.begin(), ppfweight.end(), appfweight);
	copy(pcf_counter.begin(), pcf_counter.end(), apcf_counter);
	copy(ppf_counter.begin(), ppf_counter.end(), appf_counter);

	int true_pruned_gps=0;
	for(int i=0; i<pgridinfo_arr_size; i++){
		if(apgpweight[i] > 0){

			true_pruned_gps++;
		}
	}


	//Save info into gps struct
	gps->nbins  = pgridinfo_arr_size/MAX_POINTS_PER_CLUSTER;
	gps->gridxb = apgridx;
	gps->gridyb = apgridy;
	gps->gridzb = apgridz;
        gps->gridb_sswt   = apsswt;
        gps->gridb_weight = apweight;
        gps->gridb_atm   = apiatm;
        gps->gridb_count = pgridinfo_arr_size;
	gps->dweight = apgpweight;
	gps->nbtotbf = pcfweight.size();
	gps->nbtotpf = ppfweight.size(); 	
        gps->basf  = apcfweight;
        gps->primf = appfweight;
        gps->basf_counter  = apcf_counter;
        gps->primf_counter = appf_counter;	

        free(gridx);
        free(gridy);
        free(gridz);
        free(gpweight);
        free(cfweight);
        free(pfweight);
        free(sswt);
        free(weight);


	printf("gpu grid pruning: Significant nodes: %i grid points after pruning: %i \n", pgridinfo_arr_size/MAX_POINTS_PER_CLUSTER, true_pruned_gps);
        end = clock();

        time_proc_gpu_output = ((double) (end - start)) / CLOCKS_PER_SEC;

        printf("Time for prescreening basis and primitive functions (new_imp): Process GPU output: %f s \n", time_proc_gpu_output);

}
#endif









//#define CBFPF_DEBUG

void cpu_get_pfbased_basis_function_lists_new_imp(vector<node> *octree, grd_pck_strct *gps){

        double *gridx, *gridy, *gridz, *sswt, *weight;                                          //Keeps all grid points
        unsigned int *cfweight, *pfweight;   //Holds 1 or 0 depending on the significance of each candidate
	unsigned int *bs_tracker;  //Keeps track of bin sizes 
        unsigned char *gpweight;
        int *iatm; //**************** Has to be changed into unsigned int later ************

	unsigned char *tmp_gpweight;
	unsigned int *tmp_cfweight, *tmp_pfweight;
        //get the number of octree leaves 
        unsigned int leaf_count = 0;

#ifdef CBFPF_DEBUG
        vector<node> dbg_leaf_nodes; //Store leaves for grid visialization
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
        printf("FILE: %s, LINE: %d, FUNCTION: %s, Total number of leaf nodes: %i \n", __FILE__, __LINE__, __func__, leaf_count);
	printf("FILE: %s, LINE: %d, FUNCTION: %s, Size for temporary memory allocation: %i \n", __FILE__, __LINE__, __func__, gps->arr_size);
#endif

        unsigned int init_arr_size = gps->arr_size;

        gridx = (double*) malloc(init_arr_size * sizeof(double));
        gridy = (double*) malloc(init_arr_size * sizeof(double));
        gridz = (double*) malloc(init_arr_size * sizeof(double));
        sswt  = (double*) malloc(init_arr_size * sizeof(double));
        weight= (double*) malloc(init_arr_size * sizeof(double));

#ifdef MPIV
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
        double time_prep_gpu_input;
        double time_run_gpu;
        double time_proc_gpu_output;

#ifdef MPIV
	double mpi_prep_time;
	double mpi_run_time;
	double mpi_post_proc_time;
#endif

#ifdef MPIV
        if(gmpi.mpirank == 0){
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
/*        unsigned int init_true_gpcount=0;

        for(int i=0; i<leaf_count*MAX_POINTS_PER_CLUSTER; i++){
                if(gpweight[i]>0){
                        init_true_gpcount++;
                }
        }
*/
        printf("Total number of true grid points before pruning: %i \n", init_arr_size);
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

        time_prep_gpu_input = ((double) (end - start)) / CLOCKS_PER_SEC;

        printf("Time for prescreening basis and primitive functions (new_imp): Prep GPU input: %f s \n", time_prep_gpu_input);


        start = clock();

#ifdef MPIV
	}

	setup_gpack_mpi_2(leaf_count, gridx, gridy, gridz, gps, gpweight, tmp_gpweight, cfweight, tmp_cfweight, pfweight, tmp_pfweight, sswt, weight, iatm, bs_tracker);
#endif

#ifdef MPIV
	if(gmpi.mpirank == 0){
	end = clock();

        mpi_prep_time = ((double) (end - start)) / CLOCKS_PER_SEC;

	printf("Time for prescreening basis and primitive functions (new_imp): MPI broadcast time: %f s \n", mpi_prep_time);		

	start = clock();
	}

#endif

	int bstart, bend;

#ifndef MPIV
	bstart=0;
	bend=leaf_count;
#else
	bstart=gmpi.mpi_binlst[gmpi.mpirank];
	bend=gmpi.mpi_binlst[gmpi.mpirank+1];

#endif
	for(unsigned int i=bstart; i< bend; i++){
		for(unsigned int j=bs_tracker[i]; j<bs_tracker[i+1]; j++){
			cpu_get_primf_contraf_lists_method_new_imp(gridx[j], gridy[j], gridz[j], gps, gpweight, cfweight, pfweight, i, j);	
		}	
	}

#ifdef MPIV
        if(gmpi.mpirank == 0){
        end = clock();

        mpi_run_time = ((double) (end - start)) / CLOCKS_PER_SEC;

        printf("Time for prescreening basis and primitive functions (new_imp): Actual MPI run time: %f s \n", mpi_run_time);
        
        start = clock();
        }

#endif



#ifdef MPIV

	MPI_Barrier(MPI_COMM_WORLD);

        get_slave_primf_contraf_lists(leaf_count, gps, gpweight, tmp_gpweight, cfweight, tmp_cfweight, pfweight, tmp_pfweight, bs_tracker);

        if(gmpi.mpirank == 0){

	end = clock();

	mpi_post_proc_time = ((double) (end - start)) / CLOCKS_PER_SEC;
	printf("Time for prescreening basis and primitive functions (new_imp): MPI post processing time: %f s \n", mpi_post_proc_time);

	gps -> time_bfpf_prescreen = mpi_post_proc_time+mpi_run_time+mpi_prep_time;
#else

        end = clock();

        time_run_gpu = ((double) (end - start)) / CLOCKS_PER_SEC;

        printf("Time for prescreening basis and primitive functions (new_imp): GPU run: %f s \n", time_run_gpu);

        gps -> time_bfpf_prescreen = time_run_gpu;
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

#ifdef CBFPF_DEBUG
        printf("FILE: %s, LINE: %d, FUNCTION: %s, Total number of contracted functions from GPU: %i \n", __FILE__, __LINE__, __func__, pcfweight.size());
        printf("FILE: %s, LINE: %d, FUNCTION: %s, Total number of primitive functions from GPU: %i \n", __FILE__, __LINE__, __func__, ppfweight.size());


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
        int pgridinfo_arr_size = ppt_count;

        double *apgridx, *apgridy, *apgridz, *apsswt, *apweight;
        int *apgpweight, *apiatm, *apcfweight, *appfweight, *apcf_counter, *appf_counter;
	int *apbs_tracker;

        apgridx    = (double*) malloc(pgridinfo_arr_size * sizeof(double));
        apgridy    = (double*) malloc(pgridinfo_arr_size * sizeof(double));
        apgridz    = (double*) malloc(pgridinfo_arr_size * sizeof(double));
        apsswt     = (double*) malloc(pgridinfo_arr_size * sizeof(double));
        apweight   = (double*) malloc(pgridinfo_arr_size * sizeof(double));
        apiatm     = (int*) malloc(pgridinfo_arr_size * sizeof(int));
        apcfweight = (int*) malloc(pcfweight.size() * sizeof(int));
        appfweight = (int*) malloc(ppfweight.size() * sizeof(int));
        apcf_counter = (int*) malloc((pcfweight.size() + 1) * sizeof(int));
        appf_counter = (int*) malloc((ppfweight.size() + 1) * sizeof(int));
	apbs_tracker = (int*) malloc(pbs_tracker.size() * sizeof(int));

        copy(pgridx.begin(), pgridx.end(), apgridx);
        copy(pgridy.begin(), pgridy.end(), apgridy);
        copy(pgridz.begin(), pgridz.end(), apgridz);
        copy(psswt.begin(), psswt.end(), apsswt);
        copy(pweight.begin(), pweight.end(), apweight);
        copy(piatm.begin(), piatm.end(), apiatm);
        copy(pcfweight.begin(), pcfweight.end(), apcfweight);
        copy(ppfweight.begin(), ppfweight.end(), appfweight);
        copy(pcf_counter.begin(), pcf_counter.end(), apcf_counter);
        copy(ppf_counter.begin(), ppf_counter.end(), appf_counter);
	copy(pbs_tracker.begin(), pbs_tracker.end(), apbs_tracker);

//        int true_pruned_gps=0;
//        for(int i=0; i<pgridinfo_arr_size; i++){
//                if(apgpweight[i] > 0){
//
//                        true_pruned_gps++;
//                }
//        }

        //Save info into gps struct
        gps->nbins  = pbs_tracker.size() - 1;
        gps->gridxb = apgridx;
        gps->gridyb = apgridy;
        gps->gridzb = apgridz;
        gps->gridb_sswt   = apsswt;
        gps->gridb_weight = apweight;
        gps->gridb_atm   = apiatm;
        gps->gridb_count = pgridinfo_arr_size;
        gps->nbtotbf = pcfweight.size();
        gps->nbtotpf = ppfweight.size();
        gps->basf  = apcfweight;
        gps->primf = appfweight;
        gps->basf_counter  = apcf_counter;
        gps->primf_counter = appf_counter;
	gps->bin_counter = apbs_tracker;

        printf("gpu grid pruning: Significant nodes: %i grid points after pruning: %i \n", pbs_tracker.size() - 1, pgridinfo_arr_size);

        end = clock();

        time_proc_gpu_output = ((double) (end - start)) / CLOCKS_PER_SEC;

        printf("Time for prescreening basis and primitive functions (new_imp): Process GPU output: %f s \n", time_proc_gpu_output);
#ifdef MPIV
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

	free(tmp_gpweight);
	free(tmp_cfweight);
	free(tmp_pfweight);

}



void cpu_get_primf_contraf_lists_method_new_imp(double gridx, double gridy, double gridz, grd_pck_strct *gps, unsigned char *gpweight, unsigned int *cfweight, unsigned int *pfweight, unsigned int bin_id, unsigned int gid){

        unsigned int sigcfcount=0;

        // relative coordinates between grid point and basis function I.

        for(int ibas=0; ibas<gps->nbasis;ibas++){

		unsigned int nc = (gps->ncenter[ibas])-1;
                unsigned long cfwid = bin_id * gps->nbasis + ibas; //Change here
        	double x1 = gridx - gps->xyz[0+nc*3];
        	double y1 = gridy - gps->xyz[1+nc*3];
        	double z1 = gridz - gps->xyz[2+nc*3];
			

                double x1i, y1i, z1i;
                double x1imin1, y1imin1, z1imin1;
                double x1iplus1, y1iplus1, z1iplus1;

                double phi = 0.0;
                double dphidx = 0.0;
                double dphidy = 0.0;
                double dphidz = 0.0;

		int itypex = gps->itype[0+ibas*3];
                int itypey = gps->itype[1+ibas*3];
                int itypez = gps->itype[2+ibas*3];
                double dist = x1*x1+y1*y1+z1*z1;

                       if ( dist <= gps->sigrad2[ibas]){

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
                               for(int kprim=0; kprim< gps->ncontract[ibas]; kprim++){

                                       unsigned long pfwid = bin_id * gps->nbasis * gps->maxcontract + ibas * gps->maxcontract + kprim; //Change
				       double alpha = gps->aexp[kprim + ibas * gps->maxcontract];
				       double tmp = (gps->dcoeff[kprim + ibas * gps->maxcontract]) * exp( -alpha * dist);

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

#ifdef MPIV

void setup_gpack_mpi_1(grd_pck_strct *gps){

	MPI_Comm_rank(MPI_COMM_WORLD, &gmpi.mpirank);
	MPI_Comm_size(MPI_COMM_WORLD, &gmpi.mpisize);
	MPI_Bcast(&gps->arr_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&gps->nbasis, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&gps->maxcontract, 1, MPI_INT, 0, MPI_COMM_WORLD);

}


void setup_gpack_mpi_2(unsigned int nbins, double *gridx, double *gridy, double *gridz, grd_pck_strct *gps, unsigned char *gpweight, unsigned char *tmp_gpweight, unsigned int *cfweight, unsigned int *tmp_cfweight, unsigned int *pfweight, unsigned int *tmp_pfweight, double *sswt, double *weight, int *iatm, unsigned int *bs_tracker){
	unsigned int tmp_arr[gmpi.mpisize];
	unsigned int *tmp_mpi_binlst;

	tmp_mpi_binlst = (unsigned int*) malloc((gmpi.mpisize+1)*sizeof(unsigned int));
	
        if(gmpi.mpirank == 0){

	//Set array values to zero
	tmp_mpi_binlst[0]=0;
	for(unsigned int i=1;i<gmpi.mpisize+1;i++){
		tmp_mpi_binlst[i]=0;
		tmp_arr[i-1]=0;
	}

	//Set bin count for each cpu
	unsigned int ndist = nbins;
	do{

		for(unsigned int j=0; j<gmpi.mpisize; j++){

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
	for(unsigned int i=1; i<gmpi.mpisize+1; i++){
		tmp_mpi_binlst[i] = tmp_mpi_binlst[i-1] + tmp_arr[i-1];
	}


	}	

        gmpi.mpi_binlst = tmp_mpi_binlst;

	MPI_Bcast(gmpi.mpi_binlst, gmpi.mpisize+1, MPI_INT, 0, MPI_COMM_WORLD);
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


void get_slave_primf_contraf_lists(unsigned int nbins, grd_pck_strct *gps, unsigned char *gpweight, unsigned char *tmp_gpweight, unsigned int *cfweight, unsigned int *tmp_cfweight, unsigned int *pfweight, unsigned int *tmp_pfweight, unsigned int *bs_tracker){

        MPI_Status status;
	clock_t start, end;

/*		unsigned int bstart=gmpi.mpi_binlst[gmpi.mpirank];
		unsigned int bend=gmpi.mpi_binlst[gmpi.mpirank+1];

		for(unsigned int j=0; j< bstart; j++){
			for(unsigned int k=bs_tracker[j]; k<bs_tracker[j+1]; k++){
				gpweight[k]=0;
			}
		}

                for(unsigned int j=bend; j< nbins; j++){
                        for(unsigned int k=bs_tracker[j]; k<bs_tracker[j+1]; k++){
                                gpweight[k]=0;
                        }
                }		
*/
        if(gmpi.mpirank != 0){

                        MPI_Send(gpweight, gps->arr_size, MPI_UNSIGNED_CHAR, 0, gmpi.mpirank+600, MPI_COMM_WORLD);
                        MPI_Send(cfweight, nbins*gps->nbasis, MPI_INT, 0, gmpi.mpirank+700, MPI_COMM_WORLD);
                        MPI_Send(pfweight, nbins*gps->nbasis*gps->maxcontract, MPI_INT, 0, gmpi.mpirank+800, MPI_COMM_WORLD);

        }else{

		start=clock();

		for(unsigned int i=1; i< gmpi.mpisize; i++){

			MPI_Recv(tmp_gpweight, gps->arr_size, MPI_UNSIGNED_CHAR, i, i+600, MPI_COMM_WORLD, &status);
			MPI_Recv(tmp_cfweight, nbins*gps->nbasis, MPI_INT, i, i+700, MPI_COMM_WORLD, &status);
			MPI_Recv(tmp_pfweight, nbins*gps->nbasis*gps->maxcontract, MPI_INT, i, i+800, MPI_COMM_WORLD, &status);

		        unsigned int bstart=gmpi.mpi_binlst[i];
			unsigned int bend=gmpi.mpi_binlst[i+1];	

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

/*                	for(int j=0; j<nbins*gps->nbasis; j++){
                        	cfweight[j] += tmp_cfweight[j];
                	}

                	for(int j=0; j<nbins*gps->nbasis*gps->maxcontract; j++){
                        	pfweight[j] += tmp_pfweight[j];
                	}
*/
		}

		end = clock();

		printf("Time for running through data: %f \n",  ((double) (end - start)) / CLOCKS_PER_SEC);

        }

        MPI_Barrier(MPI_COMM_WORLD);

}

void delete_gpack_mpi(){

		free(gmpi.mpi_binlst);
}

#endif

