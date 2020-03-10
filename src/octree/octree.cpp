
/* Written by Madu Manathunga 10/17/2019*/

#include "octree.h"

void get_boundaries(vector<point> *ptlst, double *xmin, double *xmax, double *ymin, double *ymax, double *zmin, double *zmax){

        point pt0= ptlst->at(0);
        double *x0 = (double*) pt0.x;
	double *y0 = (double*) pt0.y;
	double *z0 = (double*) pt0.z;

	*xmin = *x0;
	*ymin = *y0;
        *zmin = *z0;

        *xmax = *x0;
        *ymax = *y0;
        *zmax = *z0;

        for(int i=1;i<ptlst->size();i++){

		point pt = ptlst->at(i);
		double *x = (double*) pt.x;
		double *y = (double*) pt.y;
		double *z = (double*) pt.z;

                if(*xmin > *x){
                        *xmin = *x;
                }
		if(*xmax < *x){
			*xmax = *x;
		}

                if(*ymin > *y){
                        *ymin = *y;
                }
		if(*ymax < *y){
			*ymax = *y;
		}

                if(*zmin > *z){
                        *zmin = *z;
                }
		if(*zmax < *z){
                        *zmax = *z;
                }

        }	

        *xmin -= 1.0;
        *ymin -= 1.0;
        *zmin -= 1.0;

        *xmax += 1.0;
        *ymax += 1.0;
        *zmax += 1.0;

}

void generate_gridpt_list(double *arrx, double *arry, double *arrz, double *sswt, double *weight, int *iatm, int count, vector<point> *ptlst){

	for(int i=0;i<count;i++){
		point pt;
		pt.x = &arrx[i];
		pt.y = &arry[i];
		pt.z = &arrz[i];
		pt.sswt = &sswt[i];
		pt.weight = &weight[i];	
		pt.iatm = &iatm[i];		

		ptlst->push_back(pt);
	}

}

/*This method distributes parent's grid points to a child node*/
/*Takes parent grid point list and new node pointers as arguments*/
void distribute_grid_pts(vector<point> *ptlst, node *n){

	double xmin = n->xmin;
	double xmax = n->xmax;
	double ymin = n->ymin;
	double ymax = n->ymax;
	double zmin = n->zmin;
	double zmax = n->zmax;

	/*New point list for node n*/
	vector<point> nptlst;

	for(int i=0; i < (ptlst->size()); i++){
		point p = ptlst->at(i);
		double *x = (double*) p.x;
		double *y = (double*) p.y;
		double *z = (double*) p.z;

		/*Push grid point into new list if it is within bounds*/
		if( (*x >= xmin) and (*x < xmax) and (*y >= ymin) and (*y < ymax) and (*z >= zmin) and (*z <zmax) ){
			nptlst.push_back(p);
		}
	}

	/*Fianlly, store the point list in node*/
	n->ptlst = nptlst;

}

/*For a given series of grid points, this method will generate an octree and return it as a vector of nodes.*/
/* arrx, arry, arrz are pointers to grid point arrays and count is their size. sswt and weight are grid point
 properties. bin_size is the maximum amount of grid points for a node. max_lvl is the depth of tree.*/
vector<node> generate_octree(double *arrx, double *arry, double *arrz, double *sswt, double *weight, int *iatm, int count, int bin_size, int max_lvl){

	/*Organize the coordinates into point type*/
	vector<point> ptlst;

	generate_gridpt_list(arrx, arry, arrz, sswt, weight, iatm, count, &ptlst);

	/*Calculate the boundaries of the grid*/
	double xmin, ymin, zmin, xmax, ymax, zmax;

	get_boundaries(&ptlst, &xmin, &xmax, &ymin, &ymax, &zmin, &zmax);	

#ifdef OCT_DEBUG	
	printf("minX: %f, maxX: %f, minY: %f, maxY: %f, minZ: %f, maxZ: %f \n",xmin,xmax,ymin,ymax,zmin,zmax);
#endif
	/*Create the octree in terms of a node list*/
	vector<node> octree; /*We would store ONLY SIGNIFICANT NODES (i.e. nodes with at least one grid point) in this vector as follows.*/

	/*lvl 1 <--------- lvl 2 -----------------> <--
          root  <--------- First octet -----------> <--
	_______________________________________________
	|      |      |      |      |      |      |
	|  0   |   0  |  1   |  2   | ...  |  7   | ...
	|______|______|______|______|______|______|____

	To efficient access the elements of this vector, we will make use of the node counter described below. */ 

	/*A counter to assign a unique id to each node*/
	int id = 0;

	/*Create the root node and set properties*/
	node root;
	root.level = 0;
	root.id = id;
	root.parent = -1;
	root.ptlst = ptlst;
	root.xmin = xmin;
	root.ymin = ymin;
	root.zmin = zmin;
	root.xmax = xmax;
	root.ymax = ymax;
	root.zmax = zmax;

	if(root.ptlst.size() > bin_size){
		root.has_children = true;		
	}else{
		root.has_children = false;
	}

	/*Push root to octree*/
	octree.push_back(root);

	/*List to keep track of how many non-empty nodes upto each level. This helps us to easily map the octree
 	vector and access a set of nodes at a given level. */
	vector<int> lvl_node_counter; /*The structure is shown below. */

	/*   0      1      2      3     ....  max_lvl    
	  ___________________________________________
	  |      |      |      |      |      |      |
	  |  1   |   9  |  73  | 585  | ...  | ...  |
	  |______|______|______|______|______|______|
	
	value at position 1 indicates how many nodes upto 1st level and so on. */
	
	/*Update lvl_node_counter for root*/
	lvl_node_counter.push_back(1);

	/*First loop goes through each level of the octree*/
	for(int i=0;i<max_lvl;i++){

		int nlvls; /*Holds the number of levels*/
		int lvlstart; /*Holds the starting point of current level nodes in octree*/
		int lvlend; /*Holds the end point of current level nodes in octree*/
		int nnodes_at_lvl; /*Holds the number of nodes at the last level*/
		int child_count = 0; /*Keeps track of number of children generated from current level*/

		if(i==0){

			nlvls = 1;
			lvlstart = 0;
			lvlend = 1;
			nnodes_at_lvl = 1;

		}else{

			nlvls = lvl_node_counter.size(); /*May be substituted by the value of i?? */
			lvlstart = lvl_node_counter.at(nlvls-2);
			lvlend = lvl_node_counter.at(nlvls-1);
			nnodes_at_lvl = lvl_node_counter.at(nlvls-1) - lvl_node_counter.at(nlvls-2);
		
		}

#ifdef OCT_DEBUG
		printf("i: %i nlvls: %i lvlstart: %i lvlend: %i nnodes_at_lvl: %i \n", i, nlvls, lvlstart, lvlend ,nnodes_at_lvl);
#endif

		/*This loops goes through each node at a given level*/
		for(int j=lvlstart;j<lvlend;j++){

			node n = octree.at(j);

			if(n.has_children == true){

				double xmid = (n.xmax+n.xmin)/2;
				double ymid = (n.ymax+n.ymin)/2;
				double zmid = (n.zmax+n.zmin)/2;

				/*We now label new bins as follows*/
				/*Rear, B,T,L,R stands for bottom, top, L and right respectively*/
				/********************************
				*		*		*
 				*     (RTL,3)	*    (RTR,2)	*
 				*		*		*
				*********************************
 				*		*		*
 				*     (RBL,0)	*    (RBR,1)	*
 				*		*		*
 				*********************************/

				/*Front, B,T,L,R stands for bottom, top, L and right respectively*/
				/********************************
				*               *               *
				*    (FTL,7)    *    (FTR,6)	*
				*               *		*
				*********************************
				*		*		*
				*     (FBL,4)	*    (FBR,5)	*
				*		*		*               
				*********************************/ 

				/*Define a new node and set temporary boundaries*/
				node nnew;
				nnew.xmin = n.xmin;
				nnew.ymin = n.ymin;
				nnew.zmin = n.zmin;
				nnew.xmax = n.xmax;
				nnew.ymax = n.ymax;
				nnew.zmax = n.zmax;
				nnew.parent = n.id;			
				nnew.level = n.level+1;

				/*Split each node into an octet*/
				for(int k=0; k<8;k++){

					node nk = nnew;
				
					switch(k){
						case 0: 
							nk.xmax = xmid;
							nk.zmax = zmid;
							nk.ymin = ymid;
							
							break;
						case 1:
							nk.xmin = xmid;
							nk.zmax = zmid;
							nk.ymin = ymid;
							break;
						case 2:
							nk.xmin = xmid;
							nk.zmin = zmid;
							nk.ymin = ymid;
							break;
						case 3:
							nk.xmax = xmid;
							nk.zmin = zmid;
							nk.ymin = ymid;
							break;
						case 4: 
							nk.xmax = xmid;
							nk.zmax = zmid;
							nk.ymax = ymid;
							break;
						case 5:
							nk.xmin = xmid;
							nk.zmax = zmid;
							nk.ymax = ymid;
							break;
						case 6:
							nk.xmin = xmid;
							nk.zmin = zmid;
							nk.ymax = ymid;
							break;
						case 7:
							nk.xmax = xmid;
							nk.zmin = zmid;
							nk.ymax = ymid;
							break;
					}

					/*Load the grid points into node*/
					distribute_grid_pts(&(n.ptlst), &nk);

					/*Get the grid point count of new node*/
					int numpts = nk.ptlst.size();

					/*Set if node should have children*/					
					if(numpts > bin_size){
						nk.has_children = true;
					}else{
						nk.has_children = false;
					}

#ifdef OCT_DEBUG
                                        printf("i: %i j: %i k: %i xmin: %f xmax: %f ymin: %f ymax: %f zmin: %f zmax: %f numpts: %i \n", i, j, k, nk.xmin
                                        , nk.xmax, nk.ymin, nk.ymax, nk.zmin, nk.zmax, numpts);
#endif

					/*Assign an id to the node ,update the counter and insert the
  					 node into octree only if it has grid points*/
					if (numpts > 0 ){
						id++;
						nk.id = id;	
						octree.push_back(nk);
						child_count++;
					}
				}
			}
		}

#ifdef OCT_DEBUG
		printf("i: %i lvl_node_counter.at(i): %i child_count: %i\n", i, lvl_node_counter.at(i), child_count);
#endif

		/*Update the lvl_node_counter for new generation*/
		lvl_node_counter.push_back(lvl_node_counter.back()+child_count);

	}

	return octree;

}

