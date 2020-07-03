#include <iostream>
#include <vector>
#include <algorithm>
#include <stdio.h>
#include "gpack_type.h"

#ifdef MPIV
#include <mpi.h>
#endif

using namespace std;

// Write octree into the debug file
//#define WRITE_OCTREE

/*Struct to hold grid point value pointers*/
struct point{
        double *x;
        double *y;
        double *z;
	double *sswt;
	double *weight;
	int *iatm;
};


/*An struct representing a node. level=0 with has_children=true is the root.
 * parent=false nodes are leaves */
struct node{

        int level; /*node level*/
        int id; /*A unique id for each node*/
        int parent; /*parent level, -1 means the root*/
        bool has_children; /*Used to differentiate parent vs leaf nodes*/

        /*Node boundaries*/
        double xmin; /*x lower boundary*/
        double xmax; /*x upper boundary*/
        double ymin; /*y lower boundary*/
        double ymax; /*y upper boundary*/
        double zmin; /*z lower boundary*/
        double zmax; /*z upper boundary*/

        vector<point> ptlst; /*Keeps a list of points belonging to each node*/
};

vector<node> generate_octree(_gpack_type gps, int bin_size, int max_lvl);

