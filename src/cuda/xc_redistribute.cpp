/*
  !---------------------------------------------------------------------!
  ! Written by Madu Manathunga on 12/04/2020                            !
  !                                                                     ! 
  ! Copyright (C) 2020-2021 Merz lab                                    !
  ! Copyright (C) 2020-2021 Götz lab                                    !
  !                                                                     !
  ! This Source Code Form is subject to the terms of the Mozilla Public !
  ! License, v. 2.0. If a copy of the MPL was not distributed with this !
  ! file, You can obtain one at http://mozilla.org/MPL/2.0/.            !
  !_____________________________________________________________________!

  !---------------------------------------------------------------------!
  ! This source file contains functions required for QUICK multi GPU    !
  ! implementation.                                                     !
  !---------------------------------------------------------------------!
*/
#ifdef CUDA_MPIV
#include <iostream>
#include <cmath>
#include <cstring>
#include <mpi.h>
#include "xc_redistribute.h"
using namespace std;

// Distribution matrix for load balancing prior to sswder calculation.
int** distMatrix=NULL;
int* ptcount=NULL;

//--------------------------------------------------------
// Function to redistribute XC quadrature points among GPUs
// prior to sswder calculation. Sends back the adjustment to 
// size of the arrays. 
//--------------------------------------------------------
int getAdjustment(int mpisize, int mpirank, int count){

  bool master=false;
  if(mpirank == 0) master = true;

  // define arrays
  int *residuals  = new int[mpisize]{0};
  ptcount    = new int[mpisize]{0};

  distMatrix=new int*[mpisize];
  for(int i=0; i<mpisize; ++i) distMatrix[i] = new int[mpisize]{0};

  ptcount[mpirank]=count;

  MPI_Barrier(MPI_COMM_WORLD);
  // broadcast ptcount array
  for(int i=0; i<mpisize; ++i) MPI_Bcast(&ptcount[i], 1, MPI_INT, i, MPI_COMM_WORLD);

#ifdef DEBUG 
  if(master) cout << "mpirank= "<<mpirank << " init array:" << endl;
#endif

  for(int i=0; i<mpisize; ++i)
    cout << ptcount[i] << " ";
  cout << endl;

  //find sum
  int sum=0;

  for(int i=0;i<mpisize;++i){
    sum += ptcount[i]; 
  }

  int average = (int) sum/mpisize;   

#ifdef DEBUG
  if(master) cout << "mpirank= "<< mpirank << " sum= " << sum << " average= " << average << endl;  

  if(master) cout << "mpirank= "<< mpirank << " discrepencies:" << endl;
#endif  

  for(int i=0; i<mpisize; ++i)
    residuals[i]=ptcount[i]-average;
#ifdef DEBUG
  if(master){
    for(int i=0; i<mpisize; ++i) cout << residuals[i] << " ";
    cout << endl; 

    cout << "mpirank= "<< mpirank << " distributing evenly:" << endl;
  }
#endif

  bool done=false;

  for(int i=0; i<mpisize; ++i){

    if(residuals[i]>0){
      int toDist=residuals[i];
      for(int j=0; j<mpisize; ++j){
         if(residuals[j] < 0){
           if(abs(residuals[j]) >= residuals[i]){
             
             distMatrix[i][j]=residuals[i];
             residuals[j] +=residuals[i];
             residuals[i] = 0;
             break;
           }else{
             distMatrix[i][j]= abs(residuals[j]);
             residuals[i] += residuals[j];
             residuals[j] = 0;
             
           }
         }

      }

    }
  }

#ifdef DEBUG
  if(master){
    for(int i=0; i<mpisize; ++i) cout << residuals[i] << " ";
    cout << endl;

    for(int i=0; i<mpisize; ++i){
      cout << "[" << i << "] ";
      for(int j=0; j<mpisize; ++j) cout << distMatrix[i][j] << " ";
      cout << endl;
    }

   cout << "mpirank= "<< mpirank << " distributing the remainder" << endl;
 }
#endif

// add the remainder to positive ones

  for(int i=0; i<mpisize; ++i){
    if(residuals[i]>1){

      for(int j=0; j<mpisize; ++j){
         if(residuals[j] == 0){
           ++residuals[j];
           --residuals[i];
           ++distMatrix[i][j];
         }
         if(residuals[i] == 1) break;
      }

    }
  }
 
#ifdef DEBUG
  if(master){
    for(int i=0; i<mpisize; ++i) cout << residuals[i] << " ";
    cout << endl;

    for(int i=0; i<mpisize; ++i){
      cout << "[" << i << "] ";
      for(int j=0; j<mpisize; ++j) cout << distMatrix[i][j] << " ";
      cout << endl;
    }

   cout << "mpirank= "<< mpirank << " Reevaulating the strategy:" << endl;
 }
#endif
// Reevaluate the distribution strategy
  for(int i=0; i<mpisize; ++i){
    for(int j=0; j<mpisize; ++j){
      if(distMatrix[i][j] > 0){
        // check if receiver is sending to someone else
        for(int k=0; k<mpisize; ++k){
          if(distMatrix[j][k] > 0){
            if(distMatrix[j][k] >= distMatrix[i][j]){ 
              // if the receiver is sending a greater or equal amount that it receives
              distMatrix[j][k] = distMatrix[j][k] - distMatrix[i][j];
              distMatrix[i][k] += distMatrix[i][j];
              distMatrix[i][j] = 0;
            }else{
              distMatrix[i][k] += distMatrix[j][k];
              distMatrix[i][j] -= distMatrix[j][k];
              distMatrix[j][k] = 0;
            }
          }
          if(distMatrix[i][j] == 0) break;
        }
      }
    }
  }

  if(master){
    for(int i=0; i<mpisize; ++i){
      cout << "[" << i << "] ";
      for(int j=0; j<mpisize; ++j) cout << distMatrix[i][j] << " ";
      cout << endl;
    }
  }

  // Row sum of distMatrix tells what a paticular rank looses whereas coulmn sum tells what it gains
  int loss=0, gain=0;

  for(int i=0;i<mpisize;++i) loss += distMatrix[mpirank][i];
  for(int i=0;i<mpisize;++i) gain += distMatrix[i][mpirank];

#ifdef DEBUG
  if(master) cout << "mpirank= " << mpirank<< " net gain= "<< gain-loss << " adjusted size= "<< count-gain-loss << endl;
#endif

  // deallocate memory
  delete [] residuals;

  return gain-loss;

}

//--------------------------------------------------------
// Function to redistribute XC quadrature points among GPUs
// prior to sswder calculation. 
//--------------------------------------------------------
void sswderRedistribute(int mpisize, int mpirank, int count, int ncount, 
  double *gridx, double *gridy, double *gridz, double *exc, double *quadwt, int *gatm,
  double *ngridx, double *ngridy, double *ngridz, double *nexc, double *nquadwt, int *ngatm ){

  MPI_Status status;
  bool master=false;
  if(mpirank == 0) master = true;


/*  if(master){
    cout << "Printing initial arrays:" << endl;
    for(int i=0;i<count;++i)
      cout << "mpirank= " << mpirank << " i= " << i << " x= " << gridx[i] << " y= " << gridy[i] << " z= " << gridz[i]
      << " exc= " << exc[i] << " quadwt= " << quadwt[i] << " gatm= " << gatm[i] << endl;
  }
*/

  int arrsize = ncount > count ? count : ncount;
  size_t bytesize = arrsize * sizeof(double);  

  // copy existing data
  memcpy(ngridx, gridx, bytesize);
  memcpy(ngridy, gridy, bytesize);
  memcpy(ngridz, gridz, bytesize);
  memcpy(nexc, exc, bytesize);
  memcpy(nquadwt, quadwt, bytesize);
  memcpy(ngatm, gatm, arrsize * sizeof(int));

  // record senders and receivers point counts during the transfer
  int *sptcount  = new int[mpisize];
  int *rptcount  = new int[mpisize];
  memcpy(sptcount, ptcount, mpisize * sizeof(int));
  memcpy(rptcount, ptcount, mpisize * sizeof(int));

  // go through the distribution matrix and transfer data
  for(int i=0;i<mpisize;++i){
    int send_total=0;
    for(int j=0;j<mpisize;++j) send_total += distMatrix[i][j];

    if(send_total>0){
      sptcount[i] -= send_total;
 
      for(int j=0;j<mpisize;++j){
        int send_amount=distMatrix[i][j];
        if(send_amount > 0){

          if(mpirank == i){
            MPI_Send(&gridx[sptcount[i]], send_amount, MPI_DOUBLE, j, i+1, MPI_COMM_WORLD);
            MPI_Send(&gridy[sptcount[i]], send_amount, MPI_DOUBLE, j, i+2, MPI_COMM_WORLD);          
            MPI_Send(&gridz[sptcount[i]], send_amount, MPI_DOUBLE, j, i+3, MPI_COMM_WORLD);
            MPI_Send(&exc[sptcount[i]], send_amount, MPI_DOUBLE, j, i+4, MPI_COMM_WORLD);
            MPI_Send(&quadwt[sptcount[i]], send_amount, MPI_DOUBLE, j, i+5, MPI_COMM_WORLD);
            MPI_Send(&gatm[sptcount[i]], send_amount, MPI_INT, j, i+6, MPI_COMM_WORLD);
          }

          if(mpirank == j){
            MPI_Recv(&ngridx[rptcount[j]], send_amount, MPI_DOUBLE, i, i+1, MPI_COMM_WORLD, &status);                 
            MPI_Recv(&ngridy[rptcount[j]], send_amount, MPI_DOUBLE, i, i+2, MPI_COMM_WORLD, &status);
            MPI_Recv(&ngridz[rptcount[j]], send_amount, MPI_DOUBLE, i, i+3, MPI_COMM_WORLD, &status);
            MPI_Recv(&nexc[rptcount[j]], send_amount, MPI_DOUBLE, i, i+4, MPI_COMM_WORLD, &status);
            MPI_Recv(&nquadwt[rptcount[j]], send_amount, MPI_DOUBLE, i, i+5, MPI_COMM_WORLD, &status);
            MPI_Recv(&ngatm[rptcount[j]], send_amount, MPI_INT, i, i+6, MPI_COMM_WORLD, &status);
          }

          sptcount[i] += send_amount;
          rptcount[j] += send_amount;
        }
      }
    } 
  } 

  
/*  if(mpirank == 1){ 
    cout << "Printing final arrays:" << endl;
    for(int i=0;i<ncount;++i) 
      cout << "mpirank= " << mpirank << " i= " << i << " x= " << ngridx[i] << " y= " << ngridy[i] << " z= " << ngridz[i]
      << " exc= " << nexc[i] << " nquadwt= " << nquadwt[i] << " ngatm= " << ngatm[i] << endl;
  }
*/ 
  delete [] sptcount;
  delete [] rptcount;
  delete [] ptcount;
  for(int i=0; i<mpisize; ++i) delete [] distMatrix[i];
  delete [] distMatrix;  

}


#endif
