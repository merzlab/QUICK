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
#if defined(MPIV_GPU)
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
int getAdjustment(MPI_Comm mpi_comm, int mpi_comm_size, int mpi_comm_rank, int count){

  bool master=false;
  if(mpi_comm_rank == 0) master = true;

  // define arrays
  int *residuals  = new int[mpi_comm_size];
  ptcount    = new int[mpi_comm_size];

  memset(residuals,0, sizeof(int)*mpi_comm_size);
  memset(ptcount,0, sizeof(int)*mpi_comm_size);

  distMatrix=new int*[mpi_comm_size];
  for(int i=0; i<mpi_comm_size; ++i) {
    distMatrix[i] = new int[mpi_comm_size];
    memset(distMatrix[i],0, sizeof(int)*mpi_comm_size);
  }

  ptcount[mpi_comm_rank]=count;

  MPI_Barrier(mpi_comm);
  // broadcast ptcount array
  for(int i=0; i<mpi_comm_size; ++i) MPI_Bcast(&ptcount[i], 1, MPI_INT, i, mpi_comm);

#ifdef DEBUG 
  if(master) cout << "mpi_comm_rank= "<<mpi_comm_rank << " init array:" << endl;

  for(int i=0; i<mpi_comm_size; ++i)
    cout << ptcount[i] << " ";
  cout << endl;

#endif
  //find sum
  int sum=0;

  for(int i=0;i<mpi_comm_size;++i){
    sum += ptcount[i]; 
  }

  int average = (int) sum/mpi_comm_size;   

#ifdef DEBUG
  if(master) cout << "mpi_comm_rank= "<< mpi_comm_rank << " sum= " << sum << " average= " << average << endl;  

  if(master) cout << "mpi_comm_rank= "<< mpi_comm_rank << " discrepencies:" << endl;
#endif  

  for(int i=0; i<mpi_comm_size; ++i)
    residuals[i]=ptcount[i]-average;
#ifdef DEBUG
  if(master) cout << "mpi_comm_rank= "<< mpi_comm_rank << " sum= " << sum << " average= " << average << endl;  

  if(master) cout << "mpi_comm_rank= "<< mpi_comm_rank << " discrepencies:" << endl;
  
  for(int i=0; i<mpi_comm_size; ++i)
    residuals[i]=ptcount[i]-average;

  if(master){
    for(int i=0; i<mpi_comm_size; ++i) cout << residuals[i] << " ";
    cout << endl; 

    cout << "mpi_comm_rank= "<< mpi_comm_rank << " distributing evenly:" << endl;
  }
#endif

  bool done=false;

  for(int i=0; i<mpi_comm_size; ++i){

    if(residuals[i]>0){
      int toDist=residuals[i];
      for(int j=0; j<mpi_comm_size; ++j){
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
    for(int i=0; i<mpi_comm_size; ++i) cout << residuals[i] << " ";
    cout << endl;

    for(int i=0; i<mpi_comm_size; ++i){
      cout << "[" << i << "] ";
      for(int j=0; j<mpi_comm_size; ++j) cout << distMatrix[i][j] << " ";
      cout << endl;
    }

   cout << "mpi_comm_rank= "<< mpi_comm_rank << " distributing the remainder" << endl;
 }
#endif

// add the remainder to positive ones

  for(int i=0; i<mpi_comm_size; ++i){
    if(residuals[i]>1){

      for(int j=0; j<mpi_comm_size; ++j){
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
    for(int i=0; i<mpi_comm_size; ++i) cout << residuals[i] << " ";
    cout << endl;

    for(int i=0; i<mpi_comm_size; ++i){
      cout << "[" << i << "] ";
      for(int j=0; j<mpi_comm_size; ++j) cout << distMatrix[i][j] << " ";
      cout << endl;
    }

   cout << "mpi_comm_rank= "<< mpi_comm_rank << " Reevaulating the strategy:" << endl;
 }
#endif
// Reevaluate the distribution strategy
  for(int i=0; i<mpi_comm_size; ++i){
    for(int j=0; j<mpi_comm_size; ++j){
      if(distMatrix[i][j] > 0){
        // check if receiver is sending to someone else
        for(int k=0; k<mpi_comm_size; ++k){
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

#ifdef DEBUG
  if(master){
    for(int i=0; i<mpi_comm_size; ++i){
      cout << "[" << i << "] ";
      for(int j=0; j<mpi_comm_size; ++j) cout << distMatrix[i][j] << " ";
      cout << endl;
    }
  }
#endif

  // Row sum of distMatrix tells what a paticular rank looses whereas coulmn sum tells what it gains
  int loss=0, gain=0;

  for(int i=0;i<mpi_comm_size;++i) loss += distMatrix[mpi_comm_rank][i];
  for(int i=0;i<mpi_comm_size;++i) gain += distMatrix[i][mpi_comm_rank];

#ifdef DEBUG
  if(master) cout << "mpi_comm_rank= " << mpi_comm_rank<< " net gain= "<< gain-loss << " adjusted size= "<< count-gain-loss << endl;
#endif

  // deallocate memory
  delete [] residuals;

  return gain-loss;

}

//--------------------------------------------------------
// Function to redistribute XC quadrature points among GPUs
// prior to sswder calculation. 
//--------------------------------------------------------
void sswderRedistribute(MPI_Comm mpi_comm, int mpi_comm_size, int mpi_comm_rank, int count, int ncount, 
  double *gridx, double *gridy, double *gridz, double *exc, double *quadwt, int *gatm,
  double *ngridx, double *ngridy, double *ngridz, double *nexc, double *nquadwt, int *ngatm ){

  MPI_Status status;
  bool master=false;
  if(mpi_comm_rank == 0) master = true;


/*  if(master){
    cout << "Printing initial arrays:" << endl;
    for(int i=0;i<count;++i)
      cout << "mpi_comm_rank= " << mpi_comm_rank << " i= " << i << " x= " << gridx[i] << " y= " << gridy[i] << " z= " << gridz[i]
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
  int *sptcount  = new int[mpi_comm_size];
  int *rptcount  = new int[mpi_comm_size];
  memcpy(sptcount, ptcount, mpi_comm_size * sizeof(int));
  memcpy(rptcount, ptcount, mpi_comm_size * sizeof(int));

  // go through the distribution matrix and transfer data
  for(int i=0;i<mpi_comm_size;++i){
    int send_total=0;
    for(int j=0;j<mpi_comm_size;++j) send_total += distMatrix[i][j];

    if(send_total>0){
      sptcount[i] -= send_total;
 
      for(int j=0;j<mpi_comm_size;++j){
        int send_amount=distMatrix[i][j];
        if(send_amount > 0){

          if(mpi_comm_rank == i){
            MPI_Send(&gridx[sptcount[i]], send_amount, MPI_DOUBLE, j, i+1, mpi_comm);
            MPI_Send(&gridy[sptcount[i]], send_amount, MPI_DOUBLE, j, i+2, mpi_comm);          
            MPI_Send(&gridz[sptcount[i]], send_amount, MPI_DOUBLE, j, i+3, mpi_comm);
            MPI_Send(&exc[sptcount[i]], send_amount, MPI_DOUBLE, j, i+4, mpi_comm);
            MPI_Send(&quadwt[sptcount[i]], send_amount, MPI_DOUBLE, j, i+5, mpi_comm);
            MPI_Send(&gatm[sptcount[i]], send_amount, MPI_INT, j, i+6, mpi_comm);
          }

          if(mpi_comm_rank == j){
            MPI_Recv(&ngridx[rptcount[j]], send_amount, MPI_DOUBLE, i, i+1, mpi_comm, &status);                 
            MPI_Recv(&ngridy[rptcount[j]], send_amount, MPI_DOUBLE, i, i+2, mpi_comm, &status);
            MPI_Recv(&ngridz[rptcount[j]], send_amount, MPI_DOUBLE, i, i+3, mpi_comm, &status);
            MPI_Recv(&nexc[rptcount[j]], send_amount, MPI_DOUBLE, i, i+4, mpi_comm, &status);
            MPI_Recv(&nquadwt[rptcount[j]], send_amount, MPI_DOUBLE, i, i+5, mpi_comm, &status);
            MPI_Recv(&ngatm[rptcount[j]], send_amount, MPI_INT, i, i+6, mpi_comm, &status);
          }

          sptcount[i] += send_amount;
          rptcount[j] += send_amount;
        }
      }
    } 
  } 

  
/*  if(mpi_comm_rank == 1){ 
    cout << "Printing final arrays:" << endl;
    for(int i=0;i<ncount;++i) 
      cout << "mpi_comm_rank= " << mpi_comm_rank << " i= " << i << " x= " << ngridx[i] << " y= " << ngridy[i] << " z= " << ngridz[i]
      << " exc= " << nexc[i] << " nquadwt= " << nquadwt[i] << " ngatm= " << ngatm[i] << endl;
  }
*/ 
  delete [] sptcount;
  delete [] rptcount;
  delete [] ptcount;
  for(int i=0; i<mpi_comm_size; ++i) delete [] distMatrix[i];
  delete [] distMatrix;  

}


#endif
