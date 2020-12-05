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
#include <mpi.h>
#include "xc_redistribute.h"
using namespace std;

// Distribution matrix for load balancing prior to sswder calculation.
int** distMatrix=NULL;

//--------------------------------------------------------
// Function to redistribute XC quadrature points among GPUs
// prior to sswder calculation. Sends back the adjustment to 
// size of the arrays. 
//--------------------------------------------------------
void getAdjustment(int mpisize, int mpirank, int count){

  bool master=false;
  if(mpirank == 0) master = true;

  // define arrays
  int *ptcount    = new int[mpisize]{0};
  int *residuals  = new int[mpisize]{0};

  distMatrix=new int*[mpisize];
  for(int i=0; i<mpisize; ++i) distMatrix[i] = new int[mpisize]{0};

  ptcount[mpirank]=count;

  MPI_Barrier(MPI_COMM_WORLD);
  // broadcast ptcount array
  for(int i=0; i<mpisize; ++i) MPI_Bcast(&ptcount[i], 1, MPI_INT, i, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);  
 
  if(master) cout << "mpirank= "<<mpirank << " init array:" << endl;

  for(int i=0; i<mpisize; ++i)
    cout << ptcount[i] << " ";
  cout << endl;

  //find sum
  int sum=0;

  for(int i=0;i<mpisize;++i){
    sum += ptcount[i]; 
  }

  int average = (int) sum/mpisize;   

  if(master) cout << "mpirank= "<< mpirank << " sum= " << sum << " average= " << average << endl;  

  if(master) cout << "mpirank= "<< mpirank << " discrepencies:" << endl;
  
  for(int i=0; i<mpisize; ++i)
    residuals[i]=ptcount[i]-average;

  if(master){
    for(int i=0; i<mpisize; ++i) cout << residuals[i] << " ";
    cout << endl; 

    cout << "mpirank= "<< mpirank << " distributing evenly:" << endl;
  }

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

  if(master) cout << "mpirank= " << mpirank<< " net gain= "<< gain-loss << " adjusted size= "<< count-gain-loss << endl;

  // deallocate memory
  delete [] ptcount;
  delete [] residuals;
  for(int i=0; i<mpisize; ++i) delete [] distMatrix[i]; 
  delete [] distMatrix;

}

#endif
