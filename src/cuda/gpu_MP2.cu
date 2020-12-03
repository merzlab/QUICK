/*
 *  gpu_MP2.cpp
 *  new_quick
 *
 *  Created by Yipu Miao on 6/17/11.
 *  Copyright 2011 University of Florida.All rights reserved.
 *  
 *  Yipu Miao 9/15/11:  the first draft is released. And the GPUGP QM compuation can 
 *                      achieve as much as 15x faster at double precision level compared with CPU.
 */

#include "gpu.h"
#include <cuda.h>

#undef STOREDIM
//#ifdef int_spd
#define STOREDIM STOREDIM_S
//#else
//#define STOREDIM STOREDIM_L
//#endif

/*
 Constant Memory in GPU is fast but quite limited and hard to operate, usually not allocatable and 
 readonly. So we put the following variables into constant memory:
    _MP2: a gpu simluation type variable. which is to store to location of basic information about molecule and basis
 set. Note it only store the location, so it's mostly a set of pointer to GPU memory. and with some non-pointer
 value like the number of basis set. See gpu_type.h for details.
 devTrans_MP2 : arrays to save the mapping index, will be elimited by hand writing unrolling code.
 Sumindex: a array to store refect how many temp variable needed in VRR. can be elimited by hand writing code.
 */

static __constant__ gpu_simulation_type devSim_MP2;
static __constant__ int devTrans_MP2[TRANSDIM*TRANSDIM*TRANSDIM];
static __constant__ int Sumindex_MP2[10]={0,0,1,4,10,20,35,56,84,120};

/*
 upload gpu simulation type to constant memory
 */
void upload_sim_to_constant_MP2(_gpu_type gpu){
    cudaError_t status;
    //status = cudaMemcpyToSymbol("devSim_MP2", &gpu->gpu_sim, sizeof(gpu_simulation_type), 0, cudaMemcpyHostToDevice);
    status = cudaMemcpyToSymbol(devSim_MP2, &gpu->gpu_sim, sizeof(gpu_simulation_type));
	PRINTERROR(status, " cudaMemcpyToSymbol, sim copy to constants failed")
}

/*
__global__ void printY()
{
    for(int i=0;i<devSim_MP2.nbasis;i++)
    {
        for(int j=0;j<devSim_MP2.nbasis;j++)
        {
            for(int k=0;k<devSim_MP2.nbasis;k++)
            {
                for(int l=0;l<devSim_MP2.nbasis;l++)
                {       
                    printf("after get2e_MP2_kernel i,j,k,l, and Y are %d, %d, %d, %d, %lf\n",\
 i+1,j+1,k+1,l+1,LOC4(devSim_MP2.Y_Matrix, i,j,k,l, devSim_MP2.nbasis, devSim_MP2.nbasis, devSim_MP2.nbasis, devSim_MP2.nbasis));
                            
                }
            }
        }
    }
}

__global__ void printCoeff() 
{
	for(int i=0;i<devSim_MP2.nbasis;i++)
    {
        for(int j=0;j<devSim_MP2.nbasis;j++)
        {
           printf("in printcoeff, i,j, and coeff are %d, %d, %lf\n", i+1,j+1,LOC2(devSim_MP2.coefficient, i,j, devSim_MP2.nbasis, devSim_MP2.nbasis));
        }
    }
}


__global__ void printKsumtype()
{
	printf("in printKsumtype, gpu_sim.Ksumtype is:\n");
	for(int i=0;i<devSim_MP2.nshell+1;i++)
		printf("%d\n",devSim_MP2.Ksumtype[i]);
	 printf("in printKsumtype, gpu_sim.Ksumtype[2] is: %d\n", devSim_MP2.Ksumtype[2]);
	
}

__global__ void printorbmp2k331()
{
	 printf("in printorbmp2k331, devSim_MP2.orbmp2k331 is:\n");
	 for(int i=0;i<devSim_MP2.nElec/2;i++)
	 {
		for(int j=0;j<devSim_MP2.nElec/2;j++)
		{
			for(int k=0;k<devSim_MP2.nbasis-devSim_MP2.nElec/2;k++)
			{
				for(int l=0;l<devSim_MP2.nbasis;l++)
				{
					printf("%lf	",LOC4(devSim_MP2.orbmp2k331,i,j,k,l, devSim_MP2.nElec/2,devSim_MP2.nElec/2,devSim_MP2.nbasis-devSim_MP2.nElec/2,devSim_MP2.nbasis));
				}
				printf("\n");
			}
		}
	}
}

__global__ void printQsQfbasis()
{
	printf("in gpu, print Qsbasis\n");
	for(int i=0;i<devSim_MP2.nshell;i++)
	{
		for(int j=0;j<4;j++)
		{
			printf("%d ", LOC2(devSim_MP2.Qsbasis, i, j, devSim_MP2.nshell, 4));
		}
		printf("\n");
	}
	
	printf("in gpu, print Qfbasis\n");
    for(int i=0;i<devSim_MP2.nshell;i++)
    {
        for(int j=0;j<4;j++)
        {
            printf("%d ", LOC2(devSim_MP2.Qfbasis, i, j, devSim_MP2.nshell, 4));
        }
        printf("\n");
    }
	
}

__global__ void printQstartQfinal()
{
	printf("in gpu, print Qstart\n");
	for(int i=0;i<devSim_MP2.nshell;i++)
		printf("%d ", devSim_MP2.Qstart[i]);
	printf("\n");

	printf("in gpu, print Qfinal\n");
	for(int i=0;i<devSim_MP2.nshell;i++)
        printf("%d ", devSim_MP2.Qfinal[i]);
    printf("\n");

}
*/

/*
__global__ void forthQuarterTransDevice()
{
	//printf("Start 4th Quarter Transformation and finally summation\n");
	QUICKDouble MP2cor = 0;
	int nsteplength = devSim_MP2.nElec/2;	

	for(int icycle=1;icycle<=nsteplength;icycle++)
	{
		int i3 = icycle;
		for(int k3=i3;k3<=devSim_MP2.nElec/2;k3++)
		{	
			for(int j3=1;j3<=devSim_MP2.nbasis-devSim_MP2.nElec/2;j3++)
			{
				for(int l3=1;l3<=devSim_MP2.nbasis-devSim_MP2.nElec/2;l3++)
				{
					LOC2(devSim_MP2.orbmp2,l3-1,j3-1,devSim_MP2.nbasis-devSim_MP2.nElec/2, devSim_MP2.nbasis-devSim_MP2.nElec/2)=0.0;
					int l3new = l3 + devSim_MP2.nElec/2;
					for(int lll=1;lll<=devSim_MP2.nbasis;lll++)
						LOC2(devSim_MP2.orbmp2,l3-1,j3-1,devSim_MP2.nbasis-devSim_MP2.nElec/2, devSim_MP2.nbasis-devSim_MP2.nElec/2) += \
							LOC4(devSim_MP2.orbmp2k331,icycle-1,k3-1,j3-1,lll-1, devSim_MP2.nElec/2,devSim_MP2.nElec/2, \
								(devSim_MP2.nbasis-devSim_MP2.nElec/2),devSim_MP2.nbasis)*LOC2(devSim_MP2.coefficient,lll-1,l3new-1, devSim_MP2.nbasis, devSim_MP2.nbasis);
					
				}
			}
			for(int j3=1;j3<=devSim_MP2.nbasis-devSim_MP2.nElec/2;j3++)
			{
				for(int l3=1;l3<=devSim_MP2.nbasis-devSim_MP2.nElec/2;l3++)
				{
					if(k3>i3)
					{
						//here add new attribute: molorbe
						MP2cor += 2.0/(devSim_MP2.molorbe[i3-1]+devSim_MP2.molorbe[k3-1]-devSim_MP2.molorbe[j3-1+devSim_MP2.nElec/2]-devSim_MP2.molorbe[l3-1+devSim_MP2.nElec/2]) \
								*LOC2(devSim_MP2.orbmp2,j3-1,l3-1,devSim_MP2.nbasis-devSim_MP2.nElec/2, devSim_MP2.nbasis-devSim_MP2.nElec/2) \
								*(2.0*LOC2(devSim_MP2.orbmp2,j3-1,l3-1,devSim_MP2.nbasis-devSim_MP2.nElec/2, devSim_MP2.nbasis-devSim_MP2.nElec/2) \
								- LOC2(devSim_MP2.orbmp2,l3-1,j3-1,devSim_MP2.nbasis-devSim_MP2.nElec/2, devSim_MP2.nbasis-devSim_MP2.nElec/2));
					}
					if(k3==i3)
					{
						MP2cor += 1.0/(devSim_MP2.molorbe[i3-1]+devSim_MP2.molorbe[k3-1]-devSim_MP2.molorbe[j3-1+devSim_MP2.nElec/2]-devSim_MP2.molorbe[l3-1+devSim_MP2.nElec/2]) \
                                *LOC2(devSim_MP2.orbmp2,j3-1,l3-1,devSim_MP2.nbasis-devSim_MP2.nElec/2, devSim_MP2.nbasis-devSim_MP2.nElec/2) \
                                *(2.0*LOC2(devSim_MP2.orbmp2,j3-1,l3-1,devSim_MP2.nbasis-devSim_MP2.nElec/2, devSim_MP2.nbasis-devSim_MP2.nElec/2) \
                                - LOC2(devSim_MP2.orbmp2,l3-1,j3-1,devSim_MP2.nbasis-devSim_MP2.nElec/2, devSim_MP2.nbasis-devSim_MP2.nElec/2));
					}
				}
			}
	
		}
	}
	printf("On the CUDA side, final MP2 correction is %lf\n", MP2cor);
}
*/

/*
__global__ void firstThreeQuartersTransDevice()
{
	int jshell = devSim_MP2.jshell;
	for(int II=0; II<jshell; II++)
	{
		for(int JJ=II; JJ<jshell; JJ++)
		{
        	for(int ind=0;ind<devSim_MP2.nElec/2*devSim_MP2.nbasis*10*10*2;ind++)
            	devSim_MP2.orbmp2i331[ind]=0;
        	for(int ind=0;ind<devSim_MP2.nElec/2*(devSim_MP2.nbasis-devSim_MP2.nElec/2)*10*10*2;ind++)
            	devSim_MP2.orbmp2j331[ind]=0;	

			for(int KK=0; KK<jshell; KK++)
			{
				for(int LL=KK; LL<jshell; LL++)
				{	
					
					//Here call shellmp2:
					
					int NII1=devSim_MP2.Qstart[II];
   					int NII2=devSim_MP2.Qfinal[II];
   					int NJJ1=devSim_MP2.Qstart[JJ];
   					int NJJ2=devSim_MP2.Qfinal[JJ];
   					int NKK1=devSim_MP2.Qstart[KK];
   					int NKK2=devSim_MP2.Qfinal[KK];
   					int NLL1=devSim_MP2.Qstart[LL];
   					int NLL2=devSim_MP2.Qfinal[LL];

					for(int I=NII1; I<=NII2; I++)
					{
						for(int J=NJJ1; J<=NJJ2; J++)
						{
							for(int K=NKK1; K<=NKK2; K++)
							{
								for(int L=NLL1; L<=NLL2; L++)
								{
									QUICKDouble DNMax = 0;
									firstQuarterTransDevice(I, J, K, L, II, JJ, KK, LL, DNMax);
								}
							}
						}
					}
			
					
				}
			}
			//Here do 2nd and 3rd transformations
			//second and third quarter transfermation here:
			
			int NII1=devSim_MP2.Qstart[II];
            int NII2=devSim_MP2.Qfinal[II];
            int NJJ1=devSim_MP2.Qstart[JJ];
            int NJJ2=devSim_MP2.Qfinal[JJ];

			int NBI1= LOC2(devSim_MP2.Qsbasis, II, NII1, devSim_MP2.nshell, 4);			
			int NBI2= LOC2(devSim_MP2.Qfbasis, II, NII2, devSim_MP2.nshell, 4);
			int NBJ1= LOC2(devSim_MP2.Qsbasis, JJ, NJJ1, devSim_MP2.nshell, 4);
			int NBJ2= LOC2(devSim_MP2.Qfbasis, JJ, NJJ2, devSim_MP2.nshell, 4);

            int II111=NBI1;
            int II112=NBI2;
            int JJ111=NBJ1;
            int JJ112=NBJ2;



			for(int III=II111; III<=II112; III++)
                {
                    for(int JJJ=max(III,JJ111); JJJ<=JJ112; JJJ++)
                    {
                        int IIInew = III - II111 +1;
                        int JJJnew = JJJ - JJ111 +1;

                        // second quarter transformation
                        for(int LLL=1; LLL<=devSim_MP2.nbasis; LLL++)
                        {
                            for(int j33=1; j33<=devSim_MP2.nbasis-devSim_MP2.nElec/2;j33++)
                            {
                                int j33new = j33 + devSim_MP2.nElec/2;
                                QUICKDouble atemp = LOC2(devSim_MP2.coefficient, LLL-1, j33new-1, devSim_MP2.nbasis, devSim_MP2.nbasis);
                                int nsteplength = devSim_MP2.nElec/2;
                                for(int icycle=1; icycle<=nsteplength; icycle++)
                                {
                                    QUICKADD(LOC5(devSim_MP2.orbmp2j331,icycle-1,j33-1,IIInew-1,JJJnew-1,0, \
                                        devSim_MP2.nElec/2,(devSim_MP2.nbasis-devSim_MP2.nElec/2),10,10,2),\
                                            LOC5(devSim_MP2.orbmp2i331,icycle-1,LLL-1,IIInew-1,JJJnew-1,0, devSim_MP2.nElec/2,devSim_MP2.nbasis,10,10,2)*atemp);

                                    if(III!=JJJ){
                                        QUICKADD(LOC5(devSim_MP2.orbmp2j331,icycle-1,j33-1,JJJnew-1,IIInew-1,1, \
                                            devSim_MP2.nElec/2,(devSim_MP2.nbasis-devSim_MP2.nElec/2),10,10,2), \
                                                LOC5(devSim_MP2.orbmp2i331,icycle-1,LLL-1,JJJnew-1,IIInew-1,1,devSim_MP2.nElec/2,devSim_MP2.nbasis,10,10,2)*atemp);
                                    }
                                }
                            }
                        }

						 // third quarter transformation, use devSim_MP2.orbmp2k331 and QUICKADD
                        for(int j33=1; j33<=devSim_MP2.nbasis-devSim_MP2.nElec/2;j33++)
                        {
                            for(int k33=1; k33<=devSim_MP2.nElec/2; k33++)
                            {
                                QUICKDouble atemp = LOC2(devSim_MP2.coefficient, III-1, k33-1, devSim_MP2.nbasis, devSim_MP2.nbasis);
                                QUICKDouble atemp2 = LOC2(devSim_MP2.coefficient, JJJ-1, k33-1, devSim_MP2.nbasis, devSim_MP2.nbasis);
                                int nsteplength = devSim_MP2.nElec/2;
                                for(int icycle=1; icycle<=nsteplength; icycle++)
                                {
                                    QUICKADD(LOC4(devSim_MP2.orbmp2k331,icycle-1,k33-1,j33-1,JJJ-1, \
                                        devSim_MP2.nElec/2,devSim_MP2.nElec/2,(devSim_MP2.nbasis-devSim_MP2.nElec/2),devSim_MP2.nbasis),\
                                        LOC5(devSim_MP2.orbmp2j331,icycle-1,j33-1,IIInew-1,JJJnew-1,0, devSim_MP2.nElec/2, \
                                        (devSim_MP2.nbasis-devSim_MP2.nElec/2),10,10,2)*atemp);
                                    if(III!=JJJ){
                                        QUICKADD(LOC4(devSim_MP2.orbmp2k331,icycle-1,k33-1,j33-1,III-1, \
                                            devSim_MP2.nElec/2,devSim_MP2.nElec/2,(devSim_MP2.nbasis-devSim_MP2.nElec/2),devSim_MP2.nbasis),\
                                            LOC5(devSim_MP2.orbmp2j331,icycle-1,j33-1,JJJnew-1,IIInew-1,1, \
                                            devSim_MP2.nElec/2,(devSim_MP2.nbasis-devSim_MP2.nElec/2),10,10,2)*atemp2);
                                    }
                                }
                            }
                        }
                    }
				}
		}
	}
}
*/

//this kernel is to parallelize the two inner loops of the first transformation in firstThreeQuartersTransHost
__global__ void 
__launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) firstQuarterTransKernel(int II,int JJ,int nstepmp2s, int nsteplength, int nstep, int nbasistemp, QUICKDouble cutoffmp2, QUICKDouble* orbmp2i331)
{
	int offside = blockIdx.x*blockDim.x+threadIdx.x;
	int totalThreads = blockDim.x*gridDim.x;	
	int jshell =  devSim_MP2.jshell;
	int myInt = jshell*jshell/totalThreads;

	int nshell = devSim_MP2.nshell;
	int nbasis = devSim_MP2.nbasis;

	if(jshell*jshell-myInt*totalThreads>offside) myInt++;

	for(int i=1;i<=myInt;i++)
	{
		int currentInt = totalThreads*(i-1)+offside;	
		int KK = currentInt/jshell;
		int LL = currentInt%jshell;

		if(KK<jshell && LL<jshell && LL>=KK)
		{
			// Do we need prescreening here: if(testCutoff.gt.cutoffmp2)then? Yes.
			// Set ntemp later
			QUICKDouble comax = 0.0;
			QUICKDouble testCutoff = LOC2(devSim_MP2.YCutoff, II, JJ, nshell, nshell)*LOC2(devSim_MP2.YCutoff, KK, LL, nshell, nshell); 
			
			if(testCutoff>cutoffmp2)
			{
				int NII1=devSim_MP2.Qstart[II];
        		int NII2=devSim_MP2.Qfinal[II];
        		int NJJ1=devSim_MP2.Qstart[JJ];
        		int NJJ2=devSim_MP2.Qfinal[JJ];
       			int NKK1=devSim_MP2.Qstart[KK];
       			int NKK2=devSim_MP2.Qfinal[KK];
        		int NLL1=devSim_MP2.Qstart[LL];
        		int NLL2=devSim_MP2.Qfinal[LL];
	
				int NBK1 = LOC2(devSim_MP2.Qsbasis, KK, NKK1, nshell, 4);
				int NBK2 = LOC2(devSim_MP2.Qfbasis, KK, NKK2, nshell, 4);
            	int NBL1 = LOC2(devSim_MP2.Qsbasis, LL, NLL1, nshell, 4);
				int NBL2 = LOC2(devSim_MP2.Qfbasis, LL, NLL2, nshell, 4);	    
	
				int KK111 = NBK1;
				int KK112 = NBK2;
				int LL111 = NBL1;
				int LL112 = NBL2;		

				for(int icycle=1; icycle<=nsteplength; icycle++)
				{
					int i3 = nstepmp2s + icycle -1;
					for(int KKK = KK111;KKK<=KK112;KKK++)
					{
						for(int LLL=MAX(KKK,LL111);LLL<=LL112;LLL++)
						{
							comax = MAX(comax, fabs(LOC2(devSim_MP2.coefficient,KKK-1,i3-1,nbasis,nbasis)));
							comax = MAX(comax, fabs(LOC2(devSim_MP2.coefficient,LLL-1,i3-1,nbasis,nbasis)));				
						}
					}
				}
				testCutoff *= comax;

				if(testCutoff>cutoffmp2)
				{
					// Set ntemp if needed
					// from shellmp2:
					for(int I=NII1; I<=NII2; I++)
        			{
        				for(int J=NJJ1; J<=NJJ2; J++)
            			{
            				for(int K=NKK1; K<=NKK2; K++)
                			{
                				for(int L=NLL1; L<=NLL2; L++)
                    			{
									// is I,J,K,L the same iii,jjj,kkk,lll as in get2e_MP2_kernel()?
									int nshell = devSim_MP2.nshell;
									QUICKDouble DNMax = MAX(MAX(4.0*LOC2(devSim_MP2.cutMatrix, II, JJ, nshell, nshell), 4.0*LOC2(devSim_MP2.cutMatrix, KK, LL, nshell, nshell)),
                                    	MAX(MAX(LOC2(devSim_MP2.cutMatrix, II, LL, nshell, nshell),     LOC2(devSim_MP2.cutMatrix, II, KK, nshell, nshell)),
                                        	MAX(LOC2(devSim_MP2.cutMatrix, JJ, KK, nshell, nshell),     LOC2(devSim_MP2.cutMatrix, JJ, LL, nshell, nshell))));
						
									if ((LOC2(devSim_MP2.YCutoff, KK, LL, nshell, nshell) * LOC2(devSim_MP2.YCutoff, II, JJ, nshell, nshell))> devSim_MP2.integralCutoff && \
                						(LOC2(devSim_MP2.YCutoff, KK, LL, nshell, nshell) * LOC2(devSim_MP2.YCutoff, II, JJ, nshell, nshell) * DNMax) > devSim_MP2.integralCutoff) 
									{
										//equivalent to classmp2
										iclass_MP2(I,J,K,L,II,JJ,KK,LL,nstepmp2s,nsteplength,nstep, nbasistemp, DNMax,orbmp2i331);
									}
                    			}
                			}
           				}
       				}
				}
			}	
		}
	}
}

//this kernel is to parallelize the second transformation in firstThreeQuartersTransHost
__global__ void
__launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) secondQuarterTransKernel(int III, int JJJ, int IIInew,int JJJnew, int nsteplength, int nstep, int nbasistemp, QUICKDouble* orbmp2i331, QUICKDouble* orbmp2j331)
{
	int offside = blockIdx.x*blockDim.x+threadIdx.x;
    int totalThreads = blockDim.x*gridDim.x;
	
	int nbasis =  devSim_MP2.nbasis;
	int nElec = devSim_MP2.nElec;
	//nsteplength is now passed as an argument:
	//int nsteplength = devSim_MP2.nElec/2;
	int ivir = nbasis-nElec/2;
	QUICKDouble* coefficient = devSim_MP2.coefficient;	

	//int myInt = nbasis*(nbasis-nElec/2)/totalThreads;
	//if(nbasis*(nbasis-nElec/2)-myInt*totalThreads>offside) myInt++;
	
	int myInt = nbasis*ivir*nsteplength/totalThreads;
	if(nbasis*ivir*nsteplength-myInt*totalThreads>offside) myInt++;	

	for(int i=1;i<=myInt;i++)	
	{
		int currentInt = totalThreads*(i-1)+offside;
		//int LLL = currentInt/(nbasis-nElec/2);
		//int j33 = currentInt%(nbasis-nElec/2);
		int icycle = currentInt%nsteplength;
		currentInt /= nsteplength;
		int j33 = currentInt%(nbasis-nElec/2);
		int LLL = currentInt/(nbasis-nElec/2);

		if(LLL<nbasis && j33<nbasis-nElec/2)
		{
			int j33new = j33 + nElec/2;
			QUICKDouble atemp = LOC2(coefficient, LLL, j33new, nbasis, nbasis);
            //int nsteplength = nElec/2;
            //for(int icycle=0; icycle<nsteplength; icycle++)
            //{
				QUICKADD(LOC5(orbmp2j331,icycle,j33,IIInew,JJJnew,0,nstep,ivir,nbasistemp, nbasistemp,2),\
							LOC5(orbmp2i331,icycle,LLL,IIInew,JJJnew,0,nstep,nbasis,nbasistemp, nbasistemp,2)*atemp);

                if(III!=JJJ){
					QUICKADD(LOC5(orbmp2j331,icycle,j33,JJJnew,IIInew,1,nstep,ivir,nbasistemp, nbasistemp,2),\
							LOC5(orbmp2i331,icycle,LLL,JJJnew,IIInew,1,nstep,nbasis,nbasistemp, nbasistemp,2)*atemp);
			   	}
            //}	
		}	
	}
}

//this kernel is to parallelize the third transformation in firstThreeQuartersTransHost
__global__ void
__launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) thirdQuarterTransKernel(int III, int JJJ, int IIInew,int JJJnew, int nsteplength, int nstep, int nbasistemp, QUICKDouble* orbmp2j331, QUICKDouble* orbmp2k331)
//__global__ void thirdQuarterTransKernel(int III, int JJJ, int IIInew,int JJJnew,int nsteplength, int nstep, int nbasistemp, QUICKDouble* orbmp2j331, QUICKDouble* orbmp2k331)
{
	int offside = blockIdx.x*blockDim.x+threadIdx.x;
	int totalThreads = blockDim.x*gridDim.x;
	
	int nbasis =  devSim_MP2.nbasis;
    int nElec = devSim_MP2.nElec;

	//nsteplength is now passed as an argument:	
    //int nsteplength = devSim_MP2.nElec/2;
	int ivir = nbasis-nElec/2;
	int iocc = nElec/2;
	QUICKDouble* coefficient = devSim_MP2.coefficient;
    
    //int myInt = nbasis*(nbasis-nElec/2)/totalThreads;
    //if(nbasis*(nbasis-nElec/2)-myInt*totalThreads>offside) myInt++; 
	
	/*
	if(offside==0)
	{
		printf("totally %d threads, %d blocks\n", totalThreads, gridDim.x);
		printf("totally need %d threads for one iteration, nsteplength is %d\n",nbasis*(nbasis-nElec/2)*nsteplength,nsteplength );
	}
	*/
	
	int myInt = nbasis*ivir*nsteplength/totalThreads;
	if(nbasis*ivir*nsteplength-myInt*totalThreads>offside) myInt++; 

	for(int i=1;i<=myInt;i++)
	{
		int currentInt = totalThreads*(i-1)+offside;
		//int j33 = currentInt/(nElec/2);
		//int k33 = currentInt%(nElec/2);
		int icycle = currentInt%nsteplength;
		currentInt /= nsteplength;
		int k33 = currentInt%(nElec/2);
		int j33 = currentInt/(nElec/2);	
	

		if(j33<ivir && k33<nElec/2 && icycle<nsteplength)
		{
			QUICKDouble atemp = LOC2(coefficient, III-1, k33, nbasis, nbasis);
            QUICKDouble atemp2 = LOC2(coefficient, JJJ-1, k33, nbasis, nbasis);
            //int nsteplength = nElec/2;
           	//for(int icycle=0; icycle<nsteplength; icycle++)
            //{	
				//should not need atomicadd here.
            	LOC4(orbmp2k331,icycle,k33,j33,JJJ-1,nstep, iocc, ivir, nbasis) +=\
                	LOC5(orbmp2j331,icycle,j33,IIInew,JJJnew,0, nstep, ivir, nbasistemp, nbasistemp, 2)*atemp;

                if(III!=JJJ){
                	LOC4(orbmp2k331,icycle,k33,j33,III-1,nstep, iocc, ivir, nbasis) +=\
                    	LOC5(orbmp2j331,icycle,j33,JJJnew,IIInew,1,nstep, ivir, nbasistemp, nbasistemp, 2)*atemp2;
            	}
            //}
		}
		
	}

}


void fourQuarterTransHost(QUICKDouble* orbmp2i331, QUICKDouble* orbmp2j331, QUICKDouble* orbmp2k331, QUICKDouble* orbmp2, _gpu_type gpu)
{
	int jshell = gpu->jshell;
	//printf("in firstThreeQuartersTransHost, jshell is %d\n", jshell);
	int nshell = gpu->nshell;
	int nbasis = gpu->nbasis;
	int nElec = gpu->nElec;
	int iocc = nElec/2;
	int ivir = nbasis - nElec/2;
	QUICKDouble cutoffmp2 = 1.0E-8;
	QUICKDouble ttt = 0;
	QUICKDouble* YCutoff = gpu->gpu_cutoff->YCutoff->_hostData;
	//printf("to print YCutoff:\n");
	for(int i=0;i<nshell;i++)
	{
		for(int j=0;j<nshell;j++)
		{	
			ttt = MAX(ttt,LOC2(YCutoff, i, j, nshell, nshell));
		}
	}
	printf("ttt is %lf\n",ttt);
	//Shall we enlarge it here:
	printf("devSim_MP2.integralCutoff*10000 is %lf\n", gpu -> gpu_cutoff -> integralCutoff*10000);

	int* Qstart = gpu->gpu_basis->Qstart->_hostData;
	int* Qfinal = gpu->gpu_basis->Qfinal->_hostData;

	int* Qsbasis = gpu->gpu_basis->Qsbasis->_hostData;
	int* Qfbasis = gpu->gpu_basis->Qfbasis->_hostData;

	QUICKDouble* coefficient = gpu->gpu_calculated->coefficient->_hostData;
	
	QUICKDouble ememorysum = iocc*ivir*nbasis*8.0/1024.0/1024.0/1024.0;
	printf("On CUDA side, ememorysum is %lf, 1.5/ememorysum is %lf\n", ememorysum, 1.5/ememorysum);
	
	int nstep = MIN(int(1.50/ememorysum),nElec/2);
	// Alwasy use f orbitals:
	int nbasistemp = 10;


	QUICKDouble* orbmp2i331_d;
	//cudaMalloc((void **)&orbmp2i331_d, sizeof(QUICKDouble)*nElec/2*nbasis*10*10*2);
	cudaMalloc((void **)&orbmp2i331_d, sizeof(QUICKDouble)*nstep*nbasis*nbasistemp*nbasistemp*2);	

	QUICKDouble* orbmp2j331_d;
    //cudaMalloc((void **)&orbmp2j331_d, sizeof(QUICKDouble)*nElec/2*(nbasis-nElec/2)*10*10*2);	
	cudaMalloc((void **)&orbmp2j331_d, sizeof(QUICKDouble)*nstep*ivir*nbasistemp*nbasistemp*2);

	QUICKDouble* orbmp2k331_d;
	//cudaMalloc((void **)&orbmp2k331_d,sizeof(QUICKDouble)*nElec/2*nElec/2*(nbasis-nElec/2)*nbasis);
	//cudaMemset(orbmp2k331_d, 0, sizeof(QUICKDouble)*nElec/2*nElec/2*(nbasis-nElec/2)*nbasis);	
	cudaMalloc((void **)&orbmp2k331_d,sizeof(QUICKDouble)*nstep*iocc*ivir*nbasis);
	cudaMemset(orbmp2k331_d, 0, sizeof(QUICKDouble)*nstep*iocc*ivir*nbasis);	

	QUICKDouble* orbmp2_d;
	//cudaMalloc((void **)&orbmp2_d,sizeof(QUICKDouble)*(nbasis-nElec/2)*(nbasis-nElec/2));
    //cudaMemset(orbmp2_d,0,sizeof(QUICKDouble)*(nbasis-nElec/2)*(nbasis-nElec/2));
	cudaMalloc((void **)&orbmp2_d,sizeof(QUICKDouble)*ivir*ivir);
	cudaMemset(orbmp2_d,0,sizeof(QUICKDouble)*ivir*ivir);

	QUICKDouble* MP2cor_d;
	cudaMalloc((void **)&MP2cor_d,sizeof(QUICKDouble));
    cudaMemset(MP2cor_d, 0, sizeof(QUICKDouble));

    QUICKDouble* MP2cor = new QUICKDouble[1];
    MP2cor[0]= 0;
	
	//For not very large moleculars, nstepmp2 should be 1
	//This part is simplified from the serial version. Hopefully didn't mess up:
	int nstepmp2 = nElec/2/nstep;
	if(nstep*nstepmp2<nElec/2)
		nstepmp2++;
	printf("TOTAL STEP is %d\n", nstepmp2);
	
	
	for(int i3new=1;i3new<=nstepmp2;i3new++)
	{
		//int ntemp = 0;
		int nstepmp2s = (i3new-1)*nstep+1;
		int nstepmp2f = i3new*nstep;
		
		if(i3new==nstepmp2)
			nstepmp2f=nElec/2;
		//In many cases, nsteplength is simply nelec/2:
		int nsteplength = nstepmp2f-nstepmp2s+1;
	
		printf("i3new is %d\n", i3new);
		printf("nsteplength is %d\n", nsteplength);


		cudaMemset(orbmp2k331_d, 0, sizeof(QUICKDouble)*nstep*iocc*ivir*nbasis);
		for(int II=0; II<jshell; II++)
    	{
        	for(int JJ=II; JJ<jshell; JJ++)
        	{
				if(LOC2(YCutoff, II, JJ, nshell, nshell)>cutoffmp2/ttt)
				{		
	
				//cudaMemset(orbmp2i331_d,0,sizeof(QUICKDouble)*nElec/2*nbasis*10*10*2);
				//cudaMemset(orbmp2j331_d,0,sizeof(QUICKDouble)*nElec/2*(nbasis-nElec/2)*10*10*2);
				cudaMemset(orbmp2i331_d,0,sizeof(QUICKDouble)*nstep*nbasis*nbasistemp*nbasistemp*2);
				cudaMemset(orbmp2j331_d,0,sizeof(QUICKDouble)*nstep*ivir*nbasistemp*nbasistemp*2);
	
				// The following kernal should complete AO integral generation and first transfermation
            	firstQuarterTransKernel<<<gpu->blocks, gpu->twoEThreadsPerBlock>>>(II,JJ,nstepmp2s,nsteplength,nstep, nbasistemp, cutoffmp2, orbmp2i331_d);
				cudaDeviceSynchronize(); 
				//cudaMemcpy(orbmp2i331, orbmp2i331_d, sizeof(QUICKDouble)*nElec/2*nbasis*10*10*2, cudaMemcpyDeviceToHost);

				/*
            	for(int KK=0; KK<jshell; KK++)
            	{
                	for(int LL=KK; LL<jshell; LL++)
                	{
						//printf("II, JJ, KK, LL are %d, %d, %d, %d\n", II, JJ, KK, LL);			
						int NII1=Qstart[II];
                    	int NII2=Qfinal[II];
                    	int NJJ1=Qstart[JJ];
                    	int NJJ2=Qfinal[JJ];
                    	int NKK1=Qstart[KK];
                    	int NKK2=Qfinal[KK];
                    	int NLL1=Qstart[LL];
                    	int NLL2=Qfinal[LL];

						for(int I=NII1; I<=NII2; I++)
                    	{
                        	for(int J=NJJ1; J<=NJJ2; J++)
                        	{
                            	for(int K=NKK1; K<=NKK2; K++)
                            	{
                                	for(int L=NLL1; L<=NLL2; L++)
                                	{
                                    	firstQuarterTransHost(I, J, K, L, II, JJ, KK, LL, orbmp2i331,\
											Qsbasis, Qfbasis, Y_Matrix, integralCutoff, coefficient,Ksumtype, nshell, nbasis, nElec);
                                	
									}
                            	}
                        	}
                    	}
                	}
            	}
				*/
				int NII1=Qstart[II];
            	int NII2=Qfinal[II];
            	int NJJ1=Qstart[JJ];
            	int NJJ2=Qfinal[JJ];

            	int NBI1= LOC2(Qsbasis, II, NII1, nshell, 4);
            	int NBI2= LOC2(Qfbasis, II, NII2, nshell, 4);
            	int NBJ1= LOC2(Qsbasis, JJ, NJJ1, nshell, 4);
            	int NBJ2= LOC2(Qfbasis, JJ, NJJ2, nshell, 4);

            	int II111=NBI1;
            	int II112=NBI2;
            	int JJ111=NBJ1;
            	int JJ112=NBJ2;

				for(int III=II111; III<=II112; III++)
                {
                    for(int JJJ=max(III,JJ111); JJJ<=JJ112; JJJ++)
                    {
						
                        int IIInew = III - II111;
                        int JJJnew = JJJ - JJ111;

                        // second quarter transformation
						secondQuarterTransKernel<<<gpu->blocks, gpu->twoEThreadsPerBlock>>>(III,JJJ,IIInew,JJJnew,nsteplength, nstep, nbasistemp,orbmp2i331_d,orbmp2j331_d);
						cudaDeviceSynchronize();
						// this is no longer needed:
						//cudaMemcpy(orbmp2j331, orbmp2j331_d, sizeof(QUICKDouble)*nElec/2*(nbasis-nElec/2)*10*10*2, cudaMemcpyDeviceToHost);
						/*
                        for(int LLL=0; LLL<nbasis; LLL++)
                        {
                            for(int j33=0; j33<nbasis-nElec/2;j33++)
                            {
                                int j33new = j33 + nElec/2;
                                QUICKDouble atemp = LOC2(coefficient, LLL, j33new, nbasis, nbasis);
                                int nsteplength = nElec/2;
                                for(int icycle=0; icycle<nsteplength; icycle++)
                                {
									LOC5(orbmp2j331,icycle,j33,IIInew,JJJnew,0,nElec/2,(nbasis-nElec/2),10,10,2) +=\
										LOC5(orbmp2i331,icycle,LLL,IIInew,JJJnew,0, nElec/2,nbasis,10,10,2)*atemp;
									
                                    if(III!=JJJ){
										LOC5(orbmp2j331,icycle,j33,JJJnew,IIInew,1,nElec/2,(nbasis-nElec/2),10,10,2) +=\
											LOC5(orbmp2i331,icycle,LLL,JJJnew,IIInew,1,nElec/2,nbasis,10,10,2)*atemp;
                                    }
                                }
                            }
                        }
						*/
						
						// third quarter transformation, use devSim_MP2.orbmp2k331 and QUICKADD
					 	// Parallelization here is more time consuming than than the serial. Probably because of the cudaMemcpy(orbmp2k331...).
						
						thirdQuarterTransKernel<<<gpu->blocks, gpu->twoEThreadsPerBlock>>>(III,JJJ,IIInew,JJJnew,nsteplength,nstep,nbasistemp,orbmp2j331_d,orbmp2k331_d);
                        //thirdQuarterTransKernel<<<320, 1024>>>(III,JJJ,IIInew,JJJnew,nsteplength,nstep,nbasistemp,orbmp2j331_d,orbmp2k331_d);
						cudaDeviceSynchronize();
						// this is no longer needed:
                        //cudaMemcpy(orbmp2k331, orbmp2k331_d, sizeof(QUICKDouble)*nElec/2*nElec/2*(nbasis-nElec/2)*nbasis, cudaMemcpyDeviceToHost);
											
						/*
                      	for(int j33=0; j33<ivir;j33++)
                        {
                            for(int k33=0; k33<nElec/2; k33++)
                            {
                                QUICKDouble atemp = LOC2(coefficient, III-1, k33, nbasis, nbasis);
                                QUICKDouble atemp2 = LOC2(coefficient, JJJ-1, k33, nbasis, nbasis);
                                int nsteplength = nElec/2;
                                for(int icycle=0; icycle<nsteplength; icycle++)
                                {
                                	LOC4(orbmp2k331,icycle,k33,j33,JJJ-1,nstep,iocc,ivir,nbasis) +=\
										LOC5(orbmp2j331,icycle,j33,IIInew,JJJnew,0, nstep, ivir, nbasistemp, nbasistemp, 2)*atemp;
	
									//printf("orbmp2k331 is %lf\n",LOC4(orbmp2k331,icycle,k33,j33,JJJ-1,nElec/2,nElec/2,(nbasis-nElec/2),nbasis));							   
									if(III!=JJJ){
										LOC4(orbmp2k331,icycle,k33,j33,III-1,nstep,iocc,ivir,nbasis) +=\
											LOC5(orbmp2j331,icycle,j33,JJJnew,IIInew,1,nstep, ivir, nbasistemp, nbasistemp, 2)*atemp2;
                                    }
                                }
                            }
                        }
						*/					
                    }
                }
			}//corresponds to if(LOC2(devSim_MP2.YCutoff, II, JJ, nshell, nshell)>cutoffmp2/ttt)
			}
		}
	
		// first three quarters of the transformation end here
		// start the forth quarter and the final accumulation	
		// this is no longer needed:
		//cudaMemcpy(orbmp2k331_d, orbmp2k331, sizeof(QUICKDouble)*nElec/2*nElec/2*(nbasis-nElec/2)*nbasis, cudaMemcpyHostToDevice);
		for(int icycle=0;icycle<nsteplength;icycle++) 
    	{   
        	//int i3 = icycle;
        	int i3 = nstepmp2s+icycle-1;
			for(int k3=i3;k3<nElec/2;k3++) 
        	{   
            	forthQuarterTransInnerLoopsKernel<<<gpu->blocks, gpu->twoEThreadsPerBlock>>>(icycle, i3, k3,nstep, orbmp2k331_d, orbmp2_d);
            	cudaDeviceSynchronize();
            	finalMP2AccumulationInnerLoopsKernel<<<gpu->blocks, gpu->twoEThreadsPerBlock>>>(i3, k3, orbmp2_d, MP2cor_d);
            	cudaDeviceSynchronize();
			}
		}
	
	}//this corresponds to i3new
	cudaMemcpy(MP2cor, MP2cor_d, sizeof(QUICKDouble),cudaMemcpyDeviceToHost);
    printf("On the CUDA side, final MP2 correction is %lf\n", *MP2cor);

	cudaFree(orbmp2i331_d);
	cudaFree(orbmp2j331_d);
	cudaFree(orbmp2k331_d);
	cudaFree(orbmp2_d);	
	cudaFree(MP2cor_d);	
	delete[] MP2cor;

}

/*
//void firstQuarterTransHost(int I, int J, int K, int L, unsigned int II, unsigned int JJ, unsigned int KK, unsigned int LL, QUICKDouble* orbmp2i331, _gpu_type gpu)
void firstQuarterTransHost(int I, int J, int K, int L, unsigned int II, unsigned int JJ, unsigned int KK, unsigned int LL, QUICKDouble* orbmp2i331, \
	int* Qsbasis, int* Qfbasis, QUICKDouble* Y_Matrix, QUICKDouble integralCutoff, QUICKDouble* coefficient, int* Ksumtype, int nshell, int nbasis, int nElec)
{

	int III1 = LOC2(Qsbasis, II, I, nshell, 4);
    int III2 = LOC2(Qfbasis, II, I, nshell, 4);
    int JJJ1 = LOC2(Qsbasis, JJ, J, nshell, 4);
    int JJJ2 = LOC2(Qfbasis, JJ, J, nshell, 4);
    int KKK1 = LOC2(Qsbasis, KK, K, nshell, 4);
    int KKK2 = LOC2(Qfbasis, KK, K, nshell, 4);
    int LLL1 = LOC2(Qsbasis, LL, L, nshell, 4);
    int LLL2 = LOC2(Qfbasis, LL, L, nshell, 4);
	
	if(II<JJ && KK<LL){
    for (int III = III1; III <= III2; III++) {
        for (int JJJ = JJJ1; JJJ <= JJJ2; JJJ++) {
            for (int KKK = KKK1; KKK <= KKK2; KKK++) {
                for (int LLL = LLL1; LLL <= LLL2; LLL++) {
				
					QUICKDouble Y = 0;
					//permutational symmetry
					if(III<=JJJ && KKK<=LLL)
					{	
						if(III*nbasis+JJJ<=KKK*nbasis+LLL)
							Y = LOC4(Y_Matrix, III-1, JJJ-1, KKK-1, LLL-1, nbasis, nbasis, nbasis, nbasis);
                    	else
							Y = LOC4(Y_Matrix, KKK-1, LLL-1, III-1, JJJ-1, nbasis, nbasis, nbasis, nbasis);
						
					}
						//first quarter transformation of ERI!
                        if(fabs(Y)>integralCutoff)
                        {
							for(int i3mp2=1; i3mp2<=nElec/2; i3mp2++)
                            {
                                int i3mp2new = i3mp2; // i3mp2new=nstepmp2s+i3mp2-1 where nstepmp2s should be 1
                                QUICKDouble atemp = LOC2(coefficient, KKK-1, i3mp2new-1, nbasis, nbasis)*Y;
                                QUICKDouble btemp = LOC2(coefficient, LLL-1, i3mp2new-1, nbasis, nbasis)*Y;
                                int IIInew = III- Ksumtype[II]+1;
                                int JJJnew = JJJ- Ksumtype[JJ]+1;
								LOC5(orbmp2i331,i3mp2-1,LLL-1,IIInew-1,JJJnew-1,0, nElec/2, nbasis, 10, 10, 2) += atemp;
								LOC5(orbmp2i331,i3mp2-1,LLL-1,JJJnew-1,IIInew-1,1, nElec/2, nbasis, 10, 10, 2) += atemp;
								LOC5(orbmp2i331,i3mp2-1,KKK-1,IIInew-1,JJJnew-1,0, nElec/2, nbasis, 10, 10, 2) += btemp;
								LOC5(orbmp2i331,i3mp2-1,KKK-1,JJJnew-1,IIInew-1,1, nElec/2, nbasis, 10, 10, 2) += btemp;				
                            }
                        }
                    }
                }
            }
        }
    }
	else	
	{
		for(int III=III1; III<=III2; III++)
        {
            if(MAX(III,JJJ1)<=JJJ2)
            {
                for(int JJJ=MAX(III,JJJ1);JJJ<=JJJ2;JJJ++)
                {
                    for(int KKK=KKK1; KKK<=KKK2; KKK++)
                    {
                        if(MAX(KKK,LLL1)<=LLL2)
                        {
                            for(int LLL=MAX(KKK,LLL1);LLL<=LLL2;LLL++)
                            {
								QUICKDouble Y = 0;
								
								if(III*nbasis+JJJ<=KKK*nbasis+LLL)
									Y = LOC4(Y_Matrix, III-1, JJJ-1, KKK-1, LLL-1, nbasis, nbasis, nbasis, nbasis);
								else
                            		Y = LOC4(Y_Matrix, KKK-1, LLL-1, III-1, JJJ-1, nbasis, nbasis, nbasis, nbasis);								
								
								if(fabs(Y)>integralCutoff)
                                {
									for(int i3mp2=1; i3mp2<=nElec/2; i3mp2++)
                                    {   
                                        int i3mp2new = i3mp2;
                                        QUICKDouble atemp = LOC2(coefficient, KKK-1, i3mp2new-1, nbasis, nbasis)*Y;
                                        QUICKDouble btemp = LOC2(coefficient, LLL-1, i3mp2new-1, nbasis, nbasis)*Y;
                                        int IIInew = III - Ksumtype[II]+1; 
                                        int JJJnew = JJJ - Ksumtype[JJ]+1; 
                                        LOC5(orbmp2i331, i3mp2-1, LLL-1, IIInew-1, JJJnew-1, 0, nElec/2, nbasis, 10, 10, 2) += atemp;
                                        
                                        if(JJJ != III)
                                        {   
                                            LOC5(orbmp2i331, i3mp2-1, LLL-1, JJJnew-1, IIInew-1, 1, nElec/2, nbasis, 10, 10, 2) += atemp;
                                        }   
                                        if(KKK != LLL)
                                        {   
                                            LOC5(orbmp2i331, i3mp2-1, KKK-1, IIInew-1, JJJnew-1, 0, nElec/2, nbasis, 10, 10, 2) += btemp;
                                            if(III != JJJ)
                                            {   
                                                LOC5(orbmp2i331, i3mp2-1, KKK-1, JJJnew-1, IIInew-1, 1, nElec/2, nbasis, 10, 10, 2) += btemp;
                                            }   
                                        }   
                                    } 
								}
							}
						}		
					}	
				}
			}
		}		
	}			
}
*/

/*
//This is a 2D-2D(grid and block) version of the forthQuarterTransKernel below
//Problem at: LOC2(orbmp2,l3,j3,nbasis-nElec/2,nbasis-nElec/2)=0.0
//__launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1)
__global__ void forthQuarterTransKernel2D2D(QUICKDouble* orbmp2k331, QUICKDouble* orbmp2, QUICKDouble* MP2cor)
{
	int nElec = devSim_MP2.nElec;
	int nsteplength = devSim_MP2.nElec/2;
    int nbasis = devSim_MP2.nbasis;
    QUICKDouble* coefficient = devSim_MP2.coefficient;
    QUICKDouble* molorbe = devSim_MP2.molorbe;

	int icycle = blockIdx.x;
	int i3 = icycle;
	int k3 = blockIdx.y;
	//think of adding warning if not enough blocks and threads:
	if(blockIdx.x==0 && blockIdx.y==0 && threadIdx.x==0 && threadIdx.y==0)
	{
		if(gridDim.x<nsteplength || gridDim.y<nsteplength)
			printf("\nNOT ENOUGH BLOCKS!!\n\n");
	}	
	if(icycle>=0&& icycle<nsteplength && k3>=i3 && k3<nElec/2)
	{
		int j3 = threadIdx.x;
		int l3 = threadIdx.y;
		//think of adding warning if not enough blocks and threads:
		if(blockIdx.x==0 && blockIdx.y==0 && threadIdx.x==0 && threadIdx.y==0)
		{
			if(blockDim.x<(nbasis-nElec/2) || blockDim.y<(nbasis-nElec/2))
                printf("\nNOT ENOUGH THREADS!\n\n");
		}
		if(j3>0 && j3<nbasis-nElec/2 && l3>=0 && l3<nbasis-nElec/2)
		{
			//problem here. For thread(l3,j3) in each block LOC2(orbmp2,l3,j3,nbasis-nElec/2,nbasis-nElec/2) is zeroed.
			LOC2(orbmp2,l3,j3,nbasis-nElec/2,nbasis-nElec/2)=0.0;
            int l3new = l3 + nElec/2;
            for(int lll=0;lll<nbasis;lll++)
            {
                LOC2(orbmp2,l3,j3,nbasis-nElec/2, nbasis-nElec/2) += \
                            LOC4(orbmp2k331,icycle,k3,j3,lll, nElec/2,nElec/2, \
                                (nbasis-nElec/2),nbasis)*LOC2(coefficient,lll,l3new, nbasis, nbasis);
            	//printf("Now orbmp2 is %lf\n", LOC2(orbmp2,l3,j3,nbasis-nElec/2, nbasis-nElec/2)); 
			}
            __syncthreads();
			if(k3>i3)
            {
                QUICKADD(MP2cor[0], 2.0/(molorbe[i3]+molorbe[k3]-molorbe[j3+nElec/2]-molorbe[l3+nElec/2]) \
                                *LOC2(orbmp2,j3,l3,nbasis-nElec/2, nbasis-nElec/2) \
                                *(2.0*LOC2(orbmp2,j3,l3,nbasis-nElec/2, nbasis-nElec/2) \
                                - LOC2(orbmp2,l3,j3,nbasis-nElec/2, nbasis-nElec/2)));
            	printf("updated MP2cor[0] is %lf\n",MP2cor[0]);
			}
            if(k3==i3)
            {
                QUICKADD(MP2cor[0],1.0/(molorbe[i3]+molorbe[k3]-molorbe[j3+nElec/2]-molorbe[l3+nElec/2]) \
                                *LOC2(orbmp2,j3,l3,nbasis-nElec/2, nbasis-nElec/2) \
                                *(2.0*LOC2(orbmp2,j3,l3,nbasis-nElec/2, nbasis-nElec/2) \
                                - LOC2(orbmp2,l3,j3,nbasis-nElec/2, nbasis-nElec/2)));
            	printf("updated MP2cor[0] is %lf\n",MP2cor[0]);
			}
            __syncthreads();
	
		}
	}
	//if(blockIdx.x==0 && blockIdx.y==0 && threadIdx.x==0 && threadIdx.y==0)
    //{
        //printf("gridDim.x is %d, blockDim.x is %d\n", gridDim.x, blockDim.x);
        printf("At the end of forthQuarterTransKernel2D2D, final MP2 correction is %lf\n", MP2cor[0]);
    //}
    //__syncthreads();
}
*/

/*
// probably need 2d grid and 2d blocks
//__launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1)
//also problem at: LOC2(orbmp2,l3,j3,nbasis-nElec/2,nbasis-nElec/2)=0.0;
__global__ void forthQuarterTransKernel(QUICKDouble* orbmp2k331, QUICKDouble* orbmp2, QUICKDouble* MP2cor)
{
	int nElec = devSim_MP2.nElec;	
	int nsteplength = devSim_MP2.nElec/2;
	int nbasis = devSim_MP2.nbasis;
	QUICKDouble* coefficient = devSim_MP2.coefficient;
	QUICKDouble* molorbe = devSim_MP2.molorbe;
	
	int icycle = blockIdx.x/nsteplength;
	int i3 = icycle;
	int k3 = blockIdx.x%nsteplength;
	//think of adding warning if not enough blocks and threads:
	if(blockIdx.x==0 && threadIdx.x==0)
	{
		if(gridDim.x < nsteplength*nsteplength)
			printf("\nNOT ENOUGH BLOCKS!!\n\n");
	}

	if(icycle>=0 && icycle<nsteplength && k3>=i3 && k3<nElec/2)
	{
		int j3 = threadIdx.x/(nbasis-nElec/2);
		int l3 = threadIdx.x%(nbasis-nElec/2);
		//think of adding warning if not enough blocks and threads:
		if(blockIdx.x==0 && threadIdx.x==0)
    	{	
			if(blockDim.x<(nbasis-nElec/2)*(nbasis-nElec/2))	
				printf("\nNOT ENOUGH THREADS!\n\n");
		}

		if(j3>=0 && j3<nbasis-nElec/2 && l3>=0 && l3<nbasis-nElec/2)
		{
			//problem here. For thread(l3,j3) in each block LOC2(orbmp2,l3,j3,nbasis-nElec/2,nbasis-nElec/2) is zeroed?
			LOC2(orbmp2,l3,j3,nbasis-nElec/2,nbasis-nElec/2)=0.0;
			int l3new = l3 + nElec/2;
			for(int lll=0;lll<nbasis;lll++)
			{
				LOC2(orbmp2,l3,j3,nbasis-nElec/2, nbasis-nElec/2) += \
                            LOC4(orbmp2k331,icycle,k3,j3,lll, nElec/2,nElec/2, \
                                (nbasis-nElec/2),nbasis)*LOC2(coefficient,lll,l3new, nbasis, nbasis);		
				//printf("Now orbmp2 is %lf\n", LOC2(orbmp2,l3,j3,nbasis-nElec/2, nbasis-nElec/2));	
			}
			__syncthreads();
			if(k3>i3)
			{
				//MP2cor[0] += 2.0/(molorbe[i3]+molorbe[k3]-molorbe[j3+nElec/2]-molorbe[l3+nElec/2]) \
                                *LOC2(orbmp2,j3,l3,nbasis-nElec/2, nbasis-nElec/2) \
                                *(2.0*LOC2(orbmp2,j3,l3,nbasis-nElec/2, nbasis-nElec/2) \
                                - LOC2(orbmp2,l3,j3,nbasis-nElec/2, nbasis-nElec/2));

				QUICKADD(MP2cor[0], 2.0/(molorbe[i3]+molorbe[k3]-molorbe[j3+nElec/2]-molorbe[l3+nElec/2]) \
                                *LOC2(orbmp2,j3,l3,nbasis-nElec/2, nbasis-nElec/2) \
                                *(2.0*LOC2(orbmp2,j3,l3,nbasis-nElec/2, nbasis-nElec/2) \
                                - LOC2(orbmp2,l3,j3,nbasis-nElec/2, nbasis-nElec/2)));
			}
			if(k3==i3)
			{
				//MP2cor[0] += 1.0/(molorbe[i3]+molorbe[k3]-molorbe[j3+nElec/2]-molorbe[l3+nElec/2]) \
                                *LOC2(orbmp2,j3,l3,nbasis-nElec/2, nbasis-nElec/2) \
                                *(2.0*LOC2(orbmp2,j3,l3,nbasis-nElec/2, nbasis-nElec/2) \
                                - LOC2(orbmp2,l3,j3,nbasis-nElec/2, nbasis-nElec/2));
				QUICKADD(MP2cor[0],1.0/(molorbe[i3]+molorbe[k3]-molorbe[j3+nElec/2]-molorbe[l3+nElec/2]) \
                                *LOC2(orbmp2,j3,l3,nbasis-nElec/2, nbasis-nElec/2) \
                                *(2.0*LOC2(orbmp2,j3,l3,nbasis-nElec/2, nbasis-nElec/2) \
                                - LOC2(orbmp2,l3,j3,nbasis-nElec/2, nbasis-nElec/2)));
			}
			__syncthreads();
		}
	}
	if(blockIdx.x==0 && threadIdx.x==0)
	{
		printf("gridDim.x is %d, blockDim.x is %d\n", gridDim.x, blockDim.x);
		printf("At the end of forthQuarterTransKernel, final MP2 correction is %lf\n", MP2cor[0]);
	}
	//__syncthreads();
}
*/

__global__ void
__launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) forthQuarterTransInnerLoopsKernel(int icycle, int i3, int k3, int nstep, QUICKDouble* orbmp2k331, QUICKDouble* orbmp2)
{
	QUICKDouble* coefficient = devSim_MP2.coefficient;
	int nElec = devSim_MP2.nElec;
	int nbasis = devSim_MP2.nbasis;
	int ivir = nbasis-nElec/2;
	int iocc = nElec/2;

	int offside = blockIdx.x*blockDim.x+threadIdx.x;
	int totalThreads = blockDim.x*gridDim.x;
	int myInt = (nbasis-nElec/2)*(nbasis-nElec/2)/totalThreads;

	if((nbasis-nElec/2)*(nbasis-nElec/2)-myInt*totalThreads>offside) 
		myInt++;

	for(int i=1;i<=myInt;i++)
	{
		int currentInt = totalThreads*(i-1)+offside;
		int j3 = currentInt/(nbasis-nElec/2);
		int l3 = currentInt%(nbasis-nElec/2);

		if(j3<nbasis-nElec/2 && l3<nbasis-nElec/2)
		{
			LOC2(orbmp2,l3,j3,ivir,ivir)=0.0;
        	int l3new = l3 + nElec/2;
			for(int lll=0;lll<nbasis;lll++)
        	{
				//should not need atomicadd
        		LOC2(orbmp2,l3,j3,ivir,ivir) += \
            		LOC4(orbmp2k331,icycle,k3,j3,lll, nstep,iocc,ivir,nbasis)*LOC2(coefficient,lll,l3new, nbasis, nbasis);
        	}
		}
	}		
}

__global__ void 
__launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) finalMP2AccumulationInnerLoopsKernel(int i3, int k3, QUICKDouble* orbmp2, QUICKDouble* MP2cor)
{
	//QUICKDouble* coefficient = devSim_MP2.coefficient;
    QUICKDouble* molorbe = devSim_MP2.molorbe;
    int nElec = devSim_MP2.nElec;
    int nbasis = devSim_MP2.nbasis;

    int offside = blockIdx.x*blockDim.x+threadIdx.x;
	int totalThreads = blockDim.x*gridDim.x;
    int myInt = (nbasis-nElec/2)*(nbasis-nElec/2)/totalThreads; 

	if((nbasis-nElec/2)*(nbasis-nElec/2)-myInt*totalThreads>offside)
        myInt++;
  
	for(int i=1;i<=myInt;i++)
	{
		int currentInt = totalThreads*(i-1)+offside;
    	int j3 = currentInt/(nbasis-nElec/2);
    	int l3 = currentInt%(nbasis-nElec/2);

    	if(j3<nbasis-nElec/2 && l3<nbasis-nElec/2)
    	{	
        	if(k3>i3)
        	{
            	QUICKADD(MP2cor[0], 2.0/(molorbe[i3]+molorbe[k3]-molorbe[j3+nElec/2]-molorbe[l3+nElec/2]) \
                                *LOC2(orbmp2,j3,l3,nbasis-nElec/2, nbasis-nElec/2) \
                                *(2.0*LOC2(orbmp2,j3,l3,nbasis-nElec/2, nbasis-nElec/2) \
                                - LOC2(orbmp2,l3,j3,nbasis-nElec/2, nbasis-nElec/2)));

        	}
        	if(k3==i3)
       		{
            	QUICKADD(MP2cor[0],1.0/(molorbe[i3]+molorbe[k3]-molorbe[j3+nElec/2]-molorbe[l3+nElec/2]) \
                                *LOC2(orbmp2,j3,l3,nbasis-nElec/2, nbasis-nElec/2) \
                                *(2.0*LOC2(orbmp2,j3,l3,nbasis-nElec/2, nbasis-nElec/2) \
                                - LOC2(orbmp2,l3,j3,nbasis-nElec/2, nbasis-nElec/2)));
        	}
    	}
	}
}

/*
void forthQuarterTransHost(QUICKDouble* orbmp2k331, QUICKDouble* orbmp2, int nstep,  _gpu_type gpu)
{
    int nElec = gpu->nElec;
    int nsteplength = gpu->nElec/2;
    int nbasis = gpu->nbasis;

    //QUICKDouble* coefficient = gpu->gpu_calculated->coefficient->_hostData;
    //QUICKDouble* molorbe = gpu->gpu_calculated->molorbe->_hostData;

	QUICKDouble* orbmp2k331_d;
    QUICKDouble* orbmp2_d;
    QUICKDouble* MP2cor_d;
	QUICKDouble* MP2cor = new QUICKDouble[1];
	MP2cor[0]= 0;

	cudaMalloc((void **)&orbmp2k331_d,sizeof(QUICKDouble)*nElec/2*nElec/2*(nbasis-nElec/2)*nbasis);
	cudaMemcpy(orbmp2k331_d, orbmp2k331, sizeof(QUICKDouble)*nElec/2*nElec/2*(nbasis-nElec/2)*nbasis, cudaMemcpyHostToDevice);

	cudaMalloc((void **)&orbmp2_d,sizeof(QUICKDouble)*(nbasis-nElec/2)*(nbasis-nElec/2));
	cudaMemset(orbmp2_d,0,sizeof(QUICKDouble)*(nbasis-nElec/2)*(nbasis-nElec/2));

	cudaMalloc((void **)&MP2cor_d,sizeof(QUICKDouble));
    cudaMemset(MP2cor_d, 0, sizeof(QUICKDouble));
	

    for(int icycle=0;icycle<nsteplength;icycle++) //block level 1
    {
        int i3 = icycle;
        for(int k3=i3;k3<nElec/2;k3++) //block level 2
        {
			forthQuarterTransInnerLoopsKernel<<<gpu->blocks, gpu->twoEThreadsPerBlock>>>(icycle, i3, k3,nstep, orbmp2k331_d, orbmp2_d);		
			cudaDeviceSynchronize();
			finalMP2AccumulationInnerLoopsKernel<<<gpu->blocks, gpu->twoEThreadsPerBlock>>>(i3, k3, orbmp2_d, MP2cor_d);
			cudaDeviceSynchronize();		
*/	
			/*
            for(int j3=0;j3<nbasis-nElec/2;j3++) //thread level 1
            {
                for(int l3=0;l3<nbasis-nElec/2;l3++) // thread level 2
                {
                    LOC2(orbmp2,l3,j3,nbasis-nElec/2,nbasis-nElec/2)=0.0;
                    int l3new = l3 + nElec/2;
                    for(int lll=0;lll<nbasis;lll++)
                    {
                        LOC2(orbmp2,l3,j3,nbasis-nElec/2, nbasis-nElec/2) += \
                            LOC4(orbmp2k331,icycle,k3,j3,lll, nElec/2,nElec/2, \
                                (nbasis-nElec/2),nbasis)*LOC2(coefficient,lll,l3new, nbasis, nbasis);
                    }
                }
            }
            // sync all threads in a block
            for(int j3=0;j3<nbasis-nElec/2;j3++)
            {
                for(int l3=0;l3<nbasis-nElec/2;l3++)
                {
                    if(k3>i3)
                    {
                        *MP2cor += 2.0/(molorbe[i3]+molorbe[k3]-molorbe[j3+nElec/2]-molorbe[l3+nElec/2]) \
                                *LOC2(orbmp2,j3,l3,nbasis-nElec/2, nbasis-nElec/2) \
                                *(2.0*LOC2(orbmp2,j3,l3,nbasis-nElec/2, nbasis-nElec/2) \
                                - LOC2(orbmp2,l3,j3,nbasis-nElec/2, nbasis-nElec/2));
                    }
                    if(k3==i3)
                    {
                        *MP2cor += 1.0/(molorbe[i3]+molorbe[k3]-molorbe[j3+nElec/2]-molorbe[l3+nElec/2]) \
                                *LOC2(orbmp2,j3,l3,nbasis-nElec/2, nbasis-nElec/2) \
                                *(2.0*LOC2(orbmp2,j3,l3,nbasis-nElec/2, nbasis-nElec/2) \
                                - LOC2(orbmp2,l3,j3,nbasis-nElec/2, nbasis-nElec/2));
                    }
                }
            }
			*/
/*			
        }
    }
	cudaMemcpy(MP2cor, MP2cor_d, sizeof(QUICKDouble),cudaMemcpyDeviceToHost);
    printf("On the CUDA side, final MP2 correction is %lf\n", *MP2cor);

	cudaFree(orbmp2k331_d);
    cudaFree(orbmp2_d);
    cudaFree(MP2cor_d);
	delete[] MP2cor;
}
*/

/*
//This version of forthQuarterTransHost is depreciated due to the index offset. Use the one above.
void forthQuarterTransHost(QUICKDouble* orbmp2k331, QUICKDouble* orbmp2, _gpu_type gpu)
{
	QUICKDouble MP2cor = 0.0;
	int nElec = gpu->nElec;
    int nsteplength = gpu->nElec/2;
	int nbasis = gpu->nbasis;
	
    QUICKDouble* coefficient = gpu->gpu_calculated->coefficient->_hostData;
	QUICKDouble* molorbe = gpu->gpu_calculated->molorbe->_hostData;

	for(int icycle=1;icycle<=nsteplength;icycle++) //block level 1
    {   
        int i3 = icycle;
        for(int k3=i3;k3<=nElec/2;k3++)	//block level 2
        {   
            for(int j3=1;j3<=nbasis-nElec/2;j3++) //thread level 1
            {   
                for(int l3=1;l3<=nbasis-nElec/2;l3++) // thread level 2
                {   
                    LOC2(orbmp2,l3-1,j3-1,nbasis-nElec/2,nbasis-nElec/2)=0.0;
                    int l3new = l3 + nElec/2;
                    for(int lll=1;lll<=nbasis;lll++)
                    {
						LOC2(orbmp2,l3-1,j3-1,nbasis-nElec/2, nbasis-nElec/2) += \
                            LOC4(orbmp2k331,icycle-1,k3-1,j3-1,lll-1, nElec/2,nElec/2, \
                                (nbasis-nElec/2),nbasis)*LOC2(coefficient,lll-1,l3new-1, nbasis, nbasis);
                	}
				}
            }
			// sync all threads in a block
			for(int j3=1;j3<=nbasis-nElec/2;j3++)
            {
                for(int l3=1;l3<=nbasis-nElec/2;l3++)
                {
                    if(k3>i3)
                    {
                        MP2cor += 2.0/(molorbe[i3-1]+molorbe[k3-1]-molorbe[j3-1+nElec/2]-molorbe[l3-1+nElec/2]) \
                                *LOC2(orbmp2,j3-1,l3-1,nbasis-nElec/2, nbasis-nElec/2) \
                                *(2.0*LOC2(orbmp2,j3-1,l3-1,nbasis-nElec/2, nbasis-nElec/2) \
                                - LOC2(orbmp2,l3-1,j3-1,nbasis-nElec/2, nbasis-nElec/2));
					}
                    if(k3==i3)
                    {
                        MP2cor += 1.0/(molorbe[i3-1]+molorbe[k3-1]-molorbe[j3-1+nElec/2]-molorbe[l3-1+nElec/2]) \
                                *LOC2(orbmp2,j3-1,l3-1,nbasis-nElec/2, nbasis-nElec/2) \
                                *(2.0*LOC2(orbmp2,j3-1,l3-1,nbasis-nElec/2, nbasis-nElec/2) \
                                - LOC2(orbmp2,l3-1,j3-1,nbasis-nElec/2, nbasis-nElec/2));
					}
                }
            }
		}
	}
	printf("On the CUDA side, final MP2 correction is %lf\n", MP2cor);	
}
*/

/*
__device__ void firstQuarterTransDevice(int I, int J, int K, int L, unsigned int II, unsigned int JJ, unsigned int KK, unsigned int LL, QUICKDouble DNMax)
{

	int III1 = LOC2(devSim_MP2.Qsbasis, II, I, devSim_MP2.nshell, 4);
    int III2 = LOC2(devSim_MP2.Qfbasis, II, I, devSim_MP2.nshell, 4);
    int JJJ1 = LOC2(devSim_MP2.Qsbasis, JJ, J, devSim_MP2.nshell, 4);
    int JJJ2 = LOC2(devSim_MP2.Qfbasis, JJ, J, devSim_MP2.nshell, 4);
    int KKK1 = LOC2(devSim_MP2.Qsbasis, KK, K, devSim_MP2.nshell, 4);
    int KKK2 = LOC2(devSim_MP2.Qfbasis, KK, K, devSim_MP2.nshell, 4);
    int LLL1 = LOC2(devSim_MP2.Qsbasis, LL, L, devSim_MP2.nshell, 4);
    int LLL2 = LOC2(devSim_MP2.Qfbasis, LL, L, devSim_MP2.nshell, 4);

	if(II<JJ && KK<LL){
    for (int III = III1; III <= III2; III++) {
        //for (int JJJ = MAX(III,JJJ1); JJJ <= JJJ2; JJJ++) {
        for (int JJJ = JJJ1; JJJ <= JJJ2; JJJ++) {
            //for (int KKK = MAX(III,KKK1); KKK <= KKK2; KKK++) {
            for (int KKK = KKK1; KKK <= KKK2; KKK++) {
                //for (int LLL = MAX(KKK,LLL1); LLL <= LLL2; LLL++) {
                for (int LLL = LLL1; LLL <= LLL2; LLL++) {
						
						QUICKDouble Y = LOC4(devSim_MP2.Y_Matrix, III-1, JJJ-1, KKK-1, LLL-1, devSim_MP2.nbasis, devSim_MP2.nbasis, devSim_MP2.nbasis, devSim_MP2.nbasis);
                        //first quarter transformation of ERI!
						if(fabs(Y)>devSim_MP2.integralCutoff)
                        {
                            for(int i3mp2=1; i3mp2<=devSim_MP2.nElec/2; i3mp2++)
                            {
                                int i3mp2new = i3mp2; // i3mp2new=nstepmp2s+i3mp2-1 where nstepmp2s should be 1
                                QUICKDouble atemp = LOC2(devSim_MP2.coefficient, KKK-1, i3mp2new-1, devSim_MP2.nbasis, devSim_MP2.nbasis)*Y;
                                QUICKDouble btemp = LOC2(devSim_MP2.coefficient, LLL-1, i3mp2new-1, devSim_MP2.nbasis, devSim_MP2.nbasis)*Y;
                                int IIInew = III- devSim_MP2.Ksumtype[II]+1;
                                int JJJnew = JJJ- devSim_MP2.Ksumtype[JJ]+1;
                                //printf("in gpu_MP2.cu/iclass_MP2, III, JJJ, KKK, LLL and II, JJ are %d %d %d %d and %d %d %d %d\n",\
                                        III, JJJ, KKK, LLL, II, JJ, KK, LL);
                                QUICKADD(LOC5(devSim_MP2.orbmp2i331,i3mp2-1,LLL-1,IIInew-1,JJJnew-1,0, devSim_MP2.nElec/2, devSim_MP2.nbasis, 10, 10, 2), atemp);
                                QUICKADD(LOC5(devSim_MP2.orbmp2i331,i3mp2-1,LLL-1,JJJnew-1,IIInew-1,1, devSim_MP2.nElec/2, devSim_MP2.nbasis, 10, 10, 2), atemp);
                                QUICKADD(LOC5(devSim_MP2.orbmp2i331,i3mp2-1,KKK-1,IIInew-1,JJJnew-1,0, devSim_MP2.nElec/2, devSim_MP2.nbasis, 10, 10, 2), btemp);
                                QUICKADD(LOC5(devSim_MP2.orbmp2i331,i3mp2-1,KKK-1,JJJnew-1,IIInew-1,1, devSim_MP2.nElec/2, devSim_MP2.nbasis, 10, 10, 2), btemp);
                                //LOC5(orbmp2i331,i3mp2-1,LLL-1,IIInew-1,JJJnew-1,0, devSim_MP2.nElec/2, devSim_MP2.nbasis, 6, 6, 2) += atemp;
                                //LOC5(orbmp2i331,i3mp2-1,LLL-1,JJJnew-1,IIInew-1,1, devSim_MP2.nElec/2, devSim_MP2.nbasis, 6, 6, 2) += atemp;
                                //LOC5(orbmp2i331,i3mp2-1,KKK-1,IIInew-1,JJJnew-1,0, devSim_MP2.nElec/2, devSim_MP2.nbasis, 6, 6, 2) += btemp;
                                //LOC5(orbmp2i331,i3mp2-1,KKK-1,JJJnew-1,IIInew-1,1, devSim_MP2.nElec/2, devSim_MP2.nbasis, 6, 6, 2) += btemp;
                            }
                        }
					}
				}
			}
		}   
	}
	else
    {   
        for(int III=III1; III<=III2; III++)
        {   
            if(MAX(III,JJJ1)<=JJJ2)
            {   
                for(int JJJ=MAX(III,JJJ1);JJJ<=JJJ2;JJJ++)
                {   
                    for(int KKK=KKK1; KKK<=KKK2; KKK++)
                    {   
                        if(MAX(KKK,LLL1)<=LLL2)
                        {   
                            for(int LLL=MAX(KKK,LLL1);LLL<=LLL2;LLL++)
                            {   
                                //printf("in gpu_MP2.cu/iclass_MP2, after hrrwhole_MP2, III, JJJ, KKK, LLL, and Y are %d %d %d %d %lf\n", III, JJJ, KKK, LLL, Y);   
                                QUICKDouble Y = LOC4(devSim_MP2.Y_Matrix, III-1, JJJ-1, KKK-1, LLL-1, devSim_MP2.nbasis, devSim_MP2.nbasis, devSim_MP2.nbasis, devSim_MP2.nbasis);
                                //printf("else in gpu_MP2.cu/iclass_MP2, III,JJJ,KKK,LLL, and II,JJ,KK,LL, and I,J,K,L are %d %d %d %d,  %d %d %d %d,  %d %d %d %d\n", \
                                III,JJJ,KKK,LLL, II,JJ,KK,LL, I,J,K,L);
                                            
                                if(fabs(Y)>devSim_MP2.integralCutoff)
                                {   
                                    for(int i3mp2=1; i3mp2<=devSim_MP2.nElec/2; i3mp2++)
                                    {   
                                        int i3mp2new = i3mp2;
                                        QUICKDouble atemp = LOC2(devSim_MP2.coefficient, KKK-1, i3mp2new-1, devSim_MP2.nbasis, devSim_MP2.nbasis)*Y;
                                        QUICKDouble btemp = LOC2(devSim_MP2.coefficient, LLL-1, i3mp2new-1, devSim_MP2.nbasis, devSim_MP2.nbasis)*Y;
                                        int IIInew = III - devSim_MP2.Ksumtype[II]+1; 
                                        int JJJnew = JJJ - devSim_MP2.Ksumtype[JJ]+1; 
                                        QUICKADD(LOC5(devSim_MP2.orbmp2i331, i3mp2-1, LLL-1, IIInew-1, JJJnew-1, 0, devSim_MP2.nElec/2, devSim_MP2.nbasis, 10, 10, 2),atemp);
                                        //LOC5(orbmp2i331, i3mp2-1, LLL-1, IIInew-1, JJJnew-1, 0, devSim_MP2.nElec/2, devSim_MP2.nbasis, 6, 6, 2) += atemp;
                                        
                                        if(JJJ != III)
                                        {   
                                            QUICKADD(LOC5(devSim_MP2.orbmp2i331, i3mp2-1, LLL-1, JJJnew-1, IIInew-1, 1, devSim_MP2.nElec/2, devSim_MP2.nbasis, 10, 10, 2),atemp);
                                            //LOC5(orbmp2i331, i3mp2-1, LLL-1, JJJnew-1, IIInew-1, 1, devSim_MP2.nElec/2, devSim_MP2.nbasis, 6, 6, 2) += atemp;
                                        }
                                        if(KKK != LLL)
                                        {   
                                            QUICKADD(LOC5(devSim_MP2.orbmp2i331, i3mp2-1, KKK-1, IIInew-1, JJJnew-1, 0, devSim_MP2.nElec/2, devSim_MP2.nbasis, 10, 10, 2),btemp);
                                            //LOC5(orbmp2i331, i3mp2-1, KKK-1, IIInew-1, JJJnew-1, 0, devSim_MP2.nElec/2, devSim_MP2.nbasis, 6, 6, 2) += btemp;
                                            if(III != JJJ)
                                            {   
                                                QUICKADD(LOC5(devSim_MP2.orbmp2i331, i3mp2-1, KKK-1, JJJnew-1, IIInew-1, 1, devSim_MP2.nElec/2, devSim_MP2.nbasis, 10, 10, 2),btemp);
                                                //LOC5(orbmp2i331, i3mp2-1, KKK-1, JJJnew-1, IIInew-1, 1, devSim_MP2.nElec/2, devSim_MP2.nbasis, 6, 6, 2) += btemp;
                                            }
                                        }
                                    }
                                
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
*/



// totTime is the timer for GPU 2e time. Only on under debug mode

#ifdef DEBUG
static float totTime;
#endif

void get2e_MP2(_gpu_type gpu)
{
#ifdef DEBUG
    cudaEvent_t start,end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);
#endif
	float mp2_time;
	cudaEvent_t mp2_start, mp2_allocate,mp2_trans;
	cudaEventCreate(&mp2_start);
	cudaEventCreate(&mp2_allocate);
	cudaEventCreate(&mp2_trans);
	cudaEventRecord(mp2_start,0);
    
	cudaDeviceSynchronize();

	QUICKDouble* orbmp2i331 = new QUICKDouble[gpu->nElec/2*gpu->nbasis*10*10*2];	
	memset(orbmp2i331, 0, sizeof(QUICKDouble)*gpu->nElec/2*gpu->nbasis*10*10*2);
	QUICKDouble* orbmp2j331 = new QUICKDouble[gpu->nElec/2*(gpu->nbasis-gpu->nElec/2)*10*10*2]; 
	memset(orbmp2j331, 0, sizeof(QUICKDouble)*gpu->nElec/2*(gpu->nbasis-gpu->nElec/2)*10*10*2);
	QUICKDouble* orbmp2k331 = new QUICKDouble[gpu->nElec/2*gpu->nElec/2*(gpu->nbasis-gpu->nElec/2)*gpu->nbasis];
	memset(orbmp2k331, 0, sizeof(QUICKDouble)*gpu->nElec/2*gpu->nElec/2*(gpu->nbasis-gpu->nElec/2)*gpu->nbasis);
	QUICKDouble* orbmp2 = new QUICKDouble[(gpu->nbasis-gpu->nElec/2)*(gpu->nbasis-gpu->nElec/2)]; 
	memset(orbmp2, 0, sizeof(QUICKDouble)*(gpu->nbasis-gpu->nElec/2)*(gpu->nbasis-gpu->nElec/2));

	cudaEventRecord(mp2_allocate,0);
    cudaEventSynchronize(mp2_allocate);
    cudaEventElapsedTime(&mp2_time,mp2_start,mp2_allocate);
    printf("in get2e_MP2, total gpu mp2 tensor allocation time is %6.3f ms\n", mp2_time);

	//all four quarter transformation
	fourQuarterTransHost(orbmp2i331, orbmp2j331, orbmp2k331, orbmp2, gpu);
	cudaEventRecord(mp2_trans,0);
    cudaEventSynchronize(mp2_trans);
    cudaEventElapsedTime(&mp2_time,mp2_allocate,mp2_trans);
    printf("in get2e_MP2, total gpu mp2 four quarter transformation time is %6.3f ms\n", mp2_time);

	/*
	firstThreeQuartersTransHost(orbmp2i331, orbmp2j331, orbmp2k331, gpu);	
	cudaEventRecord(mp2_first3,0);
    cudaEventSynchronize(mp2_first3);
    cudaEventElapsedTime(&mp2_time,mp2_allocate,mp2_first3);
    printf("in get2e_MP2, total gpu first three transformation time is %6.3f ms\n", mp2_time);
	*/

	/*
	//4th Quarter Trans Parallelized
	forthQuarterTransHost(orbmp2k331,orbmp2, gpu);
	cudaEventRecord(mp2_forth,0);
    cudaEventSynchronize(mp2_forth);
    cudaEventElapsedTime(&mp2_time,mp2_first3,mp2_forth);
    printf("in get2e_MP2, gpu forth transformation time is %6.3f ms\n", mp2_time);
	*/	

	delete[] orbmp2i331;
	delete[] orbmp2j331;
	delete[] orbmp2k331;
	delete[] orbmp2;

	cudaEventDestroy(mp2_start);
	cudaEventDestroy(mp2_start);
	cudaEventDestroy(mp2_allocate);
	cudaEventDestroy(mp2_trans);
	//cudaEventDestroy(mp2_first3);
	//cudaEventDestroy(mp2_forth);

#ifdef DEBUG
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float time;
    cudaEventElapsedTime(&time, start, end);
    totTime+=time;
    //printf("this cycle:%f ms total time:%f ms\n", time, totTime);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
#endif
    printf("get2e_MP2 is done\n");
}

/*
__global__ void 
__launch_bounds__(SM_2X_2E_THREADS_PER_BLOCK, 1) get2e_MP2_kernel()
{

    unsigned int offside = blockIdx.x*blockDim.x+threadIdx.x;
    int totalThreads = blockDim.x*gridDim.x;
	//printf("in get2e_MP2_kernel, totalThread is %d, offside is %d\n", totalThreads, offside);

    QUICKULL jshell   = (QUICKULL) devSim_MP2.sqrQshell;
    QUICKULL myInt    = (QUICKULL) jshell*jshell / totalThreads;
	

    if ((jshell*jshell - myInt*totalThreads)> offside) myInt++;	//256>offside?
    
	for (QUICKULL i = 1; i<=myInt; i++) {

  
        QUICKULL currentInt = totalThreads * (i-1)+offside;     //currentInt=offside   
        QUICKULL a = (QUICKULL) currentInt/jshell;
        QUICKULL b = (QUICKULL) (currentInt - a*jshell);
*/        
		//printf("in get2e_MP2_kernel/for, offside is %d\n",offside);
        /* 
         the following code is to implement a better index mode.
         The original one can be see as:
         Large....Small
         Large ->->->
         ...   ->->->
         Small ->->->
         
         now it changed to 
         Large....Small
         Large ->->|  -> ->
         ...   <-<-|  |  |
         Small <-<-<-<-  |
         <-<-<-<-<-
         Theortically, it can avoid divergence but after test, we
         find it has limited effect and something it will slows down
         because of the extra FP calculation required.
         */
        /*    
         QUICKULL a, b;
         double aa = (double)((currentInt+1)*1E-4);
         QUICKULL t = (QUICKULL)(sqrt(aa)*1E2);
         if ((currentInt+1)==t*t) {
         t--;
         }
         
         QUICKULL k = currentInt-t*t;
         if (k<=t) {
         a = k;
         b = t;
         }else {
         a = t;
         b = 2*t-k;
         }*/
        
/*        
        int II = devSim_MP2.sorted_YCutoffIJ[a].x;
        int JJ = devSim_MP2.sorted_YCutoffIJ[a].y;
        int KK = devSim_MP2.sorted_YCutoffIJ[b].x;
        int LL = devSim_MP2.sorted_YCutoffIJ[b].y;        
        
        int ii = devSim_MP2.sorted_Q[II];
        int jj = devSim_MP2.sorted_Q[JJ];
        int kk = devSim_MP2.sorted_Q[KK];
        int ll = devSim_MP2.sorted_Q[LL];

		//printf("for ii,jj,kk,ll = %d, %d, %d, %d, allocate local memory orbmp2i331\n", ii, jj,kk,ll);
		////QUICKDouble* orbmp2i331 = new QUICKDouble[devSim_MP2.nElec/2*devSim_MP2.nbasis*6*6*2];
		QUICKDouble* orbmp2i331 = new QUICKDouble[1];

			if(ii<=kk){
            int nshell = devSim_MP2.nshell;
            QUICKDouble DNMax = MAX(MAX(4.0*LOC2(devSim_MP2.cutMatrix, ii, jj, nshell, nshell), 4.0*LOC2(devSim_MP2.cutMatrix, kk, ll, nshell, nshell)),
                                    MAX(MAX(LOC2(devSim_MP2.cutMatrix, ii, ll, nshell, nshell),     LOC2(devSim_MP2.cutMatrix, ii, kk, nshell, nshell)),
                                        MAX(LOC2(devSim_MP2.cutMatrix, jj, kk, nshell, nshell),     LOC2(devSim_MP2.cutMatrix, jj, ll, nshell, nshell))));
            
            if ((LOC2(devSim_MP2.YCutoff, kk, ll, nshell, nshell) * LOC2(devSim_MP2.YCutoff, ii, jj, nshell, nshell))> devSim_MP2.integralCutoff && \
                (LOC2(devSim_MP2.YCutoff, kk, ll, nshell, nshell) * LOC2(devSim_MP2.YCutoff, ii, jj, nshell, nshell) * DNMax) > devSim_MP2.integralCutoff) {
                
                int iii = devSim_MP2.sorted_Qnumber[II];
                int jjj = devSim_MP2.sorted_Qnumber[JJ];
                int kkk = devSim_MP2.sorted_Qnumber[KK];
                int lll = devSim_MP2.sorted_Qnumber[LL];
                
				//printf("to call iclass_MP2 in kernel, iii, jjj, kkk, lll are %d, %d, %d, %d\n", iii, jjj, kkk, lll);
                iclass_MP2(iii, jjj, kkk, lll, ii, jj, kk, ll, DNMax, orbmp2i331); // Y should be added to orbmp2i331;                           				
			}
		}

		delete[] orbmp2i331;
    }
}
*/

/*
 sqr for double precision. there no internal function to do that in fast-math-lib of CUDA
 */
__device__ __forceinline__ QUICKDouble quick_dsqr_MP2(QUICKDouble a)
{
    return a*a;
}


/*
 iclass subroutine is to generate 2-electron intergral using HRR and VRR method, which is the most
 performance algrithem for electron intergral evaluation. See description below for details
 */
__device__ void iclass_MP2(int I, int J, int K, int L, unsigned int II, unsigned int JJ, unsigned int KK, unsigned int LL, int nstepmp2s, int nsteplength, int nstep, int nbasistemp, QUICKDouble DNMax, QUICKDouble* orbmp2i331)
{
	//int totalThreads = blockDim.x*gridDim.x;
	//unsigned int offside = blockIdx.x*blockDim.x+threadIdx.x;
    //printf("in iclass_MP2, totalThread is %d, offside is %d\n", totalThreads, offside);

	//unsigned int offside = blockIdx.x*blockDim.x+threadIdx.x;
	//if(offside==0)
	//printf("in iclass_MP2, II, JJ, KK, LL are %d, %d, %d, %d\n", II,JJ,KK,LL);
 	//printf("in iclass_MP2, I, J, K, L are %d, %d, %d, %d\n", I,J,K,L);   
    /* 
     kAtom A, B, C ,D is the coresponding atom for shell ii, jj, kk, ll
     and be careful with the index difference between Fortran and C++, 
     Fortran starts array index with 1 and C++ starts 0.
     
     
     RA, RB, RC, and RD are the coordinates for atom katomA, katomB, katomC and katomD, 
     which means they are corrosponding coorinates for shell II, JJ, KK, and LL.
     And we don't need the coordinates now, so we will not retrieve the data now.
     */
    QUICKDouble RAx = LOC2(devSim_MP2.xyz, 0 , devSim_MP2.katom[II]-1, 3, devSim_MP2.natom);
    QUICKDouble RAy = LOC2(devSim_MP2.xyz, 1 , devSim_MP2.katom[II]-1, 3, devSim_MP2.natom);
    QUICKDouble RAz = LOC2(devSim_MP2.xyz, 2 , devSim_MP2.katom[II]-1, 3, devSim_MP2.natom);
    
    QUICKDouble RCx = LOC2(devSim_MP2.xyz, 0 , devSim_MP2.katom[KK]-1, 3, devSim_MP2.natom);
    QUICKDouble RCy = LOC2(devSim_MP2.xyz, 1 , devSim_MP2.katom[KK]-1, 3, devSim_MP2.natom);
    QUICKDouble RCz = LOC2(devSim_MP2.xyz, 2 , devSim_MP2.katom[KK]-1, 3, devSim_MP2.natom);
    
    /*
     kPrimI, J, K and L indicates the primtive gaussian function number
     kStartI, J, K, and L indicates the starting guassian function for shell I, J, K, and L.
     We retrieve from global memory and save them to register to avoid multiple retrieve.
     */
    int kPrimI = devSim_MP2.kprim[II];
    int kPrimJ = devSim_MP2.kprim[JJ];
    int kPrimK = devSim_MP2.kprim[KK];
    int kPrimL = devSim_MP2.kprim[LL];
    
    int kStartI = devSim_MP2.kstart[II]-1;
    int kStartJ = devSim_MP2.kstart[JJ]-1;
    int kStartK = devSim_MP2.kstart[KK]-1;
    int kStartL = devSim_MP2.kstart[LL]-1;
 
	int nbasis = devSim_MP2.nbasis;   
    
    /*
     store saves temp contracted integral as [as|bs] type. the dimension should be allocatable but because
     of cuda limitation, we can not do that now. 
     
     See M.Head-Gordon and J.A.Pople, Jchem.Phys., 89, No.9 (1988) for VRR algrithem details.
     */
    QUICKDouble store[STOREDIM*STOREDIM];
    
    /*
     Initial the neccessary element for 
     */
    for (int i = Sumindex_MP2[K+1]+1; i<= Sumindex_MP2[K+L+2]; i++) {
        for (int j = Sumindex_MP2[I+1]+1; j<= Sumindex_MP2[I+J+2]; j++) {
            LOC2(store, j-1, i-1, STOREDIM, STOREDIM) = 0;
        }
    }
    
    for (int i = 0; i<kPrimI*kPrimJ;i++){
        int JJJ = (int) i/kPrimI;
        int III = (int) i-kPrimI*JJJ;
        /*
         In the following comments, we have I, J, K, L denote the primitive gaussian function we use, and
         for example, expo(III, ksumtype(II)) stands for the expo for the IIIth primitive guassian function for II shell, 
         we use I to express the corresponding index.
         AB = expo(I)+expo(J)
         --->                --->
         ->     expo(I) * xyz (I) + expo(J) * xyz(J)
         P  = ---------------------------------------
         expo(I) + expo(J)
         Those two are pre-calculated in CPU stage. 
         
         */
        int ii_start = devSim_MP2.prim_start[II];
        int jj_start = devSim_MP2.prim_start[JJ];
        
        QUICKDouble AB = LOC2(devSim_MP2.expoSum, ii_start+III, jj_start+JJJ, devSim_MP2.prim_total, devSim_MP2.prim_total);
        QUICKDouble Px = LOC2(devSim_MP2.weightedCenterX, ii_start+III, jj_start+JJJ, devSim_MP2.prim_total, devSim_MP2.prim_total);
        QUICKDouble Py = LOC2(devSim_MP2.weightedCenterY, ii_start+III, jj_start+JJJ, devSim_MP2.prim_total, devSim_MP2.prim_total);
        QUICKDouble Pz = LOC2(devSim_MP2.weightedCenterZ, ii_start+III, jj_start+JJJ, devSim_MP2.prim_total, devSim_MP2.prim_total);
        
        /*
         X1 is the contracted coeffecient, which is pre-calcuated in CPU stage as well.
         cutoffprim is used to cut too small prim gaussian function when bring density matrix into consideration.
         */
        QUICKDouble cutoffPrim = DNMax * LOC2(devSim_MP2.cutPrim, kStartI+III, kStartJ+JJJ, devSim_MP2.jbasis, devSim_MP2.jbasis);
        QUICKDouble X1 = LOC4(devSim_MP2.Xcoeff, kStartI+III, kStartJ+JJJ, I - devSim_MP2.Qstart[II], J - devSim_MP2.Qstart[JJ], devSim_MP2.jbasis, devSim_MP2.jbasis, 2, 2);
        
        for (int j = 0; j<kPrimK*kPrimL; j++){
            int LLL = (int)j/kPrimK;
            int KKK = (int) j-kPrimK*LLL;
            
            if (cutoffPrim * LOC2(devSim_MP2.cutPrim, kStartK+KKK, kStartL+LLL, devSim_MP2.jbasis, devSim_MP2.jbasis) > devSim_MP2.primLimit) {
                /*
                 CD = expo(L)+expo(K)
                 ABCD = 1/ (AB + CD) = 1 / (expo(I)+expo(J)+expo(K)+expo(L))
                 AB * CD      (expo(I)+expo(J))*(expo(K)+expo(L))
                 Rou(Greek Letter) =   ----------- = ------------------------------------
                 AB + CD         expo(I)+expo(J)+expo(K)+expo(L)
                 
                 expo(I)+expo(J)                        expo(K)+expo(L)
                 ABcom = --------------------------------  CDcom = --------------------------------
                 expo(I)+expo(J)+expo(K)+expo(L)           expo(I)+expo(J)+expo(K)+expo(L)
                 
                 ABCDtemp = 1/2(expo(I)+expo(J)+expo(K)+expo(L))                    
                 */                      
                
                int kk_start = devSim_MP2.prim_start[KK];
                int ll_start = devSim_MP2.prim_start[LL];
                
                QUICKDouble CD = LOC2(devSim_MP2.expoSum, kk_start+KKK, ll_start+LLL, devSim_MP2.prim_total, devSim_MP2.prim_total);
                
                QUICKDouble ABCD = 1/(AB+CD);
                
                /*
                 X2 is the multiplication of four indices normalized coeffecient
                 */
                QUICKDouble X2 = sqrt(ABCD) * X1 * LOC4(devSim_MP2.Xcoeff, kStartK+KKK, kStartL+LLL, K - devSim_MP2.Qstart[KK], L - devSim_MP2.Qstart[LL], devSim_MP2.jbasis, devSim_MP2.jbasis, 2, 2);
                
                /*
                 Q' is the weighting center of K and L
                 --->           --->
                 ->  ------>       expo(K)*xyz(K)+expo(L)*xyz(L)
                 Q = P'(K,L)  = ------------------------------
                 expo(K) + expo(L)
                 
                 W' is the weight center for I, J, K, L
                 
                 --->             --->             --->            --->
                 ->     expo(I)*xyz(I) + expo(J)*xyz(J) + expo(K)*xyz(K) +expo(L)*xyz(L)
                 W = -------------------------------------------------------------------
                 expo(I) + expo(J) + expo(K) + expo(L)                                            
                 ->  ->  2
                 RPQ =| P - Q | 
                 
                 ->  -> 2
                 T = ROU * | P - Q|
                 */
                
                QUICKDouble Qx = LOC2(devSim_MP2.weightedCenterX, kk_start+KKK, ll_start+LLL, devSim_MP2.prim_total, devSim_MP2.prim_total);
                QUICKDouble Qy = LOC2(devSim_MP2.weightedCenterY, kk_start+KKK, ll_start+LLL, devSim_MP2.prim_total, devSim_MP2.prim_total);
                QUICKDouble Qz = LOC2(devSim_MP2.weightedCenterZ, kk_start+KKK, ll_start+LLL, devSim_MP2.prim_total, devSim_MP2.prim_total);
                
                QUICKDouble T = AB * CD * ABCD * ( quick_dsqr_MP2(Px-Qx) + quick_dsqr_MP2(Py-Qy) + quick_dsqr_MP2(Pz-Qz));
                
                QUICKDouble YVerticalTemp[VDIM1*VDIM2*VDIM3];
                FmT_MP2(I+J+K+L, T, YVerticalTemp);
                for (int i = 0; i<=I+J+K+L; i++) {
                    VY(0, 0, i) = VY(0, 0, i) * X2;
                }
                
                vertical_MP2(I, J, K, L, YVerticalTemp, store, \
                         Px - RAx, Py - RAy, Pz - RAz, (Px*AB+Qx*CD)*ABCD - Px, (Py*AB+Qy*CD)*ABCD - Py, (Pz*AB+Qz*CD)*ABCD - Pz, \
                         Qx - RCx, Qy - RCy, Qz - RCz, (Px*AB+Qx*CD)*ABCD - Qx, (Py*AB+Qy*CD)*ABCD - Qy, (Pz*AB+Qz*CD)*ABCD - Qz, \
                         0.5 * ABCD, 0.5 / AB, 0.5 / CD, AB * ABCD, CD * ABCD);
            }   
        }
    }
    
    
    // IJKLTYPE is the I, J, K,L type
    int IJKLTYPE = (int) (1000 * I + 100 *J + 10 * K + L);
    
    QUICKDouble RBx, RBy, RBz;
    QUICKDouble RDx, RDy, RDz;
    
    RBx = LOC2(devSim_MP2.xyz, 0 , devSim_MP2.katom[JJ]-1, 3, devSim_MP2.natom);
    RBy = LOC2(devSim_MP2.xyz, 1 , devSim_MP2.katom[JJ]-1, 3, devSim_MP2.natom);
    RBz = LOC2(devSim_MP2.xyz, 2 , devSim_MP2.katom[JJ]-1, 3, devSim_MP2.natom);
    
    
    RDx = LOC2(devSim_MP2.xyz, 0 , devSim_MP2.katom[LL]-1, 3, devSim_MP2.natom);
    RDy = LOC2(devSim_MP2.xyz, 1 , devSim_MP2.katom[LL]-1, 3, devSim_MP2.natom);
    RDz = LOC2(devSim_MP2.xyz, 2 , devSim_MP2.katom[LL]-1, 3, devSim_MP2.natom);
    
    int III1 = LOC2(devSim_MP2.Qsbasis, II, I, devSim_MP2.nshell, 4);
    int III2 = LOC2(devSim_MP2.Qfbasis, II, I, devSim_MP2.nshell, 4);
    int JJJ1 = LOC2(devSim_MP2.Qsbasis, JJ, J, devSim_MP2.nshell, 4);
    int JJJ2 = LOC2(devSim_MP2.Qfbasis, JJ, J, devSim_MP2.nshell, 4);
    int KKK1 = LOC2(devSim_MP2.Qsbasis, KK, K, devSim_MP2.nshell, 4);
    int KKK2 = LOC2(devSim_MP2.Qfbasis, KK, K, devSim_MP2.nshell, 4);
    int LLL1 = LOC2(devSim_MP2.Qsbasis, LL, L, devSim_MP2.nshell, 4);
    int LLL2 = LOC2(devSim_MP2.Qfbasis, LL, L, devSim_MP2.nshell, 4);

	//printf("in gpu_MP2.cu/iclass_MP2, III1,III2,JJJ1,JJJ2,KKK1,KKK2,LLL1,LLL2 are %d %d %d %d %d %d %d %d\n", III1,III2,JJJ1,JJJ2,KKK1,KKK2,LLL1,LLL2);
    

    // maxIJKL is the max of I,J,K,L
    int maxIJKL = (int)MAX(MAX(I,J),MAX(K,L));
    
    if (((maxIJKL == 2)&&(J != 0 || L!=0)) || (maxIJKL >= 3)) {
        IJKLTYPE = 999;
    }
    
    QUICKDouble hybrid_coeff = 0.0;
    if (devSim_MP2.method == HF){
        hybrid_coeff = 1.0;
    }else if (devSim_MP2.method == B3LYP){
        hybrid_coeff = 0.2;
    }else if (devSim_MP2.method == DFT){
        hybrid_coeff = 0.0;
    }
    
	if(II<JJ && KK<LL){
    for (int III = III1; III <= III2; III++) {
        //for (int JJJ = MAX(III,JJJ1); JJJ <= JJJ2; JJJ++) {
		for (int JJJ = JJJ1; JJJ <= JJJ2; JJJ++) {
            //for (int KKK = MAX(III,KKK1); KKK <= KKK2; KKK++) {
			for (int KKK = KKK1; KKK <= KKK2; KKK++) {
                //for (int LLL = MAX(KKK,LLL1); LLL <= LLL2; LLL++) {
                for (int LLL = LLL1; LLL <= LLL2; LLL++) {  
					/*
                    if (III < KKK || 
                        ((III == JJJ) && (III == LLL)) || 
                        ((III == JJJ) && (III  < LLL)) || 
                        ((JJJ == LLL) && (III  < JJJ)) ||
                        ((III == KKK) && (III  < JJJ)  && (JJJ < LLL))) {
                        
					*/
                        QUICKDouble Y = (QUICKDouble) hrrwhole_MP2( I, J, K, L,\
                                                               III, JJJ, KKK, LLL, IJKLTYPE, store, \
                                                               RAx, RAy, RAz, RBx, RBy, RBz, \
                                                               RCx, RCy, RCz, RDx, RDy, RDz);
 						
						////get Y matrix here
					    //printf("in gpu_MP2.cu/iclass_MP2, III,JJJ,KKK,LLL and Y are %d %d %d %d %lf\n", \
								III,JJJ,KKK,LLL,Y);
						//LOC4(devSim_MP2.Y_Matrix, III-1, JJJ-1, KKK-1, LLL-1, devSim_MP2.nbasis, devSim_MP2.nbasis, devSim_MP2.nbasis, devSim_MP2.nbasis) = Y;			
						
						//first quarter transformation:
						if(fabs(Y)>devSim_MP2.integralCutoff)
						{
							//for(int i3mp2=1; i3mp2<=devSim_MP2.nElec/2;i3mp2++)
							for(int i3mp2=1; i3mp2<=nsteplength;i3mp2++)
							{
								int i3mp2new = nstepmp2s+i3mp2-1; // i3mp2new=nstepmp2s+i3mp2-1 where nstepmp2s should be 1
								QUICKDouble atemp = LOC2(devSim_MP2.coefficient, KKK-1, i3mp2new-1, devSim_MP2.nbasis, devSim_MP2.nbasis)*Y;
								QUICKDouble btemp = LOC2(devSim_MP2.coefficient, LLL-1, i3mp2new-1, devSim_MP2.nbasis, devSim_MP2.nbasis)*Y;
								int IIInew = III- devSim_MP2.Ksumtype[II]+1;
								int JJJnew = JJJ- devSim_MP2.Ksumtype[JJ]+1;
								//printf("in gpu_MP2.cu/iclass_MP2, III, JJJ, KKK, LLL and II, JJ are %d %d %d %d and %d %d %d %d\n",\
										III, JJJ, KKK, LLL, II, JJ, KK, LL);
								QUICKADD(LOC5(orbmp2i331,i3mp2-1,LLL-1,IIInew-1,JJJnew-1,0, nstep, nbasis, nbasistemp, nbasistemp, 2), atemp);
								QUICKADD(LOC5(orbmp2i331,i3mp2-1,LLL-1,JJJnew-1,IIInew-1,1, nstep, nbasis, nbasistemp, nbasistemp, 2), atemp);
								QUICKADD(LOC5(orbmp2i331,i3mp2-1,KKK-1,IIInew-1,JJJnew-1,0, nstep, nbasis, nbasistemp, nbasistemp, 2), btemp);
								QUICKADD(LOC5(orbmp2i331,i3mp2-1,KKK-1,JJJnew-1,IIInew-1,1, nstep, nbasis, nbasistemp, nbasistemp, 2), btemp);
							}
						}

						
                }
            }
        }
    }
	} //This corresponds to if(II<JJ && KK<LLL)
	else
    {
        for(int III=III1; III<=III2; III++)
        {
            if(MAX(III,JJJ1)<=JJJ2)
            {
                for(int JJJ=MAX(III,JJJ1);JJJ<=JJJ2;JJJ++)
                {
                    for(int KKK=KKK1; KKK<=KKK2; KKK++)
                    {
                        if(MAX(KKK,LLL1)<=LLL2)
                        {
                            for(int LLL=MAX(KKK,LLL1);LLL<=LLL2;LLL++)
                            {
                                QUICKDouble Y = (QUICKDouble) hrrwhole_MP2( I, J, K, L,\
                                                               III, JJJ, KKK, LLL, IJKLTYPE, store, \
                                                               RAx, RAy, RAz, RBx, RBy, RBz, \
                                                               RCx, RCy, RCz, RDx, RDy, RDz);
                                //LOC4(devSim_MP2.Y_Matrix, III-1, JJJ-1, KKK-1, LLL-1, devSim_MP2.nbasis, devSim_MP2.nbasis, devSim_MP2.nbasis, devSim_MP2.nbasis) = Y;
								//printf("in gpu_MP2.cu/iclass_MP2, III,JJJ,KKK,LLL and Y are %d %d %d %d %lf\n", \
                                III,JJJ,KKK,LLL,Y);
								if(fabs(Y)>devSim_MP2.integralCutoff)
								{
									//for(int i3mp2=1;i3mp2<=devSim_MP2.nElec/2;i3mp2++)
									for(int i3mp2=1;i3mp2<=nsteplength;i3mp2++)
									{
										//int i3mp2new = i3mp2;
										int i3mp2new = nstepmp2s+i3mp2-1;
										QUICKDouble atemp = LOC2(devSim_MP2.coefficient, KKK-1, i3mp2new-1, devSim_MP2.nbasis, devSim_MP2.nbasis)*Y;
										QUICKDouble btemp = LOC2(devSim_MP2.coefficient, LLL-1, i3mp2new-1, devSim_MP2.nbasis, devSim_MP2.nbasis)*Y;
										int IIInew = III - devSim_MP2.Ksumtype[II]+1;
										int JJJnew = JJJ - devSim_MP2.Ksumtype[JJ]+1;
										QUICKADD(LOC5(orbmp2i331, i3mp2-1, LLL-1, IIInew-1, JJJnew-1, 0, nstep, nbasis, nbasistemp, nbasistemp, 2),atemp);										
										if(JJJ != III)
										{
											QUICKADD(LOC5(orbmp2i331, i3mp2-1, LLL-1, JJJnew-1, IIInew-1, 1, nstep, nbasis, nbasistemp, nbasistemp, 2),atemp);
										}
										if(KKK != LLL)
										{
											QUICKADD(LOC5(orbmp2i331, i3mp2-1, KKK-1, IIInew-1, JJJnew-1, 0, nstep, nbasis, nbasistemp, nbasistemp, 2),btemp);
											if(III != JJJ)
											{
												QUICKADD(LOC5(orbmp2i331, i3mp2-1, KKK-1, JJJnew-1, IIInew-1, 1, nstep, nbasis, nbasistemp, nbasistemp, 2),btemp);
											}
										}

									}
								}
                            }
                        }
                    }
                }
            }
        }
    }

    return;
}



__device__ void vertical_MP2(int I, int J, int K, int L, QUICKDouble* YVerticalTemp, QUICKDouble* store,
                         QUICKDouble Ptempx, QUICKDouble Ptempy, QUICKDouble Ptempz,  \
                         QUICKDouble WPtempx,QUICKDouble WPtempy,QUICKDouble WPtempz, \
                         QUICKDouble Qtempx, QUICKDouble Qtempy, QUICKDouble Qtempz,  \
                         QUICKDouble WQtempx,QUICKDouble WQtempy,QUICKDouble WQtempz, \
                         QUICKDouble ABCDtemp,QUICKDouble ABtemp, \
                         QUICKDouble CDtemp, QUICKDouble ABcom, QUICKDouble CDcom)
{
	//unsigned int offside = blockIdx.x*blockDim.x+threadIdx.x;
 
	// here is fine
	//printf("at the begining of vertical_MP2, I, J, K, L are %d, %d, %d, %d\n",I, J, K, L);   
	
	/*
	if(offside==0)
	{	
		printf("Ptempx is %lf, Ptempy is %lf, Ptempz is %lf, WPtempx is %lf, Qtempx is %lf, WQtempx is %lf, ABCDtemp is %lf, CDtemp is %lf, VY(0,0,0) is %lf\n",\
				Ptempx, Ptempy, Ptempz, WPtempx, Qtempx, WQtempx, ABCDtemp, CDtemp, VY(0,0,0));
	}
	*/
    
    LOC2(store, 0, 0, STOREDIM, STOREDIM) += VY( 0, 0, 0);
	
	/*
	if(offside==0)
    {
		printf("at the begining of vertical_MP2, store[0] is %lf, this is case %d\n", store[0], (I+J)*10+K+L);
	}
	*/
	switch ((I+J)*10+K+L){
        case 1:
        {
            //SSPS(0, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);
            LOC2(store, 0, 1, STOREDIM, STOREDIM) += Qtempx * VY( 0, 0, 0) + WQtempx * VY( 0, 0, 1);
            LOC2(store, 0, 2, STOREDIM, STOREDIM) += Qtempy * VY( 0, 0, 0) + WQtempy * VY( 0, 0, 1);
            LOC2(store, 0, 3, STOREDIM, STOREDIM) += Qtempz * VY( 0, 0, 0) + WQtempz * VY( 0, 0, 1);
            break;
        }
        case 10:
        {
            //PSSS(0, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);
            LOC2(store, 1, 0, STOREDIM, STOREDIM) += Ptempx * VY( 0, 0, 0) + WPtempx * VY( 0, 0, 1);
            LOC2(store, 2, 0, STOREDIM, STOREDIM) += Ptempy * VY( 0, 0, 0) + WPtempy * VY( 0, 0, 1);
            LOC2(store, 3, 0, STOREDIM, STOREDIM) += Ptempz * VY( 0, 0, 0) + WPtempz * VY( 0, 0, 1);
            break;
        }
            // PSPS orbital
        case 11:
        {
            
            //PSSS(0, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);
            if (K==0){
                LOC2(store, 1, 0, STOREDIM, STOREDIM) += Ptempx * VY( 0, 0, 0) + WPtempx * VY( 0, 0, 1);
                LOC2(store, 2, 0, STOREDIM, STOREDIM) += Ptempy * VY( 0, 0, 0) + WPtempy * VY( 0, 0, 1);
                LOC2(store, 3, 0, STOREDIM, STOREDIM) += Ptempz * VY( 0, 0, 0) + WPtempz * VY( 0, 0, 1);
            }
            //SSPS(0, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);
            
            
            QUICKDouble x_0_1_0 = Qtempx * VY( 0, 0, 0) + WQtempx * VY( 0, 0, 1);
            QUICKDouble x_0_2_0 = Qtempy * VY( 0, 0, 0) + WQtempy * VY( 0, 0, 1);
            QUICKDouble x_0_3_0 = Qtempz * VY( 0, 0, 0) + WQtempz * VY( 0, 0, 1);
            //SSPS(1, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);
            
            QUICKDouble x_0_1_1 = Qtempx * VY( 0, 0, 1) + WQtempx * VY( 0, 0, 2);
            QUICKDouble x_0_2_1 = Qtempy * VY( 0, 0, 1) + WQtempy * VY( 0, 0, 2);
            QUICKDouble x_0_3_1 = Qtempz * VY( 0, 0, 1) + WQtempz * VY( 0, 0, 2);
            
            if (I==0) {
                LOC2(store, 0, 1, STOREDIM, STOREDIM)+= x_0_1_0;
                LOC2(store, 0, 2, STOREDIM, STOREDIM)+= x_0_2_0;
                LOC2(store, 0, 3, STOREDIM, STOREDIM)+= x_0_3_0;
            }
            //PSPS(0, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz, ABCDtemp);
            LOC2(store, 1, 1, STOREDIM, STOREDIM) += Ptempx * x_0_1_0 + WPtempx * x_0_1_1 + ABCDtemp * VY( 0, 0, 1);
            LOC2(store, 2, 1, STOREDIM, STOREDIM) += Ptempy * x_0_1_0 + WPtempy * x_0_1_1;
            LOC2(store, 3, 1, STOREDIM, STOREDIM) += Ptempz * x_0_1_0 + WPtempz * x_0_1_1;
            
            LOC2(store, 1, 2, STOREDIM, STOREDIM) += Ptempx * x_0_2_0 + WPtempx * x_0_2_1;
            LOC2(store, 2, 2, STOREDIM, STOREDIM) += Ptempy * x_0_2_0 + WPtempy * x_0_2_1 + ABCDtemp * VY( 0, 0, 1);
            LOC2(store, 3, 2, STOREDIM, STOREDIM) += Ptempz * x_0_2_0 + WPtempz * x_0_2_1;
            
            LOC2(store, 1, 3, STOREDIM, STOREDIM) += Ptempx * x_0_3_0 + WPtempx * x_0_3_1;
            LOC2(store, 2, 3, STOREDIM, STOREDIM) += Ptempy * x_0_3_0 + WPtempy * x_0_3_1;
            LOC2(store, 3, 3, STOREDIM, STOREDIM) += Ptempz * x_0_3_0 + WPtempz * x_0_3_1 + ABCDtemp * VY( 0, 0, 1);
            
            break;
        }
        case 20:
        {
            //PSSS(0, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);
            
            QUICKDouble x_1_0_0 = Ptempx * VY( 0, 0, 0) + WPtempx * VY( 0, 0, 1);
            QUICKDouble x_2_0_0 = Ptempy * VY( 0, 0, 0) + WPtempy * VY( 0, 0, 1);
            QUICKDouble x_3_0_0 = Ptempz * VY( 0, 0, 0) + WPtempz * VY( 0, 0, 1);
            
            //PSSS(1, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);
            
            QUICKDouble x_1_0_1 = Ptempx * VY( 0, 0, 1) + WPtempx * VY( 0, 0, 2);
            QUICKDouble x_2_0_1 = Ptempy * VY( 0, 0, 1) + WPtempy * VY( 0, 0, 2);
            QUICKDouble x_3_0_1 = Ptempz * VY( 0, 0, 1) + WPtempz * VY( 0, 0, 2);
            
            //DSSS(0, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz, ABtemp, CDcom);
            
            LOC2(store, 1, 0, STOREDIM, STOREDIM) += x_1_0_0;
            LOC2(store, 2, 0, STOREDIM, STOREDIM) += x_2_0_0;
            LOC2(store, 3, 0, STOREDIM, STOREDIM) += x_3_0_0;
            LOC2(store, 4, 0, STOREDIM, STOREDIM) += Ptempx * x_2_0_0 + WPtempx * x_2_0_1;
            LOC2(store, 5, 0, STOREDIM, STOREDIM) += Ptempy * x_3_0_0 + WPtempy * x_3_0_1;
            LOC2(store, 6, 0, STOREDIM, STOREDIM) += Ptempx * x_3_0_0 + WPtempx * x_3_0_1;
            LOC2(store, 7, 0, STOREDIM, STOREDIM) += Ptempx * x_1_0_0 + WPtempx * x_1_0_1+ ABtemp*(VY( 0, 0, 0) - CDcom * VY( 0, 0, 1));
            LOC2(store, 8, 0, STOREDIM, STOREDIM) += Ptempy * x_2_0_0 + WPtempy * x_2_0_1+ ABtemp*(VY( 0, 0, 0) - CDcom * VY( 0, 0, 1));
            LOC2(store, 9, 0, STOREDIM, STOREDIM) += Ptempz * x_3_0_0 + WPtempz * x_3_0_1+ ABtemp*(VY( 0, 0, 0) - CDcom * VY( 0, 0, 1));
            break;
        }
        case 2:
        {
            //SSPS(0, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);
            QUICKDouble x_0_1_0 = Qtempx * VY( 0, 0, 0) + WQtempx * VY( 0, 0, 1);
            QUICKDouble x_0_2_0 = Qtempy * VY( 0, 0, 0) + WQtempy * VY( 0, 0, 1);
            QUICKDouble x_0_3_0 = Qtempz * VY( 0, 0, 0) + WQtempz * VY( 0, 0, 1);
            
            //SSPS(1, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);
            QUICKDouble x_0_1_1 = Qtempx * VY( 0, 0, 1) + WQtempx * VY( 0, 0, 2);
            QUICKDouble x_0_2_1 = Qtempy * VY( 0, 0, 1) + WQtempy * VY( 0, 0, 2);
            QUICKDouble x_0_3_1 = Qtempz * VY( 0, 0, 1) + WQtempz * VY( 0, 0, 2);
            
            LOC2(store, 0, 1, STOREDIM, STOREDIM) += x_0_1_0;
            LOC2(store, 0, 2, STOREDIM, STOREDIM) += x_0_2_0;
            LOC2(store, 0, 3, STOREDIM, STOREDIM) += x_0_3_0;
            
            //SSDS(0, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz, CDtemp, ABcom);
            LOC2(store, 0, 4, STOREDIM, STOREDIM) += Qtempx * x_0_2_0 + WQtempx * x_0_2_1;
            LOC2(store, 0, 5, STOREDIM, STOREDIM) += Qtempy * x_0_3_0 + WQtempy * x_0_3_1;
            LOC2(store, 0, 6, STOREDIM, STOREDIM) += Qtempx * x_0_3_0 + WQtempx * x_0_3_1;
            
            LOC2(store, 0, 7, STOREDIM, STOREDIM) += Qtempx * x_0_1_0 + WQtempx * x_0_1_1+ CDtemp*(VY( 0, 0, 0) - ABcom * VY( 0, 0, 1));
            LOC2(store, 0, 8, STOREDIM, STOREDIM) += Qtempy * x_0_2_0 + WQtempy * x_0_2_1+ CDtemp*(VY( 0, 0, 0) - ABcom * VY( 0, 0, 1));
            LOC2(store, 0, 9, STOREDIM, STOREDIM) += Qtempz * x_0_3_0 + WQtempz * x_0_3_1+ CDtemp*(VY( 0, 0, 0) - ABcom * VY( 0, 0, 1));
            
            break;
        }
        case 21:
        {
            //SSPS(0, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);
            QUICKDouble x_0_1_0 = Qtempx * VY( 0, 0, 0) + WQtempx * VY( 0, 0, 1);
            QUICKDouble x_0_2_0 = Qtempy * VY( 0, 0, 0) + WQtempy * VY( 0, 0, 1);
            QUICKDouble x_0_3_0 = Qtempz * VY( 0, 0, 0) + WQtempz * VY( 0, 0, 1);
            
            if (I==0){
                LOC2(store, 0, 1, STOREDIM, STOREDIM) += x_0_1_0;
                LOC2(store, 0, 2, STOREDIM, STOREDIM) += x_0_2_0;
                LOC2(store, 0, 3, STOREDIM, STOREDIM) += x_0_3_0;
            }
            
            //PSSS(0, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);
            QUICKDouble x_1_0_0 = Ptempx * VY( 0, 0, 0) + WPtempx * VY( 0, 0, 1);
            QUICKDouble x_2_0_0 = Ptempy * VY( 0, 0, 0) + WPtempy * VY( 0, 0, 1);
            QUICKDouble x_3_0_0 = Ptempz * VY( 0, 0, 0) + WPtempz * VY( 0, 0, 1);
            //SSPS(1, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);
            
            QUICKDouble x_0_1_1 = Qtempx * VY( 0, 0, 1) + WQtempx * VY( 0, 0, 2);
            QUICKDouble x_0_2_1 = Qtempy * VY( 0, 0, 1) + WQtempy * VY( 0, 0, 2);
            QUICKDouble x_0_3_1 = Qtempz * VY( 0, 0, 1) + WQtempz * VY( 0, 0, 2);
            
            //PSPS(0, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz, ABCDtemp);
            LOC2(store, 1, 1, STOREDIM, STOREDIM) += Ptempx * x_0_1_0 + WPtempx * x_0_1_1 + ABCDtemp * VY( 0, 0, 1);
            LOC2(store, 2, 1, STOREDIM, STOREDIM) += Ptempy * x_0_1_0 + WPtempy * x_0_1_1;
            LOC2(store, 3, 1, STOREDIM, STOREDIM) += Ptempz * x_0_1_0 + WPtempz * x_0_1_1;
            
            LOC2(store, 1, 2, STOREDIM, STOREDIM) += Ptempx * x_0_2_0 + WPtempx * x_0_2_1;
            LOC2(store, 2, 2, STOREDIM, STOREDIM) += Ptempy * x_0_2_0 + WPtempy * x_0_2_1 + ABCDtemp * VY( 0, 0, 1);
            LOC2(store, 3, 2, STOREDIM, STOREDIM) += Ptempz * x_0_2_0 + WPtempz * x_0_2_1;
            
            LOC2(store, 1, 3, STOREDIM, STOREDIM) += Ptempx * x_0_3_0 + WPtempx * x_0_3_1;
            LOC2(store, 2, 3, STOREDIM, STOREDIM) += Ptempy * x_0_3_0 + WPtempy * x_0_3_1;
            LOC2(store, 3, 3, STOREDIM, STOREDIM) += Ptempz * x_0_3_0 + WPtempz * x_0_3_1 + ABCDtemp * VY( 0, 0, 1);
            
            
            //PSSS(1, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);
            QUICKDouble x_1_0_1 = Ptempx * VY( 0, 0, 1) + WPtempx * VY( 0, 0, 2);
            QUICKDouble x_2_0_1 = Ptempy * VY( 0, 0, 1) + WPtempy * VY( 0, 0, 2);
            QUICKDouble x_3_0_1 = Ptempz * VY( 0, 0, 1) + WPtempz * VY( 0, 0, 2);
            
            //DSSS(0, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz, ABtemp, CDcom);
            QUICKDouble x_4_0_0 = Ptempx * x_2_0_0 + WPtempx * x_2_0_1;
            QUICKDouble x_5_0_0 = Ptempy * x_3_0_0 + WPtempy * x_3_0_1;
            QUICKDouble x_6_0_0 = Ptempx * x_3_0_0 + WPtempx * x_3_0_1;
            
            QUICKDouble x_7_0_0 = Ptempx * x_1_0_0 + WPtempx * x_1_0_1+ ABtemp*(VY( 0, 0, 0) - CDcom * VY( 0, 0, 1));
            QUICKDouble x_8_0_0 = Ptempy * x_2_0_0 + WPtempy * x_2_0_1+ ABtemp*(VY( 0, 0, 0) - CDcom * VY( 0, 0, 1));
            QUICKDouble x_9_0_0 = Ptempz * x_3_0_0 + WPtempz * x_3_0_1+ ABtemp*(VY( 0, 0, 0) - CDcom * VY( 0, 0, 1));
            
            if (K==0){
                LOC2(store, 1, 0, STOREDIM, STOREDIM) += x_1_0_0;
                LOC2(store, 2, 0, STOREDIM, STOREDIM) += x_2_0_0;
                LOC2(store, 3, 0, STOREDIM, STOREDIM) += x_3_0_0;
                LOC2(store, 4, 0, STOREDIM, STOREDIM) += x_4_0_0;
                LOC2(store, 5, 0, STOREDIM, STOREDIM) += x_5_0_0;
                LOC2(store, 6, 0, STOREDIM, STOREDIM) += x_6_0_0;
                LOC2(store, 7, 0, STOREDIM, STOREDIM) += x_7_0_0;
                LOC2(store, 8, 0, STOREDIM, STOREDIM) += x_8_0_0;
                LOC2(store, 9, 0, STOREDIM, STOREDIM) += x_9_0_0;
            }
            
            //PSSS(2, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);
            
            QUICKDouble x_1_0_2 = Ptempx * VY( 0, 0, 2) + WPtempx * VY( 0, 0, 3);
            QUICKDouble x_2_0_2 = Ptempy * VY( 0, 0, 2) + WPtempy * VY( 0, 0, 3);
            QUICKDouble x_3_0_2 = Ptempz * VY( 0, 0, 2) + WPtempz * VY( 0, 0, 3);
            
            //DSSS(1, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz, ABtemp, CDcom);
            //DSPS(0, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz, ABCDtemp);
            QUICKDouble x_4_0_1 = Ptempx * x_2_0_1 + WPtempx * x_2_0_2;
            LOC2(store, 4, 1, STOREDIM, STOREDIM) += Qtempx * x_4_0_0 + WQtempx * x_4_0_1 + ABCDtemp * x_2_0_1;
            LOC2(store, 4, 2, STOREDIM, STOREDIM) += Qtempy * x_4_0_0 + WQtempy * x_4_0_1 + ABCDtemp * x_1_0_1;
            LOC2(store, 4, 3, STOREDIM, STOREDIM) += Qtempz * x_4_0_0 + WQtempz * x_4_0_1;
            
            QUICKDouble x_5_0_1 = Ptempy * x_3_0_1 + WPtempy * x_3_0_2;
            LOC2(store, 5, 1, STOREDIM, STOREDIM) += Qtempx * x_5_0_0 + WQtempx * x_5_0_1;
            LOC2(store, 5, 2, STOREDIM, STOREDIM) += Qtempy * x_5_0_0 + WQtempy * x_5_0_1 + ABCDtemp * x_3_0_1;
            LOC2(store, 5, 3, STOREDIM, STOREDIM) += Qtempz * x_5_0_0 + WQtempz * x_5_0_1 + ABCDtemp * x_2_0_1;
            
            QUICKDouble x_6_0_1 = Ptempx * x_3_0_1 + WPtempx * x_3_0_2;
            LOC2(store, 6, 1, STOREDIM, STOREDIM) += Qtempx * x_6_0_0 + WQtempx * x_6_0_1 + ABCDtemp * x_3_0_1;
            LOC2(store, 6, 2, STOREDIM, STOREDIM) += Qtempy * x_6_0_0 + WQtempy * x_6_0_1;
            LOC2(store, 6, 3, STOREDIM, STOREDIM) += Qtempz * x_6_0_0 + WQtempz * x_6_0_1 + ABCDtemp * x_1_0_1;
            
            QUICKDouble x_7_0_1 = Ptempx * x_1_0_1 + WPtempx * x_1_0_2+ ABtemp*(VY( 0, 0, 1) - CDcom * VY( 0, 0, 2));
            LOC2(store, 7, 1, STOREDIM, STOREDIM) += Qtempx * x_7_0_0 + WQtempx * x_7_0_1 + ABCDtemp * x_1_0_1 * 2;
            LOC2(store, 7, 2, STOREDIM, STOREDIM) += Qtempy * x_7_0_0 + WQtempy * x_7_0_1;
            LOC2(store, 7, 3, STOREDIM, STOREDIM) += Qtempz * x_7_0_0 + WQtempz * x_7_0_1;
            
            QUICKDouble x_8_0_1 = Ptempy * x_2_0_1 + WPtempy * x_2_0_2+ ABtemp*(VY( 0, 0, 1) - CDcom * VY( 0, 0, 2));
            LOC2(store, 8, 1, STOREDIM, STOREDIM) += Qtempx * x_8_0_0 + WQtempx * x_8_0_1;
            LOC2(store, 8, 2, STOREDIM, STOREDIM) += Qtempy * x_8_0_0 + WQtempy * x_8_0_1 + ABCDtemp * x_2_0_1 * 2;
            LOC2(store, 8, 3, STOREDIM, STOREDIM) += Qtempz * x_8_0_0 + WQtempz * x_8_0_1;
            
            QUICKDouble x_9_0_1 = Ptempz * x_3_0_1 + WPtempz * x_3_0_2+ ABtemp*(VY( 0, 0, 1) - CDcom * VY( 0, 0, 2));
            LOC2(store, 9, 1, STOREDIM, STOREDIM) += Qtempx * x_9_0_0 + WQtempx * x_9_0_1;
            LOC2(store, 9, 2, STOREDIM, STOREDIM) += Qtempy * x_9_0_0 + WQtempy * x_9_0_1;
            LOC2(store, 9, 3, STOREDIM, STOREDIM) += Qtempz * x_9_0_0 + WQtempz * x_9_0_1 + ABCDtemp * x_3_0_1 * 2;break;
        }
        case 12:
        {
            // PSSS(0, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);
            if (K==0){
                LOC2(store, 1, 0, STOREDIM, STOREDIM) += Ptempx * VY( 0, 0, 0) + WPtempx * VY( 0, 0, 1);
                LOC2(store, 2, 0, STOREDIM, STOREDIM) += Ptempy * VY( 0, 0, 0) + WPtempy * VY( 0, 0, 1);
                LOC2(store, 3, 0, STOREDIM, STOREDIM) += Ptempz * VY( 0, 0, 0) + WPtempz * VY( 0, 0, 1);
            }
            
            //SSPS(0, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);
            QUICKDouble x_0_1_0 = Qtempx * VY( 0, 0, 0) + WQtempx * VY( 0, 0, 1);
            QUICKDouble x_0_2_0 = Qtempy * VY( 0, 0, 0) + WQtempy * VY( 0, 0, 1);
            QUICKDouble x_0_3_0 = Qtempz * VY( 0, 0, 0) + WQtempz * VY( 0, 0, 1);
            
            //SSPS(1, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);
            QUICKDouble x_0_1_1 = Qtempx * VY( 0, 0, 1) + WQtempx * VY( 0, 0, 2);
            QUICKDouble x_0_2_1 = Qtempy * VY( 0, 0, 1) + WQtempy * VY( 0, 0, 2);
            QUICKDouble x_0_3_1 = Qtempz * VY( 0, 0, 1) + WQtempz * VY( 0, 0, 2);
            
            //PSPS(0, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz, ABCDtemp);
            LOC2(store, 1, 1, STOREDIM, STOREDIM) += Ptempx * x_0_1_0 + WPtempx * x_0_1_1 + ABCDtemp * VY( 0, 0, 1);
            LOC2(store, 2, 1, STOREDIM, STOREDIM) += Ptempy * x_0_1_0 + WPtempy * x_0_1_1;
            LOC2(store, 3, 1, STOREDIM, STOREDIM) += Ptempz * x_0_1_0 + WPtempz * x_0_1_1;
            
            LOC2(store, 1, 2, STOREDIM, STOREDIM) += Ptempx * x_0_2_0 + WPtempx * x_0_2_1;
            LOC2(store, 2, 2, STOREDIM, STOREDIM) += Ptempy * x_0_2_0 + WPtempy * x_0_2_1 + ABCDtemp * VY( 0, 0, 1);
            LOC2(store, 3, 2, STOREDIM, STOREDIM) += Ptempz * x_0_2_0 + WPtempz * x_0_2_1;
            
            LOC2(store, 1, 3, STOREDIM, STOREDIM) += Ptempx * x_0_3_0 + WPtempx * x_0_3_1;
            LOC2(store, 2, 3, STOREDIM, STOREDIM) += Ptempy * x_0_3_0 + WPtempy * x_0_3_1;
            LOC2(store, 3, 3, STOREDIM, STOREDIM) += Ptempz * x_0_3_0 + WPtempz * x_0_3_1 + ABCDtemp * VY( 0, 0, 1);
            
            //SSDS(0, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz, CDtemp, ABcom);
            QUICKDouble x_0_4_0 = Qtempx * x_0_2_0 + WQtempx * x_0_2_1;
            QUICKDouble x_0_5_0 = Qtempy * x_0_3_0 + WQtempy * x_0_3_1;
            QUICKDouble x_0_6_0 = Qtempx * x_0_3_0 + WQtempx * x_0_3_1;
            
            QUICKDouble x_0_7_0 = Qtempx * x_0_1_0 + WQtempx * x_0_1_1+ CDtemp*(VY( 0, 0, 0) - ABcom * VY( 0, 0, 1));
            QUICKDouble x_0_8_0 = Qtempy * x_0_2_0 + WQtempy * x_0_2_1+ CDtemp*(VY( 0, 0, 0) - ABcom * VY( 0, 0, 1));
            QUICKDouble x_0_9_0 = Qtempz * x_0_3_0 + WQtempz * x_0_3_1+ CDtemp*(VY( 0, 0, 0) - ABcom * VY( 0, 0, 1));
            
            if (I==0){
                LOC2(store, 0, 1, STOREDIM, STOREDIM) += x_0_1_0;
                LOC2(store, 0, 2, STOREDIM, STOREDIM) += x_0_2_0;
                LOC2(store, 0, 3, STOREDIM, STOREDIM) += x_0_3_0;
                LOC2(store, 0, 4, STOREDIM, STOREDIM) += x_0_4_0;
                LOC2(store, 0, 5, STOREDIM, STOREDIM) += x_0_5_0;
                LOC2(store, 0, 6, STOREDIM, STOREDIM) += x_0_6_0;
                LOC2(store, 0, 7, STOREDIM, STOREDIM) += x_0_7_0;
                LOC2(store, 0, 8, STOREDIM, STOREDIM) += x_0_8_0;
                LOC2(store, 0, 9, STOREDIM, STOREDIM) += x_0_9_0;
            }
            
            //SSPS(2, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);
            QUICKDouble x_0_1_2 = Qtempx * VY( 0, 0, 2) + WQtempx * VY( 0, 0, 3);
            QUICKDouble x_0_2_2 = Qtempy * VY( 0, 0, 2) + WQtempy * VY( 0, 0, 3);
            QUICKDouble x_0_3_2 = Qtempz * VY( 0, 0, 2) + WQtempz * VY( 0, 0, 3);
            
            //SSDS(1, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz, CDtemp, ABcom);
            QUICKDouble x_0_4_1 = Qtempx * x_0_2_1 + WQtempx * x_0_2_2;
            QUICKDouble x_0_5_1 = Qtempy * x_0_3_1 + WQtempy * x_0_3_2;
            QUICKDouble x_0_6_1 = Qtempx * x_0_3_1 + WQtempx * x_0_3_2;
            QUICKDouble x_0_7_1 = Qtempx * x_0_1_1 + WQtempx * x_0_1_2+ CDtemp*(VY( 0, 0, 1) - ABcom * VY( 0, 0, 2));
            QUICKDouble x_0_8_1 = Qtempy * x_0_2_1 + WQtempy * x_0_2_2+ CDtemp*(VY( 0, 0, 1) - ABcom * VY( 0, 0, 2));
            QUICKDouble x_0_9_1 = Qtempz * x_0_3_1 + WQtempz * x_0_3_2+ CDtemp*(VY( 0, 0, 1) - ABcom * VY( 0, 0, 2));
            
            //PSDS(0, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz, ABCDtemp);
            LOC2(store, 1, 4, STOREDIM, STOREDIM) += Ptempx * x_0_4_0 + WPtempx * x_0_4_1 + ABCDtemp * x_0_2_1;
            LOC2(store, 2, 4, STOREDIM, STOREDIM) += Ptempy * x_0_4_0 + WPtempy * x_0_4_1 + ABCDtemp * x_0_1_1;
            LOC2(store, 3, 4, STOREDIM, STOREDIM) += Ptempz * x_0_4_0 + WPtempz * x_0_4_1;
            
            LOC2(store, 1, 5, STOREDIM, STOREDIM) += Ptempx * x_0_5_0 + WPtempx * x_0_5_1;
            LOC2(store, 2, 5, STOREDIM, STOREDIM) += Ptempy * x_0_5_0 + WPtempy * x_0_5_1 + ABCDtemp * x_0_3_1;
            LOC2(store, 3, 5, STOREDIM, STOREDIM) += Ptempz * x_0_5_0 + WPtempz * x_0_5_1 + ABCDtemp * x_0_2_1;
            
            LOC2(store, 1, 6, STOREDIM, STOREDIM) += Ptempx * x_0_6_0 + WPtempx * x_0_6_1 + ABCDtemp * x_0_3_1;
            LOC2(store, 2, 6, STOREDIM, STOREDIM) += Ptempy * x_0_6_0 + WPtempy * x_0_6_1;
            LOC2(store, 3, 6, STOREDIM, STOREDIM) += Ptempz * x_0_6_0 + WPtempz * x_0_6_1 + ABCDtemp * x_0_1_1;
            
            LOC2(store, 1, 7, STOREDIM, STOREDIM) += Ptempx * x_0_7_0 + WPtempx * x_0_7_1 + ABCDtemp * x_0_1_1 * 2;
            LOC2(store, 2, 7, STOREDIM, STOREDIM) += Ptempy * x_0_7_0 + WPtempy * x_0_7_1;
            LOC2(store, 3, 7, STOREDIM, STOREDIM) += Ptempz * x_0_7_0 + WPtempz * x_0_7_1;
            
            LOC2(store, 1, 8, STOREDIM, STOREDIM) += Ptempx * x_0_8_0 + WPtempx * x_0_8_1;
            LOC2(store, 2, 8, STOREDIM, STOREDIM) += Ptempy * x_0_8_0 + WPtempy * x_0_8_1 + ABCDtemp * x_0_2_1 * 2;
            LOC2(store, 3, 8, STOREDIM, STOREDIM) += Ptempz * x_0_8_0 + WPtempz * x_0_8_1;
            
            LOC2(store, 1, 9, STOREDIM, STOREDIM) += Ptempx * x_0_9_0 + WPtempx * x_0_9_1;
            LOC2(store, 2, 9, STOREDIM, STOREDIM) += Ptempy * x_0_9_0 + WPtempy * x_0_9_1;
            LOC2(store, 3, 9, STOREDIM, STOREDIM) += Ptempz * x_0_9_0 + WPtempz * x_0_9_1 + ABCDtemp * x_0_3_1 * 2; break;
        }
        case 22:
        {
            //SSPS(0, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);
            QUICKDouble x_0_1_0 = Qtempx * VY( 0, 0, 0) + WQtempx * VY( 0, 0, 1);
            QUICKDouble x_0_2_0 = Qtempy * VY( 0, 0, 0) + WQtempy * VY( 0, 0, 1);
            QUICKDouble x_0_3_0 = Qtempz * VY( 0, 0, 0) + WQtempz * VY( 0, 0, 1);
            
            //PSSS(0, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);
            QUICKDouble x_1_0_0 = Ptempx * VY( 0, 0, 0) + WPtempx * VY( 0, 0, 1);
            QUICKDouble x_2_0_0 = Ptempy * VY( 0, 0, 0) + WPtempy * VY( 0, 0, 1);
            QUICKDouble x_3_0_0 = Ptempz * VY( 0, 0, 0) + WPtempz * VY( 0, 0, 1);
            
            //SSPS(1, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);
            QUICKDouble x_0_1_1 = Qtempx * VY( 0, 0, 1) + WQtempx * VY( 0, 0, 2);
            QUICKDouble x_0_2_1 = Qtempy * VY( 0, 0, 1) + WQtempy * VY( 0, 0, 2);
            QUICKDouble x_0_3_1 = Qtempz * VY( 0, 0, 1) + WQtempz * VY( 0, 0, 2);
            
            //PSPS(0, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz, ABCDtemp);
            LOC2(store, 1, 1, STOREDIM, STOREDIM) += Ptempx * x_0_1_0 + WPtempx * x_0_1_1 + ABCDtemp * VY( 0, 0, 1);
            LOC2(store, 2, 1, STOREDIM, STOREDIM) += Ptempy * x_0_1_0 + WPtempy * x_0_1_1;
            LOC2(store, 3, 1, STOREDIM, STOREDIM) += Ptempz * x_0_1_0 + WPtempz * x_0_1_1;
            
            LOC2(store, 1, 2, STOREDIM, STOREDIM) += Ptempx * x_0_2_0 + WPtempx * x_0_2_1;
            LOC2(store, 2, 2, STOREDIM, STOREDIM) += Ptempy * x_0_2_0 + WPtempy * x_0_2_1 + ABCDtemp * VY( 0, 0, 1);
            LOC2(store, 3, 2, STOREDIM, STOREDIM) += Ptempz * x_0_2_0 + WPtempz * x_0_2_1;
            
            LOC2(store, 1, 3, STOREDIM, STOREDIM) += Ptempx * x_0_3_0 + WPtempx * x_0_3_1;
            LOC2(store, 2, 3, STOREDIM, STOREDIM) += Ptempy * x_0_3_0 + WPtempy * x_0_3_1;
            LOC2(store, 3, 3, STOREDIM, STOREDIM) += Ptempz * x_0_3_0 + WPtempz * x_0_3_1 + ABCDtemp * VY( 0, 0, 1);
            
            
            //SSDS(0, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz, CDtemp, ABcom);
            QUICKDouble x_0_4_0 = Qtempx * x_0_2_0 + WQtempx * x_0_2_1;
            QUICKDouble x_0_5_0 = Qtempy * x_0_3_0 + WQtempy * x_0_3_1;
            QUICKDouble x_0_6_0 = Qtempx * x_0_3_0 + WQtempx * x_0_3_1;
            
            QUICKDouble x_0_7_0 = Qtempx * x_0_1_0 + WQtempx * x_0_1_1+ CDtemp*(VY( 0, 0, 0) - ABcom * VY( 0, 0, 1));
            QUICKDouble x_0_8_0 = Qtempy * x_0_2_0 + WQtempy * x_0_2_1+ CDtemp*(VY( 0, 0, 0) - ABcom * VY( 0, 0, 1));
            QUICKDouble x_0_9_0 = Qtempz * x_0_3_0 + WQtempz * x_0_3_1+ CDtemp*(VY( 0, 0, 0) - ABcom * VY( 0, 0, 1));
            
            //SSPS(2, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);
            QUICKDouble x_0_1_2 = Qtempx * VY( 0, 0, 2) + WQtempx * VY( 0, 0, 3);
            QUICKDouble x_0_2_2 = Qtempy * VY( 0, 0, 2) + WQtempy * VY( 0, 0, 3);
            QUICKDouble x_0_3_2 = Qtempz * VY( 0, 0, 2) + WQtempz * VY( 0, 0, 3);
            
            //SSDS(1, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz, CDtemp, ABcom);
            QUICKDouble x_0_4_1 = Qtempx * x_0_2_1 + WQtempx * x_0_2_2;
            QUICKDouble x_0_5_1 = Qtempy * x_0_3_1 + WQtempy * x_0_3_2;
            QUICKDouble x_0_6_1 = Qtempx * x_0_3_1 + WQtempx * x_0_3_2;
            
            QUICKDouble x_0_7_1 = Qtempx * x_0_1_1 + WQtempx * x_0_1_2+ CDtemp*(VY( 0, 0, 1) - ABcom * VY( 0, 0, 2));
            QUICKDouble x_0_8_1 = Qtempy * x_0_2_1 + WQtempy * x_0_2_2+ CDtemp*(VY( 0, 0, 1) - ABcom * VY( 0, 0, 2));
            QUICKDouble x_0_9_1 = Qtempz * x_0_3_1 + WQtempz * x_0_3_2+ CDtemp*(VY( 0, 0, 1) - ABcom * VY( 0, 0, 2));
            
            //PSDS(0, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz, ABCDtemp);
            
            //PSSS(1, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);
            QUICKDouble x_1_0_1 = Ptempx * VY( 0, 0, 1) + WPtempx * VY( 0, 0, 2);
            QUICKDouble x_2_0_1 = Ptempy * VY( 0, 0, 1) + WPtempy * VY( 0, 0, 2);
            QUICKDouble x_3_0_1 = Ptempz * VY( 0, 0, 1) + WPtempz * VY( 0, 0, 2);
            
            //DSSS(0, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz, ABtemp, CDcom);
            QUICKDouble x_4_0_0 = Ptempx * x_2_0_0 + WPtempx * x_2_0_1;
            QUICKDouble x_5_0_0 = Ptempy * x_3_0_0 + WPtempy * x_3_0_1;
            QUICKDouble x_6_0_0 = Ptempx * x_3_0_0 + WPtempx * x_3_0_1;
            
            QUICKDouble x_7_0_0 = Ptempx * x_1_0_0 + WPtempx * x_1_0_1+ ABtemp*(VY( 0, 0, 0) - CDcom * VY( 0, 0, 1));
            QUICKDouble x_8_0_0 = Ptempy * x_2_0_0 + WPtempy * x_2_0_1+ ABtemp*(VY( 0, 0, 0) - CDcom * VY( 0, 0, 1));
            QUICKDouble x_9_0_0 = Ptempz * x_3_0_0 + WPtempz * x_3_0_1+ ABtemp*(VY( 0, 0, 0) - CDcom * VY( 0, 0, 1));
            
            //PSSS(2, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);
            QUICKDouble x_1_0_2 = Ptempx * VY( 0, 0, 2) + WPtempx * VY( 0, 0, 3);
            QUICKDouble x_2_0_2 = Ptempy * VY( 0, 0, 2) + WPtempy * VY( 0, 0, 3);
            QUICKDouble x_3_0_2 = Ptempz * VY( 0, 0, 2) + WPtempz * VY( 0, 0, 3);
            
            //DSSS(1, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz, ABtemp, CDcom);
            QUICKDouble x_4_0_1 = Ptempx * x_2_0_1 + WPtempx * x_2_0_2;
            QUICKDouble x_5_0_1 = Ptempy * x_3_0_1 + WPtempy * x_3_0_2;
            QUICKDouble x_6_0_1 = Ptempx * x_3_0_1 + WPtempx * x_3_0_2;
            
            QUICKDouble x_7_0_1 = Ptempx * x_1_0_1 + WPtempx * x_1_0_2+ ABtemp*(VY( 0, 0, 1) - CDcom * VY( 0, 0, 2));
            QUICKDouble x_8_0_1 = Ptempy * x_2_0_1 + WPtempy * x_2_0_2+ ABtemp*(VY( 0, 0, 1) - CDcom * VY( 0, 0, 2));
            QUICKDouble x_9_0_1 = Ptempz * x_3_0_1 + WPtempz * x_3_0_2+ ABtemp*(VY( 0, 0, 1) - CDcom * VY( 0, 0, 2));
            
            //DSPS(0, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz, ABCDtemp);
            LOC2(store, 4, 1, STOREDIM, STOREDIM) += Qtempx * x_4_0_0 + WQtempx * x_4_0_1 + ABCDtemp * x_2_0_1;
            LOC2(store, 4, 2, STOREDIM, STOREDIM) += Qtempy * x_4_0_0 + WQtempy * x_4_0_1 + ABCDtemp * x_1_0_1;
            LOC2(store, 4, 3, STOREDIM, STOREDIM) += Qtempz * x_4_0_0 + WQtempz * x_4_0_1;
            
            LOC2(store, 5, 1, STOREDIM, STOREDIM) += Qtempx * x_5_0_0 + WQtempx * x_5_0_1;
            LOC2(store, 5, 2, STOREDIM, STOREDIM) += Qtempy * x_5_0_0 + WQtempy * x_5_0_1 + ABCDtemp * x_3_0_1;
            LOC2(store, 5, 3, STOREDIM, STOREDIM) += Qtempz * x_5_0_0 + WQtempz * x_5_0_1 + ABCDtemp * x_2_0_1;
            
            LOC2(store, 6, 1, STOREDIM, STOREDIM) += Qtempx * x_6_0_0 + WQtempx * x_6_0_1 + ABCDtemp * x_3_0_1;
            LOC2(store, 6, 2, STOREDIM, STOREDIM) += Qtempy * x_6_0_0 + WQtempy * x_6_0_1;
            LOC2(store, 6, 3, STOREDIM, STOREDIM) += Qtempz * x_6_0_0 + WQtempz * x_6_0_1 + ABCDtemp * x_1_0_1;
            
            LOC2(store, 7, 1, STOREDIM, STOREDIM) += Qtempx * x_7_0_0 + WQtempx * x_7_0_1 + ABCDtemp * x_1_0_1 * 2;
            LOC2(store, 7, 2, STOREDIM, STOREDIM) += Qtempy * x_7_0_0 + WQtempy * x_7_0_1;
            LOC2(store, 7, 3, STOREDIM, STOREDIM) += Qtempz * x_7_0_0 + WQtempz * x_7_0_1;
            
            LOC2(store, 8, 1, STOREDIM, STOREDIM) += Qtempx * x_8_0_0 + WQtempx * x_8_0_1;
            LOC2(store, 8, 2, STOREDIM, STOREDIM) += Qtempy * x_8_0_0 + WQtempy * x_8_0_1 + ABCDtemp * x_2_0_1 * 2;
            LOC2(store, 8, 3, STOREDIM, STOREDIM) += Qtempz * x_8_0_0 + WQtempz * x_8_0_1;
            
            LOC2(store, 9, 1, STOREDIM, STOREDIM) += Qtempx * x_9_0_0 + WQtempx * x_9_0_1;
            LOC2(store, 9, 2, STOREDIM, STOREDIM) += Qtempy * x_9_0_0 + WQtempy * x_9_0_1;
            LOC2(store, 9, 3, STOREDIM, STOREDIM) += Qtempz * x_9_0_0 + WQtempz * x_9_0_1 + ABCDtemp * x_3_0_1 * 2;            
            
            if (K==0) {
                LOC2(store, 1, 0, STOREDIM, STOREDIM) += x_1_0_0;
                LOC2(store, 2, 0, STOREDIM, STOREDIM) += x_2_0_0;
                LOC2(store, 3, 0, STOREDIM, STOREDIM) += x_3_0_0;
                LOC2(store, 4, 0, STOREDIM, STOREDIM) += x_4_0_0;
                LOC2(store, 5, 0, STOREDIM, STOREDIM) += x_5_0_0;
                LOC2(store, 6, 0, STOREDIM, STOREDIM) += x_6_0_0;
                LOC2(store, 7, 0, STOREDIM, STOREDIM) += x_7_0_0;
                LOC2(store, 8, 0, STOREDIM, STOREDIM) += x_8_0_0;
                LOC2(store, 9, 0, STOREDIM, STOREDIM) += x_9_0_0;
            }
            
            if (I==0) {
                LOC2(store, 0, 1, STOREDIM, STOREDIM) += x_0_1_0;
                LOC2(store, 0, 2, STOREDIM, STOREDIM) += x_0_2_0;
                LOC2(store, 0, 3, STOREDIM, STOREDIM) += x_0_3_0;
                LOC2(store, 0, 4, STOREDIM, STOREDIM) += x_0_4_0;
                LOC2(store, 0, 5, STOREDIM, STOREDIM) += x_0_5_0;
                LOC2(store, 0, 6, STOREDIM, STOREDIM) += x_0_6_0;
                LOC2(store, 0, 7, STOREDIM, STOREDIM) += x_0_7_0;
                LOC2(store, 0, 8, STOREDIM, STOREDIM) += x_0_8_0;
                LOC2(store, 0, 9, STOREDIM, STOREDIM) += x_0_9_0;
            }
            
            
            //SSPS(3, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);
            QUICKDouble x_0_1_3 = Qtempx * VY( 0, 0, 3) + WQtempx * VY( 0, 0, 4);
            QUICKDouble x_0_2_3 = Qtempy * VY( 0, 0, 3) + WQtempy * VY( 0, 0, 4);
            QUICKDouble x_0_3_3 = Qtempz * VY( 0, 0, 3) + WQtempz * VY( 0, 0, 4);
            
            //SSDS(2, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz, CDtemp, ABcom);
            QUICKDouble x_0_4_2 = Qtempx * x_0_2_2 + WQtempx * x_0_2_3;
            QUICKDouble x_0_5_2 = Qtempy * x_0_3_2 + WQtempy * x_0_3_3;
            QUICKDouble x_0_6_2 = Qtempx * x_0_3_2 + WQtempx * x_0_3_3;
            
            QUICKDouble x_0_7_2 = Qtempx * x_0_1_2 + WQtempx * x_0_1_3+ CDtemp*(VY( 0, 0, 2) - ABcom * VY( 0, 0, 3));
            QUICKDouble x_0_8_2 = Qtempy * x_0_2_2 + WQtempy * x_0_2_3+ CDtemp*(VY( 0, 0, 2) - ABcom * VY( 0, 0, 3));
            QUICKDouble x_0_9_2 = Qtempz * x_0_3_2 + WQtempz * x_0_3_3+ CDtemp*(VY( 0, 0, 2) - ABcom * VY( 0, 0, 3));
            
            //PSDS(1, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz, ABCDtemp);
            QUICKDouble x_1_4_1 = Ptempx * x_0_4_1 + WPtempx * x_0_4_2 + ABCDtemp * x_0_2_2;
            QUICKDouble x_2_4_1 = Ptempy * x_0_4_1 + WPtempy * x_0_4_2 + ABCDtemp * x_0_1_2;
            QUICKDouble x_3_4_1 = Ptempz * x_0_4_1 + WPtempz * x_0_4_2;
            
            QUICKDouble x_1_5_1 = Ptempx * x_0_5_1 + WPtempx * x_0_5_2;
            QUICKDouble x_2_5_1 = Ptempy * x_0_5_1 + WPtempy * x_0_5_2 + ABCDtemp * x_0_3_2;
            QUICKDouble x_3_5_1 = Ptempz * x_0_5_1 + WPtempz * x_0_5_2 + ABCDtemp * x_0_2_2;
            
            QUICKDouble x_1_6_1 = Ptempx * x_0_6_1 + WPtempx * x_0_6_2 + ABCDtemp * x_0_3_2;
            QUICKDouble x_2_6_1 = Ptempy * x_0_6_1 + WPtempy * x_0_6_2;
            QUICKDouble x_3_6_1 = Ptempz * x_0_6_1 + WPtempz * x_0_6_2 + ABCDtemp * x_0_1_2;
            
            QUICKDouble x_1_7_1 = Ptempx * x_0_7_1 + WPtempx * x_0_7_2 + ABCDtemp * x_0_1_2 * 2;
            QUICKDouble x_2_7_1 = Ptempy * x_0_7_1 + WPtempy * x_0_7_2;
            QUICKDouble x_3_7_1 = Ptempz * x_0_7_1 + WPtempz * x_0_7_2;
            
            QUICKDouble x_1_8_1 = Ptempx * x_0_8_1 + WPtempx * x_0_8_2;
            QUICKDouble x_2_8_1 = Ptempy * x_0_8_1 + WPtempy * x_0_8_2 + ABCDtemp * x_0_2_2 * 2;
            QUICKDouble x_3_8_1 = Ptempz * x_0_8_1 + WPtempz * x_0_8_2;
            
            QUICKDouble x_1_9_1 = Ptempx * x_0_9_1 + WPtempx * x_0_9_2;
            QUICKDouble x_2_9_1 = Ptempy * x_0_9_1 + WPtempy * x_0_9_2;
            QUICKDouble x_3_9_1 = Ptempz * x_0_9_1 + WPtempz * x_0_9_2 + ABCDtemp * x_0_3_2 * 2;    
            
            //PSPS(1, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz, ABCDtemp);
            QUICKDouble x_1_1_1 = Ptempx * x_0_1_1 + WPtempx * x_0_1_2 + ABCDtemp * VY( 0, 0, 2);
            QUICKDouble x_2_1_1 = Ptempy * x_0_1_1 + WPtempy * x_0_1_2;
            QUICKDouble x_3_1_1 = Ptempz * x_0_1_1 + WPtempz * x_0_1_2;
            
            QUICKDouble x_1_2_1 = Ptempx * x_0_2_1 + WPtempx * x_0_2_2;
            QUICKDouble x_2_2_1 = Ptempy * x_0_2_1 + WPtempy * x_0_2_2 + ABCDtemp * VY( 0, 0, 2);
            QUICKDouble x_3_2_1 = Ptempz * x_0_2_1 + WPtempz * x_0_2_2;
            
            QUICKDouble x_1_3_1 = Ptempx * x_0_3_1 + WPtempx * x_0_3_2;
            QUICKDouble x_2_3_1 = Ptempy * x_0_3_1 + WPtempy * x_0_3_2;
            QUICKDouble x_3_3_1 = Ptempz * x_0_3_1 + WPtempz * x_0_3_2 + ABCDtemp * VY( 0, 0, 2);
            
            //DSDS(0, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz, ABCDtemp, ABtemp, CDcom);
            
            QUICKDouble x_1_4_0 = Ptempx * x_0_4_0 + WPtempx * x_0_4_1 + ABCDtemp * x_0_2_1;
            LOC2(store, 1, 4, STOREDIM, STOREDIM) += x_1_4_0;
            LOC2(store, 7, 4, STOREDIM, STOREDIM) += Ptempx * x_1_4_0 + WPtempx * x_1_4_1 +  ABtemp * (x_0_4_0 - CDcom * x_0_4_1) + ABCDtemp * x_1_2_1;
            
            QUICKDouble x_2_4_0 = Ptempy * x_0_4_0 + WPtempy * x_0_4_1 + ABCDtemp * x_0_1_1;
            LOC2(store, 2, 4, STOREDIM, STOREDIM) += x_2_4_0;
            LOC2(store, 4, 4, STOREDIM, STOREDIM) += Ptempx * x_2_4_0 + WPtempx * x_2_4_1 + ABCDtemp * x_2_2_1;
            LOC2(store, 8, 4, STOREDIM, STOREDIM) += Ptempy * x_2_4_0 + WPtempy * x_2_4_1 +  ABtemp * (x_0_4_0 - CDcom * x_0_4_1) + ABCDtemp * x_2_1_1;
            
            QUICKDouble x_3_4_0 = Ptempz * x_0_4_0 + WPtempz * x_0_4_1;
            LOC2(store, 3, 4, STOREDIM, STOREDIM) += x_3_4_0;
            LOC2(store, 5, 4, STOREDIM, STOREDIM) += Ptempy * x_3_4_0 + WPtempy * x_3_4_1 + ABCDtemp * x_3_1_1;
            LOC2(store, 9, 4, STOREDIM, STOREDIM) += Ptempz * x_3_4_0 + WPtempz * x_3_4_1 +  ABtemp * (x_0_4_0 - CDcom * x_0_4_1);
            LOC2(store, 6, 4, STOREDIM, STOREDIM) += Ptempx * x_3_4_0 + WPtempx * x_3_4_1 + ABCDtemp * x_3_2_1;
            
            QUICKDouble x_1_5_0 = Ptempx * x_0_5_0 + WPtempx * x_0_5_1;
            LOC2(store, 1, 5, STOREDIM, STOREDIM) += x_1_5_0;
            LOC2(store, 7, 5, STOREDIM, STOREDIM) += Ptempx * x_1_5_0 + WPtempx * x_1_5_1 +  ABtemp * (x_0_5_0 - CDcom * x_0_5_1);
            
            QUICKDouble x_2_5_0 = Ptempy * x_0_5_0 + WPtempy * x_0_5_1 + ABCDtemp * x_0_3_1;
            LOC2(store, 2, 5, STOREDIM, STOREDIM) += x_2_5_0;
            LOC2(store, 4, 5, STOREDIM, STOREDIM) += Ptempx * x_2_5_0 + WPtempx * x_2_5_1;
            LOC2(store, 8, 5, STOREDIM, STOREDIM) += Ptempy * x_2_5_0 + WPtempy * x_2_5_1 +  ABtemp * (x_0_5_0 - CDcom * x_0_5_1) + ABCDtemp * x_2_3_1;
            
            QUICKDouble x_3_5_0 = Ptempz * x_0_5_0 + WPtempz * x_0_5_1 + ABCDtemp * x_0_2_1;
            LOC2(store, 3, 5, STOREDIM, STOREDIM) += x_3_5_0;
            LOC2(store, 5, 5, STOREDIM, STOREDIM) += Ptempy * x_3_5_0 + WPtempy * x_3_5_1 + ABCDtemp * x_3_3_1;
            LOC2(store, 6, 5, STOREDIM, STOREDIM) += Ptempx * x_3_5_0 + WPtempx * x_3_5_1;
            LOC2(store, 9, 5, STOREDIM, STOREDIM) += Ptempz * x_3_5_0 + WPtempz * x_3_5_1 +  ABtemp * (x_0_5_0 - CDcom * x_0_5_1) + ABCDtemp * x_3_2_1;
            
            QUICKDouble x_1_6_0 = Ptempx * x_0_6_0 + WPtempx * x_0_6_1 + ABCDtemp * x_0_3_1;
            LOC2(store, 1, 6, STOREDIM, STOREDIM) += x_1_6_0;
            LOC2(store, 7, 6, STOREDIM, STOREDIM) += Ptempx * x_1_6_0 + WPtempx * x_1_6_1 +  ABtemp * (x_0_6_0 - CDcom * x_0_6_1) + ABCDtemp * x_1_3_1;
            
            QUICKDouble x_2_6_0 = Ptempy * x_0_6_0 + WPtempy * x_0_6_1;
            LOC2(store, 2, 6, STOREDIM, STOREDIM) += x_2_6_0;
            LOC2(store, 4, 6, STOREDIM, STOREDIM) += Ptempx * x_2_6_0 + WPtempx * x_2_6_1 + ABCDtemp * x_2_3_1;
            LOC2(store, 8, 6, STOREDIM, STOREDIM) += Ptempy * x_2_6_0 + WPtempy * x_2_6_1 +  ABtemp * (x_0_6_0 - CDcom * x_0_6_1);
            
            QUICKDouble x_3_6_0 = Ptempz * x_0_6_0 + WPtempz * x_0_6_1 + ABCDtemp * x_0_1_1;
            LOC2(store, 3, 6, STOREDIM, STOREDIM) += x_3_6_0;
            LOC2(store, 5, 6, STOREDIM, STOREDIM) += Ptempy * x_3_6_0 + WPtempy * x_3_6_1;
            LOC2(store, 6, 6, STOREDIM, STOREDIM) += Ptempx * x_3_6_0 + WPtempx * x_3_6_1 + ABCDtemp * x_3_3_1;
            LOC2(store, 9, 6, STOREDIM, STOREDIM) += Ptempz * x_3_6_0 + WPtempz * x_3_6_1 +  ABtemp * (x_0_6_0 - CDcom * x_0_6_1) + ABCDtemp * x_3_1_1;
            
            QUICKDouble x_1_7_0 = Ptempx * x_0_7_0 + WPtempx * x_0_7_1 + ABCDtemp * x_0_1_1 * 2;
            LOC2(store, 1, 7, STOREDIM, STOREDIM) += x_1_7_0;
            LOC2(store, 7, 7, STOREDIM, STOREDIM) += Ptempx * x_1_7_0 + WPtempx * x_1_7_1 +  ABtemp * (x_0_7_0 - CDcom * x_0_7_1) + 2 * ABCDtemp * x_1_1_1;
            
            QUICKDouble x_2_7_0 = Ptempy * x_0_7_0 + WPtempy * x_0_7_1;
            LOC2(store, 2, 7, STOREDIM, STOREDIM) += x_2_7_0;
            LOC2(store, 4, 7, STOREDIM, STOREDIM) += Ptempx * x_2_7_0 + WPtempx * x_2_7_1 + 2 * ABCDtemp * x_2_1_1;
            LOC2(store, 8, 7, STOREDIM, STOREDIM) += Ptempy * x_2_7_0 + WPtempy * x_2_7_1 +  ABtemp * (x_0_7_0 - CDcom * x_0_7_1);
            
            
            QUICKDouble x_3_7_0 = Ptempz * x_0_7_0 + WPtempz * x_0_7_1;
            LOC2(store, 3, 7, STOREDIM, STOREDIM) += x_3_7_0;
            LOC2(store, 5, 7, STOREDIM, STOREDIM) += Ptempy * x_3_7_0 + WPtempy * x_3_7_1;
            LOC2(store, 6, 7, STOREDIM, STOREDIM) += Ptempx * x_3_7_0 + WPtempx * x_3_7_1 + 2 * ABCDtemp * x_3_1_1;
            LOC2(store, 9, 7, STOREDIM, STOREDIM) += Ptempz * x_3_7_0 + WPtempz * x_3_7_1 +  ABtemp * (x_0_7_0 - CDcom * x_0_7_1);
            
            QUICKDouble x_1_8_0 = Ptempx * x_0_8_0 + WPtempx * x_0_8_1;
            LOC2(store, 1, 8, STOREDIM, STOREDIM) += x_1_8_0;
            LOC2(store, 7, 8, STOREDIM, STOREDIM) += Ptempx * x_1_8_0 + WPtempx * x_1_8_1 +  ABtemp * (x_0_8_0 - CDcom * x_0_8_1);
            
            QUICKDouble x_2_8_0 = Ptempy * x_0_8_0 + WPtempy * x_0_8_1 + ABCDtemp * x_0_2_1 * 2;
            LOC2(store, 2, 8, STOREDIM, STOREDIM) += x_2_8_0;
            LOC2(store, 4, 8, STOREDIM, STOREDIM) += Ptempx * x_2_8_0 + WPtempx * x_2_8_1;
            LOC2(store, 8, 8, STOREDIM, STOREDIM) += Ptempy * x_2_8_0 + WPtempy * x_2_8_1 +  ABtemp * (x_0_8_0 - CDcom * x_0_8_1) + 2 * ABCDtemp * x_2_2_1;
            
            QUICKDouble x_3_8_0 = Ptempz * x_0_8_0 + WPtempz * x_0_8_1;
            LOC2(store, 3, 8, STOREDIM, STOREDIM) += x_3_8_0;
            LOC2(store, 5, 8, STOREDIM, STOREDIM) += Ptempy * x_3_8_0 + WPtempy * x_3_8_1 + 2 * ABCDtemp * x_3_2_1;
            LOC2(store, 6, 8, STOREDIM, STOREDIM) += Ptempx * x_3_8_0 + WPtempx * x_3_8_1;
            LOC2(store, 9, 8, STOREDIM, STOREDIM) += Ptempz * x_3_8_0 + WPtempz * x_3_8_1 +  ABtemp * (x_0_8_0 - CDcom * x_0_8_1);
            
            QUICKDouble x_1_9_0 = Ptempx * x_0_9_0 + WPtempx * x_0_9_1;
            LOC2(store, 1, 9, STOREDIM, STOREDIM) += x_1_9_0;
            LOC2(store, 7, 9, STOREDIM, STOREDIM) += Ptempx * x_1_9_0 + WPtempx * x_1_9_1 +  ABtemp * (x_0_9_0 - CDcom * x_0_9_1);
            
            QUICKDouble x_2_9_0 = Ptempy * x_0_9_0 + WPtempy * x_0_9_1;
            LOC2(store, 2, 9, STOREDIM, STOREDIM) += x_2_9_0;
            LOC2(store, 8, 9, STOREDIM, STOREDIM) += Ptempy * x_2_9_0 + WPtempy * x_2_9_1 +  ABtemp * (x_0_9_0 - CDcom * x_0_9_1);
            LOC2(store, 4, 9, STOREDIM, STOREDIM) += Ptempx * x_2_9_0 + WPtempx * x_2_9_1;
            
            QUICKDouble x_3_9_0 = Ptempz * x_0_9_0 + WPtempz * x_0_9_1 + ABCDtemp * x_0_3_1 * 2; 
            LOC2(store, 3, 9, STOREDIM, STOREDIM) += x_3_9_0;
            LOC2(store, 5, 9, STOREDIM, STOREDIM) += Ptempy * x_3_9_0 + WPtempy * x_3_9_1;
            LOC2(store, 6, 9, STOREDIM, STOREDIM) += Ptempx * x_3_9_0 + WPtempx * x_3_9_1;
            LOC2(store, 9, 9, STOREDIM, STOREDIM) += Ptempz * x_3_9_0 + WPtempz * x_3_9_1 +  ABtemp * (x_0_9_0 - CDcom * x_0_9_1) + 2 * ABCDtemp * x_3_3_1;break;
        }
        default:
        {
#ifndef CUDA_SP
            //printf("fall into default case, I, J, K, L are %d, %d, %d, %d\n",I, J, K, L);
			if (K+L>=1){
                //SSPS(0, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);
                QUICKDouble x_0_1_0 = Qtempx * VY( 0, 0, 0) + WQtempx * VY( 0, 0, 1);
                QUICKDouble x_0_2_0 = Qtempy * VY( 0, 0, 0) + WQtempy * VY( 0, 0, 1);
                QUICKDouble x_0_3_0 = Qtempz * VY( 0, 0, 0) + WQtempz * VY( 0, 0, 1);
                
                LOC2(store, 0, 1, STOREDIM, STOREDIM) += x_0_1_0;
                LOC2(store, 0, 2, STOREDIM, STOREDIM) += x_0_2_0;
                LOC2(store, 0, 3, STOREDIM, STOREDIM) += x_0_3_0;
                
                //SSPS(1, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);
                QUICKDouble x_0_1_1 = Qtempx * VY( 0, 0, 1) + WQtempx * VY( 0, 0, 2);
                QUICKDouble x_0_2_1 = Qtempy * VY( 0, 0, 1) + WQtempy * VY( 0, 0, 2);
                QUICKDouble x_0_3_1 = Qtempz * VY( 0, 0, 1) + WQtempz * VY( 0, 0, 2);
                
                
                if (K+L>=1 && I+J>=1) {
                    //PSPS(0, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz, ABCDtemp);
                    QUICKDouble x_1_1_0 = Ptempx * x_0_1_0 + WPtempx * x_0_1_1 + ABCDtemp * VY( 0, 0, 1);
                    QUICKDouble x_2_1_0 = Ptempy * x_0_1_0 + WPtempy * x_0_1_1;
                    QUICKDouble x_3_1_0 = Ptempz * x_0_1_0 + WPtempz * x_0_1_1;
                    
                    QUICKDouble x_1_2_0 = Ptempx * x_0_2_0 + WPtempx * x_0_2_1;
                    QUICKDouble x_2_2_0 = Ptempy * x_0_2_0 + WPtempy * x_0_2_1 + ABCDtemp * VY( 0, 0, 1);
                    QUICKDouble x_3_2_0 = Ptempz * x_0_2_0 + WPtempz * x_0_2_1;
                    
                    QUICKDouble x_1_3_0 = Ptempx * x_0_3_0 + WPtempx * x_0_3_1;
                    QUICKDouble x_2_3_0 = Ptempy * x_0_3_0 + WPtempy * x_0_3_1;
                    QUICKDouble x_3_3_0 = Ptempz * x_0_3_0 + WPtempz * x_0_3_1 + ABCDtemp * VY( 0, 0, 1);
                    
                    LOC2(store, 1, 1, STOREDIM, STOREDIM) += x_1_1_0;
                    LOC2(store, 1, 2, STOREDIM, STOREDIM) += x_1_2_0;
                    LOC2(store, 1, 3, STOREDIM, STOREDIM) += x_1_3_0;
                    
                    LOC2(store, 2, 1, STOREDIM, STOREDIM) += x_2_1_0;
                    LOC2(store, 2, 2, STOREDIM, STOREDIM) += x_2_2_0;
                    LOC2(store, 2, 3, STOREDIM, STOREDIM) += x_2_3_0;
                    
                    LOC2(store, 3, 1, STOREDIM, STOREDIM) += x_3_1_0;
                    LOC2(store, 3, 2, STOREDIM, STOREDIM) += x_3_2_0;
                    LOC2(store, 3, 3, STOREDIM, STOREDIM) += x_3_3_0;
                }
                
                if (K+L>=2){
                    //SSDS(0, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz, CDtemp, ABcom);
                    
                    QUICKDouble x_0_4_0 = Qtempx * x_0_2_0 + WQtempx * x_0_2_1;
                    QUICKDouble x_0_5_0 = Qtempy * x_0_3_0 + WQtempy * x_0_3_1;
                    QUICKDouble x_0_6_0 = Qtempx * x_0_3_0 + WQtempx * x_0_3_1;
                    
                    QUICKDouble x_0_7_0 = Qtempx * x_0_1_0 + WQtempx * x_0_1_1+ CDtemp*(VY( 0, 0, 0) - ABcom * VY( 0, 0, 1));
                    QUICKDouble x_0_8_0 = Qtempy * x_0_2_0 + WQtempy * x_0_2_1+ CDtemp*(VY( 0, 0, 0) - ABcom * VY( 0, 0, 1));
                    QUICKDouble x_0_9_0 = Qtempz * x_0_3_0 + WQtempz * x_0_3_1+ CDtemp*(VY( 0, 0, 0) - ABcom * VY( 0, 0, 1));
                    
                    LOC2(store, 0, 4, STOREDIM, STOREDIM) += x_0_4_0;
                    LOC2(store, 0, 5, STOREDIM, STOREDIM) += x_0_5_0;
                    LOC2(store, 0, 6, STOREDIM, STOREDIM) += x_0_6_0;
                    LOC2(store, 0, 7, STOREDIM, STOREDIM) += x_0_7_0;
                    LOC2(store, 0, 8, STOREDIM, STOREDIM) += x_0_8_0;
                    LOC2(store, 0, 9, STOREDIM, STOREDIM) += x_0_9_0;
                    
                    
                    //SSPS(2, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);
                    QUICKDouble x_0_1_2 = Qtempx * VY( 0, 0, 2) + WQtempx * VY( 0, 0, 3);
                    QUICKDouble x_0_2_2 = Qtempy * VY( 0, 0, 2) + WQtempy * VY( 0, 0, 3);
                    QUICKDouble x_0_3_2 = Qtempz * VY( 0, 0, 2) + WQtempz * VY( 0, 0, 3);
                    
                    //SSPS(3, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);
                    QUICKDouble x_0_1_3 = Qtempx * VY( 0, 0, 3) + WQtempx * VY( 0, 0, 4);
                    QUICKDouble x_0_2_3 = Qtempy * VY( 0, 0, 3) + WQtempy * VY( 0, 0, 4);
                    QUICKDouble x_0_3_3 = Qtempz * VY( 0, 0, 3) + WQtempz * VY( 0, 0, 4);
                    
                    //SSDS(2, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz, CDtemp, ABcom);
                    
                    QUICKDouble x_0_4_2 = Qtempx * x_0_2_2 + WQtempx * x_0_2_3;
                    QUICKDouble x_0_5_2 = Qtempy * x_0_3_2 + WQtempy * x_0_3_3;
                    QUICKDouble x_0_6_2 = Qtempx * x_0_3_2 + WQtempx * x_0_3_3;
                    
                    QUICKDouble x_0_7_2 = Qtempx * x_0_1_2 + WQtempx * x_0_1_3+ CDtemp*(VY( 0, 0, 2) - ABcom * VY( 0, 0, 3));
                    QUICKDouble x_0_8_2 = Qtempy * x_0_2_2 + WQtempy * x_0_2_3+ CDtemp*(VY( 0, 0, 2) - ABcom * VY( 0, 0, 3));
                    QUICKDouble x_0_9_2 = Qtempz * x_0_3_2 + WQtempz * x_0_3_3+ CDtemp*(VY( 0, 0, 2) - ABcom * VY( 0, 0, 3));
                    
                    //SSDS(1, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz, CDtemp, ABcom);
                    
                    QUICKDouble x_0_4_1 = Qtempx * x_0_2_1 + WQtempx * x_0_2_2;
                    QUICKDouble x_0_5_1 = Qtempy * x_0_3_1 + WQtempy * x_0_3_2;
                    QUICKDouble x_0_6_1 = Qtempx * x_0_3_1 + WQtempx * x_0_3_2;
                    
                    QUICKDouble x_0_7_1 = Qtempx * x_0_1_1 + WQtempx * x_0_1_2+ CDtemp*(VY( 0, 0, 1) - ABcom * VY( 0, 0, 2));
                    QUICKDouble x_0_8_1 = Qtempy * x_0_2_1 + WQtempy * x_0_2_2+ CDtemp*(VY( 0, 0, 1) - ABcom * VY( 0, 0, 2));
                    QUICKDouble x_0_9_1 = Qtempz * x_0_3_1 + WQtempz * x_0_3_2+ CDtemp*(VY( 0, 0, 1) - ABcom * VY( 0, 0, 2));
                    
                    if (K+L>=2 && I+J>=1){
                        
                        //PSDS(0, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz, ABCDtemp);
                        
                        QUICKDouble x_1_4_0 = Ptempx * x_0_4_0 + WPtempx * x_0_4_1 + ABCDtemp * x_0_2_1;
                        QUICKDouble x_2_4_0 = Ptempy * x_0_4_0 + WPtempy * x_0_4_1 + ABCDtemp * x_0_1_1;
                        QUICKDouble x_3_4_0 = Ptempz * x_0_4_0 + WPtempz * x_0_4_1;
                        
                        QUICKDouble x_1_5_0 = Ptempx * x_0_5_0 + WPtempx * x_0_5_1;
                        QUICKDouble x_2_5_0 = Ptempy * x_0_5_0 + WPtempy * x_0_5_1 + ABCDtemp * x_0_3_1;
                        QUICKDouble x_3_5_0 = Ptempz * x_0_5_0 + WPtempz * x_0_5_1 + ABCDtemp * x_0_2_1;
                        
                        QUICKDouble x_1_6_0 = Ptempx * x_0_6_0 + WPtempx * x_0_6_1 + ABCDtemp * x_0_3_1;
                        QUICKDouble x_2_6_0 = Ptempy * x_0_6_0 + WPtempy * x_0_6_1;
                        QUICKDouble x_3_6_0 = Ptempz * x_0_6_0 + WPtempz * x_0_6_1 + ABCDtemp * x_0_1_1;
                        
                        QUICKDouble x_1_7_0 = Ptempx * x_0_7_0 + WPtempx * x_0_7_1 + ABCDtemp * x_0_1_1 * 2;
                        QUICKDouble x_2_7_0 = Ptempy * x_0_7_0 + WPtempy * x_0_7_1;
                        QUICKDouble x_3_7_0 = Ptempz * x_0_7_0 + WPtempz * x_0_7_1;
                        
                        QUICKDouble x_1_8_0 = Ptempx * x_0_8_0 + WPtempx * x_0_8_1;
                        QUICKDouble x_2_8_0 = Ptempy * x_0_8_0 + WPtempy * x_0_8_1 + ABCDtemp * x_0_2_1 * 2;
                        QUICKDouble x_3_8_0 = Ptempz * x_0_8_0 + WPtempz * x_0_8_1;
                        
                        QUICKDouble x_1_9_0 = Ptempx * x_0_9_0 + WPtempx * x_0_9_1;
                        QUICKDouble x_2_9_0 = Ptempy * x_0_9_0 + WPtempy * x_0_9_1;
                        QUICKDouble x_3_9_0 = Ptempz * x_0_9_0 + WPtempz * x_0_9_1 + ABCDtemp * x_0_3_1 * 2;    
                        
                        LOC2(store, 1, 4, STOREDIM, STOREDIM) += x_1_4_0;
                        LOC2(store, 1, 5, STOREDIM, STOREDIM) += x_1_5_0;
                        LOC2(store, 1, 6, STOREDIM, STOREDIM) += x_1_6_0;
                        LOC2(store, 1, 7, STOREDIM, STOREDIM) += x_1_7_0;
                        LOC2(store, 1, 8, STOREDIM, STOREDIM) += x_1_8_0;
                        LOC2(store, 1, 9, STOREDIM, STOREDIM) += x_1_9_0;
                        
                        LOC2(store, 2, 4, STOREDIM, STOREDIM) += x_2_4_0;
                        LOC2(store, 2, 5, STOREDIM, STOREDIM) += x_2_5_0;
                        LOC2(store, 2, 6, STOREDIM, STOREDIM) += x_2_6_0;
                        LOC2(store, 2, 7, STOREDIM, STOREDIM) += x_2_7_0;
                        LOC2(store, 2, 8, STOREDIM, STOREDIM) += x_2_8_0;
                        LOC2(store, 2, 9, STOREDIM, STOREDIM) += x_2_9_0;
                        
                        LOC2(store, 3, 4, STOREDIM, STOREDIM) += x_3_4_0;
                        LOC2(store, 3, 5, STOREDIM, STOREDIM) += x_3_5_0;
                        LOC2(store, 3, 6, STOREDIM, STOREDIM) += x_3_6_0;
                        LOC2(store, 3, 7, STOREDIM, STOREDIM) += x_3_7_0;
                        LOC2(store, 3, 8, STOREDIM, STOREDIM) += x_3_8_0;
                        LOC2(store, 3, 9, STOREDIM, STOREDIM) += x_3_9_0;
                        
                        if (K+L>=2 && I+J>=2) {
                            
                            //PSPS(1, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz, ABCDtemp);
                            QUICKDouble x_1_1_1 = Ptempx * x_0_1_1 + WPtempx * x_0_1_2 + ABCDtemp * VY( 0, 0, 2);
                            QUICKDouble x_2_1_1 = Ptempy * x_0_1_1 + WPtempy * x_0_1_2;
                            QUICKDouble x_3_1_1 = Ptempz * x_0_1_1 + WPtempz * x_0_1_2;
                            
                            QUICKDouble x_1_2_1 = Ptempx * x_0_2_1 + WPtempx * x_0_2_2;
                            QUICKDouble x_2_2_1 = Ptempy * x_0_2_1 + WPtempy * x_0_2_2 + ABCDtemp * VY( 0, 0, 2);
                            QUICKDouble x_3_2_1 = Ptempz * x_0_2_1 + WPtempz * x_0_2_2;
                            
                            QUICKDouble x_1_3_1 = Ptempx * x_0_3_1 + WPtempx * x_0_3_2;
                            QUICKDouble x_2_3_1 = Ptempy * x_0_3_1 + WPtempy * x_0_3_2;
                            QUICKDouble x_3_3_1 = Ptempz * x_0_3_1 + WPtempz * x_0_3_2 + ABCDtemp * VY( 0, 0, 2);
                            
                            //PSDS(1, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz, ABCDtemp);
                            
                            QUICKDouble x_1_4_1 = Ptempx * x_0_4_1 + WPtempx * x_0_4_2 + ABCDtemp * x_0_2_2;
                            QUICKDouble x_2_4_1 = Ptempy * x_0_4_1 + WPtempy * x_0_4_2 + ABCDtemp * x_0_1_2;
                            QUICKDouble x_3_4_1 = Ptempz * x_0_4_1 + WPtempz * x_0_4_2;
                            
                            QUICKDouble x_1_5_1 = Ptempx * x_0_5_1 + WPtempx * x_0_5_2;
                            QUICKDouble x_2_5_1 = Ptempy * x_0_5_1 + WPtempy * x_0_5_2 + ABCDtemp * x_0_3_2;
                            QUICKDouble x_3_5_1 = Ptempz * x_0_5_1 + WPtempz * x_0_5_2 + ABCDtemp * x_0_2_2;
                            
                            QUICKDouble x_1_6_1 = Ptempx * x_0_6_1 + WPtempx * x_0_6_2 + ABCDtemp * x_0_3_2;
                            QUICKDouble x_2_6_1 = Ptempy * x_0_6_1 + WPtempy * x_0_6_2;
                            QUICKDouble x_3_6_1 = Ptempz * x_0_6_1 + WPtempz * x_0_6_2 + ABCDtemp * x_0_1_2;
                            
                            QUICKDouble x_1_7_1 = Ptempx * x_0_7_1 + WPtempx * x_0_7_2 + ABCDtemp * x_0_1_2 * 2;
                            QUICKDouble x_2_7_1 = Ptempy * x_0_7_1 + WPtempy * x_0_7_2;
                            QUICKDouble x_3_7_1 = Ptempz * x_0_7_1 + WPtempz * x_0_7_2;
                            
                            QUICKDouble x_1_8_1 = Ptempx * x_0_8_1 + WPtempx * x_0_8_2;
                            QUICKDouble x_2_8_1 = Ptempy * x_0_8_1 + WPtempy * x_0_8_2 + ABCDtemp * x_0_2_2 * 2;
                            QUICKDouble x_3_8_1 = Ptempz * x_0_8_1 + WPtempz * x_0_8_2;
                            
                            QUICKDouble x_1_9_1 = Ptempx * x_0_9_1 + WPtempx * x_0_9_2;
                            QUICKDouble x_2_9_1 = Ptempy * x_0_9_1 + WPtempy * x_0_9_2;
                            QUICKDouble x_3_9_1 = Ptempz * x_0_9_1 + WPtempz * x_0_9_2 + ABCDtemp * x_0_3_2 * 2;    
                            
                            //DSDS(0, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz, ABCDtemp, ABtemp, CDcom);
                            
                            LOC2(store, 4, 4, STOREDIM, STOREDIM) += Ptempx * x_2_4_0 + WPtempx * x_2_4_1 + ABCDtemp * x_2_2_1;
                            LOC2(store, 4, 5, STOREDIM, STOREDIM) += Ptempx * x_2_5_0 + WPtempx * x_2_5_1;
                            LOC2(store, 4, 6, STOREDIM, STOREDIM) += Ptempx * x_2_6_0 + WPtempx * x_2_6_1 + ABCDtemp * x_2_3_1;
                            LOC2(store, 4, 7, STOREDIM, STOREDIM) += Ptempx * x_2_7_0 + WPtempx * x_2_7_1 + 2 * ABCDtemp * x_2_1_1;
                            LOC2(store, 4, 8, STOREDIM, STOREDIM) += Ptempx * x_2_8_0 + WPtempx * x_2_8_1;
                            LOC2(store, 4, 9, STOREDIM, STOREDIM) += Ptempx * x_2_9_0 + WPtempx * x_2_9_1;
                            
                            LOC2(store, 5, 4, STOREDIM, STOREDIM) += Ptempy * x_3_4_0 + WPtempy * x_3_4_1 + ABCDtemp * x_3_1_1;
                            LOC2(store, 5, 5, STOREDIM, STOREDIM) += Ptempy * x_3_5_0 + WPtempy * x_3_5_1 + ABCDtemp * x_3_3_1;
                            LOC2(store, 5, 6, STOREDIM, STOREDIM) += Ptempy * x_3_6_0 + WPtempy * x_3_6_1;
                            LOC2(store, 5, 7, STOREDIM, STOREDIM) += Ptempy * x_3_7_0 + WPtempy * x_3_7_1;
                            LOC2(store, 5, 8, STOREDIM, STOREDIM) += Ptempy * x_3_8_0 + WPtempy * x_3_8_1 + 2 * ABCDtemp * x_3_2_1;
                            LOC2(store, 5, 9, STOREDIM, STOREDIM) += Ptempy * x_3_9_0 + WPtempy * x_3_9_1;
                            
                            LOC2(store, 6, 4, STOREDIM, STOREDIM) += Ptempx * x_3_4_0 + WPtempx * x_3_4_1 + ABCDtemp * x_3_2_1;
                            LOC2(store, 6, 5, STOREDIM, STOREDIM) += Ptempx * x_3_5_0 + WPtempx * x_3_5_1;
                            LOC2(store, 6, 6, STOREDIM, STOREDIM) += Ptempx * x_3_6_0 + WPtempx * x_3_6_1 + ABCDtemp * x_3_3_1;
                            LOC2(store, 6, 7, STOREDIM, STOREDIM) += Ptempx * x_3_7_0 + WPtempx * x_3_7_1 + 2 * ABCDtemp * x_3_1_1;
                            LOC2(store, 6, 8, STOREDIM, STOREDIM) += Ptempx * x_3_8_0 + WPtempx * x_3_8_1;
                            LOC2(store, 6, 9, STOREDIM, STOREDIM) += Ptempx * x_3_9_0 + WPtempx * x_3_9_1;
                            
                            LOC2(store, 7, 4, STOREDIM, STOREDIM) += Ptempx * x_1_4_0 + WPtempx * x_1_4_1 +  ABtemp * (x_0_4_0 - CDcom * x_0_4_1) + ABCDtemp * x_1_2_1;
                            LOC2(store, 7, 5, STOREDIM, STOREDIM) += Ptempx * x_1_5_0 + WPtempx * x_1_5_1 +  ABtemp * (x_0_5_0 - CDcom * x_0_5_1);
                            LOC2(store, 7, 6, STOREDIM, STOREDIM) += Ptempx * x_1_6_0 + WPtempx * x_1_6_1 +  ABtemp * (x_0_6_0 - CDcom * x_0_6_1) + ABCDtemp * x_1_3_1;
                            LOC2(store, 7, 7, STOREDIM, STOREDIM) += Ptempx * x_1_7_0 + WPtempx * x_1_7_1 +  ABtemp * (x_0_7_0 - CDcom * x_0_7_1) + 2 * ABCDtemp * x_1_1_1;
                            LOC2(store, 7, 8, STOREDIM, STOREDIM) += Ptempx * x_1_8_0 + WPtempx * x_1_8_1 +  ABtemp * (x_0_8_0 - CDcom * x_0_8_1);
                            LOC2(store, 7, 9, STOREDIM, STOREDIM) += Ptempx * x_1_9_0 + WPtempx * x_1_9_1 +  ABtemp * (x_0_9_0 - CDcom * x_0_9_1);
                            
                            
                            LOC2(store, 8, 4, STOREDIM, STOREDIM) += Ptempy * x_2_4_0 + WPtempy * x_2_4_1 +  ABtemp * (x_0_4_0 - CDcom * x_0_4_1) + ABCDtemp * x_2_1_1;
                            LOC2(store, 8, 5, STOREDIM, STOREDIM) += Ptempy * x_2_5_0 + WPtempy * x_2_5_1 +  ABtemp * (x_0_5_0 - CDcom * x_0_5_1) + ABCDtemp * x_2_3_1;
                            LOC2(store, 8, 6, STOREDIM, STOREDIM) += Ptempy * x_2_6_0 + WPtempy * x_2_6_1 +  ABtemp * (x_0_6_0 - CDcom * x_0_6_1);
                            LOC2(store, 8, 7, STOREDIM, STOREDIM) += Ptempy * x_2_7_0 + WPtempy * x_2_7_1 +  ABtemp * (x_0_7_0 - CDcom * x_0_7_1);
                            LOC2(store, 8, 8, STOREDIM, STOREDIM) += Ptempy * x_2_8_0 + WPtempy * x_2_8_1 +  ABtemp * (x_0_8_0 - CDcom * x_0_8_1) + 2 * ABCDtemp * x_2_2_1;
                            LOC2(store, 8, 9, STOREDIM, STOREDIM) += Ptempy * x_2_9_0 + WPtempy * x_2_9_1 +  ABtemp * (x_0_9_0 - CDcom * x_0_9_1);
                            
                            LOC2(store, 9, 4, STOREDIM, STOREDIM) += Ptempz * x_3_4_0 + WPtempz * x_3_4_1 +  ABtemp * (x_0_4_0 - CDcom * x_0_4_1);
                            LOC2(store, 9, 5, STOREDIM, STOREDIM) += Ptempz * x_3_5_0 + WPtempz * x_3_5_1 +  ABtemp * (x_0_5_0 - CDcom * x_0_5_1) + ABCDtemp * x_3_2_1;
                            LOC2(store, 9, 6, STOREDIM, STOREDIM) += Ptempz * x_3_6_0 + WPtempz * x_3_6_1 +  ABtemp * (x_0_6_0 - CDcom * x_0_6_1) + ABCDtemp * x_3_1_1;
                            LOC2(store, 9, 7, STOREDIM, STOREDIM) += Ptempz * x_3_7_0 + WPtempz * x_3_7_1 +  ABtemp * (x_0_7_0 - CDcom * x_0_7_1);
                            LOC2(store, 9, 8, STOREDIM, STOREDIM) += Ptempz * x_3_8_0 + WPtempz * x_3_8_1 +  ABtemp * (x_0_8_0 - CDcom * x_0_8_1);
                            LOC2(store, 9, 9, STOREDIM, STOREDIM) += Ptempz * x_3_9_0 + WPtempz * x_3_9_1 +  ABtemp * (x_0_9_0 - CDcom * x_0_9_1) + 2 * ABCDtemp * x_3_3_1;
                            
                        }
                    }
                    if (K+L>=3) {
                        //SSFS(0, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz, CDtemp, ABcom);
                        
                        QUICKDouble x_0_10_0 = Qtempx * x_0_5_0 + WQtempx * x_0_5_1;
                        QUICKDouble x_0_11_0 = Qtempx * x_0_4_0 + WQtempx * x_0_4_1 + CDtemp * ( x_0_2_0 - ABcom * x_0_2_1);
                        QUICKDouble x_0_12_0 = Qtempx * x_0_8_0 + WQtempx * x_0_8_1;
                        QUICKDouble x_0_13_0 = Qtempx * x_0_6_0 + WQtempx * x_0_6_1 + CDtemp * ( x_0_3_0 - ABcom * x_0_3_1);
                        QUICKDouble x_0_14_0 = Qtempx * x_0_9_0 + WQtempx * x_0_9_1;
                        QUICKDouble x_0_15_0 = Qtempy * x_0_5_0 + WQtempy * x_0_5_1 + CDtemp * ( x_0_3_0 - ABcom * x_0_3_1);
                        QUICKDouble x_0_16_0 = Qtempy * x_0_9_0 + WQtempy * x_0_9_1;
                        QUICKDouble x_0_17_0 = Qtempx * x_0_7_0 + WQtempx * x_0_7_1 + 2 * CDtemp * ( x_0_1_0 - ABcom * x_0_1_1);
                        QUICKDouble x_0_18_0 = Qtempy * x_0_8_0 + WQtempy * x_0_8_1 + 2 * CDtemp * ( x_0_2_0 - ABcom * x_0_2_1);
                        QUICKDouble x_0_19_0 = Qtempz * x_0_9_0 + WQtempz * x_0_9_1 + 2 * CDtemp * ( x_0_3_0 - ABcom * x_0_3_1);
                        
                        LOC2(store, 0,10, STOREDIM, STOREDIM) += x_0_10_0;
                        LOC2(store, 0,11, STOREDIM, STOREDIM) += x_0_11_0;
                        LOC2(store, 0,12, STOREDIM, STOREDIM) += x_0_12_0;
                        LOC2(store, 0,13, STOREDIM, STOREDIM) += x_0_13_0;
                        LOC2(store, 0,14, STOREDIM, STOREDIM) += x_0_14_0;
                        LOC2(store, 0,15, STOREDIM, STOREDIM) += x_0_15_0;
                        LOC2(store, 0,16, STOREDIM, STOREDIM) += x_0_16_0;
                        LOC2(store, 0,17, STOREDIM, STOREDIM) += x_0_17_0;
                        LOC2(store, 0,18, STOREDIM, STOREDIM) += x_0_18_0;
                        LOC2(store, 0,19, STOREDIM, STOREDIM) += x_0_19_0;
                        
                        //SSPS(4, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);
                        QUICKDouble x_0_1_4 = Qtempx * VY( 0, 0, 4) + WQtempx * VY( 0, 0, 5);
                        QUICKDouble x_0_2_4 = Qtempy * VY( 0, 0, 4) + WQtempy * VY( 0, 0, 5);
                        QUICKDouble x_0_3_4 = Qtempz * VY( 0, 0, 4) + WQtempz * VY( 0, 0, 5);
                        
                        //SSDS(3, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz, CDtemp, ABcom);
                        
                        QUICKDouble x_0_4_3 = Qtempx * x_0_2_3 + WQtempx * x_0_2_4;
                        QUICKDouble x_0_5_3 = Qtempy * x_0_3_3 + WQtempy * x_0_3_4;
                        QUICKDouble x_0_6_3 = Qtempx * x_0_3_3 + WQtempx * x_0_3_4;
                        
                        QUICKDouble x_0_7_3 = Qtempx * x_0_1_3 + WQtempx * x_0_1_4+ CDtemp*(VY( 0, 0, 3) - ABcom * VY( 0, 0, 4));
                        QUICKDouble x_0_8_3 = Qtempy * x_0_2_3 + WQtempy * x_0_2_4+ CDtemp*(VY( 0, 0, 3) - ABcom * VY( 0, 0, 4));
                        QUICKDouble x_0_9_3 = Qtempz * x_0_3_3 + WQtempz * x_0_3_4+ CDtemp*(VY( 0, 0, 3) - ABcom * VY( 0, 0, 4));
                        
                        if (K+L>=3 && I+J>=1){
                            //SSFS(1, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz, CDtemp, ABcom);
                            
                            QUICKDouble x_0_10_1 = Qtempx * x_0_5_1 + WQtempx * x_0_5_2;
                            QUICKDouble x_0_11_1 = Qtempx * x_0_4_1 + WQtempx * x_0_4_2 + CDtemp * ( x_0_2_1 - ABcom * x_0_2_2);
                            QUICKDouble x_0_12_1 = Qtempx * x_0_8_1 + WQtempx * x_0_8_2;
                            QUICKDouble x_0_13_1 = Qtempx * x_0_6_1 + WQtempx * x_0_6_2 + CDtemp * ( x_0_3_1 - ABcom * x_0_3_2);
                            QUICKDouble x_0_14_1 = Qtempx * x_0_9_1 + WQtempx * x_0_9_2;
                            QUICKDouble x_0_15_1 = Qtempy * x_0_5_1 + WQtempy * x_0_5_2 + CDtemp * ( x_0_3_1 - ABcom * x_0_3_2);
                            QUICKDouble x_0_16_1 = Qtempy * x_0_9_1 + WQtempy * x_0_9_2;
                            QUICKDouble x_0_17_1 = Qtempx * x_0_7_1 + WQtempx * x_0_7_2 + 2 * CDtemp * ( x_0_1_1 - ABcom * x_0_1_2);
                            QUICKDouble x_0_18_1 = Qtempy * x_0_8_1 + WQtempy * x_0_8_2 + 2 * CDtemp * ( x_0_2_1 - ABcom * x_0_2_2);
                            QUICKDouble x_0_19_1 = Qtempz * x_0_9_1 + WQtempz * x_0_9_2 + 2 * CDtemp * ( x_0_3_1 - ABcom * x_0_3_2);
                            
                            //PSFS(0, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz, ABCDtemp);
                            QUICKDouble x_1_10_0 = Ptempx * x_0_10_0 + WPtempx * x_0_10_1 + ABCDtemp *  x_0_5_1;
                            QUICKDouble x_2_10_0 = Ptempy * x_0_10_0 + WPtempy * x_0_10_1 + ABCDtemp *  x_0_6_1;
                            QUICKDouble x_3_10_0 = Ptempz * x_0_10_0 + WPtempz * x_0_10_1 + ABCDtemp *  x_0_4_1;
                            QUICKDouble x_1_11_0 = Ptempx * x_0_11_0 + WPtempx * x_0_11_1 +  2 * ABCDtemp *  x_0_4_1;
                            QUICKDouble x_2_11_0 = Ptempy * x_0_11_0 + WPtempy * x_0_11_1 + ABCDtemp *  x_0_7_1;
                            QUICKDouble x_3_11_0 = Ptempz * x_0_11_0 + WPtempz * x_0_11_1;
                            QUICKDouble x_1_12_0 = Ptempx * x_0_12_0 + WPtempx * x_0_12_1 + ABCDtemp *  x_0_8_1;
                            QUICKDouble x_2_12_0 = Ptempy * x_0_12_0 + WPtempy * x_0_12_1 +  2 * ABCDtemp *  x_0_4_1;
                            QUICKDouble x_3_12_0 = Ptempz * x_0_12_0 + WPtempz * x_0_12_1;
                            QUICKDouble x_1_13_0 = Ptempx * x_0_13_0 + WPtempx * x_0_13_1 +  2 * ABCDtemp *  x_0_6_1;
                            QUICKDouble x_2_13_0 = Ptempy * x_0_13_0 + WPtempy * x_0_13_1;
                            QUICKDouble x_3_13_0 = Ptempz * x_0_13_0 + WPtempz * x_0_13_1 + ABCDtemp *  x_0_7_1;
                            QUICKDouble x_1_14_0 = Ptempx * x_0_14_0 + WPtempx * x_0_14_1 + ABCDtemp *  x_0_9_1;
                            QUICKDouble x_2_14_0 = Ptempy * x_0_14_0 + WPtempy * x_0_14_1;
                            QUICKDouble x_3_14_0 = Ptempz * x_0_14_0 + WPtempz * x_0_14_1 +  2 * ABCDtemp *  x_0_6_1;
                            QUICKDouble x_1_15_0 = Ptempx * x_0_15_0 + WPtempx * x_0_15_1;
                            QUICKDouble x_2_15_0 = Ptempy * x_0_15_0 + WPtempy * x_0_15_1 +  2 * ABCDtemp *  x_0_5_1;
                            QUICKDouble x_3_15_0 = Ptempz * x_0_15_0 + WPtempz * x_0_15_1 + ABCDtemp *  x_0_8_1;
                            QUICKDouble x_1_16_0 = Ptempx * x_0_16_0 + WPtempx * x_0_16_1;
                            QUICKDouble x_2_16_0 = Ptempy * x_0_16_0 + WPtempy * x_0_16_1 + ABCDtemp *  x_0_9_1;
                            QUICKDouble x_3_16_0 = Ptempz * x_0_16_0 + WPtempz * x_0_16_1 +  2 * ABCDtemp *  x_0_5_1;
                            QUICKDouble x_1_17_0 = Ptempx * x_0_17_0 + WPtempx * x_0_17_1 +  3 * ABCDtemp *  x_0_7_1;
                            QUICKDouble x_2_17_0 = Ptempy * x_0_17_0 + WPtempy * x_0_17_1;
                            QUICKDouble x_3_17_0 = Ptempz * x_0_17_0 + WPtempz * x_0_17_1;
                            QUICKDouble x_1_18_0 = Ptempx * x_0_18_0 + WPtempx * x_0_18_1;
                            QUICKDouble x_2_18_0 = Ptempy * x_0_18_0 + WPtempy * x_0_18_1 +  3 * ABCDtemp *  x_0_8_1;
                            QUICKDouble x_3_18_0 = Ptempz * x_0_18_0 + WPtempz * x_0_18_1;
                            QUICKDouble x_1_19_0 = Ptempx * x_0_19_0 + WPtempx * x_0_19_1;
                            QUICKDouble x_2_19_0 = Ptempy * x_0_19_0 + WPtempy * x_0_19_1;
                            QUICKDouble x_3_19_0 = Ptempz * x_0_19_0 + WPtempz * x_0_19_1 +  3 * ABCDtemp *  x_0_9_1;
                            LOC2(store, 1,10, STOREDIM, STOREDIM) += x_1_10_0;
                            LOC2(store, 1,11, STOREDIM, STOREDIM) += x_1_11_0;
                            LOC2(store, 1,12, STOREDIM, STOREDIM) += x_1_12_0;
                            LOC2(store, 1,13, STOREDIM, STOREDIM) += x_1_13_0;
                            LOC2(store, 1,14, STOREDIM, STOREDIM) += x_1_14_0;
                            LOC2(store, 1,15, STOREDIM, STOREDIM) += x_1_15_0;
                            LOC2(store, 1,16, STOREDIM, STOREDIM) += x_1_16_0;
                            LOC2(store, 1,17, STOREDIM, STOREDIM) += x_1_17_0;
                            LOC2(store, 1,18, STOREDIM, STOREDIM) += x_1_18_0;
                            LOC2(store, 1,19, STOREDIM, STOREDIM) += x_1_19_0;
                            LOC2(store, 2,10, STOREDIM, STOREDIM) += x_2_10_0;
                            LOC2(store, 2,11, STOREDIM, STOREDIM) += x_2_11_0;
                            LOC2(store, 2,12, STOREDIM, STOREDIM) += x_2_12_0;
                            LOC2(store, 2,13, STOREDIM, STOREDIM) += x_2_13_0;
                            LOC2(store, 2,14, STOREDIM, STOREDIM) += x_2_14_0;
                            LOC2(store, 2,15, STOREDIM, STOREDIM) += x_2_15_0;
                            LOC2(store, 2,16, STOREDIM, STOREDIM) += x_2_16_0;
                            LOC2(store, 2,17, STOREDIM, STOREDIM) += x_2_17_0;
                            LOC2(store, 2,18, STOREDIM, STOREDIM) += x_2_18_0;
                            LOC2(store, 2,19, STOREDIM, STOREDIM) += x_2_19_0;
                            LOC2(store, 3,10, STOREDIM, STOREDIM) += x_3_10_0;
                            LOC2(store, 3,11, STOREDIM, STOREDIM) += x_3_11_0;
                            LOC2(store, 3,12, STOREDIM, STOREDIM) += x_3_12_0;
                            LOC2(store, 3,13, STOREDIM, STOREDIM) += x_3_13_0;
                            LOC2(store, 3,14, STOREDIM, STOREDIM) += x_3_14_0;
                            LOC2(store, 3,15, STOREDIM, STOREDIM) += x_3_15_0;
                            LOC2(store, 3,16, STOREDIM, STOREDIM) += x_3_16_0;
                            LOC2(store, 3,17, STOREDIM, STOREDIM) += x_3_17_0;
                            LOC2(store, 3,18, STOREDIM, STOREDIM) += x_3_18_0;
                            LOC2(store, 3,19, STOREDIM, STOREDIM) += x_3_19_0;
                            if (K+L>=3 && I+J>=2) {
                                
                                //PSDS(1, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz, ABCDtemp);
                                
                                QUICKDouble x_1_4_1 = Ptempx * x_0_4_1 + WPtempx * x_0_4_2 + ABCDtemp * x_0_2_2;
                                QUICKDouble x_2_4_1 = Ptempy * x_0_4_1 + WPtempy * x_0_4_2 + ABCDtemp * x_0_1_2;
                                QUICKDouble x_3_4_1 = Ptempz * x_0_4_1 + WPtempz * x_0_4_2;
                                
                                QUICKDouble x_1_5_1 = Ptempx * x_0_5_1 + WPtempx * x_0_5_2;
                                QUICKDouble x_2_5_1 = Ptempy * x_0_5_1 + WPtempy * x_0_5_2 + ABCDtemp * x_0_3_2;
                                QUICKDouble x_3_5_1 = Ptempz * x_0_5_1 + WPtempz * x_0_5_2 + ABCDtemp * x_0_2_2;
                                
                                QUICKDouble x_1_6_1 = Ptempx * x_0_6_1 + WPtempx * x_0_6_2 + ABCDtemp * x_0_3_2;
                                QUICKDouble x_2_6_1 = Ptempy * x_0_6_1 + WPtempy * x_0_6_2;
                                QUICKDouble x_3_6_1 = Ptempz * x_0_6_1 + WPtempz * x_0_6_2 + ABCDtemp * x_0_1_2;
                                
                                QUICKDouble x_1_7_1 = Ptempx * x_0_7_1 + WPtempx * x_0_7_2 + ABCDtemp * x_0_1_2 * 2;
                                QUICKDouble x_2_7_1 = Ptempy * x_0_7_1 + WPtempy * x_0_7_2;
                                QUICKDouble x_3_7_1 = Ptempz * x_0_7_1 + WPtempz * x_0_7_2;
                                
                                QUICKDouble x_1_8_1 = Ptempx * x_0_8_1 + WPtempx * x_0_8_2;
                                QUICKDouble x_2_8_1 = Ptempy * x_0_8_1 + WPtempy * x_0_8_2 + ABCDtemp * x_0_2_2 * 2;
                                QUICKDouble x_3_8_1 = Ptempz * x_0_8_1 + WPtempz * x_0_8_2;
                                
                                QUICKDouble x_1_9_1 = Ptempx * x_0_9_1 + WPtempx * x_0_9_2;
                                QUICKDouble x_2_9_1 = Ptempy * x_0_9_1 + WPtempy * x_0_9_2;
                                QUICKDouble x_3_9_1 = Ptempz * x_0_9_1 + WPtempz * x_0_9_2 + ABCDtemp * x_0_3_2 * 2;    
                                
                                //SSFS(2, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz, CDtemp, ABcom);
                                
                                QUICKDouble x_0_10_2 = Qtempx * x_0_5_2 + WQtempx * x_0_5_3;
                                QUICKDouble x_0_11_2 = Qtempx * x_0_4_2 + WQtempx * x_0_4_3 + CDtemp * ( x_0_2_2 - ABcom * x_0_2_3);
                                QUICKDouble x_0_12_2 = Qtempx * x_0_8_2 + WQtempx * x_0_8_3;
                                QUICKDouble x_0_13_2 = Qtempx * x_0_6_2 + WQtempx * x_0_6_3 + CDtemp * ( x_0_3_2 - ABcom * x_0_3_3);
                                QUICKDouble x_0_14_2 = Qtempx * x_0_9_2 + WQtempx * x_0_9_3;
                                QUICKDouble x_0_15_2 = Qtempy * x_0_5_2 + WQtempy * x_0_5_3 + CDtemp * ( x_0_3_2 - ABcom * x_0_3_3);
                                QUICKDouble x_0_16_2 = Qtempy * x_0_9_2 + WQtempy * x_0_9_3;
                                QUICKDouble x_0_17_2 = Qtempx * x_0_7_2 + WQtempx * x_0_7_3 + 2 * CDtemp * ( x_0_1_2 - ABcom * x_0_1_3);
                                QUICKDouble x_0_18_2 = Qtempy * x_0_8_2 + WQtempy * x_0_8_3 + 2 * CDtemp * ( x_0_2_2 - ABcom * x_0_2_3);
                                QUICKDouble x_0_19_2 = Qtempz * x_0_9_2 + WQtempz * x_0_9_3 + 2 * CDtemp * ( x_0_3_2 - ABcom * x_0_3_3);
                                
                                //PSFS(1, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz, ABCDtemp);
                                
                                QUICKDouble x_1_10_1 = Ptempx * x_0_10_1 + WPtempx * x_0_10_2 + ABCDtemp *  x_0_5_2;
                                QUICKDouble x_2_10_1 = Ptempy * x_0_10_1 + WPtempy * x_0_10_2 + ABCDtemp *  x_0_6_2;
                                QUICKDouble x_3_10_1 = Ptempz * x_0_10_1 + WPtempz * x_0_10_2 + ABCDtemp *  x_0_4_2;
                                QUICKDouble x_1_11_1 = Ptempx * x_0_11_1 + WPtempx * x_0_11_2 +  2 * ABCDtemp *  x_0_4_2;
                                QUICKDouble x_2_11_1 = Ptempy * x_0_11_1 + WPtempy * x_0_11_2 + ABCDtemp *  x_0_7_2;
                                QUICKDouble x_3_11_1 = Ptempz * x_0_11_1 + WPtempz * x_0_11_2;
                                QUICKDouble x_1_12_1 = Ptempx * x_0_12_1 + WPtempx * x_0_12_2 + ABCDtemp *  x_0_8_2;
                                QUICKDouble x_2_12_1 = Ptempy * x_0_12_1 + WPtempy * x_0_12_2 +  2 * ABCDtemp *  x_0_4_2;
                                QUICKDouble x_3_12_1 = Ptempz * x_0_12_1 + WPtempz * x_0_12_2;
                                QUICKDouble x_1_13_1 = Ptempx * x_0_13_1 + WPtempx * x_0_13_2 +  2 * ABCDtemp *  x_0_6_2;
                                QUICKDouble x_2_13_1 = Ptempy * x_0_13_1 + WPtempy * x_0_13_2;
                                QUICKDouble x_3_13_1 = Ptempz * x_0_13_1 + WPtempz * x_0_13_2 + ABCDtemp *  x_0_7_2;
                                QUICKDouble x_1_14_1 = Ptempx * x_0_14_1 + WPtempx * x_0_14_2 + ABCDtemp *  x_0_9_2;
                                QUICKDouble x_2_14_1 = Ptempy * x_0_14_1 + WPtempy * x_0_14_2;
                                QUICKDouble x_3_14_1 = Ptempz * x_0_14_1 + WPtempz * x_0_14_2 +  2 * ABCDtemp *  x_0_6_2;
                                QUICKDouble x_1_15_1 = Ptempx * x_0_15_1 + WPtempx * x_0_15_2;
                                QUICKDouble x_2_15_1 = Ptempy * x_0_15_1 + WPtempy * x_0_15_2 +  2 * ABCDtemp *  x_0_5_2;
                                QUICKDouble x_3_15_1 = Ptempz * x_0_15_1 + WPtempz * x_0_15_2 + ABCDtemp *  x_0_8_2;
                                QUICKDouble x_1_16_1 = Ptempx * x_0_16_1 + WPtempx * x_0_16_2;
                                QUICKDouble x_2_16_1 = Ptempy * x_0_16_1 + WPtempy * x_0_16_2 + ABCDtemp *  x_0_9_2;
                                QUICKDouble x_3_16_1 = Ptempz * x_0_16_1 + WPtempz * x_0_16_2 +  2 * ABCDtemp *  x_0_5_2;
                                QUICKDouble x_1_17_1 = Ptempx * x_0_17_1 + WPtempx * x_0_17_2 +  3 * ABCDtemp *  x_0_7_2;
                                QUICKDouble x_2_17_1 = Ptempy * x_0_17_1 + WPtempy * x_0_17_2;
                                QUICKDouble x_3_17_1 = Ptempz * x_0_17_1 + WPtempz * x_0_17_2;
                                QUICKDouble x_1_18_1 = Ptempx * x_0_18_1 + WPtempx * x_0_18_2;
                                QUICKDouble x_2_18_1 = Ptempy * x_0_18_1 + WPtempy * x_0_18_2 +  3 * ABCDtemp *  x_0_8_2;
                                QUICKDouble x_3_18_1 = Ptempz * x_0_18_1 + WPtempz * x_0_18_2;
                                QUICKDouble x_1_19_1 = Ptempx * x_0_19_1 + WPtempx * x_0_19_2;
                                QUICKDouble x_2_19_1 = Ptempy * x_0_19_1 + WPtempy * x_0_19_2;
                                QUICKDouble x_3_19_1 = Ptempz * x_0_19_1 + WPtempz * x_0_19_2 +  3 * ABCDtemp *  x_0_9_2;
                                
                                //DSFS(0, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz, ABCDtemp, ABtemp, CDcom);
                                
                                LOC2(store, 4,10, STOREDIM, STOREDIM) += Ptempx * x_2_10_0 + WPtempx * x_2_10_1 + ABCDtemp * x_2_5_1;
                                LOC2(store, 4,11, STOREDIM, STOREDIM) += Ptempx * x_2_11_0 + WPtempx * x_2_11_1 +  2 * ABCDtemp * x_2_4_1;
                                LOC2(store, 4,12, STOREDIM, STOREDIM) += Ptempx * x_2_12_0 + WPtempx * x_2_12_1 + ABCDtemp * x_2_8_1;
                                LOC2(store, 4,13, STOREDIM, STOREDIM) += Ptempx * x_2_13_0 + WPtempx * x_2_13_1 +  2 * ABCDtemp * x_2_6_1;
                                LOC2(store, 4,14, STOREDIM, STOREDIM) += Ptempx * x_2_14_0 + WPtempx * x_2_14_1 + ABCDtemp * x_2_9_1;
                                LOC2(store, 4,15, STOREDIM, STOREDIM) += Ptempx * x_2_15_0 + WPtempx * x_2_15_1;
                                LOC2(store, 4,16, STOREDIM, STOREDIM) += Ptempx * x_2_16_0 + WPtempx * x_2_16_1;
                                LOC2(store, 4,17, STOREDIM, STOREDIM) += Ptempx * x_2_17_0 + WPtempx * x_2_17_1 +  3 * ABCDtemp * x_2_7_1;
                                LOC2(store, 4,18, STOREDIM, STOREDIM) += Ptempx * x_2_18_0 + WPtempx * x_2_18_1;
                                LOC2(store, 4,19, STOREDIM, STOREDIM) += Ptempx * x_2_19_0 + WPtempx * x_2_19_1;
                                LOC2(store, 5,10, STOREDIM, STOREDIM) += Ptempy * x_3_10_0 + WPtempy * x_3_10_1 + ABCDtemp * x_3_6_1;
                                LOC2(store, 5,11, STOREDIM, STOREDIM) += Ptempy * x_3_11_0 + WPtempy * x_3_11_1 + ABCDtemp * x_3_7_1;
                                LOC2(store, 5,12, STOREDIM, STOREDIM) += Ptempy * x_3_12_0 + WPtempy * x_3_12_1 +  2 * ABCDtemp * x_3_4_1;
                                LOC2(store, 5,13, STOREDIM, STOREDIM) += Ptempy * x_3_13_0 + WPtempy * x_3_13_1;
                                LOC2(store, 5,14, STOREDIM, STOREDIM) += Ptempy * x_3_14_0 + WPtempy * x_3_14_1;
                                LOC2(store, 5,15, STOREDIM, STOREDIM) += Ptempy * x_3_15_0 + WPtempy * x_3_15_1 +  2 * ABCDtemp * x_3_5_1;
                                LOC2(store, 5,16, STOREDIM, STOREDIM) += Ptempy * x_3_16_0 + WPtempy * x_3_16_1 + ABCDtemp * x_3_9_1;
                                LOC2(store, 5,17, STOREDIM, STOREDIM) += Ptempy * x_3_17_0 + WPtempy * x_3_17_1;
                                LOC2(store, 5,18, STOREDIM, STOREDIM) += Ptempy * x_3_18_0 + WPtempy * x_3_18_1 +  3 * ABCDtemp * x_3_8_1;
                                LOC2(store, 5,19, STOREDIM, STOREDIM) += Ptempy * x_3_19_0 + WPtempy * x_3_19_1;
                                LOC2(store, 6,10, STOREDIM, STOREDIM) += Ptempx * x_3_10_0 + WPtempx * x_3_10_1 + ABCDtemp * x_3_5_1;
                                LOC2(store, 6,11, STOREDIM, STOREDIM) += Ptempx * x_3_11_0 + WPtempx * x_3_11_1 +  2 * ABCDtemp * x_3_4_1;
                                LOC2(store, 6,12, STOREDIM, STOREDIM) += Ptempx * x_3_12_0 + WPtempx * x_3_12_1 + ABCDtemp * x_3_8_1;
                                LOC2(store, 6,13, STOREDIM, STOREDIM) += Ptempx * x_3_13_0 + WPtempx * x_3_13_1 +  2 * ABCDtemp * x_3_6_1;
                                LOC2(store, 6,14, STOREDIM, STOREDIM) += Ptempx * x_3_14_0 + WPtempx * x_3_14_1 + ABCDtemp * x_3_9_1;
                                LOC2(store, 6,15, STOREDIM, STOREDIM) += Ptempx * x_3_15_0 + WPtempx * x_3_15_1;
                                LOC2(store, 6,16, STOREDIM, STOREDIM) += Ptempx * x_3_16_0 + WPtempx * x_3_16_1;
                                LOC2(store, 6,17, STOREDIM, STOREDIM) += Ptempx * x_3_17_0 + WPtempx * x_3_17_1 +  3 * ABCDtemp * x_3_7_1;
                                LOC2(store, 6,18, STOREDIM, STOREDIM) += Ptempx * x_3_18_0 + WPtempx * x_3_18_1;
                                LOC2(store, 6,19, STOREDIM, STOREDIM) += Ptempx * x_3_19_0 + WPtempx * x_3_19_1;
                                LOC2(store, 7,10, STOREDIM, STOREDIM) += Ptempx * x_1_10_0 + WPtempx * x_1_10_1 + ABtemp * ( x_0_10_0 - CDcom * x_0_10_1) + ABCDtemp * x_1_5_1;
                                LOC2(store, 7,11, STOREDIM, STOREDIM) += Ptempx * x_1_11_0 + WPtempx * x_1_11_1 + ABtemp * ( x_0_11_0 - CDcom * x_0_11_1) +  2 * ABCDtemp * x_1_4_1;
                                LOC2(store, 7,12, STOREDIM, STOREDIM) += Ptempx * x_1_12_0 + WPtempx * x_1_12_1 + ABtemp * ( x_0_12_0 - CDcom * x_0_12_1) + ABCDtemp * x_1_8_1;
                                LOC2(store, 7,13, STOREDIM, STOREDIM) += Ptempx * x_1_13_0 + WPtempx * x_1_13_1 + ABtemp * ( x_0_13_0 - CDcom * x_0_13_1) +  2 * ABCDtemp * x_1_6_1;
                                LOC2(store, 7,14, STOREDIM, STOREDIM) += Ptempx * x_1_14_0 + WPtempx * x_1_14_1 + ABtemp * ( x_0_14_0 - CDcom * x_0_14_1) + ABCDtemp * x_1_9_1;
                                LOC2(store, 7,15, STOREDIM, STOREDIM) += Ptempx * x_1_15_0 + WPtempx * x_1_15_1 + ABtemp * ( x_0_15_0 - CDcom * x_0_15_1);
                                LOC2(store, 7,16, STOREDIM, STOREDIM) += Ptempx * x_1_16_0 + WPtempx * x_1_16_1 + ABtemp * ( x_0_16_0 - CDcom * x_0_16_1);
                                LOC2(store, 7,17, STOREDIM, STOREDIM) += Ptempx * x_1_17_0 + WPtempx * x_1_17_1 + ABtemp * ( x_0_17_0 - CDcom * x_0_17_1) +  3 * ABCDtemp * x_1_7_1;
                                LOC2(store, 7,18, STOREDIM, STOREDIM) += Ptempx * x_1_18_0 + WPtempx * x_1_18_1 + ABtemp * ( x_0_18_0 - CDcom * x_0_18_1);
                                LOC2(store, 7,19, STOREDIM, STOREDIM) += Ptempx * x_1_19_0 + WPtempx * x_1_19_1 + ABtemp * ( x_0_19_0 - CDcom * x_0_19_1);
                                LOC2(store, 8,10, STOREDIM, STOREDIM) += Ptempy * x_2_10_0 + WPtempy * x_2_10_1 + ABtemp * ( x_0_10_0 - CDcom * x_0_10_1) + ABCDtemp * x_2_6_1;
                                LOC2(store, 8,11, STOREDIM, STOREDIM) += Ptempy * x_2_11_0 + WPtempy * x_2_11_1 + ABtemp * ( x_0_11_0 - CDcom * x_0_11_1) + ABCDtemp * x_2_7_1;
                                LOC2(store, 8,12, STOREDIM, STOREDIM) += Ptempy * x_2_12_0 + WPtempy * x_2_12_1 + ABtemp * ( x_0_12_0 - CDcom * x_0_12_1) +  2 * ABCDtemp * x_2_4_1;
                                LOC2(store, 8,13, STOREDIM, STOREDIM) += Ptempy * x_2_13_0 + WPtempy * x_2_13_1 + ABtemp * ( x_0_13_0 - CDcom * x_0_13_1);
                                LOC2(store, 8,14, STOREDIM, STOREDIM) += Ptempy * x_2_14_0 + WPtempy * x_2_14_1 + ABtemp * ( x_0_14_0 - CDcom * x_0_14_1);
                                LOC2(store, 8,15, STOREDIM, STOREDIM) += Ptempy * x_2_15_0 + WPtempy * x_2_15_1 + ABtemp * ( x_0_15_0 - CDcom * x_0_15_1) +  2 * ABCDtemp * x_2_5_1;
                                LOC2(store, 8,16, STOREDIM, STOREDIM) += Ptempy * x_2_16_0 + WPtempy * x_2_16_1 + ABtemp * ( x_0_16_0 - CDcom * x_0_16_1) + ABCDtemp * x_2_9_1;
                                LOC2(store, 8,17, STOREDIM, STOREDIM) += Ptempy * x_2_17_0 + WPtempy * x_2_17_1 + ABtemp * ( x_0_17_0 - CDcom * x_0_17_1);
                                LOC2(store, 8,18, STOREDIM, STOREDIM) += Ptempy * x_2_18_0 + WPtempy * x_2_18_1 + ABtemp * ( x_0_18_0 - CDcom * x_0_18_1) +  3 * ABCDtemp * x_2_8_1;
                                LOC2(store, 8,19, STOREDIM, STOREDIM) += Ptempy * x_2_19_0 + WPtempy * x_2_19_1 + ABtemp * ( x_0_19_0 - CDcom * x_0_19_1);
                                LOC2(store, 9,10, STOREDIM, STOREDIM) += Ptempz * x_3_10_0 + WPtempz * x_3_10_1 + ABtemp * ( x_0_10_0 - CDcom * x_0_10_1) + ABCDtemp * x_3_4_1;
                                LOC2(store, 9,11, STOREDIM, STOREDIM) += Ptempz * x_3_11_0 + WPtempz * x_3_11_1 + ABtemp * ( x_0_11_0 - CDcom * x_0_11_1);
                                LOC2(store, 9,12, STOREDIM, STOREDIM) += Ptempz * x_3_12_0 + WPtempz * x_3_12_1 + ABtemp * ( x_0_12_0 - CDcom * x_0_12_1);
                                LOC2(store, 9,13, STOREDIM, STOREDIM) += Ptempz * x_3_13_0 + WPtempz * x_3_13_1 + ABtemp * ( x_0_13_0 - CDcom * x_0_13_1) + ABCDtemp * x_3_7_1;
                                LOC2(store, 9,14, STOREDIM, STOREDIM) += Ptempz * x_3_14_0 + WPtempz * x_3_14_1 + ABtemp * ( x_0_14_0 - CDcom * x_0_14_1) +  2 * ABCDtemp * x_3_6_1;
                                LOC2(store, 9,15, STOREDIM, STOREDIM) += Ptempz * x_3_15_0 + WPtempz * x_3_15_1 + ABtemp * ( x_0_15_0 - CDcom * x_0_15_1) + ABCDtemp * x_3_8_1;
                                LOC2(store, 9,16, STOREDIM, STOREDIM) += Ptempz * x_3_16_0 + WPtempz * x_3_16_1 + ABtemp * ( x_0_16_0 - CDcom * x_0_16_1) +  2 * ABCDtemp * x_3_5_1;
                                LOC2(store, 9,17, STOREDIM, STOREDIM) += Ptempz * x_3_17_0 + WPtempz * x_3_17_1 + ABtemp * ( x_0_17_0 - CDcom * x_0_17_1);
                                LOC2(store, 9,18, STOREDIM, STOREDIM) += Ptempz * x_3_18_0 + WPtempz * x_3_18_1 + ABtemp * ( x_0_18_0 - CDcom * x_0_18_1);
                                LOC2(store, 9,19, STOREDIM, STOREDIM) += Ptempz * x_3_19_0 + WPtempz * x_3_19_1 + ABtemp * ( x_0_19_0 - CDcom * x_0_19_1) +  3 * ABCDtemp * x_3_9_1;
                                
                            }
                        }
                        if (K+L>=4){
                            
                            //SSFS(1, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz, CDtemp, ABcom);
                            
                            QUICKDouble x_0_10_1 = Qtempx * x_0_5_1 + WQtempx * x_0_5_2;
                            QUICKDouble x_0_11_1 = Qtempx * x_0_4_1 + WQtempx * x_0_4_2 + CDtemp * ( x_0_2_1 - ABcom * x_0_2_2);
                            QUICKDouble x_0_12_1 = Qtempx * x_0_8_1 + WQtempx * x_0_8_2;
                            QUICKDouble x_0_13_1 = Qtempx * x_0_6_1 + WQtempx * x_0_6_2 + CDtemp * ( x_0_3_1 - ABcom * x_0_3_2);
                            QUICKDouble x_0_14_1 = Qtempx * x_0_9_1 + WQtempx * x_0_9_2;
                            QUICKDouble x_0_15_1 = Qtempy * x_0_5_1 + WQtempy * x_0_5_2 + CDtemp * ( x_0_3_1 - ABcom * x_0_3_2);
                            QUICKDouble x_0_16_1 = Qtempy * x_0_9_1 + WQtempy * x_0_9_2;
                            QUICKDouble x_0_17_1 = Qtempx * x_0_7_1 + WQtempx * x_0_7_2 + 2 * CDtemp * ( x_0_1_1 - ABcom * x_0_1_2);
                            QUICKDouble x_0_18_1 = Qtempy * x_0_8_1 + WQtempy * x_0_8_2 + 2 * CDtemp * ( x_0_2_1 - ABcom * x_0_2_2);
                            QUICKDouble x_0_19_1 = Qtempz * x_0_9_1 + WQtempz * x_0_9_2 + 2 * CDtemp * ( x_0_3_1 - ABcom * x_0_3_2);
                            
                            //SSGS(0, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz, CDtemp, ABcom);
                            
                            QUICKDouble x_0_20_0 = Qtempx * x_0_12_0 + WQtempx * x_0_12_1 + CDtemp * ( x_0_8_0 - ABcom * x_0_8_1);
                            QUICKDouble x_0_21_0 = Qtempx * x_0_14_0 + WQtempx * x_0_14_1 + CDtemp * ( x_0_9_0 - ABcom * x_0_9_1);
                            QUICKDouble x_0_22_0 = Qtempy * x_0_16_0 + WQtempy * x_0_16_1 + CDtemp * ( x_0_9_0 - ABcom * x_0_9_1);
                            QUICKDouble x_0_23_0 = Qtempx * x_0_10_0 + WQtempx * x_0_10_1 + CDtemp * ( x_0_5_0 - ABcom * x_0_5_1);
                            QUICKDouble x_0_24_0 = Qtempx * x_0_15_0 + WQtempx * x_0_15_1;
                            QUICKDouble x_0_25_0 = Qtempx * x_0_16_0 + WQtempx * x_0_16_1;
                            QUICKDouble x_0_26_0 = Qtempx * x_0_13_0 + WQtempx * x_0_13_1 +  2 * CDtemp * ( x_0_6_0 - ABcom * x_0_6_1);
                            QUICKDouble x_0_27_0 = Qtempx * x_0_19_0 + WQtempx * x_0_19_1;
                            QUICKDouble x_0_28_0 = Qtempx * x_0_11_0 + WQtempx * x_0_11_1 +  2 * CDtemp * ( x_0_4_0 - ABcom * x_0_4_1);
                            QUICKDouble x_0_29_0 = Qtempx * x_0_18_0 + WQtempx * x_0_18_1;
                            QUICKDouble x_0_30_0 = Qtempy * x_0_15_0 + WQtempy * x_0_15_1 +  2 * CDtemp * ( x_0_5_0 - ABcom * x_0_5_1);
                            QUICKDouble x_0_31_0 = Qtempy * x_0_19_0 + WQtempy * x_0_19_1;
                            QUICKDouble x_0_32_0 = Qtempx * x_0_17_0 + WQtempx * x_0_17_1 +  3 * CDtemp * ( x_0_7_0 - ABcom * x_0_7_1);
                            QUICKDouble x_0_33_0 = Qtempy * x_0_18_0 + WQtempy * x_0_18_1 +  3 * CDtemp * ( x_0_8_0 - ABcom * x_0_8_1);
                            QUICKDouble x_0_34_0 = Qtempz * x_0_19_0 + WQtempz * x_0_19_1 +  3 * CDtemp * ( x_0_9_0 - ABcom * x_0_9_1);
                            
                            LOC2(store, 0,20, STOREDIM, STOREDIM) += x_0_20_0;
                            LOC2(store, 0,21, STOREDIM, STOREDIM) += x_0_21_0;
                            LOC2(store, 0,22, STOREDIM, STOREDIM) += x_0_22_0;
                            LOC2(store, 0,23, STOREDIM, STOREDIM) += x_0_23_0;
                            LOC2(store, 0,24, STOREDIM, STOREDIM) += x_0_24_0;
                            LOC2(store, 0,25, STOREDIM, STOREDIM) += x_0_25_0;
                            LOC2(store, 0,26, STOREDIM, STOREDIM) += x_0_26_0;
                            LOC2(store, 0,27, STOREDIM, STOREDIM) += x_0_27_0;
                            LOC2(store, 0,28, STOREDIM, STOREDIM) += x_0_28_0;
                            LOC2(store, 0,29, STOREDIM, STOREDIM) += x_0_29_0;
                            LOC2(store, 0,30, STOREDIM, STOREDIM) += x_0_30_0;
                            LOC2(store, 0,31, STOREDIM, STOREDIM) += x_0_31_0;
                            LOC2(store, 0,32, STOREDIM, STOREDIM) += x_0_32_0;
                            LOC2(store, 0,33, STOREDIM, STOREDIM) += x_0_33_0;
                            LOC2(store, 0,34, STOREDIM, STOREDIM) += x_0_34_0;
                            
                            if (I+J>=1 && K+L>=4){
                                
                                //SSFS(2, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz, CDtemp, ABcom);
                                
                                QUICKDouble x_0_10_2 = Qtempx * x_0_5_2 + WQtempx * x_0_5_3;
                                QUICKDouble x_0_11_2 = Qtempx * x_0_4_2 + WQtempx * x_0_4_3 + CDtemp * ( x_0_2_2 - ABcom * x_0_2_3);
                                QUICKDouble x_0_12_2 = Qtempx * x_0_8_2 + WQtempx * x_0_8_3;
                                QUICKDouble x_0_13_2 = Qtempx * x_0_6_2 + WQtempx * x_0_6_3 + CDtemp * ( x_0_3_2 - ABcom * x_0_3_3);
                                QUICKDouble x_0_14_2 = Qtempx * x_0_9_2 + WQtempx * x_0_9_3;
                                QUICKDouble x_0_15_2 = Qtempy * x_0_5_2 + WQtempy * x_0_5_3 + CDtemp * ( x_0_3_2 - ABcom * x_0_3_3);
                                QUICKDouble x_0_16_2 = Qtempy * x_0_9_2 + WQtempy * x_0_9_3;
                                QUICKDouble x_0_17_2 = Qtempx * x_0_7_2 + WQtempx * x_0_7_3 + 2 * CDtemp * ( x_0_1_2 - ABcom * x_0_1_3);
                                QUICKDouble x_0_18_2 = Qtempy * x_0_8_2 + WQtempy * x_0_8_3 + 2 * CDtemp * ( x_0_2_2 - ABcom * x_0_2_3);
                                QUICKDouble x_0_19_2 = Qtempz * x_0_9_2 + WQtempz * x_0_9_3 + 2 * CDtemp * ( x_0_3_2 - ABcom * x_0_3_3);
                                
                                //SSGS(1, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz, CDtemp, ABcom);
                                
                                QUICKDouble x_0_20_1 = Qtempx * x_0_12_1 + WQtempx * x_0_12_2 + CDtemp * ( x_0_8_1 - ABcom * x_0_8_2);
                                QUICKDouble x_0_21_1 = Qtempx * x_0_14_1 + WQtempx * x_0_14_2 + CDtemp * ( x_0_9_1 - ABcom * x_0_9_2);
                                QUICKDouble x_0_22_1 = Qtempy * x_0_16_1 + WQtempy * x_0_16_2 + CDtemp * ( x_0_9_1 - ABcom * x_0_9_2);
                                QUICKDouble x_0_23_1 = Qtempx * x_0_10_1 + WQtempx * x_0_10_2 + CDtemp * ( x_0_5_1 - ABcom * x_0_5_2);
                                QUICKDouble x_0_24_1 = Qtempx * x_0_15_1 + WQtempx * x_0_15_2;
                                QUICKDouble x_0_25_1 = Qtempx * x_0_16_1 + WQtempx * x_0_16_2;
                                QUICKDouble x_0_26_1 = Qtempx * x_0_13_1 + WQtempx * x_0_13_2 +  2 * CDtemp * ( x_0_6_1 - ABcom * x_0_6_2);
                                QUICKDouble x_0_27_1 = Qtempx * x_0_19_1 + WQtempx * x_0_19_2;
                                QUICKDouble x_0_28_1 = Qtempx * x_0_11_1 + WQtempx * x_0_11_2 +  2 * CDtemp * ( x_0_4_1 - ABcom * x_0_4_2);
                                QUICKDouble x_0_29_1 = Qtempx * x_0_18_1 + WQtempx * x_0_18_2;
                                QUICKDouble x_0_30_1 = Qtempy * x_0_15_1 + WQtempy * x_0_15_2 +  2 * CDtemp * ( x_0_5_1 - ABcom * x_0_5_2);
                                QUICKDouble x_0_31_1 = Qtempy * x_0_19_1 + WQtempy * x_0_19_2;
                                QUICKDouble x_0_32_1 = Qtempx * x_0_17_1 + WQtempx * x_0_17_2 +  3 * CDtemp * ( x_0_7_1 - ABcom * x_0_7_2);
                                QUICKDouble x_0_33_1 = Qtempy * x_0_18_1 + WQtempy * x_0_18_2 +  3 * CDtemp * ( x_0_8_1 - ABcom * x_0_8_2);
                                QUICKDouble x_0_34_1 = Qtempz * x_0_19_1 + WQtempz * x_0_19_2 +  3 * CDtemp * ( x_0_9_1 - ABcom * x_0_9_2);
                                
                                //PSGS(0, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz, ABCDtemp);
                                
                                QUICKDouble x_1_20_0 = Ptempx * x_0_20_0 + WPtempx * x_0_20_1 +  2 * ABCDtemp *  x_0_12_1;
                                QUICKDouble x_2_20_0 = Ptempy * x_0_20_0 + WPtempy * x_0_20_1 +  2 * ABCDtemp *  x_0_11_1;
                                QUICKDouble x_3_20_0 = Ptempz * x_0_20_0 + WPtempz * x_0_20_1;
                                QUICKDouble x_1_21_0 = Ptempx * x_0_21_0 + WPtempx * x_0_21_1 +  2 * ABCDtemp *  x_0_14_1;
                                QUICKDouble x_2_21_0 = Ptempy * x_0_21_0 + WPtempy * x_0_21_1;
                                QUICKDouble x_3_21_0 = Ptempz * x_0_21_0 + WPtempz * x_0_21_1 +  2 * ABCDtemp *  x_0_13_1;
                                QUICKDouble x_1_22_0 = Ptempx * x_0_22_0 + WPtempx * x_0_22_1;
                                QUICKDouble x_2_22_0 = Ptempy * x_0_22_0 + WPtempy * x_0_22_1 +  2 * ABCDtemp *  x_0_16_1;
                                QUICKDouble x_3_22_0 = Ptempz * x_0_22_0 + WPtempz * x_0_22_1 +  2 * ABCDtemp *  x_0_15_1;
                                QUICKDouble x_1_23_0 = Ptempx * x_0_23_0 + WPtempx * x_0_23_1 +  2 * ABCDtemp *  x_0_10_1;
                                QUICKDouble x_2_23_0 = Ptempy * x_0_23_0 + WPtempy * x_0_23_1 + ABCDtemp *  x_0_13_1;
                                QUICKDouble x_3_23_0 = Ptempz * x_0_23_0 + WPtempz * x_0_23_1 + ABCDtemp *  x_0_11_1;
                                QUICKDouble x_1_24_0 = Ptempx * x_0_24_0 + WPtempx * x_0_24_1 + ABCDtemp *  x_0_15_1;
                                QUICKDouble x_2_24_0 = Ptempy * x_0_24_0 + WPtempy * x_0_24_1 +  2 * ABCDtemp *  x_0_10_1;
                                QUICKDouble x_3_24_0 = Ptempz * x_0_24_0 + WPtempz * x_0_24_1 + ABCDtemp *  x_0_12_1;
                                QUICKDouble x_1_25_0 = Ptempx * x_0_25_0 + WPtempx * x_0_25_1 + ABCDtemp *  x_0_16_1;
                                QUICKDouble x_2_25_0 = Ptempy * x_0_25_0 + WPtempy * x_0_25_1 + ABCDtemp *  x_0_14_1;
                                QUICKDouble x_3_25_0 = Ptempz * x_0_25_0 + WPtempz * x_0_25_1 +  2 * ABCDtemp *  x_0_10_1;
                                QUICKDouble x_1_26_0 = Ptempx * x_0_26_0 + WPtempx * x_0_26_1 +  3 * ABCDtemp *  x_0_13_1;
                                QUICKDouble x_2_26_0 = Ptempy * x_0_26_0 + WPtempy * x_0_26_1;
                                QUICKDouble x_3_26_0 = Ptempz * x_0_26_0 + WPtempz * x_0_26_1 + ABCDtemp *  x_0_17_1;
                                QUICKDouble x_1_27_0 = Ptempx * x_0_27_0 + WPtempx * x_0_27_1 + ABCDtemp *  x_0_19_1;
                                QUICKDouble x_2_27_0 = Ptempy * x_0_27_0 + WPtempy * x_0_27_1;
                                QUICKDouble x_3_27_0 = Ptempz * x_0_27_0 + WPtempz * x_0_27_1 +  3 * ABCDtemp *  x_0_14_1;
                                QUICKDouble x_1_28_0 = Ptempx * x_0_28_0 + WPtempx * x_0_28_1 +  3 * ABCDtemp *  x_0_11_1;
                                QUICKDouble x_2_28_0 = Ptempy * x_0_28_0 + WPtempy * x_0_28_1 + ABCDtemp *  x_0_17_1;
                                QUICKDouble x_3_28_0 = Ptempz * x_0_28_0 + WPtempz * x_0_28_1;
                                QUICKDouble x_1_29_0 = Ptempx * x_0_29_0 + WPtempx * x_0_29_1 + ABCDtemp *  x_0_18_1;
                                QUICKDouble x_2_29_0 = Ptempy * x_0_29_0 + WPtempy * x_0_29_1 +  3 * ABCDtemp *  x_0_12_1;
                                QUICKDouble x_3_29_0 = Ptempz * x_0_29_0 + WPtempz * x_0_29_1;
                                QUICKDouble x_1_30_0 = Ptempx * x_0_30_0 + WPtempx * x_0_30_1;
                                QUICKDouble x_2_30_0 = Ptempy * x_0_30_0 + WPtempy * x_0_30_1 +  3 * ABCDtemp *  x_0_15_1;
                                QUICKDouble x_3_30_0 = Ptempz * x_0_30_0 + WPtempz * x_0_30_1 + ABCDtemp *  x_0_18_1;
                                QUICKDouble x_1_31_0 = Ptempx * x_0_31_0 + WPtempx * x_0_31_1;
                                QUICKDouble x_2_31_0 = Ptempy * x_0_31_0 + WPtempy * x_0_31_1 + ABCDtemp *  x_0_19_1;
                                QUICKDouble x_3_31_0 = Ptempz * x_0_31_0 + WPtempz * x_0_31_1 +  3 * ABCDtemp *  x_0_16_1;
                                QUICKDouble x_1_32_0 = Ptempx * x_0_32_0 + WPtempx * x_0_32_1 +  4 * ABCDtemp *  x_0_17_1;
                                QUICKDouble x_2_32_0 = Ptempy * x_0_32_0 + WPtempy * x_0_32_1;
                                QUICKDouble x_3_32_0 = Ptempz * x_0_32_0 + WPtempz * x_0_32_1;
                                QUICKDouble x_1_33_0 = Ptempx * x_0_33_0 + WPtempx * x_0_33_1;
                                QUICKDouble x_2_33_0 = Ptempy * x_0_33_0 + WPtempy * x_0_33_1 +  4 * ABCDtemp *  x_0_18_1;
                                QUICKDouble x_3_33_0 = Ptempz * x_0_33_0 + WPtempz * x_0_33_1;
                                QUICKDouble x_1_34_0 = Ptempx * x_0_34_0 + WPtempx * x_0_34_1;
                                QUICKDouble x_2_34_0 = Ptempy * x_0_34_0 + WPtempy * x_0_34_1;
                                QUICKDouble x_3_34_0 = Ptempz * x_0_34_0 + WPtempz * x_0_34_1 +  4 * ABCDtemp *  x_0_19_1;
                                
                                LOC2(store, 1,20, STOREDIM, STOREDIM) += x_1_20_0;
                                LOC2(store, 1,21, STOREDIM, STOREDIM) += x_1_21_0;
                                LOC2(store, 1,22, STOREDIM, STOREDIM) += x_1_22_0;
                                LOC2(store, 1,23, STOREDIM, STOREDIM) += x_1_23_0;
                                LOC2(store, 1,24, STOREDIM, STOREDIM) += x_1_24_0;
                                LOC2(store, 1,25, STOREDIM, STOREDIM) += x_1_25_0;
                                LOC2(store, 1,26, STOREDIM, STOREDIM) += x_1_26_0;
                                LOC2(store, 1,27, STOREDIM, STOREDIM) += x_1_27_0;
                                LOC2(store, 1,28, STOREDIM, STOREDIM) += x_1_28_0;
                                LOC2(store, 1,29, STOREDIM, STOREDIM) += x_1_29_0;
                                LOC2(store, 1,30, STOREDIM, STOREDIM) += x_1_30_0;
                                LOC2(store, 1,31, STOREDIM, STOREDIM) += x_1_31_0;
                                LOC2(store, 1,32, STOREDIM, STOREDIM) += x_1_32_0;
                                LOC2(store, 1,33, STOREDIM, STOREDIM) += x_1_33_0;
                                LOC2(store, 1,34, STOREDIM, STOREDIM) += x_1_34_0;
                                LOC2(store, 2,20, STOREDIM, STOREDIM) += x_2_20_0;
                                LOC2(store, 2,21, STOREDIM, STOREDIM) += x_2_21_0;
                                LOC2(store, 2,22, STOREDIM, STOREDIM) += x_2_22_0;
                                LOC2(store, 2,23, STOREDIM, STOREDIM) += x_2_23_0;
                                LOC2(store, 2,24, STOREDIM, STOREDIM) += x_2_24_0;
                                LOC2(store, 2,25, STOREDIM, STOREDIM) += x_2_25_0;
                                LOC2(store, 2,26, STOREDIM, STOREDIM) += x_2_26_0;
                                LOC2(store, 2,27, STOREDIM, STOREDIM) += x_2_27_0;
                                LOC2(store, 2,28, STOREDIM, STOREDIM) += x_2_28_0;
                                LOC2(store, 2,29, STOREDIM, STOREDIM) += x_2_29_0;
                                LOC2(store, 2,30, STOREDIM, STOREDIM) += x_2_30_0;
                                LOC2(store, 2,31, STOREDIM, STOREDIM) += x_2_31_0;
                                LOC2(store, 2,32, STOREDIM, STOREDIM) += x_2_32_0;
                                LOC2(store, 2,33, STOREDIM, STOREDIM) += x_2_33_0;
                                LOC2(store, 2,34, STOREDIM, STOREDIM) += x_2_34_0;
                                LOC2(store, 3,20, STOREDIM, STOREDIM) += x_3_20_0;
                                LOC2(store, 3,21, STOREDIM, STOREDIM) += x_3_21_0;
                                LOC2(store, 3,22, STOREDIM, STOREDIM) += x_3_22_0;
                                LOC2(store, 3,23, STOREDIM, STOREDIM) += x_3_23_0;
                                LOC2(store, 3,24, STOREDIM, STOREDIM) += x_3_24_0;
                                LOC2(store, 3,25, STOREDIM, STOREDIM) += x_3_25_0;
                                LOC2(store, 3,26, STOREDIM, STOREDIM) += x_3_26_0;
                                LOC2(store, 3,27, STOREDIM, STOREDIM) += x_3_27_0;
                                LOC2(store, 3,28, STOREDIM, STOREDIM) += x_3_28_0;
                                LOC2(store, 3,29, STOREDIM, STOREDIM) += x_3_29_0;
                                LOC2(store, 3,30, STOREDIM, STOREDIM) += x_3_30_0;
                                LOC2(store, 3,31, STOREDIM, STOREDIM) += x_3_31_0;
                                LOC2(store, 3,32, STOREDIM, STOREDIM) += x_3_32_0;
                                LOC2(store, 3,33, STOREDIM, STOREDIM) += x_3_33_0;
                                LOC2(store, 3,34, STOREDIM, STOREDIM) += x_3_34_0;
                                if (I+J>=2 && K+L>=4){ 
                                    //SSPS(5, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);
                                    QUICKDouble x_0_1_5 = Qtempx * VY( 0, 0, 5) + WQtempx * VY( 0, 0, 6);
                                    QUICKDouble x_0_2_5 = Qtempy * VY( 0, 0, 5) + WQtempy * VY( 0, 0, 6);
                                    QUICKDouble x_0_3_5 = Qtempz * VY( 0, 0, 5) + WQtempz * VY( 0, 0, 6);
                                    
                                    //SSDS(4, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz, CDtemp, ABcom);
                                    
                                    QUICKDouble x_0_4_4 = Qtempx * x_0_2_4 + WQtempx * x_0_2_5;
                                    QUICKDouble x_0_5_4 = Qtempy * x_0_3_4 + WQtempy * x_0_3_5;
                                    QUICKDouble x_0_6_4 = Qtempx * x_0_3_4 + WQtempx * x_0_3_5;
                                    
                                    QUICKDouble x_0_7_4 = Qtempx * x_0_1_4 + WQtempx * x_0_1_5+ CDtemp*(VY( 0, 0, 4) - ABcom * VY( 0, 0, 5));
                                    QUICKDouble x_0_8_4 = Qtempy * x_0_2_4 + WQtempy * x_0_2_5+ CDtemp*(VY( 0, 0, 4) - ABcom * VY( 0, 0, 5));
                                    QUICKDouble x_0_9_4 = Qtempz * x_0_3_4 + WQtempz * x_0_3_5+ CDtemp*(VY( 0, 0, 4) - ABcom * VY( 0, 0, 5));
                                    
                                    //SSFS(3, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz, CDtemp, ABcom);
                                    
                                    QUICKDouble x_0_10_3 = Qtempx * x_0_5_3 + WQtempx * x_0_5_4;
                                    QUICKDouble x_0_11_3 = Qtempx * x_0_4_3 + WQtempx * x_0_4_4 + CDtemp * ( x_0_2_3 - ABcom * x_0_2_4);
                                    QUICKDouble x_0_12_3 = Qtempx * x_0_8_3 + WQtempx * x_0_8_4;
                                    QUICKDouble x_0_13_3 = Qtempx * x_0_6_3 + WQtempx * x_0_6_4 + CDtemp * ( x_0_3_3 - ABcom * x_0_3_4);
                                    QUICKDouble x_0_14_3 = Qtempx * x_0_9_3 + WQtempx * x_0_9_4;
                                    QUICKDouble x_0_15_3 = Qtempy * x_0_5_3 + WQtempy * x_0_5_4 + CDtemp * ( x_0_3_3 - ABcom * x_0_3_4);
                                    QUICKDouble x_0_16_3 = Qtempy * x_0_9_3 + WQtempy * x_0_9_4;
                                    QUICKDouble x_0_17_3 = Qtempx * x_0_7_3 + WQtempx * x_0_7_4 + 2 * CDtemp * ( x_0_1_3 - ABcom * x_0_1_4);
                                    QUICKDouble x_0_18_3 = Qtempy * x_0_8_3 + WQtempy * x_0_8_4 + 2 * CDtemp * ( x_0_2_3 - ABcom * x_0_2_4);
                                    QUICKDouble x_0_19_3 = Qtempz * x_0_9_3 + WQtempz * x_0_9_4 + 2 * CDtemp * ( x_0_3_3 - ABcom * x_0_3_4);
                                    
                                    //SSGS(2, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz, CDtemp, ABcom);
                                    
                                    QUICKDouble x_0_20_2 = Qtempx * x_0_12_2 + WQtempx * x_0_12_3 + CDtemp * ( x_0_8_2 - ABcom * x_0_8_3);
                                    QUICKDouble x_0_21_2 = Qtempx * x_0_14_2 + WQtempx * x_0_14_3 + CDtemp * ( x_0_9_2 - ABcom * x_0_9_3);
                                    QUICKDouble x_0_22_2 = Qtempy * x_0_16_2 + WQtempy * x_0_16_3 + CDtemp * ( x_0_9_2 - ABcom * x_0_9_3);
                                    QUICKDouble x_0_23_2 = Qtempx * x_0_10_2 + WQtempx * x_0_10_3 + CDtemp * ( x_0_5_2 - ABcom * x_0_5_3);
                                    QUICKDouble x_0_24_2 = Qtempx * x_0_15_2 + WQtempx * x_0_15_3;
                                    QUICKDouble x_0_25_2 = Qtempx * x_0_16_2 + WQtempx * x_0_16_3;
                                    QUICKDouble x_0_26_2 = Qtempx * x_0_13_2 + WQtempx * x_0_13_3 +  2 * CDtemp * ( x_0_6_2 - ABcom * x_0_6_3);
                                    QUICKDouble x_0_27_2 = Qtempx * x_0_19_2 + WQtempx * x_0_19_3;
                                    QUICKDouble x_0_28_2 = Qtempx * x_0_11_2 + WQtempx * x_0_11_3 +  2 * CDtemp * ( x_0_4_2 - ABcom * x_0_4_3);
                                    QUICKDouble x_0_29_2 = Qtempx * x_0_18_2 + WQtempx * x_0_18_3;
                                    QUICKDouble x_0_30_2 = Qtempy * x_0_15_2 + WQtempy * x_0_15_3 +  2 * CDtemp * ( x_0_5_2 - ABcom * x_0_5_3);
                                    QUICKDouble x_0_31_2 = Qtempy * x_0_19_2 + WQtempy * x_0_19_3;
                                    QUICKDouble x_0_32_2 = Qtempx * x_0_17_2 + WQtempx * x_0_17_3 +  3 * CDtemp * ( x_0_7_2 - ABcom * x_0_7_3);
                                    QUICKDouble x_0_33_2 = Qtempy * x_0_18_2 + WQtempy * x_0_18_3 +  3 * CDtemp * ( x_0_8_2 - ABcom * x_0_8_3);
                                    QUICKDouble x_0_34_2 = Qtempz * x_0_19_2 + WQtempz * x_0_19_3 +  3 * CDtemp * ( x_0_9_2 - ABcom * x_0_9_3);
                                    
                                    
                                    //PSFS(1, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz, ABCDtemp);
                                    
                                    QUICKDouble x_1_10_1 = Ptempx * x_0_10_1 + WPtempx * x_0_10_2 + ABCDtemp *  x_0_5_2;
                                    QUICKDouble x_2_10_1 = Ptempy * x_0_10_1 + WPtempy * x_0_10_2 + ABCDtemp *  x_0_6_2;
                                    QUICKDouble x_3_10_1 = Ptempz * x_0_10_1 + WPtempz * x_0_10_2 + ABCDtemp *  x_0_4_2;
                                    QUICKDouble x_1_11_1 = Ptempx * x_0_11_1 + WPtempx * x_0_11_2 +  2 * ABCDtemp *  x_0_4_2;
                                    QUICKDouble x_2_11_1 = Ptempy * x_0_11_1 + WPtempy * x_0_11_2 + ABCDtemp *  x_0_7_2;
                                    QUICKDouble x_3_11_1 = Ptempz * x_0_11_1 + WPtempz * x_0_11_2;
                                    QUICKDouble x_1_12_1 = Ptempx * x_0_12_1 + WPtempx * x_0_12_2 + ABCDtemp *  x_0_8_2;
                                    QUICKDouble x_2_12_1 = Ptempy * x_0_12_1 + WPtempy * x_0_12_2 +  2 * ABCDtemp *  x_0_4_2;
                                    QUICKDouble x_3_12_1 = Ptempz * x_0_12_1 + WPtempz * x_0_12_2;
                                    QUICKDouble x_1_13_1 = Ptempx * x_0_13_1 + WPtempx * x_0_13_2 +  2 * ABCDtemp *  x_0_6_2;
                                    QUICKDouble x_2_13_1 = Ptempy * x_0_13_1 + WPtempy * x_0_13_2;
                                    QUICKDouble x_3_13_1 = Ptempz * x_0_13_1 + WPtempz * x_0_13_2 + ABCDtemp *  x_0_7_2;
                                    QUICKDouble x_1_14_1 = Ptempx * x_0_14_1 + WPtempx * x_0_14_2 + ABCDtemp *  x_0_9_2;
                                    QUICKDouble x_2_14_1 = Ptempy * x_0_14_1 + WPtempy * x_0_14_2;
                                    QUICKDouble x_3_14_1 = Ptempz * x_0_14_1 + WPtempz * x_0_14_2 +  2 * ABCDtemp *  x_0_6_2;
                                    QUICKDouble x_1_15_1 = Ptempx * x_0_15_1 + WPtempx * x_0_15_2;
                                    QUICKDouble x_2_15_1 = Ptempy * x_0_15_1 + WPtempy * x_0_15_2 +  2 * ABCDtemp *  x_0_5_2;
                                    QUICKDouble x_3_15_1 = Ptempz * x_0_15_1 + WPtempz * x_0_15_2 + ABCDtemp *  x_0_8_2;
                                    QUICKDouble x_1_16_1 = Ptempx * x_0_16_1 + WPtempx * x_0_16_2;
                                    QUICKDouble x_2_16_1 = Ptempy * x_0_16_1 + WPtempy * x_0_16_2 + ABCDtemp *  x_0_9_2;
                                    QUICKDouble x_3_16_1 = Ptempz * x_0_16_1 + WPtempz * x_0_16_2 +  2 * ABCDtemp *  x_0_5_2;
                                    QUICKDouble x_1_17_1 = Ptempx * x_0_17_1 + WPtempx * x_0_17_2 +  3 * ABCDtemp *  x_0_7_2;
                                    QUICKDouble x_2_17_1 = Ptempy * x_0_17_1 + WPtempy * x_0_17_2;
                                    QUICKDouble x_3_17_1 = Ptempz * x_0_17_1 + WPtempz * x_0_17_2;
                                    QUICKDouble x_1_18_1 = Ptempx * x_0_18_1 + WPtempx * x_0_18_2;
                                    QUICKDouble x_2_18_1 = Ptempy * x_0_18_1 + WPtempy * x_0_18_2 +  3 * ABCDtemp *  x_0_8_2;
                                    QUICKDouble x_3_18_1 = Ptempz * x_0_18_1 + WPtempz * x_0_18_2;
                                    QUICKDouble x_1_19_1 = Ptempx * x_0_19_1 + WPtempx * x_0_19_2;
                                    QUICKDouble x_2_19_1 = Ptempy * x_0_19_1 + WPtempy * x_0_19_2;
                                    QUICKDouble x_3_19_1 = Ptempz * x_0_19_1 + WPtempz * x_0_19_2 +  3 * ABCDtemp *  x_0_9_2;
                                    
                                    //PSGS(1, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz, ABCDtemp);
                                    
                                    QUICKDouble x_1_20_1 = Ptempx * x_0_20_1 + WPtempx * x_0_20_2 +  2 * ABCDtemp *  x_0_12_2;
                                    QUICKDouble x_2_20_1 = Ptempy * x_0_20_1 + WPtempy * x_0_20_2 +  2 * ABCDtemp *  x_0_11_2;
                                    QUICKDouble x_3_20_1 = Ptempz * x_0_20_1 + WPtempz * x_0_20_2;
                                    QUICKDouble x_1_21_1 = Ptempx * x_0_21_1 + WPtempx * x_0_21_2 +  2 * ABCDtemp *  x_0_14_2;
                                    QUICKDouble x_2_21_1 = Ptempy * x_0_21_1 + WPtempy * x_0_21_2;
                                    QUICKDouble x_3_21_1 = Ptempz * x_0_21_1 + WPtempz * x_0_21_2 +  2 * ABCDtemp *  x_0_13_2;
                                    QUICKDouble x_1_22_1 = Ptempx * x_0_22_1 + WPtempx * x_0_22_2;
                                    QUICKDouble x_2_22_1 = Ptempy * x_0_22_1 + WPtempy * x_0_22_2 +  2 * ABCDtemp *  x_0_16_2;
                                    QUICKDouble x_3_22_1 = Ptempz * x_0_22_1 + WPtempz * x_0_22_2 +  2 * ABCDtemp *  x_0_15_2;
                                    QUICKDouble x_1_23_1 = Ptempx * x_0_23_1 + WPtempx * x_0_23_2 +  2 * ABCDtemp *  x_0_10_2;
                                    QUICKDouble x_2_23_1 = Ptempy * x_0_23_1 + WPtempy * x_0_23_2 + ABCDtemp *  x_0_13_2;
                                    QUICKDouble x_3_23_1 = Ptempz * x_0_23_1 + WPtempz * x_0_23_2 + ABCDtemp *  x_0_11_2;
                                    QUICKDouble x_1_24_1 = Ptempx * x_0_24_1 + WPtempx * x_0_24_2 + ABCDtemp *  x_0_15_2;
                                    QUICKDouble x_2_24_1 = Ptempy * x_0_24_1 + WPtempy * x_0_24_2 +  2 * ABCDtemp *  x_0_10_2;
                                    QUICKDouble x_3_24_1 = Ptempz * x_0_24_1 + WPtempz * x_0_24_2 + ABCDtemp *  x_0_12_2;
                                    QUICKDouble x_1_25_1 = Ptempx * x_0_25_1 + WPtempx * x_0_25_2 + ABCDtemp *  x_0_16_2;
                                    QUICKDouble x_2_25_1 = Ptempy * x_0_25_1 + WPtempy * x_0_25_2 + ABCDtemp *  x_0_14_2;
                                    QUICKDouble x_3_25_1 = Ptempz * x_0_25_1 + WPtempz * x_0_25_2 +  2 * ABCDtemp *  x_0_10_2;
                                    QUICKDouble x_1_26_1 = Ptempx * x_0_26_1 + WPtempx * x_0_26_2 +  3 * ABCDtemp *  x_0_13_2;
                                    QUICKDouble x_2_26_1 = Ptempy * x_0_26_1 + WPtempy * x_0_26_2;
                                    QUICKDouble x_3_26_1 = Ptempz * x_0_26_1 + WPtempz * x_0_26_2 + ABCDtemp *  x_0_17_2;
                                    QUICKDouble x_1_27_1 = Ptempx * x_0_27_1 + WPtempx * x_0_27_2 + ABCDtemp *  x_0_19_2;
                                    QUICKDouble x_2_27_1 = Ptempy * x_0_27_1 + WPtempy * x_0_27_2;
                                    QUICKDouble x_3_27_1 = Ptempz * x_0_27_1 + WPtempz * x_0_27_2 +  3 * ABCDtemp *  x_0_14_2;
                                    QUICKDouble x_1_28_1 = Ptempx * x_0_28_1 + WPtempx * x_0_28_2 +  3 * ABCDtemp *  x_0_11_2;
                                    QUICKDouble x_2_28_1 = Ptempy * x_0_28_1 + WPtempy * x_0_28_2 + ABCDtemp *  x_0_17_2;
                                    QUICKDouble x_3_28_1 = Ptempz * x_0_28_1 + WPtempz * x_0_28_2;
                                    QUICKDouble x_1_29_1 = Ptempx * x_0_29_1 + WPtempx * x_0_29_2 + ABCDtemp *  x_0_18_2;
                                    QUICKDouble x_2_29_1 = Ptempy * x_0_29_1 + WPtempy * x_0_29_2 +  3 * ABCDtemp *  x_0_12_2;
                                    QUICKDouble x_3_29_1 = Ptempz * x_0_29_1 + WPtempz * x_0_29_2;
                                    QUICKDouble x_1_30_1 = Ptempx * x_0_30_1 + WPtempx * x_0_30_2;
                                    QUICKDouble x_2_30_1 = Ptempy * x_0_30_1 + WPtempy * x_0_30_2 +  3 * ABCDtemp *  x_0_15_2;
                                    QUICKDouble x_3_30_1 = Ptempz * x_0_30_1 + WPtempz * x_0_30_2 + ABCDtemp *  x_0_18_2;
                                    QUICKDouble x_1_31_1 = Ptempx * x_0_31_1 + WPtempx * x_0_31_2;
                                    QUICKDouble x_2_31_1 = Ptempy * x_0_31_1 + WPtempy * x_0_31_2 + ABCDtemp *  x_0_19_2;
                                    QUICKDouble x_3_31_1 = Ptempz * x_0_31_1 + WPtempz * x_0_31_2 +  3 * ABCDtemp *  x_0_16_2;
                                    QUICKDouble x_1_32_1 = Ptempx * x_0_32_1 + WPtempx * x_0_32_2 +  4 * ABCDtemp *  x_0_17_2;
                                    QUICKDouble x_2_32_1 = Ptempy * x_0_32_1 + WPtempy * x_0_32_2;
                                    QUICKDouble x_3_32_1 = Ptempz * x_0_32_1 + WPtempz * x_0_32_2;
                                    QUICKDouble x_1_33_1 = Ptempx * x_0_33_1 + WPtempx * x_0_33_2;
                                    QUICKDouble x_2_33_1 = Ptempy * x_0_33_1 + WPtempy * x_0_33_2 +  4 * ABCDtemp *  x_0_18_2;
                                    QUICKDouble x_3_33_1 = Ptempz * x_0_33_1 + WPtempz * x_0_33_2;
                                    QUICKDouble x_1_34_1 = Ptempx * x_0_34_1 + WPtempx * x_0_34_2;
                                    QUICKDouble x_2_34_1 = Ptempy * x_0_34_1 + WPtempy * x_0_34_2;
                                    QUICKDouble x_3_34_1 = Ptempz * x_0_34_1 + WPtempz * x_0_34_2 +  4 * ABCDtemp *  x_0_19_2;
                                    
                                    //SSPS(6, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);
                                    QUICKDouble x_0_1_6 = Qtempx * VY( 0, 0, 6) + WQtempx * VY( 0, 0, 7);
                                    QUICKDouble x_0_2_6 = Qtempy * VY( 0, 0, 6) + WQtempy * VY( 0, 0, 7);
                                    QUICKDouble x_0_3_6 = Qtempz * VY( 0, 0, 6) + WQtempz * VY( 0, 0, 7);
                                    
                                    //SSDS(5, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz, CDtemp, ABcom);
                                    
                                    QUICKDouble x_0_4_5 = Qtempx * x_0_2_5 + WQtempx * x_0_2_6;
                                    QUICKDouble x_0_5_5 = Qtempy * x_0_3_5 + WQtempy * x_0_3_6;
                                    QUICKDouble x_0_6_5 = Qtempx * x_0_3_5 + WQtempx * x_0_3_6;
                                    
                                    QUICKDouble x_0_7_5 = Qtempx * x_0_1_5 + WQtempx * x_0_1_6+ CDtemp*(VY( 0, 0, 5) - ABcom * VY( 0, 0, 6));
                                    QUICKDouble x_0_8_5 = Qtempy * x_0_2_5 + WQtempy * x_0_2_6+ CDtemp*(VY( 0, 0, 5) - ABcom * VY( 0, 0, 6));
                                    QUICKDouble x_0_9_5 = Qtempz * x_0_3_5 + WQtempz * x_0_3_6+ CDtemp*(VY( 0, 0, 5) - ABcom * VY( 0, 0, 6));
                                    
                                    //SSFS(4, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz, CDtemp, ABcom);
                                    
                                    QUICKDouble x_0_10_4 = Qtempx * x_0_5_4 + WQtempx * x_0_5_5;
                                    QUICKDouble x_0_11_4 = Qtempx * x_0_4_4 + WQtempx * x_0_4_5 + CDtemp * ( x_0_2_4 - ABcom * x_0_2_5);
                                    QUICKDouble x_0_12_4 = Qtempx * x_0_8_4 + WQtempx * x_0_8_5;
                                    QUICKDouble x_0_13_4 = Qtempx * x_0_6_4 + WQtempx * x_0_6_5 + CDtemp * ( x_0_3_4 - ABcom * x_0_3_5);
                                    QUICKDouble x_0_14_4 = Qtempx * x_0_9_4 + WQtempx * x_0_9_5;
                                    QUICKDouble x_0_15_4 = Qtempy * x_0_5_4 + WQtempy * x_0_5_5 + CDtemp * ( x_0_3_4 - ABcom * x_0_3_5);
                                    QUICKDouble x_0_16_4 = Qtempy * x_0_9_4 + WQtempy * x_0_9_5;
                                    QUICKDouble x_0_17_4 = Qtempx * x_0_7_4 + WQtempx * x_0_7_5 + 2 * CDtemp * ( x_0_1_4 - ABcom * x_0_1_5);
                                    QUICKDouble x_0_18_4 = Qtempy * x_0_8_4 + WQtempy * x_0_8_5 + 2 * CDtemp * ( x_0_2_4 - ABcom * x_0_2_5);
                                    QUICKDouble x_0_19_4 = Qtempz * x_0_9_4 + WQtempz * x_0_9_5 + 2 * CDtemp * ( x_0_3_4 - ABcom * x_0_3_5);
                                    
                                    //SSGS(3, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz, CDtemp, ABcom);
                                    
                                    QUICKDouble x_0_20_3 = Qtempx * x_0_12_3 + WQtempx * x_0_12_4 + CDtemp * ( x_0_8_3 - ABcom * x_0_8_4);
                                    QUICKDouble x_0_21_3 = Qtempx * x_0_14_3 + WQtempx * x_0_14_4 + CDtemp * ( x_0_9_3 - ABcom * x_0_9_4);
                                    QUICKDouble x_0_22_3 = Qtempy * x_0_16_3 + WQtempy * x_0_16_4 + CDtemp * ( x_0_9_3 - ABcom * x_0_9_4);
                                    QUICKDouble x_0_23_3 = Qtempx * x_0_10_3 + WQtempx * x_0_10_4 + CDtemp * ( x_0_5_3 - ABcom * x_0_5_4);
                                    QUICKDouble x_0_24_3 = Qtempx * x_0_15_3 + WQtempx * x_0_15_4;
                                    QUICKDouble x_0_25_3 = Qtempx * x_0_16_3 + WQtempx * x_0_16_4;
                                    QUICKDouble x_0_26_3 = Qtempx * x_0_13_3 + WQtempx * x_0_13_4 +  2 * CDtemp * ( x_0_6_3 - ABcom * x_0_6_4);
                                    QUICKDouble x_0_27_3 = Qtempx * x_0_19_3 + WQtempx * x_0_19_4;
                                    QUICKDouble x_0_28_3 = Qtempx * x_0_11_3 + WQtempx * x_0_11_4 +  2 * CDtemp * ( x_0_4_3 - ABcom * x_0_4_4);
                                    QUICKDouble x_0_29_3 = Qtempx * x_0_18_3 + WQtempx * x_0_18_4;
                                    QUICKDouble x_0_30_3 = Qtempy * x_0_15_3 + WQtempy * x_0_15_4 +  2 * CDtemp * ( x_0_5_3 - ABcom * x_0_5_4);
                                    QUICKDouble x_0_31_3 = Qtempy * x_0_19_3 + WQtempy * x_0_19_4;
                                    QUICKDouble x_0_32_3 = Qtempx * x_0_17_3 + WQtempx * x_0_17_4 +  3 * CDtemp * ( x_0_7_3 - ABcom * x_0_7_4);
                                    QUICKDouble x_0_33_3 = Qtempy * x_0_18_3 + WQtempy * x_0_18_4 +  3 * CDtemp * ( x_0_8_3 - ABcom * x_0_8_4);
                                    QUICKDouble x_0_34_3 = Qtempz * x_0_19_3 + WQtempz * x_0_19_4 +  3 * CDtemp * ( x_0_9_3 - ABcom * x_0_9_4);
                                    
                                    //PSGS(2, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz, ABCDtemp);
                                    
                                    QUICKDouble x_1_20_2 = Ptempx * x_0_20_2 + WPtempx * x_0_20_3 +  2 * ABCDtemp *  x_0_12_3;
                                    QUICKDouble x_2_20_2 = Ptempy * x_0_20_2 + WPtempy * x_0_20_3 +  2 * ABCDtemp *  x_0_11_3;
                                    QUICKDouble x_3_20_2 = Ptempz * x_0_20_2 + WPtempz * x_0_20_3;
                                    QUICKDouble x_1_21_2 = Ptempx * x_0_21_2 + WPtempx * x_0_21_3 +  2 * ABCDtemp *  x_0_14_3;
                                    QUICKDouble x_2_21_2 = Ptempy * x_0_21_2 + WPtempy * x_0_21_3;
                                    QUICKDouble x_3_21_2 = Ptempz * x_0_21_2 + WPtempz * x_0_21_3 +  2 * ABCDtemp *  x_0_13_3;
                                    QUICKDouble x_1_22_2 = Ptempx * x_0_22_2 + WPtempx * x_0_22_3;
                                    QUICKDouble x_2_22_2 = Ptempy * x_0_22_2 + WPtempy * x_0_22_3 +  2 * ABCDtemp *  x_0_16_3;
                                    QUICKDouble x_3_22_2 = Ptempz * x_0_22_2 + WPtempz * x_0_22_3 +  2 * ABCDtemp *  x_0_15_3;
                                    QUICKDouble x_1_23_2 = Ptempx * x_0_23_2 + WPtempx * x_0_23_3 +  2 * ABCDtemp *  x_0_10_3;
                                    QUICKDouble x_2_23_2 = Ptempy * x_0_23_2 + WPtempy * x_0_23_3 + ABCDtemp *  x_0_13_3;
                                    QUICKDouble x_3_23_2 = Ptempz * x_0_23_2 + WPtempz * x_0_23_3 + ABCDtemp *  x_0_11_3;
                                    QUICKDouble x_1_24_2 = Ptempx * x_0_24_2 + WPtempx * x_0_24_3 + ABCDtemp *  x_0_15_3;
                                    QUICKDouble x_2_24_2 = Ptempy * x_0_24_2 + WPtempy * x_0_24_3 +  2 * ABCDtemp *  x_0_10_3;
                                    QUICKDouble x_3_24_2 = Ptempz * x_0_24_2 + WPtempz * x_0_24_3 + ABCDtemp *  x_0_12_3;
                                    QUICKDouble x_1_25_2 = Ptempx * x_0_25_2 + WPtempx * x_0_25_3 + ABCDtemp *  x_0_16_3;
                                    QUICKDouble x_2_25_2 = Ptempy * x_0_25_2 + WPtempy * x_0_25_3 + ABCDtemp *  x_0_14_3;
                                    QUICKDouble x_3_25_2 = Ptempz * x_0_25_2 + WPtempz * x_0_25_3 +  2 * ABCDtemp *  x_0_10_3;
                                    QUICKDouble x_1_26_2 = Ptempx * x_0_26_2 + WPtempx * x_0_26_3 +  3 * ABCDtemp *  x_0_13_3;
                                    QUICKDouble x_2_26_2 = Ptempy * x_0_26_2 + WPtempy * x_0_26_3;
                                    QUICKDouble x_3_26_2 = Ptempz * x_0_26_2 + WPtempz * x_0_26_3 + ABCDtemp *  x_0_17_3;
                                    QUICKDouble x_1_27_2 = Ptempx * x_0_27_2 + WPtempx * x_0_27_3 + ABCDtemp *  x_0_19_3;
                                    QUICKDouble x_2_27_2 = Ptempy * x_0_27_2 + WPtempy * x_0_27_3;
                                    QUICKDouble x_3_27_2 = Ptempz * x_0_27_2 + WPtempz * x_0_27_3 +  3 * ABCDtemp *  x_0_14_3;
                                    QUICKDouble x_1_28_2 = Ptempx * x_0_28_2 + WPtempx * x_0_28_3 +  3 * ABCDtemp *  x_0_11_3;
                                    QUICKDouble x_2_28_2 = Ptempy * x_0_28_2 + WPtempy * x_0_28_3 + ABCDtemp *  x_0_17_3;
                                    QUICKDouble x_3_28_2 = Ptempz * x_0_28_2 + WPtempz * x_0_28_3;
                                    QUICKDouble x_1_29_2 = Ptempx * x_0_29_2 + WPtempx * x_0_29_3 + ABCDtemp *  x_0_18_3;
                                    QUICKDouble x_2_29_2 = Ptempy * x_0_29_2 + WPtempy * x_0_29_3 +  3 * ABCDtemp *  x_0_12_3;
                                    QUICKDouble x_3_29_2 = Ptempz * x_0_29_2 + WPtempz * x_0_29_3;
                                    QUICKDouble x_1_30_2 = Ptempx * x_0_30_2 + WPtempx * x_0_30_3;
                                    QUICKDouble x_2_30_2 = Ptempy * x_0_30_2 + WPtempy * x_0_30_3 +  3 * ABCDtemp *  x_0_15_3;
                                    QUICKDouble x_3_30_2 = Ptempz * x_0_30_2 + WPtempz * x_0_30_3 + ABCDtemp *  x_0_18_3;
                                    QUICKDouble x_1_31_2 = Ptempx * x_0_31_2 + WPtempx * x_0_31_3;
                                    QUICKDouble x_2_31_2 = Ptempy * x_0_31_2 + WPtempy * x_0_31_3 + ABCDtemp *  x_0_19_3;
                                    QUICKDouble x_3_31_2 = Ptempz * x_0_31_2 + WPtempz * x_0_31_3 +  3 * ABCDtemp *  x_0_16_3;
                                    QUICKDouble x_1_32_2 = Ptempx * x_0_32_2 + WPtempx * x_0_32_3 +  4 * ABCDtemp *  x_0_17_3;
                                    QUICKDouble x_2_32_2 = Ptempy * x_0_32_2 + WPtempy * x_0_32_3;
                                    QUICKDouble x_3_32_2 = Ptempz * x_0_32_2 + WPtempz * x_0_32_3;
                                    QUICKDouble x_1_33_2 = Ptempx * x_0_33_2 + WPtempx * x_0_33_3;
                                    QUICKDouble x_2_33_2 = Ptempy * x_0_33_2 + WPtempy * x_0_33_3 +  4 * ABCDtemp *  x_0_18_3;
                                    QUICKDouble x_3_33_2 = Ptempz * x_0_33_2 + WPtempz * x_0_33_3;
                                    QUICKDouble x_1_34_2 = Ptempx * x_0_34_2 + WPtempx * x_0_34_3;
                                    QUICKDouble x_2_34_2 = Ptempy * x_0_34_2 + WPtempy * x_0_34_3;
                                    QUICKDouble x_3_34_2 = Ptempz * x_0_34_2 + WPtempz * x_0_34_3 +  4 * ABCDtemp *  x_0_19_3;
                                    
                                    
                                    //DSGS(0, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz, ABCDtemp, ABtemp, CDcom);
                                    
                                    QUICKDouble x_4_20_0 = Ptempx * x_2_20_0 + WPtempx * x_2_20_1 +  2 * ABCDtemp * x_2_12_1;
                                    QUICKDouble x_4_21_0 = Ptempx * x_2_21_0 + WPtempx * x_2_21_1 +  2 * ABCDtemp * x_2_14_1;
                                    QUICKDouble x_4_22_0 = Ptempx * x_2_22_0 + WPtempx * x_2_22_1;
                                    QUICKDouble x_4_23_0 = Ptempx * x_2_23_0 + WPtempx * x_2_23_1 +  2 * ABCDtemp * x_2_10_1;
                                    QUICKDouble x_4_24_0 = Ptempx * x_2_24_0 + WPtempx * x_2_24_1 + ABCDtemp * x_2_15_1;
                                    QUICKDouble x_4_25_0 = Ptempx * x_2_25_0 + WPtempx * x_2_25_1 + ABCDtemp * x_2_16_1;
                                    QUICKDouble x_4_26_0 = Ptempx * x_2_26_0 + WPtempx * x_2_26_1 +  3 * ABCDtemp * x_2_13_1;
                                    QUICKDouble x_4_27_0 = Ptempx * x_2_27_0 + WPtempx * x_2_27_1 + ABCDtemp * x_2_19_1;
                                    QUICKDouble x_4_28_0 = Ptempx * x_2_28_0 + WPtempx * x_2_28_1 +  3 * ABCDtemp * x_2_11_1;
                                    QUICKDouble x_4_29_0 = Ptempx * x_2_29_0 + WPtempx * x_2_29_1 + ABCDtemp * x_2_18_1;
                                    QUICKDouble x_4_30_0 = Ptempx * x_2_30_0 + WPtempx * x_2_30_1;
                                    QUICKDouble x_4_31_0 = Ptempx * x_2_31_0 + WPtempx * x_2_31_1;
                                    QUICKDouble x_4_32_0 = Ptempx * x_2_32_0 + WPtempx * x_2_32_1 +  4 * ABCDtemp * x_2_17_1;
                                    QUICKDouble x_4_33_0 = Ptempx * x_2_33_0 + WPtempx * x_2_33_1;
                                    QUICKDouble x_4_34_0 = Ptempx * x_2_34_0 + WPtempx * x_2_34_1;
                                    QUICKDouble x_5_20_0 = Ptempy * x_3_20_0 + WPtempy * x_3_20_1 +  2 * ABCDtemp * x_3_11_1;
                                    QUICKDouble x_5_21_0 = Ptempy * x_3_21_0 + WPtempy * x_3_21_1;
                                    QUICKDouble x_5_22_0 = Ptempy * x_3_22_0 + WPtempy * x_3_22_1 +  2 * ABCDtemp * x_3_16_1;
                                    QUICKDouble x_5_23_0 = Ptempy * x_3_23_0 + WPtempy * x_3_23_1 + ABCDtemp * x_3_13_1;
                                    QUICKDouble x_5_24_0 = Ptempy * x_3_24_0 + WPtempy * x_3_24_1 +  2 * ABCDtemp * x_3_10_1;
                                    QUICKDouble x_5_25_0 = Ptempy * x_3_25_0 + WPtempy * x_3_25_1 + ABCDtemp * x_3_14_1;
                                    QUICKDouble x_5_26_0 = Ptempy * x_3_26_0 + WPtempy * x_3_26_1;
                                    QUICKDouble x_5_27_0 = Ptempy * x_3_27_0 + WPtempy * x_3_27_1;
                                    QUICKDouble x_5_28_0 = Ptempy * x_3_28_0 + WPtempy * x_3_28_1 + ABCDtemp * x_3_17_1;
                                    QUICKDouble x_5_29_0 = Ptempy * x_3_29_0 + WPtempy * x_3_29_1 +  3 * ABCDtemp * x_3_12_1;
                                    QUICKDouble x_5_30_0 = Ptempy * x_3_30_0 + WPtempy * x_3_30_1 +  3 * ABCDtemp * x_3_15_1;
                                    QUICKDouble x_5_31_0 = Ptempy * x_3_31_0 + WPtempy * x_3_31_1 + ABCDtemp * x_3_19_1;
                                    QUICKDouble x_5_32_0 = Ptempy * x_3_32_0 + WPtempy * x_3_32_1;
                                    QUICKDouble x_5_33_0 = Ptempy * x_3_33_0 + WPtempy * x_3_33_1 +  4 * ABCDtemp * x_3_18_1;
                                    QUICKDouble x_5_34_0 = Ptempy * x_3_34_0 + WPtempy * x_3_34_1;
                                    QUICKDouble x_6_20_0 = Ptempx * x_3_20_0 + WPtempx * x_3_20_1 +  2 * ABCDtemp * x_3_12_1;
                                    QUICKDouble x_6_21_0 = Ptempx * x_3_21_0 + WPtempx * x_3_21_1 +  2 * ABCDtemp * x_3_14_1;
                                    QUICKDouble x_6_22_0 = Ptempx * x_3_22_0 + WPtempx * x_3_22_1;
                                    QUICKDouble x_6_23_0 = Ptempx * x_3_23_0 + WPtempx * x_3_23_1 +  2 * ABCDtemp * x_3_10_1;
                                    QUICKDouble x_6_24_0 = Ptempx * x_3_24_0 + WPtempx * x_3_24_1 + ABCDtemp * x_3_15_1;
                                    QUICKDouble x_6_25_0 = Ptempx * x_3_25_0 + WPtempx * x_3_25_1 + ABCDtemp * x_3_16_1;
                                    QUICKDouble x_6_26_0 = Ptempx * x_3_26_0 + WPtempx * x_3_26_1 +  3 * ABCDtemp * x_3_13_1;
                                    QUICKDouble x_6_27_0 = Ptempx * x_3_27_0 + WPtempx * x_3_27_1 + ABCDtemp * x_3_19_1;
                                    QUICKDouble x_6_28_0 = Ptempx * x_3_28_0 + WPtempx * x_3_28_1 +  3 * ABCDtemp * x_3_11_1;
                                    QUICKDouble x_6_29_0 = Ptempx * x_3_29_0 + WPtempx * x_3_29_1 + ABCDtemp * x_3_18_1;
                                    QUICKDouble x_6_30_0 = Ptempx * x_3_30_0 + WPtempx * x_3_30_1;
                                    QUICKDouble x_6_31_0 = Ptempx * x_3_31_0 + WPtempx * x_3_31_1;
                                    QUICKDouble x_6_32_0 = Ptempx * x_3_32_0 + WPtempx * x_3_32_1 +  4 * ABCDtemp * x_3_17_1;
                                    QUICKDouble x_6_33_0 = Ptempx * x_3_33_0 + WPtempx * x_3_33_1;
                                    QUICKDouble x_6_34_0 = Ptempx * x_3_34_0 + WPtempx * x_3_34_1;
                                    QUICKDouble x_7_20_0 = Ptempx * x_1_20_0 + WPtempx * x_1_20_1 + ABtemp * ( x_0_20_0 - CDcom * x_0_20_1) +  2 * ABCDtemp * x_1_12_1;
                                    QUICKDouble x_7_21_0 = Ptempx * x_1_21_0 + WPtempx * x_1_21_1 + ABtemp * ( x_0_21_0 - CDcom * x_0_21_1) +  2 * ABCDtemp * x_1_14_1;
                                    QUICKDouble x_7_22_0 = Ptempx * x_1_22_0 + WPtempx * x_1_22_1 + ABtemp * ( x_0_22_0 - CDcom * x_0_22_1);
                                    QUICKDouble x_7_23_0 = Ptempx * x_1_23_0 + WPtempx * x_1_23_1 + ABtemp * ( x_0_23_0 - CDcom * x_0_23_1) +  2 * ABCDtemp * x_1_10_1;
                                    QUICKDouble x_7_24_0 = Ptempx * x_1_24_0 + WPtempx * x_1_24_1 + ABtemp * ( x_0_24_0 - CDcom * x_0_24_1) + ABCDtemp * x_1_15_1;
                                    QUICKDouble x_7_25_0 = Ptempx * x_1_25_0 + WPtempx * x_1_25_1 + ABtemp * ( x_0_25_0 - CDcom * x_0_25_1) + ABCDtemp * x_1_16_1;
                                    QUICKDouble x_7_26_0 = Ptempx * x_1_26_0 + WPtempx * x_1_26_1 + ABtemp * ( x_0_26_0 - CDcom * x_0_26_1) +  3 * ABCDtemp * x_1_13_1;
                                    QUICKDouble x_7_27_0 = Ptempx * x_1_27_0 + WPtempx * x_1_27_1 + ABtemp * ( x_0_27_0 - CDcom * x_0_27_1) + ABCDtemp * x_1_19_1;
                                    QUICKDouble x_7_28_0 = Ptempx * x_1_28_0 + WPtempx * x_1_28_1 + ABtemp * ( x_0_28_0 - CDcom * x_0_28_1) +  3 * ABCDtemp * x_1_11_1;
                                    QUICKDouble x_7_29_0 = Ptempx * x_1_29_0 + WPtempx * x_1_29_1 + ABtemp * ( x_0_29_0 - CDcom * x_0_29_1) + ABCDtemp * x_1_18_1;
                                    QUICKDouble x_7_30_0 = Ptempx * x_1_30_0 + WPtempx * x_1_30_1 + ABtemp * ( x_0_30_0 - CDcom * x_0_30_1);
                                    QUICKDouble x_7_31_0 = Ptempx * x_1_31_0 + WPtempx * x_1_31_1 + ABtemp * ( x_0_31_0 - CDcom * x_0_31_1);
                                    QUICKDouble x_7_32_0 = Ptempx * x_1_32_0 + WPtempx * x_1_32_1 + ABtemp * ( x_0_32_0 - CDcom * x_0_32_1) +  4 * ABCDtemp * x_1_17_1;
                                    QUICKDouble x_7_33_0 = Ptempx * x_1_33_0 + WPtempx * x_1_33_1 + ABtemp * ( x_0_33_0 - CDcom * x_0_33_1);
                                    QUICKDouble x_7_34_0 = Ptempx * x_1_34_0 + WPtempx * x_1_34_1 + ABtemp * ( x_0_34_0 - CDcom * x_0_34_1);
                                    QUICKDouble x_8_20_0 = Ptempy * x_2_20_0 + WPtempy * x_2_20_1 + ABtemp * ( x_0_20_0 - CDcom * x_0_20_1) +  2 * ABCDtemp * x_2_11_1;
                                    QUICKDouble x_8_21_0 = Ptempy * x_2_21_0 + WPtempy * x_2_21_1 + ABtemp * ( x_0_21_0 - CDcom * x_0_21_1);
                                    QUICKDouble x_8_22_0 = Ptempy * x_2_22_0 + WPtempy * x_2_22_1 + ABtemp * ( x_0_22_0 - CDcom * x_0_22_1) +  2 * ABCDtemp * x_2_16_1;
                                    QUICKDouble x_8_23_0 = Ptempy * x_2_23_0 + WPtempy * x_2_23_1 + ABtemp * ( x_0_23_0 - CDcom * x_0_23_1) + ABCDtemp * x_2_13_1;
                                    QUICKDouble x_8_24_0 = Ptempy * x_2_24_0 + WPtempy * x_2_24_1 + ABtemp * ( x_0_24_0 - CDcom * x_0_24_1) +  2 * ABCDtemp * x_2_10_1;
                                    QUICKDouble x_8_25_0 = Ptempy * x_2_25_0 + WPtempy * x_2_25_1 + ABtemp * ( x_0_25_0 - CDcom * x_0_25_1) + ABCDtemp * x_2_14_1;
                                    QUICKDouble x_8_26_0 = Ptempy * x_2_26_0 + WPtempy * x_2_26_1 + ABtemp * ( x_0_26_0 - CDcom * x_0_26_1);
                                    QUICKDouble x_8_27_0 = Ptempy * x_2_27_0 + WPtempy * x_2_27_1 + ABtemp * ( x_0_27_0 - CDcom * x_0_27_1);
                                    QUICKDouble x_8_28_0 = Ptempy * x_2_28_0 + WPtempy * x_2_28_1 + ABtemp * ( x_0_28_0 - CDcom * x_0_28_1) + ABCDtemp * x_2_17_1;
                                    QUICKDouble x_8_29_0 = Ptempy * x_2_29_0 + WPtempy * x_2_29_1 + ABtemp * ( x_0_29_0 - CDcom * x_0_29_1) +  3 * ABCDtemp * x_2_12_1;
                                    QUICKDouble x_8_30_0 = Ptempy * x_2_30_0 + WPtempy * x_2_30_1 + ABtemp * ( x_0_30_0 - CDcom * x_0_30_1) +  3 * ABCDtemp * x_2_15_1;
                                    QUICKDouble x_8_31_0 = Ptempy * x_2_31_0 + WPtempy * x_2_31_1 + ABtemp * ( x_0_31_0 - CDcom * x_0_31_1) + ABCDtemp * x_2_19_1;
                                    QUICKDouble x_8_32_0 = Ptempy * x_2_32_0 + WPtempy * x_2_32_1 + ABtemp * ( x_0_32_0 - CDcom * x_0_32_1);
                                    QUICKDouble x_8_33_0 = Ptempy * x_2_33_0 + WPtempy * x_2_33_1 + ABtemp * ( x_0_33_0 - CDcom * x_0_33_1) +  4 * ABCDtemp * x_2_18_1;
                                    QUICKDouble x_8_34_0 = Ptempy * x_2_34_0 + WPtempy * x_2_34_1 + ABtemp * ( x_0_34_0 - CDcom * x_0_34_1);
                                    QUICKDouble x_9_20_0 = Ptempz * x_3_20_0 + WPtempz * x_3_20_1 + ABtemp * ( x_0_20_0 - CDcom * x_0_20_1);
                                    QUICKDouble x_9_21_0 = Ptempz * x_3_21_0 + WPtempz * x_3_21_1 + ABtemp * ( x_0_21_0 - CDcom * x_0_21_1) +  2 * ABCDtemp * x_3_13_1;
                                    QUICKDouble x_9_22_0 = Ptempz * x_3_22_0 + WPtempz * x_3_22_1 + ABtemp * ( x_0_22_0 - CDcom * x_0_22_1) +  2 * ABCDtemp * x_3_15_1;
                                    QUICKDouble x_9_23_0 = Ptempz * x_3_23_0 + WPtempz * x_3_23_1 + ABtemp * ( x_0_23_0 - CDcom * x_0_23_1) + ABCDtemp * x_3_11_1;
                                    QUICKDouble x_9_24_0 = Ptempz * x_3_24_0 + WPtempz * x_3_24_1 + ABtemp * ( x_0_24_0 - CDcom * x_0_24_1) + ABCDtemp * x_3_12_1;
                                    QUICKDouble x_9_25_0 = Ptempz * x_3_25_0 + WPtempz * x_3_25_1 + ABtemp * ( x_0_25_0 - CDcom * x_0_25_1) +  2 * ABCDtemp * x_3_10_1;
                                    QUICKDouble x_9_26_0 = Ptempz * x_3_26_0 + WPtempz * x_3_26_1 + ABtemp * ( x_0_26_0 - CDcom * x_0_26_1) + ABCDtemp * x_3_17_1;
                                    QUICKDouble x_9_27_0 = Ptempz * x_3_27_0 + WPtempz * x_3_27_1 + ABtemp * ( x_0_27_0 - CDcom * x_0_27_1) +  3 * ABCDtemp * x_3_14_1;
                                    QUICKDouble x_9_28_0 = Ptempz * x_3_28_0 + WPtempz * x_3_28_1 + ABtemp * ( x_0_28_0 - CDcom * x_0_28_1);
                                    QUICKDouble x_9_29_0 = Ptempz * x_3_29_0 + WPtempz * x_3_29_1 + ABtemp * ( x_0_29_0 - CDcom * x_0_29_1);
                                    QUICKDouble x_9_30_0 = Ptempz * x_3_30_0 + WPtempz * x_3_30_1 + ABtemp * ( x_0_30_0 - CDcom * x_0_30_1) + ABCDtemp * x_3_18_1;
                                    QUICKDouble x_9_31_0 = Ptempz * x_3_31_0 + WPtempz * x_3_31_1 + ABtemp * ( x_0_31_0 - CDcom * x_0_31_1) +  3 * ABCDtemp * x_3_16_1;
                                    QUICKDouble x_9_32_0 = Ptempz * x_3_32_0 + WPtempz * x_3_32_1 + ABtemp * ( x_0_32_0 - CDcom * x_0_32_1);
                                    QUICKDouble x_9_33_0 = Ptempz * x_3_33_0 + WPtempz * x_3_33_1 + ABtemp * ( x_0_33_0 - CDcom * x_0_33_1);
                                    QUICKDouble x_9_34_0 = Ptempz * x_3_34_0 + WPtempz * x_3_34_1 + ABtemp * ( x_0_34_0 - CDcom * x_0_34_1) +  4 * ABCDtemp * x_3_19_1;
                                    
                                    LOC2(store, 4,20, STOREDIM, STOREDIM) += x_4_20_0;
                                    LOC2(store, 4,21, STOREDIM, STOREDIM) += x_4_21_0;
                                    LOC2(store, 4,22, STOREDIM, STOREDIM) += x_4_22_0;
                                    LOC2(store, 4,23, STOREDIM, STOREDIM) += x_4_23_0;
                                    LOC2(store, 4,24, STOREDIM, STOREDIM) += x_4_24_0;
                                    LOC2(store, 4,25, STOREDIM, STOREDIM) += x_4_25_0;
                                    LOC2(store, 4,26, STOREDIM, STOREDIM) += x_4_26_0;
                                    LOC2(store, 4,27, STOREDIM, STOREDIM) += x_4_27_0;
                                    LOC2(store, 4,28, STOREDIM, STOREDIM) += x_4_28_0;
                                    LOC2(store, 4,29, STOREDIM, STOREDIM) += x_4_29_0;
                                    LOC2(store, 4,30, STOREDIM, STOREDIM) += x_4_30_0;
                                    LOC2(store, 4,31, STOREDIM, STOREDIM) += x_4_31_0;
                                    LOC2(store, 4,32, STOREDIM, STOREDIM) += x_4_32_0;
                                    LOC2(store, 4,33, STOREDIM, STOREDIM) += x_4_33_0;
                                    LOC2(store, 4,34, STOREDIM, STOREDIM) += x_4_34_0;
                                    LOC2(store, 5,20, STOREDIM, STOREDIM) += x_5_20_0;
                                    LOC2(store, 5,21, STOREDIM, STOREDIM) += x_5_21_0;
                                    LOC2(store, 5,22, STOREDIM, STOREDIM) += x_5_22_0;
                                    LOC2(store, 5,23, STOREDIM, STOREDIM) += x_5_23_0;
                                    LOC2(store, 5,24, STOREDIM, STOREDIM) += x_5_24_0;
                                    LOC2(store, 5,25, STOREDIM, STOREDIM) += x_5_25_0;
                                    LOC2(store, 5,26, STOREDIM, STOREDIM) += x_5_26_0;
                                    LOC2(store, 5,27, STOREDIM, STOREDIM) += x_5_27_0;
                                    LOC2(store, 5,28, STOREDIM, STOREDIM) += x_5_28_0;
                                    LOC2(store, 5,29, STOREDIM, STOREDIM) += x_5_29_0;
                                    LOC2(store, 5,30, STOREDIM, STOREDIM) += x_5_30_0;
                                    LOC2(store, 5,31, STOREDIM, STOREDIM) += x_5_31_0;
                                    LOC2(store, 5,32, STOREDIM, STOREDIM) += x_5_32_0;
                                    LOC2(store, 5,33, STOREDIM, STOREDIM) += x_5_33_0;
                                    LOC2(store, 5,34, STOREDIM, STOREDIM) += x_5_34_0;
                                    LOC2(store, 6,20, STOREDIM, STOREDIM) += x_6_20_0;
                                    LOC2(store, 6,21, STOREDIM, STOREDIM) += x_6_21_0;
                                    LOC2(store, 6,22, STOREDIM, STOREDIM) += x_6_22_0;
                                    LOC2(store, 6,23, STOREDIM, STOREDIM) += x_6_23_0;
                                    LOC2(store, 6,24, STOREDIM, STOREDIM) += x_6_24_0;
                                    LOC2(store, 6,25, STOREDIM, STOREDIM) += x_6_25_0;
                                    LOC2(store, 6,26, STOREDIM, STOREDIM) += x_6_26_0;
                                    LOC2(store, 6,27, STOREDIM, STOREDIM) += x_6_27_0;
                                    LOC2(store, 6,28, STOREDIM, STOREDIM) += x_6_28_0;
                                    LOC2(store, 6,29, STOREDIM, STOREDIM) += x_6_29_0;
                                    LOC2(store, 6,30, STOREDIM, STOREDIM) += x_6_30_0;
                                    LOC2(store, 6,31, STOREDIM, STOREDIM) += x_6_31_0;
                                    LOC2(store, 6,32, STOREDIM, STOREDIM) += x_6_32_0;
                                    LOC2(store, 6,33, STOREDIM, STOREDIM) += x_6_33_0;
                                    LOC2(store, 6,34, STOREDIM, STOREDIM) += x_6_34_0;
                                    LOC2(store, 7,20, STOREDIM, STOREDIM) += x_7_20_0;
                                    LOC2(store, 7,21, STOREDIM, STOREDIM) += x_7_21_0;
                                    LOC2(store, 7,22, STOREDIM, STOREDIM) += x_7_22_0;
                                    LOC2(store, 7,23, STOREDIM, STOREDIM) += x_7_23_0;
                                    LOC2(store, 7,24, STOREDIM, STOREDIM) += x_7_24_0;
                                    LOC2(store, 7,25, STOREDIM, STOREDIM) += x_7_25_0;
                                    LOC2(store, 7,26, STOREDIM, STOREDIM) += x_7_26_0;
                                    LOC2(store, 7,27, STOREDIM, STOREDIM) += x_7_27_0;
                                    LOC2(store, 7,28, STOREDIM, STOREDIM) += x_7_28_0;
                                    LOC2(store, 7,29, STOREDIM, STOREDIM) += x_7_29_0;
                                    LOC2(store, 7,30, STOREDIM, STOREDIM) += x_7_30_0;
                                    LOC2(store, 7,31, STOREDIM, STOREDIM) += x_7_31_0;
                                    LOC2(store, 7,32, STOREDIM, STOREDIM) += x_7_32_0;
                                    LOC2(store, 7,33, STOREDIM, STOREDIM) += x_7_33_0;
                                    LOC2(store, 7,34, STOREDIM, STOREDIM) += x_7_34_0;
                                    LOC2(store, 8,20, STOREDIM, STOREDIM) += x_8_20_0;
                                    LOC2(store, 8,21, STOREDIM, STOREDIM) += x_8_21_0;
                                    LOC2(store, 8,22, STOREDIM, STOREDIM) += x_8_22_0;
                                    LOC2(store, 8,23, STOREDIM, STOREDIM) += x_8_23_0;
                                    LOC2(store, 8,24, STOREDIM, STOREDIM) += x_8_24_0;
                                    LOC2(store, 8,25, STOREDIM, STOREDIM) += x_8_25_0;
                                    LOC2(store, 8,26, STOREDIM, STOREDIM) += x_8_26_0;
                                    LOC2(store, 8,27, STOREDIM, STOREDIM) += x_8_27_0;
                                    LOC2(store, 8,28, STOREDIM, STOREDIM) += x_8_28_0;
                                    LOC2(store, 8,29, STOREDIM, STOREDIM) += x_8_29_0;
                                    LOC2(store, 8,30, STOREDIM, STOREDIM) += x_8_30_0;
                                    LOC2(store, 8,31, STOREDIM, STOREDIM) += x_8_31_0;
                                    LOC2(store, 8,32, STOREDIM, STOREDIM) += x_8_32_0;
                                    LOC2(store, 8,33, STOREDIM, STOREDIM) += x_8_33_0;
                                    LOC2(store, 8,34, STOREDIM, STOREDIM) += x_8_34_0;
                                    LOC2(store, 9,20, STOREDIM, STOREDIM) += x_9_20_0;
                                    LOC2(store, 9,21, STOREDIM, STOREDIM) += x_9_21_0;
                                    LOC2(store, 9,22, STOREDIM, STOREDIM) += x_9_22_0;
                                    LOC2(store, 9,23, STOREDIM, STOREDIM) += x_9_23_0;
                                    LOC2(store, 9,24, STOREDIM, STOREDIM) += x_9_24_0;
                                    LOC2(store, 9,25, STOREDIM, STOREDIM) += x_9_25_0;
                                    LOC2(store, 9,26, STOREDIM, STOREDIM) += x_9_26_0;
                                    LOC2(store, 9,27, STOREDIM, STOREDIM) += x_9_27_0;
                                    LOC2(store, 9,28, STOREDIM, STOREDIM) += x_9_28_0;
                                    LOC2(store, 9,29, STOREDIM, STOREDIM) += x_9_29_0;
                                    LOC2(store, 9,30, STOREDIM, STOREDIM) += x_9_30_0;
                                    LOC2(store, 9,31, STOREDIM, STOREDIM) += x_9_31_0;
                                    LOC2(store, 9,32, STOREDIM, STOREDIM) += x_9_32_0;
                                    LOC2(store, 9,33, STOREDIM, STOREDIM) += x_9_33_0;
                                    LOC2(store, 9,34, STOREDIM, STOREDIM) += x_9_34_0;
                                    
                                    if (I+J>=3 && K+L==4){ 
                                        
                                        //PSDS(2, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz, ABCDtemp);
                                        
                                        QUICKDouble x_1_4_2 = Ptempx * x_0_4_2 + WPtempx * x_0_4_3 + ABCDtemp * x_0_2_3;
                                        QUICKDouble x_2_4_2 = Ptempy * x_0_4_2 + WPtempy * x_0_4_3 + ABCDtemp * x_0_1_3;
                                        QUICKDouble x_3_4_2 = Ptempz * x_0_4_2 + WPtempz * x_0_4_3;
                                        
                                        QUICKDouble x_1_5_2 = Ptempx * x_0_5_2 + WPtempx * x_0_5_3;
                                        QUICKDouble x_2_5_2 = Ptempy * x_0_5_2 + WPtempy * x_0_5_3 + ABCDtemp * x_0_3_3;
                                        QUICKDouble x_3_5_2 = Ptempz * x_0_5_2 + WPtempz * x_0_5_3 + ABCDtemp * x_0_2_3;
                                        
                                        QUICKDouble x_1_6_2 = Ptempx * x_0_6_2 + WPtempx * x_0_6_3 + ABCDtemp * x_0_3_3;
                                        QUICKDouble x_2_6_2 = Ptempy * x_0_6_2 + WPtempy * x_0_6_3;
                                        QUICKDouble x_3_6_2 = Ptempz * x_0_6_2 + WPtempz * x_0_6_3 + ABCDtemp * x_0_1_3;
                                        
                                        QUICKDouble x_1_7_2 = Ptempx * x_0_7_2 + WPtempx * x_0_7_3 + ABCDtemp * x_0_1_3 * 2;
                                        QUICKDouble x_2_7_2 = Ptempy * x_0_7_2 + WPtempy * x_0_7_3;
                                        QUICKDouble x_3_7_2 = Ptempz * x_0_7_2 + WPtempz * x_0_7_3;
                                        
                                        QUICKDouble x_1_8_2 = Ptempx * x_0_8_2 + WPtempx * x_0_8_3;
                                        QUICKDouble x_2_8_2 = Ptempy * x_0_8_2 + WPtempy * x_0_8_3 + ABCDtemp * x_0_2_3 * 2;
                                        QUICKDouble x_3_8_2 = Ptempz * x_0_8_2 + WPtempz * x_0_8_3;
                                        
                                        QUICKDouble x_1_9_2 = Ptempx * x_0_9_2 + WPtempx * x_0_9_3;
                                        QUICKDouble x_2_9_2 = Ptempy * x_0_9_2 + WPtempy * x_0_9_3;
                                        QUICKDouble x_3_9_2 = Ptempz * x_0_9_2 + WPtempz * x_0_9_3 + ABCDtemp * x_0_3_3 * 2;    
                                        
                                        //PSFS(2, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz, ABCDtemp);
                                        
                                        QUICKDouble x_1_10_2 = Ptempx * x_0_10_2 + WPtempx * x_0_10_3 + ABCDtemp *  x_0_5_3;
                                        QUICKDouble x_2_10_2 = Ptempy * x_0_10_2 + WPtempy * x_0_10_3 + ABCDtemp *  x_0_6_3;
                                        QUICKDouble x_3_10_2 = Ptempz * x_0_10_2 + WPtempz * x_0_10_3 + ABCDtemp *  x_0_4_3;
                                        QUICKDouble x_1_11_2 = Ptempx * x_0_11_2 + WPtempx * x_0_11_3 +  2 * ABCDtemp *  x_0_4_3;
                                        QUICKDouble x_2_11_2 = Ptempy * x_0_11_2 + WPtempy * x_0_11_3 + ABCDtemp *  x_0_7_3;
                                        QUICKDouble x_3_11_2 = Ptempz * x_0_11_2 + WPtempz * x_0_11_3;
                                        QUICKDouble x_1_12_2 = Ptempx * x_0_12_2 + WPtempx * x_0_12_3 + ABCDtemp *  x_0_8_3;
                                        QUICKDouble x_2_12_2 = Ptempy * x_0_12_2 + WPtempy * x_0_12_3 +  2 * ABCDtemp *  x_0_4_3;
                                        QUICKDouble x_3_12_2 = Ptempz * x_0_12_2 + WPtempz * x_0_12_3;
                                        QUICKDouble x_1_13_2 = Ptempx * x_0_13_2 + WPtempx * x_0_13_3 +  2 * ABCDtemp *  x_0_6_3;
                                        QUICKDouble x_2_13_2 = Ptempy * x_0_13_2 + WPtempy * x_0_13_3;
                                        QUICKDouble x_3_13_2 = Ptempz * x_0_13_2 + WPtempz * x_0_13_3 + ABCDtemp *  x_0_7_3;
                                        QUICKDouble x_1_14_2 = Ptempx * x_0_14_2 + WPtempx * x_0_14_3 + ABCDtemp *  x_0_9_3;
                                        QUICKDouble x_2_14_2 = Ptempy * x_0_14_2 + WPtempy * x_0_14_3;
                                        QUICKDouble x_3_14_2 = Ptempz * x_0_14_2 + WPtempz * x_0_14_3 +  2 * ABCDtemp *  x_0_6_3;
                                        QUICKDouble x_1_15_2 = Ptempx * x_0_15_2 + WPtempx * x_0_15_3;
                                        QUICKDouble x_2_15_2 = Ptempy * x_0_15_2 + WPtempy * x_0_15_3 +  2 * ABCDtemp *  x_0_5_3;
                                        QUICKDouble x_3_15_2 = Ptempz * x_0_15_2 + WPtempz * x_0_15_3 + ABCDtemp *  x_0_8_3;
                                        QUICKDouble x_1_16_2 = Ptempx * x_0_16_2 + WPtempx * x_0_16_3;
                                        QUICKDouble x_2_16_2 = Ptempy * x_0_16_2 + WPtempy * x_0_16_3 + ABCDtemp *  x_0_9_3;
                                        QUICKDouble x_3_16_2 = Ptempz * x_0_16_2 + WPtempz * x_0_16_3 +  2 * ABCDtemp *  x_0_5_3;
                                        QUICKDouble x_1_17_2 = Ptempx * x_0_17_2 + WPtempx * x_0_17_3 +  3 * ABCDtemp *  x_0_7_3;
                                        QUICKDouble x_2_17_2 = Ptempy * x_0_17_2 + WPtempy * x_0_17_3;
                                        QUICKDouble x_3_17_2 = Ptempz * x_0_17_2 + WPtempz * x_0_17_3;
                                        QUICKDouble x_1_18_2 = Ptempx * x_0_18_2 + WPtempx * x_0_18_3;
                                        QUICKDouble x_2_18_2 = Ptempy * x_0_18_2 + WPtempy * x_0_18_3 +  3 * ABCDtemp *  x_0_8_3;
                                        QUICKDouble x_3_18_2 = Ptempz * x_0_18_2 + WPtempz * x_0_18_3;
                                        QUICKDouble x_1_19_2 = Ptempx * x_0_19_2 + WPtempx * x_0_19_3;
                                        QUICKDouble x_2_19_2 = Ptempy * x_0_19_2 + WPtempy * x_0_19_3;
                                        QUICKDouble x_3_19_2 = Ptempz * x_0_19_2 + WPtempz * x_0_19_3 +  3 * ABCDtemp *  x_0_9_3;
                                        
                                        
                                        //DSFS(1, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz, ABCDtemp, ABtemp, CDcom);
                                        
                                        QUICKDouble x_4_10_1 = Ptempx * x_2_10_1 + WPtempx * x_2_10_2 + ABCDtemp * x_2_5_2;
                                        QUICKDouble x_4_11_1 = Ptempx * x_2_11_1 + WPtempx * x_2_11_2 +  2 * ABCDtemp * x_2_4_2;
                                        QUICKDouble x_4_12_1 = Ptempx * x_2_12_1 + WPtempx * x_2_12_2 + ABCDtemp * x_2_8_2;
                                        QUICKDouble x_4_13_1 = Ptempx * x_2_13_1 + WPtempx * x_2_13_2 +  2 * ABCDtemp * x_2_6_2;
                                        QUICKDouble x_4_14_1 = Ptempx * x_2_14_1 + WPtempx * x_2_14_2 + ABCDtemp * x_2_9_2;
                                        QUICKDouble x_4_15_1 = Ptempx * x_2_15_1 + WPtempx * x_2_15_2;
                                        QUICKDouble x_4_16_1 = Ptempx * x_2_16_1 + WPtempx * x_2_16_2;
                                        QUICKDouble x_4_17_1 = Ptempx * x_2_17_1 + WPtempx * x_2_17_2 +  3 * ABCDtemp * x_2_7_2;
                                        QUICKDouble x_4_18_1 = Ptempx * x_2_18_1 + WPtempx * x_2_18_2;
                                        QUICKDouble x_4_19_1 = Ptempx * x_2_19_1 + WPtempx * x_2_19_2;
                                        QUICKDouble x_5_10_1 = Ptempy * x_3_10_1 + WPtempy * x_3_10_2 + ABCDtemp * x_3_6_2;
                                        QUICKDouble x_5_11_1 = Ptempy * x_3_11_1 + WPtempy * x_3_11_2 + ABCDtemp * x_3_7_2;
                                        QUICKDouble x_5_12_1 = Ptempy * x_3_12_1 + WPtempy * x_3_12_2 +  2 * ABCDtemp * x_3_4_2;
                                        QUICKDouble x_5_13_1 = Ptempy * x_3_13_1 + WPtempy * x_3_13_2;
                                        QUICKDouble x_5_14_1 = Ptempy * x_3_14_1 + WPtempy * x_3_14_2;
                                        QUICKDouble x_5_15_1 = Ptempy * x_3_15_1 + WPtempy * x_3_15_2 +  2 * ABCDtemp * x_3_5_2;
                                        QUICKDouble x_5_16_1 = Ptempy * x_3_16_1 + WPtempy * x_3_16_2 + ABCDtemp * x_3_9_2;
                                        QUICKDouble x_5_17_1 = Ptempy * x_3_17_1 + WPtempy * x_3_17_2;
                                        QUICKDouble x_5_18_1 = Ptempy * x_3_18_1 + WPtempy * x_3_18_2 +  3 * ABCDtemp * x_3_8_2;
                                        QUICKDouble x_5_19_1 = Ptempy * x_3_19_1 + WPtempy * x_3_19_2;
                                        QUICKDouble x_6_10_1 = Ptempx * x_3_10_1 + WPtempx * x_3_10_2 + ABCDtemp * x_3_5_2;
                                        QUICKDouble x_6_11_1 = Ptempx * x_3_11_1 + WPtempx * x_3_11_2 +  2 * ABCDtemp * x_3_4_2;
                                        QUICKDouble x_6_12_1 = Ptempx * x_3_12_1 + WPtempx * x_3_12_2 + ABCDtemp * x_3_8_2;
                                        QUICKDouble x_6_13_1 = Ptempx * x_3_13_1 + WPtempx * x_3_13_2 +  2 * ABCDtemp * x_3_6_2;
                                        QUICKDouble x_6_14_1 = Ptempx * x_3_14_1 + WPtempx * x_3_14_2 + ABCDtemp * x_3_9_2;
                                        QUICKDouble x_6_15_1 = Ptempx * x_3_15_1 + WPtempx * x_3_15_2;
                                        QUICKDouble x_6_16_1 = Ptempx * x_3_16_1 + WPtempx * x_3_16_2;
                                        QUICKDouble x_6_17_1 = Ptempx * x_3_17_1 + WPtempx * x_3_17_2 +  3 * ABCDtemp * x_3_7_2;
                                        QUICKDouble x_6_18_1 = Ptempx * x_3_18_1 + WPtempx * x_3_18_2;
                                        QUICKDouble x_6_19_1 = Ptempx * x_3_19_1 + WPtempx * x_3_19_2;
                                        QUICKDouble x_7_10_1 = Ptempx * x_1_10_1 + WPtempx * x_1_10_2 + ABtemp * ( x_0_10_1 - CDcom * x_0_10_2) + ABCDtemp * x_1_5_2;
                                        QUICKDouble x_7_11_1 = Ptempx * x_1_11_1 + WPtempx * x_1_11_2 + ABtemp * ( x_0_11_1 - CDcom * x_0_11_2) +  2 * ABCDtemp * x_1_4_2;
                                        QUICKDouble x_7_12_1 = Ptempx * x_1_12_1 + WPtempx * x_1_12_2 + ABtemp * ( x_0_12_1 - CDcom * x_0_12_2) + ABCDtemp * x_1_8_2;
                                        QUICKDouble x_7_13_1 = Ptempx * x_1_13_1 + WPtempx * x_1_13_2 + ABtemp * ( x_0_13_1 - CDcom * x_0_13_2) +  2 * ABCDtemp * x_1_6_2;
                                        QUICKDouble x_7_14_1 = Ptempx * x_1_14_1 + WPtempx * x_1_14_2 + ABtemp * ( x_0_14_1 - CDcom * x_0_14_2) + ABCDtemp * x_1_9_2;
                                        QUICKDouble x_7_15_1 = Ptempx * x_1_15_1 + WPtempx * x_1_15_2 + ABtemp * ( x_0_15_1 - CDcom * x_0_15_2);
                                        QUICKDouble x_7_16_1 = Ptempx * x_1_16_1 + WPtempx * x_1_16_2 + ABtemp * ( x_0_16_1 - CDcom * x_0_16_2);
                                        QUICKDouble x_7_17_1 = Ptempx * x_1_17_1 + WPtempx * x_1_17_2 + ABtemp * ( x_0_17_1 - CDcom * x_0_17_2) +  3 * ABCDtemp * x_1_7_2;
                                        QUICKDouble x_7_18_1 = Ptempx * x_1_18_1 + WPtempx * x_1_18_2 + ABtemp * ( x_0_18_1 - CDcom * x_0_18_2);
                                        QUICKDouble x_7_19_1 = Ptempx * x_1_19_1 + WPtempx * x_1_19_2 + ABtemp * ( x_0_19_1 - CDcom * x_0_19_2);
                                        QUICKDouble x_8_10_1 = Ptempy * x_2_10_1 + WPtempy * x_2_10_2 + ABtemp * ( x_0_10_1 - CDcom * x_0_10_2) + ABCDtemp * x_2_6_2;
                                        QUICKDouble x_8_11_1 = Ptempy * x_2_11_1 + WPtempy * x_2_11_2 + ABtemp * ( x_0_11_1 - CDcom * x_0_11_2) + ABCDtemp * x_2_7_2;
                                        QUICKDouble x_8_12_1 = Ptempy * x_2_12_1 + WPtempy * x_2_12_2 + ABtemp * ( x_0_12_1 - CDcom * x_0_12_2) +  2 * ABCDtemp * x_2_4_2;
                                        QUICKDouble x_8_13_1 = Ptempy * x_2_13_1 + WPtempy * x_2_13_2 + ABtemp * ( x_0_13_1 - CDcom * x_0_13_2);
                                        QUICKDouble x_8_14_1 = Ptempy * x_2_14_1 + WPtempy * x_2_14_2 + ABtemp * ( x_0_14_1 - CDcom * x_0_14_2);
                                        QUICKDouble x_8_15_1 = Ptempy * x_2_15_1 + WPtempy * x_2_15_2 + ABtemp * ( x_0_15_1 - CDcom * x_0_15_2) +  2 * ABCDtemp * x_2_5_2;
                                        QUICKDouble x_8_16_1 = Ptempy * x_2_16_1 + WPtempy * x_2_16_2 + ABtemp * ( x_0_16_1 - CDcom * x_0_16_2) + ABCDtemp * x_2_9_2;
                                        QUICKDouble x_8_17_1 = Ptempy * x_2_17_1 + WPtempy * x_2_17_2 + ABtemp * ( x_0_17_1 - CDcom * x_0_17_2);
                                        QUICKDouble x_8_18_1 = Ptempy * x_2_18_1 + WPtempy * x_2_18_2 + ABtemp * ( x_0_18_1 - CDcom * x_0_18_2) +  3 * ABCDtemp * x_2_8_2;
                                        QUICKDouble x_8_19_1 = Ptempy * x_2_19_1 + WPtempy * x_2_19_2 + ABtemp * ( x_0_19_1 - CDcom * x_0_19_2);
                                        QUICKDouble x_9_10_1 = Ptempz * x_3_10_1 + WPtempz * x_3_10_2 + ABtemp * ( x_0_10_1 - CDcom * x_0_10_2) + ABCDtemp * x_3_4_2;
                                        QUICKDouble x_9_11_1 = Ptempz * x_3_11_1 + WPtempz * x_3_11_2 + ABtemp * ( x_0_11_1 - CDcom * x_0_11_2);
                                        QUICKDouble x_9_12_1 = Ptempz * x_3_12_1 + WPtempz * x_3_12_2 + ABtemp * ( x_0_12_1 - CDcom * x_0_12_2);
                                        QUICKDouble x_9_13_1 = Ptempz * x_3_13_1 + WPtempz * x_3_13_2 + ABtemp * ( x_0_13_1 - CDcom * x_0_13_2) + ABCDtemp * x_3_7_2;
                                        QUICKDouble x_9_14_1 = Ptempz * x_3_14_1 + WPtempz * x_3_14_2 + ABtemp * ( x_0_14_1 - CDcom * x_0_14_2) +  2 * ABCDtemp * x_3_6_2;
                                        QUICKDouble x_9_15_1 = Ptempz * x_3_15_1 + WPtempz * x_3_15_2 + ABtemp * ( x_0_15_1 - CDcom * x_0_15_2) + ABCDtemp * x_3_8_2;
                                        QUICKDouble x_9_16_1 = Ptempz * x_3_16_1 + WPtempz * x_3_16_2 + ABtemp * ( x_0_16_1 - CDcom * x_0_16_2) +  2 * ABCDtemp * x_3_5_2;
                                        QUICKDouble x_9_17_1 = Ptempz * x_3_17_1 + WPtempz * x_3_17_2 + ABtemp * ( x_0_17_1 - CDcom * x_0_17_2);
                                        QUICKDouble x_9_18_1 = Ptempz * x_3_18_1 + WPtempz * x_3_18_2 + ABtemp * ( x_0_18_1 - CDcom * x_0_18_2);
                                        QUICKDouble x_9_19_1 = Ptempz * x_3_19_1 + WPtempz * x_3_19_2 + ABtemp * ( x_0_19_1 - CDcom * x_0_19_2) +  3 * ABCDtemp * x_3_9_2;
                                        
                                        //DSGS(1, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz, ABCDtemp, ABtemp, CDcom);
                                        QUICKDouble x_4_20_1 = Ptempx * x_2_20_1 + WPtempx * x_2_20_2 +  2 * ABCDtemp * x_2_12_2;
                                        QUICKDouble x_4_21_1 = Ptempx * x_2_21_1 + WPtempx * x_2_21_2 +  2 * ABCDtemp * x_2_14_2;
                                        QUICKDouble x_4_22_1 = Ptempx * x_2_22_1 + WPtempx * x_2_22_2;
                                        QUICKDouble x_4_23_1 = Ptempx * x_2_23_1 + WPtempx * x_2_23_2 +  2 * ABCDtemp * x_2_10_2;
                                        QUICKDouble x_4_24_1 = Ptempx * x_2_24_1 + WPtempx * x_2_24_2 + ABCDtemp * x_2_15_2;
                                        QUICKDouble x_4_25_1 = Ptempx * x_2_25_1 + WPtempx * x_2_25_2 + ABCDtemp * x_2_16_2;
                                        QUICKDouble x_4_26_1 = Ptempx * x_2_26_1 + WPtempx * x_2_26_2 +  3 * ABCDtemp * x_2_13_2;
                                        QUICKDouble x_4_27_1 = Ptempx * x_2_27_1 + WPtempx * x_2_27_2 + ABCDtemp * x_2_19_2;
                                        QUICKDouble x_4_28_1 = Ptempx * x_2_28_1 + WPtempx * x_2_28_2 +  3 * ABCDtemp * x_2_11_2;
                                        QUICKDouble x_4_29_1 = Ptempx * x_2_29_1 + WPtempx * x_2_29_2 + ABCDtemp * x_2_18_2;
                                        QUICKDouble x_4_30_1 = Ptempx * x_2_30_1 + WPtempx * x_2_30_2;
                                        QUICKDouble x_4_31_1 = Ptempx * x_2_31_1 + WPtempx * x_2_31_2;
                                        QUICKDouble x_4_32_1 = Ptempx * x_2_32_1 + WPtempx * x_2_32_2 +  4 * ABCDtemp * x_2_17_2;
                                        QUICKDouble x_4_33_1 = Ptempx * x_2_33_1 + WPtempx * x_2_33_2;
                                        QUICKDouble x_4_34_1 = Ptempx * x_2_34_1 + WPtempx * x_2_34_2;
                                        QUICKDouble x_5_20_1 = Ptempy * x_3_20_1 + WPtempy * x_3_20_2 +  2 * ABCDtemp * x_3_11_2;
                                        QUICKDouble x_5_21_1 = Ptempy * x_3_21_1 + WPtempy * x_3_21_2;
                                        QUICKDouble x_5_22_1 = Ptempy * x_3_22_1 + WPtempy * x_3_22_2 +  2 * ABCDtemp * x_3_16_2;
                                        QUICKDouble x_5_23_1 = Ptempy * x_3_23_1 + WPtempy * x_3_23_2 + ABCDtemp * x_3_13_2;
                                        QUICKDouble x_5_24_1 = Ptempy * x_3_24_1 + WPtempy * x_3_24_2 +  2 * ABCDtemp * x_3_10_2;
                                        QUICKDouble x_5_25_1 = Ptempy * x_3_25_1 + WPtempy * x_3_25_2 + ABCDtemp * x_3_14_2;
                                        QUICKDouble x_5_26_1 = Ptempy * x_3_26_1 + WPtempy * x_3_26_2;
                                        QUICKDouble x_5_27_1 = Ptempy * x_3_27_1 + WPtempy * x_3_27_2;
                                        QUICKDouble x_5_28_1 = Ptempy * x_3_28_1 + WPtempy * x_3_28_2 + ABCDtemp * x_3_17_2;
                                        QUICKDouble x_5_29_1 = Ptempy * x_3_29_1 + WPtempy * x_3_29_2 +  3 * ABCDtemp * x_3_12_2;
                                        QUICKDouble x_5_30_1 = Ptempy * x_3_30_1 + WPtempy * x_3_30_2 +  3 * ABCDtemp * x_3_15_2;
                                        QUICKDouble x_5_31_1 = Ptempy * x_3_31_1 + WPtempy * x_3_31_2 + ABCDtemp * x_3_19_2;
                                        QUICKDouble x_5_32_1 = Ptempy * x_3_32_1 + WPtempy * x_3_32_2;
                                        QUICKDouble x_5_33_1 = Ptempy * x_3_33_1 + WPtempy * x_3_33_2 +  4 * ABCDtemp * x_3_18_2;
                                        QUICKDouble x_5_34_1 = Ptempy * x_3_34_1 + WPtempy * x_3_34_2;
                                        QUICKDouble x_6_20_1 = Ptempx * x_3_20_1 + WPtempx * x_3_20_2 +  2 * ABCDtemp * x_3_12_2;
                                        QUICKDouble x_6_21_1 = Ptempx * x_3_21_1 + WPtempx * x_3_21_2 +  2 * ABCDtemp * x_3_14_2;
                                        QUICKDouble x_6_22_1 = Ptempx * x_3_22_1 + WPtempx * x_3_22_2;
                                        QUICKDouble x_6_23_1 = Ptempx * x_3_23_1 + WPtempx * x_3_23_2 +  2 * ABCDtemp * x_3_10_2;
                                        QUICKDouble x_6_24_1 = Ptempx * x_3_24_1 + WPtempx * x_3_24_2 + ABCDtemp * x_3_15_2;
                                        QUICKDouble x_6_25_1 = Ptempx * x_3_25_1 + WPtempx * x_3_25_2 + ABCDtemp * x_3_16_2;
                                        QUICKDouble x_6_26_1 = Ptempx * x_3_26_1 + WPtempx * x_3_26_2 +  3 * ABCDtemp * x_3_13_2;
                                        QUICKDouble x_6_27_1 = Ptempx * x_3_27_1 + WPtempx * x_3_27_2 + ABCDtemp * x_3_19_2;
                                        QUICKDouble x_6_28_1 = Ptempx * x_3_28_1 + WPtempx * x_3_28_2 +  3 * ABCDtemp * x_3_11_2;
                                        QUICKDouble x_6_29_1 = Ptempx * x_3_29_1 + WPtempx * x_3_29_2 + ABCDtemp * x_3_18_2;
                                        QUICKDouble x_6_30_1 = Ptempx * x_3_30_1 + WPtempx * x_3_30_2;
                                        QUICKDouble x_6_31_1 = Ptempx * x_3_31_1 + WPtempx * x_3_31_2;
                                        QUICKDouble x_6_32_1 = Ptempx * x_3_32_1 + WPtempx * x_3_32_2 +  4 * ABCDtemp * x_3_17_2;
                                        QUICKDouble x_6_33_1 = Ptempx * x_3_33_1 + WPtempx * x_3_33_2;
                                        QUICKDouble x_6_34_1 = Ptempx * x_3_34_1 + WPtempx * x_3_34_2;
                                        QUICKDouble x_7_20_1 = Ptempx * x_1_20_1 + WPtempx * x_1_20_2 + ABtemp * ( x_0_20_1 - CDcom * x_0_20_2) +  2 * ABCDtemp * x_1_12_2;
                                        QUICKDouble x_7_21_1 = Ptempx * x_1_21_1 + WPtempx * x_1_21_2 + ABtemp * ( x_0_21_1 - CDcom * x_0_21_2) +  2 * ABCDtemp * x_1_14_2;
                                        QUICKDouble x_7_22_1 = Ptempx * x_1_22_1 + WPtempx * x_1_22_2 + ABtemp * ( x_0_22_1 - CDcom * x_0_22_2);
                                        QUICKDouble x_7_23_1 = Ptempx * x_1_23_1 + WPtempx * x_1_23_2 + ABtemp * ( x_0_23_1 - CDcom * x_0_23_2) +  2 * ABCDtemp * x_1_10_2;
                                        QUICKDouble x_7_24_1 = Ptempx * x_1_24_1 + WPtempx * x_1_24_2 + ABtemp * ( x_0_24_1 - CDcom * x_0_24_2) + ABCDtemp * x_1_15_2;
                                        QUICKDouble x_7_25_1 = Ptempx * x_1_25_1 + WPtempx * x_1_25_2 + ABtemp * ( x_0_25_1 - CDcom * x_0_25_2) + ABCDtemp * x_1_16_2;
                                        QUICKDouble x_7_26_1 = Ptempx * x_1_26_1 + WPtempx * x_1_26_2 + ABtemp * ( x_0_26_1 - CDcom * x_0_26_2) +  3 * ABCDtemp * x_1_13_2;
                                        QUICKDouble x_7_27_1 = Ptempx * x_1_27_1 + WPtempx * x_1_27_2 + ABtemp * ( x_0_27_1 - CDcom * x_0_27_2) + ABCDtemp * x_1_19_2;
                                        QUICKDouble x_7_28_1 = Ptempx * x_1_28_1 + WPtempx * x_1_28_2 + ABtemp * ( x_0_28_1 - CDcom * x_0_28_2) +  3 * ABCDtemp * x_1_11_2;
                                        QUICKDouble x_7_29_1 = Ptempx * x_1_29_1 + WPtempx * x_1_29_2 + ABtemp * ( x_0_29_1 - CDcom * x_0_29_2) + ABCDtemp * x_1_18_2;
                                        QUICKDouble x_7_30_1 = Ptempx * x_1_30_1 + WPtempx * x_1_30_2 + ABtemp * ( x_0_30_1 - CDcom * x_0_30_2);
                                        QUICKDouble x_7_31_1 = Ptempx * x_1_31_1 + WPtempx * x_1_31_2 + ABtemp * ( x_0_31_1 - CDcom * x_0_31_2);
                                        QUICKDouble x_7_32_1 = Ptempx * x_1_32_1 + WPtempx * x_1_32_2 + ABtemp * ( x_0_32_1 - CDcom * x_0_32_2) +  4 * ABCDtemp * x_1_17_2;
                                        QUICKDouble x_7_33_1 = Ptempx * x_1_33_1 + WPtempx * x_1_33_2 + ABtemp * ( x_0_33_1 - CDcom * x_0_33_2);
                                        QUICKDouble x_7_34_1 = Ptempx * x_1_34_1 + WPtempx * x_1_34_2 + ABtemp * ( x_0_34_1 - CDcom * x_0_34_2);
                                        QUICKDouble x_8_20_1 = Ptempy * x_2_20_1 + WPtempy * x_2_20_2 + ABtemp * ( x_0_20_1 - CDcom * x_0_20_2) +  2 * ABCDtemp * x_2_11_2;
                                        QUICKDouble x_8_21_1 = Ptempy * x_2_21_1 + WPtempy * x_2_21_2 + ABtemp * ( x_0_21_1 - CDcom * x_0_21_2);
                                        QUICKDouble x_8_22_1 = Ptempy * x_2_22_1 + WPtempy * x_2_22_2 + ABtemp * ( x_0_22_1 - CDcom * x_0_22_2) +  2 * ABCDtemp * x_2_16_2;
                                        QUICKDouble x_8_23_1 = Ptempy * x_2_23_1 + WPtempy * x_2_23_2 + ABtemp * ( x_0_23_1 - CDcom * x_0_23_2) + ABCDtemp * x_2_13_2;
                                        QUICKDouble x_8_24_1 = Ptempy * x_2_24_1 + WPtempy * x_2_24_2 + ABtemp * ( x_0_24_1 - CDcom * x_0_24_2) +  2 * ABCDtemp * x_2_10_2;
                                        QUICKDouble x_8_25_1 = Ptempy * x_2_25_1 + WPtempy * x_2_25_2 + ABtemp * ( x_0_25_1 - CDcom * x_0_25_2) + ABCDtemp * x_2_14_2;
                                        QUICKDouble x_8_26_1 = Ptempy * x_2_26_1 + WPtempy * x_2_26_2 + ABtemp * ( x_0_26_1 - CDcom * x_0_26_2);
                                        QUICKDouble x_8_27_1 = Ptempy * x_2_27_1 + WPtempy * x_2_27_2 + ABtemp * ( x_0_27_1 - CDcom * x_0_27_2);
                                        QUICKDouble x_8_28_1 = Ptempy * x_2_28_1 + WPtempy * x_2_28_2 + ABtemp * ( x_0_28_1 - CDcom * x_0_28_2) + ABCDtemp * x_2_17_2;
                                        QUICKDouble x_8_29_1 = Ptempy * x_2_29_1 + WPtempy * x_2_29_2 + ABtemp * ( x_0_29_1 - CDcom * x_0_29_2) +  3 * ABCDtemp * x_2_12_2;
                                        QUICKDouble x_8_30_1 = Ptempy * x_2_30_1 + WPtempy * x_2_30_2 + ABtemp * ( x_0_30_1 - CDcom * x_0_30_2) +  3 * ABCDtemp * x_2_15_2;
                                        QUICKDouble x_8_31_1 = Ptempy * x_2_31_1 + WPtempy * x_2_31_2 + ABtemp * ( x_0_31_1 - CDcom * x_0_31_2) + ABCDtemp * x_2_19_2;
                                        QUICKDouble x_8_32_1 = Ptempy * x_2_32_1 + WPtempy * x_2_32_2 + ABtemp * ( x_0_32_1 - CDcom * x_0_32_2);
                                        QUICKDouble x_8_33_1 = Ptempy * x_2_33_1 + WPtempy * x_2_33_2 + ABtemp * ( x_0_33_1 - CDcom * x_0_33_2) +  4 * ABCDtemp * x_2_18_2;
                                        QUICKDouble x_8_34_1 = Ptempy * x_2_34_1 + WPtempy * x_2_34_2 + ABtemp * ( x_0_34_1 - CDcom * x_0_34_2);
                                        QUICKDouble x_9_20_1 = Ptempz * x_3_20_1 + WPtempz * x_3_20_2 + ABtemp * ( x_0_20_1 - CDcom * x_0_20_2);
                                        QUICKDouble x_9_21_1 = Ptempz * x_3_21_1 + WPtempz * x_3_21_2 + ABtemp * ( x_0_21_1 - CDcom * x_0_21_2) +  2 * ABCDtemp * x_3_13_2;
                                        QUICKDouble x_9_22_1 = Ptempz * x_3_22_1 + WPtempz * x_3_22_2 + ABtemp * ( x_0_22_1 - CDcom * x_0_22_2) +  2 * ABCDtemp * x_3_15_2;
                                        QUICKDouble x_9_23_1 = Ptempz * x_3_23_1 + WPtempz * x_3_23_2 + ABtemp * ( x_0_23_1 - CDcom * x_0_23_2) + ABCDtemp * x_3_11_2;
                                        QUICKDouble x_9_24_1 = Ptempz * x_3_24_1 + WPtempz * x_3_24_2 + ABtemp * ( x_0_24_1 - CDcom * x_0_24_2) + ABCDtemp * x_3_12_2;
                                        QUICKDouble x_9_25_1 = Ptempz * x_3_25_1 + WPtempz * x_3_25_2 + ABtemp * ( x_0_25_1 - CDcom * x_0_25_2) +  2 * ABCDtemp * x_3_10_2;
                                        QUICKDouble x_9_26_1 = Ptempz * x_3_26_1 + WPtempz * x_3_26_2 + ABtemp * ( x_0_26_1 - CDcom * x_0_26_2) + ABCDtemp * x_3_17_2;
                                        QUICKDouble x_9_27_1 = Ptempz * x_3_27_1 + WPtempz * x_3_27_2 + ABtemp * ( x_0_27_1 - CDcom * x_0_27_2) +  3 * ABCDtemp * x_3_14_2;
                                        QUICKDouble x_9_28_1 = Ptempz * x_3_28_1 + WPtempz * x_3_28_2 + ABtemp * ( x_0_28_1 - CDcom * x_0_28_2);
                                        QUICKDouble x_9_29_1 = Ptempz * x_3_29_1 + WPtempz * x_3_29_2 + ABtemp * ( x_0_29_1 - CDcom * x_0_29_2);
                                        QUICKDouble x_9_30_1 = Ptempz * x_3_30_1 + WPtempz * x_3_30_2 + ABtemp * ( x_0_30_1 - CDcom * x_0_30_2) + ABCDtemp * x_3_18_2;
                                        QUICKDouble x_9_31_1 = Ptempz * x_3_31_1 + WPtempz * x_3_31_2 + ABtemp * ( x_0_31_1 - CDcom * x_0_31_2) +  3 * ABCDtemp * x_3_16_2;
                                        QUICKDouble x_9_32_1 = Ptempz * x_3_32_1 + WPtempz * x_3_32_2 + ABtemp * ( x_0_32_1 - CDcom * x_0_32_2);
                                        QUICKDouble x_9_33_1 = Ptempz * x_3_33_1 + WPtempz * x_3_33_2 + ABtemp * ( x_0_33_1 - CDcom * x_0_33_2);
                                        QUICKDouble x_9_34_1 = Ptempz * x_3_34_1 + WPtempz * x_3_34_2 + ABtemp * ( x_0_34_1 - CDcom * x_0_34_2) +  4 * ABCDtemp * x_3_19_2;
                                        
                                        //FSGS(0, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz, ABCDtemp, ABtemp, CDcom);
                                        LOC2(store,10,20, STOREDIM, STOREDIM) += Ptempx * x_5_20_0 + WPtempx * x_5_20_1 +  2 * ABCDtemp * x_5_12_1;
                                        LOC2(store,10,21, STOREDIM, STOREDIM) += Ptempx * x_5_21_0 + WPtempx * x_5_21_1 +  2 * ABCDtemp * x_5_14_1;
                                        LOC2(store,10,22, STOREDIM, STOREDIM) += Ptempx * x_5_22_0 + WPtempx * x_5_22_1;
                                        LOC2(store,10,23, STOREDIM, STOREDIM) += Ptempx * x_5_23_0 + WPtempx * x_5_23_1 +  2 * ABCDtemp * x_5_10_1;
                                        LOC2(store,10,24, STOREDIM, STOREDIM) += Ptempx * x_5_24_0 + WPtempx * x_5_24_1 + ABCDtemp * x_5_15_1;
                                        LOC2(store,10,25, STOREDIM, STOREDIM) += Ptempx * x_5_25_0 + WPtempx * x_5_25_1 + ABCDtemp * x_5_16_1;
                                        LOC2(store,10,26, STOREDIM, STOREDIM) += Ptempx * x_5_26_0 + WPtempx * x_5_26_1 +  3 * ABCDtemp * x_5_13_1;
                                        LOC2(store,10,27, STOREDIM, STOREDIM) += Ptempx * x_5_27_0 + WPtempx * x_5_27_1 + ABCDtemp * x_5_19_1;
                                        LOC2(store,10,28, STOREDIM, STOREDIM) += Ptempx * x_5_28_0 + WPtempx * x_5_28_1 +  3 * ABCDtemp * x_5_11_1;
                                        LOC2(store,10,29, STOREDIM, STOREDIM) += Ptempx * x_5_29_0 + WPtempx * x_5_29_1 + ABCDtemp * x_5_18_1;
                                        LOC2(store,10,30, STOREDIM, STOREDIM) += Ptempx * x_5_30_0 + WPtempx * x_5_30_1;
                                        LOC2(store,10,31, STOREDIM, STOREDIM) += Ptempx * x_5_31_0 + WPtempx * x_5_31_1;
                                        LOC2(store,10,32, STOREDIM, STOREDIM) += Ptempx * x_5_32_0 + WPtempx * x_5_32_1 +  4 * ABCDtemp * x_5_17_1;
                                        LOC2(store,10,33, STOREDIM, STOREDIM) += Ptempx * x_5_33_0 + WPtempx * x_5_33_1;
                                        LOC2(store,10,34, STOREDIM, STOREDIM) += Ptempx * x_5_34_0 + WPtempx * x_5_34_1;
                                        LOC2(store,11,20, STOREDIM, STOREDIM) += Ptempx * x_4_20_0 + WPtempx * x_4_20_1 + ABtemp * ( x_2_20_0 - CDcom * x_2_20_1) +  2 * ABCDtemp * x_4_12_1;
                                        LOC2(store,11,21, STOREDIM, STOREDIM) += Ptempx * x_4_21_0 + WPtempx * x_4_21_1 + ABtemp * ( x_2_21_0 - CDcom * x_2_21_1) +  2 * ABCDtemp * x_4_14_1;
                                        LOC2(store,11,22, STOREDIM, STOREDIM) += Ptempx * x_4_22_0 + WPtempx * x_4_22_1 + ABtemp * ( x_2_22_0 - CDcom * x_2_22_1);
                                        LOC2(store,11,23, STOREDIM, STOREDIM) += Ptempx * x_4_23_0 + WPtempx * x_4_23_1 + ABtemp * ( x_2_23_0 - CDcom * x_2_23_1) +  2 * ABCDtemp * x_4_10_1;
                                        LOC2(store,11,24, STOREDIM, STOREDIM) += Ptempx * x_4_24_0 + WPtempx * x_4_24_1 + ABtemp * ( x_2_24_0 - CDcom * x_2_24_1) + ABCDtemp * x_4_15_1;
                                        LOC2(store,11,25, STOREDIM, STOREDIM) += Ptempx * x_4_25_0 + WPtempx * x_4_25_1 + ABtemp * ( x_2_25_0 - CDcom * x_2_25_1) + ABCDtemp * x_4_16_1;
                                        LOC2(store,11,26, STOREDIM, STOREDIM) += Ptempx * x_4_26_0 + WPtempx * x_4_26_1 + ABtemp * ( x_2_26_0 - CDcom * x_2_26_1) +  3 * ABCDtemp * x_4_13_1;
                                        LOC2(store,11,27, STOREDIM, STOREDIM) += Ptempx * x_4_27_0 + WPtempx * x_4_27_1 + ABtemp * ( x_2_27_0 - CDcom * x_2_27_1) + ABCDtemp * x_4_19_1;
                                        LOC2(store,11,28, STOREDIM, STOREDIM) += Ptempx * x_4_28_0 + WPtempx * x_4_28_1 + ABtemp * ( x_2_28_0 - CDcom * x_2_28_1) +  3 * ABCDtemp * x_4_11_1;
                                        LOC2(store,11,29, STOREDIM, STOREDIM) += Ptempx * x_4_29_0 + WPtempx * x_4_29_1 + ABtemp * ( x_2_29_0 - CDcom * x_2_29_1) + ABCDtemp * x_4_18_1;
                                        LOC2(store,11,30, STOREDIM, STOREDIM) += Ptempx * x_4_30_0 + WPtempx * x_4_30_1 + ABtemp * ( x_2_30_0 - CDcom * x_2_30_1);
                                        LOC2(store,11,31, STOREDIM, STOREDIM) += Ptempx * x_4_31_0 + WPtempx * x_4_31_1 + ABtemp * ( x_2_31_0 - CDcom * x_2_31_1);
                                        LOC2(store,11,32, STOREDIM, STOREDIM) += Ptempx * x_4_32_0 + WPtempx * x_4_32_1 + ABtemp * ( x_2_32_0 - CDcom * x_2_32_1) +  4 * ABCDtemp * x_4_17_1;
                                        LOC2(store,11,33, STOREDIM, STOREDIM) += Ptempx * x_4_33_0 + WPtempx * x_4_33_1 + ABtemp * ( x_2_33_0 - CDcom * x_2_33_1);
                                        LOC2(store,11,34, STOREDIM, STOREDIM) += Ptempx * x_4_34_0 + WPtempx * x_4_34_1 + ABtemp * ( x_2_34_0 - CDcom * x_2_34_1);
                                        LOC2(store,12,20, STOREDIM, STOREDIM) += Ptempx * x_8_20_0 + WPtempx * x_8_20_1 +  2 * ABCDtemp * x_8_12_1;
                                        LOC2(store,12,21, STOREDIM, STOREDIM) += Ptempx * x_8_21_0 + WPtempx * x_8_21_1 +  2 * ABCDtemp * x_8_14_1;
                                        LOC2(store,12,22, STOREDIM, STOREDIM) += Ptempx * x_8_22_0 + WPtempx * x_8_22_1;
                                        LOC2(store,12,23, STOREDIM, STOREDIM) += Ptempx * x_8_23_0 + WPtempx * x_8_23_1 +  2 * ABCDtemp * x_8_10_1;
                                        LOC2(store,12,24, STOREDIM, STOREDIM) += Ptempx * x_8_24_0 + WPtempx * x_8_24_1 + ABCDtemp * x_8_15_1;
                                        LOC2(store,12,25, STOREDIM, STOREDIM) += Ptempx * x_8_25_0 + WPtempx * x_8_25_1 + ABCDtemp * x_8_16_1;
                                        LOC2(store,12,26, STOREDIM, STOREDIM) += Ptempx * x_8_26_0 + WPtempx * x_8_26_1 +  3 * ABCDtemp * x_8_13_1;
                                        LOC2(store,12,27, STOREDIM, STOREDIM) += Ptempx * x_8_27_0 + WPtempx * x_8_27_1 + ABCDtemp * x_8_19_1;
                                        LOC2(store,12,28, STOREDIM, STOREDIM) += Ptempx * x_8_28_0 + WPtempx * x_8_28_1 +  3 * ABCDtemp * x_8_11_1;
                                        LOC2(store,12,29, STOREDIM, STOREDIM) += Ptempx * x_8_29_0 + WPtempx * x_8_29_1 + ABCDtemp * x_8_18_1;
                                        LOC2(store,12,30, STOREDIM, STOREDIM) += Ptempx * x_8_30_0 + WPtempx * x_8_30_1;
                                        LOC2(store,12,31, STOREDIM, STOREDIM) += Ptempx * x_8_31_0 + WPtempx * x_8_31_1;
                                        LOC2(store,12,32, STOREDIM, STOREDIM) += Ptempx * x_8_32_0 + WPtempx * x_8_32_1 +  4 * ABCDtemp * x_8_17_1;
                                        LOC2(store,12,33, STOREDIM, STOREDIM) += Ptempx * x_8_33_0 + WPtempx * x_8_33_1;
                                        LOC2(store,12,34, STOREDIM, STOREDIM) += Ptempx * x_8_34_0 + WPtempx * x_8_34_1;
                                        LOC2(store,13,20, STOREDIM, STOREDIM) += Ptempx * x_6_20_0 + WPtempx * x_6_20_1 + ABtemp * ( x_3_20_0 - CDcom * x_3_20_1) +  2 * ABCDtemp * x_6_12_1;
                                        LOC2(store,13,21, STOREDIM, STOREDIM) += Ptempx * x_6_21_0 + WPtempx * x_6_21_1 + ABtemp * ( x_3_21_0 - CDcom * x_3_21_1) +  2 * ABCDtemp * x_6_14_1;
                                        LOC2(store,13,22, STOREDIM, STOREDIM) += Ptempx * x_6_22_0 + WPtempx * x_6_22_1 + ABtemp * ( x_3_22_0 - CDcom * x_3_22_1);
                                        LOC2(store,13,23, STOREDIM, STOREDIM) += Ptempx * x_6_23_0 + WPtempx * x_6_23_1 + ABtemp * ( x_3_23_0 - CDcom * x_3_23_1) +  2 * ABCDtemp * x_6_10_1;
                                        LOC2(store,13,24, STOREDIM, STOREDIM) += Ptempx * x_6_24_0 + WPtempx * x_6_24_1 + ABtemp * ( x_3_24_0 - CDcom * x_3_24_1) + ABCDtemp * x_6_15_1;
                                        LOC2(store,13,25, STOREDIM, STOREDIM) += Ptempx * x_6_25_0 + WPtempx * x_6_25_1 + ABtemp * ( x_3_25_0 - CDcom * x_3_25_1) + ABCDtemp * x_6_16_1;
                                        LOC2(store,13,26, STOREDIM, STOREDIM) += Ptempx * x_6_26_0 + WPtempx * x_6_26_1 + ABtemp * ( x_3_26_0 - CDcom * x_3_26_1) +  3 * ABCDtemp * x_6_13_1;
                                        LOC2(store,13,27, STOREDIM, STOREDIM) += Ptempx * x_6_27_0 + WPtempx * x_6_27_1 + ABtemp * ( x_3_27_0 - CDcom * x_3_27_1) + ABCDtemp * x_6_19_1;
                                        LOC2(store,13,28, STOREDIM, STOREDIM) += Ptempx * x_6_28_0 + WPtempx * x_6_28_1 + ABtemp * ( x_3_28_0 - CDcom * x_3_28_1) +  3 * ABCDtemp * x_6_11_1;
                                        LOC2(store,13,29, STOREDIM, STOREDIM) += Ptempx * x_6_29_0 + WPtempx * x_6_29_1 + ABtemp * ( x_3_29_0 - CDcom * x_3_29_1) + ABCDtemp * x_6_18_1;
                                        LOC2(store,13,30, STOREDIM, STOREDIM) += Ptempx * x_6_30_0 + WPtempx * x_6_30_1 + ABtemp * ( x_3_30_0 - CDcom * x_3_30_1);
                                        LOC2(store,13,31, STOREDIM, STOREDIM) += Ptempx * x_6_31_0 + WPtempx * x_6_31_1 + ABtemp * ( x_3_31_0 - CDcom * x_3_31_1);
                                        LOC2(store,13,32, STOREDIM, STOREDIM) += Ptempx * x_6_32_0 + WPtempx * x_6_32_1 + ABtemp * ( x_3_32_0 - CDcom * x_3_32_1) +  4 * ABCDtemp * x_6_17_1;
                                        LOC2(store,13,33, STOREDIM, STOREDIM) += Ptempx * x_6_33_0 + WPtempx * x_6_33_1 + ABtemp * ( x_3_33_0 - CDcom * x_3_33_1);
                                        LOC2(store,13,34, STOREDIM, STOREDIM) += Ptempx * x_6_34_0 + WPtempx * x_6_34_1 + ABtemp * ( x_3_34_0 - CDcom * x_3_34_1);
                                        LOC2(store,14,20, STOREDIM, STOREDIM) += Ptempx * x_9_20_0 + WPtempx * x_9_20_1 +  2 * ABCDtemp * x_9_12_1;
                                        LOC2(store,14,21, STOREDIM, STOREDIM) += Ptempx * x_9_21_0 + WPtempx * x_9_21_1 +  2 * ABCDtemp * x_9_14_1;
                                        LOC2(store,14,22, STOREDIM, STOREDIM) += Ptempx * x_9_22_0 + WPtempx * x_9_22_1;
                                        LOC2(store,14,23, STOREDIM, STOREDIM) += Ptempx * x_9_23_0 + WPtempx * x_9_23_1 +  2 * ABCDtemp * x_9_10_1;
                                        LOC2(store,14,24, STOREDIM, STOREDIM) += Ptempx * x_9_24_0 + WPtempx * x_9_24_1 + ABCDtemp * x_9_15_1;
                                        LOC2(store,14,25, STOREDIM, STOREDIM) += Ptempx * x_9_25_0 + WPtempx * x_9_25_1 + ABCDtemp * x_9_16_1;
                                        LOC2(store,14,26, STOREDIM, STOREDIM) += Ptempx * x_9_26_0 + WPtempx * x_9_26_1 +  3 * ABCDtemp * x_9_13_1;
                                        LOC2(store,14,27, STOREDIM, STOREDIM) += Ptempx * x_9_27_0 + WPtempx * x_9_27_1 + ABCDtemp * x_9_19_1;
                                        LOC2(store,14,28, STOREDIM, STOREDIM) += Ptempx * x_9_28_0 + WPtempx * x_9_28_1 +  3 * ABCDtemp * x_9_11_1;
                                        LOC2(store,14,29, STOREDIM, STOREDIM) += Ptempx * x_9_29_0 + WPtempx * x_9_29_1 + ABCDtemp * x_9_18_1;
                                        LOC2(store,14,30, STOREDIM, STOREDIM) += Ptempx * x_9_30_0 + WPtempx * x_9_30_1;
                                        LOC2(store,14,31, STOREDIM, STOREDIM) += Ptempx * x_9_31_0 + WPtempx * x_9_31_1;
                                        LOC2(store,14,32, STOREDIM, STOREDIM) += Ptempx * x_9_32_0 + WPtempx * x_9_32_1 +  4 * ABCDtemp * x_9_17_1;
                                        LOC2(store,14,33, STOREDIM, STOREDIM) += Ptempx * x_9_33_0 + WPtempx * x_9_33_1;
                                        LOC2(store,14,34, STOREDIM, STOREDIM) += Ptempx * x_9_34_0 + WPtempx * x_9_34_1;
                                        LOC2(store,15,20, STOREDIM, STOREDIM) += Ptempy * x_5_20_0 + WPtempy * x_5_20_1 + ABtemp * ( x_3_20_0 - CDcom * x_3_20_1) +  2 * ABCDtemp * x_5_11_1;
                                        LOC2(store,15,21, STOREDIM, STOREDIM) += Ptempy * x_5_21_0 + WPtempy * x_5_21_1 + ABtemp * ( x_3_21_0 - CDcom * x_3_21_1);
                                        LOC2(store,15,22, STOREDIM, STOREDIM) += Ptempy * x_5_22_0 + WPtempy * x_5_22_1 + ABtemp * ( x_3_22_0 - CDcom * x_3_22_1) +  2 * ABCDtemp * x_5_16_1;
                                        LOC2(store,15,23, STOREDIM, STOREDIM) += Ptempy * x_5_23_0 + WPtempy * x_5_23_1 + ABtemp * ( x_3_23_0 - CDcom * x_3_23_1) + ABCDtemp * x_5_13_1;
                                        LOC2(store,15,24, STOREDIM, STOREDIM) += Ptempy * x_5_24_0 + WPtempy * x_5_24_1 + ABtemp * ( x_3_24_0 - CDcom * x_3_24_1) +  2 * ABCDtemp * x_5_10_1;
                                        LOC2(store,15,25, STOREDIM, STOREDIM) += Ptempy * x_5_25_0 + WPtempy * x_5_25_1 + ABtemp * ( x_3_25_0 - CDcom * x_3_25_1) + ABCDtemp * x_5_14_1;
                                        LOC2(store,15,26, STOREDIM, STOREDIM) += Ptempy * x_5_26_0 + WPtempy * x_5_26_1 + ABtemp * ( x_3_26_0 - CDcom * x_3_26_1);
                                        LOC2(store,15,27, STOREDIM, STOREDIM) += Ptempy * x_5_27_0 + WPtempy * x_5_27_1 + ABtemp * ( x_3_27_0 - CDcom * x_3_27_1);
                                        LOC2(store,15,28, STOREDIM, STOREDIM) += Ptempy * x_5_28_0 + WPtempy * x_5_28_1 + ABtemp * ( x_3_28_0 - CDcom * x_3_28_1) + ABCDtemp * x_5_17_1;
                                        LOC2(store,15,29, STOREDIM, STOREDIM) += Ptempy * x_5_29_0 + WPtempy * x_5_29_1 + ABtemp * ( x_3_29_0 - CDcom * x_3_29_1) +  3 * ABCDtemp * x_5_12_1;
                                        LOC2(store,15,30, STOREDIM, STOREDIM) += Ptempy * x_5_30_0 + WPtempy * x_5_30_1 + ABtemp * ( x_3_30_0 - CDcom * x_3_30_1) +  3 * ABCDtemp * x_5_15_1;
                                        LOC2(store,15,31, STOREDIM, STOREDIM) += Ptempy * x_5_31_0 + WPtempy * x_5_31_1 + ABtemp * ( x_3_31_0 - CDcom * x_3_31_1) + ABCDtemp * x_5_19_1;
                                        LOC2(store,15,32, STOREDIM, STOREDIM) += Ptempy * x_5_32_0 + WPtempy * x_5_32_1 + ABtemp * ( x_3_32_0 - CDcom * x_3_32_1);
                                        LOC2(store,15,33, STOREDIM, STOREDIM) += Ptempy * x_5_33_0 + WPtempy * x_5_33_1 + ABtemp * ( x_3_33_0 - CDcom * x_3_33_1) +  4 * ABCDtemp * x_5_18_1;
                                        LOC2(store,15,34, STOREDIM, STOREDIM) += Ptempy * x_5_34_0 + WPtempy * x_5_34_1 + ABtemp * ( x_3_34_0 - CDcom * x_3_34_1);
                                        LOC2(store,16,20, STOREDIM, STOREDIM) += Ptempy * x_9_20_0 + WPtempy * x_9_20_1 +  2 * ABCDtemp * x_9_11_1;
                                        LOC2(store,16,21, STOREDIM, STOREDIM) += Ptempy * x_9_21_0 + WPtempy * x_9_21_1;
                                        LOC2(store,16,22, STOREDIM, STOREDIM) += Ptempy * x_9_22_0 + WPtempy * x_9_22_1 +  2 * ABCDtemp * x_9_16_1;
                                        LOC2(store,16,23, STOREDIM, STOREDIM) += Ptempy * x_9_23_0 + WPtempy * x_9_23_1 + ABCDtemp * x_9_13_1;
                                        LOC2(store,16,24, STOREDIM, STOREDIM) += Ptempy * x_9_24_0 + WPtempy * x_9_24_1 +  2 * ABCDtemp * x_9_10_1;
                                        LOC2(store,16,25, STOREDIM, STOREDIM) += Ptempy * x_9_25_0 + WPtempy * x_9_25_1 + ABCDtemp * x_9_14_1;
                                        LOC2(store,16,26, STOREDIM, STOREDIM) += Ptempy * x_9_26_0 + WPtempy * x_9_26_1;
                                        LOC2(store,16,27, STOREDIM, STOREDIM) += Ptempy * x_9_27_0 + WPtempy * x_9_27_1;
                                        LOC2(store,16,28, STOREDIM, STOREDIM) += Ptempy * x_9_28_0 + WPtempy * x_9_28_1 + ABCDtemp * x_9_17_1;
                                        LOC2(store,16,29, STOREDIM, STOREDIM) += Ptempy * x_9_29_0 + WPtempy * x_9_29_1 +  3 * ABCDtemp * x_9_12_1;
                                        LOC2(store,16,30, STOREDIM, STOREDIM) += Ptempy * x_9_30_0 + WPtempy * x_9_30_1 +  3 * ABCDtemp * x_9_15_1;
                                        LOC2(store,16,31, STOREDIM, STOREDIM) += Ptempy * x_9_31_0 + WPtempy * x_9_31_1 + ABCDtemp * x_9_19_1;
                                        LOC2(store,16,32, STOREDIM, STOREDIM) += Ptempy * x_9_32_0 + WPtempy * x_9_32_1;
                                        LOC2(store,16,33, STOREDIM, STOREDIM) += Ptempy * x_9_33_0 + WPtempy * x_9_33_1 +  4 * ABCDtemp * x_9_18_1;
                                        LOC2(store,16,34, STOREDIM, STOREDIM) += Ptempy * x_9_34_0 + WPtempy * x_9_34_1;
                                        LOC2(store,17,20, STOREDIM, STOREDIM) += Ptempx * x_7_20_0 + WPtempx * x_7_20_1 +  2 * ABtemp * ( x_1_20_0 - CDcom * x_1_20_1) +  2 * ABCDtemp * x_7_12_1;
                                        LOC2(store,17,21, STOREDIM, STOREDIM) += Ptempx * x_7_21_0 + WPtempx * x_7_21_1 +  2 * ABtemp * ( x_1_21_0 - CDcom * x_1_21_1) +  2 * ABCDtemp * x_7_14_1;
                                        LOC2(store,17,22, STOREDIM, STOREDIM) += Ptempx * x_7_22_0 + WPtempx * x_7_22_1 +  2 * ABtemp * ( x_1_22_0 - CDcom * x_1_22_1);
                                        LOC2(store,17,23, STOREDIM, STOREDIM) += Ptempx * x_7_23_0 + WPtempx * x_7_23_1 +  2 * ABtemp * ( x_1_23_0 - CDcom * x_1_23_1) +  2 * ABCDtemp * x_7_10_1;
                                        LOC2(store,17,24, STOREDIM, STOREDIM) += Ptempx * x_7_24_0 + WPtempx * x_7_24_1 +  2 * ABtemp * ( x_1_24_0 - CDcom * x_1_24_1) + ABCDtemp * x_7_15_1;
                                        LOC2(store,17,25, STOREDIM, STOREDIM) += Ptempx * x_7_25_0 + WPtempx * x_7_25_1 +  2 * ABtemp * ( x_1_25_0 - CDcom * x_1_25_1) + ABCDtemp * x_7_16_1;
                                        LOC2(store,17,26, STOREDIM, STOREDIM) += Ptempx * x_7_26_0 + WPtempx * x_7_26_1 +  2 * ABtemp * ( x_1_26_0 - CDcom * x_1_26_1) +  3 * ABCDtemp * x_7_13_1;
                                        LOC2(store,17,27, STOREDIM, STOREDIM) += Ptempx * x_7_27_0 + WPtempx * x_7_27_1 +  2 * ABtemp * ( x_1_27_0 - CDcom * x_1_27_1) + ABCDtemp * x_7_19_1;
                                        LOC2(store,17,28, STOREDIM, STOREDIM) += Ptempx * x_7_28_0 + WPtempx * x_7_28_1 +  2 * ABtemp * ( x_1_28_0 - CDcom * x_1_28_1) +  3 * ABCDtemp * x_7_11_1;
                                        LOC2(store,17,29, STOREDIM, STOREDIM) += Ptempx * x_7_29_0 + WPtempx * x_7_29_1 +  2 * ABtemp * ( x_1_29_0 - CDcom * x_1_29_1) + ABCDtemp * x_7_18_1;
                                        LOC2(store,17,30, STOREDIM, STOREDIM) += Ptempx * x_7_30_0 + WPtempx * x_7_30_1 +  2 * ABtemp * ( x_1_30_0 - CDcom * x_1_30_1);
                                        LOC2(store,17,31, STOREDIM, STOREDIM) += Ptempx * x_7_31_0 + WPtempx * x_7_31_1 +  2 * ABtemp * ( x_1_31_0 - CDcom * x_1_31_1);
                                        LOC2(store,17,32, STOREDIM, STOREDIM) += Ptempx * x_7_32_0 + WPtempx * x_7_32_1 +  2 * ABtemp * ( x_1_32_0 - CDcom * x_1_32_1) +  4 * ABCDtemp * x_7_17_1;
                                        LOC2(store,17,33, STOREDIM, STOREDIM) += Ptempx * x_7_33_0 + WPtempx * x_7_33_1 +  2 * ABtemp * ( x_1_33_0 - CDcom * x_1_33_1);
                                        LOC2(store,17,34, STOREDIM, STOREDIM) += Ptempx * x_7_34_0 + WPtempx * x_7_34_1 +  2 * ABtemp * ( x_1_34_0 - CDcom * x_1_34_1);
                                        LOC2(store,18,20, STOREDIM, STOREDIM) += Ptempy * x_8_20_0 + WPtempy * x_8_20_1 +  2 * ABtemp * ( x_2_20_0 - CDcom * x_2_20_1) +  2 * ABCDtemp * x_8_11_1;
                                        LOC2(store,18,21, STOREDIM, STOREDIM) += Ptempy * x_8_21_0 + WPtempy * x_8_21_1 +  2 * ABtemp * ( x_2_21_0 - CDcom * x_2_21_1);
                                        LOC2(store,18,22, STOREDIM, STOREDIM) += Ptempy * x_8_22_0 + WPtempy * x_8_22_1 +  2 * ABtemp * ( x_2_22_0 - CDcom * x_2_22_1) +  2 * ABCDtemp * x_8_16_1;
                                        LOC2(store,18,23, STOREDIM, STOREDIM) += Ptempy * x_8_23_0 + WPtempy * x_8_23_1 +  2 * ABtemp * ( x_2_23_0 - CDcom * x_2_23_1) + ABCDtemp * x_8_13_1;
                                        LOC2(store,18,24, STOREDIM, STOREDIM) += Ptempy * x_8_24_0 + WPtempy * x_8_24_1 +  2 * ABtemp * ( x_2_24_0 - CDcom * x_2_24_1) +  2 * ABCDtemp * x_8_10_1;
                                        LOC2(store,18,25, STOREDIM, STOREDIM) += Ptempy * x_8_25_0 + WPtempy * x_8_25_1 +  2 * ABtemp * ( x_2_25_0 - CDcom * x_2_25_1) + ABCDtemp * x_8_14_1;
                                        LOC2(store,18,26, STOREDIM, STOREDIM) += Ptempy * x_8_26_0 + WPtempy * x_8_26_1 +  2 * ABtemp * ( x_2_26_0 - CDcom * x_2_26_1);
                                        LOC2(store,18,27, STOREDIM, STOREDIM) += Ptempy * x_8_27_0 + WPtempy * x_8_27_1 +  2 * ABtemp * ( x_2_27_0 - CDcom * x_2_27_1);
                                        LOC2(store,18,28, STOREDIM, STOREDIM) += Ptempy * x_8_28_0 + WPtempy * x_8_28_1 +  2 * ABtemp * ( x_2_28_0 - CDcom * x_2_28_1) + ABCDtemp * x_8_17_1;
                                        LOC2(store,18,29, STOREDIM, STOREDIM) += Ptempy * x_8_29_0 + WPtempy * x_8_29_1 +  2 * ABtemp * ( x_2_29_0 - CDcom * x_2_29_1) +  3 * ABCDtemp * x_8_12_1;
                                        LOC2(store,18,30, STOREDIM, STOREDIM) += Ptempy * x_8_30_0 + WPtempy * x_8_30_1 +  2 * ABtemp * ( x_2_30_0 - CDcom * x_2_30_1) +  3 * ABCDtemp * x_8_15_1;
                                        LOC2(store,18,31, STOREDIM, STOREDIM) += Ptempy * x_8_31_0 + WPtempy * x_8_31_1 +  2 * ABtemp * ( x_2_31_0 - CDcom * x_2_31_1) + ABCDtemp * x_8_19_1;
                                        LOC2(store,18,32, STOREDIM, STOREDIM) += Ptempy * x_8_32_0 + WPtempy * x_8_32_1 +  2 * ABtemp * ( x_2_32_0 - CDcom * x_2_32_1);
                                        LOC2(store,18,33, STOREDIM, STOREDIM) += Ptempy * x_8_33_0 + WPtempy * x_8_33_1 +  2 * ABtemp * ( x_2_33_0 - CDcom * x_2_33_1) +  4 * ABCDtemp * x_8_18_1;
                                        LOC2(store,18,34, STOREDIM, STOREDIM) += Ptempy * x_8_34_0 + WPtempy * x_8_34_1 +  2 * ABtemp * ( x_2_34_0 - CDcom * x_2_34_1);
                                        LOC2(store,19,20, STOREDIM, STOREDIM) += Ptempz * x_9_20_0 + WPtempz * x_9_20_1 +  2 * ABtemp * ( x_3_20_0 - CDcom * x_3_20_1);
                                        LOC2(store,19,21, STOREDIM, STOREDIM) += Ptempz * x_9_21_0 + WPtempz * x_9_21_1 +  2 * ABtemp * ( x_3_21_0 - CDcom * x_3_21_1) +  2 * ABCDtemp * x_9_13_1;
                                        LOC2(store,19,22, STOREDIM, STOREDIM) += Ptempz * x_9_22_0 + WPtempz * x_9_22_1 +  2 * ABtemp * ( x_3_22_0 - CDcom * x_3_22_1) +  2 * ABCDtemp * x_9_15_1;
                                        LOC2(store,19,23, STOREDIM, STOREDIM) += Ptempz * x_9_23_0 + WPtempz * x_9_23_1 +  2 * ABtemp * ( x_3_23_0 - CDcom * x_3_23_1) + ABCDtemp * x_9_11_1;
                                        LOC2(store,19,24, STOREDIM, STOREDIM) += Ptempz * x_9_24_0 + WPtempz * x_9_24_1 +  2 * ABtemp * ( x_3_24_0 - CDcom * x_3_24_1) + ABCDtemp * x_9_12_1;
                                        LOC2(store,19,25, STOREDIM, STOREDIM) += Ptempz * x_9_25_0 + WPtempz * x_9_25_1 +  2 * ABtemp * ( x_3_25_0 - CDcom * x_3_25_1) +  2 * ABCDtemp * x_9_10_1;
                                        LOC2(store,19,26, STOREDIM, STOREDIM) += Ptempz * x_9_26_0 + WPtempz * x_9_26_1 +  2 * ABtemp * ( x_3_26_0 - CDcom * x_3_26_1) + ABCDtemp * x_9_17_1;
                                        LOC2(store,19,27, STOREDIM, STOREDIM) += Ptempz * x_9_27_0 + WPtempz * x_9_27_1 +  2 * ABtemp * ( x_3_27_0 - CDcom * x_3_27_1) +  3 * ABCDtemp * x_9_14_1;
                                        LOC2(store,19,28, STOREDIM, STOREDIM) += Ptempz * x_9_28_0 + WPtempz * x_9_28_1 +  2 * ABtemp * ( x_3_28_0 - CDcom * x_3_28_1);
                                        LOC2(store,19,29, STOREDIM, STOREDIM) += Ptempz * x_9_29_0 + WPtempz * x_9_29_1 +  2 * ABtemp * ( x_3_29_0 - CDcom * x_3_29_1);
                                        LOC2(store,19,30, STOREDIM, STOREDIM) += Ptempz * x_9_30_0 + WPtempz * x_9_30_1 +  2 * ABtemp * ( x_3_30_0 - CDcom * x_3_30_1) + ABCDtemp * x_9_18_1;
                                        LOC2(store,19,31, STOREDIM, STOREDIM) += Ptempz * x_9_31_0 + WPtempz * x_9_31_1 +  2 * ABtemp * ( x_3_31_0 - CDcom * x_3_31_1) +  3 * ABCDtemp * x_9_16_1;
                                        LOC2(store,19,32, STOREDIM, STOREDIM) += Ptempz * x_9_32_0 + WPtempz * x_9_32_1 +  2 * ABtemp * ( x_3_32_0 - CDcom * x_3_32_1);
                                        LOC2(store,19,33, STOREDIM, STOREDIM) += Ptempz * x_9_33_0 + WPtempz * x_9_33_1 +  2 * ABtemp * ( x_3_33_0 - CDcom * x_3_33_1);
                                        LOC2(store,19,34, STOREDIM, STOREDIM) += Ptempz * x_9_34_0 + WPtempz * x_9_34_1 +  2 * ABtemp * ( x_3_34_0 - CDcom * x_3_34_1) +  4 * ABCDtemp * x_9_19_1;
                                    }
                                    
                                }
                            }
                        }
                    }
                }
            }
            if (I+J>=1){
                //PSSS(0, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);
                QUICKDouble x_1_0_0 = Ptempx * VY( 0, 0, 0) + WPtempx * VY( 0, 0, 1);
                QUICKDouble x_2_0_0 = Ptempy * VY( 0, 0, 0) + WPtempy * VY( 0, 0, 1);
                QUICKDouble x_3_0_0 = Ptempz * VY( 0, 0, 0) + WPtempz * VY( 0, 0, 1);
                
                LOC2(store, 1, 0, STOREDIM, STOREDIM) += x_1_0_0;
                LOC2(store, 2, 0, STOREDIM, STOREDIM) += x_2_0_0;
                LOC2(store, 3, 0, STOREDIM, STOREDIM) += x_3_0_0;
                
                if (I+J>=2){
                    
                    //PSSS(1, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);
                    QUICKDouble x_1_0_1 = Ptempx * VY( 0, 0, 1) + WPtempx * VY( 0, 0, 2);
                    QUICKDouble x_2_0_1 = Ptempy * VY( 0, 0, 1) + WPtempy * VY( 0, 0, 2);
                    QUICKDouble x_3_0_1 = Ptempz * VY( 0, 0, 1) + WPtempz * VY( 0, 0, 2);
                    
                    //DSSS(0, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz, ABtemp, CDcom);
                    QUICKDouble x_4_0_0 = Ptempx * x_2_0_0 + WPtempx * x_2_0_1;
                    QUICKDouble x_5_0_0 = Ptempy * x_3_0_0 + WPtempy * x_3_0_1;
                    QUICKDouble x_6_0_0 = Ptempx * x_3_0_0 + WPtempx * x_3_0_1;
                    
                    QUICKDouble x_7_0_0 = Ptempx * x_1_0_0 + WPtempx * x_1_0_1+ ABtemp*(VY( 0, 0, 0) - CDcom * VY( 0, 0, 1));
                    QUICKDouble x_8_0_0 = Ptempy * x_2_0_0 + WPtempy * x_2_0_1+ ABtemp*(VY( 0, 0, 0) - CDcom * VY( 0, 0, 1));
                    QUICKDouble x_9_0_0 = Ptempz * x_3_0_0 + WPtempz * x_3_0_1+ ABtemp*(VY( 0, 0, 0) - CDcom * VY( 0, 0, 1));
                    
                    LOC2(store, 4, 0, STOREDIM, STOREDIM) += x_4_0_0;
                    LOC2(store, 5, 0, STOREDIM, STOREDIM) += x_5_0_0;
                    LOC2(store, 6, 0, STOREDIM, STOREDIM) += x_6_0_0;
                    LOC2(store, 7, 0, STOREDIM, STOREDIM) += x_7_0_0;
                    LOC2(store, 8, 0, STOREDIM, STOREDIM) += x_8_0_0;
                    LOC2(store, 9, 0, STOREDIM, STOREDIM) += x_9_0_0;
                    
                    //PSSS(2, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);
                    QUICKDouble x_1_0_2 = Ptempx * VY( 0, 0, 2) + WPtempx * VY( 0, 0, 3);
                    QUICKDouble x_2_0_2 = Ptempy * VY( 0, 0, 2) + WPtempy * VY( 0, 0, 3);
                    QUICKDouble x_3_0_2 = Ptempz * VY( 0, 0, 2) + WPtempz * VY( 0, 0, 3);
                    
                    //DSSS(1, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz, ABtemp, CDcom);
                    QUICKDouble x_4_0_1 = Ptempx * x_2_0_1 + WPtempx * x_2_0_2;
                    QUICKDouble x_5_0_1 = Ptempy * x_3_0_1 + WPtempy * x_3_0_2;
                    QUICKDouble x_6_0_1 = Ptempx * x_3_0_1 + WPtempx * x_3_0_2;
                    
                    QUICKDouble x_7_0_1 = Ptempx * x_1_0_1 + WPtempx * x_1_0_2+ ABtemp*(VY( 0, 0, 1) - CDcom * VY( 0, 0, 2));
                    QUICKDouble x_8_0_1 = Ptempy * x_2_0_1 + WPtempy * x_2_0_2+ ABtemp*(VY( 0, 0, 1) - CDcom * VY( 0, 0, 2));
                    QUICKDouble x_9_0_1 = Ptempz * x_3_0_1 + WPtempz * x_3_0_2+ ABtemp*(VY( 0, 0, 1) - CDcom * VY( 0, 0, 2));
                    
                    if (I+J>=2 && K+L >=1) {
                        
                        //DSPS(0, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz, ABCDtemp);
                        
                        LOC2(store, 4, 1, STOREDIM, STOREDIM) += Qtempx * x_4_0_0 + WQtempx * x_4_0_1 + ABCDtemp * x_2_0_1;
                        LOC2(store, 4, 2, STOREDIM, STOREDIM) += Qtempy * x_4_0_0 + WQtempy * x_4_0_1 + ABCDtemp * x_1_0_1;
                        LOC2(store, 4, 3, STOREDIM, STOREDIM) += Qtempz * x_4_0_0 + WQtempz * x_4_0_1;
                        
                        LOC2(store, 5, 1, STOREDIM, STOREDIM) += Qtempx * x_5_0_0 + WQtempx * x_5_0_1;
                        LOC2(store, 5, 2, STOREDIM, STOREDIM) += Qtempy * x_5_0_0 + WQtempy * x_5_0_1 + ABCDtemp * x_3_0_1;
                        LOC2(store, 5, 3, STOREDIM, STOREDIM) += Qtempz * x_5_0_0 + WQtempz * x_5_0_1 + ABCDtemp * x_2_0_1;
                        
                        LOC2(store, 6, 1, STOREDIM, STOREDIM) += Qtempx * x_6_0_0 + WQtempx * x_6_0_1 + ABCDtemp * x_3_0_1;
                        LOC2(store, 6, 2, STOREDIM, STOREDIM) += Qtempy * x_6_0_0 + WQtempy * x_6_0_1;
                        LOC2(store, 6, 3, STOREDIM, STOREDIM) += Qtempz * x_6_0_0 + WQtempz * x_6_0_1 + ABCDtemp * x_1_0_1;
                        
                        LOC2(store, 7, 1, STOREDIM, STOREDIM) += Qtempx * x_7_0_0 + WQtempx * x_7_0_1 + ABCDtemp * x_1_0_1 * 2;
                        LOC2(store, 7, 2, STOREDIM, STOREDIM) += Qtempy * x_7_0_0 + WQtempy * x_7_0_1;
                        LOC2(store, 7, 3, STOREDIM, STOREDIM) += Qtempz * x_7_0_0 + WQtempz * x_7_0_1;
                        
                        LOC2(store, 8, 1, STOREDIM, STOREDIM) += Qtempx * x_8_0_0 + WQtempx * x_8_0_1;
                        LOC2(store, 8, 2, STOREDIM, STOREDIM) += Qtempy * x_8_0_0 + WQtempy * x_8_0_1 + ABCDtemp * x_2_0_1 * 2;
                        LOC2(store, 8, 3, STOREDIM, STOREDIM) += Qtempz * x_8_0_0 + WQtempz * x_8_0_1;
                        
                        LOC2(store, 9, 1, STOREDIM, STOREDIM) += Qtempx * x_9_0_0 + WQtempx * x_9_0_1;
                        LOC2(store, 9, 2, STOREDIM, STOREDIM) += Qtempy * x_9_0_0 + WQtempy * x_9_0_1;
                        LOC2(store, 9, 3, STOREDIM, STOREDIM) += Qtempz * x_9_0_0 + WQtempz * x_9_0_1 + ABCDtemp * x_3_0_1 * 2;    
                        
                    }
                    if (I+J>=3){
                        
                        //SSPS(2, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);
                        QUICKDouble x_0_1_2 = Qtempx * VY( 0, 0, 2) + WQtempx * VY( 0, 0, 3);
                        QUICKDouble x_0_2_2 = Qtempy * VY( 0, 0, 2) + WQtempy * VY( 0, 0, 3);
                        QUICKDouble x_0_3_2 = Qtempz * VY( 0, 0, 2) + WQtempz * VY( 0, 0, 3);
                        
                        //SSPS(3, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);
                        QUICKDouble x_0_1_3 = Qtempx * VY( 0, 0, 3) + WQtempx * VY( 0, 0, 4);
                        QUICKDouble x_0_2_3 = Qtempy * VY( 0, 0, 3) + WQtempy * VY( 0, 0, 4);
                        QUICKDouble x_0_3_3 = Qtempz * VY( 0, 0, 3) + WQtempz * VY( 0, 0, 4);
                        
                        //FSSS(0, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz, ABtemp, CDcom);
                        
                        QUICKDouble x_10_0_0 = Ptempx * x_5_0_0 + WPtempx * x_5_0_1;
                        QUICKDouble x_11_0_0 = Ptempx * x_4_0_0 + WPtempx * x_4_0_1 + ABtemp * ( x_2_0_0 - CDcom * x_2_0_1);
                        QUICKDouble x_12_0_0 = Ptempx * x_8_0_0 + WPtempx * x_8_0_1;
                        QUICKDouble x_13_0_0 = Ptempx * x_6_0_0 + WPtempx * x_6_0_1 + ABtemp * ( x_3_0_0 - CDcom * x_3_0_1);
                        QUICKDouble x_14_0_0 = Ptempx * x_9_0_0 + WPtempx * x_9_0_1;
                        QUICKDouble x_15_0_0 = Ptempy * x_5_0_0 + WPtempy * x_5_0_1 + ABtemp * ( x_3_0_0 - CDcom * x_3_0_1);
                        QUICKDouble x_16_0_0 = Ptempy * x_9_0_0 + WPtempy * x_9_0_1;
                        QUICKDouble x_17_0_0 = Ptempx * x_7_0_0 + WPtempx * x_7_0_1 +  2 * ABtemp * ( x_1_0_0 - CDcom * x_1_0_1);
                        QUICKDouble x_18_0_0 = Ptempy * x_8_0_0 + WPtempy * x_8_0_1 +  2 * ABtemp * ( x_2_0_0 - CDcom * x_2_0_1);
                        QUICKDouble x_19_0_0 = Ptempz * x_9_0_0 + WPtempz * x_9_0_1 +  2 * ABtemp * ( x_3_0_0 - CDcom * x_3_0_1);
                        
                        LOC2(store,10, 0, STOREDIM, STOREDIM) += x_10_0_0;
                        LOC2(store,11, 0, STOREDIM, STOREDIM) += x_11_0_0;
                        LOC2(store,12, 0, STOREDIM, STOREDIM) += x_12_0_0;
                        LOC2(store,13, 0, STOREDIM, STOREDIM) += x_13_0_0;
                        LOC2(store,14, 0, STOREDIM, STOREDIM) += x_14_0_0;
                        LOC2(store,15, 0, STOREDIM, STOREDIM) += x_15_0_0;
                        LOC2(store,16, 0, STOREDIM, STOREDIM) += x_16_0_0;
                        LOC2(store,17, 0, STOREDIM, STOREDIM) += x_17_0_0;
                        LOC2(store,18, 0, STOREDIM, STOREDIM) += x_18_0_0;
                        LOC2(store,19, 0, STOREDIM, STOREDIM) += x_19_0_0;
                        
                        //SSPS(4, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);
                        QUICKDouble x_0_1_4 = Qtempx * VY( 0, 0, 4) + WQtempx * VY( 0, 0, 5);
                        QUICKDouble x_0_2_4 = Qtempy * VY( 0, 0, 4) + WQtempy * VY( 0, 0, 5);
                        QUICKDouble x_0_3_4 = Qtempz * VY( 0, 0, 4) + WQtempz * VY( 0, 0, 5);
                        
                        //SSDS(2, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz, CDtemp, ABcom);
                        
                        QUICKDouble x_0_4_2 = Qtempx * x_0_2_2 + WQtempx * x_0_2_3;
                        QUICKDouble x_0_5_2 = Qtempy * x_0_3_2 + WQtempy * x_0_3_3;
                        QUICKDouble x_0_6_2 = Qtempx * x_0_3_2 + WQtempx * x_0_3_3;
                        
                        QUICKDouble x_0_7_2 = Qtempx * x_0_1_2 + WQtempx * x_0_1_3+ CDtemp*(VY( 0, 0, 2) - ABcom * VY( 0, 0, 3));
                        QUICKDouble x_0_8_2 = Qtempy * x_0_2_2 + WQtempy * x_0_2_3+ CDtemp*(VY( 0, 0, 2) - ABcom * VY( 0, 0, 3));
                        QUICKDouble x_0_9_2 = Qtempz * x_0_3_2 + WQtempz * x_0_3_3+ CDtemp*(VY( 0, 0, 2) - ABcom * VY( 0, 0, 3));
                        
                        
                        //SSDS(3, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz, CDtemp, ABcom);
                        
                        QUICKDouble x_0_4_3 = Qtempx * x_0_2_3 + WQtempx * x_0_2_4;
                        QUICKDouble x_0_5_3 = Qtempy * x_0_3_3 + WQtempy * x_0_3_4;
                        QUICKDouble x_0_6_3 = Qtempx * x_0_3_3 + WQtempx * x_0_3_4;
                        
                        QUICKDouble x_0_7_3 = Qtempx * x_0_1_3 + WQtempx * x_0_1_4+ CDtemp*(VY( 0, 0, 3) - ABcom * VY( 0, 0, 4));
                        QUICKDouble x_0_8_3 = Qtempy * x_0_2_3 + WQtempy * x_0_2_4+ CDtemp*(VY( 0, 0, 3) - ABcom * VY( 0, 0, 4));
                        QUICKDouble x_0_9_3 = Qtempz * x_0_3_3 + WQtempz * x_0_3_4+ CDtemp*(VY( 0, 0, 3) - ABcom * VY( 0, 0, 4));
                        
                        //PSSS(4, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);
                        QUICKDouble x_1_0_4 = Ptempx * VY( 0, 0, 4) + WPtempx * VY( 0, 0, 5);
                        QUICKDouble x_2_0_4 = Ptempy * VY( 0, 0, 4) + WPtempy * VY( 0, 0, 5);
                        QUICKDouble x_3_0_4 = Ptempz * VY( 0, 0, 4) + WPtempz * VY( 0, 0, 5);
                        
                        
                        //PSSS(3, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);
                        QUICKDouble x_1_0_3 = Ptempx * VY( 0, 0, 3) + WPtempx * VY( 0, 0, 4);
                        QUICKDouble x_2_0_3 = Ptempy * VY( 0, 0, 3) + WPtempy * VY( 0, 0, 4);
                        QUICKDouble x_3_0_3 = Ptempz * VY( 0, 0, 3) + WPtempz * VY( 0, 0, 4);
                        
                        if (I+J>=3 && K+L>=1){
                            //DSSS(2, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz, ABtemp, CDcom);
                            QUICKDouble x_4_0_2 = Ptempx * x_2_0_2 + WPtempx * x_2_0_3;
                            QUICKDouble x_5_0_2 = Ptempy * x_3_0_2 + WPtempy * x_3_0_3;
                            QUICKDouble x_6_0_2 = Ptempx * x_3_0_2 + WPtempx * x_3_0_3;
                            
                            QUICKDouble x_7_0_2 = Ptempx * x_1_0_2 + WPtempx * x_1_0_3+ ABtemp*(VY( 0, 0, 2) - CDcom * VY( 0, 0, 3));
                            QUICKDouble x_8_0_2 = Ptempy * x_2_0_2 + WPtempy * x_2_0_3+ ABtemp*(VY( 0, 0, 2) - CDcom * VY( 0, 0, 3));
                            QUICKDouble x_9_0_2 = Ptempz * x_3_0_2 + WPtempz * x_3_0_3+ ABtemp*(VY( 0, 0, 2) - CDcom * VY( 0, 0, 3));
                            
                            //FSSS(1, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz, ABtemp, CDcom);
                            QUICKDouble x_10_0_1 = Ptempx * x_5_0_1 + WPtempx * x_5_0_2;
                            QUICKDouble x_11_0_1 = Ptempx * x_4_0_1 + WPtempx * x_4_0_2 + ABtemp * ( x_2_0_1 - CDcom * x_2_0_2);
                            QUICKDouble x_12_0_1 = Ptempx * x_8_0_1 + WPtempx * x_8_0_2;
                            QUICKDouble x_13_0_1 = Ptempx * x_6_0_1 + WPtempx * x_6_0_2 + ABtemp * ( x_3_0_1 - CDcom * x_3_0_2);
                            QUICKDouble x_14_0_1 = Ptempx * x_9_0_1 + WPtempx * x_9_0_2;
                            QUICKDouble x_15_0_1 = Ptempy * x_5_0_1 + WPtempy * x_5_0_2 + ABtemp * ( x_3_0_1 - CDcom * x_3_0_2);
                            QUICKDouble x_16_0_1 = Ptempy * x_9_0_1 + WPtempy * x_9_0_2;
                            QUICKDouble x_17_0_1 = Ptempx * x_7_0_1 + WPtempx * x_7_0_2 +  2 * ABtemp * ( x_1_0_1 - CDcom * x_1_0_2);
                            QUICKDouble x_18_0_1 = Ptempy * x_8_0_1 + WPtempy * x_8_0_2 +  2 * ABtemp * ( x_2_0_1 - CDcom * x_2_0_2);
                            QUICKDouble x_19_0_1 = Ptempz * x_9_0_1 + WPtempz * x_9_0_2 +  2 * ABtemp * ( x_3_0_1 - CDcom * x_3_0_2);
                            
                            //FSPS(0, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz, ABCDtemp);
                            
                            QUICKDouble x_10_1_0 = Qtempx * x_10_0_0 + WQtempx * x_10_0_1 + ABCDtemp *  x_5_0_1;
                            QUICKDouble x_10_2_0 = Qtempy * x_10_0_0 + WQtempy * x_10_0_1 + ABCDtemp *  x_6_0_1;
                            QUICKDouble x_10_3_0 = Qtempz * x_10_0_0 + WQtempz * x_10_0_1 + ABCDtemp *  x_4_0_1;
                            QUICKDouble x_11_1_0 = Qtempx * x_11_0_0 + WQtempx * x_11_0_1 +  2 * ABCDtemp *  x_4_0_1;
                            QUICKDouble x_11_2_0 = Qtempy * x_11_0_0 + WQtempy * x_11_0_1 + ABCDtemp *  x_7_0_1;
                            QUICKDouble x_11_3_0 = Qtempz * x_11_0_0 + WQtempz * x_11_0_1;
                            QUICKDouble x_12_1_0 = Qtempx * x_12_0_0 + WQtempx * x_12_0_1 + ABCDtemp *  x_8_0_1;
                            QUICKDouble x_12_2_0 = Qtempy * x_12_0_0 + WQtempy * x_12_0_1 +  2 * ABCDtemp *  x_4_0_1;
                            QUICKDouble x_12_3_0 = Qtempz * x_12_0_0 + WQtempz * x_12_0_1;
                            QUICKDouble x_13_1_0 = Qtempx * x_13_0_0 + WQtempx * x_13_0_1 +  2 * ABCDtemp *  x_6_0_1;
                            QUICKDouble x_13_2_0 = Qtempy * x_13_0_0 + WQtempy * x_13_0_1;
                            QUICKDouble x_13_3_0 = Qtempz * x_13_0_0 + WQtempz * x_13_0_1 + ABCDtemp *  x_7_0_1;
                            QUICKDouble x_14_1_0 = Qtempx * x_14_0_0 + WQtempx * x_14_0_1 + ABCDtemp *  x_9_0_1;
                            QUICKDouble x_14_2_0 = Qtempy * x_14_0_0 + WQtempy * x_14_0_1;
                            QUICKDouble x_14_3_0 = Qtempz * x_14_0_0 + WQtempz * x_14_0_1 +  2 * ABCDtemp *  x_6_0_1;
                            QUICKDouble x_15_1_0 = Qtempx * x_15_0_0 + WQtempx * x_15_0_1;
                            QUICKDouble x_15_2_0 = Qtempy * x_15_0_0 + WQtempy * x_15_0_1 +  2 * ABCDtemp *  x_5_0_1;
                            QUICKDouble x_15_3_0 = Qtempz * x_15_0_0 + WQtempz * x_15_0_1 + ABCDtemp *  x_8_0_1;
                            QUICKDouble x_16_1_0 = Qtempx * x_16_0_0 + WQtempx * x_16_0_1;
                            QUICKDouble x_16_2_0 = Qtempy * x_16_0_0 + WQtempy * x_16_0_1 + ABCDtemp *  x_9_0_1;
                            QUICKDouble x_16_3_0 = Qtempz * x_16_0_0 + WQtempz * x_16_0_1 +  2 * ABCDtemp *  x_5_0_1;
                            QUICKDouble x_17_1_0 = Qtempx * x_17_0_0 + WQtempx * x_17_0_1 +  3 * ABCDtemp *  x_7_0_1;
                            QUICKDouble x_17_2_0 = Qtempy * x_17_0_0 + WQtempy * x_17_0_1;
                            QUICKDouble x_17_3_0 = Qtempz * x_17_0_0 + WQtempz * x_17_0_1;
                            QUICKDouble x_18_1_0 = Qtempx * x_18_0_0 + WQtempx * x_18_0_1;
                            QUICKDouble x_18_2_0 = Qtempy * x_18_0_0 + WQtempy * x_18_0_1 +  3 * ABCDtemp *  x_8_0_1;
                            QUICKDouble x_18_3_0 = Qtempz * x_18_0_0 + WQtempz * x_18_0_1;
                            QUICKDouble x_19_1_0 = Qtempx * x_19_0_0 + WQtempx * x_19_0_1;
                            QUICKDouble x_19_2_0 = Qtempy * x_19_0_0 + WQtempy * x_19_0_1;
                            QUICKDouble x_19_3_0 = Qtempz * x_19_0_0 + WQtempz * x_19_0_1 +  3 * ABCDtemp *  x_9_0_1;
                            
                            LOC2(store,10, 1, STOREDIM, STOREDIM) += x_10_1_0;
                            LOC2(store,10, 2, STOREDIM, STOREDIM) += x_10_2_0;
                            LOC2(store,10, 3, STOREDIM, STOREDIM) += x_10_3_0;
                            
                            LOC2(store,11, 1, STOREDIM, STOREDIM) += x_11_1_0;
                            LOC2(store,11, 2, STOREDIM, STOREDIM) += x_11_2_0;
                            LOC2(store,11, 3, STOREDIM, STOREDIM) += x_11_3_0;
                            
                            LOC2(store,12, 1, STOREDIM, STOREDIM) += x_12_1_0;
                            LOC2(store,12, 2, STOREDIM, STOREDIM) += x_12_2_0;
                            LOC2(store,12, 3, STOREDIM, STOREDIM) += x_12_3_0;
                            
                            LOC2(store,13, 1, STOREDIM, STOREDIM) += x_13_1_0;
                            LOC2(store,13, 2, STOREDIM, STOREDIM) += x_13_2_0;
                            LOC2(store,13, 3, STOREDIM, STOREDIM) += x_13_3_0;
                            
                            LOC2(store,14, 1, STOREDIM, STOREDIM) += x_14_1_0;
                            LOC2(store,14, 2, STOREDIM, STOREDIM) += x_14_2_0;
                            LOC2(store,14, 3, STOREDIM, STOREDIM) += x_14_3_0;
                            
                            LOC2(store,15, 1, STOREDIM, STOREDIM) += x_15_1_0;
                            LOC2(store,15, 2, STOREDIM, STOREDIM) += x_15_2_0;
                            LOC2(store,15, 3, STOREDIM, STOREDIM) += x_15_3_0;
                            
                            LOC2(store,16, 1, STOREDIM, STOREDIM) += x_16_1_0;
                            LOC2(store,16, 2, STOREDIM, STOREDIM) += x_16_2_0;
                            LOC2(store,16, 3, STOREDIM, STOREDIM) += x_16_3_0;
                            
                            LOC2(store,17, 1, STOREDIM, STOREDIM) += x_17_1_0;
                            LOC2(store,17, 2, STOREDIM, STOREDIM) += x_17_2_0;
                            LOC2(store,17, 3, STOREDIM, STOREDIM) += x_17_3_0;
                            
                            LOC2(store,18, 1, STOREDIM, STOREDIM) += x_18_1_0;
                            LOC2(store,18, 2, STOREDIM, STOREDIM) += x_18_2_0;
                            LOC2(store,18, 3, STOREDIM, STOREDIM) += x_18_3_0;
                            
                            LOC2(store,19, 1, STOREDIM, STOREDIM) += x_19_1_0;
                            LOC2(store,19, 2, STOREDIM, STOREDIM) += x_19_2_0;
                            LOC2(store,19, 3, STOREDIM, STOREDIM) += x_19_3_0;
                            if (I+J>=3 && K+L>=2){
                                //DSSS(3, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz, ABtemp, CDcom);
                                QUICKDouble x_4_0_3 = Ptempx * x_2_0_3 + WPtempx * x_2_0_4;
                                QUICKDouble x_5_0_3 = Ptempy * x_3_0_3 + WPtempy * x_3_0_4;
                                QUICKDouble x_6_0_3 = Ptempx * x_3_0_3 + WPtempx * x_3_0_4;
                                
                                QUICKDouble x_7_0_3 = Ptempx * x_1_0_3 + WPtempx * x_1_0_4+ ABtemp*(VY( 0, 0, 3) - CDcom * VY( 0, 0, 4));
                                QUICKDouble x_8_0_3 = Ptempy * x_2_0_3 + WPtempy * x_2_0_4+ ABtemp*(VY( 0, 0, 3) - CDcom * VY( 0, 0, 4));
                                QUICKDouble x_9_0_3 = Ptempz * x_3_0_3 + WPtempz * x_3_0_4+ ABtemp*(VY( 0, 0, 3) - CDcom * VY( 0, 0, 4));
                                
                                //FSSS(2, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz, ABtemp, CDcom);
                                
                                QUICKDouble x_10_0_2 = Ptempx * x_5_0_2 + WPtempx * x_5_0_3;
                                QUICKDouble x_11_0_2 = Ptempx * x_4_0_2 + WPtempx * x_4_0_3 + ABtemp * ( x_2_0_2 - CDcom * x_2_0_3);
                                QUICKDouble x_12_0_2 = Ptempx * x_8_0_2 + WPtempx * x_8_0_3;
                                QUICKDouble x_13_0_2 = Ptempx * x_6_0_2 + WPtempx * x_6_0_3 + ABtemp * ( x_3_0_2 - CDcom * x_3_0_3);
                                QUICKDouble x_14_0_2 = Ptempx * x_9_0_2 + WPtempx * x_9_0_3;
                                QUICKDouble x_15_0_2 = Ptempy * x_5_0_2 + WPtempy * x_5_0_3 + ABtemp * ( x_3_0_2 - CDcom * x_3_0_3);
                                QUICKDouble x_16_0_2 = Ptempy * x_9_0_2 + WPtempy * x_9_0_3;
                                QUICKDouble x_17_0_2 = Ptempx * x_7_0_2 + WPtempx * x_7_0_3 +  2 * ABtemp * ( x_1_0_2 - CDcom * x_1_0_3);
                                QUICKDouble x_18_0_2 = Ptempy * x_8_0_2 + WPtempy * x_8_0_3 +  2 * ABtemp * ( x_2_0_2 - CDcom * x_2_0_3);
                                QUICKDouble x_19_0_2 = Ptempz * x_9_0_2 + WPtempz * x_9_0_3 +  2 * ABtemp * ( x_3_0_2 - CDcom * x_3_0_3);
                                
                                
                                //DSPS(1, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz, ABCDtemp);
                                
                                QUICKDouble x_4_1_1 = Qtempx * x_4_0_1 + WQtempx * x_4_0_2 + ABCDtemp * x_2_0_2;
                                QUICKDouble x_4_2_1 = Qtempy * x_4_0_1 + WQtempy * x_4_0_2 + ABCDtemp * x_1_0_2;
                                QUICKDouble x_4_3_1 = Qtempz * x_4_0_1 + WQtempz * x_4_0_2;
                                
                                QUICKDouble x_5_1_1 = Qtempx * x_5_0_1 + WQtempx * x_5_0_2;
                                QUICKDouble x_5_2_1 = Qtempy * x_5_0_1 + WQtempy * x_5_0_2 + ABCDtemp * x_3_0_2;
                                QUICKDouble x_5_3_1 = Qtempz * x_5_0_1 + WQtempz * x_5_0_2 + ABCDtemp * x_2_0_2;
                                
                                QUICKDouble x_6_1_1 = Qtempx * x_6_0_1 + WQtempx * x_6_0_2 + ABCDtemp * x_3_0_2;
                                QUICKDouble x_6_2_1 = Qtempy * x_6_0_1 + WQtempy * x_6_0_2;
                                QUICKDouble x_6_3_1 = Qtempz * x_6_0_1 + WQtempz * x_6_0_2 + ABCDtemp * x_1_0_2;
                                
                                QUICKDouble x_7_1_1 = Qtempx * x_7_0_1 + WQtempx * x_7_0_2 + ABCDtemp * x_1_0_2 * 2;
                                QUICKDouble x_7_2_1 = Qtempy * x_7_0_1 + WQtempy * x_7_0_2;
                                QUICKDouble x_7_3_1 = Qtempz * x_7_0_1 + WQtempz * x_7_0_2;
                                
                                QUICKDouble x_8_1_1 = Qtempx * x_8_0_1 + WQtempx * x_8_0_2;
                                QUICKDouble x_8_2_1 = Qtempy * x_8_0_1 + WQtempy * x_8_0_2 + ABCDtemp * x_2_0_2 * 2;
                                QUICKDouble x_8_3_1 = Qtempz * x_8_0_1 + WQtempz * x_8_0_2;
                                
                                QUICKDouble x_9_1_1 = Qtempx * x_9_0_1 + WQtempx * x_9_0_2;
                                QUICKDouble x_9_2_1 = Qtempy * x_9_0_1 + WQtempy * x_9_0_2;
                                QUICKDouble x_9_3_1 = Qtempz * x_9_0_1 + WQtempz * x_9_0_2 + ABCDtemp * x_3_0_2 * 2;  
                                //FSPS(1, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz, ABCDtemp);
                                
                                QUICKDouble x_10_1_1 = Qtempx * x_10_0_1 + WQtempx * x_10_0_2 + ABCDtemp *  x_5_0_2;
                                QUICKDouble x_10_2_1 = Qtempy * x_10_0_1 + WQtempy * x_10_0_2 + ABCDtemp *  x_6_0_2;
                                QUICKDouble x_10_3_1 = Qtempz * x_10_0_1 + WQtempz * x_10_0_2 + ABCDtemp *  x_4_0_2;
                                QUICKDouble x_11_1_1 = Qtempx * x_11_0_1 + WQtempx * x_11_0_2 +  2 * ABCDtemp *  x_4_0_2;
                                QUICKDouble x_11_2_1 = Qtempy * x_11_0_1 + WQtempy * x_11_0_2 + ABCDtemp *  x_7_0_2;
                                QUICKDouble x_11_3_1 = Qtempz * x_11_0_1 + WQtempz * x_11_0_2;
                                QUICKDouble x_12_1_1 = Qtempx * x_12_0_1 + WQtempx * x_12_0_2 + ABCDtemp *  x_8_0_2;
                                QUICKDouble x_12_2_1 = Qtempy * x_12_0_1 + WQtempy * x_12_0_2 +  2 * ABCDtemp *  x_4_0_2;
                                QUICKDouble x_12_3_1 = Qtempz * x_12_0_1 + WQtempz * x_12_0_2;
                                QUICKDouble x_13_1_1 = Qtempx * x_13_0_1 + WQtempx * x_13_0_2 +  2 * ABCDtemp *  x_6_0_2;
                                QUICKDouble x_13_2_1 = Qtempy * x_13_0_1 + WQtempy * x_13_0_2;
                                QUICKDouble x_13_3_1 = Qtempz * x_13_0_1 + WQtempz * x_13_0_2 + ABCDtemp *  x_7_0_2;
                                QUICKDouble x_14_1_1 = Qtempx * x_14_0_1 + WQtempx * x_14_0_2 + ABCDtemp *  x_9_0_2;
                                QUICKDouble x_14_2_1 = Qtempy * x_14_0_1 + WQtempy * x_14_0_2;
                                QUICKDouble x_14_3_1 = Qtempz * x_14_0_1 + WQtempz * x_14_0_2 +  2 * ABCDtemp *  x_6_0_2;
                                QUICKDouble x_15_1_1 = Qtempx * x_15_0_1 + WQtempx * x_15_0_2;
                                QUICKDouble x_15_2_1 = Qtempy * x_15_0_1 + WQtempy * x_15_0_2 +  2 * ABCDtemp *  x_5_0_2;
                                QUICKDouble x_15_3_1 = Qtempz * x_15_0_1 + WQtempz * x_15_0_2 + ABCDtemp *  x_8_0_2;
                                QUICKDouble x_16_1_1 = Qtempx * x_16_0_1 + WQtempx * x_16_0_2;
                                QUICKDouble x_16_2_1 = Qtempy * x_16_0_1 + WQtempy * x_16_0_2 + ABCDtemp *  x_9_0_2;
                                QUICKDouble x_16_3_1 = Qtempz * x_16_0_1 + WQtempz * x_16_0_2 +  2 * ABCDtemp *  x_5_0_2;
                                QUICKDouble x_17_1_1 = Qtempx * x_17_0_1 + WQtempx * x_17_0_2 +  3 * ABCDtemp *  x_7_0_2;
                                QUICKDouble x_17_2_1 = Qtempy * x_17_0_1 + WQtempy * x_17_0_2;
                                QUICKDouble x_17_3_1 = Qtempz * x_17_0_1 + WQtempz * x_17_0_2;
                                QUICKDouble x_18_1_1 = Qtempx * x_18_0_1 + WQtempx * x_18_0_2;
                                QUICKDouble x_18_2_1 = Qtempy * x_18_0_1 + WQtempy * x_18_0_2 +  3 * ABCDtemp *  x_8_0_2;
                                QUICKDouble x_18_3_1 = Qtempz * x_18_0_1 + WQtempz * x_18_0_2;
                                QUICKDouble x_19_1_1 = Qtempx * x_19_0_1 + WQtempx * x_19_0_2;
                                QUICKDouble x_19_2_1 = Qtempy * x_19_0_1 + WQtempy * x_19_0_2;
                                QUICKDouble x_19_3_1 = Qtempz * x_19_0_1 + WQtempz * x_19_0_2 +  3 * ABCDtemp *  x_9_0_2;
                                
                                
                                //FSDS(0, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz, ABCDtemp, CDtemp, ABcom);
                                QUICKDouble x_10_4_0 = Qtempx * x_10_2_0 + WQtempx * x_10_2_1 + ABCDtemp * x_5_2_1;
                                QUICKDouble x_11_4_0 = Qtempx * x_11_2_0 + WQtempx * x_11_2_1 +  2 * ABCDtemp * x_4_2_1;
                                QUICKDouble x_12_4_0 = Qtempx * x_12_2_0 + WQtempx * x_12_2_1 + ABCDtemp * x_8_2_1;
                                QUICKDouble x_13_4_0 = Qtempx * x_13_2_0 + WQtempx * x_13_2_1 +  2 * ABCDtemp * x_6_2_1;
                                QUICKDouble x_14_4_0 = Qtempx * x_14_2_0 + WQtempx * x_14_2_1 + ABCDtemp * x_9_2_1;
                                QUICKDouble x_15_4_0 = Qtempx * x_15_2_0 + WQtempx * x_15_2_1;
                                QUICKDouble x_16_4_0 = Qtempx * x_16_2_0 + WQtempx * x_16_2_1;
                                QUICKDouble x_17_4_0 = Qtempx * x_17_2_0 + WQtempx * x_17_2_1 +  3 * ABCDtemp * x_7_2_1;
                                QUICKDouble x_18_4_0 = Qtempx * x_18_2_0 + WQtempx * x_18_2_1;
                                QUICKDouble x_19_4_0 = Qtempx * x_19_2_0 + WQtempx * x_19_2_1;
                                QUICKDouble x_10_5_0 = Qtempy * x_10_3_0 + WQtempy * x_10_3_1 + ABCDtemp * x_6_3_1;
                                QUICKDouble x_11_5_0 = Qtempy * x_11_3_0 + WQtempy * x_11_3_1 + ABCDtemp * x_7_3_1;
                                QUICKDouble x_12_5_0 = Qtempy * x_12_3_0 + WQtempy * x_12_3_1 +  2 * ABCDtemp * x_4_3_1;
                                QUICKDouble x_13_5_0 = Qtempy * x_13_3_0 + WQtempy * x_13_3_1;
                                QUICKDouble x_14_5_0 = Qtempy * x_14_3_0 + WQtempy * x_14_3_1;
                                QUICKDouble x_15_5_0 = Qtempy * x_15_3_0 + WQtempy * x_15_3_1 +  2 * ABCDtemp * x_5_3_1;
                                QUICKDouble x_16_5_0 = Qtempy * x_16_3_0 + WQtempy * x_16_3_1 + ABCDtemp * x_9_3_1;
                                QUICKDouble x_17_5_0 = Qtempy * x_17_3_0 + WQtempy * x_17_3_1;
                                QUICKDouble x_18_5_0 = Qtempy * x_18_3_0 + WQtempy * x_18_3_1 +  3 * ABCDtemp * x_8_3_1;
                                QUICKDouble x_19_5_0 = Qtempy * x_19_3_0 + WQtempy * x_19_3_1;
                                QUICKDouble x_10_6_0 = Qtempx * x_10_3_0 + WQtempx * x_10_3_1 + ABCDtemp * x_5_3_1;
                                QUICKDouble x_11_6_0 = Qtempx * x_11_3_0 + WQtempx * x_11_3_1 +  2 * ABCDtemp * x_4_3_1;
                                QUICKDouble x_12_6_0 = Qtempx * x_12_3_0 + WQtempx * x_12_3_1 + ABCDtemp * x_8_3_1;
                                QUICKDouble x_13_6_0 = Qtempx * x_13_3_0 + WQtempx * x_13_3_1 +  2 * ABCDtemp * x_6_3_1;
                                QUICKDouble x_14_6_0 = Qtempx * x_14_3_0 + WQtempx * x_14_3_1 + ABCDtemp * x_9_3_1;
                                QUICKDouble x_15_6_0 = Qtempx * x_15_3_0 + WQtempx * x_15_3_1;
                                QUICKDouble x_16_6_0 = Qtempx * x_16_3_0 + WQtempx * x_16_3_1;
                                QUICKDouble x_17_6_0 = Qtempx * x_17_3_0 + WQtempx * x_17_3_1 +  3 * ABCDtemp * x_7_3_1;
                                QUICKDouble x_18_6_0 = Qtempx * x_18_3_0 + WQtempx * x_18_3_1;
                                QUICKDouble x_19_6_0 = Qtempx * x_19_3_0 + WQtempx * x_19_3_1;
                                QUICKDouble x_10_7_0 = Qtempx * x_10_1_0 + WQtempx * x_10_1_1 + CDtemp * ( x_10_0_0 - ABcom * x_10_0_1) + ABCDtemp * x_5_1_1;
                                QUICKDouble x_11_7_0 = Qtempx * x_11_1_0 + WQtempx * x_11_1_1 + CDtemp * ( x_11_0_0 - ABcom * x_11_0_1) +  2 * ABCDtemp * x_4_1_1;
                                QUICKDouble x_12_7_0 = Qtempx * x_12_1_0 + WQtempx * x_12_1_1 + CDtemp * ( x_12_0_0 - ABcom * x_12_0_1) + ABCDtemp * x_8_1_1;
                                QUICKDouble x_13_7_0 = Qtempx * x_13_1_0 + WQtempx * x_13_1_1 + CDtemp * ( x_13_0_0 - ABcom * x_13_0_1) +  2 * ABCDtemp * x_6_1_1;
                                QUICKDouble x_14_7_0 = Qtempx * x_14_1_0 + WQtempx * x_14_1_1 + CDtemp * ( x_14_0_0 - ABcom * x_14_0_1) + ABCDtemp * x_9_1_1;
                                QUICKDouble x_15_7_0 = Qtempx * x_15_1_0 + WQtempx * x_15_1_1 + CDtemp * ( x_15_0_0 - ABcom * x_15_0_1);
                                QUICKDouble x_16_7_0 = Qtempx * x_16_1_0 + WQtempx * x_16_1_1 + CDtemp * ( x_16_0_0 - ABcom * x_16_0_1);
                                QUICKDouble x_17_7_0 = Qtempx * x_17_1_0 + WQtempx * x_17_1_1 + CDtemp * ( x_17_0_0 - ABcom * x_17_0_1) +  3 * ABCDtemp * x_7_1_1;
                                QUICKDouble x_18_7_0 = Qtempx * x_18_1_0 + WQtempx * x_18_1_1 + CDtemp * ( x_18_0_0 - ABcom * x_18_0_1);
                                QUICKDouble x_19_7_0 = Qtempx * x_19_1_0 + WQtempx * x_19_1_1 + CDtemp * ( x_19_0_0 - ABcom * x_19_0_1);
                                QUICKDouble x_10_8_0 = Qtempy * x_10_2_0 + WQtempy * x_10_2_1 + CDtemp * ( x_10_0_0 - ABcom * x_10_0_1) + ABCDtemp * x_6_2_1;
                                QUICKDouble x_11_8_0 = Qtempy * x_11_2_0 + WQtempy * x_11_2_1 + CDtemp * ( x_11_0_0 - ABcom * x_11_0_1) + ABCDtemp * x_7_2_1;
                                QUICKDouble x_12_8_0 = Qtempy * x_12_2_0 + WQtempy * x_12_2_1 + CDtemp * ( x_12_0_0 - ABcom * x_12_0_1) +  2 * ABCDtemp * x_4_2_1;
                                QUICKDouble x_13_8_0 = Qtempy * x_13_2_0 + WQtempy * x_13_2_1 + CDtemp * ( x_13_0_0 - ABcom * x_13_0_1);
                                QUICKDouble x_14_8_0 = Qtempy * x_14_2_0 + WQtempy * x_14_2_1 + CDtemp * ( x_14_0_0 - ABcom * x_14_0_1);
                                QUICKDouble x_15_8_0 = Qtempy * x_15_2_0 + WQtempy * x_15_2_1 + CDtemp * ( x_15_0_0 - ABcom * x_15_0_1) +  2 * ABCDtemp * x_5_2_1;
                                QUICKDouble x_16_8_0 = Qtempy * x_16_2_0 + WQtempy * x_16_2_1 + CDtemp * ( x_16_0_0 - ABcom * x_16_0_1) + ABCDtemp * x_9_2_1;
                                QUICKDouble x_17_8_0 = Qtempy * x_17_2_0 + WQtempy * x_17_2_1 + CDtemp * ( x_17_0_0 - ABcom * x_17_0_1);
                                QUICKDouble x_18_8_0 = Qtempy * x_18_2_0 + WQtempy * x_18_2_1 + CDtemp * ( x_18_0_0 - ABcom * x_18_0_1) +  3 * ABCDtemp * x_8_2_1;
                                QUICKDouble x_19_8_0 = Qtempy * x_19_2_0 + WQtempy * x_19_2_1 + CDtemp * ( x_19_0_0 - ABcom * x_19_0_1);
                                QUICKDouble x_10_9_0 = Qtempz * x_10_3_0 + WQtempz * x_10_3_1 + CDtemp * ( x_10_0_0 - ABcom * x_10_0_1) + ABCDtemp * x_4_3_1;
                                QUICKDouble x_11_9_0 = Qtempz * x_11_3_0 + WQtempz * x_11_3_1 + CDtemp * ( x_11_0_0 - ABcom * x_11_0_1);
                                QUICKDouble x_12_9_0 = Qtempz * x_12_3_0 + WQtempz * x_12_3_1 + CDtemp * ( x_12_0_0 - ABcom * x_12_0_1);
                                QUICKDouble x_13_9_0 = Qtempz * x_13_3_0 + WQtempz * x_13_3_1 + CDtemp * ( x_13_0_0 - ABcom * x_13_0_1) + ABCDtemp * x_7_3_1;
                                QUICKDouble x_14_9_0 = Qtempz * x_14_3_0 + WQtempz * x_14_3_1 + CDtemp * ( x_14_0_0 - ABcom * x_14_0_1) +  2 * ABCDtemp * x_6_3_1;
                                QUICKDouble x_15_9_0 = Qtempz * x_15_3_0 + WQtempz * x_15_3_1 + CDtemp * ( x_15_0_0 - ABcom * x_15_0_1) + ABCDtemp * x_8_3_1;
                                QUICKDouble x_16_9_0 = Qtempz * x_16_3_0 + WQtempz * x_16_3_1 + CDtemp * ( x_16_0_0 - ABcom * x_16_0_1) +  2 * ABCDtemp * x_5_3_1;
                                QUICKDouble x_17_9_0 = Qtempz * x_17_3_0 + WQtempz * x_17_3_1 + CDtemp * ( x_17_0_0 - ABcom * x_17_0_1);
                                QUICKDouble x_18_9_0 = Qtempz * x_18_3_0 + WQtempz * x_18_3_1 + CDtemp * ( x_18_0_0 - ABcom * x_18_0_1);
                                QUICKDouble x_19_9_0 = Qtempz * x_19_3_0 + WQtempz * x_19_3_1 + CDtemp * ( x_19_0_0 - ABcom * x_19_0_1) +  3 * ABCDtemp * x_9_3_1;
                                
                                LOC2(store,10, 4, STOREDIM, STOREDIM) += x_10_4_0;
                                LOC2(store,10, 5, STOREDIM, STOREDIM) += x_10_5_0;
                                LOC2(store,10, 6, STOREDIM, STOREDIM) += x_10_6_0;
                                LOC2(store,10, 7, STOREDIM, STOREDIM) += x_10_7_0;
                                LOC2(store,10, 8, STOREDIM, STOREDIM) += x_10_8_0;
                                LOC2(store,10, 9, STOREDIM, STOREDIM) += x_10_9_0;
                                LOC2(store,11, 4, STOREDIM, STOREDIM) += x_11_4_0;
                                LOC2(store,11, 5, STOREDIM, STOREDIM) += x_11_5_0;
                                LOC2(store,11, 6, STOREDIM, STOREDIM) += x_11_6_0;
                                LOC2(store,11, 7, STOREDIM, STOREDIM) += x_11_7_0;
                                LOC2(store,11, 8, STOREDIM, STOREDIM) += x_11_8_0;
                                LOC2(store,11, 9, STOREDIM, STOREDIM) += x_11_9_0;
                                LOC2(store,12, 4, STOREDIM, STOREDIM) += x_12_4_0;
                                LOC2(store,12, 5, STOREDIM, STOREDIM) += x_12_5_0;
                                LOC2(store,12, 6, STOREDIM, STOREDIM) += x_12_6_0;
                                LOC2(store,12, 7, STOREDIM, STOREDIM) += x_12_7_0;
                                LOC2(store,12, 8, STOREDIM, STOREDIM) += x_12_8_0;
                                LOC2(store,12, 9, STOREDIM, STOREDIM) += x_12_9_0;
                                LOC2(store,13, 4, STOREDIM, STOREDIM) += x_13_4_0;
                                LOC2(store,13, 5, STOREDIM, STOREDIM) += x_13_5_0;
                                LOC2(store,13, 6, STOREDIM, STOREDIM) += x_13_6_0;
                                LOC2(store,13, 7, STOREDIM, STOREDIM) += x_13_7_0;
                                LOC2(store,13, 8, STOREDIM, STOREDIM) += x_13_8_0;
                                LOC2(store,13, 9, STOREDIM, STOREDIM) += x_13_9_0;
                                LOC2(store,14, 4, STOREDIM, STOREDIM) += x_14_4_0;
                                LOC2(store,14, 5, STOREDIM, STOREDIM) += x_14_5_0;
                                LOC2(store,14, 6, STOREDIM, STOREDIM) += x_14_6_0;
                                LOC2(store,14, 7, STOREDIM, STOREDIM) += x_14_7_0;
                                LOC2(store,14, 8, STOREDIM, STOREDIM) += x_14_8_0;
                                LOC2(store,14, 9, STOREDIM, STOREDIM) += x_14_9_0;
                                LOC2(store,15, 4, STOREDIM, STOREDIM) += x_15_4_0;
                                LOC2(store,15, 5, STOREDIM, STOREDIM) += x_15_5_0;
                                LOC2(store,15, 6, STOREDIM, STOREDIM) += x_15_6_0;
                                LOC2(store,15, 7, STOREDIM, STOREDIM) += x_15_7_0;
                                LOC2(store,15, 8, STOREDIM, STOREDIM) += x_15_8_0;
                                LOC2(store,15, 9, STOREDIM, STOREDIM) += x_15_9_0;
                                LOC2(store,16, 4, STOREDIM, STOREDIM) += x_16_4_0;
                                LOC2(store,16, 5, STOREDIM, STOREDIM) += x_16_5_0;
                                LOC2(store,16, 6, STOREDIM, STOREDIM) += x_16_6_0;
                                LOC2(store,16, 7, STOREDIM, STOREDIM) += x_16_7_0;
                                LOC2(store,16, 8, STOREDIM, STOREDIM) += x_16_8_0;
                                LOC2(store,16, 9, STOREDIM, STOREDIM) += x_16_9_0;
                                LOC2(store,17, 4, STOREDIM, STOREDIM) += x_17_4_0;
                                LOC2(store,17, 5, STOREDIM, STOREDIM) += x_17_5_0;
                                LOC2(store,17, 6, STOREDIM, STOREDIM) += x_17_6_0;
                                LOC2(store,17, 7, STOREDIM, STOREDIM) += x_17_7_0;
                                LOC2(store,17, 8, STOREDIM, STOREDIM) += x_17_8_0;
                                LOC2(store,17, 9, STOREDIM, STOREDIM) += x_17_9_0;
                                LOC2(store,18, 4, STOREDIM, STOREDIM) += x_18_4_0;
                                LOC2(store,18, 5, STOREDIM, STOREDIM) += x_18_5_0;
                                LOC2(store,18, 6, STOREDIM, STOREDIM) += x_18_6_0;
                                LOC2(store,18, 7, STOREDIM, STOREDIM) += x_18_7_0;
                                LOC2(store,18, 8, STOREDIM, STOREDIM) += x_18_8_0;
                                LOC2(store,18, 9, STOREDIM, STOREDIM) += x_18_9_0;
                                LOC2(store,19, 4, STOREDIM, STOREDIM) += x_19_4_0;
                                LOC2(store,19, 5, STOREDIM, STOREDIM) += x_19_5_0;
                                LOC2(store,19, 6, STOREDIM, STOREDIM) += x_19_6_0;
                                LOC2(store,19, 7, STOREDIM, STOREDIM) += x_19_7_0;
                                LOC2(store,19, 8, STOREDIM, STOREDIM) += x_19_8_0;
                                LOC2(store,19, 9, STOREDIM, STOREDIM) += x_19_9_0;
                                if (I+J>=3 && K+L>=3){
                                    
                                    //SSPS(1, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);
                                    QUICKDouble x_0_1_1 = Qtempx * VY( 0, 0, 1) + WQtempx * VY( 0, 0, 2);
                                    QUICKDouble x_0_2_1 = Qtempy * VY( 0, 0, 1) + WQtempy * VY( 0, 0, 2);
                                    QUICKDouble x_0_3_1 = Qtempz * VY( 0, 0, 1) + WQtempz * VY( 0, 0, 2);
                                    
                                    
                                    //SSDS(1, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz, CDtemp, ABcom);
                                    
                                    QUICKDouble x_0_4_1 = Qtempx * x_0_2_1 + WQtempx * x_0_2_2;
                                    QUICKDouble x_0_5_1 = Qtempy * x_0_3_1 + WQtempy * x_0_3_2;
                                    QUICKDouble x_0_6_1 = Qtempx * x_0_3_1 + WQtempx * x_0_3_2;
                                    
                                    QUICKDouble x_0_7_1 = Qtempx * x_0_1_1 + WQtempx * x_0_1_2+ CDtemp*(VY( 0, 0, 1) - ABcom * VY( 0, 0, 2));
                                    QUICKDouble x_0_8_1 = Qtempy * x_0_2_1 + WQtempy * x_0_2_2+ CDtemp*(VY( 0, 0, 1) - ABcom * VY( 0, 0, 2));
                                    QUICKDouble x_0_9_1 = Qtempz * x_0_3_1 + WQtempz * x_0_3_2+ CDtemp*(VY( 0, 0, 1) - ABcom * VY( 0, 0, 2));
                                    
                                    //PSDS(1, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz, ABCDtemp);
                                    
                                    QUICKDouble x_1_4_1 = Ptempx * x_0_4_1 + WPtempx * x_0_4_2 + ABCDtemp * x_0_2_2;
                                    QUICKDouble x_2_4_1 = Ptempy * x_0_4_1 + WPtempy * x_0_4_2 + ABCDtemp * x_0_1_2;
                                    QUICKDouble x_3_4_1 = Ptempz * x_0_4_1 + WPtempz * x_0_4_2;
                                    
                                    QUICKDouble x_1_5_1 = Ptempx * x_0_5_1 + WPtempx * x_0_5_2;
                                    QUICKDouble x_2_5_1 = Ptempy * x_0_5_1 + WPtempy * x_0_5_2 + ABCDtemp * x_0_3_2;
                                    QUICKDouble x_3_5_1 = Ptempz * x_0_5_1 + WPtempz * x_0_5_2 + ABCDtemp * x_0_2_2;
                                    
                                    QUICKDouble x_1_6_1 = Ptempx * x_0_6_1 + WPtempx * x_0_6_2 + ABCDtemp * x_0_3_2;
                                    QUICKDouble x_2_6_1 = Ptempy * x_0_6_1 + WPtempy * x_0_6_2;
                                    QUICKDouble x_3_6_1 = Ptempz * x_0_6_1 + WPtempz * x_0_6_2 + ABCDtemp * x_0_1_2;
                                    
                                    QUICKDouble x_1_7_1 = Ptempx * x_0_7_1 + WPtempx * x_0_7_2 + ABCDtemp * x_0_1_2 * 2;
                                    QUICKDouble x_2_7_1 = Ptempy * x_0_7_1 + WPtempy * x_0_7_2;
                                    QUICKDouble x_3_7_1 = Ptempz * x_0_7_1 + WPtempz * x_0_7_2;
                                    
                                    QUICKDouble x_1_8_1 = Ptempx * x_0_8_1 + WPtempx * x_0_8_2;
                                    QUICKDouble x_2_8_1 = Ptempy * x_0_8_1 + WPtempy * x_0_8_2 + ABCDtemp * x_0_2_2 * 2;
                                    QUICKDouble x_3_8_1 = Ptempz * x_0_8_1 + WPtempz * x_0_8_2;
                                    
                                    QUICKDouble x_1_9_1 = Ptempx * x_0_9_1 + WPtempx * x_0_9_2;
                                    QUICKDouble x_2_9_1 = Ptempy * x_0_9_1 + WPtempy * x_0_9_2;
                                    QUICKDouble x_3_9_1 = Ptempz * x_0_9_1 + WPtempz * x_0_9_2 + ABCDtemp * x_0_3_2 * 2;    
                                    
                                    //PSSS(5, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);
                                    QUICKDouble x_1_0_5 = Ptempx * VY( 0, 0, 5) + WPtempx * VY( 0, 0, 6);
                                    QUICKDouble x_2_0_5 = Ptempy * VY( 0, 0, 5) + WPtempy * VY( 0, 0, 6);
                                    QUICKDouble x_3_0_5 = Ptempz * VY( 0, 0, 5) + WPtempz * VY( 0, 0, 6);
                                    
                                    //PSDS(2, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz, ABCDtemp);
                                    
                                    QUICKDouble x_1_4_2 = Ptempx * x_0_4_2 + WPtempx * x_0_4_3 + ABCDtemp * x_0_2_3;
                                    QUICKDouble x_2_4_2 = Ptempy * x_0_4_2 + WPtempy * x_0_4_3 + ABCDtemp * x_0_1_3;
                                    QUICKDouble x_3_4_2 = Ptempz * x_0_4_2 + WPtempz * x_0_4_3;
                                    
                                    QUICKDouble x_1_5_2 = Ptempx * x_0_5_2 + WPtempx * x_0_5_3;
                                    QUICKDouble x_2_5_2 = Ptempy * x_0_5_2 + WPtempy * x_0_5_3 + ABCDtemp * x_0_3_3;
                                    QUICKDouble x_3_5_2 = Ptempz * x_0_5_2 + WPtempz * x_0_5_3 + ABCDtemp * x_0_2_3;
                                    
                                    QUICKDouble x_1_6_2 = Ptempx * x_0_6_2 + WPtempx * x_0_6_3 + ABCDtemp * x_0_3_3;
                                    QUICKDouble x_2_6_2 = Ptempy * x_0_6_2 + WPtempy * x_0_6_3;
                                    QUICKDouble x_3_6_2 = Ptempz * x_0_6_2 + WPtempz * x_0_6_3 + ABCDtemp * x_0_1_3;
                                    
                                    QUICKDouble x_1_7_2 = Ptempx * x_0_7_2 + WPtempx * x_0_7_3 + ABCDtemp * x_0_1_3 * 2;
                                    QUICKDouble x_2_7_2 = Ptempy * x_0_7_2 + WPtempy * x_0_7_3;
                                    QUICKDouble x_3_7_2 = Ptempz * x_0_7_2 + WPtempz * x_0_7_3;
                                    
                                    QUICKDouble x_1_8_2 = Ptempx * x_0_8_2 + WPtempx * x_0_8_3;
                                    QUICKDouble x_2_8_2 = Ptempy * x_0_8_2 + WPtempy * x_0_8_3 + ABCDtemp * x_0_2_3 * 2;
                                    QUICKDouble x_3_8_2 = Ptempz * x_0_8_2 + WPtempz * x_0_8_3;
                                    
                                    QUICKDouble x_1_9_2 = Ptempx * x_0_9_2 + WPtempx * x_0_9_3;
                                    QUICKDouble x_2_9_2 = Ptempy * x_0_9_2 + WPtempy * x_0_9_3;
                                    QUICKDouble x_3_9_2 = Ptempz * x_0_9_2 + WPtempz * x_0_9_3 + ABCDtemp * x_0_3_3 * 2;    
                                    
                                    //DSSS(4, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz, ABtemp, CDcom);
                                    QUICKDouble x_4_0_4 = Ptempx * x_2_0_4 + WPtempx * x_2_0_5;
                                    QUICKDouble x_5_0_4 = Ptempy * x_3_0_4 + WPtempy * x_3_0_5;
                                    QUICKDouble x_6_0_4 = Ptempx * x_3_0_4 + WPtempx * x_3_0_5;
                                    
                                    QUICKDouble x_7_0_4 = Ptempx * x_1_0_4 + WPtempx * x_1_0_5+ ABtemp*(VY( 0, 0, 4) - CDcom * VY( 0, 0, 5));
                                    QUICKDouble x_8_0_4 = Ptempy * x_2_0_4 + WPtempy * x_2_0_5+ ABtemp*(VY( 0, 0, 4) - CDcom * VY( 0, 0, 5));
                                    QUICKDouble x_9_0_4 = Ptempz * x_3_0_4 + WPtempz * x_3_0_5+ ABtemp*(VY( 0, 0, 4) - CDcom * VY( 0, 0, 5));
                                    
                                    //FSSS(3, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz, ABtemp, CDcom);
                                    
                                    QUICKDouble x_10_0_3 = Ptempx * x_5_0_3 + WPtempx * x_5_0_4;
                                    QUICKDouble x_11_0_3 = Ptempx * x_4_0_3 + WPtempx * x_4_0_4 + ABtemp * ( x_2_0_3 - CDcom * x_2_0_4);
                                    QUICKDouble x_12_0_3 = Ptempx * x_8_0_3 + WPtempx * x_8_0_4;
                                    QUICKDouble x_13_0_3 = Ptempx * x_6_0_3 + WPtempx * x_6_0_4 + ABtemp * ( x_3_0_3 - CDcom * x_3_0_4);
                                    QUICKDouble x_14_0_3 = Ptempx * x_9_0_3 + WPtempx * x_9_0_4;
                                    QUICKDouble x_15_0_3 = Ptempy * x_5_0_3 + WPtempy * x_5_0_4 + ABtemp * ( x_3_0_3 - CDcom * x_3_0_4);
                                    QUICKDouble x_16_0_3 = Ptempy * x_9_0_3 + WPtempy * x_9_0_4;
                                    QUICKDouble x_17_0_3 = Ptempx * x_7_0_3 + WPtempx * x_7_0_4 +  2 * ABtemp * ( x_1_0_3 - CDcom * x_1_0_4);
                                    QUICKDouble x_18_0_3 = Ptempy * x_8_0_3 + WPtempy * x_8_0_4 +  2 * ABtemp * ( x_2_0_3 - CDcom * x_2_0_4);
                                    QUICKDouble x_19_0_3 = Ptempz * x_9_0_3 + WPtempz * x_9_0_4 +  2 * ABtemp * ( x_3_0_3 - CDcom * x_3_0_4);
                                    
                                    //DSPS(2, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz, ABCDtemp);
                                    QUICKDouble x_4_1_2 = Qtempx * x_4_0_2 + WQtempx * x_4_0_3 + ABCDtemp * x_2_0_3;
                                    QUICKDouble x_4_2_2 = Qtempy * x_4_0_2 + WQtempy * x_4_0_3 + ABCDtemp * x_1_0_3;
                                    QUICKDouble x_4_3_2 = Qtempz * x_4_0_2 + WQtempz * x_4_0_3;
                                    
                                    QUICKDouble x_5_1_2 = Qtempx * x_5_0_2 + WQtempx * x_5_0_3;
                                    QUICKDouble x_5_2_2 = Qtempy * x_5_0_2 + WQtempy * x_5_0_3 + ABCDtemp * x_3_0_3;
                                    QUICKDouble x_5_3_2 = Qtempz * x_5_0_2 + WQtempz * x_5_0_3 + ABCDtemp * x_2_0_3;
                                    
                                    QUICKDouble x_6_1_2 = Qtempx * x_6_0_2 + WQtempx * x_6_0_3 + ABCDtemp * x_3_0_3;
                                    QUICKDouble x_6_2_2 = Qtempy * x_6_0_2 + WQtempy * x_6_0_3;
                                    QUICKDouble x_6_3_2 = Qtempz * x_6_0_2 + WQtempz * x_6_0_3 + ABCDtemp * x_1_0_3;
                                    
                                    QUICKDouble x_7_1_2 = Qtempx * x_7_0_2 + WQtempx * x_7_0_3 + ABCDtemp * x_1_0_3 * 2;
                                    QUICKDouble x_7_2_2 = Qtempy * x_7_0_2 + WQtempy * x_7_0_3;
                                    QUICKDouble x_7_3_2 = Qtempz * x_7_0_2 + WQtempz * x_7_0_3;
                                    
                                    QUICKDouble x_8_1_2 = Qtempx * x_8_0_2 + WQtempx * x_8_0_3;
                                    QUICKDouble x_8_2_2 = Qtempy * x_8_0_2 + WQtempy * x_8_0_3 + ABCDtemp * x_2_0_3 * 2;
                                    QUICKDouble x_8_3_2 = Qtempz * x_8_0_2 + WQtempz * x_8_0_3;
                                    
                                    QUICKDouble x_9_1_2 = Qtempx * x_9_0_2 + WQtempx * x_9_0_3;
                                    QUICKDouble x_9_2_2 = Qtempy * x_9_0_2 + WQtempy * x_9_0_3;
                                    QUICKDouble x_9_3_2 = Qtempz * x_9_0_2 + WQtempz * x_9_0_3 + ABCDtemp * x_3_0_3 * 2;            
                                    
                                    //FSPS(2, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz, ABCDtemp);
                                    
                                    QUICKDouble x_10_1_2 = Qtempx * x_10_0_2 + WQtempx * x_10_0_3 + ABCDtemp *  x_5_0_3;
                                    QUICKDouble x_10_2_2 = Qtempy * x_10_0_2 + WQtempy * x_10_0_3 + ABCDtemp *  x_6_0_3;
                                    QUICKDouble x_10_3_2 = Qtempz * x_10_0_2 + WQtempz * x_10_0_3 + ABCDtemp *  x_4_0_3;
                                    QUICKDouble x_11_1_2 = Qtempx * x_11_0_2 + WQtempx * x_11_0_3 +  2 * ABCDtemp *  x_4_0_3;
                                    QUICKDouble x_11_2_2 = Qtempy * x_11_0_2 + WQtempy * x_11_0_3 + ABCDtemp *  x_7_0_3;
                                    QUICKDouble x_11_3_2 = Qtempz * x_11_0_2 + WQtempz * x_11_0_3;
                                    QUICKDouble x_12_1_2 = Qtempx * x_12_0_2 + WQtempx * x_12_0_3 + ABCDtemp *  x_8_0_3;
                                    QUICKDouble x_12_2_2 = Qtempy * x_12_0_2 + WQtempy * x_12_0_3 +  2 * ABCDtemp *  x_4_0_3;
                                    QUICKDouble x_12_3_2 = Qtempz * x_12_0_2 + WQtempz * x_12_0_3;
                                    QUICKDouble x_13_1_2 = Qtempx * x_13_0_2 + WQtempx * x_13_0_3 +  2 * ABCDtemp *  x_6_0_3;
                                    QUICKDouble x_13_2_2 = Qtempy * x_13_0_2 + WQtempy * x_13_0_3;
                                    QUICKDouble x_13_3_2 = Qtempz * x_13_0_2 + WQtempz * x_13_0_3 + ABCDtemp *  x_7_0_3;
                                    QUICKDouble x_14_1_2 = Qtempx * x_14_0_2 + WQtempx * x_14_0_3 + ABCDtemp *  x_9_0_3;
                                    QUICKDouble x_14_2_2 = Qtempy * x_14_0_2 + WQtempy * x_14_0_3;
                                    QUICKDouble x_14_3_2 = Qtempz * x_14_0_2 + WQtempz * x_14_0_3 +  2 * ABCDtemp *  x_6_0_3;
                                    QUICKDouble x_15_1_2 = Qtempx * x_15_0_2 + WQtempx * x_15_0_3;
                                    QUICKDouble x_15_2_2 = Qtempy * x_15_0_2 + WQtempy * x_15_0_3 +  2 * ABCDtemp *  x_5_0_3;
                                    QUICKDouble x_15_3_2 = Qtempz * x_15_0_2 + WQtempz * x_15_0_3 + ABCDtemp *  x_8_0_3;
                                    QUICKDouble x_16_1_2 = Qtempx * x_16_0_2 + WQtempx * x_16_0_3;
                                    QUICKDouble x_16_2_2 = Qtempy * x_16_0_2 + WQtempy * x_16_0_3 + ABCDtemp *  x_9_0_3;
                                    QUICKDouble x_16_3_2 = Qtempz * x_16_0_2 + WQtempz * x_16_0_3 +  2 * ABCDtemp *  x_5_0_3;
                                    QUICKDouble x_17_1_2 = Qtempx * x_17_0_2 + WQtempx * x_17_0_3 +  3 * ABCDtemp *  x_7_0_3;
                                    QUICKDouble x_17_2_2 = Qtempy * x_17_0_2 + WQtempy * x_17_0_3;
                                    QUICKDouble x_17_3_2 = Qtempz * x_17_0_2 + WQtempz * x_17_0_3;
                                    QUICKDouble x_18_1_2 = Qtempx * x_18_0_2 + WQtempx * x_18_0_3;
                                    QUICKDouble x_18_2_2 = Qtempy * x_18_0_2 + WQtempy * x_18_0_3 +  3 * ABCDtemp *  x_8_0_3;
                                    QUICKDouble x_18_3_2 = Qtempz * x_18_0_2 + WQtempz * x_18_0_3;
                                    QUICKDouble x_19_1_2 = Qtempx * x_19_0_2 + WQtempx * x_19_0_3;
                                    QUICKDouble x_19_2_2 = Qtempy * x_19_0_2 + WQtempy * x_19_0_3;
                                    QUICKDouble x_19_3_2 = Qtempz * x_19_0_2 + WQtempz * x_19_0_3 +  3 * ABCDtemp *  x_9_0_3;
                                    
                                    //FSDS(1, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz, ABCDtemp, CDtemp, ABcom);
                                    
                                    QUICKDouble x_10_4_1 = Qtempx * x_10_2_1 + WQtempx * x_10_2_2 + ABCDtemp * x_5_2_2;
                                    QUICKDouble x_11_4_1 = Qtempx * x_11_2_1 + WQtempx * x_11_2_2 +  2 * ABCDtemp * x_4_2_2;
                                    QUICKDouble x_12_4_1 = Qtempx * x_12_2_1 + WQtempx * x_12_2_2 + ABCDtemp * x_8_2_2;
                                    QUICKDouble x_13_4_1 = Qtempx * x_13_2_1 + WQtempx * x_13_2_2 +  2 * ABCDtemp * x_6_2_2;
                                    QUICKDouble x_14_4_1 = Qtempx * x_14_2_1 + WQtempx * x_14_2_2 + ABCDtemp * x_9_2_2;
                                    QUICKDouble x_15_4_1 = Qtempx * x_15_2_1 + WQtempx * x_15_2_2;
                                    QUICKDouble x_16_4_1 = Qtempx * x_16_2_1 + WQtempx * x_16_2_2;
                                    QUICKDouble x_17_4_1 = Qtempx * x_17_2_1 + WQtempx * x_17_2_2 +  3 * ABCDtemp * x_7_2_2;
                                    QUICKDouble x_18_4_1 = Qtempx * x_18_2_1 + WQtempx * x_18_2_2;
                                    QUICKDouble x_19_4_1 = Qtempx * x_19_2_1 + WQtempx * x_19_2_2;
                                    QUICKDouble x_10_5_1 = Qtempy * x_10_3_1 + WQtempy * x_10_3_2 + ABCDtemp * x_6_3_2;
                                    QUICKDouble x_11_5_1 = Qtempy * x_11_3_1 + WQtempy * x_11_3_2 + ABCDtemp * x_7_3_2;
                                    QUICKDouble x_12_5_1 = Qtempy * x_12_3_1 + WQtempy * x_12_3_2 +  2 * ABCDtemp * x_4_3_2;
                                    QUICKDouble x_13_5_1 = Qtempy * x_13_3_1 + WQtempy * x_13_3_2;
                                    QUICKDouble x_14_5_1 = Qtempy * x_14_3_1 + WQtempy * x_14_3_2;
                                    QUICKDouble x_15_5_1 = Qtempy * x_15_3_1 + WQtempy * x_15_3_2 +  2 * ABCDtemp * x_5_3_2;
                                    QUICKDouble x_16_5_1 = Qtempy * x_16_3_1 + WQtempy * x_16_3_2 + ABCDtemp * x_9_3_2;
                                    QUICKDouble x_17_5_1 = Qtempy * x_17_3_1 + WQtempy * x_17_3_2;
                                    QUICKDouble x_18_5_1 = Qtempy * x_18_3_1 + WQtempy * x_18_3_2 +  3 * ABCDtemp * x_8_3_2;
                                    QUICKDouble x_19_5_1 = Qtempy * x_19_3_1 + WQtempy * x_19_3_2;
                                    QUICKDouble x_10_6_1 = Qtempx * x_10_3_1 + WQtempx * x_10_3_2 + ABCDtemp * x_5_3_2;
                                    QUICKDouble x_11_6_1 = Qtempx * x_11_3_1 + WQtempx * x_11_3_2 +  2 * ABCDtemp * x_4_3_2;
                                    QUICKDouble x_12_6_1 = Qtempx * x_12_3_1 + WQtempx * x_12_3_2 + ABCDtemp * x_8_3_2;
                                    QUICKDouble x_13_6_1 = Qtempx * x_13_3_1 + WQtempx * x_13_3_2 +  2 * ABCDtemp * x_6_3_2;
                                    QUICKDouble x_14_6_1 = Qtempx * x_14_3_1 + WQtempx * x_14_3_2 + ABCDtemp * x_9_3_2;
                                    QUICKDouble x_15_6_1 = Qtempx * x_15_3_1 + WQtempx * x_15_3_2;
                                    QUICKDouble x_16_6_1 = Qtempx * x_16_3_1 + WQtempx * x_16_3_2;
                                    QUICKDouble x_17_6_1 = Qtempx * x_17_3_1 + WQtempx * x_17_3_2 +  3 * ABCDtemp * x_7_3_2;
                                    QUICKDouble x_18_6_1 = Qtempx * x_18_3_1 + WQtempx * x_18_3_2;
                                    QUICKDouble x_19_6_1 = Qtempx * x_19_3_1 + WQtempx * x_19_3_2;
                                    QUICKDouble x_10_7_1 = Qtempx * x_10_1_1 + WQtempx * x_10_1_2 + CDtemp * ( x_10_0_1 - ABcom * x_10_0_2) + ABCDtemp * x_5_1_2;
                                    QUICKDouble x_11_7_1 = Qtempx * x_11_1_1 + WQtempx * x_11_1_2 + CDtemp * ( x_11_0_1 - ABcom * x_11_0_2) +  2 * ABCDtemp * x_4_1_2;
                                    QUICKDouble x_12_7_1 = Qtempx * x_12_1_1 + WQtempx * x_12_1_2 + CDtemp * ( x_12_0_1 - ABcom * x_12_0_2) + ABCDtemp * x_8_1_2;
                                    QUICKDouble x_13_7_1 = Qtempx * x_13_1_1 + WQtempx * x_13_1_2 + CDtemp * ( x_13_0_1 - ABcom * x_13_0_2) +  2 * ABCDtemp * x_6_1_2;
                                    QUICKDouble x_14_7_1 = Qtempx * x_14_1_1 + WQtempx * x_14_1_2 + CDtemp * ( x_14_0_1 - ABcom * x_14_0_2) + ABCDtemp * x_9_1_2;
                                    QUICKDouble x_15_7_1 = Qtempx * x_15_1_1 + WQtempx * x_15_1_2 + CDtemp * ( x_15_0_1 - ABcom * x_15_0_2);
                                    QUICKDouble x_16_7_1 = Qtempx * x_16_1_1 + WQtempx * x_16_1_2 + CDtemp * ( x_16_0_1 - ABcom * x_16_0_2);
                                    QUICKDouble x_17_7_1 = Qtempx * x_17_1_1 + WQtempx * x_17_1_2 + CDtemp * ( x_17_0_1 - ABcom * x_17_0_2) +  3 * ABCDtemp * x_7_1_2;
                                    QUICKDouble x_18_7_1 = Qtempx * x_18_1_1 + WQtempx * x_18_1_2 + CDtemp * ( x_18_0_1 - ABcom * x_18_0_2);
                                    QUICKDouble x_19_7_1 = Qtempx * x_19_1_1 + WQtempx * x_19_1_2 + CDtemp * ( x_19_0_1 - ABcom * x_19_0_2);
                                    QUICKDouble x_10_8_1 = Qtempy * x_10_2_1 + WQtempy * x_10_2_2 + CDtemp * ( x_10_0_1 - ABcom * x_10_0_2) + ABCDtemp * x_6_2_2;
                                    QUICKDouble x_11_8_1 = Qtempy * x_11_2_1 + WQtempy * x_11_2_2 + CDtemp * ( x_11_0_1 - ABcom * x_11_0_2) + ABCDtemp * x_7_2_2;
                                    QUICKDouble x_12_8_1 = Qtempy * x_12_2_1 + WQtempy * x_12_2_2 + CDtemp * ( x_12_0_1 - ABcom * x_12_0_2) +  2 * ABCDtemp * x_4_2_2;
                                    QUICKDouble x_13_8_1 = Qtempy * x_13_2_1 + WQtempy * x_13_2_2 + CDtemp * ( x_13_0_1 - ABcom * x_13_0_2);
                                    QUICKDouble x_14_8_1 = Qtempy * x_14_2_1 + WQtempy * x_14_2_2 + CDtemp * ( x_14_0_1 - ABcom * x_14_0_2);
                                    QUICKDouble x_15_8_1 = Qtempy * x_15_2_1 + WQtempy * x_15_2_2 + CDtemp * ( x_15_0_1 - ABcom * x_15_0_2) +  2 * ABCDtemp * x_5_2_2;
                                    QUICKDouble x_16_8_1 = Qtempy * x_16_2_1 + WQtempy * x_16_2_2 + CDtemp * ( x_16_0_1 - ABcom * x_16_0_2) + ABCDtemp * x_9_2_2;
                                    QUICKDouble x_17_8_1 = Qtempy * x_17_2_1 + WQtempy * x_17_2_2 + CDtemp * ( x_17_0_1 - ABcom * x_17_0_2);
                                    QUICKDouble x_18_8_1 = Qtempy * x_18_2_1 + WQtempy * x_18_2_2 + CDtemp * ( x_18_0_1 - ABcom * x_18_0_2) +  3 * ABCDtemp * x_8_2_2;
                                    QUICKDouble x_19_8_1 = Qtempy * x_19_2_1 + WQtempy * x_19_2_2 + CDtemp * ( x_19_0_1 - ABcom * x_19_0_2);
                                    QUICKDouble x_10_9_1 = Qtempz * x_10_3_1 + WQtempz * x_10_3_2 + CDtemp * ( x_10_0_1 - ABcom * x_10_0_2) + ABCDtemp * x_4_3_2;
                                    QUICKDouble x_11_9_1 = Qtempz * x_11_3_1 + WQtempz * x_11_3_2 + CDtemp * ( x_11_0_1 - ABcom * x_11_0_2);
                                    QUICKDouble x_12_9_1 = Qtempz * x_12_3_1 + WQtempz * x_12_3_2 + CDtemp * ( x_12_0_1 - ABcom * x_12_0_2);
                                    QUICKDouble x_13_9_1 = Qtempz * x_13_3_1 + WQtempz * x_13_3_2 + CDtemp * ( x_13_0_1 - ABcom * x_13_0_2) + ABCDtemp * x_7_3_2;
                                    QUICKDouble x_14_9_1 = Qtempz * x_14_3_1 + WQtempz * x_14_3_2 + CDtemp * ( x_14_0_1 - ABcom * x_14_0_2) +  2 * ABCDtemp * x_6_3_2;
                                    QUICKDouble x_15_9_1 = Qtempz * x_15_3_1 + WQtempz * x_15_3_2 + CDtemp * ( x_15_0_1 - ABcom * x_15_0_2) + ABCDtemp * x_8_3_2;
                                    QUICKDouble x_16_9_1 = Qtempz * x_16_3_1 + WQtempz * x_16_3_2 + CDtemp * ( x_16_0_1 - ABcom * x_16_0_2) +  2 * ABCDtemp * x_5_3_2;
                                    QUICKDouble x_17_9_1 = Qtempz * x_17_3_1 + WQtempz * x_17_3_2 + CDtemp * ( x_17_0_1 - ABcom * x_17_0_2);
                                    QUICKDouble x_18_9_1 = Qtempz * x_18_3_1 + WQtempz * x_18_3_2 + CDtemp * ( x_18_0_1 - ABcom * x_18_0_2);
                                    QUICKDouble x_19_9_1 = Qtempz * x_19_3_1 + WQtempz * x_19_3_2 + CDtemp * ( x_19_0_1 - ABcom * x_19_0_2) +  3 * ABCDtemp * x_9_3_2;
                                    
                                    //PSPS(2, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz, ABCDtemp);
                                    QUICKDouble x_1_1_2 = Ptempx * x_0_1_2 + WPtempx * x_0_1_3 + ABCDtemp * VY( 0, 0, 3);
                                    QUICKDouble x_2_1_2 = Ptempy * x_0_1_2 + WPtempy * x_0_1_3;
                                    QUICKDouble x_3_1_2 = Ptempz * x_0_1_2 + WPtempz * x_0_1_3;
                                    
                                    QUICKDouble x_1_2_2 = Ptempx * x_0_2_2 + WPtempx * x_0_2_3;
                                    QUICKDouble x_2_2_2 = Ptempy * x_0_2_2 + WPtempy * x_0_2_3 + ABCDtemp * VY( 0, 0, 3);
                                    QUICKDouble x_3_2_2 = Ptempz * x_0_2_2 + WPtempz * x_0_2_3;
                                    
                                    QUICKDouble x_1_3_2 = Ptempx * x_0_3_2 + WPtempx * x_0_3_3;
                                    QUICKDouble x_2_3_2 = Ptempy * x_0_3_2 + WPtempy * x_0_3_3;
                                    QUICKDouble x_3_3_2 = Ptempz * x_0_3_2 + WPtempz * x_0_3_3 + ABCDtemp * VY( 0, 0, 3);
                                    
                                    //DSDS(1, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz, ABCDtemp, ABtemp, CDcom);
                                    QUICKDouble x_4_4_1 = Ptempx * x_2_4_1 + WPtempx * x_2_4_2 + ABCDtemp * x_2_2_2;
                                    QUICKDouble x_4_5_1 = Ptempx * x_2_5_1 + WPtempx * x_2_5_2;
                                    QUICKDouble x_4_6_1 = Ptempx * x_2_6_1 + WPtempx * x_2_6_2 + ABCDtemp * x_2_3_2;
                                    QUICKDouble x_4_7_1 = Ptempx * x_2_7_1 + WPtempx * x_2_7_2 + 2 * ABCDtemp * x_2_1_2;
                                    QUICKDouble x_4_8_1 = Ptempx * x_2_8_1 + WPtempx * x_2_8_2;
                                    QUICKDouble x_4_9_1 = Ptempx * x_2_9_1 + WPtempx * x_2_9_2;
                                    
                                    QUICKDouble x_5_4_1 = Ptempy * x_3_4_1 + WPtempy * x_3_4_2 + ABCDtemp * x_3_1_2;
                                    QUICKDouble x_5_5_1 = Ptempy * x_3_5_1 + WPtempy * x_3_5_2 + ABCDtemp * x_3_3_2;
                                    QUICKDouble x_5_6_1 = Ptempy * x_3_6_1 + WPtempy * x_3_6_2;
                                    QUICKDouble x_5_7_1 = Ptempy * x_3_7_1 + WPtempy * x_3_7_2;
                                    QUICKDouble x_5_8_1 = Ptempy * x_3_8_1 + WPtempy * x_3_8_2 + 2 * ABCDtemp * x_3_2_2;
                                    QUICKDouble x_5_9_1 = Ptempy * x_3_9_1 + WPtempy * x_3_9_2;
                                    
                                    QUICKDouble x_6_4_1 = Ptempx * x_3_4_1 + WPtempx * x_3_4_2 + ABCDtemp * x_3_2_2;
                                    QUICKDouble x_6_5_1 = Ptempx * x_3_5_1 + WPtempx * x_3_5_2;
                                    QUICKDouble x_6_6_1 = Ptempx * x_3_6_1 + WPtempx * x_3_6_2 + ABCDtemp * x_3_3_2;
                                    QUICKDouble x_6_7_1 = Ptempx * x_3_7_1 + WPtempx * x_3_7_2 + 2 * ABCDtemp * x_3_1_2;
                                    QUICKDouble x_6_8_1 = Ptempx * x_3_8_1 + WPtempx * x_3_8_2;
                                    QUICKDouble x_6_9_1 = Ptempx * x_3_9_1 + WPtempx * x_3_9_2;
                                    
                                    QUICKDouble x_7_4_1 = Ptempx * x_1_4_1 + WPtempx * x_1_4_2 +  ABtemp * (x_0_4_1 - CDcom * x_0_4_2) + ABCDtemp * x_1_2_2;
                                    QUICKDouble x_7_5_1 = Ptempx * x_1_5_1 + WPtempx * x_1_5_2 +  ABtemp * (x_0_5_1 - CDcom * x_0_5_2);
                                    QUICKDouble x_7_6_1 = Ptempx * x_1_6_1 + WPtempx * x_1_6_2 +  ABtemp * (x_0_6_1 - CDcom * x_0_6_2) + ABCDtemp * x_1_3_2;
                                    QUICKDouble x_7_7_1 = Ptempx * x_1_7_1 + WPtempx * x_1_7_2 +  ABtemp * (x_0_7_1 - CDcom * x_0_7_2) + 2 * ABCDtemp * x_1_1_2;
                                    QUICKDouble x_7_8_1 = Ptempx * x_1_8_1 + WPtempx * x_1_8_2 +  ABtemp * (x_0_8_1 - CDcom * x_0_8_2);
                                    QUICKDouble x_7_9_1 = Ptempx * x_1_9_1 + WPtempx * x_1_9_2 +  ABtemp * (x_0_9_1 - CDcom * x_0_9_2);
                                    
                                    
                                    QUICKDouble x_8_4_1 = Ptempy * x_2_4_1 + WPtempy * x_2_4_2 +  ABtemp * (x_0_4_1 - CDcom * x_0_4_2) + ABCDtemp * x_2_1_2;
                                    QUICKDouble x_8_5_1 = Ptempy * x_2_5_1 + WPtempy * x_2_5_2 +  ABtemp * (x_0_5_1 - CDcom * x_0_5_2) + ABCDtemp * x_2_3_2;
                                    QUICKDouble x_8_6_1 = Ptempy * x_2_6_1 + WPtempy * x_2_6_2 +  ABtemp * (x_0_6_1 - CDcom * x_0_6_2);
                                    QUICKDouble x_8_7_1 = Ptempy * x_2_7_1 + WPtempy * x_2_7_2 +  ABtemp * (x_0_7_1 - CDcom * x_0_7_2);
                                    QUICKDouble x_8_8_1 = Ptempy * x_2_8_1 + WPtempy * x_2_8_2 +  ABtemp * (x_0_8_1 - CDcom * x_0_8_2) + 2 * ABCDtemp * x_2_2_2;
                                    QUICKDouble x_8_9_1 = Ptempy * x_2_9_1 + WPtempy * x_2_9_2 +  ABtemp * (x_0_9_1 - CDcom * x_0_9_2);
                                    
                                    QUICKDouble x_9_4_1 = Ptempz * x_3_4_1 + WPtempz * x_3_4_2 +  ABtemp * (x_0_4_1 - CDcom * x_0_4_2);
                                    QUICKDouble x_9_5_1 = Ptempz * x_3_5_1 + WPtempz * x_3_5_2 +  ABtemp * (x_0_5_1 - CDcom * x_0_5_2) + ABCDtemp * x_3_2_2;
                                    QUICKDouble x_9_6_1 = Ptempz * x_3_6_1 + WPtempz * x_3_6_2 +  ABtemp * (x_0_6_1 - CDcom * x_0_6_2) + ABCDtemp * x_3_1_2;
                                    QUICKDouble x_9_7_1 = Ptempz * x_3_7_1 + WPtempz * x_3_7_2 +  ABtemp * (x_0_7_1 - CDcom * x_0_7_2);
                                    QUICKDouble x_9_8_1 = Ptempz * x_3_8_1 + WPtempz * x_3_8_2 +  ABtemp * (x_0_8_1 - CDcom * x_0_8_2);
                                    QUICKDouble x_9_9_1 = Ptempz * x_3_9_1 + WPtempz * x_3_9_2 +  ABtemp * (x_0_9_1 - CDcom * x_0_9_2) + 2 * ABCDtemp * x_3_3_2;
                                    
                                    //FSFS(0, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz, ABCDtemp, CDtemp, ABcom);
                                    LOC2(store,10,10, STOREDIM, STOREDIM) += Qtempx * x_10_5_0 + WQtempx * x_10_5_1 + ABCDtemp * x_5_5_1;
                                    LOC2(store,11,10, STOREDIM, STOREDIM) += Qtempx * x_11_5_0 + WQtempx * x_11_5_1 +  2 * ABCDtemp * x_4_5_1;
                                    LOC2(store,12,10, STOREDIM, STOREDIM) += Qtempx * x_12_5_0 + WQtempx * x_12_5_1 + ABCDtemp * x_8_5_1;
                                    LOC2(store,13,10, STOREDIM, STOREDIM) += Qtempx * x_13_5_0 + WQtempx * x_13_5_1 +  2 * ABCDtemp * x_6_5_1;
                                    LOC2(store,14,10, STOREDIM, STOREDIM) += Qtempx * x_14_5_0 + WQtempx * x_14_5_1 + ABCDtemp * x_9_5_1;
                                    LOC2(store,15,10, STOREDIM, STOREDIM) += Qtempx * x_15_5_0 + WQtempx * x_15_5_1;
                                    LOC2(store,16,10, STOREDIM, STOREDIM) += Qtempx * x_16_5_0 + WQtempx * x_16_5_1;
                                    LOC2(store,17,10, STOREDIM, STOREDIM) += Qtempx * x_17_5_0 + WQtempx * x_17_5_1 +  3 * ABCDtemp * x_7_5_1;
                                    LOC2(store,18,10, STOREDIM, STOREDIM) += Qtempx * x_18_5_0 + WQtempx * x_18_5_1;
                                    LOC2(store,19,10, STOREDIM, STOREDIM) += Qtempx * x_19_5_0 + WQtempx * x_19_5_1;
                                    LOC2(store,10,11, STOREDIM, STOREDIM) += Qtempx * x_10_4_0 + WQtempx * x_10_4_1 + CDtemp * ( x_10_2_0 - ABcom * x_10_2_1) + ABCDtemp * x_5_4_1;
                                    LOC2(store,11,11, STOREDIM, STOREDIM) += Qtempx * x_11_4_0 + WQtempx * x_11_4_1 + CDtemp * ( x_11_2_0 - ABcom * x_11_2_1) +  2 * ABCDtemp * x_4_4_1;
                                    LOC2(store,12,11, STOREDIM, STOREDIM) += Qtempx * x_12_4_0 + WQtempx * x_12_4_1 + CDtemp * ( x_12_2_0 - ABcom * x_12_2_1) + ABCDtemp * x_8_4_1;
                                    LOC2(store,13,11, STOREDIM, STOREDIM) += Qtempx * x_13_4_0 + WQtempx * x_13_4_1 + CDtemp * ( x_13_2_0 - ABcom * x_13_2_1) +  2 * ABCDtemp * x_6_4_1;
                                    LOC2(store,14,11, STOREDIM, STOREDIM) += Qtempx * x_14_4_0 + WQtempx * x_14_4_1 + CDtemp * ( x_14_2_0 - ABcom * x_14_2_1) + ABCDtemp * x_9_4_1;
                                    LOC2(store,15,11, STOREDIM, STOREDIM) += Qtempx * x_15_4_0 + WQtempx * x_15_4_1 + CDtemp * ( x_15_2_0 - ABcom * x_15_2_1);
                                    LOC2(store,16,11, STOREDIM, STOREDIM) += Qtempx * x_16_4_0 + WQtempx * x_16_4_1 + CDtemp * ( x_16_2_0 - ABcom * x_16_2_1);
                                    LOC2(store,17,11, STOREDIM, STOREDIM) += Qtempx * x_17_4_0 + WQtempx * x_17_4_1 + CDtemp * ( x_17_2_0 - ABcom * x_17_2_1) +  3 * ABCDtemp * x_7_4_1;
                                    LOC2(store,18,11, STOREDIM, STOREDIM) += Qtempx * x_18_4_0 + WQtempx * x_18_4_1 + CDtemp * ( x_18_2_0 - ABcom * x_18_2_1);
                                    LOC2(store,19,11, STOREDIM, STOREDIM) += Qtempx * x_19_4_0 + WQtempx * x_19_4_1 + CDtemp * ( x_19_2_0 - ABcom * x_19_2_1);
                                    LOC2(store,10,12, STOREDIM, STOREDIM) += Qtempx * x_10_8_0 + WQtempx * x_10_8_1 + ABCDtemp * x_5_8_1;
                                    LOC2(store,11,12, STOREDIM, STOREDIM) += Qtempx * x_11_8_0 + WQtempx * x_11_8_1 +  2 * ABCDtemp * x_4_8_1;
                                    LOC2(store,12,12, STOREDIM, STOREDIM) += Qtempx * x_12_8_0 + WQtempx * x_12_8_1 + ABCDtemp * x_8_8_1;
                                    LOC2(store,13,12, STOREDIM, STOREDIM) += Qtempx * x_13_8_0 + WQtempx * x_13_8_1 +  2 * ABCDtemp * x_6_8_1;
                                    LOC2(store,14,12, STOREDIM, STOREDIM) += Qtempx * x_14_8_0 + WQtempx * x_14_8_1 + ABCDtemp * x_9_8_1;
                                    LOC2(store,15,12, STOREDIM, STOREDIM) += Qtempx * x_15_8_0 + WQtempx * x_15_8_1;
                                    LOC2(store,16,12, STOREDIM, STOREDIM) += Qtempx * x_16_8_0 + WQtempx * x_16_8_1;
                                    LOC2(store,17,12, STOREDIM, STOREDIM) += Qtempx * x_17_8_0 + WQtempx * x_17_8_1 +  3 * ABCDtemp * x_7_8_1;
                                    LOC2(store,18,12, STOREDIM, STOREDIM) += Qtempx * x_18_8_0 + WQtempx * x_18_8_1;
                                    LOC2(store,19,12, STOREDIM, STOREDIM) += Qtempx * x_19_8_0 + WQtempx * x_19_8_1;
                                    LOC2(store,10,13, STOREDIM, STOREDIM) += Qtempx * x_10_6_0 + WQtempx * x_10_6_1 + CDtemp * ( x_10_3_0 - ABcom * x_10_3_1) + ABCDtemp * x_5_6_1;
                                    LOC2(store,11,13, STOREDIM, STOREDIM) += Qtempx * x_11_6_0 + WQtempx * x_11_6_1 + CDtemp * ( x_11_3_0 - ABcom * x_11_3_1) +  2 * ABCDtemp * x_4_6_1;
                                    LOC2(store,12,13, STOREDIM, STOREDIM) += Qtempx * x_12_6_0 + WQtempx * x_12_6_1 + CDtemp * ( x_12_3_0 - ABcom * x_12_3_1) + ABCDtemp * x_8_6_1;
                                    LOC2(store,13,13, STOREDIM, STOREDIM) += Qtempx * x_13_6_0 + WQtempx * x_13_6_1 + CDtemp * ( x_13_3_0 - ABcom * x_13_3_1) +  2 * ABCDtemp * x_6_6_1;
                                    LOC2(store,14,13, STOREDIM, STOREDIM) += Qtempx * x_14_6_0 + WQtempx * x_14_6_1 + CDtemp * ( x_14_3_0 - ABcom * x_14_3_1) + ABCDtemp * x_9_6_1;
                                    LOC2(store,15,13, STOREDIM, STOREDIM) += Qtempx * x_15_6_0 + WQtempx * x_15_6_1 + CDtemp * ( x_15_3_0 - ABcom * x_15_3_1);
                                    LOC2(store,16,13, STOREDIM, STOREDIM) += Qtempx * x_16_6_0 + WQtempx * x_16_6_1 + CDtemp * ( x_16_3_0 - ABcom * x_16_3_1);
                                    LOC2(store,17,13, STOREDIM, STOREDIM) += Qtempx * x_17_6_0 + WQtempx * x_17_6_1 + CDtemp * ( x_17_3_0 - ABcom * x_17_3_1) +  3 * ABCDtemp * x_7_6_1;
                                    LOC2(store,18,13, STOREDIM, STOREDIM) += Qtempx * x_18_6_0 + WQtempx * x_18_6_1 + CDtemp * ( x_18_3_0 - ABcom * x_18_3_1);
                                    LOC2(store,19,13, STOREDIM, STOREDIM) += Qtempx * x_19_6_0 + WQtempx * x_19_6_1 + CDtemp * ( x_19_3_0 - ABcom * x_19_3_1);
                                    LOC2(store,10,14, STOREDIM, STOREDIM) += Qtempx * x_10_9_0 + WQtempx * x_10_9_1 + ABCDtemp * x_5_9_1;
                                    LOC2(store,11,14, STOREDIM, STOREDIM) += Qtempx * x_11_9_0 + WQtempx * x_11_9_1 +  2 * ABCDtemp * x_4_9_1;
                                    LOC2(store,12,14, STOREDIM, STOREDIM) += Qtempx * x_12_9_0 + WQtempx * x_12_9_1 + ABCDtemp * x_8_9_1;
                                    LOC2(store,13,14, STOREDIM, STOREDIM) += Qtempx * x_13_9_0 + WQtempx * x_13_9_1 +  2 * ABCDtemp * x_6_9_1;
                                    LOC2(store,14,14, STOREDIM, STOREDIM) += Qtempx * x_14_9_0 + WQtempx * x_14_9_1 + ABCDtemp * x_9_9_1;
                                    LOC2(store,15,14, STOREDIM, STOREDIM) += Qtempx * x_15_9_0 + WQtempx * x_15_9_1;
                                    LOC2(store,16,14, STOREDIM, STOREDIM) += Qtempx * x_16_9_0 + WQtempx * x_16_9_1;
                                    LOC2(store,17,14, STOREDIM, STOREDIM) += Qtempx * x_17_9_0 + WQtempx * x_17_9_1 +  3 * ABCDtemp * x_7_9_1;
                                    LOC2(store,18,14, STOREDIM, STOREDIM) += Qtempx * x_18_9_0 + WQtempx * x_18_9_1;
                                    LOC2(store,19,14, STOREDIM, STOREDIM) += Qtempx * x_19_9_0 + WQtempx * x_19_9_1;
                                    LOC2(store,10,15, STOREDIM, STOREDIM) += Qtempy * x_10_5_0 + WQtempy * x_10_5_1 + CDtemp * ( x_10_3_0 - ABcom * x_10_3_1) + ABCDtemp * x_6_5_1;
                                    LOC2(store,11,15, STOREDIM, STOREDIM) += Qtempy * x_11_5_0 + WQtempy * x_11_5_1 + CDtemp * ( x_11_3_0 - ABcom * x_11_3_1) + ABCDtemp * x_7_5_1;
                                    LOC2(store,12,15, STOREDIM, STOREDIM) += Qtempy * x_12_5_0 + WQtempy * x_12_5_1 + CDtemp * ( x_12_3_0 - ABcom * x_12_3_1) +  2 * ABCDtemp * x_4_5_1;
                                    LOC2(store,13,15, STOREDIM, STOREDIM) += Qtempy * x_13_5_0 + WQtempy * x_13_5_1 + CDtemp * ( x_13_3_0 - ABcom * x_13_3_1);
                                    LOC2(store,14,15, STOREDIM, STOREDIM) += Qtempy * x_14_5_0 + WQtempy * x_14_5_1 + CDtemp * ( x_14_3_0 - ABcom * x_14_3_1);
                                    LOC2(store,15,15, STOREDIM, STOREDIM) += Qtempy * x_15_5_0 + WQtempy * x_15_5_1 + CDtemp * ( x_15_3_0 - ABcom * x_15_3_1) +  2 * ABCDtemp * x_5_5_1;
                                    LOC2(store,16,15, STOREDIM, STOREDIM) += Qtempy * x_16_5_0 + WQtempy * x_16_5_1 + CDtemp * ( x_16_3_0 - ABcom * x_16_3_1) + ABCDtemp * x_9_5_1;
                                    LOC2(store,17,15, STOREDIM, STOREDIM) += Qtempy * x_17_5_0 + WQtempy * x_17_5_1 + CDtemp * ( x_17_3_0 - ABcom * x_17_3_1);
                                    LOC2(store,18,15, STOREDIM, STOREDIM) += Qtempy * x_18_5_0 + WQtempy * x_18_5_1 + CDtemp * ( x_18_3_0 - ABcom * x_18_3_1) +  3 * ABCDtemp * x_8_5_1;
                                    LOC2(store,19,15, STOREDIM, STOREDIM) += Qtempy * x_19_5_0 + WQtempy * x_19_5_1 + CDtemp * ( x_19_3_0 - ABcom * x_19_3_1);
                                    LOC2(store,10,16, STOREDIM, STOREDIM) += Qtempy * x_10_9_0 + WQtempy * x_10_9_1 + ABCDtemp * x_6_9_1;
                                    LOC2(store,11,16, STOREDIM, STOREDIM) += Qtempy * x_11_9_0 + WQtempy * x_11_9_1 + ABCDtemp * x_7_9_1;
                                    LOC2(store,12,16, STOREDIM, STOREDIM) += Qtempy * x_12_9_0 + WQtempy * x_12_9_1 +  2 * ABCDtemp * x_4_9_1;
                                    LOC2(store,13,16, STOREDIM, STOREDIM) += Qtempy * x_13_9_0 + WQtempy * x_13_9_1;
                                    LOC2(store,14,16, STOREDIM, STOREDIM) += Qtempy * x_14_9_0 + WQtempy * x_14_9_1;
                                    LOC2(store,15,16, STOREDIM, STOREDIM) += Qtempy * x_15_9_0 + WQtempy * x_15_9_1 +  2 * ABCDtemp * x_5_9_1;
                                    LOC2(store,16,16, STOREDIM, STOREDIM) += Qtempy * x_16_9_0 + WQtempy * x_16_9_1 + ABCDtemp * x_9_9_1;
                                    LOC2(store,17,16, STOREDIM, STOREDIM) += Qtempy * x_17_9_0 + WQtempy * x_17_9_1;
                                    LOC2(store,18,16, STOREDIM, STOREDIM) += Qtempy * x_18_9_0 + WQtempy * x_18_9_1 +  3 * ABCDtemp * x_8_9_1;
                                    LOC2(store,19,16, STOREDIM, STOREDIM) += Qtempy * x_19_9_0 + WQtempy * x_19_9_1;
                                    LOC2(store,10,17, STOREDIM, STOREDIM) += Qtempx * x_10_7_0 + WQtempx * x_10_7_1 +  2 * CDtemp * ( x_10_1_0 - ABcom * x_10_1_1) + ABCDtemp * x_5_7_1;
                                    LOC2(store,11,17, STOREDIM, STOREDIM) += Qtempx * x_11_7_0 + WQtempx * x_11_7_1 +  2 * CDtemp * ( x_11_1_0 - ABcom * x_11_1_1) +  2 * ABCDtemp * x_4_7_1;
                                    LOC2(store,12,17, STOREDIM, STOREDIM) += Qtempx * x_12_7_0 + WQtempx * x_12_7_1 +  2 * CDtemp * ( x_12_1_0 - ABcom * x_12_1_1) + ABCDtemp * x_8_7_1;
                                    LOC2(store,13,17, STOREDIM, STOREDIM) += Qtempx * x_13_7_0 + WQtempx * x_13_7_1 +  2 * CDtemp * ( x_13_1_0 - ABcom * x_13_1_1) +  2 * ABCDtemp * x_6_7_1;
                                    LOC2(store,14,17, STOREDIM, STOREDIM) += Qtempx * x_14_7_0 + WQtempx * x_14_7_1 +  2 * CDtemp * ( x_14_1_0 - ABcom * x_14_1_1) + ABCDtemp * x_9_7_1;
                                    LOC2(store,15,17, STOREDIM, STOREDIM) += Qtempx * x_15_7_0 + WQtempx * x_15_7_1 +  2 * CDtemp * ( x_15_1_0 - ABcom * x_15_1_1);
                                    LOC2(store,16,17, STOREDIM, STOREDIM) += Qtempx * x_16_7_0 + WQtempx * x_16_7_1 +  2 * CDtemp * ( x_16_1_0 - ABcom * x_16_1_1);
                                    LOC2(store,17,17, STOREDIM, STOREDIM) += Qtempx * x_17_7_0 + WQtempx * x_17_7_1 +  2 * CDtemp * ( x_17_1_0 - ABcom * x_17_1_1) +  3 * ABCDtemp * x_7_7_1;
                                    LOC2(store,18,17, STOREDIM, STOREDIM) += Qtempx * x_18_7_0 + WQtempx * x_18_7_1 +  2 * CDtemp * ( x_18_1_0 - ABcom * x_18_1_1);
                                    LOC2(store,19,17, STOREDIM, STOREDIM) += Qtempx * x_19_7_0 + WQtempx * x_19_7_1 +  2 * CDtemp * ( x_19_1_0 - ABcom * x_19_1_1);
                                    LOC2(store,10,18, STOREDIM, STOREDIM) += Qtempy * x_10_8_0 + WQtempy * x_10_8_1 +  2 * CDtemp * ( x_10_2_0 - ABcom * x_10_2_1) + ABCDtemp * x_6_8_1;
                                    LOC2(store,11,18, STOREDIM, STOREDIM) += Qtempy * x_11_8_0 + WQtempy * x_11_8_1 +  2 * CDtemp * ( x_11_2_0 - ABcom * x_11_2_1) + ABCDtemp * x_7_8_1;
                                    LOC2(store,12,18, STOREDIM, STOREDIM) += Qtempy * x_12_8_0 + WQtempy * x_12_8_1 +  2 * CDtemp * ( x_12_2_0 - ABcom * x_12_2_1) +  2 * ABCDtemp * x_4_8_1;
                                    LOC2(store,13,18, STOREDIM, STOREDIM) += Qtempy * x_13_8_0 + WQtempy * x_13_8_1 +  2 * CDtemp * ( x_13_2_0 - ABcom * x_13_2_1);
                                    LOC2(store,14,18, STOREDIM, STOREDIM) += Qtempy * x_14_8_0 + WQtempy * x_14_8_1 +  2 * CDtemp * ( x_14_2_0 - ABcom * x_14_2_1);
                                    LOC2(store,15,18, STOREDIM, STOREDIM) += Qtempy * x_15_8_0 + WQtempy * x_15_8_1 +  2 * CDtemp * ( x_15_2_0 - ABcom * x_15_2_1) +  2 * ABCDtemp * x_5_8_1;
                                    LOC2(store,16,18, STOREDIM, STOREDIM) += Qtempy * x_16_8_0 + WQtempy * x_16_8_1 +  2 * CDtemp * ( x_16_2_0 - ABcom * x_16_2_1) + ABCDtemp * x_9_8_1;
                                    LOC2(store,17,18, STOREDIM, STOREDIM) += Qtempy * x_17_8_0 + WQtempy * x_17_8_1 +  2 * CDtemp * ( x_17_2_0 - ABcom * x_17_2_1);
                                    LOC2(store,18,18, STOREDIM, STOREDIM) += Qtempy * x_18_8_0 + WQtempy * x_18_8_1 +  2 * CDtemp * ( x_18_2_0 - ABcom * x_18_2_1) +  3 * ABCDtemp * x_8_8_1;
                                    LOC2(store,19,18, STOREDIM, STOREDIM) += Qtempy * x_19_8_0 + WQtempy * x_19_8_1 +  2 * CDtemp * ( x_19_2_0 - ABcom * x_19_2_1);
                                    LOC2(store,10,19, STOREDIM, STOREDIM) += Qtempz * x_10_9_0 + WQtempz * x_10_9_1 +  2 * CDtemp * ( x_10_3_0 - ABcom * x_10_3_1) + ABCDtemp * x_4_9_1;
                                    LOC2(store,11,19, STOREDIM, STOREDIM) += Qtempz * x_11_9_0 + WQtempz * x_11_9_1 +  2 * CDtemp * ( x_11_3_0 - ABcom * x_11_3_1);
                                    LOC2(store,12,19, STOREDIM, STOREDIM) += Qtempz * x_12_9_0 + WQtempz * x_12_9_1 +  2 * CDtemp * ( x_12_3_0 - ABcom * x_12_3_1);
                                    LOC2(store,13,19, STOREDIM, STOREDIM) += Qtempz * x_13_9_0 + WQtempz * x_13_9_1 +  2 * CDtemp * ( x_13_3_0 - ABcom * x_13_3_1) + ABCDtemp * x_7_9_1;
                                    LOC2(store,14,19, STOREDIM, STOREDIM) += Qtempz * x_14_9_0 + WQtempz * x_14_9_1 +  2 * CDtemp * ( x_14_3_0 - ABcom * x_14_3_1) +  2 * ABCDtemp * x_6_9_1;
                                    LOC2(store,15,19, STOREDIM, STOREDIM) += Qtempz * x_15_9_0 + WQtempz * x_15_9_1 +  2 * CDtemp * ( x_15_3_0 - ABcom * x_15_3_1) + ABCDtemp * x_8_9_1;
                                    LOC2(store,16,19, STOREDIM, STOREDIM) += Qtempz * x_16_9_0 + WQtempz * x_16_9_1 +  2 * CDtemp * ( x_16_3_0 - ABcom * x_16_3_1) +  2 * ABCDtemp * x_5_9_1;
                                    LOC2(store,17,19, STOREDIM, STOREDIM) += Qtempz * x_17_9_0 + WQtempz * x_17_9_1 +  2 * CDtemp * ( x_17_3_0 - ABcom * x_17_3_1);
                                    LOC2(store,18,19, STOREDIM, STOREDIM) += Qtempz * x_18_9_0 + WQtempz * x_18_9_1 +  2 * CDtemp * ( x_18_3_0 - ABcom * x_18_3_1);
                                    LOC2(store,19,19, STOREDIM, STOREDIM) += Qtempz * x_19_9_0 + WQtempz * x_19_9_1 +  2 * CDtemp * ( x_19_3_0 - ABcom * x_19_3_1) +  3 * ABCDtemp * x_9_9_1;
                                }
                            }
                        }
                        if (I+J>=4){
                            //DSSS(2, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz, ABtemp, CDcom);
                            QUICKDouble x_4_0_2 = Ptempx * x_2_0_2 + WPtempx * x_2_0_3;
                            QUICKDouble x_5_0_2 = Ptempy * x_3_0_2 + WPtempy * x_3_0_3;
                            QUICKDouble x_6_0_2 = Ptempx * x_3_0_2 + WPtempx * x_3_0_3;
                            
                            QUICKDouble x_7_0_2 = Ptempx * x_1_0_2 + WPtempx * x_1_0_3+ ABtemp*(VY( 0, 0, 2) - CDcom * VY( 0, 0, 3));
                            QUICKDouble x_8_0_2 = Ptempy * x_2_0_2 + WPtempy * x_2_0_3+ ABtemp*(VY( 0, 0, 2) - CDcom * VY( 0, 0, 3));
                            QUICKDouble x_9_0_2 = Ptempz * x_3_0_2 + WPtempz * x_3_0_3+ ABtemp*(VY( 0, 0, 2) - CDcom * VY( 0, 0, 3));
                            
                            //FSSS(1, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz, ABtemp, CDcom);
                            QUICKDouble x_10_0_1 = Ptempx * x_5_0_1 + WPtempx * x_5_0_2;
                            QUICKDouble x_11_0_1 = Ptempx * x_4_0_1 + WPtempx * x_4_0_2 + ABtemp * ( x_2_0_1 - CDcom * x_2_0_2);
                            QUICKDouble x_12_0_1 = Ptempx * x_8_0_1 + WPtempx * x_8_0_2;
                            QUICKDouble x_13_0_1 = Ptempx * x_6_0_1 + WPtempx * x_6_0_2 + ABtemp * ( x_3_0_1 - CDcom * x_3_0_2);
                            QUICKDouble x_14_0_1 = Ptempx * x_9_0_1 + WPtempx * x_9_0_2;
                            QUICKDouble x_15_0_1 = Ptempy * x_5_0_1 + WPtempy * x_5_0_2 + ABtemp * ( x_3_0_1 - CDcom * x_3_0_2);
                            QUICKDouble x_16_0_1 = Ptempy * x_9_0_1 + WPtempy * x_9_0_2;
                            QUICKDouble x_17_0_1 = Ptempx * x_7_0_1 + WPtempx * x_7_0_2 +  2 * ABtemp * ( x_1_0_1 - CDcom * x_1_0_2);
                            QUICKDouble x_18_0_1 = Ptempy * x_8_0_1 + WPtempy * x_8_0_2 +  2 * ABtemp * ( x_2_0_1 - CDcom * x_2_0_2);
                            QUICKDouble x_19_0_1 = Ptempz * x_9_0_1 + WPtempz * x_9_0_2 +  2 * ABtemp * ( x_3_0_1 - CDcom * x_3_0_2);
                            
                            //GSSS(0, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz, ABtemp, CDcom);
                            
                            QUICKDouble x_20_0_0 = Ptempx * x_12_0_0 + WPtempx * x_12_0_1 + ABtemp * ( x_8_0_0 - CDcom * x_8_0_1);
                            QUICKDouble x_21_0_0 = Ptempx * x_14_0_0 + WPtempx * x_14_0_1 + ABtemp * ( x_9_0_0 - CDcom * x_9_0_1);
                            QUICKDouble x_22_0_0 = Ptempy * x_16_0_0 + WPtempy * x_16_0_1 + ABtemp * ( x_9_0_0 - CDcom * x_9_0_1);
                            QUICKDouble x_23_0_0 = Ptempx * x_10_0_0 + WPtempx * x_10_0_1 + ABtemp * ( x_5_0_0 - CDcom * x_5_0_1);
                            QUICKDouble x_24_0_0 = Ptempx * x_15_0_0 + WPtempx * x_15_0_1;
                            QUICKDouble x_25_0_0 = Ptempx * x_16_0_0 + WPtempx * x_16_0_1;
                            QUICKDouble x_26_0_0 = Ptempx * x_13_0_0 + WPtempx * x_13_0_1 + 2 * ABtemp * ( x_6_0_0 - CDcom * x_6_0_1);
                            QUICKDouble x_27_0_0 = Ptempx * x_19_0_0 + WPtempx * x_19_0_1;
                            QUICKDouble x_28_0_0 = Ptempx * x_11_0_0 + WPtempx * x_11_0_1 + 2 * ABtemp * ( x_4_0_0 - CDcom * x_4_0_1);
                            QUICKDouble x_29_0_0 = Ptempx * x_18_0_0 + WPtempx * x_18_0_1;
                            QUICKDouble x_30_0_0 = Ptempy * x_15_0_0 + WPtempy * x_15_0_1 + 2 * ABtemp * ( x_5_0_0 - CDcom * x_5_0_1);
                            QUICKDouble x_31_0_0 = Ptempy * x_19_0_0 + WPtempy * x_19_0_1;
                            QUICKDouble x_32_0_0 = Ptempx * x_17_0_0 + WPtempx * x_17_0_1 + 3 * ABtemp * ( x_7_0_0 - CDcom * x_7_0_1);
                            QUICKDouble x_33_0_0 = Ptempy * x_18_0_0 + WPtempy * x_18_0_1 + 3 * ABtemp * ( x_8_0_0 - CDcom * x_8_0_1);
                            QUICKDouble x_34_0_0 = Ptempz * x_19_0_0 + WPtempz * x_19_0_1 + 3 * ABtemp * ( x_9_0_0 - CDcom * x_9_0_1);
                            
                            LOC2(store,20, 0, STOREDIM, STOREDIM) += x_20_0_0;
                            LOC2(store,21, 0, STOREDIM, STOREDIM) += x_21_0_0;
                            LOC2(store,22, 0, STOREDIM, STOREDIM) += x_22_0_0;
                            LOC2(store,23, 0, STOREDIM, STOREDIM) += x_23_0_0;
                            LOC2(store,24, 0, STOREDIM, STOREDIM) += x_24_0_0;
                            LOC2(store,25, 0, STOREDIM, STOREDIM) += x_25_0_0;
                            LOC2(store,26, 0, STOREDIM, STOREDIM) += x_26_0_0;
                            LOC2(store,27, 0, STOREDIM, STOREDIM) += x_27_0_0;
                            LOC2(store,28, 0, STOREDIM, STOREDIM) += x_28_0_0;
                            LOC2(store,29, 0, STOREDIM, STOREDIM) += x_29_0_0;
                            LOC2(store,30, 0, STOREDIM, STOREDIM) += x_30_0_0;
                            LOC2(store,31, 0, STOREDIM, STOREDIM) += x_31_0_0;
                            LOC2(store,32, 0, STOREDIM, STOREDIM) += x_32_0_0;
                            LOC2(store,33, 0, STOREDIM, STOREDIM) += x_33_0_0;
                            LOC2(store,34, 0, STOREDIM, STOREDIM) += x_34_0_0;
                            if (I+J==4 && K+L>=1){
                                //DSSS(3, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz, ABtemp, CDcom);
                                QUICKDouble x_4_0_3 = Ptempx * x_2_0_3 + WPtempx * x_2_0_4;
                                QUICKDouble x_5_0_3 = Ptempy * x_3_0_3 + WPtempy * x_3_0_4;
                                QUICKDouble x_6_0_3 = Ptempx * x_3_0_3 + WPtempx * x_3_0_4;
                                
                                QUICKDouble x_7_0_3 = Ptempx * x_1_0_3 + WPtempx * x_1_0_4+ ABtemp*(VY( 0, 0, 3) - CDcom * VY( 0, 0, 4));
                                QUICKDouble x_8_0_3 = Ptempy * x_2_0_3 + WPtempy * x_2_0_4+ ABtemp*(VY( 0, 0, 3) - CDcom * VY( 0, 0, 4));
                                QUICKDouble x_9_0_3 = Ptempz * x_3_0_3 + WPtempz * x_3_0_4+ ABtemp*(VY( 0, 0, 3) - CDcom * VY( 0, 0, 4));
                                
                                //FSSS(2, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz, ABtemp, CDcom);
                                
                                QUICKDouble x_10_0_2 = Ptempx * x_5_0_2 + WPtempx * x_5_0_3;
                                QUICKDouble x_11_0_2 = Ptempx * x_4_0_2 + WPtempx * x_4_0_3 + ABtemp * ( x_2_0_2 - CDcom * x_2_0_3);
                                QUICKDouble x_12_0_2 = Ptempx * x_8_0_2 + WPtempx * x_8_0_3;
                                QUICKDouble x_13_0_2 = Ptempx * x_6_0_2 + WPtempx * x_6_0_3 + ABtemp * ( x_3_0_2 - CDcom * x_3_0_3);
                                QUICKDouble x_14_0_2 = Ptempx * x_9_0_2 + WPtempx * x_9_0_3;
                                QUICKDouble x_15_0_2 = Ptempy * x_5_0_2 + WPtempy * x_5_0_3 + ABtemp * ( x_3_0_2 - CDcom * x_3_0_3);
                                QUICKDouble x_16_0_2 = Ptempy * x_9_0_2 + WPtempy * x_9_0_3;
                                QUICKDouble x_17_0_2 = Ptempx * x_7_0_2 + WPtempx * x_7_0_3 +  2 * ABtemp * ( x_1_0_2 - CDcom * x_1_0_3);
                                QUICKDouble x_18_0_2 = Ptempy * x_8_0_2 + WPtempy * x_8_0_3 +  2 * ABtemp * ( x_2_0_2 - CDcom * x_2_0_3);
                                QUICKDouble x_19_0_2 = Ptempz * x_9_0_2 + WPtempz * x_9_0_3 +  2 * ABtemp * ( x_3_0_2 - CDcom * x_3_0_3);
                                
                                //GSSS(1, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz, ABtemp, CDcom);
                                
                                QUICKDouble x_20_0_1 = Ptempx * x_12_0_1 + WPtempx * x_12_0_2 + ABtemp * ( x_8_0_1 - CDcom * x_8_0_2);
                                QUICKDouble x_21_0_1 = Ptempx * x_14_0_1 + WPtempx * x_14_0_2 + ABtemp * ( x_9_0_1 - CDcom * x_9_0_2);
                                QUICKDouble x_22_0_1 = Ptempy * x_16_0_1 + WPtempy * x_16_0_2 + ABtemp * ( x_9_0_1 - CDcom * x_9_0_2);
                                QUICKDouble x_23_0_1 = Ptempx * x_10_0_1 + WPtempx * x_10_0_2 + ABtemp * ( x_5_0_1 - CDcom * x_5_0_2);
                                QUICKDouble x_24_0_1 = Ptempx * x_15_0_1 + WPtempx * x_15_0_2;
                                QUICKDouble x_25_0_1 = Ptempx * x_16_0_1 + WPtempx * x_16_0_2;
                                QUICKDouble x_26_0_1 = Ptempx * x_13_0_1 + WPtempx * x_13_0_2 + 2 * ABtemp * ( x_6_0_1 - CDcom * x_6_0_2);
                                QUICKDouble x_27_0_1 = Ptempx * x_19_0_1 + WPtempx * x_19_0_2;
                                QUICKDouble x_28_0_1 = Ptempx * x_11_0_1 + WPtempx * x_11_0_2 + 2 * ABtemp * ( x_4_0_1 - CDcom * x_4_0_2);
                                QUICKDouble x_29_0_1 = Ptempx * x_18_0_1 + WPtempx * x_18_0_2;
                                QUICKDouble x_30_0_1 = Ptempy * x_15_0_1 + WPtempy * x_15_0_2 + 2 * ABtemp * ( x_5_0_1 - CDcom * x_5_0_2);
                                QUICKDouble x_31_0_1 = Ptempy * x_19_0_1 + WPtempy * x_19_0_2;
                                QUICKDouble x_32_0_1 = Ptempx * x_17_0_1 + WPtempx * x_17_0_2 + 3 * ABtemp * ( x_7_0_1 - CDcom * x_7_0_2);
                                QUICKDouble x_33_0_1 = Ptempy * x_18_0_1 + WPtempy * x_18_0_2 + 3 * ABtemp * ( x_8_0_1 - CDcom * x_8_0_2);
                                QUICKDouble x_34_0_1 = Ptempz * x_19_0_1 + WPtempz * x_19_0_2 + 3 * ABtemp * ( x_9_0_1 - CDcom * x_9_0_2);
                                
                                //GSPS(0, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz, ABCDtemp);
                                
                                QUICKDouble x_20_1_0 = Qtempx * x_20_0_0 + WQtempx * x_20_0_1 +  2 * ABCDtemp *  x_12_0_1;
                                QUICKDouble x_20_2_0 = Qtempy * x_20_0_0 + WQtempy * x_20_0_1 +  2 * ABCDtemp *  x_11_0_1;
                                QUICKDouble x_20_3_0 = Qtempz * x_20_0_0 + WQtempz * x_20_0_1;
                                QUICKDouble x_21_1_0 = Qtempx * x_21_0_0 + WQtempx * x_21_0_1 +  2 * ABCDtemp *  x_14_0_1;
                                QUICKDouble x_21_2_0 = Qtempy * x_21_0_0 + WQtempy * x_21_0_1;
                                QUICKDouble x_21_3_0 = Qtempz * x_21_0_0 + WQtempz * x_21_0_1 +  2 * ABCDtemp *  x_13_0_1;
                                QUICKDouble x_22_1_0 = Qtempx * x_22_0_0 + WQtempx * x_22_0_1;
                                QUICKDouble x_22_2_0 = Qtempy * x_22_0_0 + WQtempy * x_22_0_1 +  2 * ABCDtemp *  x_16_0_1;
                                QUICKDouble x_22_3_0 = Qtempz * x_22_0_0 + WQtempz * x_22_0_1 +  2 * ABCDtemp *  x_15_0_1;
                                QUICKDouble x_23_1_0 = Qtempx * x_23_0_0 + WQtempx * x_23_0_1 +  2 * ABCDtemp *  x_10_0_1;
                                QUICKDouble x_23_2_0 = Qtempy * x_23_0_0 + WQtempy * x_23_0_1 + ABCDtemp *  x_13_0_1;
                                QUICKDouble x_23_3_0 = Qtempz * x_23_0_0 + WQtempz * x_23_0_1 + ABCDtemp *  x_11_0_1;
                                QUICKDouble x_24_1_0 = Qtempx * x_24_0_0 + WQtempx * x_24_0_1 + ABCDtemp *  x_15_0_1;
                                QUICKDouble x_24_2_0 = Qtempy * x_24_0_0 + WQtempy * x_24_0_1 +  2 * ABCDtemp *  x_10_0_1;
                                QUICKDouble x_24_3_0 = Qtempz * x_24_0_0 + WQtempz * x_24_0_1 + ABCDtemp *  x_12_0_1;
                                QUICKDouble x_25_1_0 = Qtempx * x_25_0_0 + WQtempx * x_25_0_1 + ABCDtemp *  x_16_0_1;
                                QUICKDouble x_25_2_0 = Qtempy * x_25_0_0 + WQtempy * x_25_0_1 + ABCDtemp *  x_14_0_1;
                                QUICKDouble x_25_3_0 = Qtempz * x_25_0_0 + WQtempz * x_25_0_1 +  2 * ABCDtemp *  x_10_0_1;
                                QUICKDouble x_26_1_0 = Qtempx * x_26_0_0 + WQtempx * x_26_0_1 +  3 * ABCDtemp *  x_13_0_1;
                                QUICKDouble x_26_2_0 = Qtempy * x_26_0_0 + WQtempy * x_26_0_1;
                                QUICKDouble x_26_3_0 = Qtempz * x_26_0_0 + WQtempz * x_26_0_1 + ABCDtemp *  x_17_0_1;
                                QUICKDouble x_27_1_0 = Qtempx * x_27_0_0 + WQtempx * x_27_0_1 + ABCDtemp *  x_19_0_1;
                                QUICKDouble x_27_2_0 = Qtempy * x_27_0_0 + WQtempy * x_27_0_1;
                                QUICKDouble x_27_3_0 = Qtempz * x_27_0_0 + WQtempz * x_27_0_1 +  3 * ABCDtemp *  x_14_0_1;
                                QUICKDouble x_28_1_0 = Qtempx * x_28_0_0 + WQtempx * x_28_0_1 +  3 * ABCDtemp *  x_11_0_1;
                                QUICKDouble x_28_2_0 = Qtempy * x_28_0_0 + WQtempy * x_28_0_1 + ABCDtemp *  x_17_0_1;
                                QUICKDouble x_28_3_0 = Qtempz * x_28_0_0 + WQtempz * x_28_0_1;
                                QUICKDouble x_29_1_0 = Qtempx * x_29_0_0 + WQtempx * x_29_0_1 + ABCDtemp *  x_18_0_1;
                                QUICKDouble x_29_2_0 = Qtempy * x_29_0_0 + WQtempy * x_29_0_1 +  3 * ABCDtemp *  x_12_0_1;
                                QUICKDouble x_29_3_0 = Qtempz * x_29_0_0 + WQtempz * x_29_0_1;
                                QUICKDouble x_30_1_0 = Qtempx * x_30_0_0 + WQtempx * x_30_0_1;
                                QUICKDouble x_30_2_0 = Qtempy * x_30_0_0 + WQtempy * x_30_0_1 +  3 * ABCDtemp *  x_15_0_1;
                                QUICKDouble x_30_3_0 = Qtempz * x_30_0_0 + WQtempz * x_30_0_1 + ABCDtemp *  x_18_0_1;
                                QUICKDouble x_31_1_0 = Qtempx * x_31_0_0 + WQtempx * x_31_0_1;
                                QUICKDouble x_31_2_0 = Qtempy * x_31_0_0 + WQtempy * x_31_0_1 + ABCDtemp *  x_19_0_1;
                                QUICKDouble x_31_3_0 = Qtempz * x_31_0_0 + WQtempz * x_31_0_1 +  3 * ABCDtemp *  x_16_0_1;    
                                QUICKDouble x_32_1_0 = Qtempx * x_32_0_0 + WQtempx * x_32_0_1 +  4 * ABCDtemp *  x_17_0_1;
                                QUICKDouble x_32_2_0 = Qtempy * x_32_0_0 + WQtempy * x_32_0_1;
                                QUICKDouble x_32_3_0 = Qtempz * x_32_0_0 + WQtempz * x_32_0_1;
                                QUICKDouble x_33_1_0 = Qtempx * x_33_0_0 + WQtempx * x_33_0_1;
                                QUICKDouble x_33_2_0 = Qtempy * x_33_0_0 + WQtempy * x_33_0_1 +  4 * ABCDtemp *  x_18_0_1;
                                QUICKDouble x_33_3_0 = Qtempz * x_33_0_0 + WQtempz * x_33_0_1;
                                QUICKDouble x_34_1_0 = Qtempx * x_34_0_0 + WQtempx * x_34_0_1;
                                QUICKDouble x_34_2_0 = Qtempy * x_34_0_0 + WQtempy * x_34_0_1;
                                QUICKDouble x_34_3_0 = Qtempz * x_34_0_0 + WQtempz * x_34_0_1 +  4 * ABCDtemp *  x_19_0_1;
                                
                                LOC2(store,20, 1, STOREDIM, STOREDIM) += x_20_1_0;
                                LOC2(store,20, 2, STOREDIM, STOREDIM) += x_20_2_0;
                                LOC2(store,20, 3, STOREDIM, STOREDIM) += x_20_3_0;
                                LOC2(store,21, 1, STOREDIM, STOREDIM) += x_21_1_0;
                                LOC2(store,21, 2, STOREDIM, STOREDIM) += x_21_2_0;
                                LOC2(store,21, 3, STOREDIM, STOREDIM) += x_21_3_0;
                                LOC2(store,22, 1, STOREDIM, STOREDIM) += x_22_1_0;
                                LOC2(store,22, 2, STOREDIM, STOREDIM) += x_22_2_0;
                                LOC2(store,22, 3, STOREDIM, STOREDIM) += x_22_3_0;
                                LOC2(store,23, 1, STOREDIM, STOREDIM) += x_23_1_0;
                                LOC2(store,23, 2, STOREDIM, STOREDIM) += x_23_2_0;
                                LOC2(store,23, 3, STOREDIM, STOREDIM) += x_23_3_0;
                                LOC2(store,24, 1, STOREDIM, STOREDIM) += x_24_1_0;
                                LOC2(store,24, 2, STOREDIM, STOREDIM) += x_24_2_0;
                                LOC2(store,24, 3, STOREDIM, STOREDIM) += x_24_3_0;
                                LOC2(store,25, 1, STOREDIM, STOREDIM) += x_25_1_0;
                                LOC2(store,25, 2, STOREDIM, STOREDIM) += x_25_2_0;
                                LOC2(store,25, 3, STOREDIM, STOREDIM) += x_25_3_0;
                                LOC2(store,26, 1, STOREDIM, STOREDIM) += x_26_1_0;
                                LOC2(store,26, 2, STOREDIM, STOREDIM) += x_26_2_0;
                                LOC2(store,26, 3, STOREDIM, STOREDIM) += x_26_3_0;
                                LOC2(store,27, 1, STOREDIM, STOREDIM) += x_27_1_0;
                                LOC2(store,27, 2, STOREDIM, STOREDIM) += x_27_2_0;
                                LOC2(store,27, 3, STOREDIM, STOREDIM) += x_27_3_0;
                                LOC2(store,28, 1, STOREDIM, STOREDIM) += x_28_1_0;
                                LOC2(store,28, 2, STOREDIM, STOREDIM) += x_28_2_0;
                                LOC2(store,28, 3, STOREDIM, STOREDIM) += x_28_3_0;
                                LOC2(store,29, 1, STOREDIM, STOREDIM) += x_29_1_0;
                                LOC2(store,29, 2, STOREDIM, STOREDIM) += x_29_2_0;
                                LOC2(store,29, 3, STOREDIM, STOREDIM) += x_29_3_0;
                                LOC2(store,30, 1, STOREDIM, STOREDIM) += x_30_1_0;
                                LOC2(store,30, 2, STOREDIM, STOREDIM) += x_30_2_0;
                                LOC2(store,30, 3, STOREDIM, STOREDIM) += x_30_3_0;
                                LOC2(store,31, 1, STOREDIM, STOREDIM) += x_31_1_0;
                                LOC2(store,31, 2, STOREDIM, STOREDIM) += x_31_2_0;
                                LOC2(store,31, 3, STOREDIM, STOREDIM) += x_31_3_0;
                                LOC2(store,32, 1, STOREDIM, STOREDIM) += x_32_1_0;
                                LOC2(store,32, 2, STOREDIM, STOREDIM) += x_32_2_0;
                                LOC2(store,32, 3, STOREDIM, STOREDIM) += x_32_3_0;
                                LOC2(store,33, 1, STOREDIM, STOREDIM) += x_33_1_0;
                                LOC2(store,33, 2, STOREDIM, STOREDIM) += x_33_2_0;
                                LOC2(store,33, 3, STOREDIM, STOREDIM) += x_33_3_0;
                                LOC2(store,34, 1, STOREDIM, STOREDIM) += x_34_1_0;
                                LOC2(store,34, 2, STOREDIM, STOREDIM) += x_34_2_0;
                                LOC2(store,34, 3, STOREDIM, STOREDIM) += x_34_3_0;
                                
                                if (I+J==4 && K+L>=2){
                                    //FSPS(1, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz, ABCDtemp);
                                    
                                    QUICKDouble x_10_1_1 = Qtempx * x_10_0_1 + WQtempx * x_10_0_2 + ABCDtemp *  x_5_0_2;
                                    QUICKDouble x_10_2_1 = Qtempy * x_10_0_1 + WQtempy * x_10_0_2 + ABCDtemp *  x_6_0_2;
                                    QUICKDouble x_10_3_1 = Qtempz * x_10_0_1 + WQtempz * x_10_0_2 + ABCDtemp *  x_4_0_2;
                                    QUICKDouble x_11_1_1 = Qtempx * x_11_0_1 + WQtempx * x_11_0_2 +  2 * ABCDtemp *  x_4_0_2;
                                    QUICKDouble x_11_2_1 = Qtempy * x_11_0_1 + WQtempy * x_11_0_2 + ABCDtemp *  x_7_0_2;
                                    QUICKDouble x_11_3_1 = Qtempz * x_11_0_1 + WQtempz * x_11_0_2;
                                    QUICKDouble x_12_1_1 = Qtempx * x_12_0_1 + WQtempx * x_12_0_2 + ABCDtemp *  x_8_0_2;
                                    QUICKDouble x_12_2_1 = Qtempy * x_12_0_1 + WQtempy * x_12_0_2 +  2 * ABCDtemp *  x_4_0_2;
                                    QUICKDouble x_12_3_1 = Qtempz * x_12_0_1 + WQtempz * x_12_0_2;
                                    QUICKDouble x_13_1_1 = Qtempx * x_13_0_1 + WQtempx * x_13_0_2 +  2 * ABCDtemp *  x_6_0_2;
                                    QUICKDouble x_13_2_1 = Qtempy * x_13_0_1 + WQtempy * x_13_0_2;
                                    QUICKDouble x_13_3_1 = Qtempz * x_13_0_1 + WQtempz * x_13_0_2 + ABCDtemp *  x_7_0_2;
                                    QUICKDouble x_14_1_1 = Qtempx * x_14_0_1 + WQtempx * x_14_0_2 + ABCDtemp *  x_9_0_2;
                                    QUICKDouble x_14_2_1 = Qtempy * x_14_0_1 + WQtempy * x_14_0_2;
                                    QUICKDouble x_14_3_1 = Qtempz * x_14_0_1 + WQtempz * x_14_0_2 +  2 * ABCDtemp *  x_6_0_2;
                                    QUICKDouble x_15_1_1 = Qtempx * x_15_0_1 + WQtempx * x_15_0_2;
                                    QUICKDouble x_15_2_1 = Qtempy * x_15_0_1 + WQtempy * x_15_0_2 +  2 * ABCDtemp *  x_5_0_2;
                                    QUICKDouble x_15_3_1 = Qtempz * x_15_0_1 + WQtempz * x_15_0_2 + ABCDtemp *  x_8_0_2;
                                    QUICKDouble x_16_1_1 = Qtempx * x_16_0_1 + WQtempx * x_16_0_2;
                                    QUICKDouble x_16_2_1 = Qtempy * x_16_0_1 + WQtempy * x_16_0_2 + ABCDtemp *  x_9_0_2;
                                    QUICKDouble x_16_3_1 = Qtempz * x_16_0_1 + WQtempz * x_16_0_2 +  2 * ABCDtemp *  x_5_0_2;
                                    QUICKDouble x_17_1_1 = Qtempx * x_17_0_1 + WQtempx * x_17_0_2 +  3 * ABCDtemp *  x_7_0_2;
                                    QUICKDouble x_17_2_1 = Qtempy * x_17_0_1 + WQtempy * x_17_0_2;
                                    QUICKDouble x_17_3_1 = Qtempz * x_17_0_1 + WQtempz * x_17_0_2;
                                    QUICKDouble x_18_1_1 = Qtempx * x_18_0_1 + WQtempx * x_18_0_2;
                                    QUICKDouble x_18_2_1 = Qtempy * x_18_0_1 + WQtempy * x_18_0_2 +  3 * ABCDtemp *  x_8_0_2;
                                    QUICKDouble x_18_3_1 = Qtempz * x_18_0_1 + WQtempz * x_18_0_2;
                                    QUICKDouble x_19_1_1 = Qtempx * x_19_0_1 + WQtempx * x_19_0_2;
                                    QUICKDouble x_19_2_1 = Qtempy * x_19_0_1 + WQtempy * x_19_0_2;
                                    QUICKDouble x_19_3_1 = Qtempz * x_19_0_1 + WQtempz * x_19_0_2 +  3 * ABCDtemp *  x_9_0_2;
                                    
                                    //PSSS(5, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);
                                    QUICKDouble x_1_0_5 = Ptempx * VY( 0, 0, 5) + WPtempx * VY( 0, 0, 6);
                                    QUICKDouble x_2_0_5 = Ptempy * VY( 0, 0, 5) + WPtempy * VY( 0, 0, 6);
                                    QUICKDouble x_3_0_5 = Ptempz * VY( 0, 0, 5) + WPtempz * VY( 0, 0, 6);
                                    
                                    //DSSS(4, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz, ABtemp, CDcom);
                                    QUICKDouble x_4_0_4 = Ptempx * x_2_0_4 + WPtempx * x_2_0_5;
                                    QUICKDouble x_5_0_4 = Ptempy * x_3_0_4 + WPtempy * x_3_0_5;
                                    QUICKDouble x_6_0_4 = Ptempx * x_3_0_4 + WPtempx * x_3_0_5;
                                    
                                    QUICKDouble x_7_0_4 = Ptempx * x_1_0_4 + WPtempx * x_1_0_5+ ABtemp*(VY( 0, 0, 4) - CDcom * VY( 0, 0, 5));
                                    QUICKDouble x_8_0_4 = Ptempy * x_2_0_4 + WPtempy * x_2_0_5+ ABtemp*(VY( 0, 0, 4) - CDcom * VY( 0, 0, 5));
                                    QUICKDouble x_9_0_4 = Ptempz * x_3_0_4 + WPtempz * x_3_0_5+ ABtemp*(VY( 0, 0, 4) - CDcom * VY( 0, 0, 5));
                                    
                                    //FSSS(3, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz, ABtemp, CDcom);
                                    
                                    QUICKDouble x_10_0_3 = Ptempx * x_5_0_3 + WPtempx * x_5_0_4;
                                    QUICKDouble x_11_0_3 = Ptempx * x_4_0_3 + WPtempx * x_4_0_4 + ABtemp * ( x_2_0_3 - CDcom * x_2_0_4);
                                    QUICKDouble x_12_0_3 = Ptempx * x_8_0_3 + WPtempx * x_8_0_4;
                                    QUICKDouble x_13_0_3 = Ptempx * x_6_0_3 + WPtempx * x_6_0_4 + ABtemp * ( x_3_0_3 - CDcom * x_3_0_4);
                                    QUICKDouble x_14_0_3 = Ptempx * x_9_0_3 + WPtempx * x_9_0_4;
                                    QUICKDouble x_15_0_3 = Ptempy * x_5_0_3 + WPtempy * x_5_0_4 + ABtemp * ( x_3_0_3 - CDcom * x_3_0_4);
                                    QUICKDouble x_16_0_3 = Ptempy * x_9_0_3 + WPtempy * x_9_0_4;
                                    QUICKDouble x_17_0_3 = Ptempx * x_7_0_3 + WPtempx * x_7_0_4 +  2 * ABtemp * ( x_1_0_3 - CDcom * x_1_0_4);
                                    QUICKDouble x_18_0_3 = Ptempy * x_8_0_3 + WPtempy * x_8_0_4 +  2 * ABtemp * ( x_2_0_3 - CDcom * x_2_0_4);
                                    QUICKDouble x_19_0_3 = Ptempz * x_9_0_3 + WPtempz * x_9_0_4 +  2 * ABtemp * ( x_3_0_3 - CDcom * x_3_0_4);
                                    
                                    //GSSS(2, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz, ABtemp, CDcom);
                                    
                                    QUICKDouble x_20_0_2 = Ptempx * x_12_0_2 + WPtempx * x_12_0_3 + ABtemp * ( x_8_0_2 - CDcom * x_8_0_3);
                                    QUICKDouble x_21_0_2 = Ptempx * x_14_0_2 + WPtempx * x_14_0_3 + ABtemp * ( x_9_0_2 - CDcom * x_9_0_3);
                                    QUICKDouble x_22_0_2 = Ptempy * x_16_0_2 + WPtempy * x_16_0_3 + ABtemp * ( x_9_0_2 - CDcom * x_9_0_3);
                                    QUICKDouble x_23_0_2 = Ptempx * x_10_0_2 + WPtempx * x_10_0_3 + ABtemp * ( x_5_0_2 - CDcom * x_5_0_3);
                                    QUICKDouble x_24_0_2 = Ptempx * x_15_0_2 + WPtempx * x_15_0_3;
                                    QUICKDouble x_25_0_2 = Ptempx * x_16_0_2 + WPtempx * x_16_0_3;
                                    QUICKDouble x_26_0_2 = Ptempx * x_13_0_2 + WPtempx * x_13_0_3 + 2 * ABtemp * ( x_6_0_2 - CDcom * x_6_0_3);
                                    QUICKDouble x_27_0_2 = Ptempx * x_19_0_2 + WPtempx * x_19_0_3;
                                    QUICKDouble x_28_0_2 = Ptempx * x_11_0_2 + WPtempx * x_11_0_3 + 2 * ABtemp * ( x_4_0_2 - CDcom * x_4_0_3);
                                    QUICKDouble x_29_0_2 = Ptempx * x_18_0_2 + WPtempx * x_18_0_3;
                                    QUICKDouble x_30_0_2 = Ptempy * x_15_0_2 + WPtempy * x_15_0_3 + 2 * ABtemp * ( x_5_0_2 - CDcom * x_5_0_3);
                                    QUICKDouble x_31_0_2 = Ptempy * x_19_0_2 + WPtempy * x_19_0_3;
                                    QUICKDouble x_32_0_2 = Ptempx * x_17_0_2 + WPtempx * x_17_0_3 + 3 * ABtemp * ( x_7_0_2 - CDcom * x_7_0_3);
                                    QUICKDouble x_33_0_2 = Ptempy * x_18_0_2 + WPtempy * x_18_0_3 + 3 * ABtemp * ( x_8_0_2 - CDcom * x_8_0_3);
                                    QUICKDouble x_34_0_2 = Ptempz * x_19_0_2 + WPtempz * x_19_0_3 + 3 * ABtemp * ( x_9_0_2 - CDcom * x_9_0_3);
                                    
                                    
                                    //GSPS(1, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz, ABCDtemp);
                                    
                                    QUICKDouble x_20_1_1 = Qtempx * x_20_0_1 + WQtempx * x_20_0_2 +  2 * ABCDtemp *  x_12_0_2;
                                    QUICKDouble x_20_2_1 = Qtempy * x_20_0_1 + WQtempy * x_20_0_2 +  2 * ABCDtemp *  x_11_0_2;
                                    QUICKDouble x_20_3_1 = Qtempz * x_20_0_1 + WQtempz * x_20_0_2;
                                    QUICKDouble x_21_1_1 = Qtempx * x_21_0_1 + WQtempx * x_21_0_2 +  2 * ABCDtemp *  x_14_0_2;
                                    QUICKDouble x_21_2_1 = Qtempy * x_21_0_1 + WQtempy * x_21_0_2;
                                    QUICKDouble x_21_3_1 = Qtempz * x_21_0_1 + WQtempz * x_21_0_2 +  2 * ABCDtemp *  x_13_0_2;
                                    QUICKDouble x_22_1_1 = Qtempx * x_22_0_1 + WQtempx * x_22_0_2;
                                    QUICKDouble x_22_2_1 = Qtempy * x_22_0_1 + WQtempy * x_22_0_2 +  2 * ABCDtemp *  x_16_0_2;
                                    QUICKDouble x_22_3_1 = Qtempz * x_22_0_1 + WQtempz * x_22_0_2 +  2 * ABCDtemp *  x_15_0_2;
                                    QUICKDouble x_23_1_1 = Qtempx * x_23_0_1 + WQtempx * x_23_0_2 +  2 * ABCDtemp *  x_10_0_2;
                                    QUICKDouble x_23_2_1 = Qtempy * x_23_0_1 + WQtempy * x_23_0_2 + ABCDtemp *  x_13_0_2;
                                    QUICKDouble x_23_3_1 = Qtempz * x_23_0_1 + WQtempz * x_23_0_2 + ABCDtemp *  x_11_0_2;
                                    QUICKDouble x_24_1_1 = Qtempx * x_24_0_1 + WQtempx * x_24_0_2 + ABCDtemp *  x_15_0_2;
                                    QUICKDouble x_24_2_1 = Qtempy * x_24_0_1 + WQtempy * x_24_0_2 +  2 * ABCDtemp *  x_10_0_2;
                                    QUICKDouble x_24_3_1 = Qtempz * x_24_0_1 + WQtempz * x_24_0_2 + ABCDtemp *  x_12_0_2;
                                    QUICKDouble x_25_1_1 = Qtempx * x_25_0_1 + WQtempx * x_25_0_2 + ABCDtemp *  x_16_0_2;
                                    QUICKDouble x_25_2_1 = Qtempy * x_25_0_1 + WQtempy * x_25_0_2 + ABCDtemp *  x_14_0_2;
                                    QUICKDouble x_25_3_1 = Qtempz * x_25_0_1 + WQtempz * x_25_0_2 +  2 * ABCDtemp *  x_10_0_2;
                                    QUICKDouble x_26_1_1 = Qtempx * x_26_0_1 + WQtempx * x_26_0_2 +  3 * ABCDtemp *  x_13_0_2;
                                    QUICKDouble x_26_2_1 = Qtempy * x_26_0_1 + WQtempy * x_26_0_2;
                                    QUICKDouble x_26_3_1 = Qtempz * x_26_0_1 + WQtempz * x_26_0_2 + ABCDtemp *  x_17_0_2;
                                    QUICKDouble x_27_1_1 = Qtempx * x_27_0_1 + WQtempx * x_27_0_2 + ABCDtemp *  x_19_0_2;
                                    QUICKDouble x_27_2_1 = Qtempy * x_27_0_1 + WQtempy * x_27_0_2;
                                    QUICKDouble x_27_3_1 = Qtempz * x_27_0_1 + WQtempz * x_27_0_2 +  3 * ABCDtemp *  x_14_0_2;
                                    QUICKDouble x_28_1_1 = Qtempx * x_28_0_1 + WQtempx * x_28_0_2 +  3 * ABCDtemp *  x_11_0_2;
                                    QUICKDouble x_28_2_1 = Qtempy * x_28_0_1 + WQtempy * x_28_0_2 + ABCDtemp *  x_17_0_2;
                                    QUICKDouble x_28_3_1 = Qtempz * x_28_0_1 + WQtempz * x_28_0_2;
                                    QUICKDouble x_29_1_1 = Qtempx * x_29_0_1 + WQtempx * x_29_0_2 + ABCDtemp *  x_18_0_2;
                                    QUICKDouble x_29_2_1 = Qtempy * x_29_0_1 + WQtempy * x_29_0_2 +  3 * ABCDtemp *  x_12_0_2;
                                    QUICKDouble x_29_3_1 = Qtempz * x_29_0_1 + WQtempz * x_29_0_2;
                                    QUICKDouble x_30_1_1 = Qtempx * x_30_0_1 + WQtempx * x_30_0_2;
                                    QUICKDouble x_30_2_1 = Qtempy * x_30_0_1 + WQtempy * x_30_0_2 +  3 * ABCDtemp *  x_15_0_2;
                                    QUICKDouble x_30_3_1 = Qtempz * x_30_0_1 + WQtempz * x_30_0_2 + ABCDtemp *  x_18_0_2;
                                    QUICKDouble x_31_1_1 = Qtempx * x_31_0_1 + WQtempx * x_31_0_2;
                                    QUICKDouble x_31_2_1 = Qtempy * x_31_0_1 + WQtempy * x_31_0_2 + ABCDtemp *  x_19_0_2;
                                    QUICKDouble x_31_3_1 = Qtempz * x_31_0_1 + WQtempz * x_31_0_2 +  3 * ABCDtemp *  x_16_0_2;    
                                    QUICKDouble x_32_1_1 = Qtempx * x_32_0_1 + WQtempx * x_32_0_2 +  4 * ABCDtemp *  x_17_0_2;
                                    QUICKDouble x_32_2_1 = Qtempy * x_32_0_1 + WQtempy * x_32_0_2;
                                    QUICKDouble x_32_3_1 = Qtempz * x_32_0_1 + WQtempz * x_32_0_2;
                                    QUICKDouble x_33_1_1 = Qtempx * x_33_0_1 + WQtempx * x_33_0_2;
                                    QUICKDouble x_33_2_1 = Qtempy * x_33_0_1 + WQtempy * x_33_0_2 +  4 * ABCDtemp *  x_18_0_2;
                                    QUICKDouble x_33_3_1 = Qtempz * x_33_0_1 + WQtempz * x_33_0_2;
                                    QUICKDouble x_34_1_1 = Qtempx * x_34_0_1 + WQtempx * x_34_0_2;
                                    QUICKDouble x_34_2_1 = Qtempy * x_34_0_1 + WQtempy * x_34_0_2;
                                    QUICKDouble x_34_3_1 = Qtempz * x_34_0_1 + WQtempz * x_34_0_2 +  4 * ABCDtemp *  x_19_0_2;
                                    
                                    
                                    //GSDS(0, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz, ABCDtemp, CDtemp, ABcom);
                                    
                                    QUICKDouble x_20_4_0 = Qtempx * x_20_2_0 + WQtempx * x_20_2_1 +  2 * ABCDtemp * x_12_2_1;
                                    QUICKDouble x_21_4_0 = Qtempx * x_21_2_0 + WQtempx * x_21_2_1 +  2 * ABCDtemp * x_14_2_1;
                                    QUICKDouble x_22_4_0 = Qtempx * x_22_2_0 + WQtempx * x_22_2_1;
                                    QUICKDouble x_23_4_0 = Qtempx * x_23_2_0 + WQtempx * x_23_2_1 +  2 * ABCDtemp * x_10_2_1;
                                    QUICKDouble x_24_4_0 = Qtempx * x_24_2_0 + WQtempx * x_24_2_1 + ABCDtemp * x_15_2_1;
                                    QUICKDouble x_25_4_0 = Qtempx * x_25_2_0 + WQtempx * x_25_2_1 + ABCDtemp * x_16_2_1;
                                    QUICKDouble x_26_4_0 = Qtempx * x_26_2_0 + WQtempx * x_26_2_1 +  3 * ABCDtemp * x_13_2_1;
                                    QUICKDouble x_27_4_0 = Qtempx * x_27_2_0 + WQtempx * x_27_2_1 + ABCDtemp * x_19_2_1;
                                    QUICKDouble x_28_4_0 = Qtempx * x_28_2_0 + WQtempx * x_28_2_1 +  3 * ABCDtemp * x_11_2_1;
                                    QUICKDouble x_29_4_0 = Qtempx * x_29_2_0 + WQtempx * x_29_2_1 + ABCDtemp * x_18_2_1;
                                    QUICKDouble x_30_4_0 = Qtempx * x_30_2_0 + WQtempx * x_30_2_1;
                                    QUICKDouble x_31_4_0 = Qtempx * x_31_2_0 + WQtempx * x_31_2_1;
                                    QUICKDouble x_32_4_0 = Qtempx * x_32_2_0 + WQtempx * x_32_2_1 +  4 * ABCDtemp * x_17_2_1;
                                    QUICKDouble x_33_4_0 = Qtempx * x_33_2_0 + WQtempx * x_33_2_1;
                                    QUICKDouble x_34_4_0 = Qtempx * x_34_2_0 + WQtempx * x_34_2_1;
                                    QUICKDouble x_20_5_0 = Qtempy * x_20_3_0 + WQtempy * x_20_3_1 +  2 * ABCDtemp * x_11_3_1;
                                    QUICKDouble x_21_5_0 = Qtempy * x_21_3_0 + WQtempy * x_21_3_1;
                                    QUICKDouble x_22_5_0 = Qtempy * x_22_3_0 + WQtempy * x_22_3_1 +  2 * ABCDtemp * x_16_3_1;
                                    QUICKDouble x_23_5_0 = Qtempy * x_23_3_0 + WQtempy * x_23_3_1 + ABCDtemp * x_13_3_1;
                                    QUICKDouble x_24_5_0 = Qtempy * x_24_3_0 + WQtempy * x_24_3_1 +  2 * ABCDtemp * x_10_3_1;
                                    QUICKDouble x_25_5_0 = Qtempy * x_25_3_0 + WQtempy * x_25_3_1 + ABCDtemp * x_14_3_1;
                                    QUICKDouble x_26_5_0 = Qtempy * x_26_3_0 + WQtempy * x_26_3_1;
                                    QUICKDouble x_27_5_0 = Qtempy * x_27_3_0 + WQtempy * x_27_3_1;
                                    QUICKDouble x_28_5_0 = Qtempy * x_28_3_0 + WQtempy * x_28_3_1 + ABCDtemp * x_17_3_1;
                                    QUICKDouble x_29_5_0 = Qtempy * x_29_3_0 + WQtempy * x_29_3_1 +  3 * ABCDtemp * x_12_3_1;
                                    QUICKDouble x_30_5_0 = Qtempy * x_30_3_0 + WQtempy * x_30_3_1 +  3 * ABCDtemp * x_15_3_1;
                                    QUICKDouble x_31_5_0 = Qtempy * x_31_3_0 + WQtempy * x_31_3_1 + ABCDtemp * x_19_3_1;
                                    QUICKDouble x_32_5_0 = Qtempy * x_32_3_0 + WQtempy * x_32_3_1;
                                    QUICKDouble x_33_5_0 = Qtempy * x_33_3_0 + WQtempy * x_33_3_1 +  4 * ABCDtemp * x_18_3_1;
                                    QUICKDouble x_34_5_0 = Qtempy * x_34_3_0 + WQtempy * x_34_3_1;
                                    QUICKDouble x_20_6_0 = Qtempx * x_20_3_0 + WQtempx * x_20_3_1 +  2 * ABCDtemp * x_12_3_1;
                                    QUICKDouble x_21_6_0 = Qtempx * x_21_3_0 + WQtempx * x_21_3_1 +  2 * ABCDtemp * x_14_3_1;
                                    QUICKDouble x_22_6_0 = Qtempx * x_22_3_0 + WQtempx * x_22_3_1;
                                    QUICKDouble x_23_6_0 = Qtempx * x_23_3_0 + WQtempx * x_23_3_1 +  2 * ABCDtemp * x_10_3_1;
                                    QUICKDouble x_24_6_0 = Qtempx * x_24_3_0 + WQtempx * x_24_3_1 + ABCDtemp * x_15_3_1;
                                    QUICKDouble x_25_6_0 = Qtempx * x_25_3_0 + WQtempx * x_25_3_1 + ABCDtemp * x_16_3_1;
                                    QUICKDouble x_26_6_0 = Qtempx * x_26_3_0 + WQtempx * x_26_3_1 +  3 * ABCDtemp * x_13_3_1;
                                    QUICKDouble x_27_6_0 = Qtempx * x_27_3_0 + WQtempx * x_27_3_1 + ABCDtemp * x_19_3_1;
                                    QUICKDouble x_28_6_0 = Qtempx * x_28_3_0 + WQtempx * x_28_3_1 +  3 * ABCDtemp * x_11_3_1;
                                    QUICKDouble x_29_6_0 = Qtempx * x_29_3_0 + WQtempx * x_29_3_1 + ABCDtemp * x_18_3_1;
                                    QUICKDouble x_30_6_0 = Qtempx * x_30_3_0 + WQtempx * x_30_3_1;
                                    QUICKDouble x_31_6_0 = Qtempx * x_31_3_0 + WQtempx * x_31_3_1;
                                    QUICKDouble x_32_6_0 = Qtempx * x_32_3_0 + WQtempx * x_32_3_1 +  4 * ABCDtemp * x_17_3_1;
                                    QUICKDouble x_33_6_0 = Qtempx * x_33_3_0 + WQtempx * x_33_3_1;
                                    QUICKDouble x_34_6_0 = Qtempx * x_34_3_0 + WQtempx * x_34_3_1;
                                    QUICKDouble x_20_7_0 = Qtempx * x_20_1_0 + WQtempx * x_20_1_1 + CDtemp * ( x_20_0_0 - ABcom * x_20_0_1) +  2 * ABCDtemp * x_12_1_1;
                                    QUICKDouble x_21_7_0 = Qtempx * x_21_1_0 + WQtempx * x_21_1_1 + CDtemp * ( x_21_0_0 - ABcom * x_21_0_1) +  2 * ABCDtemp * x_14_1_1;
                                    QUICKDouble x_22_7_0 = Qtempx * x_22_1_0 + WQtempx * x_22_1_1 + CDtemp * ( x_22_0_0 - ABcom * x_22_0_1);
                                    QUICKDouble x_23_7_0 = Qtempx * x_23_1_0 + WQtempx * x_23_1_1 + CDtemp * ( x_23_0_0 - ABcom * x_23_0_1) +  2 * ABCDtemp * x_10_1_1;
                                    QUICKDouble x_24_7_0 = Qtempx * x_24_1_0 + WQtempx * x_24_1_1 + CDtemp * ( x_24_0_0 - ABcom * x_24_0_1) + ABCDtemp * x_15_1_1;
                                    QUICKDouble x_25_7_0 = Qtempx * x_25_1_0 + WQtempx * x_25_1_1 + CDtemp * ( x_25_0_0 - ABcom * x_25_0_1) + ABCDtemp * x_16_1_1;
                                    QUICKDouble x_26_7_0 = Qtempx * x_26_1_0 + WQtempx * x_26_1_1 + CDtemp * ( x_26_0_0 - ABcom * x_26_0_1) +  3 * ABCDtemp * x_13_1_1;
                                    QUICKDouble x_27_7_0 = Qtempx * x_27_1_0 + WQtempx * x_27_1_1 + CDtemp * ( x_27_0_0 - ABcom * x_27_0_1) + ABCDtemp * x_19_1_1;
                                    QUICKDouble x_28_7_0 = Qtempx * x_28_1_0 + WQtempx * x_28_1_1 + CDtemp * ( x_28_0_0 - ABcom * x_28_0_1) +  3 * ABCDtemp * x_11_1_1;
                                    QUICKDouble x_29_7_0 = Qtempx * x_29_1_0 + WQtempx * x_29_1_1 + CDtemp * ( x_29_0_0 - ABcom * x_29_0_1) + ABCDtemp * x_18_1_1;
                                    QUICKDouble x_30_7_0 = Qtempx * x_30_1_0 + WQtempx * x_30_1_1 + CDtemp * ( x_30_0_0 - ABcom * x_30_0_1);
                                    QUICKDouble x_31_7_0 = Qtempx * x_31_1_0 + WQtempx * x_31_1_1 + CDtemp * ( x_31_0_0 - ABcom * x_31_0_1);
                                    QUICKDouble x_32_7_0 = Qtempx * x_32_1_0 + WQtempx * x_32_1_1 + CDtemp * ( x_32_0_0 - ABcom * x_32_0_1) +  4 * ABCDtemp * x_17_1_1;
                                    QUICKDouble x_33_7_0 = Qtempx * x_33_1_0 + WQtempx * x_33_1_1 + CDtemp * ( x_33_0_0 - ABcom * x_33_0_1);
                                    QUICKDouble x_34_7_0 = Qtempx * x_34_1_0 + WQtempx * x_34_1_1 + CDtemp * ( x_34_0_0 - ABcom * x_34_0_1);
                                    QUICKDouble x_20_8_0 = Qtempy * x_20_2_0 + WQtempy * x_20_2_1 + CDtemp * ( x_20_0_0 - ABcom * x_20_0_1) +  2 * ABCDtemp * x_11_2_1;
                                    QUICKDouble x_21_8_0 = Qtempy * x_21_2_0 + WQtempy * x_21_2_1 + CDtemp * ( x_21_0_0 - ABcom * x_21_0_1);
                                    QUICKDouble x_22_8_0 = Qtempy * x_22_2_0 + WQtempy * x_22_2_1 + CDtemp * ( x_22_0_0 - ABcom * x_22_0_1) +  2 * ABCDtemp * x_16_2_1;
                                    QUICKDouble x_23_8_0 = Qtempy * x_23_2_0 + WQtempy * x_23_2_1 + CDtemp * ( x_23_0_0 - ABcom * x_23_0_1) + ABCDtemp * x_13_2_1;
                                    QUICKDouble x_24_8_0 = Qtempy * x_24_2_0 + WQtempy * x_24_2_1 + CDtemp * ( x_24_0_0 - ABcom * x_24_0_1) +  2 * ABCDtemp * x_10_2_1;
                                    QUICKDouble x_25_8_0 = Qtempy * x_25_2_0 + WQtempy * x_25_2_1 + CDtemp * ( x_25_0_0 - ABcom * x_25_0_1) + ABCDtemp * x_14_2_1;
                                    QUICKDouble x_26_8_0 = Qtempy * x_26_2_0 + WQtempy * x_26_2_1 + CDtemp * ( x_26_0_0 - ABcom * x_26_0_1);
                                    QUICKDouble x_27_8_0 = Qtempy * x_27_2_0 + WQtempy * x_27_2_1 + CDtemp * ( x_27_0_0 - ABcom * x_27_0_1);
                                    QUICKDouble x_28_8_0 = Qtempy * x_28_2_0 + WQtempy * x_28_2_1 + CDtemp * ( x_28_0_0 - ABcom * x_28_0_1) + ABCDtemp * x_17_2_1;
                                    QUICKDouble x_29_8_0 = Qtempy * x_29_2_0 + WQtempy * x_29_2_1 + CDtemp * ( x_29_0_0 - ABcom * x_29_0_1) +  3 * ABCDtemp * x_12_2_1;
                                    QUICKDouble x_30_8_0 = Qtempy * x_30_2_0 + WQtempy * x_30_2_1 + CDtemp * ( x_30_0_0 - ABcom * x_30_0_1) +  3 * ABCDtemp * x_15_2_1;
                                    QUICKDouble x_31_8_0 = Qtempy * x_31_2_0 + WQtempy * x_31_2_1 + CDtemp * ( x_31_0_0 - ABcom * x_31_0_1) + ABCDtemp * x_19_2_1;
                                    QUICKDouble x_32_8_0 = Qtempy * x_32_2_0 + WQtempy * x_32_2_1 + CDtemp * ( x_32_0_0 - ABcom * x_32_0_1);
                                    QUICKDouble x_33_8_0 = Qtempy * x_33_2_0 + WQtempy * x_33_2_1 + CDtemp * ( x_33_0_0 - ABcom * x_33_0_1) +  4 * ABCDtemp * x_18_2_1;
                                    QUICKDouble x_34_8_0 = Qtempy * x_34_2_0 + WQtempy * x_34_2_1 + CDtemp * ( x_34_0_0 - ABcom * x_34_0_1);
                                    QUICKDouble x_20_9_0 = Qtempz * x_20_3_0 + WQtempz * x_20_3_1 + CDtemp * ( x_20_0_0 - ABcom * x_20_0_1);
                                    QUICKDouble x_21_9_0 = Qtempz * x_21_3_0 + WQtempz * x_21_3_1 + CDtemp * ( x_21_0_0 - ABcom * x_21_0_1) +  2 * ABCDtemp * x_13_3_1;
                                    QUICKDouble x_22_9_0 = Qtempz * x_22_3_0 + WQtempz * x_22_3_1 + CDtemp * ( x_22_0_0 - ABcom * x_22_0_1) +  2 * ABCDtemp * x_15_3_1;
                                    QUICKDouble x_23_9_0 = Qtempz * x_23_3_0 + WQtempz * x_23_3_1 + CDtemp * ( x_23_0_0 - ABcom * x_23_0_1) + ABCDtemp * x_11_3_1;
                                    QUICKDouble x_24_9_0 = Qtempz * x_24_3_0 + WQtempz * x_24_3_1 + CDtemp * ( x_24_0_0 - ABcom * x_24_0_1) + ABCDtemp * x_12_3_1;
                                    QUICKDouble x_25_9_0 = Qtempz * x_25_3_0 + WQtempz * x_25_3_1 + CDtemp * ( x_25_0_0 - ABcom * x_25_0_1) +  2 * ABCDtemp * x_10_3_1;
                                    QUICKDouble x_26_9_0 = Qtempz * x_26_3_0 + WQtempz * x_26_3_1 + CDtemp * ( x_26_0_0 - ABcom * x_26_0_1) + ABCDtemp * x_17_3_1;
                                    QUICKDouble x_27_9_0 = Qtempz * x_27_3_0 + WQtempz * x_27_3_1 + CDtemp * ( x_27_0_0 - ABcom * x_27_0_1) +  3 * ABCDtemp * x_14_3_1;
                                    QUICKDouble x_28_9_0 = Qtempz * x_28_3_0 + WQtempz * x_28_3_1 + CDtemp * ( x_28_0_0 - ABcom * x_28_0_1);
                                    QUICKDouble x_29_9_0 = Qtempz * x_29_3_0 + WQtempz * x_29_3_1 + CDtemp * ( x_29_0_0 - ABcom * x_29_0_1);
                                    QUICKDouble x_30_9_0 = Qtempz * x_30_3_0 + WQtempz * x_30_3_1 + CDtemp * ( x_30_0_0 - ABcom * x_30_0_1) + ABCDtemp * x_18_3_1;
                                    QUICKDouble x_31_9_0 = Qtempz * x_31_3_0 + WQtempz * x_31_3_1 + CDtemp * ( x_31_0_0 - ABcom * x_31_0_1) +  3 * ABCDtemp * x_16_3_1;
                                    QUICKDouble x_32_9_0 = Qtempz * x_32_3_0 + WQtempz * x_32_3_1 + CDtemp * ( x_32_0_0 - ABcom * x_32_0_1);
                                    QUICKDouble x_33_9_0 = Qtempz * x_33_3_0 + WQtempz * x_33_3_1 + CDtemp * ( x_33_0_0 - ABcom * x_33_0_1);
                                    QUICKDouble x_34_9_0 = Qtempz * x_34_3_0 + WQtempz * x_34_3_1 + CDtemp * ( x_34_0_0 - ABcom * x_34_0_1) +  4 * ABCDtemp * x_19_3_1;
                                    
                                    LOC2(store,20, 4, STOREDIM, STOREDIM) += x_20_4_0;
                                    LOC2(store,20, 5, STOREDIM, STOREDIM) += x_20_5_0;
                                    LOC2(store,20, 6, STOREDIM, STOREDIM) += x_20_6_0;
                                    LOC2(store,20, 7, STOREDIM, STOREDIM) += x_20_7_0;
                                    LOC2(store,20, 8, STOREDIM, STOREDIM) += x_20_8_0;
                                    LOC2(store,20, 9, STOREDIM, STOREDIM) += x_20_9_0;
                                    LOC2(store,21, 4, STOREDIM, STOREDIM) += x_21_4_0;
                                    LOC2(store,21, 5, STOREDIM, STOREDIM) += x_21_5_0;
                                    LOC2(store,21, 6, STOREDIM, STOREDIM) += x_21_6_0;
                                    LOC2(store,21, 7, STOREDIM, STOREDIM) += x_21_7_0;
                                    LOC2(store,21, 8, STOREDIM, STOREDIM) += x_21_8_0;
                                    LOC2(store,21, 9, STOREDIM, STOREDIM) += x_21_9_0;
                                    LOC2(store,22, 4, STOREDIM, STOREDIM) += x_22_4_0;
                                    LOC2(store,22, 5, STOREDIM, STOREDIM) += x_22_5_0;
                                    LOC2(store,22, 6, STOREDIM, STOREDIM) += x_22_6_0;
                                    LOC2(store,22, 7, STOREDIM, STOREDIM) += x_22_7_0;
                                    LOC2(store,22, 8, STOREDIM, STOREDIM) += x_22_8_0;
                                    LOC2(store,22, 9, STOREDIM, STOREDIM) += x_22_9_0;
                                    LOC2(store,23, 4, STOREDIM, STOREDIM) += x_23_4_0;
                                    LOC2(store,23, 5, STOREDIM, STOREDIM) += x_23_5_0;
                                    LOC2(store,23, 6, STOREDIM, STOREDIM) += x_23_6_0;
                                    LOC2(store,23, 7, STOREDIM, STOREDIM) += x_23_7_0;
                                    LOC2(store,23, 8, STOREDIM, STOREDIM) += x_23_8_0;
                                    LOC2(store,23, 9, STOREDIM, STOREDIM) += x_23_9_0;
                                    LOC2(store,24, 4, STOREDIM, STOREDIM) += x_24_4_0;
                                    LOC2(store,24, 5, STOREDIM, STOREDIM) += x_24_5_0;
                                    LOC2(store,24, 6, STOREDIM, STOREDIM) += x_24_6_0;
                                    LOC2(store,24, 7, STOREDIM, STOREDIM) += x_24_7_0;
                                    LOC2(store,24, 8, STOREDIM, STOREDIM) += x_24_8_0;
                                    LOC2(store,24, 9, STOREDIM, STOREDIM) += x_24_9_0;
                                    LOC2(store,25, 4, STOREDIM, STOREDIM) += x_25_4_0;
                                    LOC2(store,25, 5, STOREDIM, STOREDIM) += x_25_5_0;
                                    LOC2(store,25, 6, STOREDIM, STOREDIM) += x_25_6_0;
                                    LOC2(store,25, 7, STOREDIM, STOREDIM) += x_25_7_0;
                                    LOC2(store,25, 8, STOREDIM, STOREDIM) += x_25_8_0;
                                    LOC2(store,25, 9, STOREDIM, STOREDIM) += x_25_9_0;
                                    LOC2(store,26, 4, STOREDIM, STOREDIM) += x_26_4_0;
                                    LOC2(store,26, 5, STOREDIM, STOREDIM) += x_26_5_0;
                                    LOC2(store,26, 6, STOREDIM, STOREDIM) += x_26_6_0;
                                    LOC2(store,26, 7, STOREDIM, STOREDIM) += x_26_7_0;
                                    LOC2(store,26, 8, STOREDIM, STOREDIM) += x_26_8_0;
                                    LOC2(store,26, 9, STOREDIM, STOREDIM) += x_26_9_0;
                                    LOC2(store,27, 4, STOREDIM, STOREDIM) += x_27_4_0;
                                    LOC2(store,27, 5, STOREDIM, STOREDIM) += x_27_5_0;
                                    LOC2(store,27, 6, STOREDIM, STOREDIM) += x_27_6_0;
                                    LOC2(store,27, 7, STOREDIM, STOREDIM) += x_27_7_0;
                                    LOC2(store,27, 8, STOREDIM, STOREDIM) += x_27_8_0;
                                    LOC2(store,27, 9, STOREDIM, STOREDIM) += x_27_9_0;
                                    LOC2(store,28, 4, STOREDIM, STOREDIM) += x_28_4_0;
                                    LOC2(store,28, 5, STOREDIM, STOREDIM) += x_28_5_0;
                                    LOC2(store,28, 6, STOREDIM, STOREDIM) += x_28_6_0;
                                    LOC2(store,28, 7, STOREDIM, STOREDIM) += x_28_7_0;
                                    LOC2(store,28, 8, STOREDIM, STOREDIM) += x_28_8_0;
                                    LOC2(store,28, 9, STOREDIM, STOREDIM) += x_28_9_0;
                                    LOC2(store,29, 4, STOREDIM, STOREDIM) += x_29_4_0;
                                    LOC2(store,29, 5, STOREDIM, STOREDIM) += x_29_5_0;
                                    LOC2(store,29, 6, STOREDIM, STOREDIM) += x_29_6_0;
                                    LOC2(store,29, 7, STOREDIM, STOREDIM) += x_29_7_0;
                                    LOC2(store,29, 8, STOREDIM, STOREDIM) += x_29_8_0;
                                    LOC2(store,29, 9, STOREDIM, STOREDIM) += x_29_9_0;
                                    LOC2(store,30, 4, STOREDIM, STOREDIM) += x_30_4_0;
                                    LOC2(store,30, 5, STOREDIM, STOREDIM) += x_30_5_0;
                                    LOC2(store,30, 6, STOREDIM, STOREDIM) += x_30_6_0;
                                    LOC2(store,30, 7, STOREDIM, STOREDIM) += x_30_7_0;
                                    LOC2(store,30, 8, STOREDIM, STOREDIM) += x_30_8_0;
                                    LOC2(store,30, 9, STOREDIM, STOREDIM) += x_30_9_0;
                                    LOC2(store,31, 4, STOREDIM, STOREDIM) += x_31_4_0;
                                    LOC2(store,31, 5, STOREDIM, STOREDIM) += x_31_5_0;
                                    LOC2(store,31, 6, STOREDIM, STOREDIM) += x_31_6_0;
                                    LOC2(store,31, 7, STOREDIM, STOREDIM) += x_31_7_0;
                                    LOC2(store,31, 8, STOREDIM, STOREDIM) += x_31_8_0;
                                    LOC2(store,31, 9, STOREDIM, STOREDIM) += x_31_9_0;
                                    LOC2(store,32, 4, STOREDIM, STOREDIM) += x_32_4_0;
                                    LOC2(store,32, 5, STOREDIM, STOREDIM) += x_32_5_0;
                                    LOC2(store,32, 6, STOREDIM, STOREDIM) += x_32_6_0;
                                    LOC2(store,32, 7, STOREDIM, STOREDIM) += x_32_7_0;
                                    LOC2(store,32, 8, STOREDIM, STOREDIM) += x_32_8_0;
                                    LOC2(store,32, 9, STOREDIM, STOREDIM) += x_32_9_0;
                                    LOC2(store,33, 4, STOREDIM, STOREDIM) += x_33_4_0;
                                    LOC2(store,33, 5, STOREDIM, STOREDIM) += x_33_5_0;
                                    LOC2(store,33, 6, STOREDIM, STOREDIM) += x_33_6_0;
                                    LOC2(store,33, 7, STOREDIM, STOREDIM) += x_33_7_0;
                                    LOC2(store,33, 8, STOREDIM, STOREDIM) += x_33_8_0;
                                    LOC2(store,33, 9, STOREDIM, STOREDIM) += x_33_9_0;
                                    LOC2(store,34, 4, STOREDIM, STOREDIM) += x_34_4_0;
                                    LOC2(store,34, 5, STOREDIM, STOREDIM) += x_34_5_0;
                                    LOC2(store,34, 6, STOREDIM, STOREDIM) += x_34_6_0;
                                    LOC2(store,34, 7, STOREDIM, STOREDIM) += x_34_7_0;
                                    LOC2(store,34, 8, STOREDIM, STOREDIM) += x_34_8_0;
                                    LOC2(store,34, 9, STOREDIM, STOREDIM) += x_34_9_0;
                                    
                                    if (I+J==4 && K+L>=3){
                                        //PSSS(6, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);
                                        QUICKDouble x_1_0_6 = Ptempx * VY( 0, 0, 6) + WPtempx * VY( 0, 0, 7);
                                        QUICKDouble x_2_0_6 = Ptempy * VY( 0, 0, 6) + WPtempy * VY( 0, 0, 7);
                                        QUICKDouble x_3_0_6 = Ptempz * VY( 0, 0, 6) + WPtempz * VY( 0, 0, 7);
                                        
                                        //DSSS(5, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz, ABtemp, CDcom);
                                        QUICKDouble x_4_0_5 = Ptempx * x_2_0_5 + WPtempx * x_2_0_6;
                                        QUICKDouble x_5_0_5 = Ptempy * x_3_0_5 + WPtempy * x_3_0_6;
                                        QUICKDouble x_6_0_5 = Ptempx * x_3_0_5 + WPtempx * x_3_0_6;
                                        
                                        QUICKDouble x_7_0_5 = Ptempx * x_1_0_5 + WPtempx * x_1_0_6+ ABtemp*(VY( 0, 0, 5) - CDcom * VY( 0, 0, 6));
                                        QUICKDouble x_8_0_5 = Ptempy * x_2_0_5 + WPtempy * x_2_0_6+ ABtemp*(VY( 0, 0, 5) - CDcom * VY( 0, 0, 6));
                                        QUICKDouble x_9_0_5 = Ptempz * x_3_0_5 + WPtempz * x_3_0_6+ ABtemp*(VY( 0, 0, 5) - CDcom * VY( 0, 0, 6));
                                        
                                        //FSSS(4, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz, ABtemp, CDcom);
                                        
                                        QUICKDouble x_10_0_4 = Ptempx * x_5_0_4 + WPtempx * x_5_0_5;
                                        QUICKDouble x_11_0_4 = Ptempx * x_4_0_4 + WPtempx * x_4_0_5 + ABtemp * ( x_2_0_4 - CDcom * x_2_0_5);
                                        QUICKDouble x_12_0_4 = Ptempx * x_8_0_4 + WPtempx * x_8_0_5;
                                        QUICKDouble x_13_0_4 = Ptempx * x_6_0_4 + WPtempx * x_6_0_5 + ABtemp * ( x_3_0_4 - CDcom * x_3_0_5);
                                        QUICKDouble x_14_0_4 = Ptempx * x_9_0_4 + WPtempx * x_9_0_5;
                                        QUICKDouble x_15_0_4 = Ptempy * x_5_0_4 + WPtempy * x_5_0_5 + ABtemp * ( x_3_0_4 - CDcom * x_3_0_5);
                                        QUICKDouble x_16_0_4 = Ptempy * x_9_0_4 + WPtempy * x_9_0_5;
                                        QUICKDouble x_17_0_4 = Ptempx * x_7_0_4 + WPtempx * x_7_0_5 +  2 * ABtemp * ( x_1_0_4 - CDcom * x_1_0_5);
                                        QUICKDouble x_18_0_4 = Ptempy * x_8_0_4 + WPtempy * x_8_0_5 +  2 * ABtemp * ( x_2_0_4 - CDcom * x_2_0_5);
                                        QUICKDouble x_19_0_4 = Ptempz * x_9_0_4 + WPtempz * x_9_0_5 +  2 * ABtemp * ( x_3_0_4 - CDcom * x_3_0_5);
                                        
                                        //GSSS(3, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz, ABtemp, CDcom);
                                        
                                        QUICKDouble x_20_0_3 = Ptempx * x_12_0_3 + WPtempx * x_12_0_4 + ABtemp * ( x_8_0_3 - CDcom * x_8_0_4);
                                        QUICKDouble x_21_0_3 = Ptempx * x_14_0_3 + WPtempx * x_14_0_4 + ABtemp * ( x_9_0_3 - CDcom * x_9_0_4);
                                        QUICKDouble x_22_0_3 = Ptempy * x_16_0_3 + WPtempy * x_16_0_4 + ABtemp * ( x_9_0_3 - CDcom * x_9_0_4);
                                        QUICKDouble x_23_0_3 = Ptempx * x_10_0_3 + WPtempx * x_10_0_4 + ABtemp * ( x_5_0_3 - CDcom * x_5_0_4);
                                        QUICKDouble x_24_0_3 = Ptempx * x_15_0_3 + WPtempx * x_15_0_4;
                                        QUICKDouble x_25_0_3 = Ptempx * x_16_0_3 + WPtempx * x_16_0_4;
                                        QUICKDouble x_26_0_3 = Ptempx * x_13_0_3 + WPtempx * x_13_0_4 + 2 * ABtemp * ( x_6_0_3 - CDcom * x_6_0_4);
                                        QUICKDouble x_27_0_3 = Ptempx * x_19_0_3 + WPtempx * x_19_0_4;
                                        QUICKDouble x_28_0_3 = Ptempx * x_11_0_3 + WPtempx * x_11_0_4 + 2 * ABtemp * ( x_4_0_3 - CDcom * x_4_0_4);
                                        QUICKDouble x_29_0_3 = Ptempx * x_18_0_3 + WPtempx * x_18_0_4;
                                        QUICKDouble x_30_0_3 = Ptempy * x_15_0_3 + WPtempy * x_15_0_4 + 2 * ABtemp * ( x_5_0_3 - CDcom * x_5_0_4);
                                        QUICKDouble x_31_0_3 = Ptempy * x_19_0_3 + WPtempy * x_19_0_4;
                                        QUICKDouble x_32_0_3 = Ptempx * x_17_0_3 + WPtempx * x_17_0_4 + 3 * ABtemp * ( x_7_0_3 - CDcom * x_7_0_4);
                                        QUICKDouble x_33_0_3 = Ptempy * x_18_0_3 + WPtempy * x_18_0_4 + 3 * ABtemp * ( x_8_0_3 - CDcom * x_8_0_4);
                                        QUICKDouble x_34_0_3 = Ptempz * x_19_0_3 + WPtempz * x_19_0_4 + 3 * ABtemp * ( x_9_0_3 - CDcom * x_9_0_4);
                                        
                                        //FSPS(2, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz, ABCDtemp);
                                        
                                        QUICKDouble x_10_1_2 = Qtempx * x_10_0_2 + WQtempx * x_10_0_3 + ABCDtemp *  x_5_0_3;
                                        QUICKDouble x_10_2_2 = Qtempy * x_10_0_2 + WQtempy * x_10_0_3 + ABCDtemp *  x_6_0_3;
                                        QUICKDouble x_10_3_2 = Qtempz * x_10_0_2 + WQtempz * x_10_0_3 + ABCDtemp *  x_4_0_3;
                                        QUICKDouble x_11_1_2 = Qtempx * x_11_0_2 + WQtempx * x_11_0_3 +  2 * ABCDtemp *  x_4_0_3;
                                        QUICKDouble x_11_2_2 = Qtempy * x_11_0_2 + WQtempy * x_11_0_3 + ABCDtemp *  x_7_0_3;
                                        QUICKDouble x_11_3_2 = Qtempz * x_11_0_2 + WQtempz * x_11_0_3;
                                        QUICKDouble x_12_1_2 = Qtempx * x_12_0_2 + WQtempx * x_12_0_3 + ABCDtemp *  x_8_0_3;
                                        QUICKDouble x_12_2_2 = Qtempy * x_12_0_2 + WQtempy * x_12_0_3 +  2 * ABCDtemp *  x_4_0_3;
                                        QUICKDouble x_12_3_2 = Qtempz * x_12_0_2 + WQtempz * x_12_0_3;
                                        QUICKDouble x_13_1_2 = Qtempx * x_13_0_2 + WQtempx * x_13_0_3 +  2 * ABCDtemp *  x_6_0_3;
                                        QUICKDouble x_13_2_2 = Qtempy * x_13_0_2 + WQtempy * x_13_0_3;
                                        QUICKDouble x_13_3_2 = Qtempz * x_13_0_2 + WQtempz * x_13_0_3 + ABCDtemp *  x_7_0_3;
                                        QUICKDouble x_14_1_2 = Qtempx * x_14_0_2 + WQtempx * x_14_0_3 + ABCDtemp *  x_9_0_3;
                                        QUICKDouble x_14_2_2 = Qtempy * x_14_0_2 + WQtempy * x_14_0_3;
                                        QUICKDouble x_14_3_2 = Qtempz * x_14_0_2 + WQtempz * x_14_0_3 +  2 * ABCDtemp *  x_6_0_3;
                                        QUICKDouble x_15_1_2 = Qtempx * x_15_0_2 + WQtempx * x_15_0_3;
                                        QUICKDouble x_15_2_2 = Qtempy * x_15_0_2 + WQtempy * x_15_0_3 +  2 * ABCDtemp *  x_5_0_3;
                                        QUICKDouble x_15_3_2 = Qtempz * x_15_0_2 + WQtempz * x_15_0_3 + ABCDtemp *  x_8_0_3;
                                        QUICKDouble x_16_1_2 = Qtempx * x_16_0_2 + WQtempx * x_16_0_3;
                                        QUICKDouble x_16_2_2 = Qtempy * x_16_0_2 + WQtempy * x_16_0_3 + ABCDtemp *  x_9_0_3;
                                        QUICKDouble x_16_3_2 = Qtempz * x_16_0_2 + WQtempz * x_16_0_3 +  2 * ABCDtemp *  x_5_0_3;
                                        QUICKDouble x_17_1_2 = Qtempx * x_17_0_2 + WQtempx * x_17_0_3 +  3 * ABCDtemp *  x_7_0_3;
                                        QUICKDouble x_17_2_2 = Qtempy * x_17_0_2 + WQtempy * x_17_0_3;
                                        QUICKDouble x_17_3_2 = Qtempz * x_17_0_2 + WQtempz * x_17_0_3;
                                        QUICKDouble x_18_1_2 = Qtempx * x_18_0_2 + WQtempx * x_18_0_3;
                                        QUICKDouble x_18_2_2 = Qtempy * x_18_0_2 + WQtempy * x_18_0_3 +  3 * ABCDtemp *  x_8_0_3;
                                        QUICKDouble x_18_3_2 = Qtempz * x_18_0_2 + WQtempz * x_18_0_3;
                                        QUICKDouble x_19_1_2 = Qtempx * x_19_0_2 + WQtempx * x_19_0_3;
                                        QUICKDouble x_19_2_2 = Qtempy * x_19_0_2 + WQtempy * x_19_0_3;
                                        QUICKDouble x_19_3_2 = Qtempz * x_19_0_2 + WQtempz * x_19_0_3 +  3 * ABCDtemp *  x_9_0_3;
                                        
                                        //GSPS(2, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz, ABCDtemp);
                                        
                                        QUICKDouble x_20_1_2 = Qtempx * x_20_0_2 + WQtempx * x_20_0_3 +  2 * ABCDtemp *  x_12_0_3;
                                        QUICKDouble x_20_2_2 = Qtempy * x_20_0_2 + WQtempy * x_20_0_3 +  2 * ABCDtemp *  x_11_0_3;
                                        QUICKDouble x_20_3_2 = Qtempz * x_20_0_2 + WQtempz * x_20_0_3;
                                        QUICKDouble x_21_1_2 = Qtempx * x_21_0_2 + WQtempx * x_21_0_3 +  2 * ABCDtemp *  x_14_0_3;
                                        QUICKDouble x_21_2_2 = Qtempy * x_21_0_2 + WQtempy * x_21_0_3;
                                        QUICKDouble x_21_3_2 = Qtempz * x_21_0_2 + WQtempz * x_21_0_3 +  2 * ABCDtemp *  x_13_0_3;
                                        QUICKDouble x_22_1_2 = Qtempx * x_22_0_2 + WQtempx * x_22_0_3;
                                        QUICKDouble x_22_2_2 = Qtempy * x_22_0_2 + WQtempy * x_22_0_3 +  2 * ABCDtemp *  x_16_0_3;
                                        QUICKDouble x_22_3_2 = Qtempz * x_22_0_2 + WQtempz * x_22_0_3 +  2 * ABCDtemp *  x_15_0_3;
                                        QUICKDouble x_23_1_2 = Qtempx * x_23_0_2 + WQtempx * x_23_0_3 +  2 * ABCDtemp *  x_10_0_3;
                                        QUICKDouble x_23_2_2 = Qtempy * x_23_0_2 + WQtempy * x_23_0_3 + ABCDtemp *  x_13_0_3;
                                        QUICKDouble x_23_3_2 = Qtempz * x_23_0_2 + WQtempz * x_23_0_3 + ABCDtemp *  x_11_0_3;
                                        QUICKDouble x_24_1_2 = Qtempx * x_24_0_2 + WQtempx * x_24_0_3 + ABCDtemp *  x_15_0_3;
                                        QUICKDouble x_24_2_2 = Qtempy * x_24_0_2 + WQtempy * x_24_0_3 +  2 * ABCDtemp *  x_10_0_3;
                                        QUICKDouble x_24_3_2 = Qtempz * x_24_0_2 + WQtempz * x_24_0_3 + ABCDtemp *  x_12_0_3;
                                        QUICKDouble x_25_1_2 = Qtempx * x_25_0_2 + WQtempx * x_25_0_3 + ABCDtemp *  x_16_0_3;
                                        QUICKDouble x_25_2_2 = Qtempy * x_25_0_2 + WQtempy * x_25_0_3 + ABCDtemp *  x_14_0_3;
                                        QUICKDouble x_25_3_2 = Qtempz * x_25_0_2 + WQtempz * x_25_0_3 +  2 * ABCDtemp *  x_10_0_3;
                                        QUICKDouble x_26_1_2 = Qtempx * x_26_0_2 + WQtempx * x_26_0_3 +  3 * ABCDtemp *  x_13_0_3;
                                        QUICKDouble x_26_2_2 = Qtempy * x_26_0_2 + WQtempy * x_26_0_3;
                                        QUICKDouble x_26_3_2 = Qtempz * x_26_0_2 + WQtempz * x_26_0_3 + ABCDtemp *  x_17_0_3;
                                        QUICKDouble x_27_1_2 = Qtempx * x_27_0_2 + WQtempx * x_27_0_3 + ABCDtemp *  x_19_0_3;
                                        QUICKDouble x_27_2_2 = Qtempy * x_27_0_2 + WQtempy * x_27_0_3;
                                        QUICKDouble x_27_3_2 = Qtempz * x_27_0_2 + WQtempz * x_27_0_3 +  3 * ABCDtemp *  x_14_0_3;
                                        QUICKDouble x_28_1_2 = Qtempx * x_28_0_2 + WQtempx * x_28_0_3 +  3 * ABCDtemp *  x_11_0_3;
                                        QUICKDouble x_28_2_2 = Qtempy * x_28_0_2 + WQtempy * x_28_0_3 + ABCDtemp *  x_17_0_3;
                                        QUICKDouble x_28_3_2 = Qtempz * x_28_0_2 + WQtempz * x_28_0_3;
                                        QUICKDouble x_29_1_2 = Qtempx * x_29_0_2 + WQtempx * x_29_0_3 + ABCDtemp *  x_18_0_3;
                                        QUICKDouble x_29_2_2 = Qtempy * x_29_0_2 + WQtempy * x_29_0_3 +  3 * ABCDtemp *  x_12_0_3;
                                        QUICKDouble x_29_3_2 = Qtempz * x_29_0_2 + WQtempz * x_29_0_3;
                                        QUICKDouble x_30_1_2 = Qtempx * x_30_0_2 + WQtempx * x_30_0_3;
                                        QUICKDouble x_30_2_2 = Qtempy * x_30_0_2 + WQtempy * x_30_0_3 +  3 * ABCDtemp *  x_15_0_3;
                                        QUICKDouble x_30_3_2 = Qtempz * x_30_0_2 + WQtempz * x_30_0_3 + ABCDtemp *  x_18_0_3;
                                        QUICKDouble x_31_1_2 = Qtempx * x_31_0_2 + WQtempx * x_31_0_3;
                                        QUICKDouble x_31_2_2 = Qtempy * x_31_0_2 + WQtempy * x_31_0_3 + ABCDtemp *  x_19_0_3;
                                        QUICKDouble x_31_3_2 = Qtempz * x_31_0_2 + WQtempz * x_31_0_3 +  3 * ABCDtemp *  x_16_0_3;    
                                        QUICKDouble x_32_1_2 = Qtempx * x_32_0_2 + WQtempx * x_32_0_3 +  4 * ABCDtemp *  x_17_0_3;
                                        QUICKDouble x_32_2_2 = Qtempy * x_32_0_2 + WQtempy * x_32_0_3;
                                        QUICKDouble x_32_3_2 = Qtempz * x_32_0_2 + WQtempz * x_32_0_3;
                                        QUICKDouble x_33_1_2 = Qtempx * x_33_0_2 + WQtempx * x_33_0_3;
                                        QUICKDouble x_33_2_2 = Qtempy * x_33_0_2 + WQtempy * x_33_0_3 +  4 * ABCDtemp *  x_18_0_3;
                                        QUICKDouble x_33_3_2 = Qtempz * x_33_0_2 + WQtempz * x_33_0_3;
                                        QUICKDouble x_34_1_2 = Qtempx * x_34_0_2 + WQtempx * x_34_0_3;
                                        QUICKDouble x_34_2_2 = Qtempy * x_34_0_2 + WQtempy * x_34_0_3;
                                        QUICKDouble x_34_3_2 = Qtempz * x_34_0_2 + WQtempz * x_34_0_3 +  4 * ABCDtemp *  x_19_0_3;
                                        
                                        //PSSS(7, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);
                                        QUICKDouble x_1_0_7 = Ptempx * VY( 0, 0, 7) + WPtempx * VY( 0, 0, 8);
                                        QUICKDouble x_2_0_7 = Ptempy * VY( 0, 0, 7) + WPtempy * VY( 0, 0, 8);
                                        QUICKDouble x_3_0_7 = Ptempz * VY( 0, 0, 7) + WPtempz * VY( 0, 0, 8);
                                        
                                        //DSSS(6, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz, ABtemp, CDcom);
                                        QUICKDouble x_4_0_6 = Ptempx * x_2_0_6 + WPtempx * x_2_0_7;
                                        QUICKDouble x_5_0_6 = Ptempy * x_3_0_6 + WPtempy * x_3_0_7;
                                        QUICKDouble x_6_0_6 = Ptempx * x_3_0_6 + WPtempx * x_3_0_7;
                                        
                                        QUICKDouble x_7_0_6 = Ptempx * x_1_0_6 + WPtempx * x_1_0_7+ ABtemp*(VY( 0, 0, 6) - CDcom * VY( 0, 0, 7));
                                        QUICKDouble x_8_0_6 = Ptempy * x_2_0_6 + WPtempy * x_2_0_7+ ABtemp*(VY( 0, 0, 6) - CDcom * VY( 0, 0, 7));
                                        QUICKDouble x_9_0_6 = Ptempz * x_3_0_6 + WPtempz * x_3_0_7+ ABtemp*(VY( 0, 0, 6) - CDcom * VY( 0, 0, 7));
                                        
                                        //FSSS(5, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz, ABtemp, CDcom);
                                        
                                        QUICKDouble x_10_0_5 = Ptempx * x_5_0_5 + WPtempx * x_5_0_6;
                                        QUICKDouble x_11_0_5 = Ptempx * x_4_0_5 + WPtempx * x_4_0_6 + ABtemp * ( x_2_0_5 - CDcom * x_2_0_6);
                                        QUICKDouble x_12_0_5 = Ptempx * x_8_0_5 + WPtempx * x_8_0_6;
                                        QUICKDouble x_13_0_5 = Ptempx * x_6_0_5 + WPtempx * x_6_0_6 + ABtemp * ( x_3_0_5 - CDcom * x_3_0_6);
                                        QUICKDouble x_14_0_5 = Ptempx * x_9_0_5 + WPtempx * x_9_0_6;
                                        QUICKDouble x_15_0_5 = Ptempy * x_5_0_5 + WPtempy * x_5_0_6 + ABtemp * ( x_3_0_5 - CDcom * x_3_0_6);
                                        QUICKDouble x_16_0_5 = Ptempy * x_9_0_5 + WPtempy * x_9_0_6;
                                        QUICKDouble x_17_0_5 = Ptempx * x_7_0_5 + WPtempx * x_7_0_6 +  2 * ABtemp * ( x_1_0_5 - CDcom * x_1_0_6);
                                        QUICKDouble x_18_0_5 = Ptempy * x_8_0_5 + WPtempy * x_8_0_6 +  2 * ABtemp * ( x_2_0_5 - CDcom * x_2_0_6);
                                        QUICKDouble x_19_0_5 = Ptempz * x_9_0_5 + WPtempz * x_9_0_6 +  2 * ABtemp * ( x_3_0_5 - CDcom * x_3_0_6);
                                        
                                        //GSSS(4, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz, ABtemp, CDcom);
                                        
                                        QUICKDouble x_20_0_4 = Ptempx * x_12_0_4 + WPtempx * x_12_0_5 + ABtemp * ( x_8_0_4 - CDcom * x_8_0_5);
                                        QUICKDouble x_21_0_4 = Ptempx * x_14_0_4 + WPtempx * x_14_0_5 + ABtemp * ( x_9_0_4 - CDcom * x_9_0_5);
                                        QUICKDouble x_22_0_4 = Ptempy * x_16_0_4 + WPtempy * x_16_0_5 + ABtemp * ( x_9_0_4 - CDcom * x_9_0_5);
                                        QUICKDouble x_23_0_4 = Ptempx * x_10_0_4 + WPtempx * x_10_0_5 + ABtemp * ( x_5_0_4 - CDcom * x_5_0_5);
                                        QUICKDouble x_24_0_4 = Ptempx * x_15_0_4 + WPtempx * x_15_0_5;
                                        QUICKDouble x_25_0_4 = Ptempx * x_16_0_4 + WPtempx * x_16_0_5;
                                        QUICKDouble x_26_0_4 = Ptempx * x_13_0_4 + WPtempx * x_13_0_5 + 2 * ABtemp * ( x_6_0_4 - CDcom * x_6_0_5);
                                        QUICKDouble x_27_0_4 = Ptempx * x_19_0_4 + WPtempx * x_19_0_5;
                                        QUICKDouble x_28_0_4 = Ptempx * x_11_0_4 + WPtempx * x_11_0_5 + 2 * ABtemp * ( x_4_0_4 - CDcom * x_4_0_5);
                                        QUICKDouble x_29_0_4 = Ptempx * x_18_0_4 + WPtempx * x_18_0_5;
                                        QUICKDouble x_30_0_4 = Ptempy * x_15_0_4 + WPtempy * x_15_0_5 + 2 * ABtemp * ( x_5_0_4 - CDcom * x_5_0_5);
                                        QUICKDouble x_31_0_4 = Ptempy * x_19_0_4 + WPtempy * x_19_0_5;
                                        QUICKDouble x_32_0_4 = Ptempx * x_17_0_4 + WPtempx * x_17_0_5 + 3 * ABtemp * ( x_7_0_4 - CDcom * x_7_0_5);
                                        QUICKDouble x_33_0_4 = Ptempy * x_18_0_4 + WPtempy * x_18_0_5 + 3 * ABtemp * ( x_8_0_4 - CDcom * x_8_0_5);
                                        QUICKDouble x_34_0_4 = Ptempz * x_19_0_4 + WPtempz * x_19_0_5 + 3 * ABtemp * ( x_9_0_4 - CDcom * x_9_0_5);
                                        
                                        //FSPS(3, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz, ABCDtemp);
                                        
                                        QUICKDouble x_10_1_3 = Qtempx * x_10_0_3 + WQtempx * x_10_0_4 + ABCDtemp *  x_5_0_4;
                                        QUICKDouble x_10_2_3 = Qtempy * x_10_0_3 + WQtempy * x_10_0_4 + ABCDtemp *  x_6_0_4;
                                        QUICKDouble x_10_3_3 = Qtempz * x_10_0_3 + WQtempz * x_10_0_4 + ABCDtemp *  x_4_0_4;
                                        QUICKDouble x_11_1_3 = Qtempx * x_11_0_3 + WQtempx * x_11_0_4 +  2 * ABCDtemp *  x_4_0_4;
                                        QUICKDouble x_11_2_3 = Qtempy * x_11_0_3 + WQtempy * x_11_0_4 + ABCDtemp *  x_7_0_4;
                                        QUICKDouble x_11_3_3 = Qtempz * x_11_0_3 + WQtempz * x_11_0_4;
                                        QUICKDouble x_12_1_3 = Qtempx * x_12_0_3 + WQtempx * x_12_0_4 + ABCDtemp *  x_8_0_4;
                                        QUICKDouble x_12_2_3 = Qtempy * x_12_0_3 + WQtempy * x_12_0_4 +  2 * ABCDtemp *  x_4_0_4;
                                        QUICKDouble x_12_3_3 = Qtempz * x_12_0_3 + WQtempz * x_12_0_4;
                                        QUICKDouble x_13_1_3 = Qtempx * x_13_0_3 + WQtempx * x_13_0_4 +  2 * ABCDtemp *  x_6_0_4;
                                        QUICKDouble x_13_2_3 = Qtempy * x_13_0_3 + WQtempy * x_13_0_4;
                                        QUICKDouble x_13_3_3 = Qtempz * x_13_0_3 + WQtempz * x_13_0_4 + ABCDtemp *  x_7_0_4;
                                        QUICKDouble x_14_1_3 = Qtempx * x_14_0_3 + WQtempx * x_14_0_4 + ABCDtemp *  x_9_0_4;
                                        QUICKDouble x_14_2_3 = Qtempy * x_14_0_3 + WQtempy * x_14_0_4;
                                        QUICKDouble x_14_3_3 = Qtempz * x_14_0_3 + WQtempz * x_14_0_4 +  2 * ABCDtemp *  x_6_0_4;
                                        QUICKDouble x_15_1_3 = Qtempx * x_15_0_3 + WQtempx * x_15_0_4;
                                        QUICKDouble x_15_2_3 = Qtempy * x_15_0_3 + WQtempy * x_15_0_4 +  2 * ABCDtemp *  x_5_0_4;
                                        QUICKDouble x_15_3_3 = Qtempz * x_15_0_3 + WQtempz * x_15_0_4 + ABCDtemp *  x_8_0_4;
                                        QUICKDouble x_16_1_3 = Qtempx * x_16_0_3 + WQtempx * x_16_0_4;
                                        QUICKDouble x_16_2_3 = Qtempy * x_16_0_3 + WQtempy * x_16_0_4 + ABCDtemp *  x_9_0_4;
                                        QUICKDouble x_16_3_3 = Qtempz * x_16_0_3 + WQtempz * x_16_0_4 +  2 * ABCDtemp *  x_5_0_4;
                                        QUICKDouble x_17_1_3 = Qtempx * x_17_0_3 + WQtempx * x_17_0_4 +  3 * ABCDtemp *  x_7_0_4;
                                        QUICKDouble x_17_2_3 = Qtempy * x_17_0_3 + WQtempy * x_17_0_4;
                                        QUICKDouble x_17_3_3 = Qtempz * x_17_0_3 + WQtempz * x_17_0_4;
                                        QUICKDouble x_18_1_3 = Qtempx * x_18_0_3 + WQtempx * x_18_0_4;
                                        QUICKDouble x_18_2_3 = Qtempy * x_18_0_3 + WQtempy * x_18_0_4 +  3 * ABCDtemp *  x_8_0_4;
                                        QUICKDouble x_18_3_3 = Qtempz * x_18_0_3 + WQtempz * x_18_0_4;
                                        QUICKDouble x_19_1_3 = Qtempx * x_19_0_3 + WQtempx * x_19_0_4;
                                        QUICKDouble x_19_2_3 = Qtempy * x_19_0_3 + WQtempy * x_19_0_4;
                                        QUICKDouble x_19_3_3 = Qtempz * x_19_0_3 + WQtempz * x_19_0_4 +  3 * ABCDtemp *  x_9_0_4;
                                        
                                        //GSPS(3, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz, ABCDtemp);
                                        
                                        QUICKDouble x_20_1_3 = Qtempx * x_20_0_3 + WQtempx * x_20_0_4 +  2 * ABCDtemp *  x_12_0_4;
                                        QUICKDouble x_20_2_3 = Qtempy * x_20_0_3 + WQtempy * x_20_0_4 +  2 * ABCDtemp *  x_11_0_4;
                                        QUICKDouble x_20_3_3 = Qtempz * x_20_0_3 + WQtempz * x_20_0_4;
                                        QUICKDouble x_21_1_3 = Qtempx * x_21_0_3 + WQtempx * x_21_0_4 +  2 * ABCDtemp *  x_14_0_4;
                                        QUICKDouble x_21_2_3 = Qtempy * x_21_0_3 + WQtempy * x_21_0_4;
                                        QUICKDouble x_21_3_3 = Qtempz * x_21_0_3 + WQtempz * x_21_0_4 +  2 * ABCDtemp *  x_13_0_4;
                                        QUICKDouble x_22_1_3 = Qtempx * x_22_0_3 + WQtempx * x_22_0_4;
                                        QUICKDouble x_22_2_3 = Qtempy * x_22_0_3 + WQtempy * x_22_0_4 +  2 * ABCDtemp *  x_16_0_4;
                                        QUICKDouble x_22_3_3 = Qtempz * x_22_0_3 + WQtempz * x_22_0_4 +  2 * ABCDtemp *  x_15_0_4;
                                        QUICKDouble x_23_1_3 = Qtempx * x_23_0_3 + WQtempx * x_23_0_4 +  2 * ABCDtemp *  x_10_0_4;
                                        QUICKDouble x_23_2_3 = Qtempy * x_23_0_3 + WQtempy * x_23_0_4 + ABCDtemp *  x_13_0_4;
                                        QUICKDouble x_23_3_3 = Qtempz * x_23_0_3 + WQtempz * x_23_0_4 + ABCDtemp *  x_11_0_4;
                                        QUICKDouble x_24_1_3 = Qtempx * x_24_0_3 + WQtempx * x_24_0_4 + ABCDtemp *  x_15_0_4;
                                        QUICKDouble x_24_2_3 = Qtempy * x_24_0_3 + WQtempy * x_24_0_4 +  2 * ABCDtemp *  x_10_0_4;
                                        QUICKDouble x_24_3_3 = Qtempz * x_24_0_3 + WQtempz * x_24_0_4 + ABCDtemp *  x_12_0_4;
                                        QUICKDouble x_25_1_3 = Qtempx * x_25_0_3 + WQtempx * x_25_0_4 + ABCDtemp *  x_16_0_4;
                                        QUICKDouble x_25_2_3 = Qtempy * x_25_0_3 + WQtempy * x_25_0_4 + ABCDtemp *  x_14_0_4;
                                        QUICKDouble x_25_3_3 = Qtempz * x_25_0_3 + WQtempz * x_25_0_4 +  2 * ABCDtemp *  x_10_0_4;
                                        QUICKDouble x_26_1_3 = Qtempx * x_26_0_3 + WQtempx * x_26_0_4 +  3 * ABCDtemp *  x_13_0_4;
                                        QUICKDouble x_26_2_3 = Qtempy * x_26_0_3 + WQtempy * x_26_0_4;
                                        QUICKDouble x_26_3_3 = Qtempz * x_26_0_3 + WQtempz * x_26_0_4 + ABCDtemp *  x_17_0_4;
                                        QUICKDouble x_27_1_3 = Qtempx * x_27_0_3 + WQtempx * x_27_0_4 + ABCDtemp *  x_19_0_4;
                                        QUICKDouble x_27_2_3 = Qtempy * x_27_0_3 + WQtempy * x_27_0_4;
                                        QUICKDouble x_27_3_3 = Qtempz * x_27_0_3 + WQtempz * x_27_0_4 +  3 * ABCDtemp *  x_14_0_4;
                                        QUICKDouble x_28_1_3 = Qtempx * x_28_0_3 + WQtempx * x_28_0_4 +  3 * ABCDtemp *  x_11_0_4;
                                        QUICKDouble x_28_2_3 = Qtempy * x_28_0_3 + WQtempy * x_28_0_4 + ABCDtemp *  x_17_0_4;
                                        QUICKDouble x_28_3_3 = Qtempz * x_28_0_3 + WQtempz * x_28_0_4;
                                        QUICKDouble x_29_1_3 = Qtempx * x_29_0_3 + WQtempx * x_29_0_4 + ABCDtemp *  x_18_0_4;
                                        QUICKDouble x_29_2_3 = Qtempy * x_29_0_3 + WQtempy * x_29_0_4 +  3 * ABCDtemp *  x_12_0_4;
                                        QUICKDouble x_29_3_3 = Qtempz * x_29_0_3 + WQtempz * x_29_0_4;
                                        QUICKDouble x_30_1_3 = Qtempx * x_30_0_3 + WQtempx * x_30_0_4;
                                        QUICKDouble x_30_2_3 = Qtempy * x_30_0_3 + WQtempy * x_30_0_4 +  3 * ABCDtemp *  x_15_0_4;
                                        QUICKDouble x_30_3_3 = Qtempz * x_30_0_3 + WQtempz * x_30_0_4 + ABCDtemp *  x_18_0_4;
                                        QUICKDouble x_31_1_3 = Qtempx * x_31_0_3 + WQtempx * x_31_0_4;
                                        QUICKDouble x_31_2_3 = Qtempy * x_31_0_3 + WQtempy * x_31_0_4 + ABCDtemp *  x_19_0_4;
                                        QUICKDouble x_31_3_3 = Qtempz * x_31_0_3 + WQtempz * x_31_0_4 +  3 * ABCDtemp *  x_16_0_4;    
                                        QUICKDouble x_32_1_3 = Qtempx * x_32_0_3 + WQtempx * x_32_0_4 +  4 * ABCDtemp *  x_17_0_4;
                                        QUICKDouble x_32_2_3 = Qtempy * x_32_0_3 + WQtempy * x_32_0_4;
                                        QUICKDouble x_32_3_3 = Qtempz * x_32_0_3 + WQtempz * x_32_0_4;
                                        QUICKDouble x_33_1_3 = Qtempx * x_33_0_3 + WQtempx * x_33_0_4;
                                        QUICKDouble x_33_2_3 = Qtempy * x_33_0_3 + WQtempy * x_33_0_4 +  4 * ABCDtemp *  x_18_0_4;
                                        QUICKDouble x_33_3_3 = Qtempz * x_33_0_3 + WQtempz * x_33_0_4;
                                        QUICKDouble x_34_1_3 = Qtempx * x_34_0_3 + WQtempx * x_34_0_4;
                                        QUICKDouble x_34_2_3 = Qtempy * x_34_0_3 + WQtempy * x_34_0_4;
                                        QUICKDouble x_34_3_3 = Qtempz * x_34_0_3 + WQtempz * x_34_0_4 +  4 * ABCDtemp *  x_19_0_4;
                                        
                                        //GSDS(2, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz, ABCDtemp, CDtemp, ABcom);
                                        QUICKDouble x_20_4_2 = Qtempx * x_20_2_2 + WQtempx * x_20_2_3 +  2 * ABCDtemp * x_12_2_3;
                                        QUICKDouble x_21_4_2 = Qtempx * x_21_2_2 + WQtempx * x_21_2_3 +  2 * ABCDtemp * x_14_2_3;
                                        QUICKDouble x_22_4_2 = Qtempx * x_22_2_2 + WQtempx * x_22_2_3;
                                        QUICKDouble x_23_4_2 = Qtempx * x_23_2_2 + WQtempx * x_23_2_3 +  2 * ABCDtemp * x_10_2_3;
                                        QUICKDouble x_24_4_2 = Qtempx * x_24_2_2 + WQtempx * x_24_2_3 + ABCDtemp * x_15_2_3;
                                        QUICKDouble x_25_4_2 = Qtempx * x_25_2_2 + WQtempx * x_25_2_3 + ABCDtemp * x_16_2_3;
                                        QUICKDouble x_26_4_2 = Qtempx * x_26_2_2 + WQtempx * x_26_2_3 +  3 * ABCDtemp * x_13_2_3;
                                        QUICKDouble x_27_4_2 = Qtempx * x_27_2_2 + WQtempx * x_27_2_3 + ABCDtemp * x_19_2_3;
                                        QUICKDouble x_28_4_2 = Qtempx * x_28_2_2 + WQtempx * x_28_2_3 +  3 * ABCDtemp * x_11_2_3;
                                        QUICKDouble x_29_4_2 = Qtempx * x_29_2_2 + WQtempx * x_29_2_3 + ABCDtemp * x_18_2_3;
                                        QUICKDouble x_30_4_2 = Qtempx * x_30_2_2 + WQtempx * x_30_2_3;
                                        QUICKDouble x_31_4_2 = Qtempx * x_31_2_2 + WQtempx * x_31_2_3;
                                        QUICKDouble x_32_4_2 = Qtempx * x_32_2_2 + WQtempx * x_32_2_3 +  4 * ABCDtemp * x_17_2_3;
                                        QUICKDouble x_33_4_2 = Qtempx * x_33_2_2 + WQtempx * x_33_2_3;
                                        QUICKDouble x_34_4_2 = Qtempx * x_34_2_2 + WQtempx * x_34_2_3;
                                        QUICKDouble x_20_5_2 = Qtempy * x_20_3_2 + WQtempy * x_20_3_3 +  2 * ABCDtemp * x_11_3_3;
                                        QUICKDouble x_21_5_2 = Qtempy * x_21_3_2 + WQtempy * x_21_3_3;
                                        QUICKDouble x_22_5_2 = Qtempy * x_22_3_2 + WQtempy * x_22_3_3 +  2 * ABCDtemp * x_16_3_3;
                                        QUICKDouble x_23_5_2 = Qtempy * x_23_3_2 + WQtempy * x_23_3_3 + ABCDtemp * x_13_3_3;
                                        QUICKDouble x_24_5_2 = Qtempy * x_24_3_2 + WQtempy * x_24_3_3 +  2 * ABCDtemp * x_10_3_3;
                                        QUICKDouble x_25_5_2 = Qtempy * x_25_3_2 + WQtempy * x_25_3_3 + ABCDtemp * x_14_3_3;
                                        QUICKDouble x_26_5_2 = Qtempy * x_26_3_2 + WQtempy * x_26_3_3;
                                        QUICKDouble x_27_5_2 = Qtempy * x_27_3_2 + WQtempy * x_27_3_3;
                                        QUICKDouble x_28_5_2 = Qtempy * x_28_3_2 + WQtempy * x_28_3_3 + ABCDtemp * x_17_3_3;
                                        QUICKDouble x_29_5_2 = Qtempy * x_29_3_2 + WQtempy * x_29_3_3 +  3 * ABCDtemp * x_12_3_3;
                                        QUICKDouble x_30_5_2 = Qtempy * x_30_3_2 + WQtempy * x_30_3_3 +  3 * ABCDtemp * x_15_3_3;
                                        QUICKDouble x_31_5_2 = Qtempy * x_31_3_2 + WQtempy * x_31_3_3 + ABCDtemp * x_19_3_3;
                                        QUICKDouble x_32_5_2 = Qtempy * x_32_3_2 + WQtempy * x_32_3_3;
                                        QUICKDouble x_33_5_2 = Qtempy * x_33_3_2 + WQtempy * x_33_3_3 +  4 * ABCDtemp * x_18_3_3;
                                        QUICKDouble x_34_5_2 = Qtempy * x_34_3_2 + WQtempy * x_34_3_3;
                                        QUICKDouble x_20_6_2 = Qtempx * x_20_3_2 + WQtempx * x_20_3_3 +  2 * ABCDtemp * x_12_3_3;
                                        QUICKDouble x_21_6_2 = Qtempx * x_21_3_2 + WQtempx * x_21_3_3 +  2 * ABCDtemp * x_14_3_3;
                                        QUICKDouble x_22_6_2 = Qtempx * x_22_3_2 + WQtempx * x_22_3_3;
                                        QUICKDouble x_23_6_2 = Qtempx * x_23_3_2 + WQtempx * x_23_3_3 +  2 * ABCDtemp * x_10_3_3;
                                        QUICKDouble x_24_6_2 = Qtempx * x_24_3_2 + WQtempx * x_24_3_3 + ABCDtemp * x_15_3_3;
                                        QUICKDouble x_25_6_2 = Qtempx * x_25_3_2 + WQtempx * x_25_3_3 + ABCDtemp * x_16_3_3;
                                        QUICKDouble x_26_6_2 = Qtempx * x_26_3_2 + WQtempx * x_26_3_3 +  3 * ABCDtemp * x_13_3_3;
                                        QUICKDouble x_27_6_2 = Qtempx * x_27_3_2 + WQtempx * x_27_3_3 + ABCDtemp * x_19_3_3;
                                        QUICKDouble x_28_6_2 = Qtempx * x_28_3_2 + WQtempx * x_28_3_3 +  3 * ABCDtemp * x_11_3_3;
                                        QUICKDouble x_29_6_2 = Qtempx * x_29_3_2 + WQtempx * x_29_3_3 + ABCDtemp * x_18_3_3;
                                        QUICKDouble x_30_6_2 = Qtempx * x_30_3_2 + WQtempx * x_30_3_3;
                                        QUICKDouble x_31_6_2 = Qtempx * x_31_3_2 + WQtempx * x_31_3_3;
                                        QUICKDouble x_32_6_2 = Qtempx * x_32_3_2 + WQtempx * x_32_3_3 +  4 * ABCDtemp * x_17_3_3;
                                        QUICKDouble x_33_6_2 = Qtempx * x_33_3_2 + WQtempx * x_33_3_3;
                                        QUICKDouble x_34_6_2 = Qtempx * x_34_3_2 + WQtempx * x_34_3_3;
                                        QUICKDouble x_20_7_2 = Qtempx * x_20_1_2 + WQtempx * x_20_1_3 + CDtemp * ( x_20_0_2 - ABcom * x_20_0_3) +  2 * ABCDtemp * x_12_1_3;
                                        QUICKDouble x_21_7_2 = Qtempx * x_21_1_2 + WQtempx * x_21_1_3 + CDtemp * ( x_21_0_2 - ABcom * x_21_0_3) +  2 * ABCDtemp * x_14_1_3;
                                        QUICKDouble x_22_7_2 = Qtempx * x_22_1_2 + WQtempx * x_22_1_3 + CDtemp * ( x_22_0_2 - ABcom * x_22_0_3);
                                        QUICKDouble x_23_7_2 = Qtempx * x_23_1_2 + WQtempx * x_23_1_3 + CDtemp * ( x_23_0_2 - ABcom * x_23_0_3) +  2 * ABCDtemp * x_10_1_3;
                                        QUICKDouble x_24_7_2 = Qtempx * x_24_1_2 + WQtempx * x_24_1_3 + CDtemp * ( x_24_0_2 - ABcom * x_24_0_3) + ABCDtemp * x_15_1_3;
                                        QUICKDouble x_25_7_2 = Qtempx * x_25_1_2 + WQtempx * x_25_1_3 + CDtemp * ( x_25_0_2 - ABcom * x_25_0_3) + ABCDtemp * x_16_1_3;
                                        QUICKDouble x_26_7_2 = Qtempx * x_26_1_2 + WQtempx * x_26_1_3 + CDtemp * ( x_26_0_2 - ABcom * x_26_0_3) +  3 * ABCDtemp * x_13_1_3;
                                        QUICKDouble x_27_7_2 = Qtempx * x_27_1_2 + WQtempx * x_27_1_3 + CDtemp * ( x_27_0_2 - ABcom * x_27_0_3) + ABCDtemp * x_19_1_3;
                                        QUICKDouble x_28_7_2 = Qtempx * x_28_1_2 + WQtempx * x_28_1_3 + CDtemp * ( x_28_0_2 - ABcom * x_28_0_3) +  3 * ABCDtemp * x_11_1_3;
                                        QUICKDouble x_29_7_2 = Qtempx * x_29_1_2 + WQtempx * x_29_1_3 + CDtemp * ( x_29_0_2 - ABcom * x_29_0_3) + ABCDtemp * x_18_1_3;
                                        QUICKDouble x_30_7_2 = Qtempx * x_30_1_2 + WQtempx * x_30_1_3 + CDtemp * ( x_30_0_2 - ABcom * x_30_0_3);
                                        QUICKDouble x_31_7_2 = Qtempx * x_31_1_2 + WQtempx * x_31_1_3 + CDtemp * ( x_31_0_2 - ABcom * x_31_0_3);
                                        QUICKDouble x_32_7_2 = Qtempx * x_32_1_2 + WQtempx * x_32_1_3 + CDtemp * ( x_32_0_2 - ABcom * x_32_0_3) +  4 * ABCDtemp * x_17_1_3;
                                        QUICKDouble x_33_7_2 = Qtempx * x_33_1_2 + WQtempx * x_33_1_3 + CDtemp * ( x_33_0_2 - ABcom * x_33_0_3);
                                        QUICKDouble x_34_7_2 = Qtempx * x_34_1_2 + WQtempx * x_34_1_3 + CDtemp * ( x_34_0_2 - ABcom * x_34_0_3);
                                        QUICKDouble x_20_8_2 = Qtempy * x_20_2_2 + WQtempy * x_20_2_3 + CDtemp * ( x_20_0_2 - ABcom * x_20_0_3) +  2 * ABCDtemp * x_11_2_3;
                                        QUICKDouble x_21_8_2 = Qtempy * x_21_2_2 + WQtempy * x_21_2_3 + CDtemp * ( x_21_0_2 - ABcom * x_21_0_3);
                                        QUICKDouble x_22_8_2 = Qtempy * x_22_2_2 + WQtempy * x_22_2_3 + CDtemp * ( x_22_0_2 - ABcom * x_22_0_3) +  2 * ABCDtemp * x_16_2_3;
                                        QUICKDouble x_23_8_2 = Qtempy * x_23_2_2 + WQtempy * x_23_2_3 + CDtemp * ( x_23_0_2 - ABcom * x_23_0_3) + ABCDtemp * x_13_2_3;
                                        QUICKDouble x_24_8_2 = Qtempy * x_24_2_2 + WQtempy * x_24_2_3 + CDtemp * ( x_24_0_2 - ABcom * x_24_0_3) +  2 * ABCDtemp * x_10_2_3;
                                        QUICKDouble x_25_8_2 = Qtempy * x_25_2_2 + WQtempy * x_25_2_3 + CDtemp * ( x_25_0_2 - ABcom * x_25_0_3) + ABCDtemp * x_14_2_3;
                                        QUICKDouble x_26_8_2 = Qtempy * x_26_2_2 + WQtempy * x_26_2_3 + CDtemp * ( x_26_0_2 - ABcom * x_26_0_3);
                                        QUICKDouble x_27_8_2 = Qtempy * x_27_2_2 + WQtempy * x_27_2_3 + CDtemp * ( x_27_0_2 - ABcom * x_27_0_3);
                                        QUICKDouble x_28_8_2 = Qtempy * x_28_2_2 + WQtempy * x_28_2_3 + CDtemp * ( x_28_0_2 - ABcom * x_28_0_3) + ABCDtemp * x_17_2_3;
                                        QUICKDouble x_29_8_2 = Qtempy * x_29_2_2 + WQtempy * x_29_2_3 + CDtemp * ( x_29_0_2 - ABcom * x_29_0_3) +  3 * ABCDtemp * x_12_2_3;
                                        QUICKDouble x_30_8_2 = Qtempy * x_30_2_2 + WQtempy * x_30_2_3 + CDtemp * ( x_30_0_2 - ABcom * x_30_0_3) +  3 * ABCDtemp * x_15_2_3;
                                        QUICKDouble x_31_8_2 = Qtempy * x_31_2_2 + WQtempy * x_31_2_3 + CDtemp * ( x_31_0_2 - ABcom * x_31_0_3) + ABCDtemp * x_19_2_3;
                                        QUICKDouble x_32_8_2 = Qtempy * x_32_2_2 + WQtempy * x_32_2_3 + CDtemp * ( x_32_0_2 - ABcom * x_32_0_3);
                                        QUICKDouble x_33_8_2 = Qtempy * x_33_2_2 + WQtempy * x_33_2_3 + CDtemp * ( x_33_0_2 - ABcom * x_33_0_3) +  4 * ABCDtemp * x_18_2_3;
                                        QUICKDouble x_34_8_2 = Qtempy * x_34_2_2 + WQtempy * x_34_2_3 + CDtemp * ( x_34_0_2 - ABcom * x_34_0_3);
                                        QUICKDouble x_20_9_2 = Qtempz * x_20_3_2 + WQtempz * x_20_3_3 + CDtemp * ( x_20_0_2 - ABcom * x_20_0_3);
                                        QUICKDouble x_21_9_2 = Qtempz * x_21_3_2 + WQtempz * x_21_3_3 + CDtemp * ( x_21_0_2 - ABcom * x_21_0_3) +  2 * ABCDtemp * x_13_3_3;
                                        QUICKDouble x_22_9_2 = Qtempz * x_22_3_2 + WQtempz * x_22_3_3 + CDtemp * ( x_22_0_2 - ABcom * x_22_0_3) +  2 * ABCDtemp * x_15_3_3;
                                        QUICKDouble x_23_9_2 = Qtempz * x_23_3_2 + WQtempz * x_23_3_3 + CDtemp * ( x_23_0_2 - ABcom * x_23_0_3) + ABCDtemp * x_11_3_3;
                                        QUICKDouble x_24_9_2 = Qtempz * x_24_3_2 + WQtempz * x_24_3_3 + CDtemp * ( x_24_0_2 - ABcom * x_24_0_3) + ABCDtemp * x_12_3_3;
                                        QUICKDouble x_25_9_2 = Qtempz * x_25_3_2 + WQtempz * x_25_3_3 + CDtemp * ( x_25_0_2 - ABcom * x_25_0_3) +  2 * ABCDtemp * x_10_3_3;
                                        QUICKDouble x_26_9_2 = Qtempz * x_26_3_2 + WQtempz * x_26_3_3 + CDtemp * ( x_26_0_2 - ABcom * x_26_0_3) + ABCDtemp * x_17_3_3;
                                        QUICKDouble x_27_9_2 = Qtempz * x_27_3_2 + WQtempz * x_27_3_3 + CDtemp * ( x_27_0_2 - ABcom * x_27_0_3) +  3 * ABCDtemp * x_14_3_3;
                                        QUICKDouble x_28_9_2 = Qtempz * x_28_3_2 + WQtempz * x_28_3_3 + CDtemp * ( x_28_0_2 - ABcom * x_28_0_3);
                                        QUICKDouble x_29_9_2 = Qtempz * x_29_3_2 + WQtempz * x_29_3_3 + CDtemp * ( x_29_0_2 - ABcom * x_29_0_3);
                                        QUICKDouble x_30_9_2 = Qtempz * x_30_3_2 + WQtempz * x_30_3_3 + CDtemp * ( x_30_0_2 - ABcom * x_30_0_3) + ABCDtemp * x_18_3_3;
                                        QUICKDouble x_31_9_2 = Qtempz * x_31_3_2 + WQtempz * x_31_3_3 + CDtemp * ( x_31_0_2 - ABcom * x_31_0_3) +  3 * ABCDtemp * x_16_3_3;
                                        QUICKDouble x_32_9_2 = Qtempz * x_32_3_2 + WQtempz * x_32_3_3 + CDtemp * ( x_32_0_2 - ABcom * x_32_0_3);
                                        QUICKDouble x_33_9_2 = Qtempz * x_33_3_2 + WQtempz * x_33_3_3 + CDtemp * ( x_33_0_2 - ABcom * x_33_0_3);
                                        QUICKDouble x_34_9_2 = Qtempz * x_34_3_2 + WQtempz * x_34_3_3 + CDtemp * ( x_34_0_2 - ABcom * x_34_0_3) +  4 * ABCDtemp * x_19_3_3;
                                        
                                        //DSPS(2, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz, ABCDtemp);
                                        QUICKDouble x_4_1_2 = Qtempx * x_4_0_2 + WQtempx * x_4_0_3 + ABCDtemp * x_2_0_3;
                                        QUICKDouble x_4_2_2 = Qtempy * x_4_0_2 + WQtempy * x_4_0_3 + ABCDtemp * x_1_0_3;
                                        QUICKDouble x_4_3_2 = Qtempz * x_4_0_2 + WQtempz * x_4_0_3;
                                        
                                        QUICKDouble x_5_1_2 = Qtempx * x_5_0_2 + WQtempx * x_5_0_3;
                                        QUICKDouble x_5_2_2 = Qtempy * x_5_0_2 + WQtempy * x_5_0_3 + ABCDtemp * x_3_0_3;
                                        QUICKDouble x_5_3_2 = Qtempz * x_5_0_2 + WQtempz * x_5_0_3 + ABCDtemp * x_2_0_3;
                                        
                                        QUICKDouble x_6_1_2 = Qtempx * x_6_0_2 + WQtempx * x_6_0_3 + ABCDtemp * x_3_0_3;
                                        QUICKDouble x_6_2_2 = Qtempy * x_6_0_2 + WQtempy * x_6_0_3;
                                        QUICKDouble x_6_3_2 = Qtempz * x_6_0_2 + WQtempz * x_6_0_3 + ABCDtemp * x_1_0_3;
                                        
                                        QUICKDouble x_7_1_2 = Qtempx * x_7_0_2 + WQtempx * x_7_0_3 + ABCDtemp * x_1_0_3 * 2;
                                        QUICKDouble x_7_2_2 = Qtempy * x_7_0_2 + WQtempy * x_7_0_3;
                                        QUICKDouble x_7_3_2 = Qtempz * x_7_0_2 + WQtempz * x_7_0_3;
                                        
                                        QUICKDouble x_8_1_2 = Qtempx * x_8_0_2 + WQtempx * x_8_0_3;
                                        QUICKDouble x_8_2_2 = Qtempy * x_8_0_2 + WQtempy * x_8_0_3 + ABCDtemp * x_2_0_3 * 2;
                                        QUICKDouble x_8_3_2 = Qtempz * x_8_0_2 + WQtempz * x_8_0_3;
                                        
                                        QUICKDouble x_9_1_2 = Qtempx * x_9_0_2 + WQtempx * x_9_0_3;
                                        QUICKDouble x_9_2_2 = Qtempy * x_9_0_2 + WQtempy * x_9_0_3;
                                        QUICKDouble x_9_3_2 = Qtempz * x_9_0_2 + WQtempz * x_9_0_3 + ABCDtemp * x_3_0_3 * 2;            
                                        
                                        //FSDS(1, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz, ABCDtemp, CDtemp, ABcom);
                                        
                                        QUICKDouble x_10_4_1 = Qtempx * x_10_2_1 + WQtempx * x_10_2_2 + ABCDtemp * x_5_2_2;
                                        QUICKDouble x_11_4_1 = Qtempx * x_11_2_1 + WQtempx * x_11_2_2 +  2 * ABCDtemp * x_4_2_2;
                                        QUICKDouble x_12_4_1 = Qtempx * x_12_2_1 + WQtempx * x_12_2_2 + ABCDtemp * x_8_2_2;
                                        QUICKDouble x_13_4_1 = Qtempx * x_13_2_1 + WQtempx * x_13_2_2 +  2 * ABCDtemp * x_6_2_2;
                                        QUICKDouble x_14_4_1 = Qtempx * x_14_2_1 + WQtempx * x_14_2_2 + ABCDtemp * x_9_2_2;
                                        QUICKDouble x_15_4_1 = Qtempx * x_15_2_1 + WQtempx * x_15_2_2;
                                        QUICKDouble x_16_4_1 = Qtempx * x_16_2_1 + WQtempx * x_16_2_2;
                                        QUICKDouble x_17_4_1 = Qtempx * x_17_2_1 + WQtempx * x_17_2_2 +  3 * ABCDtemp * x_7_2_2;
                                        QUICKDouble x_18_4_1 = Qtempx * x_18_2_1 + WQtempx * x_18_2_2;
                                        QUICKDouble x_19_4_1 = Qtempx * x_19_2_1 + WQtempx * x_19_2_2;
                                        QUICKDouble x_10_5_1 = Qtempy * x_10_3_1 + WQtempy * x_10_3_2 + ABCDtemp * x_6_3_2;
                                        QUICKDouble x_11_5_1 = Qtempy * x_11_3_1 + WQtempy * x_11_3_2 + ABCDtemp * x_7_3_2;
                                        QUICKDouble x_12_5_1 = Qtempy * x_12_3_1 + WQtempy * x_12_3_2 +  2 * ABCDtemp * x_4_3_2;
                                        QUICKDouble x_13_5_1 = Qtempy * x_13_3_1 + WQtempy * x_13_3_2;
                                        QUICKDouble x_14_5_1 = Qtempy * x_14_3_1 + WQtempy * x_14_3_2;
                                        QUICKDouble x_15_5_1 = Qtempy * x_15_3_1 + WQtempy * x_15_3_2 +  2 * ABCDtemp * x_5_3_2;
                                        QUICKDouble x_16_5_1 = Qtempy * x_16_3_1 + WQtempy * x_16_3_2 + ABCDtemp * x_9_3_2;
                                        QUICKDouble x_17_5_1 = Qtempy * x_17_3_1 + WQtempy * x_17_3_2;
                                        QUICKDouble x_18_5_1 = Qtempy * x_18_3_1 + WQtempy * x_18_3_2 +  3 * ABCDtemp * x_8_3_2;
                                        QUICKDouble x_19_5_1 = Qtempy * x_19_3_1 + WQtempy * x_19_3_2;
                                        QUICKDouble x_10_6_1 = Qtempx * x_10_3_1 + WQtempx * x_10_3_2 + ABCDtemp * x_5_3_2;
                                        QUICKDouble x_11_6_1 = Qtempx * x_11_3_1 + WQtempx * x_11_3_2 +  2 * ABCDtemp * x_4_3_2;
                                        QUICKDouble x_12_6_1 = Qtempx * x_12_3_1 + WQtempx * x_12_3_2 + ABCDtemp * x_8_3_2;
                                        QUICKDouble x_13_6_1 = Qtempx * x_13_3_1 + WQtempx * x_13_3_2 +  2 * ABCDtemp * x_6_3_2;
                                        QUICKDouble x_14_6_1 = Qtempx * x_14_3_1 + WQtempx * x_14_3_2 + ABCDtemp * x_9_3_2;
                                        QUICKDouble x_15_6_1 = Qtempx * x_15_3_1 + WQtempx * x_15_3_2;
                                        QUICKDouble x_16_6_1 = Qtempx * x_16_3_1 + WQtempx * x_16_3_2;
                                        QUICKDouble x_17_6_1 = Qtempx * x_17_3_1 + WQtempx * x_17_3_2 +  3 * ABCDtemp * x_7_3_2;
                                        QUICKDouble x_18_6_1 = Qtempx * x_18_3_1 + WQtempx * x_18_3_2;
                                        QUICKDouble x_19_6_1 = Qtempx * x_19_3_1 + WQtempx * x_19_3_2;
                                        QUICKDouble x_10_7_1 = Qtempx * x_10_1_1 + WQtempx * x_10_1_2 + CDtemp * ( x_10_0_1 - ABcom * x_10_0_2) + ABCDtemp * x_5_1_2;
                                        QUICKDouble x_11_7_1 = Qtempx * x_11_1_1 + WQtempx * x_11_1_2 + CDtemp * ( x_11_0_1 - ABcom * x_11_0_2) +  2 * ABCDtemp * x_4_1_2;
                                        QUICKDouble x_12_7_1 = Qtempx * x_12_1_1 + WQtempx * x_12_1_2 + CDtemp * ( x_12_0_1 - ABcom * x_12_0_2) + ABCDtemp * x_8_1_2;
                                        QUICKDouble x_13_7_1 = Qtempx * x_13_1_1 + WQtempx * x_13_1_2 + CDtemp * ( x_13_0_1 - ABcom * x_13_0_2) +  2 * ABCDtemp * x_6_1_2;
                                        QUICKDouble x_14_7_1 = Qtempx * x_14_1_1 + WQtempx * x_14_1_2 + CDtemp * ( x_14_0_1 - ABcom * x_14_0_2) + ABCDtemp * x_9_1_2;
                                        QUICKDouble x_15_7_1 = Qtempx * x_15_1_1 + WQtempx * x_15_1_2 + CDtemp * ( x_15_0_1 - ABcom * x_15_0_2);
                                        QUICKDouble x_16_7_1 = Qtempx * x_16_1_1 + WQtempx * x_16_1_2 + CDtemp * ( x_16_0_1 - ABcom * x_16_0_2);
                                        QUICKDouble x_17_7_1 = Qtempx * x_17_1_1 + WQtempx * x_17_1_2 + CDtemp * ( x_17_0_1 - ABcom * x_17_0_2) +  3 * ABCDtemp * x_7_1_2;
                                        QUICKDouble x_18_7_1 = Qtempx * x_18_1_1 + WQtempx * x_18_1_2 + CDtemp * ( x_18_0_1 - ABcom * x_18_0_2);
                                        QUICKDouble x_19_7_1 = Qtempx * x_19_1_1 + WQtempx * x_19_1_2 + CDtemp * ( x_19_0_1 - ABcom * x_19_0_2);
                                        QUICKDouble x_10_8_1 = Qtempy * x_10_2_1 + WQtempy * x_10_2_2 + CDtemp * ( x_10_0_1 - ABcom * x_10_0_2) + ABCDtemp * x_6_2_2;
                                        QUICKDouble x_11_8_1 = Qtempy * x_11_2_1 + WQtempy * x_11_2_2 + CDtemp * ( x_11_0_1 - ABcom * x_11_0_2) + ABCDtemp * x_7_2_2;
                                        QUICKDouble x_12_8_1 = Qtempy * x_12_2_1 + WQtempy * x_12_2_2 + CDtemp * ( x_12_0_1 - ABcom * x_12_0_2) +  2 * ABCDtemp * x_4_2_2;
                                        QUICKDouble x_13_8_1 = Qtempy * x_13_2_1 + WQtempy * x_13_2_2 + CDtemp * ( x_13_0_1 - ABcom * x_13_0_2);
                                        QUICKDouble x_14_8_1 = Qtempy * x_14_2_1 + WQtempy * x_14_2_2 + CDtemp * ( x_14_0_1 - ABcom * x_14_0_2);
                                        QUICKDouble x_15_8_1 = Qtempy * x_15_2_1 + WQtempy * x_15_2_2 + CDtemp * ( x_15_0_1 - ABcom * x_15_0_2) +  2 * ABCDtemp * x_5_2_2;
                                        QUICKDouble x_16_8_1 = Qtempy * x_16_2_1 + WQtempy * x_16_2_2 + CDtemp * ( x_16_0_1 - ABcom * x_16_0_2) + ABCDtemp * x_9_2_2;
                                        QUICKDouble x_17_8_1 = Qtempy * x_17_2_1 + WQtempy * x_17_2_2 + CDtemp * ( x_17_0_1 - ABcom * x_17_0_2);
                                        QUICKDouble x_18_8_1 = Qtempy * x_18_2_1 + WQtempy * x_18_2_2 + CDtemp * ( x_18_0_1 - ABcom * x_18_0_2) +  3 * ABCDtemp * x_8_2_2;
                                        QUICKDouble x_19_8_1 = Qtempy * x_19_2_1 + WQtempy * x_19_2_2 + CDtemp * ( x_19_0_1 - ABcom * x_19_0_2);
                                        QUICKDouble x_10_9_1 = Qtempz * x_10_3_1 + WQtempz * x_10_3_2 + CDtemp * ( x_10_0_1 - ABcom * x_10_0_2) + ABCDtemp * x_4_3_2;
                                        QUICKDouble x_11_9_1 = Qtempz * x_11_3_1 + WQtempz * x_11_3_2 + CDtemp * ( x_11_0_1 - ABcom * x_11_0_2);
                                        QUICKDouble x_12_9_1 = Qtempz * x_12_3_1 + WQtempz * x_12_3_2 + CDtemp * ( x_12_0_1 - ABcom * x_12_0_2);
                                        QUICKDouble x_13_9_1 = Qtempz * x_13_3_1 + WQtempz * x_13_3_2 + CDtemp * ( x_13_0_1 - ABcom * x_13_0_2) + ABCDtemp * x_7_3_2;
                                        QUICKDouble x_14_9_1 = Qtempz * x_14_3_1 + WQtempz * x_14_3_2 + CDtemp * ( x_14_0_1 - ABcom * x_14_0_2) +  2 * ABCDtemp * x_6_3_2;
                                        QUICKDouble x_15_9_1 = Qtempz * x_15_3_1 + WQtempz * x_15_3_2 + CDtemp * ( x_15_0_1 - ABcom * x_15_0_2) + ABCDtemp * x_8_3_2;
                                        QUICKDouble x_16_9_1 = Qtempz * x_16_3_1 + WQtempz * x_16_3_2 + CDtemp * ( x_16_0_1 - ABcom * x_16_0_2) +  2 * ABCDtemp * x_5_3_2;
                                        QUICKDouble x_17_9_1 = Qtempz * x_17_3_1 + WQtempz * x_17_3_2 + CDtemp * ( x_17_0_1 - ABcom * x_17_0_2);
                                        QUICKDouble x_18_9_1 = Qtempz * x_18_3_1 + WQtempz * x_18_3_2 + CDtemp * ( x_18_0_1 - ABcom * x_18_0_2);
                                        QUICKDouble x_19_9_1 = Qtempz * x_19_3_1 + WQtempz * x_19_3_2 + CDtemp * ( x_19_0_1 - ABcom * x_19_0_2) +  3 * ABCDtemp * x_9_3_2;
                                        
                                        //GSDS(1, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz, ABCDtemp, CDtemp, ABcom);
                                        
                                        QUICKDouble x_20_4_1 = Qtempx * x_20_2_1 + WQtempx * x_20_2_2 +  2 * ABCDtemp * x_12_2_2;
                                        QUICKDouble x_21_4_1 = Qtempx * x_21_2_1 + WQtempx * x_21_2_2 +  2 * ABCDtemp * x_14_2_2;
                                        QUICKDouble x_22_4_1 = Qtempx * x_22_2_1 + WQtempx * x_22_2_2;
                                        QUICKDouble x_23_4_1 = Qtempx * x_23_2_1 + WQtempx * x_23_2_2 +  2 * ABCDtemp * x_10_2_2;
                                        QUICKDouble x_24_4_1 = Qtempx * x_24_2_1 + WQtempx * x_24_2_2 + ABCDtemp * x_15_2_2;
                                        QUICKDouble x_25_4_1 = Qtempx * x_25_2_1 + WQtempx * x_25_2_2 + ABCDtemp * x_16_2_2;
                                        QUICKDouble x_26_4_1 = Qtempx * x_26_2_1 + WQtempx * x_26_2_2 +  3 * ABCDtemp * x_13_2_2;
                                        QUICKDouble x_27_4_1 = Qtempx * x_27_2_1 + WQtempx * x_27_2_2 + ABCDtemp * x_19_2_2;
                                        QUICKDouble x_28_4_1 = Qtempx * x_28_2_1 + WQtempx * x_28_2_2 +  3 * ABCDtemp * x_11_2_2;
                                        QUICKDouble x_29_4_1 = Qtempx * x_29_2_1 + WQtempx * x_29_2_2 + ABCDtemp * x_18_2_2;
                                        QUICKDouble x_30_4_1 = Qtempx * x_30_2_1 + WQtempx * x_30_2_2;
                                        QUICKDouble x_31_4_1 = Qtempx * x_31_2_1 + WQtempx * x_31_2_2;
                                        QUICKDouble x_32_4_1 = Qtempx * x_32_2_1 + WQtempx * x_32_2_2 +  4 * ABCDtemp * x_17_2_2;
                                        QUICKDouble x_33_4_1 = Qtempx * x_33_2_1 + WQtempx * x_33_2_2;
                                        QUICKDouble x_34_4_1 = Qtempx * x_34_2_1 + WQtempx * x_34_2_2;
                                        QUICKDouble x_20_5_1 = Qtempy * x_20_3_1 + WQtempy * x_20_3_2 +  2 * ABCDtemp * x_11_3_2;
                                        QUICKDouble x_21_5_1 = Qtempy * x_21_3_1 + WQtempy * x_21_3_2;
                                        QUICKDouble x_22_5_1 = Qtempy * x_22_3_1 + WQtempy * x_22_3_2 +  2 * ABCDtemp * x_16_3_2;
                                        QUICKDouble x_23_5_1 = Qtempy * x_23_3_1 + WQtempy * x_23_3_2 + ABCDtemp * x_13_3_2;
                                        QUICKDouble x_24_5_1 = Qtempy * x_24_3_1 + WQtempy * x_24_3_2 +  2 * ABCDtemp * x_10_3_2;
                                        QUICKDouble x_25_5_1 = Qtempy * x_25_3_1 + WQtempy * x_25_3_2 + ABCDtemp * x_14_3_2;
                                        QUICKDouble x_26_5_1 = Qtempy * x_26_3_1 + WQtempy * x_26_3_2;
                                        QUICKDouble x_27_5_1 = Qtempy * x_27_3_1 + WQtempy * x_27_3_2;
                                        QUICKDouble x_28_5_1 = Qtempy * x_28_3_1 + WQtempy * x_28_3_2 + ABCDtemp * x_17_3_2;
                                        QUICKDouble x_29_5_1 = Qtempy * x_29_3_1 + WQtempy * x_29_3_2 +  3 * ABCDtemp * x_12_3_2;
                                        QUICKDouble x_30_5_1 = Qtempy * x_30_3_1 + WQtempy * x_30_3_2 +  3 * ABCDtemp * x_15_3_2;
                                        QUICKDouble x_31_5_1 = Qtempy * x_31_3_1 + WQtempy * x_31_3_2 + ABCDtemp * x_19_3_2;
                                        QUICKDouble x_32_5_1 = Qtempy * x_32_3_1 + WQtempy * x_32_3_2;
                                        QUICKDouble x_33_5_1 = Qtempy * x_33_3_1 + WQtempy * x_33_3_2 +  4 * ABCDtemp * x_18_3_2;
                                        QUICKDouble x_34_5_1 = Qtempy * x_34_3_1 + WQtempy * x_34_3_2;
                                        QUICKDouble x_20_6_1 = Qtempx * x_20_3_1 + WQtempx * x_20_3_2 +  2 * ABCDtemp * x_12_3_2;
                                        QUICKDouble x_21_6_1 = Qtempx * x_21_3_1 + WQtempx * x_21_3_2 +  2 * ABCDtemp * x_14_3_2;
                                        QUICKDouble x_22_6_1 = Qtempx * x_22_3_1 + WQtempx * x_22_3_2;
                                        QUICKDouble x_23_6_1 = Qtempx * x_23_3_1 + WQtempx * x_23_3_2 +  2 * ABCDtemp * x_10_3_2;
                                        QUICKDouble x_24_6_1 = Qtempx * x_24_3_1 + WQtempx * x_24_3_2 + ABCDtemp * x_15_3_2;
                                        QUICKDouble x_25_6_1 = Qtempx * x_25_3_1 + WQtempx * x_25_3_2 + ABCDtemp * x_16_3_2;
                                        QUICKDouble x_26_6_1 = Qtempx * x_26_3_1 + WQtempx * x_26_3_2 +  3 * ABCDtemp * x_13_3_2;
                                        QUICKDouble x_27_6_1 = Qtempx * x_27_3_1 + WQtempx * x_27_3_2 + ABCDtemp * x_19_3_2;
                                        QUICKDouble x_28_6_1 = Qtempx * x_28_3_1 + WQtempx * x_28_3_2 +  3 * ABCDtemp * x_11_3_2;
                                        QUICKDouble x_29_6_1 = Qtempx * x_29_3_1 + WQtempx * x_29_3_2 + ABCDtemp * x_18_3_2;
                                        QUICKDouble x_30_6_1 = Qtempx * x_30_3_1 + WQtempx * x_30_3_2;
                                        QUICKDouble x_31_6_1 = Qtempx * x_31_3_1 + WQtempx * x_31_3_2;
                                        QUICKDouble x_32_6_1 = Qtempx * x_32_3_1 + WQtempx * x_32_3_2 +  4 * ABCDtemp * x_17_3_2;
                                        QUICKDouble x_33_6_1 = Qtempx * x_33_3_1 + WQtempx * x_33_3_2;
                                        QUICKDouble x_34_6_1 = Qtempx * x_34_3_1 + WQtempx * x_34_3_2;
                                        QUICKDouble x_20_7_1 = Qtempx * x_20_1_1 + WQtempx * x_20_1_2 + CDtemp * ( x_20_0_1 - ABcom * x_20_0_2) +  2 * ABCDtemp * x_12_1_2;
                                        QUICKDouble x_21_7_1 = Qtempx * x_21_1_1 + WQtempx * x_21_1_2 + CDtemp * ( x_21_0_1 - ABcom * x_21_0_2) +  2 * ABCDtemp * x_14_1_2;
                                        QUICKDouble x_22_7_1 = Qtempx * x_22_1_1 + WQtempx * x_22_1_2 + CDtemp * ( x_22_0_1 - ABcom * x_22_0_2);
                                        QUICKDouble x_23_7_1 = Qtempx * x_23_1_1 + WQtempx * x_23_1_2 + CDtemp * ( x_23_0_1 - ABcom * x_23_0_2) +  2 * ABCDtemp * x_10_1_2;
                                        QUICKDouble x_24_7_1 = Qtempx * x_24_1_1 + WQtempx * x_24_1_2 + CDtemp * ( x_24_0_1 - ABcom * x_24_0_2) + ABCDtemp * x_15_1_2;
                                        QUICKDouble x_25_7_1 = Qtempx * x_25_1_1 + WQtempx * x_25_1_2 + CDtemp * ( x_25_0_1 - ABcom * x_25_0_2) + ABCDtemp * x_16_1_2;
                                        QUICKDouble x_26_7_1 = Qtempx * x_26_1_1 + WQtempx * x_26_1_2 + CDtemp * ( x_26_0_1 - ABcom * x_26_0_2) +  3 * ABCDtemp * x_13_1_2;
                                        QUICKDouble x_27_7_1 = Qtempx * x_27_1_1 + WQtempx * x_27_1_2 + CDtemp * ( x_27_0_1 - ABcom * x_27_0_2) + ABCDtemp * x_19_1_2;
                                        QUICKDouble x_28_7_1 = Qtempx * x_28_1_1 + WQtempx * x_28_1_2 + CDtemp * ( x_28_0_1 - ABcom * x_28_0_2) +  3 * ABCDtemp * x_11_1_2;
                                        QUICKDouble x_29_7_1 = Qtempx * x_29_1_1 + WQtempx * x_29_1_2 + CDtemp * ( x_29_0_1 - ABcom * x_29_0_2) + ABCDtemp * x_18_1_2;
                                        QUICKDouble x_30_7_1 = Qtempx * x_30_1_1 + WQtempx * x_30_1_2 + CDtemp * ( x_30_0_1 - ABcom * x_30_0_2);
                                        QUICKDouble x_31_7_1 = Qtempx * x_31_1_1 + WQtempx * x_31_1_2 + CDtemp * ( x_31_0_1 - ABcom * x_31_0_2);
                                        QUICKDouble x_32_7_1 = Qtempx * x_32_1_1 + WQtempx * x_32_1_2 + CDtemp * ( x_32_0_1 - ABcom * x_32_0_2) +  4 * ABCDtemp * x_17_1_2;
                                        QUICKDouble x_33_7_1 = Qtempx * x_33_1_1 + WQtempx * x_33_1_2 + CDtemp * ( x_33_0_1 - ABcom * x_33_0_2);
                                        QUICKDouble x_34_7_1 = Qtempx * x_34_1_1 + WQtempx * x_34_1_2 + CDtemp * ( x_34_0_1 - ABcom * x_34_0_2);
                                        QUICKDouble x_20_8_1 = Qtempy * x_20_2_1 + WQtempy * x_20_2_2 + CDtemp * ( x_20_0_1 - ABcom * x_20_0_2) +  2 * ABCDtemp * x_11_2_2;
                                        QUICKDouble x_21_8_1 = Qtempy * x_21_2_1 + WQtempy * x_21_2_2 + CDtemp * ( x_21_0_1 - ABcom * x_21_0_2);
                                        QUICKDouble x_22_8_1 = Qtempy * x_22_2_1 + WQtempy * x_22_2_2 + CDtemp * ( x_22_0_1 - ABcom * x_22_0_2) +  2 * ABCDtemp * x_16_2_2;
                                        QUICKDouble x_23_8_1 = Qtempy * x_23_2_1 + WQtempy * x_23_2_2 + CDtemp * ( x_23_0_1 - ABcom * x_23_0_2) + ABCDtemp * x_13_2_2;
                                        QUICKDouble x_24_8_1 = Qtempy * x_24_2_1 + WQtempy * x_24_2_2 + CDtemp * ( x_24_0_1 - ABcom * x_24_0_2) +  2 * ABCDtemp * x_10_2_2;
                                        QUICKDouble x_25_8_1 = Qtempy * x_25_2_1 + WQtempy * x_25_2_2 + CDtemp * ( x_25_0_1 - ABcom * x_25_0_2) + ABCDtemp * x_14_2_2;
                                        QUICKDouble x_26_8_1 = Qtempy * x_26_2_1 + WQtempy * x_26_2_2 + CDtemp * ( x_26_0_1 - ABcom * x_26_0_2);
                                        QUICKDouble x_27_8_1 = Qtempy * x_27_2_1 + WQtempy * x_27_2_2 + CDtemp * ( x_27_0_1 - ABcom * x_27_0_2);
                                        QUICKDouble x_28_8_1 = Qtempy * x_28_2_1 + WQtempy * x_28_2_2 + CDtemp * ( x_28_0_1 - ABcom * x_28_0_2) + ABCDtemp * x_17_2_2;
                                        QUICKDouble x_29_8_1 = Qtempy * x_29_2_1 + WQtempy * x_29_2_2 + CDtemp * ( x_29_0_1 - ABcom * x_29_0_2) +  3 * ABCDtemp * x_12_2_2;
                                        QUICKDouble x_30_8_1 = Qtempy * x_30_2_1 + WQtempy * x_30_2_2 + CDtemp * ( x_30_0_1 - ABcom * x_30_0_2) +  3 * ABCDtemp * x_15_2_2;
                                        QUICKDouble x_31_8_1 = Qtempy * x_31_2_1 + WQtempy * x_31_2_2 + CDtemp * ( x_31_0_1 - ABcom * x_31_0_2) + ABCDtemp * x_19_2_2;
                                        QUICKDouble x_32_8_1 = Qtempy * x_32_2_1 + WQtempy * x_32_2_2 + CDtemp * ( x_32_0_1 - ABcom * x_32_0_2);
                                        QUICKDouble x_33_8_1 = Qtempy * x_33_2_1 + WQtempy * x_33_2_2 + CDtemp * ( x_33_0_1 - ABcom * x_33_0_2) +  4 * ABCDtemp * x_18_2_2;
                                        QUICKDouble x_34_8_1 = Qtempy * x_34_2_1 + WQtempy * x_34_2_2 + CDtemp * ( x_34_0_1 - ABcom * x_34_0_2);
                                        QUICKDouble x_20_9_1 = Qtempz * x_20_3_1 + WQtempz * x_20_3_2 + CDtemp * ( x_20_0_1 - ABcom * x_20_0_2);
                                        QUICKDouble x_21_9_1 = Qtempz * x_21_3_1 + WQtempz * x_21_3_2 + CDtemp * ( x_21_0_1 - ABcom * x_21_0_2) +  2 * ABCDtemp * x_13_3_2;
                                        QUICKDouble x_22_9_1 = Qtempz * x_22_3_1 + WQtempz * x_22_3_2 + CDtemp * ( x_22_0_1 - ABcom * x_22_0_2) +  2 * ABCDtemp * x_15_3_2;
                                        QUICKDouble x_23_9_1 = Qtempz * x_23_3_1 + WQtempz * x_23_3_2 + CDtemp * ( x_23_0_1 - ABcom * x_23_0_2) + ABCDtemp * x_11_3_2;
                                        QUICKDouble x_24_9_1 = Qtempz * x_24_3_1 + WQtempz * x_24_3_2 + CDtemp * ( x_24_0_1 - ABcom * x_24_0_2) + ABCDtemp * x_12_3_2;
                                        QUICKDouble x_25_9_1 = Qtempz * x_25_3_1 + WQtempz * x_25_3_2 + CDtemp * ( x_25_0_1 - ABcom * x_25_0_2) +  2 * ABCDtemp * x_10_3_2;
                                        QUICKDouble x_26_9_1 = Qtempz * x_26_3_1 + WQtempz * x_26_3_2 + CDtemp * ( x_26_0_1 - ABcom * x_26_0_2) + ABCDtemp * x_17_3_2;
                                        QUICKDouble x_27_9_1 = Qtempz * x_27_3_1 + WQtempz * x_27_3_2 + CDtemp * ( x_27_0_1 - ABcom * x_27_0_2) +  3 * ABCDtemp * x_14_3_2;
                                        QUICKDouble x_28_9_1 = Qtempz * x_28_3_1 + WQtempz * x_28_3_2 + CDtemp * ( x_28_0_1 - ABcom * x_28_0_2);
                                        QUICKDouble x_29_9_1 = Qtempz * x_29_3_1 + WQtempz * x_29_3_2 + CDtemp * ( x_29_0_1 - ABcom * x_29_0_2);
                                        QUICKDouble x_30_9_1 = Qtempz * x_30_3_1 + WQtempz * x_30_3_2 + CDtemp * ( x_30_0_1 - ABcom * x_30_0_2) + ABCDtemp * x_18_3_2;
                                        QUICKDouble x_31_9_1 = Qtempz * x_31_3_1 + WQtempz * x_31_3_2 + CDtemp * ( x_31_0_1 - ABcom * x_31_0_2) +  3 * ABCDtemp * x_16_3_2;
                                        QUICKDouble x_32_9_1 = Qtempz * x_32_3_1 + WQtempz * x_32_3_2 + CDtemp * ( x_32_0_1 - ABcom * x_32_0_2);
                                        QUICKDouble x_33_9_1 = Qtempz * x_33_3_1 + WQtempz * x_33_3_2 + CDtemp * ( x_33_0_1 - ABcom * x_33_0_2);
                                        QUICKDouble x_34_9_1 = Qtempz * x_34_3_1 + WQtempz * x_34_3_2 + CDtemp * ( x_34_0_1 - ABcom * x_34_0_2) +  4 * ABCDtemp * x_19_3_2;
                                        
                                        //GSFS(0, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz, ABCDtemp, CDtemp, ABcom);
                                        
                                        QUICKDouble x_20_10_0 = Qtempx * x_20_5_0 + WQtempx * x_20_5_1 +  2 * ABCDtemp * x_12_5_1;
                                        QUICKDouble x_21_10_0 = Qtempx * x_21_5_0 + WQtempx * x_21_5_1 +  2 * ABCDtemp * x_14_5_1;
                                        QUICKDouble x_22_10_0 = Qtempx * x_22_5_0 + WQtempx * x_22_5_1;
                                        QUICKDouble x_23_10_0 = Qtempx * x_23_5_0 + WQtempx * x_23_5_1 +  2 * ABCDtemp * x_10_5_1;
                                        QUICKDouble x_24_10_0 = Qtempx * x_24_5_0 + WQtempx * x_24_5_1 + ABCDtemp * x_15_5_1;
                                        QUICKDouble x_25_10_0 = Qtempx * x_25_5_0 + WQtempx * x_25_5_1 + ABCDtemp * x_16_5_1;
                                        QUICKDouble x_26_10_0 = Qtempx * x_26_5_0 + WQtempx * x_26_5_1 +  3 * ABCDtemp * x_13_5_1;
                                        QUICKDouble x_27_10_0 = Qtempx * x_27_5_0 + WQtempx * x_27_5_1 + ABCDtemp * x_19_5_1;
                                        QUICKDouble x_28_10_0 = Qtempx * x_28_5_0 + WQtempx * x_28_5_1 +  3 * ABCDtemp * x_11_5_1;
                                        QUICKDouble x_29_10_0 = Qtempx * x_29_5_0 + WQtempx * x_29_5_1 + ABCDtemp * x_18_5_1;
                                        QUICKDouble x_30_10_0 = Qtempx * x_30_5_0 + WQtempx * x_30_5_1;
                                        QUICKDouble x_31_10_0 = Qtempx * x_31_5_0 + WQtempx * x_31_5_1;
                                        QUICKDouble x_32_10_0 = Qtempx * x_32_5_0 + WQtempx * x_32_5_1 +  4 * ABCDtemp * x_17_5_1;
                                        QUICKDouble x_33_10_0 = Qtempx * x_33_5_0 + WQtempx * x_33_5_1;
                                        QUICKDouble x_34_10_0 = Qtempx * x_34_5_0 + WQtempx * x_34_5_1;
                                        QUICKDouble x_20_11_0 = Qtempx * x_20_4_0 + WQtempx * x_20_4_1 + CDtemp * ( x_20_2_0 - ABcom * x_20_2_1) +  2 * ABCDtemp * x_12_4_1;
                                        QUICKDouble x_21_11_0 = Qtempx * x_21_4_0 + WQtempx * x_21_4_1 + CDtemp * ( x_21_2_0 - ABcom * x_21_2_1) +  2 * ABCDtemp * x_14_4_1;
                                        QUICKDouble x_22_11_0 = Qtempx * x_22_4_0 + WQtempx * x_22_4_1 + CDtemp * ( x_22_2_0 - ABcom * x_22_2_1);
                                        QUICKDouble x_23_11_0 = Qtempx * x_23_4_0 + WQtempx * x_23_4_1 + CDtemp * ( x_23_2_0 - ABcom * x_23_2_1) +  2 * ABCDtemp * x_10_4_1;
                                        QUICKDouble x_24_11_0 = Qtempx * x_24_4_0 + WQtempx * x_24_4_1 + CDtemp * ( x_24_2_0 - ABcom * x_24_2_1) + ABCDtemp * x_15_4_1;
                                        QUICKDouble x_25_11_0 = Qtempx * x_25_4_0 + WQtempx * x_25_4_1 + CDtemp * ( x_25_2_0 - ABcom * x_25_2_1) + ABCDtemp * x_16_4_1;
                                        QUICKDouble x_26_11_0 = Qtempx * x_26_4_0 + WQtempx * x_26_4_1 + CDtemp * ( x_26_2_0 - ABcom * x_26_2_1) +  3 * ABCDtemp * x_13_4_1;
                                        QUICKDouble x_27_11_0 = Qtempx * x_27_4_0 + WQtempx * x_27_4_1 + CDtemp * ( x_27_2_0 - ABcom * x_27_2_1) + ABCDtemp * x_19_4_1;
                                        QUICKDouble x_28_11_0 = Qtempx * x_28_4_0 + WQtempx * x_28_4_1 + CDtemp * ( x_28_2_0 - ABcom * x_28_2_1) +  3 * ABCDtemp * x_11_4_1;
                                        QUICKDouble x_29_11_0 = Qtempx * x_29_4_0 + WQtempx * x_29_4_1 + CDtemp * ( x_29_2_0 - ABcom * x_29_2_1) + ABCDtemp * x_18_4_1;
                                        QUICKDouble x_30_11_0 = Qtempx * x_30_4_0 + WQtempx * x_30_4_1 + CDtemp * ( x_30_2_0 - ABcom * x_30_2_1);
                                        QUICKDouble x_31_11_0 = Qtempx * x_31_4_0 + WQtempx * x_31_4_1 + CDtemp * ( x_31_2_0 - ABcom * x_31_2_1);
                                        QUICKDouble x_32_11_0 = Qtempx * x_32_4_0 + WQtempx * x_32_4_1 + CDtemp * ( x_32_2_0 - ABcom * x_32_2_1) +  4 * ABCDtemp * x_17_4_1;
                                        QUICKDouble x_33_11_0 = Qtempx * x_33_4_0 + WQtempx * x_33_4_1 + CDtemp * ( x_33_2_0 - ABcom * x_33_2_1);
                                        QUICKDouble x_34_11_0 = Qtempx * x_34_4_0 + WQtempx * x_34_4_1 + CDtemp * ( x_34_2_0 - ABcom * x_34_2_1);
                                        QUICKDouble x_20_12_0 = Qtempx * x_20_8_0 + WQtempx * x_20_8_1 +  2 * ABCDtemp * x_12_8_1;
                                        QUICKDouble x_21_12_0 = Qtempx * x_21_8_0 + WQtempx * x_21_8_1 +  2 * ABCDtemp * x_14_8_1;
                                        QUICKDouble x_22_12_0 = Qtempx * x_22_8_0 + WQtempx * x_22_8_1;
                                        QUICKDouble x_23_12_0 = Qtempx * x_23_8_0 + WQtempx * x_23_8_1 +  2 * ABCDtemp * x_10_8_1;
                                        QUICKDouble x_24_12_0 = Qtempx * x_24_8_0 + WQtempx * x_24_8_1 + ABCDtemp * x_15_8_1;
                                        QUICKDouble x_25_12_0 = Qtempx * x_25_8_0 + WQtempx * x_25_8_1 + ABCDtemp * x_16_8_1;
                                        QUICKDouble x_26_12_0 = Qtempx * x_26_8_0 + WQtempx * x_26_8_1 +  3 * ABCDtemp * x_13_8_1;
                                        QUICKDouble x_27_12_0 = Qtempx * x_27_8_0 + WQtempx * x_27_8_1 + ABCDtemp * x_19_8_1;
                                        QUICKDouble x_28_12_0 = Qtempx * x_28_8_0 + WQtempx * x_28_8_1 +  3 * ABCDtemp * x_11_8_1;
                                        QUICKDouble x_29_12_0 = Qtempx * x_29_8_0 + WQtempx * x_29_8_1 + ABCDtemp * x_18_8_1;
                                        QUICKDouble x_30_12_0 = Qtempx * x_30_8_0 + WQtempx * x_30_8_1;
                                        QUICKDouble x_31_12_0 = Qtempx * x_31_8_0 + WQtempx * x_31_8_1;
                                        QUICKDouble x_32_12_0 = Qtempx * x_32_8_0 + WQtempx * x_32_8_1 +  4 * ABCDtemp * x_17_8_1;
                                        QUICKDouble x_33_12_0 = Qtempx * x_33_8_0 + WQtempx * x_33_8_1;
                                        QUICKDouble x_34_12_0 = Qtempx * x_34_8_0 + WQtempx * x_34_8_1;
                                        QUICKDouble x_20_13_0 = Qtempx * x_20_6_0 + WQtempx * x_20_6_1 + CDtemp * ( x_20_3_0 - ABcom * x_20_3_1) +  2 * ABCDtemp * x_12_6_1;
                                        QUICKDouble x_21_13_0 = Qtempx * x_21_6_0 + WQtempx * x_21_6_1 + CDtemp * ( x_21_3_0 - ABcom * x_21_3_1) +  2 * ABCDtemp * x_14_6_1;
                                        QUICKDouble x_22_13_0 = Qtempx * x_22_6_0 + WQtempx * x_22_6_1 + CDtemp * ( x_22_3_0 - ABcom * x_22_3_1);
                                        QUICKDouble x_23_13_0 = Qtempx * x_23_6_0 + WQtempx * x_23_6_1 + CDtemp * ( x_23_3_0 - ABcom * x_23_3_1) +  2 * ABCDtemp * x_10_6_1;
                                        QUICKDouble x_24_13_0 = Qtempx * x_24_6_0 + WQtempx * x_24_6_1 + CDtemp * ( x_24_3_0 - ABcom * x_24_3_1) + ABCDtemp * x_15_6_1;
                                        QUICKDouble x_25_13_0 = Qtempx * x_25_6_0 + WQtempx * x_25_6_1 + CDtemp * ( x_25_3_0 - ABcom * x_25_3_1) + ABCDtemp * x_16_6_1;
                                        QUICKDouble x_26_13_0 = Qtempx * x_26_6_0 + WQtempx * x_26_6_1 + CDtemp * ( x_26_3_0 - ABcom * x_26_3_1) +  3 * ABCDtemp * x_13_6_1;
                                        QUICKDouble x_27_13_0 = Qtempx * x_27_6_0 + WQtempx * x_27_6_1 + CDtemp * ( x_27_3_0 - ABcom * x_27_3_1) + ABCDtemp * x_19_6_1;
                                        QUICKDouble x_28_13_0 = Qtempx * x_28_6_0 + WQtempx * x_28_6_1 + CDtemp * ( x_28_3_0 - ABcom * x_28_3_1) +  3 * ABCDtemp * x_11_6_1;
                                        QUICKDouble x_29_13_0 = Qtempx * x_29_6_0 + WQtempx * x_29_6_1 + CDtemp * ( x_29_3_0 - ABcom * x_29_3_1) + ABCDtemp * x_18_6_1;
                                        QUICKDouble x_30_13_0 = Qtempx * x_30_6_0 + WQtempx * x_30_6_1 + CDtemp * ( x_30_3_0 - ABcom * x_30_3_1);
                                        QUICKDouble x_31_13_0 = Qtempx * x_31_6_0 + WQtempx * x_31_6_1 + CDtemp * ( x_31_3_0 - ABcom * x_31_3_1);
                                        QUICKDouble x_32_13_0 = Qtempx * x_32_6_0 + WQtempx * x_32_6_1 + CDtemp * ( x_32_3_0 - ABcom * x_32_3_1) +  4 * ABCDtemp * x_17_6_1;
                                        QUICKDouble x_33_13_0 = Qtempx * x_33_6_0 + WQtempx * x_33_6_1 + CDtemp * ( x_33_3_0 - ABcom * x_33_3_1);
                                        QUICKDouble x_34_13_0 = Qtempx * x_34_6_0 + WQtempx * x_34_6_1 + CDtemp * ( x_34_3_0 - ABcom * x_34_3_1);
                                        QUICKDouble x_20_14_0 = Qtempx * x_20_9_0 + WQtempx * x_20_9_1 +  2 * ABCDtemp * x_12_9_1;
                                        QUICKDouble x_21_14_0 = Qtempx * x_21_9_0 + WQtempx * x_21_9_1 +  2 * ABCDtemp * x_14_9_1;
                                        QUICKDouble x_22_14_0 = Qtempx * x_22_9_0 + WQtempx * x_22_9_1;
                                        QUICKDouble x_23_14_0 = Qtempx * x_23_9_0 + WQtempx * x_23_9_1 +  2 * ABCDtemp * x_10_9_1;
                                        QUICKDouble x_24_14_0 = Qtempx * x_24_9_0 + WQtempx * x_24_9_1 + ABCDtemp * x_15_9_1;
                                        QUICKDouble x_25_14_0 = Qtempx * x_25_9_0 + WQtempx * x_25_9_1 + ABCDtemp * x_16_9_1;
                                        QUICKDouble x_26_14_0 = Qtempx * x_26_9_0 + WQtempx * x_26_9_1 +  3 * ABCDtemp * x_13_9_1;
                                        QUICKDouble x_27_14_0 = Qtempx * x_27_9_0 + WQtempx * x_27_9_1 + ABCDtemp * x_19_9_1;
                                        QUICKDouble x_28_14_0 = Qtempx * x_28_9_0 + WQtempx * x_28_9_1 +  3 * ABCDtemp * x_11_9_1;
                                        QUICKDouble x_29_14_0 = Qtempx * x_29_9_0 + WQtempx * x_29_9_1 + ABCDtemp * x_18_9_1;
                                        QUICKDouble x_30_14_0 = Qtempx * x_30_9_0 + WQtempx * x_30_9_1;
                                        QUICKDouble x_31_14_0 = Qtempx * x_31_9_0 + WQtempx * x_31_9_1;
                                        QUICKDouble x_32_14_0 = Qtempx * x_32_9_0 + WQtempx * x_32_9_1 +  4 * ABCDtemp * x_17_9_1;
                                        QUICKDouble x_33_14_0 = Qtempx * x_33_9_0 + WQtempx * x_33_9_1;
                                        QUICKDouble x_34_14_0 = Qtempx * x_34_9_0 + WQtempx * x_34_9_1;
                                        QUICKDouble x_20_15_0 = Qtempy * x_20_5_0 + WQtempy * x_20_5_1 + CDtemp * ( x_20_3_0 - ABcom * x_20_3_1) +  2 * ABCDtemp * x_11_5_1;
                                        QUICKDouble x_21_15_0 = Qtempy * x_21_5_0 + WQtempy * x_21_5_1 + CDtemp * ( x_21_3_0 - ABcom * x_21_3_1);
                                        QUICKDouble x_22_15_0 = Qtempy * x_22_5_0 + WQtempy * x_22_5_1 + CDtemp * ( x_22_3_0 - ABcom * x_22_3_1) +  2 * ABCDtemp * x_16_5_1;
                                        QUICKDouble x_23_15_0 = Qtempy * x_23_5_0 + WQtempy * x_23_5_1 + CDtemp * ( x_23_3_0 - ABcom * x_23_3_1) + ABCDtemp * x_13_5_1;
                                        QUICKDouble x_24_15_0 = Qtempy * x_24_5_0 + WQtempy * x_24_5_1 + CDtemp * ( x_24_3_0 - ABcom * x_24_3_1) +  2 * ABCDtemp * x_10_5_1;
                                        QUICKDouble x_25_15_0 = Qtempy * x_25_5_0 + WQtempy * x_25_5_1 + CDtemp * ( x_25_3_0 - ABcom * x_25_3_1) + ABCDtemp * x_14_5_1;
                                        QUICKDouble x_26_15_0 = Qtempy * x_26_5_0 + WQtempy * x_26_5_1 + CDtemp * ( x_26_3_0 - ABcom * x_26_3_1);
                                        QUICKDouble x_27_15_0 = Qtempy * x_27_5_0 + WQtempy * x_27_5_1 + CDtemp * ( x_27_3_0 - ABcom * x_27_3_1);
                                        QUICKDouble x_28_15_0 = Qtempy * x_28_5_0 + WQtempy * x_28_5_1 + CDtemp * ( x_28_3_0 - ABcom * x_28_3_1) + ABCDtemp * x_17_5_1;
                                        QUICKDouble x_29_15_0 = Qtempy * x_29_5_0 + WQtempy * x_29_5_1 + CDtemp * ( x_29_3_0 - ABcom * x_29_3_1) +  3 * ABCDtemp * x_12_5_1;
                                        QUICKDouble x_30_15_0 = Qtempy * x_30_5_0 + WQtempy * x_30_5_1 + CDtemp * ( x_30_3_0 - ABcom * x_30_3_1) +  3 * ABCDtemp * x_15_5_1;
                                        QUICKDouble x_31_15_0 = Qtempy * x_31_5_0 + WQtempy * x_31_5_1 + CDtemp * ( x_31_3_0 - ABcom * x_31_3_1) + ABCDtemp * x_19_5_1;
                                        QUICKDouble x_32_15_0 = Qtempy * x_32_5_0 + WQtempy * x_32_5_1 + CDtemp * ( x_32_3_0 - ABcom * x_32_3_1);
                                        QUICKDouble x_33_15_0 = Qtempy * x_33_5_0 + WQtempy * x_33_5_1 + CDtemp * ( x_33_3_0 - ABcom * x_33_3_1) +  4 * ABCDtemp * x_18_5_1;
                                        QUICKDouble x_34_15_0 = Qtempy * x_34_5_0 + WQtempy * x_34_5_1 + CDtemp * ( x_34_3_0 - ABcom * x_34_3_1);
                                        QUICKDouble x_20_16_0 = Qtempy * x_20_9_0 + WQtempy * x_20_9_1 +  2 * ABCDtemp * x_11_9_1;
                                        QUICKDouble x_21_16_0 = Qtempy * x_21_9_0 + WQtempy * x_21_9_1;
                                        QUICKDouble x_22_16_0 = Qtempy * x_22_9_0 + WQtempy * x_22_9_1 +  2 * ABCDtemp * x_16_9_1;
                                        QUICKDouble x_23_16_0 = Qtempy * x_23_9_0 + WQtempy * x_23_9_1 + ABCDtemp * x_13_9_1;
                                        QUICKDouble x_24_16_0 = Qtempy * x_24_9_0 + WQtempy * x_24_9_1 +  2 * ABCDtemp * x_10_9_1;
                                        QUICKDouble x_25_16_0 = Qtempy * x_25_9_0 + WQtempy * x_25_9_1 + ABCDtemp * x_14_9_1;
                                        QUICKDouble x_26_16_0 = Qtempy * x_26_9_0 + WQtempy * x_26_9_1;
                                        QUICKDouble x_27_16_0 = Qtempy * x_27_9_0 + WQtempy * x_27_9_1;
                                        QUICKDouble x_28_16_0 = Qtempy * x_28_9_0 + WQtempy * x_28_9_1 + ABCDtemp * x_17_9_1;
                                        QUICKDouble x_29_16_0 = Qtempy * x_29_9_0 + WQtempy * x_29_9_1 +  3 * ABCDtemp * x_12_9_1;
                                        QUICKDouble x_30_16_0 = Qtempy * x_30_9_0 + WQtempy * x_30_9_1 +  3 * ABCDtemp * x_15_9_1;
                                        QUICKDouble x_31_16_0 = Qtempy * x_31_9_0 + WQtempy * x_31_9_1 + ABCDtemp * x_19_9_1;
                                        QUICKDouble x_32_16_0 = Qtempy * x_32_9_0 + WQtempy * x_32_9_1;
                                        QUICKDouble x_33_16_0 = Qtempy * x_33_9_0 + WQtempy * x_33_9_1 +  4 * ABCDtemp * x_18_9_1;
                                        QUICKDouble x_34_16_0 = Qtempy * x_34_9_0 + WQtempy * x_34_9_1;
                                        QUICKDouble x_20_17_0 = Qtempx * x_20_7_0 + WQtempx * x_20_7_1 +  2 * CDtemp * ( x_20_1_0 - ABcom * x_20_1_1) +  2 * ABCDtemp * x_12_7_1;
                                        QUICKDouble x_21_17_0 = Qtempx * x_21_7_0 + WQtempx * x_21_7_1 +  2 * CDtemp * ( x_21_1_0 - ABcom * x_21_1_1) +  2 * ABCDtemp * x_14_7_1;
                                        QUICKDouble x_22_17_0 = Qtempx * x_22_7_0 + WQtempx * x_22_7_1 +  2 * CDtemp * ( x_22_1_0 - ABcom * x_22_1_1);
                                        QUICKDouble x_23_17_0 = Qtempx * x_23_7_0 + WQtempx * x_23_7_1 +  2 * CDtemp * ( x_23_1_0 - ABcom * x_23_1_1) +  2 * ABCDtemp * x_10_7_1;
                                        QUICKDouble x_24_17_0 = Qtempx * x_24_7_0 + WQtempx * x_24_7_1 +  2 * CDtemp * ( x_24_1_0 - ABcom * x_24_1_1) + ABCDtemp * x_15_7_1;
                                        QUICKDouble x_25_17_0 = Qtempx * x_25_7_0 + WQtempx * x_25_7_1 +  2 * CDtemp * ( x_25_1_0 - ABcom * x_25_1_1) + ABCDtemp * x_16_7_1;
                                        QUICKDouble x_26_17_0 = Qtempx * x_26_7_0 + WQtempx * x_26_7_1 +  2 * CDtemp * ( x_26_1_0 - ABcom * x_26_1_1) +  3 * ABCDtemp * x_13_7_1;
                                        QUICKDouble x_27_17_0 = Qtempx * x_27_7_0 + WQtempx * x_27_7_1 +  2 * CDtemp * ( x_27_1_0 - ABcom * x_27_1_1) + ABCDtemp * x_19_7_1;
                                        QUICKDouble x_28_17_0 = Qtempx * x_28_7_0 + WQtempx * x_28_7_1 +  2 * CDtemp * ( x_28_1_0 - ABcom * x_28_1_1) +  3 * ABCDtemp * x_11_7_1;
                                        QUICKDouble x_29_17_0 = Qtempx * x_29_7_0 + WQtempx * x_29_7_1 +  2 * CDtemp * ( x_29_1_0 - ABcom * x_29_1_1) + ABCDtemp * x_18_7_1;
                                        QUICKDouble x_30_17_0 = Qtempx * x_30_7_0 + WQtempx * x_30_7_1 +  2 * CDtemp * ( x_30_1_0 - ABcom * x_30_1_1);
                                        QUICKDouble x_31_17_0 = Qtempx * x_31_7_0 + WQtempx * x_31_7_1 +  2 * CDtemp * ( x_31_1_0 - ABcom * x_31_1_1);
                                        QUICKDouble x_32_17_0 = Qtempx * x_32_7_0 + WQtempx * x_32_7_1 +  2 * CDtemp * ( x_32_1_0 - ABcom * x_32_1_1) +  4 * ABCDtemp * x_17_7_1;
                                        QUICKDouble x_33_17_0 = Qtempx * x_33_7_0 + WQtempx * x_33_7_1 +  2 * CDtemp * ( x_33_1_0 - ABcom * x_33_1_1);
                                        QUICKDouble x_34_17_0 = Qtempx * x_34_7_0 + WQtempx * x_34_7_1 +  2 * CDtemp * ( x_34_1_0 - ABcom * x_34_1_1);
                                        QUICKDouble x_20_18_0 = Qtempy * x_20_8_0 + WQtempy * x_20_8_1 +  2 * CDtemp * ( x_20_2_0 - ABcom * x_20_2_1) +  2 * ABCDtemp * x_11_8_1;
                                        QUICKDouble x_21_18_0 = Qtempy * x_21_8_0 + WQtempy * x_21_8_1 +  2 * CDtemp * ( x_21_2_0 - ABcom * x_21_2_1);
                                        QUICKDouble x_22_18_0 = Qtempy * x_22_8_0 + WQtempy * x_22_8_1 +  2 * CDtemp * ( x_22_2_0 - ABcom * x_22_2_1) +  2 * ABCDtemp * x_16_8_1;
                                        QUICKDouble x_23_18_0 = Qtempy * x_23_8_0 + WQtempy * x_23_8_1 +  2 * CDtemp * ( x_23_2_0 - ABcom * x_23_2_1) + ABCDtemp * x_13_8_1;
                                        QUICKDouble x_24_18_0 = Qtempy * x_24_8_0 + WQtempy * x_24_8_1 +  2 * CDtemp * ( x_24_2_0 - ABcom * x_24_2_1) +  2 * ABCDtemp * x_10_8_1;
                                        QUICKDouble x_25_18_0 = Qtempy * x_25_8_0 + WQtempy * x_25_8_1 +  2 * CDtemp * ( x_25_2_0 - ABcom * x_25_2_1) + ABCDtemp * x_14_8_1;
                                        QUICKDouble x_26_18_0 = Qtempy * x_26_8_0 + WQtempy * x_26_8_1 +  2 * CDtemp * ( x_26_2_0 - ABcom * x_26_2_1);
                                        QUICKDouble x_27_18_0 = Qtempy * x_27_8_0 + WQtempy * x_27_8_1 +  2 * CDtemp * ( x_27_2_0 - ABcom * x_27_2_1);
                                        QUICKDouble x_28_18_0 = Qtempy * x_28_8_0 + WQtempy * x_28_8_1 +  2 * CDtemp * ( x_28_2_0 - ABcom * x_28_2_1) + ABCDtemp * x_17_8_1;
                                        QUICKDouble x_29_18_0 = Qtempy * x_29_8_0 + WQtempy * x_29_8_1 +  2 * CDtemp * ( x_29_2_0 - ABcom * x_29_2_1) +  3 * ABCDtemp * x_12_8_1;
                                        QUICKDouble x_30_18_0 = Qtempy * x_30_8_0 + WQtempy * x_30_8_1 +  2 * CDtemp * ( x_30_2_0 - ABcom * x_30_2_1) +  3 * ABCDtemp * x_15_8_1;
                                        QUICKDouble x_31_18_0 = Qtempy * x_31_8_0 + WQtempy * x_31_8_1 +  2 * CDtemp * ( x_31_2_0 - ABcom * x_31_2_1) + ABCDtemp * x_19_8_1;
                                        QUICKDouble x_32_18_0 = Qtempy * x_32_8_0 + WQtempy * x_32_8_1 +  2 * CDtemp * ( x_32_2_0 - ABcom * x_32_2_1);
                                        QUICKDouble x_33_18_0 = Qtempy * x_33_8_0 + WQtempy * x_33_8_1 +  2 * CDtemp * ( x_33_2_0 - ABcom * x_33_2_1) +  4 * ABCDtemp * x_18_8_1;
                                        QUICKDouble x_34_18_0 = Qtempy * x_34_8_0 + WQtempy * x_34_8_1 +  2 * CDtemp * ( x_34_2_0 - ABcom * x_34_2_1);
                                        QUICKDouble x_20_19_0 = Qtempz * x_20_9_0 + WQtempz * x_20_9_1 +  2 * CDtemp * ( x_20_3_0 - ABcom * x_20_3_1);
                                        QUICKDouble x_21_19_0 = Qtempz * x_21_9_0 + WQtempz * x_21_9_1 +  2 * CDtemp * ( x_21_3_0 - ABcom * x_21_3_1) +  2 * ABCDtemp * x_13_9_1;
                                        QUICKDouble x_22_19_0 = Qtempz * x_22_9_0 + WQtempz * x_22_9_1 +  2 * CDtemp * ( x_22_3_0 - ABcom * x_22_3_1) +  2 * ABCDtemp * x_15_9_1;
                                        QUICKDouble x_23_19_0 = Qtempz * x_23_9_0 + WQtempz * x_23_9_1 +  2 * CDtemp * ( x_23_3_0 - ABcom * x_23_3_1) + ABCDtemp * x_11_9_1;
                                        QUICKDouble x_24_19_0 = Qtempz * x_24_9_0 + WQtempz * x_24_9_1 +  2 * CDtemp * ( x_24_3_0 - ABcom * x_24_3_1) + ABCDtemp * x_12_9_1;
                                        QUICKDouble x_25_19_0 = Qtempz * x_25_9_0 + WQtempz * x_25_9_1 +  2 * CDtemp * ( x_25_3_0 - ABcom * x_25_3_1) +  2 * ABCDtemp * x_10_9_1;
                                        QUICKDouble x_26_19_0 = Qtempz * x_26_9_0 + WQtempz * x_26_9_1 +  2 * CDtemp * ( x_26_3_0 - ABcom * x_26_3_1) + ABCDtemp * x_17_9_1;
                                        QUICKDouble x_27_19_0 = Qtempz * x_27_9_0 + WQtempz * x_27_9_1 +  2 * CDtemp * ( x_27_3_0 - ABcom * x_27_3_1) +  3 * ABCDtemp * x_14_9_1;
                                        QUICKDouble x_28_19_0 = Qtempz * x_28_9_0 + WQtempz * x_28_9_1 +  2 * CDtemp * ( x_28_3_0 - ABcom * x_28_3_1);
                                        QUICKDouble x_29_19_0 = Qtempz * x_29_9_0 + WQtempz * x_29_9_1 +  2 * CDtemp * ( x_29_3_0 - ABcom * x_29_3_1);
                                        QUICKDouble x_30_19_0 = Qtempz * x_30_9_0 + WQtempz * x_30_9_1 +  2 * CDtemp * ( x_30_3_0 - ABcom * x_30_3_1) + ABCDtemp * x_18_9_1;
                                        QUICKDouble x_31_19_0 = Qtempz * x_31_9_0 + WQtempz * x_31_9_1 +  2 * CDtemp * ( x_31_3_0 - ABcom * x_31_3_1) +  3 * ABCDtemp * x_16_9_1;
                                        QUICKDouble x_32_19_0 = Qtempz * x_32_9_0 + WQtempz * x_32_9_1 +  2 * CDtemp * ( x_32_3_0 - ABcom * x_32_3_1);
                                        QUICKDouble x_33_19_0 = Qtempz * x_33_9_0 + WQtempz * x_33_9_1 +  2 * CDtemp * ( x_33_3_0 - ABcom * x_33_3_1);
                                        QUICKDouble x_34_19_0 = Qtempz * x_34_9_0 + WQtempz * x_34_9_1 +  2 * CDtemp * ( x_34_3_0 - ABcom * x_34_3_1) +  4 * ABCDtemp * x_19_9_1;
                                        
                                        LOC2(store,20,10, STOREDIM, STOREDIM) += x_20_10_0;
                                        LOC2(store,20,11, STOREDIM, STOREDIM) += x_20_11_0;
                                        LOC2(store,20,12, STOREDIM, STOREDIM) += x_20_12_0;
                                        LOC2(store,20,13, STOREDIM, STOREDIM) += x_20_13_0;
                                        LOC2(store,20,14, STOREDIM, STOREDIM) += x_20_14_0;
                                        LOC2(store,20,15, STOREDIM, STOREDIM) += x_20_15_0;
                                        LOC2(store,20,16, STOREDIM, STOREDIM) += x_20_16_0;
                                        LOC2(store,20,17, STOREDIM, STOREDIM) += x_20_17_0;
                                        LOC2(store,20,18, STOREDIM, STOREDIM) += x_20_18_0;
                                        LOC2(store,20,19, STOREDIM, STOREDIM) += x_20_19_0;
                                        LOC2(store,21,10, STOREDIM, STOREDIM) += x_21_10_0;
                                        LOC2(store,21,11, STOREDIM, STOREDIM) += x_21_11_0;
                                        LOC2(store,21,12, STOREDIM, STOREDIM) += x_21_12_0;
                                        LOC2(store,21,13, STOREDIM, STOREDIM) += x_21_13_0;
                                        LOC2(store,21,14, STOREDIM, STOREDIM) += x_21_14_0;
                                        LOC2(store,21,15, STOREDIM, STOREDIM) += x_21_15_0;
                                        LOC2(store,21,16, STOREDIM, STOREDIM) += x_21_16_0;
                                        LOC2(store,21,17, STOREDIM, STOREDIM) += x_21_17_0;
                                        LOC2(store,21,18, STOREDIM, STOREDIM) += x_21_18_0;
                                        LOC2(store,21,19, STOREDIM, STOREDIM) += x_21_19_0;
                                        LOC2(store,22,10, STOREDIM, STOREDIM) += x_22_10_0;
                                        LOC2(store,22,11, STOREDIM, STOREDIM) += x_22_11_0;
                                        LOC2(store,22,12, STOREDIM, STOREDIM) += x_22_12_0;
                                        LOC2(store,22,13, STOREDIM, STOREDIM) += x_22_13_0;
                                        LOC2(store,22,14, STOREDIM, STOREDIM) += x_22_14_0;
                                        LOC2(store,22,15, STOREDIM, STOREDIM) += x_22_15_0;
                                        LOC2(store,22,16, STOREDIM, STOREDIM) += x_22_16_0;
                                        LOC2(store,22,17, STOREDIM, STOREDIM) += x_22_17_0;
                                        LOC2(store,22,18, STOREDIM, STOREDIM) += x_22_18_0;
                                        LOC2(store,22,19, STOREDIM, STOREDIM) += x_22_19_0;
                                        LOC2(store,23,10, STOREDIM, STOREDIM) += x_23_10_0;
                                        LOC2(store,23,11, STOREDIM, STOREDIM) += x_23_11_0;
                                        LOC2(store,23,12, STOREDIM, STOREDIM) += x_23_12_0;
                                        LOC2(store,23,13, STOREDIM, STOREDIM) += x_23_13_0;
                                        LOC2(store,23,14, STOREDIM, STOREDIM) += x_23_14_0;
                                        LOC2(store,23,15, STOREDIM, STOREDIM) += x_23_15_0;
                                        LOC2(store,23,16, STOREDIM, STOREDIM) += x_23_16_0;
                                        LOC2(store,23,17, STOREDIM, STOREDIM) += x_23_17_0;
                                        LOC2(store,23,18, STOREDIM, STOREDIM) += x_23_18_0;
                                        LOC2(store,23,19, STOREDIM, STOREDIM) += x_23_19_0;
                                        LOC2(store,24,10, STOREDIM, STOREDIM) += x_24_10_0;
                                        LOC2(store,24,11, STOREDIM, STOREDIM) += x_24_11_0;
                                        LOC2(store,24,12, STOREDIM, STOREDIM) += x_24_12_0;
                                        LOC2(store,24,13, STOREDIM, STOREDIM) += x_24_13_0;
                                        LOC2(store,24,14, STOREDIM, STOREDIM) += x_24_14_0;
                                        LOC2(store,24,15, STOREDIM, STOREDIM) += x_24_15_0;
                                        LOC2(store,24,16, STOREDIM, STOREDIM) += x_24_16_0;
                                        LOC2(store,24,17, STOREDIM, STOREDIM) += x_24_17_0;
                                        LOC2(store,24,18, STOREDIM, STOREDIM) += x_24_18_0;
                                        LOC2(store,24,19, STOREDIM, STOREDIM) += x_24_19_0;
                                        LOC2(store,25,10, STOREDIM, STOREDIM) += x_25_10_0;
                                        LOC2(store,25,11, STOREDIM, STOREDIM) += x_25_11_0;
                                        LOC2(store,25,12, STOREDIM, STOREDIM) += x_25_12_0;
                                        LOC2(store,25,13, STOREDIM, STOREDIM) += x_25_13_0;
                                        LOC2(store,25,14, STOREDIM, STOREDIM) += x_25_14_0;
                                        LOC2(store,25,15, STOREDIM, STOREDIM) += x_25_15_0;
                                        LOC2(store,25,16, STOREDIM, STOREDIM) += x_25_16_0;
                                        LOC2(store,25,17, STOREDIM, STOREDIM) += x_25_17_0;
                                        LOC2(store,25,18, STOREDIM, STOREDIM) += x_25_18_0;
                                        LOC2(store,25,19, STOREDIM, STOREDIM) += x_25_19_0;
                                        LOC2(store,26,10, STOREDIM, STOREDIM) += x_26_10_0;
                                        LOC2(store,26,11, STOREDIM, STOREDIM) += x_26_11_0;
                                        LOC2(store,26,12, STOREDIM, STOREDIM) += x_26_12_0;
                                        LOC2(store,26,13, STOREDIM, STOREDIM) += x_26_13_0;
                                        LOC2(store,26,14, STOREDIM, STOREDIM) += x_26_14_0;
                                        LOC2(store,26,15, STOREDIM, STOREDIM) += x_26_15_0;
                                        LOC2(store,26,16, STOREDIM, STOREDIM) += x_26_16_0;
                                        LOC2(store,26,17, STOREDIM, STOREDIM) += x_26_17_0;
                                        LOC2(store,26,18, STOREDIM, STOREDIM) += x_26_18_0;
                                        LOC2(store,26,19, STOREDIM, STOREDIM) += x_26_19_0;
                                        LOC2(store,27,10, STOREDIM, STOREDIM) += x_27_10_0;
                                        LOC2(store,27,11, STOREDIM, STOREDIM) += x_27_11_0;
                                        LOC2(store,27,12, STOREDIM, STOREDIM) += x_27_12_0;
                                        LOC2(store,27,13, STOREDIM, STOREDIM) += x_27_13_0;
                                        LOC2(store,27,14, STOREDIM, STOREDIM) += x_27_14_0;
                                        LOC2(store,27,15, STOREDIM, STOREDIM) += x_27_15_0;
                                        LOC2(store,27,16, STOREDIM, STOREDIM) += x_27_16_0;
                                        LOC2(store,27,17, STOREDIM, STOREDIM) += x_27_17_0;
                                        LOC2(store,27,18, STOREDIM, STOREDIM) += x_27_18_0;
                                        LOC2(store,27,19, STOREDIM, STOREDIM) += x_27_19_0;
                                        LOC2(store,28,10, STOREDIM, STOREDIM) += x_28_10_0;
                                        LOC2(store,28,11, STOREDIM, STOREDIM) += x_28_11_0;
                                        LOC2(store,28,12, STOREDIM, STOREDIM) += x_28_12_0;
                                        LOC2(store,28,13, STOREDIM, STOREDIM) += x_28_13_0;
                                        LOC2(store,28,14, STOREDIM, STOREDIM) += x_28_14_0;
                                        LOC2(store,28,15, STOREDIM, STOREDIM) += x_28_15_0;
                                        LOC2(store,28,16, STOREDIM, STOREDIM) += x_28_16_0;
                                        LOC2(store,28,17, STOREDIM, STOREDIM) += x_28_17_0;
                                        LOC2(store,28,18, STOREDIM, STOREDIM) += x_28_18_0;
                                        LOC2(store,28,19, STOREDIM, STOREDIM) += x_28_19_0;
                                        LOC2(store,29,10, STOREDIM, STOREDIM) += x_29_10_0;
                                        LOC2(store,29,11, STOREDIM, STOREDIM) += x_29_11_0;
                                        LOC2(store,29,12, STOREDIM, STOREDIM) += x_29_12_0;
                                        LOC2(store,29,13, STOREDIM, STOREDIM) += x_29_13_0;
                                        LOC2(store,29,14, STOREDIM, STOREDIM) += x_29_14_0;
                                        LOC2(store,29,15, STOREDIM, STOREDIM) += x_29_15_0;
                                        LOC2(store,29,16, STOREDIM, STOREDIM) += x_29_16_0;
                                        LOC2(store,29,17, STOREDIM, STOREDIM) += x_29_17_0;
                                        LOC2(store,29,18, STOREDIM, STOREDIM) += x_29_18_0;
                                        LOC2(store,29,19, STOREDIM, STOREDIM) += x_29_19_0;
                                        LOC2(store,30,10, STOREDIM, STOREDIM) += x_30_10_0;
                                        LOC2(store,30,11, STOREDIM, STOREDIM) += x_30_11_0;
                                        LOC2(store,30,12, STOREDIM, STOREDIM) += x_30_12_0;
                                        LOC2(store,30,13, STOREDIM, STOREDIM) += x_30_13_0;
                                        LOC2(store,30,14, STOREDIM, STOREDIM) += x_30_14_0;
                                        LOC2(store,30,15, STOREDIM, STOREDIM) += x_30_15_0;
                                        LOC2(store,30,16, STOREDIM, STOREDIM) += x_30_16_0;
                                        LOC2(store,30,17, STOREDIM, STOREDIM) += x_30_17_0;
                                        LOC2(store,30,18, STOREDIM, STOREDIM) += x_30_18_0;
                                        LOC2(store,30,19, STOREDIM, STOREDIM) += x_30_19_0;
                                        LOC2(store,31,10, STOREDIM, STOREDIM) += x_31_10_0;
                                        LOC2(store,31,11, STOREDIM, STOREDIM) += x_31_11_0;
                                        LOC2(store,31,12, STOREDIM, STOREDIM) += x_31_12_0;
                                        LOC2(store,31,13, STOREDIM, STOREDIM) += x_31_13_0;
                                        LOC2(store,31,14, STOREDIM, STOREDIM) += x_31_14_0;
                                        LOC2(store,31,15, STOREDIM, STOREDIM) += x_31_15_0;
                                        LOC2(store,31,16, STOREDIM, STOREDIM) += x_31_16_0;
                                        LOC2(store,31,17, STOREDIM, STOREDIM) += x_31_17_0;
                                        LOC2(store,31,18, STOREDIM, STOREDIM) += x_31_18_0;
                                        LOC2(store,31,19, STOREDIM, STOREDIM) += x_31_19_0;
                                        LOC2(store,32,10, STOREDIM, STOREDIM) += x_32_10_0;
                                        LOC2(store,32,11, STOREDIM, STOREDIM) += x_32_11_0;
                                        LOC2(store,32,12, STOREDIM, STOREDIM) += x_32_12_0;
                                        LOC2(store,32,13, STOREDIM, STOREDIM) += x_32_13_0;
                                        LOC2(store,32,14, STOREDIM, STOREDIM) += x_32_14_0;
                                        LOC2(store,32,15, STOREDIM, STOREDIM) += x_32_15_0;
                                        LOC2(store,32,16, STOREDIM, STOREDIM) += x_32_16_0;
                                        LOC2(store,32,17, STOREDIM, STOREDIM) += x_32_17_0;
                                        LOC2(store,32,18, STOREDIM, STOREDIM) += x_32_18_0;
                                        LOC2(store,32,19, STOREDIM, STOREDIM) += x_32_19_0;
                                        LOC2(store,33,10, STOREDIM, STOREDIM) += x_33_10_0;
                                        LOC2(store,33,11, STOREDIM, STOREDIM) += x_33_11_0;
                                        LOC2(store,33,12, STOREDIM, STOREDIM) += x_33_12_0;
                                        LOC2(store,33,13, STOREDIM, STOREDIM) += x_33_13_0;
                                        LOC2(store,33,14, STOREDIM, STOREDIM) += x_33_14_0;
                                        LOC2(store,33,15, STOREDIM, STOREDIM) += x_33_15_0;
                                        LOC2(store,33,16, STOREDIM, STOREDIM) += x_33_16_0;
                                        LOC2(store,33,17, STOREDIM, STOREDIM) += x_33_17_0;
                                        LOC2(store,33,18, STOREDIM, STOREDIM) += x_33_18_0;
                                        LOC2(store,33,19, STOREDIM, STOREDIM) += x_33_19_0;
                                        LOC2(store,34,10, STOREDIM, STOREDIM) += x_34_10_0;
                                        LOC2(store,34,11, STOREDIM, STOREDIM) += x_34_11_0;
                                        LOC2(store,34,12, STOREDIM, STOREDIM) += x_34_12_0;
                                        LOC2(store,34,13, STOREDIM, STOREDIM) += x_34_13_0;
                                        LOC2(store,34,14, STOREDIM, STOREDIM) += x_34_14_0;
                                        LOC2(store,34,15, STOREDIM, STOREDIM) += x_34_15_0;
                                        LOC2(store,34,16, STOREDIM, STOREDIM) += x_34_16_0;
                                        LOC2(store,34,17, STOREDIM, STOREDIM) += x_34_17_0;
                                        LOC2(store,34,18, STOREDIM, STOREDIM) += x_34_18_0;
                                        LOC2(store,34,19, STOREDIM, STOREDIM) += x_34_19_0;
                                        
                                        
                                        if (I+J ==4 && K+L ==4) {
                                            //SSPS(5, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);
                                            QUICKDouble x_0_1_5 = Qtempx * VY( 0, 0, 5) + WQtempx * VY( 0, 0, 6);
                                            QUICKDouble x_0_2_5 = Qtempy * VY( 0, 0, 5) + WQtempy * VY( 0, 0, 6);
                                            QUICKDouble x_0_3_5 = Qtempz * VY( 0, 0, 5) + WQtempz * VY( 0, 0, 6);
                                            
                                            //SSDS(4, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz, CDtemp, ABcom);
                                            
                                            QUICKDouble x_0_4_4 = Qtempx * x_0_2_4 + WQtempx * x_0_2_5;
                                            QUICKDouble x_0_5_4 = Qtempy * x_0_3_4 + WQtempy * x_0_3_5;
                                            QUICKDouble x_0_6_4 = Qtempx * x_0_3_4 + WQtempx * x_0_3_5;
                                            
                                            QUICKDouble x_0_7_4 = Qtempx * x_0_1_4 + WQtempx * x_0_1_5+ CDtemp*(VY( 0, 0, 4) - ABcom * VY( 0, 0, 5));
                                            QUICKDouble x_0_8_4 = Qtempy * x_0_2_4 + WQtempy * x_0_2_5+ CDtemp*(VY( 0, 0, 4) - ABcom * VY( 0, 0, 5));
                                            QUICKDouble x_0_9_4 = Qtempz * x_0_3_4 + WQtempz * x_0_3_5+ CDtemp*(VY( 0, 0, 4) - ABcom * VY( 0, 0, 5));
                                            
                                            //PSDS(2, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz, ABCDtemp);
                                            
                                            QUICKDouble x_1_4_2 = Ptempx * x_0_4_2 + WPtempx * x_0_4_3 + ABCDtemp * x_0_2_3;
                                            QUICKDouble x_2_4_2 = Ptempy * x_0_4_2 + WPtempy * x_0_4_3 + ABCDtemp * x_0_1_3;
                                            QUICKDouble x_3_4_2 = Ptempz * x_0_4_2 + WPtempz * x_0_4_3;
                                            
                                            QUICKDouble x_1_5_2 = Ptempx * x_0_5_2 + WPtempx * x_0_5_3;
                                            QUICKDouble x_2_5_2 = Ptempy * x_0_5_2 + WPtempy * x_0_5_3 + ABCDtemp * x_0_3_3;
                                            QUICKDouble x_3_5_2 = Ptempz * x_0_5_2 + WPtempz * x_0_5_3 + ABCDtemp * x_0_2_3;
                                            
                                            QUICKDouble x_1_6_2 = Ptempx * x_0_6_2 + WPtempx * x_0_6_3 + ABCDtemp * x_0_3_3;
                                            QUICKDouble x_2_6_2 = Ptempy * x_0_6_2 + WPtempy * x_0_6_3;
                                            QUICKDouble x_3_6_2 = Ptempz * x_0_6_2 + WPtempz * x_0_6_3 + ABCDtemp * x_0_1_3;
                                            
                                            QUICKDouble x_1_7_2 = Ptempx * x_0_7_2 + WPtempx * x_0_7_3 + ABCDtemp * x_0_1_3 * 2;
                                            QUICKDouble x_2_7_2 = Ptempy * x_0_7_2 + WPtempy * x_0_7_3;
                                            QUICKDouble x_3_7_2 = Ptempz * x_0_7_2 + WPtempz * x_0_7_3;
                                            
                                            QUICKDouble x_1_8_2 = Ptempx * x_0_8_2 + WPtempx * x_0_8_3;
                                            QUICKDouble x_2_8_2 = Ptempy * x_0_8_2 + WPtempy * x_0_8_3 + ABCDtemp * x_0_2_3 * 2;
                                            QUICKDouble x_3_8_2 = Ptempz * x_0_8_2 + WPtempz * x_0_8_3;
                                            
                                            QUICKDouble x_1_9_2 = Ptempx * x_0_9_2 + WPtempx * x_0_9_3;
                                            QUICKDouble x_2_9_2 = Ptempy * x_0_9_2 + WPtempy * x_0_9_3;
                                            QUICKDouble x_3_9_2 = Ptempz * x_0_9_2 + WPtempz * x_0_9_3 + ABCDtemp * x_0_3_3 * 2;    
                                            //DSPS(3, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz, ABCDtemp);
                                            
                                            QUICKDouble x_4_1_3 = Qtempx * x_4_0_3 + WQtempx * x_4_0_4 + ABCDtemp * x_2_0_4;
                                            QUICKDouble x_4_2_3 = Qtempy * x_4_0_3 + WQtempy * x_4_0_4 + ABCDtemp * x_1_0_4;
                                            QUICKDouble x_4_3_3 = Qtempz * x_4_0_3 + WQtempz * x_4_0_4;
                                            
                                            QUICKDouble x_5_1_3 = Qtempx * x_5_0_3 + WQtempx * x_5_0_4;
                                            QUICKDouble x_5_2_3 = Qtempy * x_5_0_3 + WQtempy * x_5_0_4 + ABCDtemp * x_3_0_4;
                                            QUICKDouble x_5_3_3 = Qtempz * x_5_0_3 + WQtempz * x_5_0_4 + ABCDtemp * x_2_0_4;
                                            
                                            QUICKDouble x_6_1_3 = Qtempx * x_6_0_3 + WQtempx * x_6_0_4 + ABCDtemp * x_3_0_4;
                                            QUICKDouble x_6_2_3 = Qtempy * x_6_0_3 + WQtempy * x_6_0_4;
                                            QUICKDouble x_6_3_3 = Qtempz * x_6_0_3 + WQtempz * x_6_0_4 + ABCDtemp * x_1_0_4;
                                            
                                            QUICKDouble x_7_1_3 = Qtempx * x_7_0_3 + WQtempx * x_7_0_4 + ABCDtemp * x_1_0_4 * 2;
                                            QUICKDouble x_7_2_3 = Qtempy * x_7_0_3 + WQtempy * x_7_0_4;
                                            QUICKDouble x_7_3_3 = Qtempz * x_7_0_3 + WQtempz * x_7_0_4;
                                            
                                            QUICKDouble x_8_1_3 = Qtempx * x_8_0_3 + WQtempx * x_8_0_4;
                                            QUICKDouble x_8_2_3 = Qtempy * x_8_0_3 + WQtempy * x_8_0_4 + ABCDtemp * x_2_0_4 * 2;
                                            QUICKDouble x_8_3_3 = Qtempz * x_8_0_3 + WQtempz * x_8_0_4;
                                            
                                            QUICKDouble x_9_1_3 = Qtempx * x_9_0_3 + WQtempx * x_9_0_4;
                                            QUICKDouble x_9_2_3 = Qtempy * x_9_0_3 + WQtempy * x_9_0_4;
                                            QUICKDouble x_9_3_3 = Qtempz * x_9_0_3 + WQtempz * x_9_0_4 + ABCDtemp * x_3_0_4 * 2;            
                                            
                                            //FSDS(2, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz, ABCDtemp, CDtemp, ABcom);
                                            
                                            QUICKDouble x_10_4_2 = Qtempx * x_10_2_2 + WQtempx * x_10_2_3 + ABCDtemp * x_5_2_3;
                                            QUICKDouble x_11_4_2 = Qtempx * x_11_2_2 + WQtempx * x_11_2_3 +  2 * ABCDtemp * x_4_2_3;
                                            QUICKDouble x_12_4_2 = Qtempx * x_12_2_2 + WQtempx * x_12_2_3 + ABCDtemp * x_8_2_3;
                                            QUICKDouble x_13_4_2 = Qtempx * x_13_2_2 + WQtempx * x_13_2_3 +  2 * ABCDtemp * x_6_2_3;
                                            QUICKDouble x_14_4_2 = Qtempx * x_14_2_2 + WQtempx * x_14_2_3 + ABCDtemp * x_9_2_3;
                                            QUICKDouble x_15_4_2 = Qtempx * x_15_2_2 + WQtempx * x_15_2_3;
                                            QUICKDouble x_16_4_2 = Qtempx * x_16_2_2 + WQtempx * x_16_2_3;
                                            QUICKDouble x_17_4_2 = Qtempx * x_17_2_2 + WQtempx * x_17_2_3 +  3 * ABCDtemp * x_7_2_3;
                                            QUICKDouble x_18_4_2 = Qtempx * x_18_2_2 + WQtempx * x_18_2_3;
                                            QUICKDouble x_19_4_2 = Qtempx * x_19_2_2 + WQtempx * x_19_2_3;
                                            QUICKDouble x_10_5_2 = Qtempy * x_10_3_2 + WQtempy * x_10_3_3 + ABCDtemp * x_6_3_3;
                                            QUICKDouble x_11_5_2 = Qtempy * x_11_3_2 + WQtempy * x_11_3_3 + ABCDtemp * x_7_3_3;
                                            QUICKDouble x_12_5_2 = Qtempy * x_12_3_2 + WQtempy * x_12_3_3 +  2 * ABCDtemp * x_4_3_3;
                                            QUICKDouble x_13_5_2 = Qtempy * x_13_3_2 + WQtempy * x_13_3_3;
                                            QUICKDouble x_14_5_2 = Qtempy * x_14_3_2 + WQtempy * x_14_3_3;
                                            QUICKDouble x_15_5_2 = Qtempy * x_15_3_2 + WQtempy * x_15_3_3 +  2 * ABCDtemp * x_5_3_3;
                                            QUICKDouble x_16_5_2 = Qtempy * x_16_3_2 + WQtempy * x_16_3_3 + ABCDtemp * x_9_3_3;
                                            QUICKDouble x_17_5_2 = Qtempy * x_17_3_2 + WQtempy * x_17_3_3;
                                            QUICKDouble x_18_5_2 = Qtempy * x_18_3_2 + WQtempy * x_18_3_3 +  3 * ABCDtemp * x_8_3_3;
                                            QUICKDouble x_19_5_2 = Qtempy * x_19_3_2 + WQtempy * x_19_3_3;
                                            QUICKDouble x_10_6_2 = Qtempx * x_10_3_2 + WQtempx * x_10_3_3 + ABCDtemp * x_5_3_3;
                                            QUICKDouble x_11_6_2 = Qtempx * x_11_3_2 + WQtempx * x_11_3_3 +  2 * ABCDtemp * x_4_3_3;
                                            QUICKDouble x_12_6_2 = Qtempx * x_12_3_2 + WQtempx * x_12_3_3 + ABCDtemp * x_8_3_3;
                                            QUICKDouble x_13_6_2 = Qtempx * x_13_3_2 + WQtempx * x_13_3_3 +  2 * ABCDtemp * x_6_3_3;
                                            QUICKDouble x_14_6_2 = Qtempx * x_14_3_2 + WQtempx * x_14_3_3 + ABCDtemp * x_9_3_3;
                                            QUICKDouble x_15_6_2 = Qtempx * x_15_3_2 + WQtempx * x_15_3_3;
                                            QUICKDouble x_16_6_2 = Qtempx * x_16_3_2 + WQtempx * x_16_3_3;
                                            QUICKDouble x_17_6_2 = Qtempx * x_17_3_2 + WQtempx * x_17_3_3 +  3 * ABCDtemp * x_7_3_3;
                                            QUICKDouble x_18_6_2 = Qtempx * x_18_3_2 + WQtempx * x_18_3_3;
                                            QUICKDouble x_19_6_2 = Qtempx * x_19_3_2 + WQtempx * x_19_3_3;
                                            QUICKDouble x_10_7_2 = Qtempx * x_10_1_2 + WQtempx * x_10_1_3 + CDtemp * ( x_10_0_2 - ABcom * x_10_0_3) + ABCDtemp * x_5_1_3;
                                            QUICKDouble x_11_7_2 = Qtempx * x_11_1_2 + WQtempx * x_11_1_3 + CDtemp * ( x_11_0_2 - ABcom * x_11_0_3) +  2 * ABCDtemp * x_4_1_3;
                                            QUICKDouble x_12_7_2 = Qtempx * x_12_1_2 + WQtempx * x_12_1_3 + CDtemp * ( x_12_0_2 - ABcom * x_12_0_3) + ABCDtemp * x_8_1_3;
                                            QUICKDouble x_13_7_2 = Qtempx * x_13_1_2 + WQtempx * x_13_1_3 + CDtemp * ( x_13_0_2 - ABcom * x_13_0_3) +  2 * ABCDtemp * x_6_1_3;
                                            QUICKDouble x_14_7_2 = Qtempx * x_14_1_2 + WQtempx * x_14_1_3 + CDtemp * ( x_14_0_2 - ABcom * x_14_0_3) + ABCDtemp * x_9_1_3;
                                            QUICKDouble x_15_7_2 = Qtempx * x_15_1_2 + WQtempx * x_15_1_3 + CDtemp * ( x_15_0_2 - ABcom * x_15_0_3);
                                            QUICKDouble x_16_7_2 = Qtempx * x_16_1_2 + WQtempx * x_16_1_3 + CDtemp * ( x_16_0_2 - ABcom * x_16_0_3);
                                            QUICKDouble x_17_7_2 = Qtempx * x_17_1_2 + WQtempx * x_17_1_3 + CDtemp * ( x_17_0_2 - ABcom * x_17_0_3) +  3 * ABCDtemp * x_7_1_3;
                                            QUICKDouble x_18_7_2 = Qtempx * x_18_1_2 + WQtempx * x_18_1_3 + CDtemp * ( x_18_0_2 - ABcom * x_18_0_3);
                                            QUICKDouble x_19_7_2 = Qtempx * x_19_1_2 + WQtempx * x_19_1_3 + CDtemp * ( x_19_0_2 - ABcom * x_19_0_3);
                                            QUICKDouble x_10_8_2 = Qtempy * x_10_2_2 + WQtempy * x_10_2_3 + CDtemp * ( x_10_0_2 - ABcom * x_10_0_3) + ABCDtemp * x_6_2_3;
                                            QUICKDouble x_11_8_2 = Qtempy * x_11_2_2 + WQtempy * x_11_2_3 + CDtemp * ( x_11_0_2 - ABcom * x_11_0_3) + ABCDtemp * x_7_2_3;
                                            QUICKDouble x_12_8_2 = Qtempy * x_12_2_2 + WQtempy * x_12_2_3 + CDtemp * ( x_12_0_2 - ABcom * x_12_0_3) +  2 * ABCDtemp * x_4_2_3;
                                            QUICKDouble x_13_8_2 = Qtempy * x_13_2_2 + WQtempy * x_13_2_3 + CDtemp * ( x_13_0_2 - ABcom * x_13_0_3);
                                            QUICKDouble x_14_8_2 = Qtempy * x_14_2_2 + WQtempy * x_14_2_3 + CDtemp * ( x_14_0_2 - ABcom * x_14_0_3);
                                            QUICKDouble x_15_8_2 = Qtempy * x_15_2_2 + WQtempy * x_15_2_3 + CDtemp * ( x_15_0_2 - ABcom * x_15_0_3) +  2 * ABCDtemp * x_5_2_3;
                                            QUICKDouble x_16_8_2 = Qtempy * x_16_2_2 + WQtempy * x_16_2_3 + CDtemp * ( x_16_0_2 - ABcom * x_16_0_3) + ABCDtemp * x_9_2_3;
                                            QUICKDouble x_17_8_2 = Qtempy * x_17_2_2 + WQtempy * x_17_2_3 + CDtemp * ( x_17_0_2 - ABcom * x_17_0_3);
                                            QUICKDouble x_18_8_2 = Qtempy * x_18_2_2 + WQtempy * x_18_2_3 + CDtemp * ( x_18_0_2 - ABcom * x_18_0_3) +  3 * ABCDtemp * x_8_2_3;
                                            QUICKDouble x_19_8_2 = Qtempy * x_19_2_2 + WQtempy * x_19_2_3 + CDtemp * ( x_19_0_2 - ABcom * x_19_0_3);
                                            QUICKDouble x_10_9_2 = Qtempz * x_10_3_2 + WQtempz * x_10_3_3 + CDtemp * ( x_10_0_2 - ABcom * x_10_0_3) + ABCDtemp * x_4_3_3;
                                            QUICKDouble x_11_9_2 = Qtempz * x_11_3_2 + WQtempz * x_11_3_3 + CDtemp * ( x_11_0_2 - ABcom * x_11_0_3);
                                            QUICKDouble x_12_9_2 = Qtempz * x_12_3_2 + WQtempz * x_12_3_3 + CDtemp * ( x_12_0_2 - ABcom * x_12_0_3);
                                            QUICKDouble x_13_9_2 = Qtempz * x_13_3_2 + WQtempz * x_13_3_3 + CDtemp * ( x_13_0_2 - ABcom * x_13_0_3) + ABCDtemp * x_7_3_3;
                                            QUICKDouble x_14_9_2 = Qtempz * x_14_3_2 + WQtempz * x_14_3_3 + CDtemp * ( x_14_0_2 - ABcom * x_14_0_3) +  2 * ABCDtemp * x_6_3_3;
                                            QUICKDouble x_15_9_2 = Qtempz * x_15_3_2 + WQtempz * x_15_3_3 + CDtemp * ( x_15_0_2 - ABcom * x_15_0_3) + ABCDtemp * x_8_3_3;
                                            QUICKDouble x_16_9_2 = Qtempz * x_16_3_2 + WQtempz * x_16_3_3 + CDtemp * ( x_16_0_2 - ABcom * x_16_0_3) +  2 * ABCDtemp * x_5_3_3;
                                            QUICKDouble x_17_9_2 = Qtempz * x_17_3_2 + WQtempz * x_17_3_3 + CDtemp * ( x_17_0_2 - ABcom * x_17_0_3);
                                            QUICKDouble x_18_9_2 = Qtempz * x_18_3_2 + WQtempz * x_18_3_3 + CDtemp * ( x_18_0_2 - ABcom * x_18_0_3);
                                            QUICKDouble x_19_9_2 = Qtempz * x_19_3_2 + WQtempz * x_19_3_3 + CDtemp * ( x_19_0_2 - ABcom * x_19_0_3) +  3 * ABCDtemp * x_9_3_3;
                                            //PSDS(3, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz, ABCDtemp);
                                            
                                            QUICKDouble x_1_4_3 = Ptempx * x_0_4_3 + WPtempx * x_0_4_4 + ABCDtemp * x_0_2_4;
                                            QUICKDouble x_2_4_3 = Ptempy * x_0_4_3 + WPtempy * x_0_4_4 + ABCDtemp * x_0_1_4;
                                            QUICKDouble x_3_4_3 = Ptempz * x_0_4_3 + WPtempz * x_0_4_4;
                                            
                                            QUICKDouble x_1_5_3 = Ptempx * x_0_5_3 + WPtempx * x_0_5_4;
                                            QUICKDouble x_2_5_3 = Ptempy * x_0_5_3 + WPtempy * x_0_5_4 + ABCDtemp * x_0_3_4;
                                            QUICKDouble x_3_5_3 = Ptempz * x_0_5_3 + WPtempz * x_0_5_4 + ABCDtemp * x_0_2_4;
                                            
                                            QUICKDouble x_1_6_3 = Ptempx * x_0_6_3 + WPtempx * x_0_6_4 + ABCDtemp * x_0_3_4;
                                            QUICKDouble x_2_6_3 = Ptempy * x_0_6_3 + WPtempy * x_0_6_4;
                                            QUICKDouble x_3_6_3 = Ptempz * x_0_6_3 + WPtempz * x_0_6_4 + ABCDtemp * x_0_1_4;
                                            
                                            QUICKDouble x_1_7_3 = Ptempx * x_0_7_3 + WPtempx * x_0_7_4 + ABCDtemp * x_0_1_4 * 2;
                                            QUICKDouble x_2_7_3 = Ptempy * x_0_7_3 + WPtempy * x_0_7_4;
                                            QUICKDouble x_3_7_3 = Ptempz * x_0_7_3 + WPtempz * x_0_7_4;
                                            
                                            QUICKDouble x_1_8_3 = Ptempx * x_0_8_3 + WPtempx * x_0_8_4;
                                            QUICKDouble x_2_8_3 = Ptempy * x_0_8_3 + WPtempy * x_0_8_4 + ABCDtemp * x_0_2_4 * 2;
                                            QUICKDouble x_3_8_3 = Ptempz * x_0_8_3 + WPtempz * x_0_8_4;
                                            
                                            QUICKDouble x_1_9_3 = Ptempx * x_0_9_3 + WPtempx * x_0_9_4;
                                            QUICKDouble x_2_9_3 = Ptempy * x_0_9_3 + WPtempy * x_0_9_4;
                                            QUICKDouble x_3_9_3 = Ptempz * x_0_9_3 + WPtempz * x_0_9_4 + ABCDtemp * x_0_3_4 * 2;    
                                            
                                            //PSPS(3, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz, ABCDtemp);
                                            QUICKDouble x_1_1_3 = Ptempx * x_0_1_3 + WPtempx * x_0_1_4 + ABCDtemp * VY( 0, 0, 4);
                                            QUICKDouble x_2_1_3 = Ptempy * x_0_1_3 + WPtempy * x_0_1_4;
                                            QUICKDouble x_3_1_3 = Ptempz * x_0_1_3 + WPtempz * x_0_1_4;
                                            
                                            QUICKDouble x_1_2_3 = Ptempx * x_0_2_3 + WPtempx * x_0_2_4;
                                            QUICKDouble x_2_2_3 = Ptempy * x_0_2_3 + WPtempy * x_0_2_4 + ABCDtemp * VY( 0, 0, 4);
                                            QUICKDouble x_3_2_3 = Ptempz * x_0_2_3 + WPtempz * x_0_2_4;
                                            
                                            QUICKDouble x_1_3_3 = Ptempx * x_0_3_3 + WPtempx * x_0_3_4;
                                            QUICKDouble x_2_3_3 = Ptempy * x_0_3_3 + WPtempy * x_0_3_4;
                                            QUICKDouble x_3_3_3 = Ptempz * x_0_3_3 + WPtempz * x_0_3_4 + ABCDtemp * VY( 0, 0, 4);
                                            
                                            //DSDS(2, YVerticalTemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz, ABCDtemp, ABtemp, CDcom);
                                            QUICKDouble x_4_4_2 = Ptempx * x_2_4_2 + WPtempx * x_2_4_3 + ABCDtemp * x_2_2_3;
                                            QUICKDouble x_4_5_2 = Ptempx * x_2_5_2 + WPtempx * x_2_5_3;
                                            QUICKDouble x_4_6_2 = Ptempx * x_2_6_2 + WPtempx * x_2_6_3 + ABCDtemp * x_2_3_3;
                                            QUICKDouble x_4_7_2 = Ptempx * x_2_7_2 + WPtempx * x_2_7_3 + 2 * ABCDtemp * x_2_1_3;
                                            QUICKDouble x_4_8_2 = Ptempx * x_2_8_2 + WPtempx * x_2_8_3;
                                            QUICKDouble x_4_9_2 = Ptempx * x_2_9_2 + WPtempx * x_2_9_3;
                                            
                                            QUICKDouble x_5_4_2 = Ptempy * x_3_4_2 + WPtempy * x_3_4_3 + ABCDtemp * x_3_1_3;
                                            QUICKDouble x_5_5_2 = Ptempy * x_3_5_2 + WPtempy * x_3_5_3 + ABCDtemp * x_3_3_3;
                                            QUICKDouble x_5_6_2 = Ptempy * x_3_6_2 + WPtempy * x_3_6_3;
                                            QUICKDouble x_5_7_2 = Ptempy * x_3_7_2 + WPtempy * x_3_7_3;
                                            QUICKDouble x_5_8_2 = Ptempy * x_3_8_2 + WPtempy * x_3_8_3 + 2 * ABCDtemp * x_3_2_3;
                                            QUICKDouble x_5_9_2 = Ptempy * x_3_9_2 + WPtempy * x_3_9_3;
                                            
                                            QUICKDouble x_6_4_2 = Ptempx * x_3_4_2 + WPtempx * x_3_4_3 + ABCDtemp * x_3_2_3;
                                            QUICKDouble x_6_5_2 = Ptempx * x_3_5_2 + WPtempx * x_3_5_3;
                                            QUICKDouble x_6_6_2 = Ptempx * x_3_6_2 + WPtempx * x_3_6_3 + ABCDtemp * x_3_3_3;
                                            QUICKDouble x_6_7_2 = Ptempx * x_3_7_2 + WPtempx * x_3_7_3 + 2 * ABCDtemp * x_3_1_3;
                                            QUICKDouble x_6_8_2 = Ptempx * x_3_8_2 + WPtempx * x_3_8_3;
                                            QUICKDouble x_6_9_2 = Ptempx * x_3_9_2 + WPtempx * x_3_9_3;
                                            
                                            QUICKDouble x_7_4_2 = Ptempx * x_1_4_2 + WPtempx * x_1_4_3 +  ABtemp * (x_0_4_2 - CDcom * x_0_4_3) + ABCDtemp * x_1_2_3;
                                            QUICKDouble x_7_5_2 = Ptempx * x_1_5_2 + WPtempx * x_1_5_3 +  ABtemp * (x_0_5_2 - CDcom * x_0_5_3);
                                            QUICKDouble x_7_6_2 = Ptempx * x_1_6_2 + WPtempx * x_1_6_3 +  ABtemp * (x_0_6_2 - CDcom * x_0_6_3) + ABCDtemp * x_1_3_3;
                                            QUICKDouble x_7_7_2 = Ptempx * x_1_7_2 + WPtempx * x_1_7_3 +  ABtemp * (x_0_7_2 - CDcom * x_0_7_3) + 2 * ABCDtemp * x_1_1_3;
                                            QUICKDouble x_7_8_2 = Ptempx * x_1_8_2 + WPtempx * x_1_8_3 +  ABtemp * (x_0_8_2 - CDcom * x_0_8_3);
                                            QUICKDouble x_7_9_2 = Ptempx * x_1_9_2 + WPtempx * x_1_9_3 +  ABtemp * (x_0_9_2 - CDcom * x_0_9_3);
                                            
                                            
                                            QUICKDouble x_8_4_2 = Ptempy * x_2_4_2 + WPtempy * x_2_4_3 +  ABtemp * (x_0_4_2 - CDcom * x_0_4_3) + ABCDtemp * x_2_1_3;
                                            QUICKDouble x_8_5_2 = Ptempy * x_2_5_2 + WPtempy * x_2_5_3 +  ABtemp * (x_0_5_2 - CDcom * x_0_5_3) + ABCDtemp * x_2_3_3;
                                            QUICKDouble x_8_6_2 = Ptempy * x_2_6_2 + WPtempy * x_2_6_3 +  ABtemp * (x_0_6_2 - CDcom * x_0_6_3);
                                            QUICKDouble x_8_7_2 = Ptempy * x_2_7_2 + WPtempy * x_2_7_3 +  ABtemp * (x_0_7_2 - CDcom * x_0_7_3);
                                            QUICKDouble x_8_8_2 = Ptempy * x_2_8_2 + WPtempy * x_2_8_3 +  ABtemp * (x_0_8_2 - CDcom * x_0_8_3) + 2 * ABCDtemp * x_2_2_3;
                                            QUICKDouble x_8_9_2 = Ptempy * x_2_9_2 + WPtempy * x_2_9_3 +  ABtemp * (x_0_9_2 - CDcom * x_0_9_3);
                                            
                                            QUICKDouble x_9_4_2 = Ptempz * x_3_4_2 + WPtempz * x_3_4_3 +  ABtemp * (x_0_4_2 - CDcom * x_0_4_3);
                                            QUICKDouble x_9_5_2 = Ptempz * x_3_5_2 + WPtempz * x_3_5_3 +  ABtemp * (x_0_5_2 - CDcom * x_0_5_3) + ABCDtemp * x_3_2_3;
                                            QUICKDouble x_9_6_2 = Ptempz * x_3_6_2 + WPtempz * x_3_6_3 +  ABtemp * (x_0_6_2 - CDcom * x_0_6_3) + ABCDtemp * x_3_1_3;
                                            QUICKDouble x_9_7_2 = Ptempz * x_3_7_2 + WPtempz * x_3_7_3 +  ABtemp * (x_0_7_2 - CDcom * x_0_7_3);
                                            QUICKDouble x_9_8_2 = Ptempz * x_3_8_2 + WPtempz * x_3_8_3 +  ABtemp * (x_0_8_2 - CDcom * x_0_8_3);
                                            QUICKDouble x_9_9_2 = Ptempz * x_3_9_2 + WPtempz * x_3_9_3 +  ABtemp * (x_0_9_2 - CDcom * x_0_9_3) + 2 * ABCDtemp * x_3_3_3;
                                            
                                            //FSFS(1, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz, ABCDtemp, CDtemp, ABcom);
                                            
                                            QUICKDouble x_10_10_1 = Qtempx * x_10_5_1 + WQtempx * x_10_5_2 + ABCDtemp * x_5_5_2;
                                            QUICKDouble x_11_10_1 = Qtempx * x_11_5_1 + WQtempx * x_11_5_2 +  2 * ABCDtemp * x_4_5_2;
                                            QUICKDouble x_12_10_1 = Qtempx * x_12_5_1 + WQtempx * x_12_5_2 + ABCDtemp * x_8_5_2;
                                            QUICKDouble x_13_10_1 = Qtempx * x_13_5_1 + WQtempx * x_13_5_2 +  2 * ABCDtemp * x_6_5_2;
                                            QUICKDouble x_14_10_1 = Qtempx * x_14_5_1 + WQtempx * x_14_5_2 + ABCDtemp * x_9_5_2;
                                            QUICKDouble x_15_10_1 = Qtempx * x_15_5_1 + WQtempx * x_15_5_2;
                                            QUICKDouble x_16_10_1 = Qtempx * x_16_5_1 + WQtempx * x_16_5_2;
                                            QUICKDouble x_17_10_1 = Qtempx * x_17_5_1 + WQtempx * x_17_5_2 +  3 * ABCDtemp * x_7_5_2;
                                            QUICKDouble x_18_10_1 = Qtempx * x_18_5_1 + WQtempx * x_18_5_2;
                                            QUICKDouble x_19_10_1 = Qtempx * x_19_5_1 + WQtempx * x_19_5_2;
                                            QUICKDouble x_10_11_1 = Qtempx * x_10_4_1 + WQtempx * x_10_4_2 + CDtemp * ( x_10_2_1 - ABcom * x_10_2_2) + ABCDtemp * x_5_4_2;
                                            QUICKDouble x_11_11_1 = Qtempx * x_11_4_1 + WQtempx * x_11_4_2 + CDtemp * ( x_11_2_1 - ABcom * x_11_2_2) +  2 * ABCDtemp * x_4_4_2;
                                            QUICKDouble x_12_11_1 = Qtempx * x_12_4_1 + WQtempx * x_12_4_2 + CDtemp * ( x_12_2_1 - ABcom * x_12_2_2) + ABCDtemp * x_8_4_2;
                                            QUICKDouble x_13_11_1 = Qtempx * x_13_4_1 + WQtempx * x_13_4_2 + CDtemp * ( x_13_2_1 - ABcom * x_13_2_2) +  2 * ABCDtemp * x_6_4_2;
                                            QUICKDouble x_14_11_1 = Qtempx * x_14_4_1 + WQtempx * x_14_4_2 + CDtemp * ( x_14_2_1 - ABcom * x_14_2_2) + ABCDtemp * x_9_4_2;
                                            QUICKDouble x_15_11_1 = Qtempx * x_15_4_1 + WQtempx * x_15_4_2 + CDtemp * ( x_15_2_1 - ABcom * x_15_2_2);
                                            QUICKDouble x_16_11_1 = Qtempx * x_16_4_1 + WQtempx * x_16_4_2 + CDtemp * ( x_16_2_1 - ABcom * x_16_2_2);
                                            QUICKDouble x_17_11_1 = Qtempx * x_17_4_1 + WQtempx * x_17_4_2 + CDtemp * ( x_17_2_1 - ABcom * x_17_2_2) +  3 * ABCDtemp * x_7_4_2;
                                            QUICKDouble x_18_11_1 = Qtempx * x_18_4_1 + WQtempx * x_18_4_2 + CDtemp * ( x_18_2_1 - ABcom * x_18_2_2);
                                            QUICKDouble x_19_11_1 = Qtempx * x_19_4_1 + WQtempx * x_19_4_2 + CDtemp * ( x_19_2_1 - ABcom * x_19_2_2);
                                            QUICKDouble x_10_12_1 = Qtempx * x_10_8_1 + WQtempx * x_10_8_2 + ABCDtemp * x_5_8_2;
                                            QUICKDouble x_11_12_1 = Qtempx * x_11_8_1 + WQtempx * x_11_8_2 +  2 * ABCDtemp * x_4_8_2;
                                            QUICKDouble x_12_12_1 = Qtempx * x_12_8_1 + WQtempx * x_12_8_2 + ABCDtemp * x_8_8_2;
                                            QUICKDouble x_13_12_1 = Qtempx * x_13_8_1 + WQtempx * x_13_8_2 +  2 * ABCDtemp * x_6_8_2;
                                            QUICKDouble x_14_12_1 = Qtempx * x_14_8_1 + WQtempx * x_14_8_2 + ABCDtemp * x_9_8_2;
                                            QUICKDouble x_15_12_1 = Qtempx * x_15_8_1 + WQtempx * x_15_8_2;
                                            QUICKDouble x_16_12_1 = Qtempx * x_16_8_1 + WQtempx * x_16_8_2;
                                            QUICKDouble x_17_12_1 = Qtempx * x_17_8_1 + WQtempx * x_17_8_2 +  3 * ABCDtemp * x_7_8_2;
                                            QUICKDouble x_18_12_1 = Qtempx * x_18_8_1 + WQtempx * x_18_8_2;
                                            QUICKDouble x_19_12_1 = Qtempx * x_19_8_1 + WQtempx * x_19_8_2;
                                            QUICKDouble x_10_13_1 = Qtempx * x_10_6_1 + WQtempx * x_10_6_2 + CDtemp * ( x_10_3_1 - ABcom * x_10_3_2) + ABCDtemp * x_5_6_2;
                                            QUICKDouble x_11_13_1 = Qtempx * x_11_6_1 + WQtempx * x_11_6_2 + CDtemp * ( x_11_3_1 - ABcom * x_11_3_2) +  2 * ABCDtemp * x_4_6_2;
                                            QUICKDouble x_12_13_1 = Qtempx * x_12_6_1 + WQtempx * x_12_6_2 + CDtemp * ( x_12_3_1 - ABcom * x_12_3_2) + ABCDtemp * x_8_6_2;
                                            QUICKDouble x_13_13_1 = Qtempx * x_13_6_1 + WQtempx * x_13_6_2 + CDtemp * ( x_13_3_1 - ABcom * x_13_3_2) +  2 * ABCDtemp * x_6_6_2;
                                            QUICKDouble x_14_13_1 = Qtempx * x_14_6_1 + WQtempx * x_14_6_2 + CDtemp * ( x_14_3_1 - ABcom * x_14_3_2) + ABCDtemp * x_9_6_2;
                                            QUICKDouble x_15_13_1 = Qtempx * x_15_6_1 + WQtempx * x_15_6_2 + CDtemp * ( x_15_3_1 - ABcom * x_15_3_2);
                                            QUICKDouble x_16_13_1 = Qtempx * x_16_6_1 + WQtempx * x_16_6_2 + CDtemp * ( x_16_3_1 - ABcom * x_16_3_2);
                                            QUICKDouble x_17_13_1 = Qtempx * x_17_6_1 + WQtempx * x_17_6_2 + CDtemp * ( x_17_3_1 - ABcom * x_17_3_2) +  3 * ABCDtemp * x_7_6_2;
                                            QUICKDouble x_18_13_1 = Qtempx * x_18_6_1 + WQtempx * x_18_6_2 + CDtemp * ( x_18_3_1 - ABcom * x_18_3_2);
                                            QUICKDouble x_19_13_1 = Qtempx * x_19_6_1 + WQtempx * x_19_6_2 + CDtemp * ( x_19_3_1 - ABcom * x_19_3_2);
                                            QUICKDouble x_10_14_1 = Qtempx * x_10_9_1 + WQtempx * x_10_9_2 + ABCDtemp * x_5_9_2;
                                            QUICKDouble x_11_14_1 = Qtempx * x_11_9_1 + WQtempx * x_11_9_2 +  2 * ABCDtemp * x_4_9_2;
                                            QUICKDouble x_12_14_1 = Qtempx * x_12_9_1 + WQtempx * x_12_9_2 + ABCDtemp * x_8_9_2;
                                            QUICKDouble x_13_14_1 = Qtempx * x_13_9_1 + WQtempx * x_13_9_2 +  2 * ABCDtemp * x_6_9_2;
                                            QUICKDouble x_14_14_1 = Qtempx * x_14_9_1 + WQtempx * x_14_9_2 + ABCDtemp * x_9_9_2;
                                            QUICKDouble x_15_14_1 = Qtempx * x_15_9_1 + WQtempx * x_15_9_2;
                                            QUICKDouble x_16_14_1 = Qtempx * x_16_9_1 + WQtempx * x_16_9_2;
                                            QUICKDouble x_17_14_1 = Qtempx * x_17_9_1 + WQtempx * x_17_9_2 +  3 * ABCDtemp * x_7_9_2;
                                            QUICKDouble x_18_14_1 = Qtempx * x_18_9_1 + WQtempx * x_18_9_2;
                                            QUICKDouble x_19_14_1 = Qtempx * x_19_9_1 + WQtempx * x_19_9_2;
                                            QUICKDouble x_10_15_1 = Qtempy * x_10_5_1 + WQtempy * x_10_5_2 + CDtemp * ( x_10_3_1 - ABcom * x_10_3_2) + ABCDtemp * x_6_5_2;
                                            QUICKDouble x_11_15_1 = Qtempy * x_11_5_1 + WQtempy * x_11_5_2 + CDtemp * ( x_11_3_1 - ABcom * x_11_3_2) + ABCDtemp * x_7_5_2;
                                            QUICKDouble x_12_15_1 = Qtempy * x_12_5_1 + WQtempy * x_12_5_2 + CDtemp * ( x_12_3_1 - ABcom * x_12_3_2) +  2 * ABCDtemp * x_4_5_2;
                                            QUICKDouble x_13_15_1 = Qtempy * x_13_5_1 + WQtempy * x_13_5_2 + CDtemp * ( x_13_3_1 - ABcom * x_13_3_2);
                                            QUICKDouble x_14_15_1 = Qtempy * x_14_5_1 + WQtempy * x_14_5_2 + CDtemp * ( x_14_3_1 - ABcom * x_14_3_2);
                                            QUICKDouble x_15_15_1 = Qtempy * x_15_5_1 + WQtempy * x_15_5_2 + CDtemp * ( x_15_3_1 - ABcom * x_15_3_2) +  2 * ABCDtemp * x_5_5_2;
                                            QUICKDouble x_16_15_1 = Qtempy * x_16_5_1 + WQtempy * x_16_5_2 + CDtemp * ( x_16_3_1 - ABcom * x_16_3_2) + ABCDtemp * x_9_5_2;
                                            QUICKDouble x_17_15_1 = Qtempy * x_17_5_1 + WQtempy * x_17_5_2 + CDtemp * ( x_17_3_1 - ABcom * x_17_3_2);
                                            QUICKDouble x_18_15_1 = Qtempy * x_18_5_1 + WQtempy * x_18_5_2 + CDtemp * ( x_18_3_1 - ABcom * x_18_3_2) +  3 * ABCDtemp * x_8_5_2;
                                            QUICKDouble x_19_15_1 = Qtempy * x_19_5_1 + WQtempy * x_19_5_2 + CDtemp * ( x_19_3_1 - ABcom * x_19_3_2);
                                            QUICKDouble x_10_16_1 = Qtempy * x_10_9_1 + WQtempy * x_10_9_2 + ABCDtemp * x_6_9_2;
                                            QUICKDouble x_11_16_1 = Qtempy * x_11_9_1 + WQtempy * x_11_9_2 + ABCDtemp * x_7_9_2;
                                            QUICKDouble x_12_16_1 = Qtempy * x_12_9_1 + WQtempy * x_12_9_2 +  2 * ABCDtemp * x_4_9_2;
                                            QUICKDouble x_13_16_1 = Qtempy * x_13_9_1 + WQtempy * x_13_9_2;
                                            QUICKDouble x_14_16_1 = Qtempy * x_14_9_1 + WQtempy * x_14_9_2;
                                            QUICKDouble x_15_16_1 = Qtempy * x_15_9_1 + WQtempy * x_15_9_2 +  2 * ABCDtemp * x_5_9_2;
                                            QUICKDouble x_16_16_1 = Qtempy * x_16_9_1 + WQtempy * x_16_9_2 + ABCDtemp * x_9_9_2;
                                            QUICKDouble x_17_16_1 = Qtempy * x_17_9_1 + WQtempy * x_17_9_2;
                                            QUICKDouble x_18_16_1 = Qtempy * x_18_9_1 + WQtempy * x_18_9_2 +  3 * ABCDtemp * x_8_9_2;
                                            QUICKDouble x_19_16_1 = Qtempy * x_19_9_1 + WQtempy * x_19_9_2;
                                            QUICKDouble x_10_17_1 = Qtempx * x_10_7_1 + WQtempx * x_10_7_2 +  2 * CDtemp * ( x_10_1_1 - ABcom * x_10_1_2) + ABCDtemp * x_5_7_2;
                                            QUICKDouble x_11_17_1 = Qtempx * x_11_7_1 + WQtempx * x_11_7_2 +  2 * CDtemp * ( x_11_1_1 - ABcom * x_11_1_2) +  2 * ABCDtemp * x_4_7_2;
                                            QUICKDouble x_12_17_1 = Qtempx * x_12_7_1 + WQtempx * x_12_7_2 +  2 * CDtemp * ( x_12_1_1 - ABcom * x_12_1_2) + ABCDtemp * x_8_7_2;
                                            QUICKDouble x_13_17_1 = Qtempx * x_13_7_1 + WQtempx * x_13_7_2 +  2 * CDtemp * ( x_13_1_1 - ABcom * x_13_1_2) +  2 * ABCDtemp * x_6_7_2;
                                            QUICKDouble x_14_17_1 = Qtempx * x_14_7_1 + WQtempx * x_14_7_2 +  2 * CDtemp * ( x_14_1_1 - ABcom * x_14_1_2) + ABCDtemp * x_9_7_2;
                                            QUICKDouble x_15_17_1 = Qtempx * x_15_7_1 + WQtempx * x_15_7_2 +  2 * CDtemp * ( x_15_1_1 - ABcom * x_15_1_2);
                                            QUICKDouble x_16_17_1 = Qtempx * x_16_7_1 + WQtempx * x_16_7_2 +  2 * CDtemp * ( x_16_1_1 - ABcom * x_16_1_2);
                                            QUICKDouble x_17_17_1 = Qtempx * x_17_7_1 + WQtempx * x_17_7_2 +  2 * CDtemp * ( x_17_1_1 - ABcom * x_17_1_2) +  3 * ABCDtemp * x_7_7_2;
                                            QUICKDouble x_18_17_1 = Qtempx * x_18_7_1 + WQtempx * x_18_7_2 +  2 * CDtemp * ( x_18_1_1 - ABcom * x_18_1_2);
                                            QUICKDouble x_19_17_1 = Qtempx * x_19_7_1 + WQtempx * x_19_7_2 +  2 * CDtemp * ( x_19_1_1 - ABcom * x_19_1_2);
                                            QUICKDouble x_10_18_1 = Qtempy * x_10_8_1 + WQtempy * x_10_8_2 +  2 * CDtemp * ( x_10_2_1 - ABcom * x_10_2_2) + ABCDtemp * x_6_8_2;
                                            QUICKDouble x_11_18_1 = Qtempy * x_11_8_1 + WQtempy * x_11_8_2 +  2 * CDtemp * ( x_11_2_1 - ABcom * x_11_2_2) + ABCDtemp * x_7_8_2;
                                            QUICKDouble x_12_18_1 = Qtempy * x_12_8_1 + WQtempy * x_12_8_2 +  2 * CDtemp * ( x_12_2_1 - ABcom * x_12_2_2) +  2 * ABCDtemp * x_4_8_2;
                                            QUICKDouble x_13_18_1 = Qtempy * x_13_8_1 + WQtempy * x_13_8_2 +  2 * CDtemp * ( x_13_2_1 - ABcom * x_13_2_2);
                                            QUICKDouble x_14_18_1 = Qtempy * x_14_8_1 + WQtempy * x_14_8_2 +  2 * CDtemp * ( x_14_2_1 - ABcom * x_14_2_2);
                                            QUICKDouble x_15_18_1 = Qtempy * x_15_8_1 + WQtempy * x_15_8_2 +  2 * CDtemp * ( x_15_2_1 - ABcom * x_15_2_2) +  2 * ABCDtemp * x_5_8_2;
                                            QUICKDouble x_16_18_1 = Qtempy * x_16_8_1 + WQtempy * x_16_8_2 +  2 * CDtemp * ( x_16_2_1 - ABcom * x_16_2_2) + ABCDtemp * x_9_8_2;
                                            QUICKDouble x_17_18_1 = Qtempy * x_17_8_1 + WQtempy * x_17_8_2 +  2 * CDtemp * ( x_17_2_1 - ABcom * x_17_2_2);
                                            QUICKDouble x_18_18_1 = Qtempy * x_18_8_1 + WQtempy * x_18_8_2 +  2 * CDtemp * ( x_18_2_1 - ABcom * x_18_2_2) +  3 * ABCDtemp * x_8_8_2;
                                            QUICKDouble x_19_18_1 = Qtempy * x_19_8_1 + WQtempy * x_19_8_2 +  2 * CDtemp * ( x_19_2_1 - ABcom * x_19_2_2);
                                            QUICKDouble x_10_19_1 = Qtempz * x_10_9_1 + WQtempz * x_10_9_2 +  2 * CDtemp * ( x_10_3_1 - ABcom * x_10_3_2) + ABCDtemp * x_4_9_2;
                                            QUICKDouble x_11_19_1 = Qtempz * x_11_9_1 + WQtempz * x_11_9_2 +  2 * CDtemp * ( x_11_3_1 - ABcom * x_11_3_2);
                                            QUICKDouble x_12_19_1 = Qtempz * x_12_9_1 + WQtempz * x_12_9_2 +  2 * CDtemp * ( x_12_3_1 - ABcom * x_12_3_2);
                                            QUICKDouble x_13_19_1 = Qtempz * x_13_9_1 + WQtempz * x_13_9_2 +  2 * CDtemp * ( x_13_3_1 - ABcom * x_13_3_2) + ABCDtemp * x_7_9_2;
                                            QUICKDouble x_14_19_1 = Qtempz * x_14_9_1 + WQtempz * x_14_9_2 +  2 * CDtemp * ( x_14_3_1 - ABcom * x_14_3_2) +  2 * ABCDtemp * x_6_9_2;
                                            QUICKDouble x_15_19_1 = Qtempz * x_15_9_1 + WQtempz * x_15_9_2 +  2 * CDtemp * ( x_15_3_1 - ABcom * x_15_3_2) + ABCDtemp * x_8_9_2;
                                            QUICKDouble x_16_19_1 = Qtempz * x_16_9_1 + WQtempz * x_16_9_2 +  2 * CDtemp * ( x_16_3_1 - ABcom * x_16_3_2) +  2 * ABCDtemp * x_5_9_2;
                                            QUICKDouble x_17_19_1 = Qtempz * x_17_9_1 + WQtempz * x_17_9_2 +  2 * CDtemp * ( x_17_3_1 - ABcom * x_17_3_2);
                                            QUICKDouble x_18_19_1 = Qtempz * x_18_9_1 + WQtempz * x_18_9_2 +  2 * CDtemp * ( x_18_3_1 - ABcom * x_18_3_2);
                                            QUICKDouble x_19_19_1 = Qtempz * x_19_9_1 + WQtempz * x_19_9_2 +  2 * CDtemp * ( x_19_3_1 - ABcom * x_19_3_2) +  3 * ABCDtemp * x_9_9_2;
                                            
                                            
                                            //GSFS(1, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz, ABCDtemp, CDtemp, ABcom);
                                            
                                            QUICKDouble x_20_10_1 = Qtempx * x_20_5_1 + WQtempx * x_20_5_2 +  2 * ABCDtemp * x_12_5_2;
                                            QUICKDouble x_21_10_1 = Qtempx * x_21_5_1 + WQtempx * x_21_5_2 +  2 * ABCDtemp * x_14_5_2;
                                            QUICKDouble x_22_10_1 = Qtempx * x_22_5_1 + WQtempx * x_22_5_2;
                                            QUICKDouble x_23_10_1 = Qtempx * x_23_5_1 + WQtempx * x_23_5_2 +  2 * ABCDtemp * x_10_5_2;
                                            QUICKDouble x_24_10_1 = Qtempx * x_24_5_1 + WQtempx * x_24_5_2 + ABCDtemp * x_15_5_2;
                                            QUICKDouble x_25_10_1 = Qtempx * x_25_5_1 + WQtempx * x_25_5_2 + ABCDtemp * x_16_5_2;
                                            QUICKDouble x_26_10_1 = Qtempx * x_26_5_1 + WQtempx * x_26_5_2 +  3 * ABCDtemp * x_13_5_2;
                                            QUICKDouble x_27_10_1 = Qtempx * x_27_5_1 + WQtempx * x_27_5_2 + ABCDtemp * x_19_5_2;
                                            QUICKDouble x_28_10_1 = Qtempx * x_28_5_1 + WQtempx * x_28_5_2 +  3 * ABCDtemp * x_11_5_2;
                                            QUICKDouble x_29_10_1 = Qtempx * x_29_5_1 + WQtempx * x_29_5_2 + ABCDtemp * x_18_5_2;
                                            QUICKDouble x_30_10_1 = Qtempx * x_30_5_1 + WQtempx * x_30_5_2;
                                            QUICKDouble x_31_10_1 = Qtempx * x_31_5_1 + WQtempx * x_31_5_2;
                                            QUICKDouble x_32_10_1 = Qtempx * x_32_5_1 + WQtempx * x_32_5_2 +  4 * ABCDtemp * x_17_5_2;
                                            QUICKDouble x_33_10_1 = Qtempx * x_33_5_1 + WQtempx * x_33_5_2;
                                            QUICKDouble x_34_10_1 = Qtempx * x_34_5_1 + WQtempx * x_34_5_2;
                                            QUICKDouble x_20_11_1 = Qtempx * x_20_4_1 + WQtempx * x_20_4_2 + CDtemp * ( x_20_2_1 - ABcom * x_20_2_2) +  2 * ABCDtemp * x_12_4_2;
                                            QUICKDouble x_21_11_1 = Qtempx * x_21_4_1 + WQtempx * x_21_4_2 + CDtemp * ( x_21_2_1 - ABcom * x_21_2_2) +  2 * ABCDtemp * x_14_4_2;
                                            QUICKDouble x_22_11_1 = Qtempx * x_22_4_1 + WQtempx * x_22_4_2 + CDtemp * ( x_22_2_1 - ABcom * x_22_2_2);
                                            QUICKDouble x_23_11_1 = Qtempx * x_23_4_1 + WQtempx * x_23_4_2 + CDtemp * ( x_23_2_1 - ABcom * x_23_2_2) +  2 * ABCDtemp * x_10_4_2;
                                            QUICKDouble x_24_11_1 = Qtempx * x_24_4_1 + WQtempx * x_24_4_2 + CDtemp * ( x_24_2_1 - ABcom * x_24_2_2) + ABCDtemp * x_15_4_2;
                                            QUICKDouble x_25_11_1 = Qtempx * x_25_4_1 + WQtempx * x_25_4_2 + CDtemp * ( x_25_2_1 - ABcom * x_25_2_2) + ABCDtemp * x_16_4_2;
                                            QUICKDouble x_26_11_1 = Qtempx * x_26_4_1 + WQtempx * x_26_4_2 + CDtemp * ( x_26_2_1 - ABcom * x_26_2_2) +  3 * ABCDtemp * x_13_4_2;
                                            QUICKDouble x_27_11_1 = Qtempx * x_27_4_1 + WQtempx * x_27_4_2 + CDtemp * ( x_27_2_1 - ABcom * x_27_2_2) + ABCDtemp * x_19_4_2;
                                            QUICKDouble x_28_11_1 = Qtempx * x_28_4_1 + WQtempx * x_28_4_2 + CDtemp * ( x_28_2_1 - ABcom * x_28_2_2) +  3 * ABCDtemp * x_11_4_2;
                                            QUICKDouble x_29_11_1 = Qtempx * x_29_4_1 + WQtempx * x_29_4_2 + CDtemp * ( x_29_2_1 - ABcom * x_29_2_2) + ABCDtemp * x_18_4_2;
                                            QUICKDouble x_30_11_1 = Qtempx * x_30_4_1 + WQtempx * x_30_4_2 + CDtemp * ( x_30_2_1 - ABcom * x_30_2_2);
                                            QUICKDouble x_31_11_1 = Qtempx * x_31_4_1 + WQtempx * x_31_4_2 + CDtemp * ( x_31_2_1 - ABcom * x_31_2_2);
                                            QUICKDouble x_32_11_1 = Qtempx * x_32_4_1 + WQtempx * x_32_4_2 + CDtemp * ( x_32_2_1 - ABcom * x_32_2_2) +  4 * ABCDtemp * x_17_4_2;
                                            QUICKDouble x_33_11_1 = Qtempx * x_33_4_1 + WQtempx * x_33_4_2 + CDtemp * ( x_33_2_1 - ABcom * x_33_2_2);
                                            QUICKDouble x_34_11_1 = Qtempx * x_34_4_1 + WQtempx * x_34_4_2 + CDtemp * ( x_34_2_1 - ABcom * x_34_2_2);
                                            QUICKDouble x_20_12_1 = Qtempx * x_20_8_1 + WQtempx * x_20_8_2 +  2 * ABCDtemp * x_12_8_2;
                                            QUICKDouble x_21_12_1 = Qtempx * x_21_8_1 + WQtempx * x_21_8_2 +  2 * ABCDtemp * x_14_8_2;
                                            QUICKDouble x_22_12_1 = Qtempx * x_22_8_1 + WQtempx * x_22_8_2;
                                            QUICKDouble x_23_12_1 = Qtempx * x_23_8_1 + WQtempx * x_23_8_2 +  2 * ABCDtemp * x_10_8_2;
                                            QUICKDouble x_24_12_1 = Qtempx * x_24_8_1 + WQtempx * x_24_8_2 + ABCDtemp * x_15_8_2;
                                            QUICKDouble x_25_12_1 = Qtempx * x_25_8_1 + WQtempx * x_25_8_2 + ABCDtemp * x_16_8_2;
                                            QUICKDouble x_26_12_1 = Qtempx * x_26_8_1 + WQtempx * x_26_8_2 +  3 * ABCDtemp * x_13_8_2;
                                            QUICKDouble x_27_12_1 = Qtempx * x_27_8_1 + WQtempx * x_27_8_2 + ABCDtemp * x_19_8_2;
                                            QUICKDouble x_28_12_1 = Qtempx * x_28_8_1 + WQtempx * x_28_8_2 +  3 * ABCDtemp * x_11_8_2;
                                            QUICKDouble x_29_12_1 = Qtempx * x_29_8_1 + WQtempx * x_29_8_2 + ABCDtemp * x_18_8_2;
                                            QUICKDouble x_30_12_1 = Qtempx * x_30_8_1 + WQtempx * x_30_8_2;
                                            QUICKDouble x_31_12_1 = Qtempx * x_31_8_1 + WQtempx * x_31_8_2;
                                            QUICKDouble x_32_12_1 = Qtempx * x_32_8_1 + WQtempx * x_32_8_2 +  4 * ABCDtemp * x_17_8_2;
                                            QUICKDouble x_33_12_1 = Qtempx * x_33_8_1 + WQtempx * x_33_8_2;
                                            QUICKDouble x_34_12_1 = Qtempx * x_34_8_1 + WQtempx * x_34_8_2;
                                            QUICKDouble x_20_13_1 = Qtempx * x_20_6_1 + WQtempx * x_20_6_2 + CDtemp * ( x_20_3_1 - ABcom * x_20_3_2) +  2 * ABCDtemp * x_12_6_2;
                                            QUICKDouble x_21_13_1 = Qtempx * x_21_6_1 + WQtempx * x_21_6_2 + CDtemp * ( x_21_3_1 - ABcom * x_21_3_2) +  2 * ABCDtemp * x_14_6_2;
                                            QUICKDouble x_22_13_1 = Qtempx * x_22_6_1 + WQtempx * x_22_6_2 + CDtemp * ( x_22_3_1 - ABcom * x_22_3_2);
                                            QUICKDouble x_23_13_1 = Qtempx * x_23_6_1 + WQtempx * x_23_6_2 + CDtemp * ( x_23_3_1 - ABcom * x_23_3_2) +  2 * ABCDtemp * x_10_6_2;
                                            QUICKDouble x_24_13_1 = Qtempx * x_24_6_1 + WQtempx * x_24_6_2 + CDtemp * ( x_24_3_1 - ABcom * x_24_3_2) + ABCDtemp * x_15_6_2;
                                            QUICKDouble x_25_13_1 = Qtempx * x_25_6_1 + WQtempx * x_25_6_2 + CDtemp * ( x_25_3_1 - ABcom * x_25_3_2) + ABCDtemp * x_16_6_2;
                                            QUICKDouble x_26_13_1 = Qtempx * x_26_6_1 + WQtempx * x_26_6_2 + CDtemp * ( x_26_3_1 - ABcom * x_26_3_2) +  3 * ABCDtemp * x_13_6_2;
                                            QUICKDouble x_27_13_1 = Qtempx * x_27_6_1 + WQtempx * x_27_6_2 + CDtemp * ( x_27_3_1 - ABcom * x_27_3_2) + ABCDtemp * x_19_6_2;
                                            QUICKDouble x_28_13_1 = Qtempx * x_28_6_1 + WQtempx * x_28_6_2 + CDtemp * ( x_28_3_1 - ABcom * x_28_3_2) +  3 * ABCDtemp * x_11_6_2;
                                            QUICKDouble x_29_13_1 = Qtempx * x_29_6_1 + WQtempx * x_29_6_2 + CDtemp * ( x_29_3_1 - ABcom * x_29_3_2) + ABCDtemp * x_18_6_2;
                                            QUICKDouble x_30_13_1 = Qtempx * x_30_6_1 + WQtempx * x_30_6_2 + CDtemp * ( x_30_3_1 - ABcom * x_30_3_2);
                                            QUICKDouble x_31_13_1 = Qtempx * x_31_6_1 + WQtempx * x_31_6_2 + CDtemp * ( x_31_3_1 - ABcom * x_31_3_2);
                                            QUICKDouble x_32_13_1 = Qtempx * x_32_6_1 + WQtempx * x_32_6_2 + CDtemp * ( x_32_3_1 - ABcom * x_32_3_2) +  4 * ABCDtemp * x_17_6_2;
                                            QUICKDouble x_33_13_1 = Qtempx * x_33_6_1 + WQtempx * x_33_6_2 + CDtemp * ( x_33_3_1 - ABcom * x_33_3_2);
                                            QUICKDouble x_34_13_1 = Qtempx * x_34_6_1 + WQtempx * x_34_6_2 + CDtemp * ( x_34_3_1 - ABcom * x_34_3_2);
                                            QUICKDouble x_20_14_1 = Qtempx * x_20_9_1 + WQtempx * x_20_9_2 +  2 * ABCDtemp * x_12_9_2;
                                            QUICKDouble x_21_14_1 = Qtempx * x_21_9_1 + WQtempx * x_21_9_2 +  2 * ABCDtemp * x_14_9_2;
                                            QUICKDouble x_22_14_1 = Qtempx * x_22_9_1 + WQtempx * x_22_9_2;
                                            QUICKDouble x_23_14_1 = Qtempx * x_23_9_1 + WQtempx * x_23_9_2 +  2 * ABCDtemp * x_10_9_2;
                                            QUICKDouble x_24_14_1 = Qtempx * x_24_9_1 + WQtempx * x_24_9_2 + ABCDtemp * x_15_9_2;
                                            QUICKDouble x_25_14_1 = Qtempx * x_25_9_1 + WQtempx * x_25_9_2 + ABCDtemp * x_16_9_2;
                                            QUICKDouble x_26_14_1 = Qtempx * x_26_9_1 + WQtempx * x_26_9_2 +  3 * ABCDtemp * x_13_9_2;
                                            QUICKDouble x_27_14_1 = Qtempx * x_27_9_1 + WQtempx * x_27_9_2 + ABCDtemp * x_19_9_2;
                                            QUICKDouble x_28_14_1 = Qtempx * x_28_9_1 + WQtempx * x_28_9_2 +  3 * ABCDtemp * x_11_9_2;
                                            QUICKDouble x_29_14_1 = Qtempx * x_29_9_1 + WQtempx * x_29_9_2 + ABCDtemp * x_18_9_2;
                                            QUICKDouble x_30_14_1 = Qtempx * x_30_9_1 + WQtempx * x_30_9_2;
                                            QUICKDouble x_31_14_1 = Qtempx * x_31_9_1 + WQtempx * x_31_9_2;
                                            QUICKDouble x_32_14_1 = Qtempx * x_32_9_1 + WQtempx * x_32_9_2 +  4 * ABCDtemp * x_17_9_2;
                                            QUICKDouble x_33_14_1 = Qtempx * x_33_9_1 + WQtempx * x_33_9_2;
                                            QUICKDouble x_34_14_1 = Qtempx * x_34_9_1 + WQtempx * x_34_9_2;
                                            QUICKDouble x_20_15_1 = Qtempy * x_20_5_1 + WQtempy * x_20_5_2 + CDtemp * ( x_20_3_1 - ABcom * x_20_3_2) +  2 * ABCDtemp * x_11_5_2;
                                            QUICKDouble x_21_15_1 = Qtempy * x_21_5_1 + WQtempy * x_21_5_2 + CDtemp * ( x_21_3_1 - ABcom * x_21_3_2);
                                            QUICKDouble x_22_15_1 = Qtempy * x_22_5_1 + WQtempy * x_22_5_2 + CDtemp * ( x_22_3_1 - ABcom * x_22_3_2) +  2 * ABCDtemp * x_16_5_2;
                                            QUICKDouble x_23_15_1 = Qtempy * x_23_5_1 + WQtempy * x_23_5_2 + CDtemp * ( x_23_3_1 - ABcom * x_23_3_2) + ABCDtemp * x_13_5_2;
                                            QUICKDouble x_24_15_1 = Qtempy * x_24_5_1 + WQtempy * x_24_5_2 + CDtemp * ( x_24_3_1 - ABcom * x_24_3_2) +  2 * ABCDtemp * x_10_5_2;
                                            QUICKDouble x_25_15_1 = Qtempy * x_25_5_1 + WQtempy * x_25_5_2 + CDtemp * ( x_25_3_1 - ABcom * x_25_3_2) + ABCDtemp * x_14_5_2;
                                            QUICKDouble x_26_15_1 = Qtempy * x_26_5_1 + WQtempy * x_26_5_2 + CDtemp * ( x_26_3_1 - ABcom * x_26_3_2);
                                            QUICKDouble x_27_15_1 = Qtempy * x_27_5_1 + WQtempy * x_27_5_2 + CDtemp * ( x_27_3_1 - ABcom * x_27_3_2);
                                            QUICKDouble x_28_15_1 = Qtempy * x_28_5_1 + WQtempy * x_28_5_2 + CDtemp * ( x_28_3_1 - ABcom * x_28_3_2) + ABCDtemp * x_17_5_2;
                                            QUICKDouble x_29_15_1 = Qtempy * x_29_5_1 + WQtempy * x_29_5_2 + CDtemp * ( x_29_3_1 - ABcom * x_29_3_2) +  3 * ABCDtemp * x_12_5_2;
                                            QUICKDouble x_30_15_1 = Qtempy * x_30_5_1 + WQtempy * x_30_5_2 + CDtemp * ( x_30_3_1 - ABcom * x_30_3_2) +  3 * ABCDtemp * x_15_5_2;
                                            QUICKDouble x_31_15_1 = Qtempy * x_31_5_1 + WQtempy * x_31_5_2 + CDtemp * ( x_31_3_1 - ABcom * x_31_3_2) + ABCDtemp * x_19_5_2;
                                            QUICKDouble x_32_15_1 = Qtempy * x_32_5_1 + WQtempy * x_32_5_2 + CDtemp * ( x_32_3_1 - ABcom * x_32_3_2);
                                            QUICKDouble x_33_15_1 = Qtempy * x_33_5_1 + WQtempy * x_33_5_2 + CDtemp * ( x_33_3_1 - ABcom * x_33_3_2) +  4 * ABCDtemp * x_18_5_2;
                                            QUICKDouble x_34_15_1 = Qtempy * x_34_5_1 + WQtempy * x_34_5_2 + CDtemp * ( x_34_3_1 - ABcom * x_34_3_2);
                                            QUICKDouble x_20_16_1 = Qtempy * x_20_9_1 + WQtempy * x_20_9_2 +  2 * ABCDtemp * x_11_9_2;
                                            QUICKDouble x_21_16_1 = Qtempy * x_21_9_1 + WQtempy * x_21_9_2;
                                            QUICKDouble x_22_16_1 = Qtempy * x_22_9_1 + WQtempy * x_22_9_2 +  2 * ABCDtemp * x_16_9_2;
                                            QUICKDouble x_23_16_1 = Qtempy * x_23_9_1 + WQtempy * x_23_9_2 + ABCDtemp * x_13_9_2;
                                            QUICKDouble x_24_16_1 = Qtempy * x_24_9_1 + WQtempy * x_24_9_2 +  2 * ABCDtemp * x_10_9_2;
                                            QUICKDouble x_25_16_1 = Qtempy * x_25_9_1 + WQtempy * x_25_9_2 + ABCDtemp * x_14_9_2;
                                            QUICKDouble x_26_16_1 = Qtempy * x_26_9_1 + WQtempy * x_26_9_2;
                                            QUICKDouble x_27_16_1 = Qtempy * x_27_9_1 + WQtempy * x_27_9_2;
                                            QUICKDouble x_28_16_1 = Qtempy * x_28_9_1 + WQtempy * x_28_9_2 + ABCDtemp * x_17_9_2;
                                            QUICKDouble x_29_16_1 = Qtempy * x_29_9_1 + WQtempy * x_29_9_2 +  3 * ABCDtemp * x_12_9_2;
                                            QUICKDouble x_30_16_1 = Qtempy * x_30_9_1 + WQtempy * x_30_9_2 +  3 * ABCDtemp * x_15_9_2;
                                            QUICKDouble x_31_16_1 = Qtempy * x_31_9_1 + WQtempy * x_31_9_2 + ABCDtemp * x_19_9_2;
                                            QUICKDouble x_32_16_1 = Qtempy * x_32_9_1 + WQtempy * x_32_9_2;
                                            QUICKDouble x_33_16_1 = Qtempy * x_33_9_1 + WQtempy * x_33_9_2 +  4 * ABCDtemp * x_18_9_2;
                                            QUICKDouble x_34_16_1 = Qtempy * x_34_9_1 + WQtempy * x_34_9_2;
                                            QUICKDouble x_20_17_1 = Qtempx * x_20_7_1 + WQtempx * x_20_7_2 +  2 * CDtemp * ( x_20_1_1 - ABcom * x_20_1_2) +  2 * ABCDtemp * x_12_7_2;
                                            QUICKDouble x_21_17_1 = Qtempx * x_21_7_1 + WQtempx * x_21_7_2 +  2 * CDtemp * ( x_21_1_1 - ABcom * x_21_1_2) +  2 * ABCDtemp * x_14_7_2;
                                            QUICKDouble x_22_17_1 = Qtempx * x_22_7_1 + WQtempx * x_22_7_2 +  2 * CDtemp * ( x_22_1_1 - ABcom * x_22_1_2);
                                            QUICKDouble x_23_17_1 = Qtempx * x_23_7_1 + WQtempx * x_23_7_2 +  2 * CDtemp * ( x_23_1_1 - ABcom * x_23_1_2) +  2 * ABCDtemp * x_10_7_2;
                                            QUICKDouble x_24_17_1 = Qtempx * x_24_7_1 + WQtempx * x_24_7_2 +  2 * CDtemp * ( x_24_1_1 - ABcom * x_24_1_2) + ABCDtemp * x_15_7_2;
                                            QUICKDouble x_25_17_1 = Qtempx * x_25_7_1 + WQtempx * x_25_7_2 +  2 * CDtemp * ( x_25_1_1 - ABcom * x_25_1_2) + ABCDtemp * x_16_7_2;
                                            QUICKDouble x_26_17_1 = Qtempx * x_26_7_1 + WQtempx * x_26_7_2 +  2 * CDtemp * ( x_26_1_1 - ABcom * x_26_1_2) +  3 * ABCDtemp * x_13_7_2;
                                            QUICKDouble x_27_17_1 = Qtempx * x_27_7_1 + WQtempx * x_27_7_2 +  2 * CDtemp * ( x_27_1_1 - ABcom * x_27_1_2) + ABCDtemp * x_19_7_2;
                                            QUICKDouble x_28_17_1 = Qtempx * x_28_7_1 + WQtempx * x_28_7_2 +  2 * CDtemp * ( x_28_1_1 - ABcom * x_28_1_2) +  3 * ABCDtemp * x_11_7_2;
                                            QUICKDouble x_29_17_1 = Qtempx * x_29_7_1 + WQtempx * x_29_7_2 +  2 * CDtemp * ( x_29_1_1 - ABcom * x_29_1_2) + ABCDtemp * x_18_7_2;
                                            QUICKDouble x_30_17_1 = Qtempx * x_30_7_1 + WQtempx * x_30_7_2 +  2 * CDtemp * ( x_30_1_1 - ABcom * x_30_1_2);
                                            QUICKDouble x_31_17_1 = Qtempx * x_31_7_1 + WQtempx * x_31_7_2 +  2 * CDtemp * ( x_31_1_1 - ABcom * x_31_1_2);
                                            QUICKDouble x_32_17_1 = Qtempx * x_32_7_1 + WQtempx * x_32_7_2 +  2 * CDtemp * ( x_32_1_1 - ABcom * x_32_1_2) +  4 * ABCDtemp * x_17_7_2;
                                            QUICKDouble x_33_17_1 = Qtempx * x_33_7_1 + WQtempx * x_33_7_2 +  2 * CDtemp * ( x_33_1_1 - ABcom * x_33_1_2);
                                            QUICKDouble x_34_17_1 = Qtempx * x_34_7_1 + WQtempx * x_34_7_2 +  2 * CDtemp * ( x_34_1_1 - ABcom * x_34_1_2);
                                            QUICKDouble x_20_18_1 = Qtempy * x_20_8_1 + WQtempy * x_20_8_2 +  2 * CDtemp * ( x_20_2_1 - ABcom * x_20_2_2) +  2 * ABCDtemp * x_11_8_2;
                                            QUICKDouble x_21_18_1 = Qtempy * x_21_8_1 + WQtempy * x_21_8_2 +  2 * CDtemp * ( x_21_2_1 - ABcom * x_21_2_2);
                                            QUICKDouble x_22_18_1 = Qtempy * x_22_8_1 + WQtempy * x_22_8_2 +  2 * CDtemp * ( x_22_2_1 - ABcom * x_22_2_2) +  2 * ABCDtemp * x_16_8_2;
                                            QUICKDouble x_23_18_1 = Qtempy * x_23_8_1 + WQtempy * x_23_8_2 +  2 * CDtemp * ( x_23_2_1 - ABcom * x_23_2_2) + ABCDtemp * x_13_8_2;
                                            QUICKDouble x_24_18_1 = Qtempy * x_24_8_1 + WQtempy * x_24_8_2 +  2 * CDtemp * ( x_24_2_1 - ABcom * x_24_2_2) +  2 * ABCDtemp * x_10_8_2;
                                            QUICKDouble x_25_18_1 = Qtempy * x_25_8_1 + WQtempy * x_25_8_2 +  2 * CDtemp * ( x_25_2_1 - ABcom * x_25_2_2) + ABCDtemp * x_14_8_2;
                                            QUICKDouble x_26_18_1 = Qtempy * x_26_8_1 + WQtempy * x_26_8_2 +  2 * CDtemp * ( x_26_2_1 - ABcom * x_26_2_2);
                                            QUICKDouble x_27_18_1 = Qtempy * x_27_8_1 + WQtempy * x_27_8_2 +  2 * CDtemp * ( x_27_2_1 - ABcom * x_27_2_2);
                                            QUICKDouble x_28_18_1 = Qtempy * x_28_8_1 + WQtempy * x_28_8_2 +  2 * CDtemp * ( x_28_2_1 - ABcom * x_28_2_2) + ABCDtemp * x_17_8_2;
                                            QUICKDouble x_29_18_1 = Qtempy * x_29_8_1 + WQtempy * x_29_8_2 +  2 * CDtemp * ( x_29_2_1 - ABcom * x_29_2_2) +  3 * ABCDtemp * x_12_8_2;
                                            QUICKDouble x_30_18_1 = Qtempy * x_30_8_1 + WQtempy * x_30_8_2 +  2 * CDtemp * ( x_30_2_1 - ABcom * x_30_2_2) +  3 * ABCDtemp * x_15_8_2;
                                            QUICKDouble x_31_18_1 = Qtempy * x_31_8_1 + WQtempy * x_31_8_2 +  2 * CDtemp * ( x_31_2_1 - ABcom * x_31_2_2) + ABCDtemp * x_19_8_2;
                                            QUICKDouble x_32_18_1 = Qtempy * x_32_8_1 + WQtempy * x_32_8_2 +  2 * CDtemp * ( x_32_2_1 - ABcom * x_32_2_2);
                                            QUICKDouble x_33_18_1 = Qtempy * x_33_8_1 + WQtempy * x_33_8_2 +  2 * CDtemp * ( x_33_2_1 - ABcom * x_33_2_2) +  4 * ABCDtemp * x_18_8_2;
                                            QUICKDouble x_34_18_1 = Qtempy * x_34_8_1 + WQtempy * x_34_8_2 +  2 * CDtemp * ( x_34_2_1 - ABcom * x_34_2_2);
                                            QUICKDouble x_20_19_1 = Qtempz * x_20_9_1 + WQtempz * x_20_9_2 +  2 * CDtemp * ( x_20_3_1 - ABcom * x_20_3_2);
                                            QUICKDouble x_21_19_1 = Qtempz * x_21_9_1 + WQtempz * x_21_9_2 +  2 * CDtemp * ( x_21_3_1 - ABcom * x_21_3_2) +  2 * ABCDtemp * x_13_9_2;
                                            QUICKDouble x_22_19_1 = Qtempz * x_22_9_1 + WQtempz * x_22_9_2 +  2 * CDtemp * ( x_22_3_1 - ABcom * x_22_3_2) +  2 * ABCDtemp * x_15_9_2;
                                            QUICKDouble x_23_19_1 = Qtempz * x_23_9_1 + WQtempz * x_23_9_2 +  2 * CDtemp * ( x_23_3_1 - ABcom * x_23_3_2) + ABCDtemp * x_11_9_2;
                                            QUICKDouble x_24_19_1 = Qtempz * x_24_9_1 + WQtempz * x_24_9_2 +  2 * CDtemp * ( x_24_3_1 - ABcom * x_24_3_2) + ABCDtemp * x_12_9_2;
                                            QUICKDouble x_25_19_1 = Qtempz * x_25_9_1 + WQtempz * x_25_9_2 +  2 * CDtemp * ( x_25_3_1 - ABcom * x_25_3_2) +  2 * ABCDtemp * x_10_9_2;
                                            QUICKDouble x_26_19_1 = Qtempz * x_26_9_1 + WQtempz * x_26_9_2 +  2 * CDtemp * ( x_26_3_1 - ABcom * x_26_3_2) + ABCDtemp * x_17_9_2;
                                            QUICKDouble x_27_19_1 = Qtempz * x_27_9_1 + WQtempz * x_27_9_2 +  2 * CDtemp * ( x_27_3_1 - ABcom * x_27_3_2) +  3 * ABCDtemp * x_14_9_2;
                                            QUICKDouble x_28_19_1 = Qtempz * x_28_9_1 + WQtempz * x_28_9_2 +  2 * CDtemp * ( x_28_3_1 - ABcom * x_28_3_2);
                                            QUICKDouble x_29_19_1 = Qtempz * x_29_9_1 + WQtempz * x_29_9_2 +  2 * CDtemp * ( x_29_3_1 - ABcom * x_29_3_2);
                                            QUICKDouble x_30_19_1 = Qtempz * x_30_9_1 + WQtempz * x_30_9_2 +  2 * CDtemp * ( x_30_3_1 - ABcom * x_30_3_2) + ABCDtemp * x_18_9_2;
                                            QUICKDouble x_31_19_1 = Qtempz * x_31_9_1 + WQtempz * x_31_9_2 +  2 * CDtemp * ( x_31_3_1 - ABcom * x_31_3_2) +  3 * ABCDtemp * x_16_9_2;
                                            QUICKDouble x_32_19_1 = Qtempz * x_32_9_1 + WQtempz * x_32_9_2 +  2 * CDtemp * ( x_32_3_1 - ABcom * x_32_3_2);
                                            QUICKDouble x_33_19_1 = Qtempz * x_33_9_1 + WQtempz * x_33_9_2 +  2 * CDtemp * ( x_33_3_1 - ABcom * x_33_3_2);
                                            QUICKDouble x_34_19_1 = Qtempz * x_34_9_1 + WQtempz * x_34_9_2 +  2 * CDtemp * ( x_34_3_1 - ABcom * x_34_3_2) +  4 * ABCDtemp * x_19_9_2;
                                            
                                            //GSGS(0, YVerticalTemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz, ABCDtemp, CDtemp, ABcom);
                                            
                                            LOC2(store,20,20, STOREDIM, STOREDIM) += Qtempx * x_20_12_0 + WQtempx * x_20_12_1 + CDtemp * ( x_20_8_0 - ABcom * x_20_8_1) +  2 * ABCDtemp * x_12_12_1;
                                            LOC2(store,21,20, STOREDIM, STOREDIM) += Qtempx * x_21_12_0 + WQtempx * x_21_12_1 + CDtemp * ( x_21_8_0 - ABcom * x_21_8_1) +  2 * ABCDtemp * x_14_12_1;
                                            LOC2(store,22,20, STOREDIM, STOREDIM) += Qtempx * x_22_12_0 + WQtempx * x_22_12_1 + CDtemp * ( x_22_8_0 - ABcom * x_22_8_1);
                                            LOC2(store,23,20, STOREDIM, STOREDIM) += Qtempx * x_23_12_0 + WQtempx * x_23_12_1 + CDtemp * ( x_23_8_0 - ABcom * x_23_8_1) +  2 * ABCDtemp * x_10_12_1;
                                            LOC2(store,24,20, STOREDIM, STOREDIM) += Qtempx * x_24_12_0 + WQtempx * x_24_12_1 + CDtemp * ( x_24_8_0 - ABcom * x_24_8_1) + ABCDtemp * x_15_12_1;
                                            LOC2(store,25,20, STOREDIM, STOREDIM) += Qtempx * x_25_12_0 + WQtempx * x_25_12_1 + CDtemp * ( x_25_8_0 - ABcom * x_25_8_1) + ABCDtemp * x_16_12_1;
                                            LOC2(store,26,20, STOREDIM, STOREDIM) += Qtempx * x_26_12_0 + WQtempx * x_26_12_1 + CDtemp * ( x_26_8_0 - ABcom * x_26_8_1) +  3 * ABCDtemp * x_13_12_1;
                                            LOC2(store,27,20, STOREDIM, STOREDIM) += Qtempx * x_27_12_0 + WQtempx * x_27_12_1 + CDtemp * ( x_27_8_0 - ABcom * x_27_8_1) + ABCDtemp * x_19_12_1;
                                            LOC2(store,28,20, STOREDIM, STOREDIM) += Qtempx * x_28_12_0 + WQtempx * x_28_12_1 + CDtemp * ( x_28_8_0 - ABcom * x_28_8_1) +  3 * ABCDtemp * x_11_12_1;
                                            LOC2(store,29,20, STOREDIM, STOREDIM) += Qtempx * x_29_12_0 + WQtempx * x_29_12_1 + CDtemp * ( x_29_8_0 - ABcom * x_29_8_1) + ABCDtemp * x_18_12_1;
                                            LOC2(store,30,20, STOREDIM, STOREDIM) += Qtempx * x_30_12_0 + WQtempx * x_30_12_1 + CDtemp * ( x_30_8_0 - ABcom * x_30_8_1);
                                            LOC2(store,31,20, STOREDIM, STOREDIM) += Qtempx * x_31_12_0 + WQtempx * x_31_12_1 + CDtemp * ( x_31_8_0 - ABcom * x_31_8_1);
                                            LOC2(store,32,20, STOREDIM, STOREDIM) += Qtempx * x_32_12_0 + WQtempx * x_32_12_1 + CDtemp * ( x_32_8_0 - ABcom * x_32_8_1) +  4 * ABCDtemp * x_17_12_1;
                                            LOC2(store,33,20, STOREDIM, STOREDIM) += Qtempx * x_33_12_0 + WQtempx * x_33_12_1 + CDtemp * ( x_33_8_0 - ABcom * x_33_8_1);
                                            LOC2(store,34,20, STOREDIM, STOREDIM) += Qtempx * x_34_12_0 + WQtempx * x_34_12_1 + CDtemp * ( x_34_8_0 - ABcom * x_34_8_1);
                                            LOC2(store,20,21, STOREDIM, STOREDIM) += Qtempx * x_20_14_0 + WQtempx * x_20_14_1 + CDtemp * ( x_20_9_0 - ABcom * x_20_9_1) +  2 * ABCDtemp * x_12_14_1;
                                            LOC2(store,21,21, STOREDIM, STOREDIM) += Qtempx * x_21_14_0 + WQtempx * x_21_14_1 + CDtemp * ( x_21_9_0 - ABcom * x_21_9_1) +  2 * ABCDtemp * x_14_14_1;
                                            LOC2(store,22,21, STOREDIM, STOREDIM) += Qtempx * x_22_14_0 + WQtempx * x_22_14_1 + CDtemp * ( x_22_9_0 - ABcom * x_22_9_1);
                                            LOC2(store,23,21, STOREDIM, STOREDIM) += Qtempx * x_23_14_0 + WQtempx * x_23_14_1 + CDtemp * ( x_23_9_0 - ABcom * x_23_9_1) +  2 * ABCDtemp * x_10_14_1;
                                            LOC2(store,24,21, STOREDIM, STOREDIM) += Qtempx * x_24_14_0 + WQtempx * x_24_14_1 + CDtemp * ( x_24_9_0 - ABcom * x_24_9_1) + ABCDtemp * x_15_14_1;
                                            LOC2(store,25,21, STOREDIM, STOREDIM) += Qtempx * x_25_14_0 + WQtempx * x_25_14_1 + CDtemp * ( x_25_9_0 - ABcom * x_25_9_1) + ABCDtemp * x_16_14_1;
                                            LOC2(store,26,21, STOREDIM, STOREDIM) += Qtempx * x_26_14_0 + WQtempx * x_26_14_1 + CDtemp * ( x_26_9_0 - ABcom * x_26_9_1) +  3 * ABCDtemp * x_13_14_1;
                                            LOC2(store,27,21, STOREDIM, STOREDIM) += Qtempx * x_27_14_0 + WQtempx * x_27_14_1 + CDtemp * ( x_27_9_0 - ABcom * x_27_9_1) + ABCDtemp * x_19_14_1;
                                            LOC2(store,28,21, STOREDIM, STOREDIM) += Qtempx * x_28_14_0 + WQtempx * x_28_14_1 + CDtemp * ( x_28_9_0 - ABcom * x_28_9_1) +  3 * ABCDtemp * x_11_14_1;
                                            LOC2(store,29,21, STOREDIM, STOREDIM) += Qtempx * x_29_14_0 + WQtempx * x_29_14_1 + CDtemp * ( x_29_9_0 - ABcom * x_29_9_1) + ABCDtemp * x_18_14_1;
                                            LOC2(store,30,21, STOREDIM, STOREDIM) += Qtempx * x_30_14_0 + WQtempx * x_30_14_1 + CDtemp * ( x_30_9_0 - ABcom * x_30_9_1);
                                            LOC2(store,31,21, STOREDIM, STOREDIM) += Qtempx * x_31_14_0 + WQtempx * x_31_14_1 + CDtemp * ( x_31_9_0 - ABcom * x_31_9_1);
                                            LOC2(store,32,21, STOREDIM, STOREDIM) += Qtempx * x_32_14_0 + WQtempx * x_32_14_1 + CDtemp * ( x_32_9_0 - ABcom * x_32_9_1) +  4 * ABCDtemp * x_17_14_1;
                                            LOC2(store,33,21, STOREDIM, STOREDIM) += Qtempx * x_33_14_0 + WQtempx * x_33_14_1 + CDtemp * ( x_33_9_0 - ABcom * x_33_9_1);
                                            LOC2(store,34,21, STOREDIM, STOREDIM) += Qtempx * x_34_14_0 + WQtempx * x_34_14_1 + CDtemp * ( x_34_9_0 - ABcom * x_34_9_1);
                                            LOC2(store,20,22, STOREDIM, STOREDIM) += Qtempy * x_20_16_0 + WQtempy * x_20_16_1 + CDtemp * ( x_20_9_0 - ABcom * x_20_9_1) +  2 * ABCDtemp * x_11_16_1;
                                            LOC2(store,21,22, STOREDIM, STOREDIM) += Qtempy * x_21_16_0 + WQtempy * x_21_16_1 + CDtemp * ( x_21_9_0 - ABcom * x_21_9_1);
                                            LOC2(store,22,22, STOREDIM, STOREDIM) += Qtempy * x_22_16_0 + WQtempy * x_22_16_1 + CDtemp * ( x_22_9_0 - ABcom * x_22_9_1) +  2 * ABCDtemp * x_16_16_1;
                                            LOC2(store,23,22, STOREDIM, STOREDIM) += Qtempy * x_23_16_0 + WQtempy * x_23_16_1 + CDtemp * ( x_23_9_0 - ABcom * x_23_9_1) + ABCDtemp * x_13_16_1;
                                            LOC2(store,24,22, STOREDIM, STOREDIM) += Qtempy * x_24_16_0 + WQtempy * x_24_16_1 + CDtemp * ( x_24_9_0 - ABcom * x_24_9_1) +  2 * ABCDtemp * x_10_16_1;
                                            LOC2(store,25,22, STOREDIM, STOREDIM) += Qtempy * x_25_16_0 + WQtempy * x_25_16_1 + CDtemp * ( x_25_9_0 - ABcom * x_25_9_1) + ABCDtemp * x_14_16_1;
                                            LOC2(store,26,22, STOREDIM, STOREDIM) += Qtempy * x_26_16_0 + WQtempy * x_26_16_1 + CDtemp * ( x_26_9_0 - ABcom * x_26_9_1);
                                            LOC2(store,27,22, STOREDIM, STOREDIM) += Qtempy * x_27_16_0 + WQtempy * x_27_16_1 + CDtemp * ( x_27_9_0 - ABcom * x_27_9_1);
                                            LOC2(store,28,22, STOREDIM, STOREDIM) += Qtempy * x_28_16_0 + WQtempy * x_28_16_1 + CDtemp * ( x_28_9_0 - ABcom * x_28_9_1) + ABCDtemp * x_17_16_1;
                                            LOC2(store,29,22, STOREDIM, STOREDIM) += Qtempy * x_29_16_0 + WQtempy * x_29_16_1 + CDtemp * ( x_29_9_0 - ABcom * x_29_9_1) +  3 * ABCDtemp * x_12_16_1;
                                            LOC2(store,30,22, STOREDIM, STOREDIM) += Qtempy * x_30_16_0 + WQtempy * x_30_16_1 + CDtemp * ( x_30_9_0 - ABcom * x_30_9_1) +  3 * ABCDtemp * x_15_16_1;
                                            LOC2(store,31,22, STOREDIM, STOREDIM) += Qtempy * x_31_16_0 + WQtempy * x_31_16_1 + CDtemp * ( x_31_9_0 - ABcom * x_31_9_1) + ABCDtemp * x_19_16_1;
                                            LOC2(store,32,22, STOREDIM, STOREDIM) += Qtempy * x_32_16_0 + WQtempy * x_32_16_1 + CDtemp * ( x_32_9_0 - ABcom * x_32_9_1);
                                            LOC2(store,33,22, STOREDIM, STOREDIM) += Qtempy * x_33_16_0 + WQtempy * x_33_16_1 + CDtemp * ( x_33_9_0 - ABcom * x_33_9_1) +  4 * ABCDtemp * x_18_16_1;
                                            LOC2(store,34,22, STOREDIM, STOREDIM) += Qtempy * x_34_16_0 + WQtempy * x_34_16_1 + CDtemp * ( x_34_9_0 - ABcom * x_34_9_1);
                                            LOC2(store,20,23, STOREDIM, STOREDIM) += Qtempx * x_20_10_0 + WQtempx * x_20_10_1 + CDtemp * ( x_20_5_0 - ABcom * x_20_5_1) +  2 * ABCDtemp * x_12_10_1;
                                            LOC2(store,21,23, STOREDIM, STOREDIM) += Qtempx * x_21_10_0 + WQtempx * x_21_10_1 + CDtemp * ( x_21_5_0 - ABcom * x_21_5_1) +  2 * ABCDtemp * x_14_10_1;
                                            LOC2(store,22,23, STOREDIM, STOREDIM) += Qtempx * x_22_10_0 + WQtempx * x_22_10_1 + CDtemp * ( x_22_5_0 - ABcom * x_22_5_1);
                                            LOC2(store,23,23, STOREDIM, STOREDIM) += Qtempx * x_23_10_0 + WQtempx * x_23_10_1 + CDtemp * ( x_23_5_0 - ABcom * x_23_5_1) +  2 * ABCDtemp * x_10_10_1;
                                            LOC2(store,24,23, STOREDIM, STOREDIM) += Qtempx * x_24_10_0 + WQtempx * x_24_10_1 + CDtemp * ( x_24_5_0 - ABcom * x_24_5_1) + ABCDtemp * x_15_10_1;
                                            LOC2(store,25,23, STOREDIM, STOREDIM) += Qtempx * x_25_10_0 + WQtempx * x_25_10_1 + CDtemp * ( x_25_5_0 - ABcom * x_25_5_1) + ABCDtemp * x_16_10_1;
                                            LOC2(store,26,23, STOREDIM, STOREDIM) += Qtempx * x_26_10_0 + WQtempx * x_26_10_1 + CDtemp * ( x_26_5_0 - ABcom * x_26_5_1) +  3 * ABCDtemp * x_13_10_1;
                                            LOC2(store,27,23, STOREDIM, STOREDIM) += Qtempx * x_27_10_0 + WQtempx * x_27_10_1 + CDtemp * ( x_27_5_0 - ABcom * x_27_5_1) + ABCDtemp * x_19_10_1;
                                            LOC2(store,28,23, STOREDIM, STOREDIM) += Qtempx * x_28_10_0 + WQtempx * x_28_10_1 + CDtemp * ( x_28_5_0 - ABcom * x_28_5_1) +  3 * ABCDtemp * x_11_10_1;
                                            LOC2(store,29,23, STOREDIM, STOREDIM) += Qtempx * x_29_10_0 + WQtempx * x_29_10_1 + CDtemp * ( x_29_5_0 - ABcom * x_29_5_1) + ABCDtemp * x_18_10_1;
                                            LOC2(store,30,23, STOREDIM, STOREDIM) += Qtempx * x_30_10_0 + WQtempx * x_30_10_1 + CDtemp * ( x_30_5_0 - ABcom * x_30_5_1);
                                            LOC2(store,31,23, STOREDIM, STOREDIM) += Qtempx * x_31_10_0 + WQtempx * x_31_10_1 + CDtemp * ( x_31_5_0 - ABcom * x_31_5_1);
                                            LOC2(store,32,23, STOREDIM, STOREDIM) += Qtempx * x_32_10_0 + WQtempx * x_32_10_1 + CDtemp * ( x_32_5_0 - ABcom * x_32_5_1) +  4 * ABCDtemp * x_17_10_1;
                                            LOC2(store,33,23, STOREDIM, STOREDIM) += Qtempx * x_33_10_0 + WQtempx * x_33_10_1 + CDtemp * ( x_33_5_0 - ABcom * x_33_5_1);
                                            LOC2(store,34,23, STOREDIM, STOREDIM) += Qtempx * x_34_10_0 + WQtempx * x_34_10_1 + CDtemp * ( x_34_5_0 - ABcom * x_34_5_1);
                                            LOC2(store,20,24, STOREDIM, STOREDIM) += Qtempx * x_20_15_0 + WQtempx * x_20_15_1 +  2 * ABCDtemp * x_12_15_1;
                                            LOC2(store,21,24, STOREDIM, STOREDIM) += Qtempx * x_21_15_0 + WQtempx * x_21_15_1 +  2 * ABCDtemp * x_14_15_1;
                                            LOC2(store,22,24, STOREDIM, STOREDIM) += Qtempx * x_22_15_0 + WQtempx * x_22_15_1;
                                            LOC2(store,23,24, STOREDIM, STOREDIM) += Qtempx * x_23_15_0 + WQtempx * x_23_15_1 +  2 * ABCDtemp * x_10_15_1;
                                            LOC2(store,24,24, STOREDIM, STOREDIM) += Qtempx * x_24_15_0 + WQtempx * x_24_15_1 + ABCDtemp * x_15_15_1;
                                            LOC2(store,25,24, STOREDIM, STOREDIM) += Qtempx * x_25_15_0 + WQtempx * x_25_15_1 + ABCDtemp * x_16_15_1;
                                            LOC2(store,26,24, STOREDIM, STOREDIM) += Qtempx * x_26_15_0 + WQtempx * x_26_15_1 +  3 * ABCDtemp * x_13_15_1;
                                            LOC2(store,27,24, STOREDIM, STOREDIM) += Qtempx * x_27_15_0 + WQtempx * x_27_15_1 + ABCDtemp * x_19_15_1;
                                            LOC2(store,28,24, STOREDIM, STOREDIM) += Qtempx * x_28_15_0 + WQtempx * x_28_15_1 +  3 * ABCDtemp * x_11_15_1;
                                            LOC2(store,29,24, STOREDIM, STOREDIM) += Qtempx * x_29_15_0 + WQtempx * x_29_15_1 + ABCDtemp * x_18_15_1;
                                            LOC2(store,30,24, STOREDIM, STOREDIM) += Qtempx * x_30_15_0 + WQtempx * x_30_15_1;
                                            LOC2(store,31,24, STOREDIM, STOREDIM) += Qtempx * x_31_15_0 + WQtempx * x_31_15_1;
                                            LOC2(store,32,24, STOREDIM, STOREDIM) += Qtempx * x_32_15_0 + WQtempx * x_32_15_1 +  4 * ABCDtemp * x_17_15_1;
                                            LOC2(store,33,24, STOREDIM, STOREDIM) += Qtempx * x_33_15_0 + WQtempx * x_33_15_1;
                                            LOC2(store,34,24, STOREDIM, STOREDIM) += Qtempx * x_34_15_0 + WQtempx * x_34_15_1;
                                            LOC2(store,20,25, STOREDIM, STOREDIM) += Qtempx * x_20_16_0 + WQtempx * x_20_16_1 +  2 * ABCDtemp * x_12_16_1;
                                            LOC2(store,21,25, STOREDIM, STOREDIM) += Qtempx * x_21_16_0 + WQtempx * x_21_16_1 +  2 * ABCDtemp * x_14_16_1;
                                            LOC2(store,22,25, STOREDIM, STOREDIM) += Qtempx * x_22_16_0 + WQtempx * x_22_16_1;
                                            LOC2(store,23,25, STOREDIM, STOREDIM) += Qtempx * x_23_16_0 + WQtempx * x_23_16_1 +  2 * ABCDtemp * x_10_16_1;
                                            LOC2(store,24,25, STOREDIM, STOREDIM) += Qtempx * x_24_16_0 + WQtempx * x_24_16_1 + ABCDtemp * x_15_16_1;
                                            LOC2(store,25,25, STOREDIM, STOREDIM) += Qtempx * x_25_16_0 + WQtempx * x_25_16_1 + ABCDtemp * x_16_16_1;
                                            LOC2(store,26,25, STOREDIM, STOREDIM) += Qtempx * x_26_16_0 + WQtempx * x_26_16_1 +  3 * ABCDtemp * x_13_16_1;
                                            LOC2(store,27,25, STOREDIM, STOREDIM) += Qtempx * x_27_16_0 + WQtempx * x_27_16_1 + ABCDtemp * x_19_16_1;
                                            LOC2(store,28,25, STOREDIM, STOREDIM) += Qtempx * x_28_16_0 + WQtempx * x_28_16_1 +  3 * ABCDtemp * x_11_16_1;
                                            LOC2(store,29,25, STOREDIM, STOREDIM) += Qtempx * x_29_16_0 + WQtempx * x_29_16_1 + ABCDtemp * x_18_16_1;
                                            LOC2(store,30,25, STOREDIM, STOREDIM) += Qtempx * x_30_16_0 + WQtempx * x_30_16_1;
                                            LOC2(store,31,25, STOREDIM, STOREDIM) += Qtempx * x_31_16_0 + WQtempx * x_31_16_1;
                                            LOC2(store,32,25, STOREDIM, STOREDIM) += Qtempx * x_32_16_0 + WQtempx * x_32_16_1 +  4 * ABCDtemp * x_17_16_1;
                                            LOC2(store,33,25, STOREDIM, STOREDIM) += Qtempx * x_33_16_0 + WQtempx * x_33_16_1;
                                            LOC2(store,34,25, STOREDIM, STOREDIM) += Qtempx * x_34_16_0 + WQtempx * x_34_16_1;
                                            LOC2(store,20,26, STOREDIM, STOREDIM) += Qtempx * x_20_13_0 + WQtempx * x_20_13_1 +  2 * CDtemp * ( x_20_6_0 - ABcom * x_20_6_1) +  2 * ABCDtemp * x_12_13_1;
                                            LOC2(store,21,26, STOREDIM, STOREDIM) += Qtempx * x_21_13_0 + WQtempx * x_21_13_1 +  2 * CDtemp * ( x_21_6_0 - ABcom * x_21_6_1) +  2 * ABCDtemp * x_14_13_1;
                                            LOC2(store,22,26, STOREDIM, STOREDIM) += Qtempx * x_22_13_0 + WQtempx * x_22_13_1 +  2 * CDtemp * ( x_22_6_0 - ABcom * x_22_6_1);
                                            LOC2(store,23,26, STOREDIM, STOREDIM) += Qtempx * x_23_13_0 + WQtempx * x_23_13_1 +  2 * CDtemp * ( x_23_6_0 - ABcom * x_23_6_1) +  2 * ABCDtemp * x_10_13_1;
                                            LOC2(store,24,26, STOREDIM, STOREDIM) += Qtempx * x_24_13_0 + WQtempx * x_24_13_1 +  2 * CDtemp * ( x_24_6_0 - ABcom * x_24_6_1) + ABCDtemp * x_15_13_1;
                                            LOC2(store,25,26, STOREDIM, STOREDIM) += Qtempx * x_25_13_0 + WQtempx * x_25_13_1 +  2 * CDtemp * ( x_25_6_0 - ABcom * x_25_6_1) + ABCDtemp * x_16_13_1;
                                            LOC2(store,26,26, STOREDIM, STOREDIM) += Qtempx * x_26_13_0 + WQtempx * x_26_13_1 +  2 * CDtemp * ( x_26_6_0 - ABcom * x_26_6_1) +  3 * ABCDtemp * x_13_13_1;
                                            LOC2(store,27,26, STOREDIM, STOREDIM) += Qtempx * x_27_13_0 + WQtempx * x_27_13_1 +  2 * CDtemp * ( x_27_6_0 - ABcom * x_27_6_1) + ABCDtemp * x_19_13_1;
                                            LOC2(store,28,26, STOREDIM, STOREDIM) += Qtempx * x_28_13_0 + WQtempx * x_28_13_1 +  2 * CDtemp * ( x_28_6_0 - ABcom * x_28_6_1) +  3 * ABCDtemp * x_11_13_1;
                                            LOC2(store,29,26, STOREDIM, STOREDIM) += Qtempx * x_29_13_0 + WQtempx * x_29_13_1 +  2 * CDtemp * ( x_29_6_0 - ABcom * x_29_6_1) + ABCDtemp * x_18_13_1;
                                            LOC2(store,30,26, STOREDIM, STOREDIM) += Qtempx * x_30_13_0 + WQtempx * x_30_13_1 +  2 * CDtemp * ( x_30_6_0 - ABcom * x_30_6_1);
                                            LOC2(store,31,26, STOREDIM, STOREDIM) += Qtempx * x_31_13_0 + WQtempx * x_31_13_1 +  2 * CDtemp * ( x_31_6_0 - ABcom * x_31_6_1);
                                            LOC2(store,32,26, STOREDIM, STOREDIM) += Qtempx * x_32_13_0 + WQtempx * x_32_13_1 +  2 * CDtemp * ( x_32_6_0 - ABcom * x_32_6_1) +  4 * ABCDtemp * x_17_13_1;
                                            LOC2(store,33,26, STOREDIM, STOREDIM) += Qtempx * x_33_13_0 + WQtempx * x_33_13_1 +  2 * CDtemp * ( x_33_6_0 - ABcom * x_33_6_1);
                                            LOC2(store,34,26, STOREDIM, STOREDIM) += Qtempx * x_34_13_0 + WQtempx * x_34_13_1 +  2 * CDtemp * ( x_34_6_0 - ABcom * x_34_6_1);
                                            LOC2(store,20,27, STOREDIM, STOREDIM) += Qtempx * x_20_19_0 + WQtempx * x_20_19_1 +  2 * ABCDtemp * x_12_19_1;
                                            LOC2(store,21,27, STOREDIM, STOREDIM) += Qtempx * x_21_19_0 + WQtempx * x_21_19_1 +  2 * ABCDtemp * x_14_19_1;
                                            LOC2(store,22,27, STOREDIM, STOREDIM) += Qtempx * x_22_19_0 + WQtempx * x_22_19_1;
                                            LOC2(store,23,27, STOREDIM, STOREDIM) += Qtempx * x_23_19_0 + WQtempx * x_23_19_1 +  2 * ABCDtemp * x_10_19_1;
                                            LOC2(store,24,27, STOREDIM, STOREDIM) += Qtempx * x_24_19_0 + WQtempx * x_24_19_1 + ABCDtemp * x_15_19_1;
                                            LOC2(store,25,27, STOREDIM, STOREDIM) += Qtempx * x_25_19_0 + WQtempx * x_25_19_1 + ABCDtemp * x_16_19_1;
                                            LOC2(store,26,27, STOREDIM, STOREDIM) += Qtempx * x_26_19_0 + WQtempx * x_26_19_1 +  3 * ABCDtemp * x_13_19_1;
                                            LOC2(store,27,27, STOREDIM, STOREDIM) += Qtempx * x_27_19_0 + WQtempx * x_27_19_1 + ABCDtemp * x_19_19_1;
                                            LOC2(store,28,27, STOREDIM, STOREDIM) += Qtempx * x_28_19_0 + WQtempx * x_28_19_1 +  3 * ABCDtemp * x_11_19_1;
                                            LOC2(store,29,27, STOREDIM, STOREDIM) += Qtempx * x_29_19_0 + WQtempx * x_29_19_1 + ABCDtemp * x_18_19_1;
                                            LOC2(store,30,27, STOREDIM, STOREDIM) += Qtempx * x_30_19_0 + WQtempx * x_30_19_1;
                                            LOC2(store,31,27, STOREDIM, STOREDIM) += Qtempx * x_31_19_0 + WQtempx * x_31_19_1;
                                            LOC2(store,32,27, STOREDIM, STOREDIM) += Qtempx * x_32_19_0 + WQtempx * x_32_19_1 +  4 * ABCDtemp * x_17_19_1;
                                            LOC2(store,33,27, STOREDIM, STOREDIM) += Qtempx * x_33_19_0 + WQtempx * x_33_19_1;
                                            LOC2(store,34,27, STOREDIM, STOREDIM) += Qtempx * x_34_19_0 + WQtempx * x_34_19_1;
                                            LOC2(store,20,28, STOREDIM, STOREDIM) += Qtempx * x_20_11_0 + WQtempx * x_20_11_1 +  2 * CDtemp * ( x_20_4_0 - ABcom * x_20_4_1) +  2 * ABCDtemp * x_12_11_1;
                                            LOC2(store,21,28, STOREDIM, STOREDIM) += Qtempx * x_21_11_0 + WQtempx * x_21_11_1 +  2 * CDtemp * ( x_21_4_0 - ABcom * x_21_4_1) +  2 * ABCDtemp * x_14_11_1;
                                            LOC2(store,22,28, STOREDIM, STOREDIM) += Qtempx * x_22_11_0 + WQtempx * x_22_11_1 +  2 * CDtemp * ( x_22_4_0 - ABcom * x_22_4_1);
                                            LOC2(store,23,28, STOREDIM, STOREDIM) += Qtempx * x_23_11_0 + WQtempx * x_23_11_1 +  2 * CDtemp * ( x_23_4_0 - ABcom * x_23_4_1) +  2 * ABCDtemp * x_10_11_1;
                                            LOC2(store,24,28, STOREDIM, STOREDIM) += Qtempx * x_24_11_0 + WQtempx * x_24_11_1 +  2 * CDtemp * ( x_24_4_0 - ABcom * x_24_4_1) + ABCDtemp * x_15_11_1;
                                            LOC2(store,25,28, STOREDIM, STOREDIM) += Qtempx * x_25_11_0 + WQtempx * x_25_11_1 +  2 * CDtemp * ( x_25_4_0 - ABcom * x_25_4_1) + ABCDtemp * x_16_11_1;
                                            LOC2(store,26,28, STOREDIM, STOREDIM) += Qtempx * x_26_11_0 + WQtempx * x_26_11_1 +  2 * CDtemp * ( x_26_4_0 - ABcom * x_26_4_1) +  3 * ABCDtemp * x_13_11_1;
                                            LOC2(store,27,28, STOREDIM, STOREDIM) += Qtempx * x_27_11_0 + WQtempx * x_27_11_1 +  2 * CDtemp * ( x_27_4_0 - ABcom * x_27_4_1) + ABCDtemp * x_19_11_1;
                                            LOC2(store,28,28, STOREDIM, STOREDIM) += Qtempx * x_28_11_0 + WQtempx * x_28_11_1 +  2 * CDtemp * ( x_28_4_0 - ABcom * x_28_4_1) +  3 * ABCDtemp * x_11_11_1;
                                            LOC2(store,29,28, STOREDIM, STOREDIM) += Qtempx * x_29_11_0 + WQtempx * x_29_11_1 +  2 * CDtemp * ( x_29_4_0 - ABcom * x_29_4_1) + ABCDtemp * x_18_11_1;
                                            LOC2(store,30,28, STOREDIM, STOREDIM) += Qtempx * x_30_11_0 + WQtempx * x_30_11_1 +  2 * CDtemp * ( x_30_4_0 - ABcom * x_30_4_1);
                                            LOC2(store,31,28, STOREDIM, STOREDIM) += Qtempx * x_31_11_0 + WQtempx * x_31_11_1 +  2 * CDtemp * ( x_31_4_0 - ABcom * x_31_4_1);
                                            LOC2(store,32,28, STOREDIM, STOREDIM) += Qtempx * x_32_11_0 + WQtempx * x_32_11_1 +  2 * CDtemp * ( x_32_4_0 - ABcom * x_32_4_1) +  4 * ABCDtemp * x_17_11_1;
                                            LOC2(store,33,28, STOREDIM, STOREDIM) += Qtempx * x_33_11_0 + WQtempx * x_33_11_1 +  2 * CDtemp * ( x_33_4_0 - ABcom * x_33_4_1);
                                            LOC2(store,34,28, STOREDIM, STOREDIM) += Qtempx * x_34_11_0 + WQtempx * x_34_11_1 +  2 * CDtemp * ( x_34_4_0 - ABcom * x_34_4_1);
                                            LOC2(store,20,29, STOREDIM, STOREDIM) += Qtempx * x_20_18_0 + WQtempx * x_20_18_1 +  2 * ABCDtemp * x_12_18_1;
                                            LOC2(store,21,29, STOREDIM, STOREDIM) += Qtempx * x_21_18_0 + WQtempx * x_21_18_1 +  2 * ABCDtemp * x_14_18_1;
                                            LOC2(store,22,29, STOREDIM, STOREDIM) += Qtempx * x_22_18_0 + WQtempx * x_22_18_1;
                                            LOC2(store,23,29, STOREDIM, STOREDIM) += Qtempx * x_23_18_0 + WQtempx * x_23_18_1 +  2 * ABCDtemp * x_10_18_1;
                                            LOC2(store,24,29, STOREDIM, STOREDIM) += Qtempx * x_24_18_0 + WQtempx * x_24_18_1 + ABCDtemp * x_15_18_1;
                                            LOC2(store,25,29, STOREDIM, STOREDIM) += Qtempx * x_25_18_0 + WQtempx * x_25_18_1 + ABCDtemp * x_16_18_1;
                                            LOC2(store,26,29, STOREDIM, STOREDIM) += Qtempx * x_26_18_0 + WQtempx * x_26_18_1 +  3 * ABCDtemp * x_13_18_1;
                                            LOC2(store,27,29, STOREDIM, STOREDIM) += Qtempx * x_27_18_0 + WQtempx * x_27_18_1 + ABCDtemp * x_19_18_1;
                                            LOC2(store,28,29, STOREDIM, STOREDIM) += Qtempx * x_28_18_0 + WQtempx * x_28_18_1 +  3 * ABCDtemp * x_11_18_1;
                                            LOC2(store,29,29, STOREDIM, STOREDIM) += Qtempx * x_29_18_0 + WQtempx * x_29_18_1 + ABCDtemp * x_18_18_1;
                                            LOC2(store,30,29, STOREDIM, STOREDIM) += Qtempx * x_30_18_0 + WQtempx * x_30_18_1;
                                            LOC2(store,31,29, STOREDIM, STOREDIM) += Qtempx * x_31_18_0 + WQtempx * x_31_18_1;
                                            LOC2(store,32,29, STOREDIM, STOREDIM) += Qtempx * x_32_18_0 + WQtempx * x_32_18_1 +  4 * ABCDtemp * x_17_18_1;
                                            LOC2(store,33,29, STOREDIM, STOREDIM) += Qtempx * x_33_18_0 + WQtempx * x_33_18_1;
                                            LOC2(store,34,29, STOREDIM, STOREDIM) += Qtempx * x_34_18_0 + WQtempx * x_34_18_1;
                                            LOC2(store,20,30, STOREDIM, STOREDIM) += Qtempy * x_20_15_0 + WQtempy * x_20_15_1 +  2 * CDtemp * ( x_20_5_0 - ABcom * x_20_5_1) +  2 * ABCDtemp * x_11_15_1;
                                            LOC2(store,21,30, STOREDIM, STOREDIM) += Qtempy * x_21_15_0 + WQtempy * x_21_15_1 +  2 * CDtemp * ( x_21_5_0 - ABcom * x_21_5_1);
                                            LOC2(store,22,30, STOREDIM, STOREDIM) += Qtempy * x_22_15_0 + WQtempy * x_22_15_1 +  2 * CDtemp * ( x_22_5_0 - ABcom * x_22_5_1) +  2 * ABCDtemp * x_16_15_1;
                                            LOC2(store,23,30, STOREDIM, STOREDIM) += Qtempy * x_23_15_0 + WQtempy * x_23_15_1 +  2 * CDtemp * ( x_23_5_0 - ABcom * x_23_5_1) + ABCDtemp * x_13_15_1;
                                            LOC2(store,24,30, STOREDIM, STOREDIM) += Qtempy * x_24_15_0 + WQtempy * x_24_15_1 +  2 * CDtemp * ( x_24_5_0 - ABcom * x_24_5_1) +  2 * ABCDtemp * x_10_15_1;
                                            LOC2(store,25,30, STOREDIM, STOREDIM) += Qtempy * x_25_15_0 + WQtempy * x_25_15_1 +  2 * CDtemp * ( x_25_5_0 - ABcom * x_25_5_1) + ABCDtemp * x_14_15_1;
                                            LOC2(store,26,30, STOREDIM, STOREDIM) += Qtempy * x_26_15_0 + WQtempy * x_26_15_1 +  2 * CDtemp * ( x_26_5_0 - ABcom * x_26_5_1);
                                            LOC2(store,27,30, STOREDIM, STOREDIM) += Qtempy * x_27_15_0 + WQtempy * x_27_15_1 +  2 * CDtemp * ( x_27_5_0 - ABcom * x_27_5_1);
                                            LOC2(store,28,30, STOREDIM, STOREDIM) += Qtempy * x_28_15_0 + WQtempy * x_28_15_1 +  2 * CDtemp * ( x_28_5_0 - ABcom * x_28_5_1) + ABCDtemp * x_17_15_1;
                                            LOC2(store,29,30, STOREDIM, STOREDIM) += Qtempy * x_29_15_0 + WQtempy * x_29_15_1 +  2 * CDtemp * ( x_29_5_0 - ABcom * x_29_5_1) +  3 * ABCDtemp * x_12_15_1;
                                            LOC2(store,30,30, STOREDIM, STOREDIM) += Qtempy * x_30_15_0 + WQtempy * x_30_15_1 +  2 * CDtemp * ( x_30_5_0 - ABcom * x_30_5_1) +  3 * ABCDtemp * x_15_15_1;
                                            LOC2(store,31,30, STOREDIM, STOREDIM) += Qtempy * x_31_15_0 + WQtempy * x_31_15_1 +  2 * CDtemp * ( x_31_5_0 - ABcom * x_31_5_1) + ABCDtemp * x_19_15_1;
                                            LOC2(store,32,30, STOREDIM, STOREDIM) += Qtempy * x_32_15_0 + WQtempy * x_32_15_1 +  2 * CDtemp * ( x_32_5_0 - ABcom * x_32_5_1);
                                            LOC2(store,33,30, STOREDIM, STOREDIM) += Qtempy * x_33_15_0 + WQtempy * x_33_15_1 +  2 * CDtemp * ( x_33_5_0 - ABcom * x_33_5_1) +  4 * ABCDtemp * x_18_15_1;
                                            LOC2(store,34,30, STOREDIM, STOREDIM) += Qtempy * x_34_15_0 + WQtempy * x_34_15_1 +  2 * CDtemp * ( x_34_5_0 - ABcom * x_34_5_1);
                                            LOC2(store,20,31, STOREDIM, STOREDIM) += Qtempy * x_20_19_0 + WQtempy * x_20_19_1 +  2 * ABCDtemp * x_11_19_1;
                                            LOC2(store,21,31, STOREDIM, STOREDIM) += Qtempy * x_21_19_0 + WQtempy * x_21_19_1;
                                            LOC2(store,22,31, STOREDIM, STOREDIM) += Qtempy * x_22_19_0 + WQtempy * x_22_19_1 +  2 * ABCDtemp * x_16_19_1;
                                            LOC2(store,23,31, STOREDIM, STOREDIM) += Qtempy * x_23_19_0 + WQtempy * x_23_19_1 + ABCDtemp * x_13_19_1;
                                            LOC2(store,24,31, STOREDIM, STOREDIM) += Qtempy * x_24_19_0 + WQtempy * x_24_19_1 +  2 * ABCDtemp * x_10_19_1;
                                            LOC2(store,25,31, STOREDIM, STOREDIM) += Qtempy * x_25_19_0 + WQtempy * x_25_19_1 + ABCDtemp * x_14_19_1;
                                            LOC2(store,26,31, STOREDIM, STOREDIM) += Qtempy * x_26_19_0 + WQtempy * x_26_19_1;
                                            LOC2(store,27,31, STOREDIM, STOREDIM) += Qtempy * x_27_19_0 + WQtempy * x_27_19_1;
                                            LOC2(store,28,31, STOREDIM, STOREDIM) += Qtempy * x_28_19_0 + WQtempy * x_28_19_1 + ABCDtemp * x_17_19_1;
                                            LOC2(store,29,31, STOREDIM, STOREDIM) += Qtempy * x_29_19_0 + WQtempy * x_29_19_1 +  3 * ABCDtemp * x_12_19_1;
                                            LOC2(store,30,31, STOREDIM, STOREDIM) += Qtempy * x_30_19_0 + WQtempy * x_30_19_1 +  3 * ABCDtemp * x_15_19_1;
                                            LOC2(store,31,31, STOREDIM, STOREDIM) += Qtempy * x_31_19_0 + WQtempy * x_31_19_1 + ABCDtemp * x_19_19_1;
                                            LOC2(store,32,31, STOREDIM, STOREDIM) += Qtempy * x_32_19_0 + WQtempy * x_32_19_1;
                                            LOC2(store,33,31, STOREDIM, STOREDIM) += Qtempy * x_33_19_0 + WQtempy * x_33_19_1 +  4 * ABCDtemp * x_18_19_1;
                                            LOC2(store,34,31, STOREDIM, STOREDIM) += Qtempy * x_34_19_0 + WQtempy * x_34_19_1;
                                            LOC2(store,20,32, STOREDIM, STOREDIM) += Qtempx * x_20_17_0 + WQtempx * x_20_17_1 +  3 * CDtemp * ( x_20_7_0 - ABcom * x_20_7_1) +  2 * ABCDtemp * x_12_17_1;
                                            LOC2(store,21,32, STOREDIM, STOREDIM) += Qtempx * x_21_17_0 + WQtempx * x_21_17_1 +  3 * CDtemp * ( x_21_7_0 - ABcom * x_21_7_1) +  2 * ABCDtemp * x_14_17_1;
                                            LOC2(store,22,32, STOREDIM, STOREDIM) += Qtempx * x_22_17_0 + WQtempx * x_22_17_1 +  3 * CDtemp * ( x_22_7_0 - ABcom * x_22_7_1);
                                            LOC2(store,23,32, STOREDIM, STOREDIM) += Qtempx * x_23_17_0 + WQtempx * x_23_17_1 +  3 * CDtemp * ( x_23_7_0 - ABcom * x_23_7_1) +  2 * ABCDtemp * x_10_17_1;
                                            LOC2(store,24,32, STOREDIM, STOREDIM) += Qtempx * x_24_17_0 + WQtempx * x_24_17_1 +  3 * CDtemp * ( x_24_7_0 - ABcom * x_24_7_1) + ABCDtemp * x_15_17_1;
                                            LOC2(store,25,32, STOREDIM, STOREDIM) += Qtempx * x_25_17_0 + WQtempx * x_25_17_1 +  3 * CDtemp * ( x_25_7_0 - ABcom * x_25_7_1) + ABCDtemp * x_16_17_1;
                                            LOC2(store,26,32, STOREDIM, STOREDIM) += Qtempx * x_26_17_0 + WQtempx * x_26_17_1 +  3 * CDtemp * ( x_26_7_0 - ABcom * x_26_7_1) +  3 * ABCDtemp * x_13_17_1;
                                            LOC2(store,27,32, STOREDIM, STOREDIM) += Qtempx * x_27_17_0 + WQtempx * x_27_17_1 +  3 * CDtemp * ( x_27_7_0 - ABcom * x_27_7_1) + ABCDtemp * x_19_17_1;
                                            LOC2(store,28,32, STOREDIM, STOREDIM) += Qtempx * x_28_17_0 + WQtempx * x_28_17_1 +  3 * CDtemp * ( x_28_7_0 - ABcom * x_28_7_1) +  3 * ABCDtemp * x_11_17_1;
                                            LOC2(store,29,32, STOREDIM, STOREDIM) += Qtempx * x_29_17_0 + WQtempx * x_29_17_1 +  3 * CDtemp * ( x_29_7_0 - ABcom * x_29_7_1) + ABCDtemp * x_18_17_1;
                                            LOC2(store,30,32, STOREDIM, STOREDIM) += Qtempx * x_30_17_0 + WQtempx * x_30_17_1 +  3 * CDtemp * ( x_30_7_0 - ABcom * x_30_7_1);
                                            LOC2(store,31,32, STOREDIM, STOREDIM) += Qtempx * x_31_17_0 + WQtempx * x_31_17_1 +  3 * CDtemp * ( x_31_7_0 - ABcom * x_31_7_1);
                                            LOC2(store,32,32, STOREDIM, STOREDIM) += Qtempx * x_32_17_0 + WQtempx * x_32_17_1 +  3 * CDtemp * ( x_32_7_0 - ABcom * x_32_7_1) +  4 * ABCDtemp * x_17_17_1;
                                            LOC2(store,33,32, STOREDIM, STOREDIM) += Qtempx * x_33_17_0 + WQtempx * x_33_17_1 +  3 * CDtemp * ( x_33_7_0 - ABcom * x_33_7_1);
                                            LOC2(store,34,32, STOREDIM, STOREDIM) += Qtempx * x_34_17_0 + WQtempx * x_34_17_1 +  3 * CDtemp * ( x_34_7_0 - ABcom * x_34_7_1);
                                            LOC2(store,20,33, STOREDIM, STOREDIM) += Qtempy * x_20_18_0 + WQtempy * x_20_18_1 +  3 * CDtemp * ( x_20_8_0 - ABcom * x_20_8_1) +  2 * ABCDtemp * x_11_18_1;
                                            LOC2(store,21,33, STOREDIM, STOREDIM) += Qtempy * x_21_18_0 + WQtempy * x_21_18_1 +  3 * CDtemp * ( x_21_8_0 - ABcom * x_21_8_1);
                                            LOC2(store,22,33, STOREDIM, STOREDIM) += Qtempy * x_22_18_0 + WQtempy * x_22_18_1 +  3 * CDtemp * ( x_22_8_0 - ABcom * x_22_8_1) +  2 * ABCDtemp * x_16_18_1;
                                            LOC2(store,23,33, STOREDIM, STOREDIM) += Qtempy * x_23_18_0 + WQtempy * x_23_18_1 +  3 * CDtemp * ( x_23_8_0 - ABcom * x_23_8_1) + ABCDtemp * x_13_18_1;
                                            LOC2(store,24,33, STOREDIM, STOREDIM) += Qtempy * x_24_18_0 + WQtempy * x_24_18_1 +  3 * CDtemp * ( x_24_8_0 - ABcom * x_24_8_1) +  2 * ABCDtemp * x_10_18_1;
                                            LOC2(store,25,33, STOREDIM, STOREDIM) += Qtempy * x_25_18_0 + WQtempy * x_25_18_1 +  3 * CDtemp * ( x_25_8_0 - ABcom * x_25_8_1) + ABCDtemp * x_14_18_1;
                                            LOC2(store,26,33, STOREDIM, STOREDIM) += Qtempy * x_26_18_0 + WQtempy * x_26_18_1 +  3 * CDtemp * ( x_26_8_0 - ABcom * x_26_8_1);
                                            LOC2(store,27,33, STOREDIM, STOREDIM) += Qtempy * x_27_18_0 + WQtempy * x_27_18_1 +  3 * CDtemp * ( x_27_8_0 - ABcom * x_27_8_1);
                                            LOC2(store,28,33, STOREDIM, STOREDIM) += Qtempy * x_28_18_0 + WQtempy * x_28_18_1 +  3 * CDtemp * ( x_28_8_0 - ABcom * x_28_8_1) + ABCDtemp * x_17_18_1;
                                            LOC2(store,29,33, STOREDIM, STOREDIM) += Qtempy * x_29_18_0 + WQtempy * x_29_18_1 +  3 * CDtemp * ( x_29_8_0 - ABcom * x_29_8_1) +  3 * ABCDtemp * x_12_18_1;
                                            LOC2(store,30,33, STOREDIM, STOREDIM) += Qtempy * x_30_18_0 + WQtempy * x_30_18_1 +  3 * CDtemp * ( x_30_8_0 - ABcom * x_30_8_1) +  3 * ABCDtemp * x_15_18_1;
                                            LOC2(store,31,33, STOREDIM, STOREDIM) += Qtempy * x_31_18_0 + WQtempy * x_31_18_1 +  3 * CDtemp * ( x_31_8_0 - ABcom * x_31_8_1) + ABCDtemp * x_19_18_1;
                                            LOC2(store,32,33, STOREDIM, STOREDIM) += Qtempy * x_32_18_0 + WQtempy * x_32_18_1 +  3 * CDtemp * ( x_32_8_0 - ABcom * x_32_8_1);
                                            LOC2(store,33,33, STOREDIM, STOREDIM) += Qtempy * x_33_18_0 + WQtempy * x_33_18_1 +  3 * CDtemp * ( x_33_8_0 - ABcom * x_33_8_1) +  4 * ABCDtemp * x_18_18_1;
                                            LOC2(store,34,33, STOREDIM, STOREDIM) += Qtempy * x_34_18_0 + WQtempy * x_34_18_1 +  3 * CDtemp * ( x_34_8_0 - ABcom * x_34_8_1);
                                            LOC2(store,20,34, STOREDIM, STOREDIM) += Qtempz * x_20_19_0 + WQtempz * x_20_19_1 +  3 * CDtemp * ( x_20_9_0 - ABcom * x_20_9_1);
                                            LOC2(store,21,34, STOREDIM, STOREDIM) += Qtempz * x_21_19_0 + WQtempz * x_21_19_1 +  3 * CDtemp * ( x_21_9_0 - ABcom * x_21_9_1) +  2 * ABCDtemp * x_13_19_1;
                                            LOC2(store,22,34, STOREDIM, STOREDIM) += Qtempz * x_22_19_0 + WQtempz * x_22_19_1 +  3 * CDtemp * ( x_22_9_0 - ABcom * x_22_9_1) +  2 * ABCDtemp * x_15_19_1;
                                            LOC2(store,23,34, STOREDIM, STOREDIM) += Qtempz * x_23_19_0 + WQtempz * x_23_19_1 +  3 * CDtemp * ( x_23_9_0 - ABcom * x_23_9_1) + ABCDtemp * x_11_19_1;
                                            LOC2(store,24,34, STOREDIM, STOREDIM) += Qtempz * x_24_19_0 + WQtempz * x_24_19_1 +  3 * CDtemp * ( x_24_9_0 - ABcom * x_24_9_1) + ABCDtemp * x_12_19_1;
                                            LOC2(store,25,34, STOREDIM, STOREDIM) += Qtempz * x_25_19_0 + WQtempz * x_25_19_1 +  3 * CDtemp * ( x_25_9_0 - ABcom * x_25_9_1) +  2 * ABCDtemp * x_10_19_1;
                                            LOC2(store,26,34, STOREDIM, STOREDIM) += Qtempz * x_26_19_0 + WQtempz * x_26_19_1 +  3 * CDtemp * ( x_26_9_0 - ABcom * x_26_9_1) + ABCDtemp * x_17_19_1;
                                            LOC2(store,27,34, STOREDIM, STOREDIM) += Qtempz * x_27_19_0 + WQtempz * x_27_19_1 +  3 * CDtemp * ( x_27_9_0 - ABcom * x_27_9_1) +  3 * ABCDtemp * x_14_19_1;
                                            LOC2(store,28,34, STOREDIM, STOREDIM) += Qtempz * x_28_19_0 + WQtempz * x_28_19_1 +  3 * CDtemp * ( x_28_9_0 - ABcom * x_28_9_1);
                                            LOC2(store,29,34, STOREDIM, STOREDIM) += Qtempz * x_29_19_0 + WQtempz * x_29_19_1 +  3 * CDtemp * ( x_29_9_0 - ABcom * x_29_9_1);
                                            LOC2(store,30,34, STOREDIM, STOREDIM) += Qtempz * x_30_19_0 + WQtempz * x_30_19_1 +  3 * CDtemp * ( x_30_9_0 - ABcom * x_30_9_1) + ABCDtemp * x_18_19_1;
                                            LOC2(store,31,34, STOREDIM, STOREDIM) += Qtempz * x_31_19_0 + WQtempz * x_31_19_1 +  3 * CDtemp * ( x_31_9_0 - ABcom * x_31_9_1) +  3 * ABCDtemp * x_16_19_1;
                                            LOC2(store,32,34, STOREDIM, STOREDIM) += Qtempz * x_32_19_0 + WQtempz * x_32_19_1 +  3 * CDtemp * ( x_32_9_0 - ABcom * x_32_9_1);
                                            LOC2(store,33,34, STOREDIM, STOREDIM) += Qtempz * x_33_19_0 + WQtempz * x_33_19_1 +  3 * CDtemp * ( x_33_9_0 - ABcom * x_33_9_1);
                                            LOC2(store,34,34, STOREDIM, STOREDIM) += Qtempz * x_34_19_0 + WQtempz * x_34_19_1 +  3 * CDtemp * ( x_34_9_0 - ABcom * x_34_9_1) +  4 * ABCDtemp * x_19_19_1;
                                            
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
#endif
            break;
        }
    }  
}

__device__ void FmT_MP2(int MaxM, QUICKDouble X, QUICKDouble* YVerticalTemp)
{
    
    const QUICKDouble PIE4 = (QUICKDouble) PI/4.0 ;
    
    const QUICKDouble XINV = (QUICKDouble) 1.0 /X;
    const QUICKDouble E = (QUICKDouble) exp(-X);
    QUICKDouble WW1;
    
    if (X > 5.0 ) {
        if (X>15.0 ) {
            if (X>33.0 ) {
                WW1 = sqrt(PIE4 * XINV);
            }else {
                WW1 = (( 1.9623264149430E-01 *XINV-4.9695241464490E-01 )*XINV - \
                       6.0156581186481E-05 )*E + sqrt(PIE4*XINV);
            }
        }else if (X>10.0 ) {
            WW1 = (((-1.8784686463512E-01 *XINV+2.2991849164985E-01 )*XINV - \
                    4.9893752514047E-01 )*XINV-2.1916512131607E-05 )*E + sqrt(PIE4*XINV);
        }else {
            WW1 = (((((( 4.6897511375022E-01  *XINV-6.9955602298985E-01 )*XINV + \
                       5.3689283271887E-01 )*XINV-3.2883030418398E-01 )*XINV + \
                     2.4645596956002E-01 )*XINV-4.9984072848436E-01 )*XINV - \
                   3.1501078774085E-06 )*E + sqrt(PIE4*XINV);
        }
    }else if (X >1.0 ) {
        if (X>3.0 ) {
            QUICKDouble Y = (QUICKDouble) X - 4.0 ;
            QUICKDouble F1 = ((((((((((-2.62453564772299E-11 *Y+3.24031041623823E-10  )*Y- \
                                      3.614965656163E-09 )*Y+3.760256799971E-08 )*Y- \
                                    3.553558319675E-07 )*Y+3.022556449731E-06 )*Y- \
                                  2.290098979647E-05 )*Y+1.526537461148E-04 )*Y- \
                                8.81947375894379E-04 )*Y+4.33207949514611E-03 )*Y- \
                              1.75257821619926E-02 )*Y+5.28406320615584E-02 ;
            WW1 = (X+X)*F1+E;
        }else {
            QUICKDouble Y = (QUICKDouble) X - 2.0 ;
            QUICKDouble F1 = ((((((((((-1.61702782425558E-10 *Y+1.96215250865776E-09  )*Y- \
                                      2.14234468198419E-08  )*Y+2.17216556336318E-07  )*Y- \
                                    1.98850171329371E-06  )*Y+1.62429321438911E-05  )*Y- \
                                  1.16740298039895E-04  )*Y+7.24888732052332E-04  )*Y- \
                                3.79490003707156E-03  )*Y+1.61723488664661E-02  )*Y- \
                              5.29428148329736E-02  )*Y+1.15702180856167E-01 ;
            WW1 = (X+X)*F1+E;
        }
        
    }else if (X > 3.0E-7 ) {
        QUICKDouble F1 =(((((((( -8.36313918003957E-08 *X+1.21222603512827E-06  )*X- \
                               1.15662609053481E-05  )*X+9.25197374512647E-05  )*X- \
                             6.40994113129432E-04  )*X+3.78787044215009E-03  )*X- \
                           1.85185172458485E-02  )*X+7.14285713298222E-02  )*X- \
                         1.99999999997023E-01  )*X+3.33333333333318E-01 ;
        WW1 = (X+X)*F1+E;
    }else {
        WW1 = (1.0 -X)/(QUICKDouble)(2.0 * MaxM+1);
    }
    if (X > 3.0E-7 ) {
        LOC3(YVerticalTemp, 0, 0, 0, VDIM1, VDIM2, VDIM3) = WW1;
        for (int m = 1; m<= MaxM; m++) {
            LOC3(YVerticalTemp, 0, 0, m, VDIM1, VDIM2, VDIM3) = (((2*m-1)*LOC3(YVerticalTemp, 0, 0, m-1, VDIM1, VDIM2, VDIM3))- E)*0.5*XINV;
        }
    }else {
        LOC3(YVerticalTemp, 0, 0, MaxM, VDIM1, VDIM2, VDIM3) = WW1;
        for (int m = MaxM-1; m >=0; m--) {
            LOC3(YVerticalTemp, 0, 0, m, VDIM1, VDIM2, VDIM3) = (2.0 * X * LOC3(YVerticalTemp, 0, 0, m+1, VDIM1, VDIM2, VDIM3) + E) / (QUICKDouble)(m*2+1);
        }
    }
    return;
}


__device__ QUICKDouble hrrwhole_MP2(int I, int J, int K, int L, \
                                int III, int JJJ, int KKK, int LLL, int IJKLTYPE, QUICKDouble* store, \
                                QUICKDouble RAx,QUICKDouble RAy,QUICKDouble RAz, \
                                QUICKDouble RBx,QUICKDouble RBy,QUICKDouble RBz, \
                                QUICKDouble RCx,QUICKDouble RCy,QUICKDouble RCz, \
                                QUICKDouble RDx,QUICKDouble RDy,QUICKDouble RDz)
{
	//printf("at the beginning of hrrwhole_MP2\n");
    QUICKDouble Y;
#ifdef CUDA_SP
	printf("def CUDA_SP\n")
    int NAx = LOC2(devSim_MP2.KLMN,0,III-1,3,devSim_MP2.nbasis);
    int NAy = LOC2(devSim_MP2.KLMN,1,III-1,3,devSim_MP2.nbasis);
    int NAz = LOC2(devSim_MP2.KLMN,2,III-1,3,devSim_MP2.nbasis);
   	//printf("NAx, NAy, NAz are done\n");
	
    int NBx = LOC2(devSim_MP2.KLMN,0,JJJ-1,3,devSim_MP2.nbasis);
    int NBy = LOC2(devSim_MP2.KLMN,1,JJJ-1,3,devSim_MP2.nbasis);
    int NBz = LOC2(devSim_MP2.KLMN,2,JJJ-1,3,devSim_MP2.nbasis);
	//printf("NBx, NBy, NBz are done\n");    

    int NCx = LOC2(devSim_MP2.KLMN,0,KKK-1,3,devSim_MP2.nbasis);
    int NCy = LOC2(devSim_MP2.KLMN,1,KKK-1,3,devSim_MP2.nbasis);
    int NCz = LOC2(devSim_MP2.KLMN,2,KKK-1,3,devSim_MP2.nbasis);
	//printf("NCx, NCy, NCz are done\n");
    
    int NDx = LOC2(devSim_MP2.KLMN,0,LLL-1,3,devSim_MP2.nbasis);
    int NDy = LOC2(devSim_MP2.KLMN,1,LLL-1,3,devSim_MP2.nbasis);
    int NDz = LOC2(devSim_MP2.KLMN,2,LLL-1,3,devSim_MP2.nbasis);
	//printf("NDx, NDy, NDz are done\n");    

    
    int MA = LOC3(devTrans_MP2, NAx, NAy, NAz, TRANSDIM, TRANSDIM, TRANSDIM);
    int MB = LOC3(devTrans_MP2, NBx, NBy, NBz, TRANSDIM, TRANSDIM, TRANSDIM);
    int MC = LOC3(devTrans_MP2, NCx, NCy, NCz, TRANSDIM, TRANSDIM, TRANSDIM);
    int MD = LOC3(devTrans_MP2, NDx, NDy, NDz, TRANSDIM, TRANSDIM, TRANSDIM);
    
	//printf("inside hrrwhole_MP2, IJKLTYPE is %d\n", IJKLTYPE);

    switch (IJKLTYPE) {
        case 0:
        case 10:
        case 1000:
        case 1010:
        {
            Y = (QUICKDouble) LOC2(store, MA-1, MC-1, STOREDIM, STOREDIM);
            break;
        }
        case 2000:
        case 20:
        case 2010:
        case 1020:
        case 2020:
        {
            Y = (QUICKDouble) LOC2(store, MA-1, MC-1, STOREDIM, STOREDIM) * devSim_MP2.cons[III-1] * devSim_MP2.cons[JJJ-1] * devSim_MP2.cons[KKK-1] * devSim_MP2.cons[LLL-1];
            break;
        }
        case 100:
        {
            if (NBx != 0) {
                Y = (QUICKDouble) LOC2(store, MB-1, 0, STOREDIM, STOREDIM) + (RAx-RBx)*LOC2(store, 0, 0, STOREDIM, STOREDIM);
            }else if (NBy != 0) {
                Y = (QUICKDouble) LOC2(store, MB-1, 0, STOREDIM, STOREDIM) + (RAy-RBy)*LOC2(store, 0, 0, STOREDIM, STOREDIM);
            }else if (NBz != 0) {
                Y = (QUICKDouble) LOC2(store, MB-1, 0, STOREDIM, STOREDIM) + (RAz-RBz)*LOC2(store, 0, 0, STOREDIM, STOREDIM);
            }
            break;
        }
        case 110:
        {
            
            if (NBx != 0) {
                Y = (QUICKDouble) LOC2(store, MB-1, MC-1, STOREDIM, STOREDIM) + (RAx-RBx)*LOC2(store, 0, MC-1, STOREDIM, STOREDIM);
            }else if (NBy != 0) {
                Y = (QUICKDouble) LOC2(store, MB-1, MC-1, STOREDIM, STOREDIM) + (RAy-RBy)*LOC2(store, 0, MC-1, STOREDIM, STOREDIM);
            }else if (NBz != 0) {
                Y = (QUICKDouble) LOC2(store, MB-1, MC-1, STOREDIM, STOREDIM) + (RAz-RBz)*LOC2(store, 0, MC-1, STOREDIM, STOREDIM);
            }
            break;
        }
        case 101:
        {
            QUICKDouble Y1,Y2;
            if (NDx != 0) {
                QUICKDouble c = (QUICKDouble) (RCx - RDx);
                Y1 = (QUICKDouble) LOC2(store, MB-1, MD-1 , STOREDIM, STOREDIM) + c * LOC2(store, MB-1,  0, STOREDIM, STOREDIM);
                Y2 = (QUICKDouble) LOC2(store,    0, MD-1 , STOREDIM, STOREDIM) + c * LOC2(store,    0,  0, STOREDIM, STOREDIM);
            }else if (NDy != 0) {
                QUICKDouble c = (QUICKDouble) (RCy - RDy);
                Y1 = (QUICKDouble) LOC2(store, MB-1, MD-1 , STOREDIM, STOREDIM) + c * LOC2(store, MB-1,  0, STOREDIM, STOREDIM);
                Y2 = (QUICKDouble) LOC2(store,    0, MD-1 , STOREDIM, STOREDIM) + c * LOC2(store,    0,  0, STOREDIM, STOREDIM);
            }else if (NDz != 0) {
                QUICKDouble c = (QUICKDouble) (RCz - RDz);
                Y1 = (QUICKDouble) LOC2(store, MB-1, MD-1 , STOREDIM, STOREDIM) + c * LOC2(store, MB-1,  0, STOREDIM, STOREDIM);
                Y2 = (QUICKDouble) LOC2(store,    0, MD-1 , STOREDIM, STOREDIM) + c * LOC2(store,    0,  0, STOREDIM, STOREDIM);
            }
            
            if (NBx != 0) {
                Y = Y1 + (RAx-RBx)*Y2;
            }else if (NBy != 0) {
                Y = Y1 + (RAy-RBy)*Y2;
            }else if (NBz != 0) {
                Y = Y1 + (RAz-RBz)*Y2;
            }
            break;
        }
        case 111:
        {
            QUICKDouble Y1,Y2;
            int MCD = (int) LOC3(devTrans_MP2, NCx+NDx, NCy+NDy, NCz+NDz, TRANSDIM, TRANSDIM, TRANSDIM);
            if (NDx != 0) {
                QUICKDouble c = (QUICKDouble) (RCx - RDx);
                Y1 = (QUICKDouble) LOC2(store, MB-1, MCD-1 , STOREDIM, STOREDIM) + c * LOC2(store, MB-1,  MC-1, STOREDIM, STOREDIM);
                Y2 = (QUICKDouble) LOC2(store,    0, MCD-1 , STOREDIM, STOREDIM) + c * LOC2(store,    0,  MC-1, STOREDIM, STOREDIM);
            }else if (NDy != 0) {
                QUICKDouble c = (QUICKDouble) (RCy - RDy);
                Y1 = (QUICKDouble) LOC2(store, MB-1, MCD-1 , STOREDIM, STOREDIM) + c * LOC2(store, MB-1,  MC-1, STOREDIM, STOREDIM);
                Y2 = (QUICKDouble) LOC2(store,    0, MCD-1 , STOREDIM, STOREDIM) + c * LOC2(store,    0,  MC-1, STOREDIM, STOREDIM);
            }else if (NDz != 0) {
                QUICKDouble c = (QUICKDouble) (RCz - RDz);
                Y1 = (QUICKDouble) LOC2(store, MB-1, MCD-1 , STOREDIM, STOREDIM) + c * LOC2(store, MB-1,  MC-1, STOREDIM, STOREDIM);
                Y2 = (QUICKDouble) LOC2(store,    0, MCD-1 , STOREDIM, STOREDIM) + c * LOC2(store,    0,  MC-1, STOREDIM, STOREDIM);
            }
            
            if (NBx != 0) {
                Y = Y1 + (RAx-RBx)*Y2;
            }else if (NBy != 0) {
                Y = Y1 + (RAy-RBy)*Y2;
            }else if (NBz != 0) {
                Y = Y1 + (RAz-RBz)*Y2;
            }
            break;
        }
        case 1100:
        {
            int MAB = (int) LOC3(devTrans_MP2, NAx+NBx, NAy+NBy, NAz+NBz, TRANSDIM, TRANSDIM, TRANSDIM);
            if (NBx != 0) {
                Y = (QUICKDouble) LOC2(store, MAB-1, 0 , STOREDIM, STOREDIM) + (RAx-RBx)*LOC2(store, MA-1, 0, STOREDIM, STOREDIM);
            }else if (NBy != 0) {
                Y = (QUICKDouble) LOC2(store, MAB-1, 0 , STOREDIM, STOREDIM) + (RAy-RBy)*LOC2(store, MA-1, 0, STOREDIM, STOREDIM);
            }else if (NBz != 0) {
                Y = (QUICKDouble) LOC2(store, MAB-1, 0 , STOREDIM, STOREDIM) + (RAz-RBz)*LOC2(store, MA-1, 0, STOREDIM, STOREDIM);
            }
            break;
        }
        case 1110:
        {   
            int MAB = (int) LOC3(devTrans_MP2, NAx+NBx, NAy+NBy, NAz+NBz, TRANSDIM, TRANSDIM, TRANSDIM);
            
            if (NBx != 0) {
                Y = (QUICKDouble) LOC2(store, MAB-1, MC-1 , STOREDIM, STOREDIM) + (RAx-RBx)*LOC2(store, MA-1, MC-1, STOREDIM, STOREDIM);
            }else if (NBy != 0) {
                Y = (QUICKDouble) LOC2(store, MAB-1, MC-1 , STOREDIM, STOREDIM) + (RAy-RBy)*LOC2(store, MA-1, MC-1, STOREDIM, STOREDIM);
            }else if (NBz != 0) {
                Y = (QUICKDouble) LOC2(store, MAB-1, MC-1 , STOREDIM, STOREDIM) + (RAz-RBz)*LOC2(store, MA-1, MC-1, STOREDIM, STOREDIM);
            }
            break;
        }
        case 1101:
        {
            QUICKDouble Y1,Y2;
            int MAB = (int) LOC3(devTrans_MP2, NAx+NBx, NAy+NBy, NAz+NBz, TRANSDIM, TRANSDIM, TRANSDIM);
            if (NDx != 0) {
                QUICKDouble c = (QUICKDouble) (RCx - RDx);
                Y1 = (QUICKDouble) LOC2(store, MAB-1, MD-1 , STOREDIM, STOREDIM) + c * LOC2(store, MAB-1,  0, STOREDIM, STOREDIM);
                Y2 = (QUICKDouble) LOC2(store,  MA-1, MD-1 , STOREDIM, STOREDIM) + c * LOC2(store,  MA-1,  0, STOREDIM, STOREDIM);
            }else if (NDy != 0) {
                QUICKDouble c = (QUICKDouble) (RCy - RDy);
                Y1 = (QUICKDouble) LOC2(store, MAB-1, MD-1 , STOREDIM, STOREDIM) + c * LOC2(store, MAB-1,  0, STOREDIM, STOREDIM);
                Y2 = (QUICKDouble) LOC2(store,  MA-1, MD-1 , STOREDIM, STOREDIM) + c * LOC2(store,  MA-1,  0, STOREDIM, STOREDIM);
            }else if (NDz != 0) {
                QUICKDouble c = (QUICKDouble) (RCz - RDz);
                Y1 = (QUICKDouble) LOC2(store, MAB-1, MD-1 , STOREDIM, STOREDIM) + c * LOC2(store, MAB-1,  0, STOREDIM, STOREDIM);
                Y2 = (QUICKDouble) LOC2(store,  MA-1, MD-1 , STOREDIM, STOREDIM) + c * LOC2(store,  MA-1,  0, STOREDIM, STOREDIM);
            }
            
            if (NBx != 0) {
                Y = Y1 + (RAx-RBx)*Y2;
            }else if (NBy != 0) {
                Y = Y1 + (RAy-RBy)*Y2;
            }else if (NBz != 0) {
                Y = Y1 + (RAz-RBz)*Y2;
            }
            break;
        }
        case 1111:
        {
            QUICKDouble Y1,Y2;
            int MAB = (int) LOC3(devTrans_MP2, NAx+NBx, NAy+NBy, NAz+NBz, TRANSDIM, TRANSDIM, TRANSDIM);
            int MCD = (int) LOC3(devTrans_MP2, NCx+NDx, NCy+NDy, NCz+NDz, TRANSDIM, TRANSDIM, TRANSDIM);
            
            if (NDx != 0) {
                QUICKDouble c = (QUICKDouble) (RCx - RDx);
                Y1 = (QUICKDouble) LOC2(store, MAB-1, MCD-1 , STOREDIM, STOREDIM) + c * LOC2(store, MAB-1, MC-1, STOREDIM, STOREDIM);
                Y2 = (QUICKDouble) LOC2(store,  MA-1, MCD-1 , STOREDIM, STOREDIM) + c * LOC2(store,  MA-1, MC-1, STOREDIM, STOREDIM);
            }else if (NDy != 0) {
                QUICKDouble c = (QUICKDouble) (RCy - RDy);
                Y1 = (QUICKDouble) LOC2(store, MAB-1, MCD-1 , STOREDIM, STOREDIM) + c * LOC2(store, MAB-1, MC-1, STOREDIM, STOREDIM);
                Y2 = (QUICKDouble) LOC2(store,  MA-1, MCD-1 , STOREDIM, STOREDIM) + c * LOC2(store,  MA-1, MC-1, STOREDIM, STOREDIM);
            }else if (NDz != 0) {
                QUICKDouble c = (QUICKDouble) (RCz - RDz);
                Y1 = (QUICKDouble) LOC2(store, MAB-1, MCD-1 , STOREDIM, STOREDIM) + c * LOC2(store, MAB-1, MC-1, STOREDIM, STOREDIM);
                Y2 = (QUICKDouble) LOC2(store,  MA-1, MCD-1 , STOREDIM, STOREDIM) + c * LOC2(store,  MA-1, MC-1, STOREDIM, STOREDIM);
            }
            
            if (NBx != 0) {
                Y = Y1 + (RAx-RBx)*Y2;
            }else if (NBy != 0) {
                Y = Y1 + (RAy-RBy)*Y2;
            }else if (NBz != 0) {
                Y = Y1 + (RAz-RBz)*Y2;
            }
            
            break;
        }
        case 1:
        {
            if (NDx != 0) {
                Y = (QUICKDouble) LOC2(store, 0, MD-1, STOREDIM, STOREDIM) + (RCx-RDx)*LOC2(store, 0, 0, STOREDIM, STOREDIM);
            }else if (NDy != 0) {
                Y = (QUICKDouble) LOC2(store, 0, MD-1, STOREDIM, STOREDIM) + (RCy-RDy)*LOC2(store, 0, 0, STOREDIM, STOREDIM);
            }else if (NDz != 0) {
                Y = (QUICKDouble) LOC2(store, 0, MD-1, STOREDIM, STOREDIM) + (RCz-RDz)*LOC2(store, 0, 0, STOREDIM, STOREDIM);
            }
            break;
        }
        case 11:
        {
            int MCD = (int) LOC3(devTrans_MP2, NCx+NDx, NCy+NDy, NCz+NDz, TRANSDIM, TRANSDIM, TRANSDIM);
            if (NDx != 0) {
                Y = (QUICKDouble) LOC2(store, 0, MCD-1, STOREDIM, STOREDIM) + (RCx-RDx)*LOC2(store, 0, MC-1, STOREDIM, STOREDIM);
            }else if (NDy != 0) {
                Y = (QUICKDouble) LOC2(store, 0, MCD-1, STOREDIM, STOREDIM) + (RCy-RDy)*LOC2(store, 0, MC-1, STOREDIM, STOREDIM);
            }else if (NDz != 0) {
                Y = (QUICKDouble) LOC2(store, 0, MCD-1, STOREDIM, STOREDIM) + (RCz-RDz)*LOC2(store, 0, MC-1, STOREDIM, STOREDIM);
            }
            break;
        }
        case 1001:
        {   
            if (NDx != 0) {
                Y = (QUICKDouble) LOC2(store, MA-1, MD-1, STOREDIM, STOREDIM) + (RCx-RDx)*LOC2(store, MA-1, 0, STOREDIM, STOREDIM);
            }else if (NDy != 0) {
                Y = (QUICKDouble) LOC2(store, MA-1, MD-1, STOREDIM, STOREDIM) + (RCy-RDy)*LOC2(store, MA-1, 0, STOREDIM, STOREDIM);
            }else if (NDz != 0) {
                Y = (QUICKDouble) LOC2(store, MA-1, MD-1, STOREDIM, STOREDIM) + (RCz-RDz)*LOC2(store, MA-1, 0, STOREDIM, STOREDIM);
            }
        }
        case 1011:
        {
            int MCD = (int) LOC3(devTrans_MP2, NCx+NDx, NCy+NDy, NCz+NDz, TRANSDIM, TRANSDIM, TRANSDIM);
            if (NDx != 0) {
                Y = (QUICKDouble) LOC2(store, MA-1, MCD-1, STOREDIM, STOREDIM) + (RCx-RDx)*LOC2(store, MA-1, MC-1, STOREDIM, STOREDIM);
            }else if (NDy != 0) {
                Y = (QUICKDouble) LOC2(store, MA-1, MCD-1, STOREDIM, STOREDIM) + (RCy-RDy)*LOC2(store, MA-1, MC-1, STOREDIM, STOREDIM);
            }else if (NDz != 0) {
                Y = (QUICKDouble) LOC2(store, MA-1, MCD-1, STOREDIM, STOREDIM) + (RCz-RDz)*LOC2(store, MA-1, MC-1, STOREDIM, STOREDIM);
            }
            break;
        }
        default:
            break;
    }
#else
	//unsigned int offside = blockIdx.x*blockDim.x+threadIdx.x;
 	  
    int angularL[8], angularR[8];
    QUICKDouble coefAngularL[8], coefAngularR[8];
	Y = (QUICKDouble) 0.0;
		
    int numAngularL = lefthrr_MP2(RAx, RAy, RAz, RBx, RBy, RBz, 
                              LOC2(devSim_MP2.KLMN,0,III-1,3,devSim_MP2.nbasis), LOC2(devSim_MP2.KLMN,1,III-1,3,devSim_MP2.nbasis), LOC2(devSim_MP2.KLMN,2,III-1,3,devSim_MP2.nbasis),
                              LOC2(devSim_MP2.KLMN,0,JJJ-1,3,devSim_MP2.nbasis), LOC2(devSim_MP2.KLMN,1,JJJ-1,3,devSim_MP2.nbasis), LOC2(devSim_MP2.KLMN,2,JJJ-1,3,devSim_MP2.nbasis),
                              J, coefAngularL, angularL);
    int numAngularR = lefthrr_MP2(RCx, RCy, RCz, RDx, RDy, RDz,
                              LOC2(devSim_MP2.KLMN,0,KKK-1,3,devSim_MP2.nbasis), LOC2(devSim_MP2.KLMN,1,KKK-1,3,devSim_MP2.nbasis), LOC2(devSim_MP2.KLMN,2,KKK-1,3,devSim_MP2.nbasis),
                              LOC2(devSim_MP2.KLMN,0,LLL-1,3,devSim_MP2.nbasis), LOC2(devSim_MP2.KLMN,1,LLL-1,3,devSim_MP2.nbasis), LOC2(devSim_MP2.KLMN,2,LLL-1,3,devSim_MP2.nbasis),
                              L, coefAngularR, angularR);
    
    //printf(//"after calling lefthrr_MP2,coefAngularR[0] is %lf \n",coefAngularR[0]);

    for (int i = 0; i<numAngularL; i++) {
        for (int j = 0; j<numAngularR; j++) {
    		//printf("coefAngularL[%d] is %lf, coefAngularR[%d] is %lf, angularL[%d] is %d, angularR[%d]is %d, STOREDIM is %d, LOC2 is %lf, LOC2_IND is %d \n",\
i,coefAngularL[i],j,coefAngularR[j], i, angularL[i], j, angularR[j], STOREDIM, LOC2(store, angularL[i]-1, angularR[j]-1 , STOREDIM, STOREDIM), (angularL[i]-1)+(angularR[j]-1)*STOREDIM); 
			Y += coefAngularL[i] * coefAngularR[j] * LOC2(store, angularL[i]-1, angularR[j]-1 , STOREDIM, STOREDIM);
		}
    }
	//printf("after two for loop statement, Y is %lf\n",Y);
    
    Y = Y * devSim_MP2.cons[III-1] * devSim_MP2.cons[JJJ-1] * devSim_MP2.cons[KKK-1] * devSim_MP2.cons[LLL-1];

	//get Y
	//printf("III, JJJ, KKK, LLL, and Y, offside are %d %d %d %d %lf %d\n", III, JJJ, KKK, LLL, Y, offside); 
#endif
    return Y;
}  


#ifndef CUDA_SP
__device__ int lefthrr_MP2(QUICKDouble RAx, QUICKDouble RAy, QUICKDouble RAz, 
                       QUICKDouble RBx, QUICKDouble RBy, QUICKDouble RBz,
                       int KLMNAx, int KLMNAy, int KLMNAz,
                       int KLMNBx, int KLMNBy, int KLMNBz,
                       int IJTYPE,QUICKDouble* coefAngularL, int* angularL)
{           

    int numAngularL;
    switch (IJTYPE) {
            
        case 0:
        {
            numAngularL = 1;
            coefAngularL[0] = 1.0;
            angularL[0] = (int) LOC3(devTrans_MP2, KLMNAx, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            //printf("in lefthrr_MP2 case 0, LOC3_ind is %d, angularL[0] is %d\n", KLMNAz+(KLMNAy+KLMNAx*TRANSDIM)*TRANSDIM, angularL[0]);
			break;
        }
        case 1:
        {
            coefAngularL[0] = 1.0;
            numAngularL = 2;
            angularL[0] = (int) LOC3(devTrans_MP2, KLMNAx + KLMNBx, KLMNAy + KLMNBy, KLMNAz + KLMNBz, TRANSDIM, TRANSDIM, TRANSDIM);
            angularL[1] = (int) LOC3(devTrans_MP2, KLMNAx, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            
            if (KLMNBx != 0) {
                coefAngularL[1] = RAx-RBx;
            }else if(KLMNBy !=0 ){
                coefAngularL[1] = RAy-RBy;
            }else if (KLMNBz != 0) {
                coefAngularL[1] = RAz-RBz;
            }
			//printf("in lefthrr_MP2 case 1, LOC3_ind is %d, angularL[0] is %d, LOC3_ind is %d,angularL[1] is %d\n",(KLMNAz + KLMNBz)+((KLMNAy + KLMNBy)+TRANSDIM*(KLMNAx + KLMNBx))*TRANSDIM, angularL[0],  KLMNAz+(KLMNAy+KLMNAx*TRANSDIM)*TRANSDIM, angularL[1]);
            break;
        }
        case 2:
        {
            coefAngularL[0] = 1.0;
            angularL[0] = (int) LOC3(devTrans_MP2, KLMNAx + KLMNBx, KLMNAy + KLMNBy, KLMNAz + KLMNBz, TRANSDIM, TRANSDIM, TRANSDIM);
            
            if (KLMNBx == 2) {
                numAngularL = 3;
                QUICKDouble tmp = RAx - RBx;
                coefAngularL[1] = 2 * tmp;
                angularL[1] = (int) LOC3(devTrans_MP2, KLMNAx+1, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
                coefAngularL[2]= tmp * tmp;
                angularL[2] = (int) LOC3(devTrans_MP2, KLMNAx, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            }else if(KLMNBy == 2) {
                numAngularL = 3;
                QUICKDouble tmp = RAy - RBy;
                coefAngularL[1] = 2 * tmp;
                angularL[1] = (int) LOC3(devTrans_MP2, KLMNAx, KLMNAy+1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
                coefAngularL[2]= tmp * tmp;
                angularL[2] = (int) LOC3(devTrans_MP2, KLMNAx, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            }else if (KLMNBz == 2 ){
                numAngularL = 3;
                QUICKDouble tmp = RAz - RBz;
                coefAngularL[1] = 2 * tmp;
                angularL[1] = (int) LOC3(devTrans_MP2, KLMNAx, KLMNAy, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
                coefAngularL[2]= tmp * tmp;
                angularL[2] = (int) LOC3(devTrans_MP2, KLMNAx, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            }else if (KLMNBx == 1 && KLMNBy == 1){
                numAngularL = 4;
                coefAngularL[1] = RAx - RBx;
                coefAngularL[2] = RAy - RBy;
                coefAngularL[3] = (RAx - RBx) * (RAy - RBy);
                angularL[1] = (int) LOC3(devTrans_MP2, KLMNAx, KLMNAy+1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
                angularL[2] = (int) LOC3(devTrans_MP2, KLMNAx+1, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
                angularL[3] = (int) LOC3(devTrans_MP2, KLMNAx, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
                
            }else if (KLMNBx == 1 && KLMNBz == 1) {
                numAngularL = 4;
                coefAngularL[1] = RAx - RBx;
                coefAngularL[2] = RAz - RBz;
                coefAngularL[3] = (RAx - RBx) * (RAz - RBz);
                angularL[1] = (int) LOC3(devTrans_MP2, KLMNAx, KLMNAy, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
                angularL[2] = (int) LOC3(devTrans_MP2, KLMNAx+1, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
                angularL[3] = (int) LOC3(devTrans_MP2, KLMNAx, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            }else if (KLMNBy == 1 && KLMNBz == 1) {
                numAngularL = 4;
                coefAngularL[1] = RAy - RBy;
                coefAngularL[2] = RAz - RBz;
                coefAngularL[3] = (RAy - RBy) * (RAz - RBz);
                angularL[1] = (int) LOC3(devTrans_MP2, KLMNAx, KLMNAy, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
                angularL[2] = (int) LOC3(devTrans_MP2, KLMNAx, KLMNAy+1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
                angularL[3] = (int) LOC3(devTrans_MP2, KLMNAx, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            }
            break;
        }
        case 3:
        {
            coefAngularL[0] = 1.0;
            angularL[0] = (int) LOC3(devTrans_MP2, KLMNAx + KLMNBx, KLMNAy + KLMNBy, KLMNAz + KLMNBz, TRANSDIM, TRANSDIM, TRANSDIM);
            if (KLMNBx == 3) {
                numAngularL = 4;
                QUICKDouble tmp = RAx - RBx;
                
                coefAngularL[1] = 3 * tmp;
                coefAngularL[2] = 3 * tmp * tmp;
                coefAngularL[3] = tmp * tmp * tmp;
                
                angularL[1] = (int) LOC3(devTrans_MP2, KLMNAx+2, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
                angularL[2] = (int) LOC3(devTrans_MP2, KLMNAx+1, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
                angularL[3] = (int) LOC3(devTrans_MP2, KLMNAx, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            }else if (KLMNBy == 3) {
                numAngularL = 4;
                QUICKDouble tmp = RAy - RBy;
                coefAngularL[1] = 3 * tmp;
                coefAngularL[2] = 3 * tmp * tmp;
                coefAngularL[3] = tmp * tmp * tmp;
                
                angularL[1] = (int) LOC3(devTrans_MP2, KLMNAx, KLMNAy+2, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
                angularL[2] = (int) LOC3(devTrans_MP2, KLMNAx, KLMNAy+1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
                angularL[3] = (int) LOC3(devTrans_MP2, KLMNAx, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            }else if (KLMNBz == 3) {
                numAngularL = 4;
                
                QUICKDouble tmp = RAz - RBz;
                coefAngularL[1] = 3 * tmp;
                coefAngularL[2] = 3 * tmp * tmp;
                coefAngularL[3] = tmp * tmp * tmp;
                
                angularL[1] = (int) LOC3(devTrans_MP2, KLMNAx, KLMNAy, KLMNAz+2, TRANSDIM, TRANSDIM, TRANSDIM);
                angularL[2] = (int) LOC3(devTrans_MP2, KLMNAx, KLMNAy, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
                angularL[3] = (int) LOC3(devTrans_MP2, KLMNAx, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            }else if (KLMNBx == 1 && KLMNBy ==2) {
                numAngularL = 6;
                QUICKDouble tmp = RAx - RBx;
                QUICKDouble tmp2 = RAy - RBy;
                
                coefAngularL[1] = tmp;
                coefAngularL[2] = 2 * tmp2;
                coefAngularL[3] = 2 * tmp * tmp2;
                coefAngularL[4] = tmp2 * tmp2;
                coefAngularL[5] = tmp * tmp2 * tmp2;
                
                angularL[1] = (int) LOC3(devTrans_MP2, KLMNAx, KLMNAy+2, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
                angularL[2] = (int) LOC3(devTrans_MP2, KLMNAx+1, KLMNAy+1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
                angularL[3] = (int) LOC3(devTrans_MP2, KLMNAx, KLMNAy+1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
                angularL[4] = (int) LOC3(devTrans_MP2, KLMNAx+1, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);   
                angularL[5] = (int) LOC3(devTrans_MP2, KLMNAx, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            }else if (KLMNBx == 1 && KLMNBz ==2) {
                numAngularL = 6;
                QUICKDouble tmp = RAx - RBx;
                QUICKDouble tmp2 = RAz - RBz;
                coefAngularL[1] = tmp;
                coefAngularL[2] = 2 * tmp2;
                coefAngularL[3] = 2 * tmp * tmp2;
                coefAngularL[4] = tmp2 * tmp2;
                coefAngularL[5] = tmp * tmp2 * tmp2;
                
                angularL[1] = (int) LOC3(devTrans_MP2, KLMNAx, KLMNAy, KLMNAz+2, TRANSDIM, TRANSDIM, TRANSDIM);
                angularL[2] = (int) LOC3(devTrans_MP2, KLMNAx+1, KLMNAy, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
                angularL[3] = (int) LOC3(devTrans_MP2, KLMNAx, KLMNAy, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
                angularL[4] = (int) LOC3(devTrans_MP2, KLMNAx+1, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);   
                angularL[5] = (int) LOC3(devTrans_MP2, KLMNAx, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            }else if (KLMNBy == 1 && KLMNBz ==2) {
                numAngularL = 6;
                QUICKDouble tmp = RAy - RBy;
                QUICKDouble tmp2 = RAz - RBz;
                coefAngularL[1] = tmp;
                coefAngularL[2] = 2 * tmp2;
                coefAngularL[3] = 2 * tmp * tmp2;
                coefAngularL[4] = tmp2 * tmp2;
                coefAngularL[5] = tmp * tmp2 * tmp2;
                
                angularL[1] = (int) LOC3(devTrans_MP2, KLMNAx, KLMNAy, KLMNAz+2, TRANSDIM, TRANSDIM, TRANSDIM);
                angularL[2] = (int) LOC3(devTrans_MP2, KLMNAx, KLMNAy+1, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
                angularL[3] = (int) LOC3(devTrans_MP2, KLMNAx, KLMNAy, KLMNAz+1, TRANSDIM, TRANSDIM, TRANSDIM);
                angularL[4] = (int) LOC3(devTrans_MP2, KLMNAx, KLMNAy+1, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);   
                angularL[5] = (int) LOC3(devTrans_MP2, KLMNAx, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            }else if (KLMNBx == 1 && KLMNBy == 1) {
                numAngularL = 8;
                QUICKDouble tmp = RAx - RBx;
                QUICKDouble tmp2 = RAy - RBy;
                QUICKDouble tmp3 = RAz - RBz;
                
                coefAngularL[1] = tmp;
                coefAngularL[2] = tmp2;
                coefAngularL[3] = tmp3;
                coefAngularL[4] = tmp * tmp2;
                coefAngularL[5] = tmp * tmp3;                
                coefAngularL[6] = tmp2 * tmp3;
                coefAngularL[7] = tmp * tmp2 * tmp3;
                
                angularL[1] = (int) LOC3(devTrans_MP2, KLMNAx, KLMNAx+1, KLMNAx+1, TRANSDIM, TRANSDIM, TRANSDIM);
                angularL[2] = (int) LOC3(devTrans_MP2, KLMNAx+1, KLMNAx, KLMNAx+1, TRANSDIM, TRANSDIM, TRANSDIM);
                angularL[3] = (int) LOC3(devTrans_MP2, KLMNAx+1, KLMNAx+1, KLMNAx, TRANSDIM, TRANSDIM, TRANSDIM);
                angularL[4] = (int) LOC3(devTrans_MP2, KLMNAx, KLMNAx, KLMNAx+1, TRANSDIM, TRANSDIM, TRANSDIM);
                angularL[5] = (int) LOC3(devTrans_MP2, KLMNAx, KLMNAx+1, KLMNAx, TRANSDIM, TRANSDIM, TRANSDIM);
                angularL[6] = (int) LOC3(devTrans_MP2, KLMNAx+1, KLMNAx, KLMNAx, TRANSDIM, TRANSDIM, TRANSDIM);
                angularL[7] = (int) LOC3(devTrans_MP2, KLMNAx, KLMNAy, KLMNAz, TRANSDIM, TRANSDIM, TRANSDIM);
            }
            break;
            
        }
    }
    return numAngularL;
}

#endif

void upload_para_to_const_MP2(){
    
    int trans[TRANSDIM*TRANSDIM*TRANSDIM];
    // Data to trans
    {
        LOC3(trans, 0, 0, 0, TRANSDIM, TRANSDIM, TRANSDIM) =   1;
        LOC3(trans, 0, 0, 1, TRANSDIM, TRANSDIM, TRANSDIM) =   4;
        LOC3(trans, 0, 0, 2, TRANSDIM, TRANSDIM, TRANSDIM) =  10;
        LOC3(trans, 0, 0, 3, TRANSDIM, TRANSDIM, TRANSDIM) =  20;
        LOC3(trans, 0, 0, 4, TRANSDIM, TRANSDIM, TRANSDIM) =  35;
        LOC3(trans, 0, 0, 5, TRANSDIM, TRANSDIM, TRANSDIM) =  56;
        LOC3(trans, 0, 0, 6, TRANSDIM, TRANSDIM, TRANSDIM) =  84;
        LOC3(trans, 0, 0, 7, TRANSDIM, TRANSDIM, TRANSDIM) = 120;
        LOC3(trans, 0, 1, 0, TRANSDIM, TRANSDIM, TRANSDIM) =   3;
        LOC3(trans, 0, 1, 1, TRANSDIM, TRANSDIM, TRANSDIM) =   6;
        LOC3(trans, 0, 1, 2, TRANSDIM, TRANSDIM, TRANSDIM) =  17;
        LOC3(trans, 0, 1, 3, TRANSDIM, TRANSDIM, TRANSDIM) =  32;
        LOC3(trans, 0, 1, 4, TRANSDIM, TRANSDIM, TRANSDIM) =  48;
        LOC3(trans, 0, 1, 5, TRANSDIM, TRANSDIM, TRANSDIM) =  67;
        LOC3(trans, 0, 1, 6, TRANSDIM, TRANSDIM, TRANSDIM) = 100;
        LOC3(trans, 0, 2, 0, TRANSDIM, TRANSDIM, TRANSDIM) =   9;
        LOC3(trans, 0, 2, 1, TRANSDIM, TRANSDIM, TRANSDIM) =  16;
        LOC3(trans, 0, 2, 2, TRANSDIM, TRANSDIM, TRANSDIM) =  23;
        LOC3(trans, 0, 2, 3, TRANSDIM, TRANSDIM, TRANSDIM) =  42;
        LOC3(trans, 0, 2, 4, TRANSDIM, TRANSDIM, TRANSDIM) =  73;
        LOC3(trans, 0, 2, 5, TRANSDIM, TRANSDIM, TRANSDIM) = 106;
        LOC3(trans, 0, 3, 0, TRANSDIM, TRANSDIM, TRANSDIM) =  19;
        LOC3(trans, 0, 3, 1, TRANSDIM, TRANSDIM, TRANSDIM) =  31;
        LOC3(trans, 0, 3, 2, TRANSDIM, TRANSDIM, TRANSDIM) =  43;
        LOC3(trans, 0, 3, 3, TRANSDIM, TRANSDIM, TRANSDIM) =  79;
        LOC3(trans, 0, 3, 4, TRANSDIM, TRANSDIM, TRANSDIM) = 112;
        LOC3(trans, 0, 4, 0, TRANSDIM, TRANSDIM, TRANSDIM) =  34;
        LOC3(trans, 0, 4, 1, TRANSDIM, TRANSDIM, TRANSDIM) =  49;
        LOC3(trans, 0, 4, 2, TRANSDIM, TRANSDIM, TRANSDIM) =  74;
        LOC3(trans, 0, 4, 3, TRANSDIM, TRANSDIM, TRANSDIM) = 113;
        LOC3(trans, 0, 5, 0, TRANSDIM, TRANSDIM, TRANSDIM) =  55;
        LOC3(trans, 0, 5, 1, TRANSDIM, TRANSDIM, TRANSDIM) =  68;
        LOC3(trans, 0, 5, 2, TRANSDIM, TRANSDIM, TRANSDIM) = 107;
        LOC3(trans, 0, 6, 0, TRANSDIM, TRANSDIM, TRANSDIM) =  83;
        LOC3(trans, 0, 6, 1, TRANSDIM, TRANSDIM, TRANSDIM) = 101;
        LOC3(trans, 0, 7, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 119;
        LOC3(trans, 1, 0, 0, TRANSDIM, TRANSDIM, TRANSDIM) =   2;
        LOC3(trans, 1, 0, 1, TRANSDIM, TRANSDIM, TRANSDIM) =   7;
        LOC3(trans, 1, 0, 2, TRANSDIM, TRANSDIM, TRANSDIM) =  15;
        LOC3(trans, 1, 0, 3, TRANSDIM, TRANSDIM, TRANSDIM) =  28;
        LOC3(trans, 1, 0, 4, TRANSDIM, TRANSDIM, TRANSDIM) =  50;
        LOC3(trans, 1, 0, 5, TRANSDIM, TRANSDIM, TRANSDIM) =  69;
        LOC3(trans, 1, 0, 6, TRANSDIM, TRANSDIM, TRANSDIM) = 102;
        LOC3(trans, 1, 1, 0, TRANSDIM, TRANSDIM, TRANSDIM) =   5;
        LOC3(trans, 1, 1, 1, TRANSDIM, TRANSDIM, TRANSDIM) =  11;
        LOC3(trans, 1, 1, 2, TRANSDIM, TRANSDIM, TRANSDIM) =  26;
        LOC3(trans, 1, 1, 3, TRANSDIM, TRANSDIM, TRANSDIM) =  41;
        LOC3(trans, 1, 1, 4, TRANSDIM, TRANSDIM, TRANSDIM) =  59;
        LOC3(trans, 1, 1, 5, TRANSDIM, TRANSDIM, TRANSDIM) =  87;
        LOC3(trans, 1, 2, 0, TRANSDIM, TRANSDIM, TRANSDIM) =  13;
        LOC3(trans, 1, 2, 1, TRANSDIM, TRANSDIM, TRANSDIM) =  25;
        LOC3(trans, 1, 2, 2, TRANSDIM, TRANSDIM, TRANSDIM) =  36;
        LOC3(trans, 1, 2, 3, TRANSDIM, TRANSDIM, TRANSDIM) =  60;
        LOC3(trans, 1, 2, 4, TRANSDIM, TRANSDIM, TRANSDIM) =  88;
        LOC3(trans, 1, 3, 0, TRANSDIM, TRANSDIM, TRANSDIM) =  30;
        LOC3(trans, 1, 3, 1, TRANSDIM, TRANSDIM, TRANSDIM) =  40;
        LOC3(trans, 1, 3, 2, TRANSDIM, TRANSDIM, TRANSDIM) =  61;
        LOC3(trans, 1, 3, 3, TRANSDIM, TRANSDIM, TRANSDIM) =  94;
        LOC3(trans, 1, 4, 0, TRANSDIM, TRANSDIM, TRANSDIM) =  52;
        LOC3(trans, 1, 4, 1, TRANSDIM, TRANSDIM, TRANSDIM) =  58;
        LOC3(trans, 1, 4, 2, TRANSDIM, TRANSDIM, TRANSDIM) =  89;
        LOC3(trans, 1, 5, 0, TRANSDIM, TRANSDIM, TRANSDIM) =  71;
        LOC3(trans, 1, 5, 1, TRANSDIM, TRANSDIM, TRANSDIM) =  86;
        LOC3(trans, 1, 6, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 104;
        LOC3(trans, 2, 0, 0, TRANSDIM, TRANSDIM, TRANSDIM) =   8;
        LOC3(trans, 2, 0, 1, TRANSDIM, TRANSDIM, TRANSDIM) =  14;
        LOC3(trans, 2, 0, 2, TRANSDIM, TRANSDIM, TRANSDIM) =  22;
        LOC3(trans, 2, 0, 3, TRANSDIM, TRANSDIM, TRANSDIM) =  44;
        LOC3(trans, 2, 0, 4, TRANSDIM, TRANSDIM, TRANSDIM) =  75;
        LOC3(trans, 2, 0, 5, TRANSDIM, TRANSDIM, TRANSDIM) = 108;
        LOC3(trans, 2, 1, 0, TRANSDIM, TRANSDIM, TRANSDIM) =  12;
        LOC3(trans, 2, 1, 1, TRANSDIM, TRANSDIM, TRANSDIM) =  24;
        LOC3(trans, 2, 1, 2, TRANSDIM, TRANSDIM, TRANSDIM) =  37;
        LOC3(trans, 2, 1, 3, TRANSDIM, TRANSDIM, TRANSDIM) =  62;
        LOC3(trans, 2, 1, 4, TRANSDIM, TRANSDIM, TRANSDIM) =  90;
        LOC3(trans, 2, 2, 0, TRANSDIM, TRANSDIM, TRANSDIM) =  21;
        LOC3(trans, 2, 2, 1, TRANSDIM, TRANSDIM, TRANSDIM) =  38;
        LOC3(trans, 2, 2, 2, TRANSDIM, TRANSDIM, TRANSDIM) =  66;
        LOC3(trans, 2, 2, 3, TRANSDIM, TRANSDIM, TRANSDIM) =  99;
        LOC3(trans, 2, 3, 0, TRANSDIM, TRANSDIM, TRANSDIM) =  46;
        LOC3(trans, 2, 3, 1, TRANSDIM, TRANSDIM, TRANSDIM) =  64;
        LOC3(trans, 2, 3, 2, TRANSDIM, TRANSDIM, TRANSDIM) =  98;
        LOC3(trans, 2, 4, 0, TRANSDIM, TRANSDIM, TRANSDIM) =  77;
        LOC3(trans, 2, 4, 1, TRANSDIM, TRANSDIM, TRANSDIM) =  92;
        LOC3(trans, 2, 5, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 110;
        LOC3(trans, 3, 0, 0, TRANSDIM, TRANSDIM, TRANSDIM) =  18;
        LOC3(trans, 3, 0, 1, TRANSDIM, TRANSDIM, TRANSDIM) =  27;
        LOC3(trans, 3, 0, 2, TRANSDIM, TRANSDIM, TRANSDIM) =  45;
        LOC3(trans, 3, 0, 3, TRANSDIM, TRANSDIM, TRANSDIM) =  80;
        LOC3(trans, 3, 0, 4, TRANSDIM, TRANSDIM, TRANSDIM) = 114;
        LOC3(trans, 3, 1, 0, TRANSDIM, TRANSDIM, TRANSDIM) =  29;
        LOC3(trans, 3, 1, 1, TRANSDIM, TRANSDIM, TRANSDIM) =  39;
        LOC3(trans, 3, 1, 2, TRANSDIM, TRANSDIM, TRANSDIM) =  63;
        LOC3(trans, 3, 1, 3, TRANSDIM, TRANSDIM, TRANSDIM) =  95;
        LOC3(trans, 3, 2, 0, TRANSDIM, TRANSDIM, TRANSDIM) =  47;
        LOC3(trans, 3, 2, 1, TRANSDIM, TRANSDIM, TRANSDIM) =  65;
        LOC3(trans, 3, 2, 2, TRANSDIM, TRANSDIM, TRANSDIM) =  97;
        LOC3(trans, 3, 3, 0, TRANSDIM, TRANSDIM, TRANSDIM) =  81;
        LOC3(trans, 3, 3, 1, TRANSDIM, TRANSDIM, TRANSDIM) =  96;
        LOC3(trans, 3, 4, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 116;
        LOC3(trans, 4, 0, 0, TRANSDIM, TRANSDIM, TRANSDIM) =  33;
        LOC3(trans, 4, 0, 1, TRANSDIM, TRANSDIM, TRANSDIM) =  51;
        LOC3(trans, 4, 0, 2, TRANSDIM, TRANSDIM, TRANSDIM) =  76;
        LOC3(trans, 4, 0, 3, TRANSDIM, TRANSDIM, TRANSDIM) = 115;
        LOC3(trans, 4, 1, 0, TRANSDIM, TRANSDIM, TRANSDIM) =  53;
        LOC3(trans, 4, 1, 1, TRANSDIM, TRANSDIM, TRANSDIM) =  57;
        LOC3(trans, 4, 1, 2, TRANSDIM, TRANSDIM, TRANSDIM) =  91;
        LOC3(trans, 4, 2, 0, TRANSDIM, TRANSDIM, TRANSDIM) =  78;
        LOC3(trans, 4, 2, 1, TRANSDIM, TRANSDIM, TRANSDIM) =  93;
        LOC3(trans, 4, 3, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 117;
        LOC3(trans, 5, 0, 0, TRANSDIM, TRANSDIM, TRANSDIM) =  54;
        LOC3(trans, 5, 0, 1, TRANSDIM, TRANSDIM, TRANSDIM) =  70;
        LOC3(trans, 5, 0, 2, TRANSDIM, TRANSDIM, TRANSDIM) = 109;
        LOC3(trans, 5, 1, 0, TRANSDIM, TRANSDIM, TRANSDIM) =  72;
        LOC3(trans, 5, 1, 1, TRANSDIM, TRANSDIM, TRANSDIM) =  85;
        LOC3(trans, 5, 2, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 111;
        LOC3(trans, 6, 0, 0, TRANSDIM, TRANSDIM, TRANSDIM) =  82;
        LOC3(trans, 6, 0, 1, TRANSDIM, TRANSDIM, TRANSDIM) = 103;
        LOC3(trans, 6, 1, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 105;
        LOC3(trans, 7, 0, 0, TRANSDIM, TRANSDIM, TRANSDIM) = 118;
    }
    // upload to trans device location
    cudaError_t status;
    
    status = cudaMemcpyToSymbol(devTrans_MP2, trans, sizeof(int)*TRANSDIM*TRANSDIM*TRANSDIM);
    PRINTERROR(status, " cudaMemcpyToSymbol, Trans copy to constants failed")
    
}


