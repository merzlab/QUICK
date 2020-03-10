#include "gpu.h"
#include <cuda.h>
#include "libxc_dev_funcs.h"
#include "gpu_work_gga_x.cu"
#include "gpu_work_gga_c.cu"
#include "gpu_work_lda.cu"

static __constant__ gpu_simulation_type devSim_dft;
/*
static __constant__ QUICKDouble radii[19] = {1.e0, 0.5882e0, 3.0769e0, 
                         2.0513e0,1.5385e0, 1.2308e0, 1.0256e0,
                         0.8791e0,0.7692e0, 0.6838e0, 4.0909e0,
                         3.1579e0,2.5714e0, 2.1687e0, 1.8750e0,
                         1.6514e0,1.4754e0, 1.3333e0, 65*2.25e0};

static __constant__ QUICKDouble radii2[19] = {1.30e0,  0.0e0, 1.95e0, 2.20e0, 1.45e0, 
                            1.20e0, 1.10e0, 1.10e0, 1.20e0,  0.0e0, 
                            2.30e0, 2.20e0, 2.10e0, 1.30e0, 1.30e0, 
                            1.10e0, 1.45e0,  0.0e0, 65*2.25e0};

static __constant__ QUICKDouble RGRID[50] = { 0.000400000000, 0.001665972511, 0.003906250000, 
                            0.007243096424, 0.011814744802, 0.017777777778, 0.025309917355, 
                            0.034613304489, 0.045918367347, 0.059488399762, 0.075625000000, 
                            0.094674556213, 0.117036011080, 0.143170197224, 0.173611111111, 
                            0.208979591837, 0.250000000000, 0.297520661157, 0.352539062500, 
                            0.416233090531, 0.490000000000, 0.575505350773, 0.674744897959, 
                            0.790123456790, 0.924556213018, 1.081600000000, 1.265625000000, 
                            1.482041587902, 1.737603305785, 2.040816326531, 2.402500000000, 
                            2.836565096953, 3.361111111111, 4.000000000000, 4.785156250000, 
                            5.760000000000, 6.984693877551, 8.544378698225,10.562500000000,
                            13.223140495868,16.810000000000,21.777777777778,28.890625000000,
                            39.510204081633,56.250000000000,84.640000000000,138.062500000000,
                            256.000000000000,600.250000000000,2500.000000000000};

static __constant__ QUICKDouble RWT[50] = {0.000000000131, 0.000000004813, 0.000000042220, 
                            0.000000206165, 0.000000731384, 0.000002122603, 0.000005369349, 
                            0.000012296200, 0.000026125685, 0.000052373690, 0.000100263684, 
                            0.000184949922, 0.000331002868, 0.000577867516, 0.000988414167, 
                            0.001662355002, 0.002757352941, 0.004522359864, 0.007350524480, 
                            0.011863649113, 0.019047933333, 0.030473868600, 0.048655633610, 
                            0.077644380858, 0.124018587115, 0.198557753672, 0.319109916687, 
                            0.515579304527, 0.838747802817, 1.376167769422, 2.281385470312, 
                            3.828907341705, 6.520221288676,11.294117647059,19.957241602242,
                            36.097228800000,67.098458410034,128.799651727227,256.835164388021,
                            535.983391329609,1181.733250200000,2787.077579637250,7150.106995582581,
                            20425.715291126267,67236.328125000000,268905.261465599993,1427807.050415039062,
                            11883861.333333332092,225097464.046875000000,31875000000.000000000000};
*/                            

static __constant__ QUICKDouble hpartpara[4] = {0.2500e0,0.5000e0,1.0000e0,4.5000e0};
static __constant__ QUICKDouble lpartpara[4] = {0.1667e0,0.5000e0,0.9000e0,3.5000e0};
static __constant__ QUICKDouble npartpara[4] = {0.1000e0,0.4000e0,0.8000e0,2.5000e0};

/*
 upload gpu simulation type to constant memory
 */
void upload_sim_to_constant_dft(_gpu_type gpu){
    cudaError_t status;
    PRINTDEBUG("UPLOAD CONSTANT DFT");
    status = cudaMemcpyToSymbol(devSim_dft, &gpu->gpu_sim, sizeof(gpu_simulation_type));
//    status = cudaMemcpyToSymbol("devSim_dft", &gpu->gpu_sim, sizeof(gpu_simulation_type), 0, cudaMemcpyHostToDevice);
    PRINTERROR(status, " cudaMemcpyToSymbol, dft sim copy to constants failed")
    PRINTDEBUG("FINISH UPLOAD CONSTANT DFT");
}


#ifdef DEBUG
static float totTime;
#endif

/*Madu Manathunga 07/01/2019 added 3 new variables
glinfo and nof_functionals are for libxc. xc_calc_type determines where we would compute
energy or gradient */
/*void getxc(_gpu_type gpu, gpu_libxc_info** glinfo, int nof_functionals, int xc_calc_type, double* exc_dev_grad)
{
#ifdef DEBUG
    cudaEvent_t start,end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);
#endif
    if(xc_calc_type == 0){
#ifdef DEBUG
	printf("Running getxc_kernel with BLOCK = %i, THREADS PER BLOCK = %i \n", gpu->blocks, gpu->XCThreadsPerBlock);
#endif
        getxc_kernel<<<gpu->blocks, gpu->XCThreadsPerBlock>>>(glinfo, nof_functionals, exc_dev_grad);
    }else{
#ifdef DEBUG
	printf("Running getxc_grad_kernel with BLOCK = %i, THREADS PER BLOCK = %i \n", gpu->blocks, gpu->XCThreadsPerBlock);
#endif
        getxc_grad_kernel<<<gpu->blocks, gpu->XCThreadsPerBlock>>>(glinfo, nof_functionals, exc_dev_grad);
    }
		//cudaDeviceSynchronize();
#ifdef DEBUG
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float time;
    cudaEventElapsedTime(&time, start, end);
    totTime+=time;
    printf("this DFT cycle:%f ms total time:%f ms\n", time, totTime);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
#endif
    
}

*/

void get_ssw_new_imp(_gpu_type gpu){

#ifdef DEBUG
    cudaEvent_t start,end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);
#endif

	get_ssw_kernel_new_imp<<< gpu -> xc_blocks, gpu -> xc_threadsPerBlock>>>();

#ifdef DEBUG
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float time;
    cudaEventElapsedTime(&time, start, end);
    totTime+=time;
    printf("Time to compute grid weights on gpu:%f ms total time:%f ms\n", time, totTime);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
#endif

}

//Calls the device kernel to compute significant grid pts, contracted and primitive function
//indices on gpu.

/*void get_primf_contraf_lists(_gpu_type gpu){

#ifdef DEBUG
    cudaEvent_t start,end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);
#endif
	
    get_primf_contraf_lists_kernel<<< gpu -> xc_blocks, gpu -> xc_threadsPerBlock>>>();

#ifdef DEBUG
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float time;
    cudaEventElapsedTime(&time, start, end);
    totTime+=time;
    printf("Time to compute primitive and contracted indices on gpu: %f ms total time:%f ms\n", time, totTime);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
#endif

}
*/

void get_primf_contraf_lists_new_imp(_gpu_type gpu, unsigned char *gpweight, unsigned int *cfweight, unsigned int *pfweight){

#ifdef DEBUG
    cudaEvent_t start,end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);
#endif

    get_primf_contraf_lists_kernel_new_imp<<< gpu -> xc_blocks, gpu -> xc_threadsPerBlock>>>(gpweight, cfweight, pfweight);

#ifdef DEBUG
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float time;
    cudaEventElapsedTime(&time, start, end);
    totTime+=time;
    printf("Time to compute primitive and contracted indices on gpu: %f ms total time:%f ms\n", time, totTime);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
#endif

}

void getxc_new_imp(_gpu_type gpu, gpu_libxc_info** glinfo, int nof_functionals){

#ifdef DEBUG
    cudaEvent_t start,end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);
#endif

//        nvtxRangePushA("SCF XC: density");

	get_density_kernel_new_imp<<<gpu->xc_blocks, gpu->xc_threadsPerBlock>>>();

	cudaDeviceSynchronize();

//	nvtxRangePop();

//	nvtxRangePushA("SCF XC");

	getxc_kernel_new_imp<<<gpu->xc_blocks, gpu->xc_threadsPerBlock>>>(glinfo, nof_functionals);

#ifdef DEBUG
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float time;
    cudaEventElapsedTime(&time, start, end);
    totTime+=time;
    printf("this DFT cycle:%f ms total time:%f ms\n", time, totTime);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
#endif

//    nvtxRangePop();

}


void getxc_grad_new_imp(_gpu_type gpu, QUICKDouble* dev_grad, gpu_libxc_info** glinfo, int nof_functionals){

#ifdef DEBUG
    cudaEvent_t start,end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, 0);
#endif

//    nvtxRangePushA("XC grad: density");	

    get_density_kernel_new_imp<<<gpu->xc_blocks, gpu->xc_threadsPerBlock>>>();

    cudaDeviceSynchronize();
 
//    nvtxRangePop();

//    nvtxRangePushA("XC grad");

    get_xcgrad_kernel_new_imp<<<gpu->xc_blocks, gpu->xc_threadsPerBlock>>>(dev_grad, glinfo, nof_functionals);

    cudaDeviceSynchronize();

    prune_grid_sswgrad();

    get_sswgrad_kernel_new_imp<<<gpu->xc_blocks, gpu->xc_threadsPerBlock>>>(dev_grad);

    gpu_delete_sswgrad_vars();

#ifdef DEBUG
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    float time;
    cudaEventElapsedTime(&time, start, end);
    totTime+=time;
    printf("this DFT cycle:%f ms total time:%f ms\n", time, totTime);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
#endif

//    nvtxRangePop();

}

__global__ void get_density_kernel_new_imp()
{
        int gid = blockIdx.x * blockDim.x + threadIdx.x;

        if(gid<devSim_dft.npoints){

                int dweight = devSim_dft.dweight[gid];

                if(dweight >0){

	                QUICKDouble density = 0.0;
        	        QUICKDouble gax = 0.0;
	                QUICKDouble gay = 0.0;
        	        QUICKDouble gaz = 0.0;

                        int bin_id = (int) (gid/devSim_dft.bin_size);
                        //int nofbfs = devSim_dft.basf_locator[bin_id+1] - devSim_dft.basf_locator[bin_id];

                        QUICKDouble gridx = devSim_dft.gridx[gid];
                        QUICKDouble gridy = devSim_dft.gridy[gid];
                        QUICKDouble gridz = devSim_dft.gridz[gid];

	                for(int i=devSim_dft.basf_locator[bin_id]; i<devSim_dft.basf_locator[bin_id+1] ; i++){
        	                int ibas = devSim_dft.basf[i];
                	        QUICKDouble phi, dphidx, dphidy, dphidz;
			
				pteval_new(gridx, gridy, gridz, &phi, &dphidx, &dphidy, &dphidz, devSim_dft.primf, devSim_dft.primf_locator, ibas, i);

				if (abs(phi+dphidx+dphidy+dphidz) >= devSim_dft.DMCutoff ) {

					QUICKDouble denseii = LOC2(devSim_dft.dense, ibas, ibas, devSim_dft.nbasis, devSim_dft.nbasis) * phi;
					density = density + denseii * phi / 2.0;
					gax = gax + denseii * dphidx;
					gay = gay + denseii * dphidy;
					gaz = gaz + denseii * dphidz;

					for(int j=i+1; j< devSim_dft.basf_locator[bin_id+1]; j++){
						int jbas = devSim_dft.basf[j];
						QUICKDouble phi2, dphidx2, dphidy2, dphidz2;
						pteval_new(gridx, gridy, gridz, &phi2, &dphidx2, &dphidy2, &dphidz2, devSim_dft.primf, devSim_dft.primf_locator, jbas, j);
						QUICKDouble denseij = LOC2(devSim_dft.dense, ibas, jbas, devSim_dft.nbasis, devSim_dft.nbasis);
						density = density + denseij * phi * phi2;
						gax = gax + denseij * ( phi * dphidx2 + phi2 * dphidx );
						gay = gay + denseij * ( phi * dphidy2 + phi2 * dphidy );
						gaz = gaz + denseij * ( phi * dphidz2 + phi2 * dphidz );
					}
				}
			}

			devSim_dft.densa[gid] = density;
			devSim_dft.densb[gid] = density;
			devSim_dft.gax[gid] = gax;
			devSim_dft.gbx[gid] = gax;
			devSim_dft.gay[gid] = gay;
			devSim_dft.gby[gid] = gay;
			devSim_dft.gaz[gid] = gaz;
			devSim_dft.gbz[gid] = gaz;

#ifdef DEBUG
//              printf("TEST_NEW_UPLOAD gid: %i x: %f y: %f z: %f sswt: %f weight: %f gatm: %i dweight: %i \n ", gid, gridx, gridy, gridz, sswt, weight, gatm, dweight);
#endif

#ifdef DEBUG
//        printf("gridx: %f  gridy: %f  gridz: %f, weight: %.10e, density: %.10e \n",gridx, gridy, gridz, weight, density);
#endif
		}
	}
}

__global__ void getxc_kernel_new_imp(gpu_libxc_info** glinfo, int nof_functionals){

        int gid = blockIdx.x * blockDim.x + threadIdx.x;

        if(gid<devSim_dft.npoints){

                int dweight = devSim_dft.dweight[gid];

                if(dweight>0){
                        int bin_id = (int) (gid/devSim_dft.bin_size);
                        //int nofbfs = devSim_dft.basf_locator[bin_id+1] - devSim_dft.basf_locator[bin_id];

                        QUICKDouble gridx = devSim_dft.gridx[gid];
                        QUICKDouble gridy = devSim_dft.gridy[gid];
                        QUICKDouble gridz = devSim_dft.gridz[gid];
                        QUICKDouble weight = devSim_dft.weight[gid];
                        QUICKDouble density = devSim_dft.densa[gid];
                        QUICKDouble densityb = devSim_dft.densb[gid];
                        QUICKDouble gax = devSim_dft.gax[gid];
                        QUICKDouble gay = devSim_dft.gay[gid];
                        QUICKDouble gaz = devSim_dft.gaz[gid];
                        QUICKDouble gbx = devSim_dft.gbx[gid];
                        QUICKDouble gby = devSim_dft.gby[gid];
                        QUICKDouble gbz = devSim_dft.gbz[gid];

                        if(density >devSim_dft.DMCutoff){
                                QUICKDouble sigma = 4.0 * (gax * gax + gay * gay + gaz * gaz);
                                QUICKDouble _tmp ;

                                if (devSim_dft.method == B3LYP) {
                                        _tmp = b3lyp_e(2.0*density, sigma) * weight;
                                }else if(devSim_dft.method == DFT){
                                        _tmp = (becke_e(density, densityb, gax, gay, gaz, gbx, gby, gbz)
                                        + lyp_e(density, densityb, gax, gay, gaz, gbx, gby, gbz)) * weight;
                                }

                                QUICKDouble dfdr;
                                QUICKDouble dot, xdot, ydot, zdot;

                                if (devSim_dft.method == B3LYP) {
                                        dot = b3lypf(2.0*density, sigma, &dfdr);
                                        xdot = dot * gax;
                                        ydot = dot * gay;
                                        zdot = dot * gaz;
                                }else if(devSim_dft.method == DFT){
                                        QUICKDouble dfdgaa, dfdgab, dfdgaa2, dfdgab2;
                                        QUICKDouble dfdr2;

					becke(density, gax, gay, gaz, gbx, gby, gbz, &dfdr, &dfdgaa, &dfdgab);
                                        lyp(density, densityb, gax, gay, gaz, gbx, gby, gbz, &dfdr2, &dfdgaa2, &dfdgab2);
                                        dfdr += dfdr2;
                                        dfdgaa += dfdgaa2;
                                        dfdgab += dfdgab2;
                                        //Calculate the first term in the dot product shown above,i.e.:
                                        //(2 df/dgaa Grad(rho a) + df/dgab Grad(rho b)) doT Grad(Phimu Phinu))
                                        xdot = 2.0 * dfdgaa * gax + dfdgab * gbx;
                                        ydot = 2.0 * dfdgaa * gay + dfdgab * gby;
                                        zdot = 2.0 * dfdgaa * gaz + dfdgab * gbz;

                                }else if(devSim_dft.method == LIBXC){
                                        //Prepare in/out for libxc call
                                        double d_rhoa = (double) density;
                                        double d_rhob = (double) densityb;
                                        double d_sigma = (double)sigma;
                                        double d_zk, d_vrho, d_vsigma;
                                        d_zk = d_vrho = d_vsigma = 0.0;
        
                                        for(int i=0; i<nof_functionals; i++){
                                                double tmp_d_zk, tmp_d_vrho, tmp_d_vsigma;
                                                tmp_d_zk=tmp_d_vrho=tmp_d_vsigma=0.0;

                                                gpu_libxc_info* tmp_glinfo = glinfo[i];

                                                switch(tmp_glinfo->gpu_worker){
                                                        case GPU_WORK_LDA:
                                                                gpu_work_lda_c(tmp_glinfo, d_rhoa, d_rhob, &tmp_d_zk, &tmp_d_vrho, 1);
                                                                break;
                                        
                                                        case GPU_WORK_GGA_X:
                                                                gpu_work_gga_x(tmp_glinfo, d_rhoa, d_rhob, d_sigma, &tmp_d_zk, &tmp_d_vrho, &tmp_d_vsigma);
                                                                break;
        
                                                        case GPU_WORK_GGA_C:
                                                                gpu_work_gga_c(tmp_glinfo, d_rhoa, d_rhob, d_sigma, &tmp_d_zk, &tmp_d_vrho, &tmp_d_vsigma, 1);
                                                                break;
                                                }
                                                d_zk += (tmp_d_zk*tmp_glinfo->mix_coeff);
                                                d_vrho += (tmp_d_vrho*tmp_glinfo->mix_coeff);
                                                d_vsigma += (tmp_d_vsigma*tmp_glinfo->mix_coeff);
                                        }
                        
                                        _tmp = ((QUICKDouble) (d_zk * (d_rhoa + d_rhob)) * weight);                              

                                        QUICKDouble dfdgaa;
					//QUICKDouble dfdgab, dfdgaa2, dfdgab2;
                                        //QUICKDouble dfdr2;
                                        dfdr = (QUICKDouble)d_vrho;
                                        dfdgaa = (QUICKDouble)d_vsigma*4.0;

                                        xdot = dfdgaa * gax;
                                        ydot = dfdgaa * gay;
                                        zdot = dfdgaa * gaz;
                                }

				QUICKULL val1 = (QUICKULL) (fabs( _tmp * OSCALE) + (QUICKDouble)0.5);
                                if ( _tmp * weight < (QUICKDouble)0.0)
                                    val1 = 0ull - val1;
                                QUICKADD(devSim_dft.DFT_calculated[0].Eelxc, val1);
        
                                _tmp = weight*density;
                                val1 = (QUICKULL) (fabs( _tmp * OSCALE) + (QUICKDouble)0.5);
                                if ( _tmp * weight < (QUICKDouble)0.0)
                                    val1 = 0ull - val1;
                                QUICKADD(devSim_dft.DFT_calculated[0].aelec, val1);
        
        
                                _tmp = weight*densityb;
                                val1 = (QUICKULL) (fabs( _tmp * OSCALE) + (QUICKDouble)0.5);
                                if ( _tmp * weight < (QUICKDouble)0.0)
                                    val1 = 0ull - val1;
                                QUICKADD(devSim_dft.DFT_calculated[0].belec, val1);

                                for (int i = devSim_dft.basf_locator[bin_id]; i<devSim_dft.basf_locator[bin_id+1]; i++) {
                                        int ibas = devSim_dft.basf[i];
                                        QUICKDouble phi, dphidx, dphidy, dphidz;
                                        pteval_new(gridx, gridy, gridz, &phi, &dphidx, &dphidy, &dphidz, devSim_dft.primf, devSim_dft.primf_locator, ibas, i);

                                        if (abs(phi+dphidx+dphidy+dphidz)> devSim_dft.DMCutoff ) {

                                                for (int j = devSim_dft.basf_locator[bin_id]; j <devSim_dft.basf_locator[bin_id+1]; j++) {

                                                        int jbas = devSim_dft.basf[j];

                                                        QUICKDouble phi2, dphidx2, dphidy2, dphidz2;

                                                        pteval_new(gridx, gridy, gridz, &phi2, &dphidx2, &dphidy2, &dphidz2, devSim_dft.primf, devSim_dft.primf_locator, jbas, j);

														QUICKDouble _tmp = (phi * phi2 * dfdr + xdot * (phi*dphidx2 + phi2*dphidx) \
															+ ydot * (phi*dphidy2 + phi2*dphidy) + zdot * (phi*dphidz2 + phi2*dphidz))*weight;

														QUICKULL val1 = (QUICKULL) (fabs( _tmp * OSCALE) + (QUICKDouble)0.5);
														if ( _tmp * weight < (QUICKDouble)0.0)
															val1 = 0ull - val1;
														QUICKADD(LOC2(devSim_dft.oULL, jbas, ibas, devSim_dft.nbasis, devSim_dft.nbasis), val1);

                                                }
                                        }
                                }
                        }
                }
        }
}


__global__ void get_xcgrad_kernel_new_imp(QUICKDouble* dev_grad, gpu_libxc_info** glinfo, int nof_functionals){

	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(gid<devSim_dft.npoints){

		int dweight = devSim_dft.dweight[gid];

		if(dweight>0){

                        int bin_id = (int) (gid/devSim_dft.bin_size);
                        //int nofbfs = devSim_dft.basf_locator[bin_id+1] - devSim_dft.basf_locator[bin_id];

                        QUICKDouble gridx = devSim_dft.gridx[gid];
                        QUICKDouble gridy = devSim_dft.gridy[gid];
                        QUICKDouble gridz = devSim_dft.gridz[gid];
			QUICKDouble weight = devSim_dft.weight[gid];
                        QUICKDouble density = devSim_dft.densa[gid];
                        QUICKDouble densityb = devSim_dft.densb[gid];
                        QUICKDouble gax = devSim_dft.gax[gid];
                        QUICKDouble gay = devSim_dft.gay[gid];
                        QUICKDouble gaz = devSim_dft.gaz[gid];
                        QUICKDouble gbx = devSim_dft.gbx[gid];
                        QUICKDouble gby = devSim_dft.gby[gid];
                        QUICKDouble gbz = devSim_dft.gbz[gid];			
		
			if(density >devSim_dft.DMCutoff){	
				QUICKDouble sigma = 4.0 * (gax * gax + gay * gay + gaz * gaz);
				QUICKDouble _tmp ;
				
				if (devSim_dft.method == B3LYP) {
					_tmp = b3lyp_e(2.0*density, sigma);
				}else if(devSim_dft.method == DFT){
					_tmp = (becke_e(density, densityb, gax, gay, gaz, gbx, gby, gbz)
					+ lyp_e(density, densityb, gax, gay, gaz, gbx, gby, gbz));
				}

				QUICKDouble dfdr;
				QUICKDouble dot, xdot, ydot, zdot;

				if (devSim_dft.method == B3LYP) {
					dot = b3lypf(2.0*density, sigma, &dfdr);
					xdot = dot * gax;
					ydot = dot * gay;
					zdot = dot * gaz;
				}else if(devSim_dft.method == DFT){
					QUICKDouble dfdgaa, dfdgab, dfdgaa2, dfdgab2;
					QUICKDouble dfdr2;
					
					lyp(density, densityb, gax, gay, gaz, gbx, gby, gbz, &dfdr2, &dfdgaa2, &dfdgab2);
                                	dfdr = dfdr2;
                                	dfdgaa = dfdgaa2;
                                	dfdgab = dfdgab2;	
					
	                                //Calculate the first term in the dot product shown above,i.e.:
	                                //(2 df/dgaa Grad(rho a) + df/dgab Grad(rho b)) doT Grad(Phimu Phinu))
	                                xdot = 2.0 * dfdgaa * gax + dfdgab * gbx;
	                                ydot = 2.0 * dfdgaa * gay + dfdgab * gby;
	                                zdot = 2.0 * dfdgaa * gaz + dfdgab * gbz;					
				
				}else if(devSim_dft.method == LIBXC){
	                                //Prepare in/out for libxc call
        	                        double d_rhoa = (double) density;
	                                double d_rhob = (double) densityb;
     		                        double d_sigma = (double)sigma;
                	                double d_zk, d_vrho, d_vsigma;
                        	        d_zk = d_vrho = d_vsigma = 0.0;
	
        	                        for(int i=0; i<nof_functionals; i++){
                	                        double tmp_d_zk, tmp_d_vrho, tmp_d_vsigma;
                        	                tmp_d_zk=tmp_d_vrho=tmp_d_vsigma=0.0;

                                	        gpu_libxc_info* tmp_glinfo = glinfo[i];

                                        	switch(tmp_glinfo->gpu_worker){
	                                                case GPU_WORK_LDA:
        	                                                gpu_work_lda_c(tmp_glinfo, d_rhoa, d_rhob, &tmp_d_zk, &tmp_d_vrho, 1);
                	                                        break;
                                        
                        	                        case GPU_WORK_GGA_X:
                                	                        gpu_work_gga_x(tmp_glinfo, d_rhoa, d_rhob, d_sigma, &tmp_d_zk, &tmp_d_vrho, &tmp_d_vsigma);
                                        	                break;
	
        	                                        case GPU_WORK_GGA_C:
                	                                        gpu_work_gga_c(tmp_glinfo, d_rhoa, d_rhob, d_sigma, &tmp_d_zk, &tmp_d_vrho, &tmp_d_vsigma, 1);
                        	                                break;
                                	        }
	                                        d_zk += (tmp_d_zk*tmp_glinfo->mix_coeff);
	                                        d_vrho += (tmp_d_vrho*tmp_glinfo->mix_coeff);
	                                        d_vsigma += (tmp_d_vsigma*tmp_glinfo->mix_coeff);
	                                }
                        
	                                _tmp = ((QUICKDouble) (d_zk * (d_rhoa + d_rhob)));                              

	                                QUICKDouble dfdgaa;
					//QUICKDouble dfdgab, dfdgaa2, dfdgab2;
	                                //QUICKDouble dfdr2;
	                                dfdr = (QUICKDouble)d_vrho;
	                                dfdgaa = (QUICKDouble)d_vsigma*4.0;

	                                xdot = dfdgaa * gax;
	                                ydot = dfdgaa * gay;
	                                zdot = dfdgaa * gaz;
				}
				devSim_dft.exc[gid] = _tmp;
			
				for (int i = devSim_dft.basf_locator[bin_id]; i<devSim_dft.basf_locator[bin_id+1]; i++) {
					int ibas = devSim_dft.basf[i];
					QUICKDouble phi, dphidx, dphidy, dphidz;
					pteval_new(gridx, gridy, gridz, &phi, &dphidx, &dphidy, &dphidz, devSim_dft.primf, devSim_dft.primf_locator, ibas, i);

					if (abs(phi+dphidx+dphidy+dphidz)> devSim_dft.DMCutoff ) {

						QUICKDouble dxdx, dxdy, dxdz, dydy, dydz, dzdz;

						pt2der_new(gridx, gridy, gridz, &dxdx, &dxdy, &dxdz, &dydy, &dydz, &dzdz, devSim_dft.primf, devSim_dft.primf_locator, ibas, i);
				
						int Istart = (devSim_dft.ncenter[ibas]-1) * 3;
					
						for (int j = devSim_dft.basf_locator[bin_id]; j <devSim_dft.basf_locator[bin_id+1]; j++) {

							int jbas = devSim_dft.basf[j];

							QUICKDouble phi2, dphidx2, dphidy2, dphidz2;

							pteval_new(gridx, gridy, gridz, &phi2, &dphidx2, &dphidy2, &dphidz2, devSim_dft.primf, devSim_dft.primf_locator, jbas, j);
							
							QUICKDouble denseij = (QUICKDouble) LOC2(devSim_dft.dense, ibas, jbas, devSim_dft.nbasis, devSim_dft.nbasis);

	                                                QUICKDouble Gradx = - 2.0 * denseij * weight * (dfdr * dphidx * phi2
                                                                + xdot * (dxdx * phi2 + dphidx * dphidx2)
                                                                + ydot * (dxdy * phi2 + dphidx * dphidy2)
                                                                + zdot * (dxdz * phi2 + dphidx * dphidz2));

        	                                        QUICKDouble Grady = - 2.0 * denseij * weight * (dfdr * dphidy * phi2
                                                                + xdot * (dxdy * phi2 + dphidy * dphidx2)
                                                                + ydot * (dydy * phi2 + dphidy * dphidy2)
                                                                + zdot * (dydz * phi2 + dphidy * dphidz2));

                	                                QUICKDouble Gradz = - 2.0 * denseij * weight * (dfdr * dphidz * phi2
                                                                + xdot * (dxdz * phi2 + dphidz * dphidx2)
                                                                + ydot * (dydz * phi2 + dphidz * dphidy2)
                                                                + zdot * (dzdz * phi2 + dphidz * dphidz2));							
						//	printf("Test xc_grad: %i gradx: %f grady: %f gradz: %f \n", gid, Gradx, Grady, Gradz);
	/*                                                atomicAdd(&devSim_dft.xc_grad[Istart], Gradx);
        	                                        atomicAdd(&devSim_dft.xc_grad[Istart+1], Grady);
                	                                atomicAdd(&devSim_dft.xc_grad[Istart+2], Gradz);
	*/						atomicAdd(&dev_grad[Istart], Gradx);
							atomicAdd(&dev_grad[Istart+1], Grady);
							atomicAdd(&dev_grad[Istart+2], Gradz);
						}
						//printf("Test xc_grad: %i %f %f %f \n", gid, devSim_dft.xc_grad[Istart], devSim_dft.xc_grad[Istart+1], devSim_dft.xc_grad[Istart+2]);
					}
				}
			}

                        //Set weights for sswder calculation
                        if(density < devSim_dft.DMCutoff){
                                devSim_dft.dweight_ssd[gid] = 0;
                        }

                        if(devSim_dft.sswt[gid] == 1){
                                devSim_dft.dweight_ssd[gid] = 0;
                        }				
		}
	}
}

//device kernel to compute significant grid pts, contracted and primitive functions for octree
//__global__ void get_primf_contraf_lists_kernel(){
/*
        int gid = blockIdx.x * blockDim.x + threadIdx.x;

        if(gid<devSim_dft.npoints){	
		QUICKDouble gridx = devSim_dft.gridx[gid];
		QUICKDouble gridy = devSim_dft.gridy[gid];
		QUICKDouble gridz = devSim_dft.gridz[gid];
		unsigned char gpweight = devSim_dft.gpweight[gid];

		// relative coordinates between grid point and basis function I.

		for(int ibas=0; ibas<devSim_dft.nbasis;ibas++){

			unsigned long cfwid = blockIdx.x * devSim_dft.nbasis + ibas;			

			QUICKDouble x1 = gridx - LOC2(devSim_dft.xyz, 0, devSim_dft.ncenter[ibas]-1, 3, devSim_dft.natom);
		        QUICKDouble y1 = gridy - LOC2(devSim_dft.xyz, 1, devSim_dft.ncenter[ibas]-1, 3, devSim_dft.natom);
			QUICKDouble z1 = gridz - LOC2(devSim_dft.xyz, 2, devSim_dft.ncenter[ibas]-1, 3, devSim_dft.natom);		

			QUICKDouble x1i, y1i, z1i;
			QUICKDouble x1imin1, y1imin1, z1imin1;
			QUICKDouble x1iplus1, y1iplus1, z1iplus1;

			QUICKDouble phi = 0.0;
			QUICKDouble dphidx = 0.0;
			QUICKDouble dphidy = 0.0;
			QUICKDouble dphidz = 0.0;

			int itypex = LOC2(devSim_dft.itype, 0, ibas, 3, devSim_dft.nbasis);
			int itypey = LOC2(devSim_dft.itype, 1, ibas, 3, devSim_dft.nbasis);
			int itypez = LOC2(devSim_dft.itype, 2, ibas, 3, devSim_dft.nbasis);

			QUICKDouble dist = x1*x1+y1*y1+z1*z1;

			
			//if ( dist <= devSim_dft.sigrad2[ibas]){

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

				for(int kprim=0; kprim< devSim_dft.ncontract[ibas]; kprim++){

					unsigned long pfwid = blockIdx.x * devSim_dft.nbasis * devSim_dft.maxcontract + ibas * devSim_dft.maxcontract + kprim;

					QUICKDouble tmp = LOC2(devSim_dft.dcoeff, kprim, ibas, devSim_dft.maxcontract, devSim_dft.nbasis) *
						exp( - LOC2(devSim_dft.aexp, kprim, ibas, devSim_dft.maxcontract, devSim_dft.nbasis) * dist);
					QUICKDouble tmpdx = tmp * ( -2.0 * LOC2(devSim_dft.aexp, kprim, ibas, devSim_dft.maxcontract, devSim_dft.nbasis)* x1iplus1 + (QUICKDouble)itypex * x1imin1);
					QUICKDouble tmpdy = tmp * ( -2.0 * LOC2(devSim_dft.aexp, kprim, ibas, devSim_dft.maxcontract, devSim_dft.nbasis)* y1iplus1 + (QUICKDouble)itypey * y1imin1);
					QUICKDouble tmpdz = tmp * ( -2.0 * LOC2(devSim_dft.aexp, kprim, ibas, devSim_dft.maxcontract, devSim_dft.nbasis)* z1iplus1 + (QUICKDouble)itypez * z1imin1);

					phi = phi + tmp;
					dphidx = dphidx + tmpdx;
					dphidy = dphidy + tmpdy;
					dphidz = dphidz + tmpdz;

					//Check the significance of the primitive
					if(abs(tmp+tmpdx+tmpdy+tmpdz) > devSim_dft.DMCutoff){
						atomicAdd(&devSim_dft.pfweight, 1);	
					}
					
				}
	
				phi = phi * x1i * y1i * z1i;
				dphidx = dphidx * y1i * z1i;
				dphidy = dphidy * x1i * z1i;
				dphidz = dphidz * x1i * y1i;
				
			//}
			

			if (abs(phi+dphidx+dphidy+dphidz)> devSim_dft.DMCutoff ){
				atomicAdd(&devSim_dft.pfweight, 1);
			}
				
		}
		
		unsigned int sum_cfweight = 0; 

		for(int ibas=0; ibas<devSim_dft.nbasis; ibas++){
			sum_cfweight += devSim_dft.cfweight[gid * devSim_dft.nbasis + ibas];
		}		

		if(sum_cfweight > 0){
			devSim_dft.gpweight[gid] = 1;
		}else{
			devSim_dft.gpweight[gid] = 0;
		}

	}
*/
//}


//device kernel to compute significant grid pts, contracted and primitive functions for octree
__global__ void get_primf_contraf_lists_kernel_new_imp(unsigned char *gpweight, unsigned int *cfweight, unsigned int *pfweight){

        int gid = blockIdx.x * blockDim.x + threadIdx.x;

        if(gid < devSim_dft.npoints){
		if(gpweight[gid]>0){

        	        QUICKDouble gridx = devSim_dft.gridx[gid];
	                QUICKDouble gridy = devSim_dft.gridy[gid];
	                QUICKDouble gridz = devSim_dft.gridz[gid];

			unsigned int sigcfcount=0;
	
        	        // relative coordinates between grid point and basis function I.

                	for(int ibas=0; ibas<devSim_dft.nbasis;ibas++){

                        	unsigned long cfwid = blockIdx.x * devSim_dft.nbasis + ibas; 

                        	QUICKDouble x1 = gridx - LOC2(devSim_dft.xyz, 0, devSim_dft.ncenter[ibas]-1, 3, devSim_dft.natom);
                        	QUICKDouble y1 = gridy - LOC2(devSim_dft.xyz, 1, devSim_dft.ncenter[ibas]-1, 3, devSim_dft.natom);
                        	QUICKDouble z1 = gridz - LOC2(devSim_dft.xyz, 2, devSim_dft.ncenter[ibas]-1, 3, devSim_dft.natom);

                        	QUICKDouble x1i, y1i, z1i;
                        	QUICKDouble x1imin1, y1imin1, z1imin1;
                        	QUICKDouble x1iplus1, y1iplus1, z1iplus1;

                        	QUICKDouble phi = 0.0;
                        	QUICKDouble dphidx = 0.0;
                        	QUICKDouble dphidy = 0.0;
                        	QUICKDouble dphidz = 0.0;

                        	int itypex = LOC2(devSim_dft.itype, 0, ibas, 3, devSim_dft.nbasis);
                        	int itypey = LOC2(devSim_dft.itype, 1, ibas, 3, devSim_dft.nbasis);
                        	int itypez = LOC2(devSim_dft.itype, 2, ibas, 3, devSim_dft.nbasis);

                        	QUICKDouble dist = x1*x1+y1*y1+z1*z1;

                        	if ( dist <= devSim_dft.sigrad2[ibas]){

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


                                	for(int kprim=0; kprim< devSim_dft.ncontract[ibas]; kprim++){

                                        	unsigned long pfwid = blockIdx.x * devSim_dft.nbasis * devSim_dft.maxcontract + ibas * devSim_dft.maxcontract + kprim;

                                        	QUICKDouble tmp = LOC2(devSim_dft.dcoeff, kprim, ibas, devSim_dft.maxcontract, devSim_dft.nbasis) *
                                                	exp( - LOC2(devSim_dft.aexp, kprim, ibas, devSim_dft.maxcontract, devSim_dft.nbasis) * dist);
                                        	QUICKDouble tmpdx = tmp * ( -2.0 * LOC2(devSim_dft.aexp, kprim, ibas, devSim_dft.maxcontract, devSim_dft.nbasis)* x1iplus1 + (QUICKDouble)itypex * x1imin1);
                                        	QUICKDouble tmpdy = tmp * ( -2.0 * LOC2(devSim_dft.aexp, kprim, ibas, devSim_dft.maxcontract, devSim_dft.nbasis)* y1iplus1 + (QUICKDouble)itypey * y1imin1);
                                        	QUICKDouble tmpdz = tmp * ( -2.0 * LOC2(devSim_dft.aexp, kprim, ibas, devSim_dft.maxcontract, devSim_dft.nbasis)* z1iplus1 + (QUICKDouble)itypez * z1imin1);

                                        	phi = phi + tmp;
                                        	dphidx = dphidx + tmpdx;
                                        	dphidy = dphidy + tmpdy;
                                        	dphidz = dphidz + tmpdz;

                                        	//Check the significance of the primitive
						if(abs(tmp+tmpdx+tmpdy+tmpdz) > devSim_dft.DMCutoff){
							atomicAdd(&pfweight[pfwid], 1);
                                        	}
                                	}

                                	phi = phi * x1i * y1i * z1i;
                                	dphidx = dphidx * y1i * z1i;
                                	dphidy = dphidy * x1i * z1i;
                                	dphidz = dphidz * x1i * y1i;

                        	}

                        	if (abs(phi+dphidx+dphidy+dphidz)> devSim_dft.DMCutoff ){
					atomicAdd(&cfweight[cfwid], 1);
					sigcfcount++;
                        	}
			
                	}
			
			if(sigcfcount < 1){
				gpweight[gid] = 0;
			}
		}
        }

}

__global__ void get_ssw_kernel_new_imp(){

        int gid = blockIdx.x * blockDim.x + threadIdx.x;

        if(gid<devSim_dft.npoints){

                QUICKDouble gridx = devSim_dft.gridx[gid];
                QUICKDouble gridy = devSim_dft.gridy[gid];
                QUICKDouble gridz = devSim_dft.gridz[gid];
		QUICKDouble wtang = devSim_dft.wtang[gid];
		QUICKDouble rwt = devSim_dft.rwt[gid];
		QUICKDouble rad3 = devSim_dft.rad3[gid];
                int gatm = devSim_dft.gatm[gid];
		
                QUICKDouble sswt = SSW(gridx, gridy, gridz, gatm);
                QUICKDouble weight = sswt*wtang*rwt*rad3;

		devSim_dft.sswt[gid] = sswt;
		devSim_dft.weight[gid] = weight;

        }

}


__global__ void get_sswgrad_kernel_new_imp(QUICKDouble* dev_grad){

        int gid = blockIdx.x * blockDim.x + threadIdx.x;
     
        if(gid<devSim_dft.npoints_ssd){

                QUICKDouble gridx = devSim_dft.gridx_ssd[gid];
                QUICKDouble gridy = devSim_dft.gridy_ssd[gid];
                QUICKDouble gridz = devSim_dft.gridz_ssd[gid];
                QUICKDouble exc = devSim_dft.exc_ssd[gid];
		QUICKDouble quadwt = devSim_dft.quadwt[gid];
                int gatm = devSim_dft.gatm_ssd[gid];

		//sswder(gridx, gridy, gridz, exc, quadwt, gatm, dev_grad);
		sswder_new_imp(gridx, gridy, gridz, exc, quadwt, gatm, gid, dev_grad);
	}

}

/*__global__ void get_sswgrad_kernel_new_imp(QUICKDouble* dev_grad){

        int gid = blockIdx.x * blockDim.x + threadIdx.x;

        if(gid<devSim_dft.npoints){

                int dweight = devSim_dft.dweight[gid];

                if(dweight>0){

                        QUICKDouble gridx = devSim_dft.gridx[gid];
                        QUICKDouble gridy = devSim_dft.gridy[gid];
                        QUICKDouble gridz = devSim_dft.gridz[gid];
                        QUICKDouble weight = devSim_dft.weight[gid];
                        QUICKDouble sswt = devSim_dft.sswt[gid];
                        QUICKDouble exc = devSim_dft.exc[gid];
                        int gatm = devSim_dft.gatm[gid];

                        sswder(gridx, gridy, gridz, exc, weight/sswt, gatm, dev_grad);
                }
        }

}
*/

/*__launch_bounds__(SM_2X_XC_THREADS_PER_BLOCK, 1)
__global__ void get_ssw_kernel(QUICKDouble* d_grid_ptx, QUICKDouble* d_grid_pty, QUICKDouble* d_grid_ptz, int* d_grid_atm, QUICKDouble* d_grid_sswt, QUICKDouble* d_grid_weights, QUICKDouble* wtang, QUICKDouble* rwt, QUICKDouble* rad3, int size)
{
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	if(gid<size){
		d_grid_sswt[gid] = SSW(d_grid_ptx[gid], d_grid_pty[gid], d_grid_ptz[gid], d_grid_atm[gid]);
		d_grid_weights[gid] = d_grid_sswt[gid]*wtang[gid]*rwt[gid]*rad3[gid];
#ifdef DEBUG
//              printf("gid: %i d_grid_ptx: %f d_grid_pty: %f d_grid_ptz: %f d_grid_weights: %f \n", gid, d_grid_ptx[gid], d_grid_pty[gid], d_grid_ptz[gid],d_grid_weights[gid]);
#endif
	}
}
*/

/*__launch_bounds__(SM_2X_XC_THREADS_PER_BLOCK, 1)
__global__ void get_density_kernel(QUICKDouble* d_grid_ptx, QUICKDouble* d_grid_pty, QUICKDouble* d_grid_ptz, QUICKDouble* d_density, QUICKDouble* d_densityb, QUICKDouble* d_gax, QUICKDouble* d_gay, QUICKDouble* d_gaz, QUICKDouble* d_gbx, QUICKDouble* d_gby, QUICKDouble* d_gbz, int size)
{
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	if(gid<size){

		QUICKDouble gridx = d_grid_ptx[gid];
		QUICKDouble gridy = d_grid_pty[gid];
		QUICKDouble gridz = d_grid_ptz[gid];
		QUICKDouble density, densityb;
		QUICKDouble gax, gay, gaz;
		QUICKDouble gbx, gby, gbz;

		denspt_new(gridx, gridy, gridz, &density, &densityb, &gax, &gay, &gaz, &gbx, &gby, &gbz, gid);

		d_density[gid] = density;
		d_densityb[gid] = densityb;
		d_gax[gid] = gax;
		d_gay[gid] = gay;
		d_gaz[gid] = gaz;
		d_gbx[gid] = gbx;
		d_gby[gid] = gby;
		d_gbz[gid] = gbz;

#ifdef DEBUG
//        printf("gid: %i x: %f y: %f z: %f dens: %f gax: %f gay: %f gaz: %f \n", gid, gridx, gridy, gridz, density, gax, gay, gaz);
#endif

#ifdef DEBUG
//        printf("gid: %i x: %f y: %f z: %f dens: %f densb: %f gax: %f gay: %f gaz: %f gbx: %f gby: %f gbz: %f \n", gid, d_grid_ptx[gid], d_grid_pty[gid], d_grid_ptz[gid], d_density[gid],d_densityb[gid],d_gax[gid],d_gay[gid],d_gaz[gid],d_gbx[gid],d_gby[gid],d_gbz[gid]);
#endif

	}

}
*/

/*__launch_bounds__(SM_2X_XC_THREADS_PER_BLOCK, 1)
__global__ void get_density_kernel_new(QUICKDouble* grid_ptx, QUICKDouble* grid_pty, QUICKDouble* grid_ptz, int *dweight, int *basf, int *primf, int *basf_counter, int *primf_counter, QUICKDouble* arr_density, QUICKDouble* arr_densityb, QUICKDouble* arr_gax, QUICKDouble* arr_gay, QUICKDouble* arr_gaz, QUICKDouble* arr_gbx, QUICKDouble* arr_gby, QUICKDouble* arr_gbz, int size)
{

	   
	//Given a point in space, this function calculates the densities and
	//gradient  at that point.  The gradients are stored in the common block
	//three element arrays ga and gb for alpha and beta electron gradients. Thus
	//the x, y, z component of the alpha density is stored in gax, gay, gaz.
	 

        int gid = blockIdx.x * blockDim.x + threadIdx.x;
	int bin_size = MAX_POINTS_PER_CLUSTER; //This is the bin size of octree. Should be replaced by a paramter of a macro

        if(gid<size){

                QUICKDouble density = 0.0;
                QUICKDouble gax = 0.0;
                QUICKDouble gay = 0.0;
                QUICKDouble gaz = 0.0;

		int dummy_weight = dweight[gid];

		if(dummy_weight >0){

		int bin_id = (int) (gid/bin_size);
		int nofbfs = basf_counter[bin_id+1] - basf_counter[bin_id];
		QUICKDouble gridx = grid_ptx[gid];
		QUICKDouble gridy = grid_pty[gid];
		QUICKDouble gridz = grid_ptz[gid];

		//for (int i = 0; i<devSim_dft.nbasis; i++) {
		for(int i=basf_counter[bin_id]; i<basf_counter[bin_id+1] ; i++){
			int ibas = basf[i];
			QUICKDouble phi, dphidx, dphidy, dphidz;

		//	pteval(gridx, gridy, gridz, &phi, &dphidx, &dphidy, &dphidz, i+1);		
			pteval_new(gridx, gridy, gridz, &phi, &dphidx, &dphidy, &dphidz, primf, primf_counter, ibas, i);

			if (abs(phi+dphidx+dphidy+dphidz) >= devSim_dft.DMCutoff ) {
		//	QUICKDouble denseii = LOC2(devSim_dft.dense, i, i, devSim_dft.nbasis, devSim_dft.nbasis) * phi;
			QUICKDouble denseii = LOC2(devSim_dft.dense, ibas, ibas, devSim_dft.nbasis, devSim_dft.nbasis) * phi;

			density = density + denseii * phi / 2.0;
			gax = gax + denseii * dphidx;
			gay = gay + denseii * dphidy;
			gaz = gaz + denseii * dphidz;

		        if(gid ==126031){
                        //	printf("Selected: %f %f %f i: %i ibas: %i densii: %10e dens: %10e  gax: %10e gay: %10e  gaz: %10e  \n", gridx, gridy, gridz, i, ibas, denseii, density, gax, gay, gaz);
                	}

			for(int j=i+1; j< basf_counter[bin_id+1]; j++){
				int jbas = basf[j];
		//	for (int j = i + 1; j< devSim_dft.nbasis; j++) {
				QUICKDouble phi2, dphidx2, dphidy2, dphidz2;

		//		pteval(gridx, gridy, gridz, &phi2, &dphidx2, &dphidy2, &dphidz2, j+1);
				pteval_new(gridx, gridy, gridz, &phi2, &dphidx2, &dphidy2, &dphidz2, primf, primf_counter, jbas, j);

				QUICKDouble denseij = LOC2(devSim_dft.dense, ibas, jbas, devSim_dft.nbasis, devSim_dft.nbasis);

		//		QUICKDouble denseij = LOC2(devSim_dft.dense, i, j, devSim_dft.nbasis, devSim_dft.nbasis);
				density = density + denseij * phi * phi2;
				gax = gax + denseij * ( phi * dphidx2 + phi2 * dphidx );
				gay = gay + denseij * ( phi * dphidy2 + phi2 * dphidy );
				gaz = gaz + denseij * ( phi * dphidz2 + phi2 * dphidz );
			}

			}
		}
		}
	
	arr_density[gid] = density;
	arr_densityb[gid] = density;
	arr_gax[gid] = gax;
	arr_gay[gid] = gay;
	arr_gaz[gid] = gaz;
	arr_gbx[gid] = gax;
	arr_gby[gid] = gay;
	arr_gbz[gid] = gaz;

#ifdef DEBUG
//	printf("gid: %i: bin: %i x: %f y: %f z: %f dens: %f gax: %f gay: %f gaz: %f \n", gid, bin_id, gridx, gridy, gridz, density, gax, gay, gaz);
//        printf("gid: %i x: %f y: %f z: %f dens: %f densb: %f gax: %f gay: %f gaz: %f gbx: %f gby: %f gbz: %f \n", gid, d_grid_ptx[gid], d_grid_pty[gid], d_grid_ptz[gid], d_density[gid],d_densityb[gid],d_gax[gid],d_gay[gid],d_gaz[gid],d_gbx[gid],d_gby[gid],d_gbz[gid]);
#endif

        }

}

*/

//__launch_bounds__(SM_2X_XC_THREADS_PER_BLOCK, 1)
/*__global__ void gpu_getxc_grad_new(QUICKDouble* d_grid_ptx, QUICKDouble* d_grid_pty, QUICKDouble* d_grid_ptz, QUICKDouble* d_grid_weights, QUICKDouble* d_density, QUICKDouble* d_densityb, QUICKDouble* d_gax, QUICKDouble* d_gay, QUICKDouble* d_gaz, QUICKDouble* d_gbx, QUICKDouble* d_gby, QUICKDouble* d_gbz, QUICKDouble* d_exc, QUICKDouble* exc_dev_grad, int size, QUICKDouble* tst_dfdr, QUICKDouble* tst_dot)
{
        int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if(gid<size){

		QUICKDouble gridx = d_grid_ptx[gid];
		QUICKDouble gridy = d_grid_pty[gid];
		QUICKDouble gridz = d_grid_ptz[gid];
		QUICKDouble weight = d_grid_weights[gid];
		QUICKDouble density = d_density[gid];
		QUICKDouble densityb = d_densityb[gid];
		QUICKDouble gax = d_gax[gid];
		QUICKDouble gay = d_gay[gid];
		QUICKDouble gaz = d_gaz[gid];
		QUICKDouble gbx = d_gbx[gid];
		QUICKDouble gby = d_gby[gid];
		QUICKDouble gbz = d_gbz[gid];

#ifdef DEBUG
//              printf("gid: %i x: %f y: %f z: %f dens: %f densb: %f gax: %f gay: %f gaz: %f weights: %f \n ", gid, gridx, gridy, gridz, density, densityb, gax, gay, gaz, weight);
#endif

#ifdef DEBUG
//        printf("gridx: %f  gridy: %f  gridz: %f, weight: %.10e, density: %.10e \n",gridx, gridy, gridz, weight, density);
#endif

                        QUICKDouble sigma = 4.0 * (gax * gax + gay * gay + gaz * gaz);
                        QUICKDouble _tmp ;

                        if (devSim_dft.method == B3LYP) {
                                _tmp = b3lyp_e(2.0*density, sigma);
                        }else if(devSim_dft.method == DFT){
                                _tmp = (becke_e(density, densityb, gax, gay, gaz, gbx, gby, gbz)
                                + lyp_e(density, densityb, gax, gay, gaz, gbx, gby, gbz));
                        }

                        QUICKDouble dfdr;
                        QUICKDouble dot, xdot, ydot, zdot;

                        if (devSim_dft.method == B3LYP) {
                                dot = b3lypf(2.0*density, sigma, &dfdr);
                                xdot = dot * gax;
                                ydot = dot * gay;
                                zdot = dot * gaz;
                        }else if(devSim_dft.method == DFT){

                                QUICKDouble dfdgaa, dfdgab, dfdgaa2, dfdgab2;
                                QUICKDouble dfdr2;

                                lyp(density, densityb, gax, gay, gaz, gbx, gby, gbz, &dfdr2, &dfdgaa2, &dfdgab2);
                                dfdr = dfdr2;
                                dfdgaa = dfdgaa2;
                                dfdgab = dfdgab2;		
                                // This subroutine will never run,
                                // however, it will speed up the program for about 4 times. Yes, you are right, 4 times
                                // in another word, if you delete it, the program will be slown up to 25%.
                                // you don't need to know why
                                if (false) lyp(density, densityb, gax, gay, gaz, gbx, gby, gbz, &dfdr2, &dfdgaa2, &dfdgab2);

                                //Calculate the first term in the dot product shown above,i.e.:
                                //(2 df/dgaa Grad(rho a) + df/dgab Grad(rho b)) doT Grad(Phimu Phinu))
                                xdot = 2.0 * dfdgaa * gax + dfdgab * gbx;
                                ydot = 2.0 * dfdgaa * gay + dfdgab * gby;
                                zdot = 2.0 * dfdgaa * gaz + dfdgab * gbz;
                                
                        }else if(devSim_dft.method == LIBXC){
                                //Prepare in/out for libxc call
                                double d_rhoa = (double) density;
                                double d_rhob = (double) densityb;
                                double d_sigma = (double)sigma;
                                double d_zk, d_vrho, d_vsigma;
                                d_zk = d_vrho = d_vsigma = 0.0;

                                for(int i=0; i<nof_functionals; i++){
                                        double tmp_d_zk, tmp_d_vrho, tmp_d_vsigma;
                                        tmp_d_zk=tmp_d_vrho=tmp_d_vsigma=0.0;

                                        gpu_libxc_info* tmp_glinfo = glinfo[i];

                                        switch(tmp_glinfo->gpu_worker){
                                                case GPU_WORK_LDA:
                                                        gpu_work_lda_c(tmp_glinfo, d_rhoa, d_rhob, &tmp_d_zk, &tmp_d_vrho, 1);
                                                        break;
                                        
                                                case GPU_WORK_GGA_X:
                                                        gpu_work_gga_x(tmp_glinfo, d_rhoa, d_rhob, d_sigma, &tmp_d_zk, &tmp_d_vrho, &tmp_d_vsigma);
                                                        break;

                                                case GPU_WORK_GGA_C:
                                                        gpu_work_gga_c(tmp_glinfo, d_rhoa, d_rhob, d_sigma, &tmp_d_zk, &tmp_d_vrho, &tmp_d_vsigma, 1);
                                                        break;
                                        }
                                        d_zk += (tmp_d_zk*tmp_glinfo->mix_coeff);
                                        d_vrho += (tmp_d_vrho*tmp_glinfo->mix_coeff);
                                        d_vsigma += (tmp_d_vsigma*tmp_glinfo->mix_coeff);
                                }
                        
                                _tmp = ((QUICKDouble) (d_zk * (d_rhoa + d_rhob)));                              

                                QUICKDouble dfdgaa, dfdgab, dfdgaa2, dfdgab2;
                                QUICKDouble dfdr2;
                                dfdr = (QUICKDouble)d_vrho;
                                dfdgaa = (QUICKDouble)d_vsigma*4.0;

                                xdot = dfdgaa * gax;
                                ydot = dfdgaa * gay;
                                zdot = dfdgaa * gaz;
                        }

			d_exc[gid] = _tmp;

                        tst_dfdr[gid] = dfdr;
                        tst_dot[gid] = dot;

#ifdef DEBUG
//              printf("gid: %i x: %f y: %f z: %f dens: %f densb: %f gax: %f gay: %f gaz: %f weights: %f _tmp: %f dfdr: %f dot: %f \n ", gid, gridx, gridy, gridz, density, densityb, gax, gay, gaz, weight, _tmp, dfdr, dot);
#endif

                        for (int i = 0; i< devSim_dft.nbasis; i++) {
                                QUICKDouble phi, dphidx, dphidy, dphidz;
                                pteval(gridx, gridy, gridz, &phi, &dphidx, &dphidy, &dphidz, i+1);

                                if (abs(phi+dphidx+dphidy+dphidz)> devSim_dft.DMCutoff ) {

                                        QUICKDouble dxdx, dxdy, dxdz, dydy, dydz, dzdz;

                                        dxdx = 0.0;
                                        dxdy = 0.0;
                                        dxdz = 0.0;
                                        dydy = 0.0;
                                        dydz = 0.0;
                                        dzdz = 0.0;
                                        
                                        pt2der(gridx, gridy, gridz, &dxdx, &dxdy, &dxdz, &dydy, &dydz, &dzdz, i+1);
                                        
                                        int Istart = (devSim_dft.ncenter[i]-1) * 3;

                                        for (int j = 0; j<devSim_dft.nbasis; j++) {

                                                QUICKDouble phi2, dphidx2, dphidy2, dphidz2;

                                                pteval(gridx, gridy, gridz, &phi2, &dphidx2, &dphidy2, &dphidz2, j+1);
                                                
                                                QUICKDouble denseij = (QUICKDouble) LOC2(devSim_dft.dense, i, j, devSim_dft.nbasis, devSim_dft.nbasis);
                                                
                                                QUICKDouble Gradx = - 2.0 * denseij * weight * (dfdr * dphidx * phi2
                                                                + xdot * (dxdx * phi2 + dphidx * dphidx2)
                                                                + ydot * (dxdy * phi2 + dphidx * dphidy2)
                                                                + zdot * (dxdz * phi2 + dphidx * dphidz2));

                                                QUICKDouble Grady = - 2.0 * denseij * weight * (dfdr * dphidy * phi2
                                                                + xdot * (dxdy * phi2 + dphidy * dphidx2)
                                                                + ydot * (dydy * phi2 + dphidy * dphidy2)
                                                                + zdot * (dydz * phi2 + dphidy * dphidz2));

                                                QUICKDouble Gradz = - 2.0 * denseij * weight * (dfdr * dphidz * phi2
                                                                + xdot * (dxdz * phi2 + dphidz * dphidx2)
                                                                + ydot * (dydz * phi2 + dphidz * dphidy2)
                                                                + zdot * (dzdz * phi2 + dphidz * dphidz2));

                                                atomicAdd(&(exc_dev_grad[Istart]), Gradx);
                                                atomicAdd(&(exc_dev_grad[Istart+1]), Grady);
                                                atomicAdd(&(exc_dev_grad[Istart+2]), Gradz);
                                        }

                                }
                        }
        
        }
}
*/

/*
__global__ void gpu_getxc_grad_new2(QUICKDouble* grid_ptx, QUICKDouble* grid_pty, QUICKDouble* grid_ptz, QUICKDouble* grid_weights, int *dweight, int *basf, int *primf, int *basf_counter, int *primf_counter, QUICKDouble* density_in, QUICKDouble* densityb_in, QUICKDouble* gax_in, QUICKDouble* gay_in, QUICKDouble* gaz_in, QUICKDouble* gbx_in, QUICKDouble* gby_in, QUICKDouble* gbz_in, QUICKDouble* exc, QUICKDouble* xc_dev_grad, int size, QUICKDouble* tst_dfdr, QUICKDouble* tst_dot)
{
        int gid = blockIdx.x * blockDim.x + threadIdx.x;
	int bin_size = MAX_POINTS_PER_CLUSTER; //This is the bin size of octree. Should be replaced by a paramter of a macro

        if(gid<size){

		int dummy_weight = dweight[gid];

		if(dummy_weight >0){

			int bin_id = (int) (gid/bin_size);
			int nofbfs = basf_counter[bin_id+1] - basf_counter[bin_id];			

                	QUICKDouble gridx = grid_ptx[gid];
                	QUICKDouble gridy = grid_pty[gid];
                	QUICKDouble gridz = grid_ptz[gid];
                	QUICKDouble weight = grid_weights[gid];
                	QUICKDouble density = density_in[gid];
                	QUICKDouble densityb = densityb_in[gid];
                	QUICKDouble gax = gax_in[gid];
                	QUICKDouble gay = gay_in[gid];
                	QUICKDouble gaz = gaz_in[gid];
                	QUICKDouble gbx = gbx_in[gid];
                	QUICKDouble gby = gby_in[gid];
                	QUICKDouble gbz = gbz_in[gid];

			if(density >devSim_dft.DMCutoff){

#ifdef DEBUG
//              printf("gid: %i x: %f y: %f z: %f dens: %f densb: %f gax: %f gay: %f gaz: %f weights: %f dweight: %i \n ", gid, gridx, gridy, gridz, density, densityb, gax, gay, gaz, weight, dummy_weight);
#endif

#ifdef DEBUG
//        printf("gridx: %f  gridy: %f  gridz: %f, weight: %.10e, density: %.10e \n",gridx, gridy, gridz, weight, density);
#endif

                        QUICKDouble sigma = 4.0 * (gax * gax + gay * gay + gaz * gaz);
                        QUICKDouble _tmp ;

                        if (devSim_dft.method == B3LYP) {
                                _tmp = b3lyp_e(2.0*density, sigma);
                        }else if(devSim_dft.method == DFT){
                                _tmp = (becke_e(density, densityb, gax, gay, gaz, gbx, gby, gbz)
                                + lyp_e(density, densityb, gax, gay, gaz, gbx, gby, gbz));
                        }

                        QUICKDouble dfdr;
                        QUICKDouble dot, xdot, ydot, zdot;

                        if (devSim_dft.method == B3LYP) {
                                dot = b3lypf(2.0*density, sigma, &dfdr);
                                xdot = dot * gax;
                                ydot = dot * gay;
                                zdot = dot * gaz;

			//printf("gid: %i dens: %f sigma: %f _tmp: %f dot:%f dfdr: %f \n", gid, density, sigma, _tmp, dot, dfdr);

                        }else if(devSim_dft.method == DFT){

                                QUICKDouble dfdgaa, dfdgab, dfdgaa2, dfdgab2;
                                QUICKDouble dfdr2;

                                lyp(density, densityb, gax, gay, gaz, gbx, gby, gbz, &dfdr2, &dfdgaa2, &dfdgab2);
                                dfdr = dfdr2;
                                dfdgaa = dfdgaa2;
                                dfdgab = dfdgab2;
                                // This subroutine will never run,
                                // however, it will speed up the program for about 4 times. Yes, you are right, 4 times
                                // in another word, if you delete it, the program will be slown up to 25%.
                                // you don't need to know why
                                if (false) lyp(density, densityb, gax, gay, gaz, gbx, gby, gbz, &dfdr2, &dfdgaa2, &dfdgab2);

                                //Calculate the first term in the dot product shown above,i.e.:
                                //(2 df/dgaa Grad(rho a) + df/dgab Grad(rho b)) doT Grad(Phimu Phinu))
                                xdot = 2.0 * dfdgaa * gax + dfdgab * gbx;
                                ydot = 2.0 * dfdgaa * gay + dfdgab * gby;
                                zdot = 2.0 * dfdgaa * gaz + dfdgab * gbz;

                        }else if(devSim_dft.method == LIBXC){
                                //Prepare in/out for libxc call
                                double d_rhoa = (double) density;
                                double d_rhob = (double) densityb;
                                double d_sigma = (double)sigma;
                                double d_zk, d_vrho, d_vsigma;
                                d_zk = d_vrho = d_vsigma = 0.0;

                                for(int i=0; i<nof_functionals; i++){
                                        double tmp_d_zk, tmp_d_vrho, tmp_d_vsigma;
                                        tmp_d_zk=tmp_d_vrho=tmp_d_vsigma=0.0;

                                        gpu_libxc_info* tmp_glinfo = glinfo[i];

                                        switch(tmp_glinfo->gpu_worker){
                                                case GPU_WORK_LDA:
                                                        gpu_work_lda_c(tmp_glinfo, d_rhoa, d_rhob, &tmp_d_zk, &tmp_d_vrho, 1);
                                                        break;
                                        
                                                case GPU_WORK_GGA_X:
                                                        gpu_work_gga_x(tmp_glinfo, d_rhoa, d_rhob, d_sigma, &tmp_d_zk, &tmp_d_vrho, &tmp_d_vsigma);
                                                        break;

                                                case GPU_WORK_GGA_C:
                                                        gpu_work_gga_c(tmp_glinfo, d_rhoa, d_rhob, d_sigma, &tmp_d_zk, &tmp_d_vrho, &tmp_d_vsigma, 1);
                                                        break;
                                        }
                                        d_zk += (tmp_d_zk*tmp_glinfo->mix_coeff);
                                        d_vrho += (tmp_d_vrho*tmp_glinfo->mix_coeff);
                                        d_vsigma += (tmp_d_vsigma*tmp_glinfo->mix_coeff);
                                }
                        
                                _tmp = ((QUICKDouble) (d_zk * (d_rhoa + d_rhob)));                              

                                QUICKDouble dfdgaa, dfdgab, dfdgaa2, dfdgab2;
                                QUICKDouble dfdr2;
                                dfdr = (QUICKDouble)d_vrho;
                                dfdgaa = (QUICKDouble)d_vsigma*4.0;

                                xdot = dfdgaa * gax;
                                ydot = dfdgaa * gay;
                                zdot = dfdgaa * gaz;
                        }

                        exc[gid] = _tmp;

			tst_dfdr[gid] = dfdr;
			tst_dot[gid] = dot;

#ifdef DEBUG
//printf("gid: %i x: %f y: %f z: %f dens: %f densb: %f gax: %f gay: %f gaz: %f weights: %f _tmp: %f dfdr: %f dot: %f dweight: %i \n ", gid, gridx, gridy, gridz, density, densityb, gax, gay, gaz, weight, _tmp, dfdr, dot, dummy_weight);
#endif

//			for (int i = 0; i< devSim_dft.nbasis; i++) {
                        for (int i = basf_counter[bin_id]; i<basf_counter[bin_id+1]; i++) {

				int ibas = basf[i];

                                QUICKDouble phi, dphidx, dphidy, dphidz;
				pteval_new(gridx, gridy, gridz, &phi, &dphidx, &dphidy, &dphidz, primf, primf_counter, ibas, i);
//                                pteval(gridx, gridy, gridz, &phi, &dphidx, &dphidy, &dphidz, i+1);


                                if (abs(phi+dphidx+dphidy+dphidz)> devSim_dft.DMCutoff ) {

                                        QUICKDouble dxdx, dxdy, dxdz, dydy, dydz, dzdz;
					dxdx = 0.0;
					dxdy = 0.0;
					dxdz = 0.0;
					dydy = 0.0;
					dydz = 0.0;
					dzdz = 0.0;
				
					
                                       pt2der_new(gridx, gridy, gridz, &dxdx, &dxdy, &dxdz, &dydy, &dydz, &dzdz, primf, primf_counter, ibas, i);
//					pt2der(gridx, gridy, gridz, &dxdx, &dxdy, &dxdz, &dydy, &dydz, &dzdz, i+1);
                                        int Istart = (devSim_dft.ncenter[ibas]-1) * 3;
//					int Istart = (devSim_dft.ncenter[i]-1) * 3;

//					for (int j = 0; j<devSim_dft.nbasis; j++) {
                                        for (int j = basf_counter[bin_id]; j <basf_counter[bin_id+1]; j++) {
						int jbas = basf[j];

                                                QUICKDouble phi2, dphidx2, dphidy2, dphidz2;
						pteval_new(gridx, gridy, gridz, &phi2, &dphidx2, &dphidy2, &dphidz2, primf, primf_counter, jbas, j);
//                                                pteval(gridx, gridy, gridz, &phi2, &dphidx2, &dphidy2, &dphidz2, j+1);
				

                                                QUICKDouble denseij = (QUICKDouble) LOC2(devSim_dft.dense, ibas, jbas, devSim_dft.nbasis, devSim_dft.nbasis);

//						QUICKDouble denseij = (QUICKDouble) LOC2(devSim_dft.dense, i, j, devSim_dft.nbasis, devSim_dft.nbasis);

						

                                                QUICKDouble Gradx = - 2.0 * denseij * weight * (dfdr * dphidx * phi2
                                                                + xdot * (dxdx * phi2 + dphidx * dphidx2)
                                                                + ydot * (dxdy * phi2 + dphidx * dphidy2)
                                                                + zdot * (dxdz * phi2 + dphidx * dphidz2));

                                                QUICKDouble Grady = - 2.0 * denseij * weight * (dfdr * dphidy * phi2
                                                                + xdot * (dxdy * phi2 + dphidy * dphidx2)
                                                                + ydot * (dydy * phi2 + dphidy * dphidy2)
                                                                + zdot * (dydz * phi2 + dphidy * dphidz2));

                                                QUICKDouble Gradz = - 2.0 * denseij * weight * (dfdr * dphidz * phi2
                                                                + xdot * (dxdz * phi2 + dphidz * dphidx2)
                                                                + ydot * (dydz * phi2 + dphidz * dphidy2)
                                                                + zdot * (dzdz * phi2 + dphidz * dphidz2));

			//			printf(" gid: %i x: %f y: %f z: %f density: %f sigma: %f denseij: %f weight: %f xdot: %f ydot: %f zdot: %f dfdr: %f \n", gid, gridx, gridy, gridz, density, sigma, denseij, weight, xdot, ydot, zdot, dfdr);

                                                atomicAdd(&(xc_dev_grad[Istart]), Gradx);
                                                atomicAdd(&(xc_dev_grad[Istart+1]), Grady);
                                                atomicAdd(&(xc_dev_grad[Istart+2]), Gradz);
                                        }

                                }
                        }
			}
		}
        }
}
*/

/*__launch_bounds__(SM_2X_XC_THREADS_PER_BLOCK, 1)
__global__ void get_sswder_grad(QUICKDouble* d_grid_ptx, QUICKDouble* d_grid_pty, QUICKDouble* d_grid_ptz, QUICKDouble* d_exc, QUICKDouble* d_grid_sswt, QUICKDouble* d_grid_weights, int* d_grid_atm, QUICKDouble* exc_dev_grad, int size)
{
        int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if(gid<size){

                int iatm = d_grid_atm[gid];
                QUICKDouble sswt = d_grid_sswt[gid];
                QUICKDouble weight = d_grid_weights[gid];
		
		sswder(d_grid_ptx[gid], d_grid_pty[gid], d_grid_ptz[gid], d_exc[gid], weight/sswt, iatm, exc_dev_grad);
#ifdef DEBUG
//              printf("gid: %i d_grid_ptx: %f d_grid_pty: %f d_grid_ptz: %f d_grid_weights: %f \n", gid, d_grid_ptx[gid], d_grid_pty[gid], d_grid_ptz[gid],d_grid_weights[gid]);
#endif
        }
}
*/

//Madu Manathunga 07/01/2019 added libxc variable
/*__launch_bounds__(SM_2X_XC_THREADS_PER_BLOCK, 1)
__global__ void getxc_kernel(gpu_libxc_info** glinfo, int nof_functionals, double* exc_dev_grad)
{
    unsigned int offset = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int totalThreads = blockDim.x * gridDim.x;
    QUICKULL currentPoint = 0;
    QUICKULL myPoint = offset;
    
    int radTotal = 50; // SG1 default value.
    int iiangt = 0;    // number of ;angular point for every radical.  
    
    // x, y, z value and weight value.
    QUICKDouble XAng[MAX_GRID];
    QUICKDouble YAng[MAX_GRID];
    QUICKDouble ZAng[MAX_GRID];
    QUICKDouble WAng[MAX_GRID];
    
    for (int i = 0; i< devSim_dft.natom; i++) {
        if (devSim_dft.isg == 0) {
            if (devSim_dft.iattype[i]<=10) {
                radTotal = 23;
            }else {
                radTotal = 26;
            }
        }
        
        for (int j = 0; j<radTotal; j++) {
             
            // Generates grids points according to central atom type and 
            // distance to atoms. See subroutines for details.
            if (devSim_dft.isg == 1){ // SG1 scheme
                iiangt = gridFormSG1(i+1, RGRID[j], XAng, YAng, ZAng, WAng);
            }else {                   // SG0 scheme
                // iiangt = gridFormSG0(*atm, *iradtemp+1-*irad, RGRID, RWT);
            }
            
            if (currentPoint <= myPoint && currentPoint + iiangt > myPoint) {
                unsigned int pointId = (unsigned int) myPoint - currentPoint;

                gpu_grid_xc(j+1, radTotal, i+1, XAng[pointId], YAng[pointId], ZAng[pointId], WAng[pointId], 
		glinfo, nof_functionals);

                myPoint = myPoint + totalThreads;
            }
            currentPoint = currentPoint + iiangt;
        }
    }
}*/

/*
__launch_bounds__(SM_2X_XC_THREADS_PER_BLOCK, 1)
__global__ void getxc_grad_kernel(gpu_libxc_info** glinfo, int nof_functionals, double* exc_dev_grad)
{

    unsigned int offset = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int totalThreads = blockDim.x * gridDim.x;
    QUICKULL currentPoint = 0;
    QUICKULL myPoint = offset;

    int radTotal = 50; // SG1 default value.
    int iiangt = 0;    // number of ;angular point for every radical.  

    // x, y, z value and weight value.
    QUICKDouble XAng[MAX_GRID];
    QUICKDouble YAng[MAX_GRID];
    QUICKDouble ZAng[MAX_GRID];
    QUICKDouble WAng[MAX_GRID];

    for (int i = 0; i< devSim_dft.natom; i++) {
        if (devSim_dft.isg == 0) {
            if (devSim_dft.iattype[i]<=10) {
                radTotal = 23;
            }else {
                radTotal = 26;
            }
        }

        for (int j = 0; j<radTotal; j++) {

            // Generates grids points according to central atom type and 
            // distance to atoms. See subroutines for details.
            if (devSim_dft.isg == 1){ // SG1 scheme
                iiangt = gridFormSG1(i+1, RGRID[j], XAng, YAng, ZAng, WAng);
            }else {                   // SG0 scheme
                // iiangt = gridFormSG0(*atm, *iradtemp+1-*irad, RGRID, RWT);
            }

            if (currentPoint <= myPoint && currentPoint + iiangt > myPoint) {
                unsigned int pointId = (unsigned int) myPoint - currentPoint;

                gpu_grid_xc_grad(j+1, radTotal, i+1, XAng[pointId], YAng[pointId], ZAng[pointId], WAng[pointId],
                glinfo, nof_functionals, exc_dev_grad);

                myPoint = myPoint + totalThreads;
            }
            currentPoint = currentPoint + iiangt;
        }
    }

}
*/

/*
 This subroutine is to get energy, electron density, deviation and operator change
 if given a point in grid.
 */
//Madu Manathunga 07/01/2019 added libxc variable
/*__device__ void gpu_grid_xc(int irad, int iradtemp, int iatm, QUICKDouble XAng, QUICKDouble YAng, QUICKDouble ZAng, QUICKDouble WAng, gpu_libxc_info** glinfo, int nof_functionals){
    
    QUICKDouble rad, rad3;    
    QUICKDouble atomx, atomy, atomz;
    
    
     //Read atom coordinates(atomx, atomy and atomz) from global memory (DRAM). If the atom is shared by the 
     //whole block, we could put these variables into block memory (shared memory)
     
    atomx = LOC2(devSim_dft.xyz, 0, iatm-1, 3, devSim_dft.natom);
    atomy = LOC2(devSim_dft.xyz, 1, iatm-1, 3, devSim_dft.natom);
    atomz = LOC2(devSim_dft.xyz, 2, iatm-1, 3, devSim_dft.natom);
    
    if (devSim_dft.isg == 1){
        rad = radii[devSim_dft.iattype[iatm-1]-1];
        rad3 = pow(rad,3) * RWT[irad-1];
    }else {
        rad = radii2[devSim_dft.iattype[iatm-1]-1];
        rad3 = pow(rad,3) * RWT[irad-1];
    }
    
    // grid point coordinates.
    QUICKDouble gridx = atomx + rad * RGRID[irad-1] * XAng;
    QUICKDouble gridy = atomy + rad * RGRID[irad-1] * YAng;
    QUICKDouble gridz = atomz + rad * RGRID[irad-1] * ZAng;

#ifdef DEBUG
        //printf("gridx: %f, gridy: %f, gridz: %f \n",gridx, gridy, gridz);
#endif
    
    // calculate Scuseria-Stratmann weights, and times rad3 and the point basic weights to get comprhensive point weight
    QUICKDouble weight = SSW(gridx, gridy, gridz, iatm) * WAng * rad3;

#ifdef DEBUG
	//printf("gridx: %f  gridy: %f  gridz: %f, weight: %.10e \n",gridx, gridy, gridz, weight);
       // printf("weight: %.10e, devSim_dft.DMCutoff: %.10e, devSim_dft.isg: %d \n",weight, devSim_dft.DMCutoff, devSim_dft.isg);
#endif
    
    if (weight > devSim_dft.DMCutoff ) { 
        
        QUICKDouble density, densityb;
        QUICKDouble gax, gay, gaz;
        QUICKDouble gbx, gby, gbz;
        denspt(gridx, gridy, gridz, &density, &densityb, &gax, &gay, &gaz, &gbx, &gby, &gbz);

#ifdef DEBUG
       // printf("gridx: %f  gridy: %f  gridz: %f, weight: %.10e, density: %.10e \n",gridx, gridy, gridz, weight, density);
//        printf("rho: %f, rhob: %f, weight: %.10e, devSim_dft.DMCutoff: %.10e \n", density, densityb, weight, devSim_dft.DMCutoff);
#endif
        
        if (density > devSim_dft.DMCutoff ) { 
            QUICKDouble sigma = 4.0 * (gax * gax + gay * gay + gaz * gaz);

#ifdef DEBUG
//        printf("gridx: %f  gridy: %f  gridz: %f, weight: %.10e, density: %.10e sigma: %.10e \n",gridx, gridy, gridz, weight, density, sigma);
//        printf("rho: %f, rhob: %f, weight: %.10e, devSim_dft.DMCutoff: %.10e \n", density, densityb, weight, devSim_dft.DMCutoff);
#endif
  
            QUICKDouble _tmp ;
            
            if (devSim_dft.method == B3LYP) {

#ifdef DEBUG
//    printf("FILE: %s, LINE: %d, FUNCTION: %s,  B3LYP enregy \n", __FILE__, __LINE__, __func__);
#endif
                _tmp = b3lyp_e(2.0*density, sigma) * weight;
            }else if(devSim_dft.method == DFT){// !!! remember to change it to BLYP
		_tmp = (becke_e(density, densityb, gax, gay, gaz, gbx, gby, gbz)
                + lyp_e(density, densityb, gax, gay, gaz, gbx, gby, gbz)) * weight;
#ifdef DEBUG
 //printf("gridx: %f  gridy: %f  gridz: %f, weight: %.10e, density: %.10e sigma: %.10e _tmp: %.10e \n",gridx, gridy, gridz, weight, density, sigma, _tmp);
//                printf("rho: %.10e sigma: %.10e _tmp/weight: %.10e \n", (density+densityb), sigma, _tmp/weight);
#endif
            }else if(devSim_dft.method == LIBXC){ //Madu: Change this conditional statement content
               // _tmp = (becke_e(density, densityb, gax, gay, gaz, gbx, gby, gbz)
               // + lyp_e(density, densityb, gax, gay, gaz, gbx, gby, gbz)) * weight;
	    }

            QUICKDouble dfdr;
            QUICKDouble dot, xdot, ydot, zdot;
            
            if (devSim_dft.method == B3LYP) {
                dot = b3lypf(2.0*density, sigma, &dfdr);
                xdot = dot * gax;
                ydot = dot * gay;
                zdot = dot * gaz;
#ifdef DEBUG
//    printf("FILE: %s, LINE: %d, FUNCTION: %s, devSim_dft.method == B3LYP \n", __FILE__, __LINE__, __func__);
#endif

            }else if (devSim_dft.method == DFT){
                
                // This allows the calculation of the derivative of the functional
                // with regard to the density (dfdr), with regard to the alpha-alpha
                // density invariant (df/dgaa), and the alpha-beta density invariant.

                QUICKDouble dfdgaa, dfdgab, dfdgaa2, dfdgab2;
                QUICKDouble dfdr2;
                becke(density, gax, gay, gaz, gbx, gby, gbz, &dfdr, &dfdgaa, &dfdgab);
                lyp(density, densityb, gax, gay, gaz, gbx, gby, gbz, &dfdr2, &dfdgaa2, &dfdgab2);

                dfdr += dfdr2;
                dfdgaa += dfdgaa2;
                dfdgab += dfdgab2;

#ifdef DEBUG
//		printf("rho: %.10e sigma: %.10e _tmp/weight: %.10e dfdr: %.10e dfdgaa: %.10e dfdgab: %.10e \n", density, sigma, _tmp/weight, dfdr2, dfdgaa2, dfdgab2);
#endif    
                // This subroutine will never run,
                // however, it will speed up the program for about 4 times. Yes, you are right, 4 times
                // in another word, if you delete it, the program will be slown up to 25%.
                // you don't need to know why
                if (false) lyp(density, densityb, gax, gay, gaz, gbx, gby, gbz, &dfdr2, &dfdgaa2, &dfdgab2); 
                
                //Calculate the first term in the dot product shown above,i.e.:
                //(2 df/dgaa Grad(rho a) + df/dgab Grad(rho b)) doT Grad(Phimu Phinu))
                xdot = 2.0 * dfdgaa * gax + dfdgab * gbx;
                ydot = 2.0 * dfdgaa * gay + dfdgab * gby;
                zdot = 2.0 * dfdgaa * gaz + dfdgab * gbz;
            }else if(devSim_dft.method == LIBXC){ //Madu: Change this conditional statement content

#ifdef DEBUG
    //printf("FILE: %s, LINE: %d, FUNCTION: %s, devSim_dft.method == LIBXC \n", __FILE__, __LINE__, __func__);
#endif

		//Prepare in/out for libxc call
		double d_rhoa = (double) density;
		double d_rhob = (double) densityb;
		double d_sigma = (double)sigma;
		double d_zk, d_vrho, d_vsigma;
		d_zk = d_vrho = d_vsigma = 0.0;

		for(int i=0; i<nof_functionals; i++){

			double tmp_d_zk, tmp_d_vrho, tmp_d_vsigma;
			tmp_d_zk=tmp_d_vrho=tmp_d_vsigma=0.0;

			gpu_libxc_info* tmp_glinfo = glinfo[i];
			
			switch(tmp_glinfo->gpu_worker){
				case GPU_WORK_LDA:

					gpu_work_lda_c(tmp_glinfo, d_rhoa, d_rhob, &tmp_d_zk, &tmp_d_vrho, 1);
#ifdef DEBUG
//                printf("func_id: %d rho: %.10e sigma: %.10e tmp_d_zk: %.10e tmp_d_vrho: %.10e \n", tmp_glinfo->func_id, (d_rhoa+d_rhob), 0.0, tmp_d_zk, tmp_d_vrho );
#endif
					break;
				case GPU_WORK_GGA_X:
					gpu_work_gga_x(tmp_glinfo, d_rhoa, d_rhob, d_sigma, &tmp_d_zk, &tmp_d_vrho, &tmp_d_vsigma);
#ifdef DEBUG
//                printf("func_id: %d rho: %.10e sigma: %.10e tmp_d_zk: %.10e tmp_d_vrho: %.10e tmp_d_sigma: %.10e \n", tmp_glinfo->func_id, (d_rhoa+d_rhob), d_sigma, tmp_d_zk, tmp_d_vrho, tmp_d_vsigma );
#endif
					break;
				case GPU_WORK_GGA_C:
					gpu_work_gga_c(tmp_glinfo, d_rhoa, d_rhob, d_sigma, &tmp_d_zk, &tmp_d_vrho, &tmp_d_vsigma, 1);
#ifdef DEBUG                            
//                printf("func_id: %d rho: %.10e sigma: %.10e tmp_d_zk: %.10e tmp_d_vrho: %.10e tmp_d_sigma: %.10e \n", tmp_glinfo->func_id, (d_rhoa+d_rhob), d_sigma, tmp_d_zk, tmp_d_vrho, tmp_d_vsigma );
#endif
					break;
			}			

			d_zk += (tmp_d_zk*tmp_glinfo->mix_coeff) ;
			d_vrho += (tmp_d_vrho*tmp_glinfo->mix_coeff) ;
			d_vsigma += (tmp_d_vsigma*tmp_glinfo->mix_coeff);
#ifdef DEBUG
	//printf("FILE: %s, LINE: %d, FUNCTION: %s, nof_functionals: %d, functional_id: %d, worker_id: %d, mix_coeff: %f \n", __FILE__, __LINE__, __func__, nof_functionals, tmp_glinfo->func_id, tmp_glinfo->gpu_worker, tmp_glinfo->mix_coeff);
//                printf("func_id: %d rho: %.10e zk: %.10e d_vrho: %.10e d_vsigma: %.10e \n", tmp_glinfo->func_id,(d_rhoa+d_rhob), d_zk, d_vrho, d_vsigma);
#endif
		}

		_tmp = ((QUICKDouble) (d_zk * (d_rhoa + d_rhob))) * weight;

#ifdef DEBUG
//	printf("rho: %.10e sigma: %.10e d_zk: %.10e  d_vrho: %.10e  d_vsigma: %.10e \n", d_rho_sum, d_sigma, d_zk, d_vrho, d_vsigma);
//	 printf("gridx: %f  gridy: %f  gridz: %f, weight: %.10e, density: %.10e sigma: %.10e _tmp: %.10e \n",gridx, gridy, gridz, weight, density, sigma, _tmp);
		//printf("rho: %f, d_rho[1]: %f, sigma: %f, d_sigma[1]: %f, d_zk[0]: %.10e \n", (density+densityb), d_rho[0], sigma, d_sigma[0],d_zk[0]);
        //printf("FILE: %s, LINE: %d, FUNCTION: %s, rho: %f, sigma: %f, zk: %f, vrho: %f, vsigma: %f \n", __FILE__, __LINE__, __func__, d_rho[0],
        //d_sigma[0], d_zk[0], d_vrho[0], d_vsigma[0]);
#endif
	
		QUICKDouble dfdgaa, dfdgab, dfdgaa2, dfdgab2;
                QUICKDouble dfdr2;

		// *******************************Only for testing********************************
		//d_vsigma = 0.0;
		// *******************************Only for testing********************************

		dfdr = (QUICKDouble)d_vrho;
		dfdgaa = (QUICKDouble)d_vsigma*4.0;
		//dfdgab = (QUICKDouble)d_vsigma*4.0; //Currently we can only handle closed shell systems		

#ifdef DEBUG
        //printf("FILE: %s, LINE: %d, FUNCTION: %s, dfdgaa: %f, dfdgab: %f, gax: %f, gbx: %f \n", __FILE__, __LINE__, __func__, dfdgaa, dfdgab, gax, gbx);
#endif

                if (false) lyp(density, densityb, gax, gay, gaz, gbx, gby, gbz, &dfdr2, &dfdgaa2, &dfdgab2);

                //xdot = 2.0 * dfdgaa * gax + dfdgab * gbx;
                //ydot = 2.0 * dfdgaa * gay + dfdgab * gby;
                //zdot = 2.0 * dfdgaa * gaz + dfdgab * gbz;

		xdot = dfdgaa * gax;
		ydot = dfdgaa * gay;
		zdot = dfdgaa * gaz;

#ifdef DEBUG
        //printf("FILE: %s, LINE: %d, FUNCTION: %s, xdot: %f, ydot: %f, zdot: %f \n", __FILE__, __LINE__, __func__, xdot, ydot, zdot);
//	printf("rho: %.10e d_zk: %.10e dfdr: %.10e dfdgaa: %.10e xdot: %.10e ydot: %.10e zdot: %.10e \n", (d_rhoa + d_rhob), d_zk, dfdr, dfdgaa, xdot, ydot, zdot);
#endif

	    }

	           
            QUICKULL val1 = (QUICKULL) (fabs( _tmp * OSCALE) + (QUICKDouble)0.5);
            if ( _tmp * weight < (QUICKDouble)0.0)
                val1 = 0ull - val1;
            QUICKADD(devSim_dft.DFT_calculated[0].Eelxc, val1);

            _tmp = weight*density;
            val1 = (QUICKULL) (fabs( _tmp * OSCALE) + (QUICKDouble)0.5);
            if ( _tmp * weight < (QUICKDouble)0.0)
                val1 = 0ull - val1;
            QUICKADD(devSim_dft.DFT_calculated[0].aelec, val1);


            _tmp = weight*densityb;
            val1 = (QUICKULL) (fabs( _tmp * OSCALE) + (QUICKDouble)0.5);
            if ( _tmp * weight < (QUICKDouble)0.0)
                val1 = 0ull - val1;
            QUICKADD(devSim_dft.DFT_calculated[0].belec, val1); 

            
            for (int i = 0; i< devSim_dft.nbasis; i++) {
                QUICKDouble phi, dphidx, dphidy, dphidz;
                pteval(gridx, gridy, gridz, &phi, &dphidx, &dphidy, &dphidz, i+1);
                
                if (abs(phi+dphidx+dphidy+dphidz)> devSim_dft.DMCutoff ) { 
                    for (int j = i; j<devSim_dft.nbasis; j++) {
                        QUICKDouble phi2, dphidx2, dphidy2, dphidz2;
                        pteval(gridx, gridy, gridz, &phi2, &dphidx2, &dphidy2, &dphidz2, j+1);
                        
                        QUICKDouble _tmp = (phi * phi2 * dfdr + xdot * (phi*dphidx2 + phi2*dphidx) \
                                            + ydot * (phi*dphidy2 + phi2*dphidy) + zdot * (phi*dphidz2 + phi2*dphidz))*weight;
                        
                        QUICKULL val1 = (QUICKULL) (fabs( _tmp * OSCALE) + (QUICKDouble)0.5);
                        if ( _tmp * weight < (QUICKDouble)0.0)
                            val1 = 0ull - val1;                               
                        QUICKADD(LOC2(devSim_dft.oULL, j, i, devSim_dft.nbasis, devSim_dft.nbasis), val1);
                        
                    }
                }
            }
        }
    }
}
*/

/*	Madu Manathunga 08/16/2019
	Following method calls the device kernel that computes exchange correlation
	gradients
*/
/*__device__ void gpu_grid_xc_grad(int irad, int iradtemp, int iatm, QUICKDouble XAng, QUICKDouble YAng, QUICKDouble ZAng, QUICKDouble WAng, gpu_libxc_info** glinfo, int nof_functionals, double* exc_dev_grad){

    QUICKDouble rad, rad3;
    QUICKDouble atomx, atomy, atomz;

    
     //Read atom coordinates(atomx, atomy and atomz) from global memory (DRAM). If the atom is shared by the 
     //whole block, we could put these variables into block memory (shared memory)
     
    atomx = LOC2(devSim_dft.xyz, 0, iatm-1, 3, devSim_dft.natom);
    atomy = LOC2(devSim_dft.xyz, 1, iatm-1, 3, devSim_dft.natom);
    atomz = LOC2(devSim_dft.xyz, 2, iatm-1, 3, devSim_dft.natom);

    if (devSim_dft.isg == 1){
        rad = radii[devSim_dft.iattype[iatm-1]-1];
        rad3 = pow(rad,3) * RWT[irad-1];
    }else {
        rad = radii2[devSim_dft.iattype[iatm-1]-1];
        rad3 = pow(rad,3) * RWT[irad-1];
    }


    // grid point coordinates.
    QUICKDouble gridx = atomx + rad * RGRID[irad-1] * XAng;
    QUICKDouble gridy = atomy + rad * RGRID[irad-1] * YAng;
    QUICKDouble gridz = atomz + rad * RGRID[irad-1] * ZAng;
*/
/*	QUICKDouble sswt = SSW(gridx, gridy, gridz, iatm);
    QUICKDouble weight = sswt * WAng * rad3;


    if (weight > devSim_dft.DMCutoff ) {


        QUICKDouble density, densityb;
        QUICKDouble gax, gay, gaz;
        QUICKDouble gbx, gby, gbz;
        denspt(gridx, gridy, gridz, &density, &densityb, &gax, &gay, &gaz, &gbx, &gby, &gbz);

		if (density > devSim_dft.DMCutoff ) {

			QUICKDouble sigma = 4.0 * (gax * gax + gay * gay + gaz * gaz);
			QUICKDouble _tmp ;

			if (devSim_dft.method == B3LYP) {
				_tmp = b3lyp_e(2.0*density, sigma);
			}else if(devSim_dft.method == DFT){
				_tmp = (becke_e(density, densityb, gax, gay, gaz, gbx, gby, gbz)
				+ lyp_e(density, densityb, gax, gay, gaz, gbx, gby, gbz));
			}

			QUICKDouble dfdr;
			QUICKDouble dot, xdot, ydot, zdot;

			if (devSim_dft.method == B3LYP) {
				dot = b3lypf(2.0*density, sigma, &dfdr);
				xdot = dot * gax;
				ydot = dot * gay;
				zdot = dot * gaz;
			}else if(devSim_dft.method == DFT){

				QUICKDouble dfdgaa, dfdgab, dfdgaa2, dfdgab2;
				QUICKDouble dfdr2;

				lyp(density, densityb, gax, gay, gaz, gbx, gby, gbz, &dfdr2, &dfdgaa2, &dfdgab2);
				dfdr = dfdr2;
				dfdgaa = dfdgaa2;
				dfdgab = dfdgab2;

				// This subroutine will never run,
				// however, it will speed up the program for about 4 times. Yes, you are right, 4 times
				// in another word, if you delete it, the program will be slown up to 25%.
				// you don't need to know why
				if (false) lyp(density, densityb, gax, gay, gaz, gbx, gby, gbz, &dfdr2, &dfdgaa2, &dfdgab2);

				//Calculate the first term in the dot product shown above,i.e.:
				//(2 df/dgaa Grad(rho a) + df/dgab Grad(rho b)) doT Grad(Phimu Phinu))
				xdot = 2.0 * dfdgaa * gax + dfdgab * gbx;
				ydot = 2.0 * dfdgaa * gay + dfdgab * gby;
				zdot = 2.0 * dfdgaa * gaz + dfdgab * gbz;
				
			}else if(devSim_dft.method == LIBXC){
				//Prepare in/out for libxc call
				double d_rhoa = (double) density;
				double d_rhob = (double) densityb;
				double d_sigma = (double)sigma;
				double d_zk, d_vrho, d_vsigma;
				d_zk = d_vrho = d_vsigma = 0.0;

				for(int i=0; i<nof_functionals; i++){
					double tmp_d_zk, tmp_d_vrho, tmp_d_vsigma;
					tmp_d_zk=tmp_d_vrho=tmp_d_vsigma=0.0;

					gpu_libxc_info* tmp_glinfo = glinfo[i];

					switch(tmp_glinfo->gpu_worker){
						case GPU_WORK_LDA:
							gpu_work_lda_c(tmp_glinfo, d_rhoa, d_rhob, &tmp_d_zk, &tmp_d_vrho, 1);
							break;
					
						case GPU_WORK_GGA_X:
							gpu_work_gga_x(tmp_glinfo, d_rhoa, d_rhob, d_sigma, &tmp_d_zk, &tmp_d_vrho, &tmp_d_vsigma);
							break;

						case GPU_WORK_GGA_C:
							gpu_work_gga_c(tmp_glinfo, d_rhoa, d_rhob, d_sigma, &tmp_d_zk, &tmp_d_vrho, &tmp_d_vsigma, 1);
							break;
					}
					d_zk += (tmp_d_zk*tmp_glinfo->mix_coeff);
					d_vrho += (tmp_d_vrho*tmp_glinfo->mix_coeff);
					d_vsigma += (tmp_d_vsigma*tmp_glinfo->mix_coeff);
				}
			
				_tmp = ((QUICKDouble) (d_zk * (d_rhoa + d_rhob)));				

				QUICKDouble dfdgaa, dfdgab, dfdgaa2, dfdgab2;
				QUICKDouble dfdr2;
				dfdr = (QUICKDouble)d_vrho;
				dfdgaa = (QUICKDouble)d_vsigma*4.0;

				xdot = dfdgaa * gax;
				ydot = dfdgaa * gay;
				zdot = dfdgaa * gaz;
			}

	
			for (int i = 0; i< devSim_dft.nbasis; i++) {
				QUICKDouble phi, dphidx, dphidy, dphidz;
				pteval(gridx, gridy, gridz, &phi, &dphidx, &dphidy, &dphidz, i+1);

				if (abs(phi+dphidx+dphidy+dphidz)> devSim_dft.DMCutoff ) {

					QUICKDouble dxdx, dxdy, dxdz, dydy, dydz, dzdz;
					
					pt2der(gridx, gridy, gridz, &dxdx, &dxdy, &dxdz, &dydy, &dydz, &dzdz, i+1);
					
					int Istart = (devSim_dft.ncenter[i]-1) * 3;

					for (int j = 0; j<devSim_dft.nbasis; j++) {

						QUICKDouble phi2, dphidx2, dphidy2, dphidz2;

						pteval(gridx, gridy, gridz, &phi2, &dphidx2, &dphidy2, &dphidz2, j+1);
						
						QUICKDouble denseij = (QUICKDouble) LOC2(devSim_dft.dense, i, j, devSim_dft.nbasis, devSim_dft.nbasis);
						
						QUICKDouble Gradx = - 2.0 * denseij * weight * (dfdr * dphidx * phi2
								+ xdot * (dxdx * phi2 + dphidx * dphidx2)
								+ ydot * (dxdy * phi2 + dphidx * dphidy2)
								+ zdot * (dxdz * phi2 + dphidx * dphidz2));

//        printf("iatm: %d, gridx: %f, gridy: %f, gridz: %f, i: %d, j:%d, denseij: %f, weight: %f, dfdr: %f, dphidx: %f, phi2: %f, dxdx: %f, dxdy: %f, dxdz: %f, dphidx2: %f, dphidy2: %f, dphidz2: %f, xdot: %f, ydot: %f, zdot: %f, Gradx: %f \n", iatm, gridx, gridy, gridz, i, j, denseij, weight, dfdr, dphidx, phi2, dxdx, dxdy, dxdz, dphidx2, dphidy2, dphidz2, xdot, ydot, zdot, Gradx); 
				
						QUICKDouble Grady = - 2.0 * denseij * weight * (dfdr * dphidy * phi2
								+ xdot * (dxdy * phi2 + dphidy * dphidx2)
								+ ydot * (dydy * phi2 + dphidy * dphidy2)
								+ zdot * (dydz * phi2 + dphidy * dphidz2));

						QUICKDouble Gradz = - 2.0 * denseij * weight * (dfdr * dphidz * phi2
								+ xdot * (dxdz * phi2 + dphidz * dphidx2)
								+ ydot * (dydz * phi2 + dphidz * dphidy2)
								+ zdot * (dzdz * phi2 + dphidz * dphidz2));

						atomicAdd(&(exc_dev_grad[Istart]), Gradx);
						atomicAdd(&(exc_dev_grad[Istart+1]), Grady);
						atomicAdd(&(exc_dev_grad[Istart+2]), Gradz);
					}

				}
			}

			if(sswt != 1.0){
				sswder(gridx, gridy, gridz, _tmp, weight/sswt, iatm, exc_dev_grad);
			}
		}
	}
*/
//}


__device__ int gridFormSG1(int iitype, QUICKDouble distance, \
    QUICKDouble* XAng, QUICKDouble* YAng, QUICKDouble* ZAng, QUICKDouble* WAng){
    int iiang;
    int N = 0;
    
    if (devSim_dft.iattype[iitype-1] >= 1 && devSim_dft.iattype[iitype-1] <=2) {
        if (distance<hpartpara[0]) {
            iiang = 6;
        }else if (distance<hpartpara[1]) {
            iiang = 38;
        }else if (distance<hpartpara[2]) {
            iiang = 86;
        }else if (distance<hpartpara[3]) {
            iiang = 194;
        }else{
            iiang = 86;
        }
    }else if (devSim_dft.iattype[iitype-1] >= 3 && devSim_dft.iattype[iitype-1] <=10) {
        if (distance<lpartpara[0]) {
            iiang = 6;
        }else if (distance<lpartpara[1]) {
            iiang = 38;
        }else if (distance<lpartpara[2]) {
            iiang = 86;
        }else if (distance<lpartpara[3]) {
            iiang = 194;
        }else{
            iiang = 86;
        }
    }else if (devSim_dft.iattype[iitype-1]>= 11 && devSim_dft.iattype[iitype-1]<=18) {
        if (distance<npartpara[0]) {
            iiang = 6;
        }else if (distance<npartpara[1]) {
            iiang = 38;
        }else if (distance<npartpara[2]) {
            iiang = 86;
        }else if (distance<npartpara[3]) {
            iiang = 194;
        }else{
            iiang = 86;
        }
    } else {
        iiang = 194;
    }
    
    if (iiang == 6) {
        LD0006(XAng, YAng, ZAng, WAng, N);
    }else if (iiang == 38)  {
        LD0038(XAng, YAng, ZAng, WAng, N);
    }else if (iiang == 86)  {
        LD0086(XAng, YAng, ZAng, WAng, N);
    }else if (iiang == 194) {
        LD0194(XAng, YAng, ZAng, WAng, N);
    }
    
    for (int i = 0; i<iiang; i++) {
        WAng[i] = WAng[i] * 12.56637061435917295385;
    }
    return iiang;
}

__device__ QUICKDouble SSW( QUICKDouble gridx, QUICKDouble gridy, QUICKDouble gridz, int atm)
{
    
    /*
     This subroutie calculates the Scuseria-Stratmann wieghts.  There are
     two conditions that cause the weights to be unity: If there is only
     one atom:
    */
    QUICKDouble ssw;
    if (devSim_dft.natom == 1) {
        ssw = 1.0e0;
        return ssw;
    }
    
    /*
     Another time the weight is unity is r(iparent,g)<.5*(1-a)*R(i,n)
     where r(iparent,g) is the distance from the parent atom to the grid
     point, a is a parameter (=.64) and R(i,n) is the distance from the
     parent atom to it's nearest neighbor.
    */
    
    QUICKDouble xparent = LOC2(devSim_dft.xyz, 0, atm-1, 3, devSim_dft.natom);
    QUICKDouble yparent = LOC2(devSim_dft.xyz, 1, atm-1, 3, devSim_dft.natom);
    QUICKDouble zparent = LOC2(devSim_dft.xyz, 2, atm-1, 3, devSim_dft.natom);
    
    QUICKDouble rig = sqrt(pow((gridx-xparent),2) + 
                           pow((gridy-yparent),2) + 
                           pow((gridz-zparent),2)); 

    /* !!!! this part can be done in CPU*/
    QUICKDouble distnbor = 1e3;
    for (int i = 0; i<devSim_dft.natom; i++) {
        if (i != atm-1) {        
            QUICKDouble distance = sqrt(pow(xparent - LOC2(devSim_dft.xyz, 0, i, 3, devSim_dft.natom),2) + 
                                    pow(yparent - LOC2(devSim_dft.xyz, 1, i, 3, devSim_dft.natom),2) +
                                    pow(zparent - LOC2(devSim_dft.xyz, 2, i, 3, devSim_dft.natom),2));
            distnbor = (distnbor<distance)? distnbor: distance;
        }
    }   
    
    if (rig < 0.18 * distnbor) {
        ssw = 1.0e0;
        return ssw;
    }
    
    /*
     If neither of those are the case, we have to actually calculate the
     weight.  First we must calculate the unnormalized wieght of the grid point
     with respect to the parent atom.
    
     Step one of calculating the unnormalized weight is finding the confocal
     elliptical coordinate between each cell.  This it the mu with subscripted
     i and j in the paper:
     Stratmann, Scuseria, and Frisch, Chem. Phys. Lett., v 257,
     1996, pg 213-223.
     */
    QUICKDouble wofparent = 1.0e0; // weight of parents
    
    for (int jatm = 1; jatm <= devSim_dft.natom; jatm ++)
    {
        if (jatm != atm && wofparent != 0.0e0){
            QUICKDouble xjatm = LOC2(devSim_dft.xyz, 0, jatm-1, 3, devSim_dft.natom) ;
            QUICKDouble yjatm = LOC2(devSim_dft.xyz, 1, jatm-1, 3, devSim_dft.natom) ;
            QUICKDouble zjatm = LOC2(devSim_dft.xyz, 2, jatm-1, 3, devSim_dft.natom) ;
            
            QUICKDouble rjg = sqrt(pow((gridx-xjatm),2) + pow((gridy-yjatm),2) + pow((gridz-zjatm),2)); 
            QUICKDouble rij = sqrt(pow((xparent-xjatm),2) + pow((yparent-yjatm),2) + pow((zparent-zjatm),2)); 
            QUICKDouble confocal = (rig - rjg)/rij;
            
            if (confocal >= 0.64) {
                wofparent = 0.0e0;
            }else if (confocal>=-0.64e0) {
                QUICKDouble frctn = confocal/0.64;
                QUICKDouble gofconfocal = (35.0*frctn-35.0*pow(frctn,3)+21.0*pow(frctn,5)-5.0*pow(frctn,7))/16.0;
                wofparent = wofparent*0.5*(1.0-gofconfocal);
            }
        }
    }
    
    QUICKDouble totalw = wofparent;
    if (wofparent == 0.0e0) {
        ssw = 0.0e0;
        return ssw;
    }
    
    /*    
     Now we have the unnormalized weight of the grid point with regard to the
     parent atom.  Now we have to do this for all other atom pairs to
     normalize the grid weight.
     */
    
    // !!!! this part should be rewrite
    for (int i = 0; i<devSim_dft.natom; i++) {
        if (i!=atm-1) {
            QUICKDouble xiatm = LOC2(devSim_dft.xyz, 0, i, 3, devSim_dft.natom) ;
            QUICKDouble yiatm = LOC2(devSim_dft.xyz, 1, i, 3, devSim_dft.natom) ;
            QUICKDouble ziatm = LOC2(devSim_dft.xyz, 2, i, 3, devSim_dft.natom) ;
            
            rig = sqrt(pow((gridx-xiatm),2) + pow((gridy-yiatm),2) + pow((gridz-ziatm),2)); 
            QUICKDouble wofiatom = 1.0;
            for (int jatm = 1; jatm<=devSim_dft.natom;jatm++){
                if (jatm != i+1 && wofiatom != 0.0e0) {
                    QUICKDouble xjatm = LOC2(devSim_dft.xyz, 0, jatm-1, 3, devSim_dft.natom) ;
                    QUICKDouble yjatm = LOC2(devSim_dft.xyz, 1, jatm-1, 3, devSim_dft.natom) ;
                    QUICKDouble zjatm = LOC2(devSim_dft.xyz, 2, jatm-1, 3, devSim_dft.natom) ;
                    
                    QUICKDouble rjg = sqrt(pow((gridx-xjatm),2) + pow((gridy-yjatm),2) + pow((gridz-zjatm),2)); 
                    QUICKDouble rij = sqrt(pow((xiatm-xjatm),2) + pow((yiatm-yjatm),2) + pow((ziatm-zjatm),2)); 
                    QUICKDouble confocal = (rig - rjg)/rij;
                    if (confocal >= 0.64) {
                        wofiatom = 0.0e0;
                    }else if (confocal>=-0.64e0) {
                        QUICKDouble frctn = confocal/0.64;
                        QUICKDouble gofconfocal = (35.0*frctn-35.0*pow(frctn,3)+21.0*pow(frctn,5)-5.0*pow(frctn,7))/16.0;
                        wofiatom = wofiatom*0.5*(1.0-gofconfocal);
                    }
                }
            }
            totalw = totalw + wofiatom;
        }
    }
    ssw = wofparent/totalw;
    return ssw;
}

/*
	Madu Manathunga 09/11/2019
*/
/*
__device__ void sswder(QUICKDouble gridx, QUICKDouble gridy, QUICKDouble gridz, QUICKDouble Exc, QUICKDouble quadwt, int iparent, double* exc_dev_grad){


	This subroutine calculates the derivatives of weight found in
	Stratmann, Scuseria, and Frisch, Chem. Phys. Lett., v 257,
	1996, pg 213-223.  The mathematical development is similar to
	the developement found in the papers of Gill and Pople, but uses
	SSW method rather than Becke's weights.

	The derivative of the weight with repsect to the displacement of the
	parent atom is mathematically quite complicated, thus rotational
	invariance is used.  Rotational invariance simply is the condition that
	the sum of all derivatives must be zero.

	Certain things will be needed again and again in this subroutine.  We
	are therefore goint to store them.  They are the unnormalized weights.
	The array is called UW for obvious reasons.
	

#ifdef DEBUG
//        printf("gridx: %f  gridy: %f  gridz: %f, exc: %.10e, quadwt: %.10e, iparent: %i \n",gridx, gridy, gridz, Exc, quadwt, iparent);
#endif

	QUICKDouble sumUW= 0.0;
	QUICKDouble xiatm, yiatm, ziatm, xjatm, yjatm, zjatm, xlatm, ylatm, zlatm;
	QUICKDouble rig, rjg, rlg, rij, rjl;
		
	for(int iatm=0;iatm<devSim_dft.natom;iatm++){
		sumUW = sumUW + get_unnormalized_weight(gridx, gridy, gridz, iatm);
	}

#ifdef DEBUG
        //printf("gridx: %f  gridy: %f  gridz: %f, exc: %.10e, quadwt: %.10e, iparent: %i, sumUW: %.10e \n",gridx, gridy, gridz, Exc, quadwt, iparent, sumUW);
#endif
	

	At this point we now have the unnormalized weight and the sum of same.
	Calculate the parent atom - grid point distance, and then start the loop.

	
	xiatm = LOC2(devSim_dft.xyz, 0, iparent-1, 3, devSim_dft.natom);
	yiatm = LOC2(devSim_dft.xyz, 1, iparent-1, 3, devSim_dft.natom);
	ziatm = LOC2(devSim_dft.xyz, 2, iparent-1, 3, devSim_dft.natom);

	rig = sqrt(pow((gridx-xiatm),2) + pow((gridy-yiatm),2) + pow((gridz-ziatm),2));	

	QUICKDouble a = 0.64;

	int istart = (iparent-1)*3;
	
	QUICKDouble wtgradix = 0.0;
	QUICKDouble wtgradiy = 0.0;
	QUICKDouble wtgradiz = 0.0;

	for(int jatm=0;jatm<devSim_dft.natom;jatm++){

		int jstart = jatm*3;

		QUICKDouble wtgradjx = 0.0;
		QUICKDouble wtgradjy = 0.0;
		QUICKDouble wtgradjz = 0.0;
		
		if(jatm != iparent-1){

			xjatm = LOC2(devSim_dft.xyz, 0, jatm, 3, devSim_dft.natom);
			yjatm = LOC2(devSim_dft.xyz, 1, jatm, 3, devSim_dft.natom);
			zjatm = LOC2(devSim_dft.xyz, 2, jatm, 3, devSim_dft.natom);

			rjg = sqrt(pow((gridx-xjatm),2) + pow((gridy-yjatm),2) + pow((gridz-zjatm),2));
			rij = sqrt(pow((xiatm-xjatm),2) + pow((yiatm-yjatm),2) + pow((ziatm-zjatm),2));	

			QUICKDouble dmudx = (-1.0/rij)*(1/rjg)*(xjatm-gridx) + ((rig-rjg)/pow(rij,3))*(xiatm-xjatm);
			QUICKDouble dmudy = (-1.0/rij)*(1/rjg)*(yjatm-gridy) + ((rig-rjg)/pow(rij,3))*(yiatm-yjatm);
			QUICKDouble dmudz = (-1.0/rij)*(1/rjg)*(zjatm-gridz) + ((rig-rjg)/pow(rij,3))*(ziatm-zjatm);

			QUICKDouble u = (rig-rjg)/rij;
			QUICKDouble t = (-35.0*pow((a+u),3)) / ((a-u)*(16.0*pow(a,3)+29.0*pow(a,2)*u+20.0*a*pow(u,2)+5.0*pow(u,3)));
			QUICKDouble uw_iparent = get_unnormalized_weight(gridx, gridy, gridz, iparent-1);
			QUICKDouble uw_jatm = get_unnormalized_weight(gridx, gridy, gridz, jatm);

			wtgradjx = wtgradjx + uw_iparent*dmudx*t/sumUW;
			wtgradjy = wtgradjy + uw_iparent*dmudy*t/sumUW;
			wtgradjz = wtgradjz + uw_iparent*dmudz*t/sumUW;

#ifdef DEBUG
      //  printf("gridx: %f  gridy: %f  gridz: %f \n",wtgradjx, wtgradjy, wtgradjz);
#endif

			for(int latm=0; latm<devSim_dft.natom;latm++){
				if(latm != jatm){
					xlatm = LOC2(devSim_dft.xyz, 0, latm, 3, devSim_dft.natom);
					ylatm = LOC2(devSim_dft.xyz, 1, latm, 3, devSim_dft.natom);
					zlatm = LOC2(devSim_dft.xyz, 2, latm, 3, devSim_dft.natom);

					rlg = sqrt(pow((gridx-xlatm),2) + pow((gridy-ylatm),2) + pow((gridz-zlatm),2));
					rjl = sqrt(pow((xjatm-xlatm),2) + pow((yjatm-ylatm),2) + pow((zjatm-zlatm),2));

					dmudx = (-1.0/rjl)*(1/rjg)*(xjatm-gridx) + ((rlg-rjg)/pow(rjl,3))*(xlatm-xjatm);
					dmudy = (-1.0/rjl)*(1/rjg)*(yjatm-gridy) + ((rlg-rjg)/pow(rjl,3))*(ylatm-yjatm);
					dmudz = (-1.0/rjl)*(1/rjg)*(zjatm-gridz) + ((rlg-rjg)/pow(rjl,3))*(zlatm-zjatm);

					u = (rlg-rjg)/rjl; 
					t = (-35.0*pow((a+u),3)) / ((a-u)*(16.0*pow(a,3)+29.0*pow(a,2)*u+20.0*a*pow(u,2)+5.0*pow(u,3)));
					QUICKDouble uw_latm = get_unnormalized_weight(gridx, gridy, gridz, latm);					
					wtgradjx = wtgradjx - uw_latm*uw_iparent*dmudx*t/pow(sumUW,2);
					wtgradjy = wtgradjy - uw_latm*uw_iparent*dmudy*t/pow(sumUW,2);
					wtgradjz = wtgradjz - uw_latm*uw_iparent*dmudz*t/pow(sumUW,2);
				}
			}
#ifdef DEBUG
      //  printf("gridx: %f  gridy: %f  gridz: %f \n",wtgradjx, wtgradjy, wtgradjz);
#endif
			for(int latm=0; latm<devSim_dft.natom;latm++){
				if(latm != jatm){
					xlatm = LOC2(devSim_dft.xyz, 0, latm, 3, devSim_dft.natom);
					ylatm = LOC2(devSim_dft.xyz, 1, latm, 3, devSim_dft.natom);
					zlatm = LOC2(devSim_dft.xyz, 2, latm, 3, devSim_dft.natom);

					rlg = sqrt(pow((gridx-xlatm),2) + pow((gridy-ylatm),2) + pow((gridz-zlatm),2));
					rjl = sqrt(pow((xjatm-xlatm),2) + pow((yjatm-ylatm),2) + pow((zjatm-zlatm),2));

					dmudx = (-1.0/rjl)*(1/rlg)*(xlatm-gridx) + ((rjg-rlg)/pow(rjl,3))*(xjatm-xlatm);
					dmudy = (-1.0/rjl)*(1/rlg)*(ylatm-gridy) + ((rjg-rlg)/pow(rjl,3))*(yjatm-ylatm);
					dmudz = (-1.0/rjl)*(1/rlg)*(zlatm-gridz) + ((rjg-rlg)/pow(rjl,3))*(zjatm-zlatm);

					u = (rjg-rlg)/rjl;
					t = (-35.0*pow((a+u),3)) / ((a-u)*(16.0*pow(a,3)+29.0*pow(a,2)*u+20.0*a*pow(u,2)+5.0*pow(u,3)));

					wtgradjx = wtgradjx + uw_jatm*uw_iparent*dmudx*t/pow(sumUW,2);
					wtgradjy = wtgradjy + uw_jatm*uw_iparent*dmudy*t/pow(sumUW,2);
					wtgradjz = wtgradjz + uw_jatm*uw_iparent*dmudz*t/pow(sumUW,2);
				}
			}
#ifdef DEBUG
      //  printf("gridx: %f  gridy: %f  gridz: %f \n",wtgradjx, wtgradjy, wtgradjz);
#endif

//      Now do the rotational invariance part of the derivatives.

		wtgradix = wtgradix - wtgradjx;
		wtgradiy = wtgradiy - wtgradjy;
		wtgradiz = wtgradiz - wtgradjz;

#ifdef DEBUG
        //printf("gridx: %f  gridy: %f  gridz: %f Exc: %e quadwt: %e\n",wtgradjx, wtgradjy, wtgradjz, Exc, quadwt);
#endif

//	We should now have the derivatives of the SS weights.  Now just add it.

		atomicAdd(&(exc_dev_grad[jstart+0]), wtgradjx*Exc*quadwt);
		atomicAdd(&(exc_dev_grad[jstart+1]), wtgradjy*Exc*quadwt);
		atomicAdd(&(exc_dev_grad[jstart+2]), wtgradjz*Exc*quadwt);
		}

	}

#ifdef DEBUG
        //printf("istart: %i  gridx: %f  gridy: %f  gridz: %f Exc: %e quadwt: %e\n",istart, wtgradix, wtgradiy, wtgradiz, Exc, quadwt);
#endif

	atomicAdd(&(exc_dev_grad[istart+0]), wtgradix*Exc*quadwt);
	atomicAdd(&(exc_dev_grad[istart+1]), wtgradiy*Exc*quadwt);
	atomicAdd(&(exc_dev_grad[istart+2]), wtgradiz*Exc*quadwt);


}
*/

__device__ void sswder_new_imp(QUICKDouble gridx, QUICKDouble gridy, QUICKDouble gridz, QUICKDouble Exc, QUICKDouble quadwt, int iparent, int gid, double* exc_dev_grad){

/*
        This subroutine calculates the derivatives of weight found in
        Stratmann, Scuseria, and Frisch, Chem. Phys. Lett., v 257,
        1996, pg 213-223.  The mathematical development is similar to
        the developement found in the papers of Gill and Pople, but uses
        SSW method rather than Becke's weights.

        The derivative of the weight with repsect to the displacement of the
        parent atom is mathematically quite complicated, thus rotational
        invariance is used.  Rotational invariance simply is the condition that
        the sum of all derivatives must be zero.

        Certain things will be needed again and again in this subroutine.  We
        are therefore goint to store them.  They are the unnormalized weights.
        The array is called UW for obvious reasons.
*/

#ifdef DEBUG
//        printf("gridx: %f  gridy: %f  gridz: %f, exc: %.10e, quadwt: %.10e, iparent: %i \n",gridx, gridy, gridz, Exc, quadwt, iparent);
#endif

        QUICKDouble sumUW= 0.0;
        QUICKDouble xiatm, yiatm, ziatm, xjatm, yjatm, zjatm, xlatm, ylatm, zlatm;
        QUICKDouble rig, rjg, rlg, rij, rjl;

        for(int iatm=0;iatm<devSim_dft.natom;iatm++){
		QUICKDouble tmp_uw = get_unnormalized_weight(gridx, gridy, gridz, iatm);
		devSim_dft.uw_ssd[gid*devSim_dft.natom+iatm] = tmp_uw;
                sumUW = sumUW + tmp_uw;
        }

#ifdef DEBUG
        //printf("gridx: %f  gridy: %f  gridz: %f, exc: %.10e, quadwt: %.10e, iparent: %i, sumUW: %.10e \n",gridx, gridy, gridz, Exc, quadwt, iparent, sumUW);
#endif

/*
        At this point we now have the unnormalized weight and the sum of same.
        Calculate the parent atom - grid point distance, and then start the loop.
*/

        xiatm = LOC2(devSim_dft.xyz, 0, iparent-1, 3, devSim_dft.natom);
        yiatm = LOC2(devSim_dft.xyz, 1, iparent-1, 3, devSim_dft.natom);
        ziatm = LOC2(devSim_dft.xyz, 2, iparent-1, 3, devSim_dft.natom);

        rig = sqrt(pow((gridx-xiatm),2) + pow((gridy-yiatm),2) + pow((gridz-ziatm),2));

        QUICKDouble a = 0.64;

        int istart = (iparent-1)*3;

        QUICKDouble wtgradix = 0.0;
        QUICKDouble wtgradiy = 0.0;
        QUICKDouble wtgradiz = 0.0;

        for(int jatm=0;jatm<devSim_dft.natom;jatm++){

                int jstart = jatm*3;

                QUICKDouble wtgradjx = 0.0;
                QUICKDouble wtgradjy = 0.0;
                QUICKDouble wtgradjz = 0.0;

                if(jatm != iparent-1){

                        xjatm = LOC2(devSim_dft.xyz, 0, jatm, 3, devSim_dft.natom);
                        yjatm = LOC2(devSim_dft.xyz, 1, jatm, 3, devSim_dft.natom);
                        zjatm = LOC2(devSim_dft.xyz, 2, jatm, 3, devSim_dft.natom);

                        rjg = sqrt(pow((gridx-xjatm),2) + pow((gridy-yjatm),2) + pow((gridz-zjatm),2));
                        rij = sqrt(pow((xiatm-xjatm),2) + pow((yiatm-yjatm),2) + pow((ziatm-zjatm),2));

                        QUICKDouble dmudx = (-1.0/rij)*(1/rjg)*(xjatm-gridx) + ((rig-rjg)/pow(rij,3))*(xiatm-xjatm);
                        QUICKDouble dmudy = (-1.0/rij)*(1/rjg)*(yjatm-gridy) + ((rig-rjg)/pow(rij,3))*(yiatm-yjatm);
                        QUICKDouble dmudz = (-1.0/rij)*(1/rjg)*(zjatm-gridz) + ((rig-rjg)/pow(rij,3))*(ziatm-zjatm);

                        QUICKDouble u = (rig-rjg)/rij;
                        QUICKDouble t = (-35.0*pow((a+u),3)) / ((a-u)*(16.0*pow(a,3)+29.0*pow(a,2)*u+20.0*a*pow(u,2)+5.0*pow(u,3)));
                        //QUICKDouble uw_iparent = get_unnormalized_weight(gridx, gridy, gridz, iparent-1);
			QUICKDouble uw_iparent = devSim_dft.uw_ssd[gid*devSim_dft.natom+(iparent-1)];
                        //QUICKDouble uw_jatm = get_unnormalized_weight(gridx, gridy, gridz, jatm);
			QUICKDouble uw_jatm = devSim_dft.uw_ssd[gid*devSim_dft.natom+jatm];

                        wtgradjx = wtgradjx + uw_iparent*dmudx*t/sumUW;
                        wtgradjy = wtgradjy + uw_iparent*dmudy*t/sumUW;
                        wtgradjz = wtgradjz + uw_iparent*dmudz*t/sumUW;

#ifdef DEBUG
      //  printf("gridx: %f  gridy: %f  gridz: %f \n",wtgradjx, wtgradjy, wtgradjz);
#endif

                        for(int latm=0; latm<devSim_dft.natom;latm++){
                                if(latm != jatm){
                                        xlatm = LOC2(devSim_dft.xyz, 0, latm, 3, devSim_dft.natom);
                                        ylatm = LOC2(devSim_dft.xyz, 1, latm, 3, devSim_dft.natom);
                                        zlatm = LOC2(devSim_dft.xyz, 2, latm, 3, devSim_dft.natom);

                                        rlg = sqrt(pow((gridx-xlatm),2) + pow((gridy-ylatm),2) + pow((gridz-zlatm),2));
                                        rjl = sqrt(pow((xjatm-xlatm),2) + pow((yjatm-ylatm),2) + pow((zjatm-zlatm),2));

                                        dmudx = (-1.0/rjl)*(1/rjg)*(xjatm-gridx) + ((rlg-rjg)/pow(rjl,3))*(xlatm-xjatm);
                                        dmudy = (-1.0/rjl)*(1/rjg)*(yjatm-gridy) + ((rlg-rjg)/pow(rjl,3))*(ylatm-yjatm);
                                        dmudz = (-1.0/rjl)*(1/rjg)*(zjatm-gridz) + ((rlg-rjg)/pow(rjl,3))*(zlatm-zjatm);

                                        u = (rlg-rjg)/rjl;
                                        t = (-35.0*pow((a+u),3)) / ((a-u)*(16.0*pow(a,3)+29.0*pow(a,2)*u+20.0*a*pow(u,2)+5.0*pow(u,3)));
                                        //QUICKDouble uw_latm = get_unnormalized_weight(gridx, gridy, gridz, latm);                                   
					QUICKDouble uw_latm = devSim_dft.uw_ssd[gid*devSim_dft.natom+latm];
                                        wtgradjx = wtgradjx - uw_latm*uw_iparent*dmudx*t/pow(sumUW,2);
                                        wtgradjy = wtgradjy - uw_latm*uw_iparent*dmudy*t/pow(sumUW,2);
                                        wtgradjz = wtgradjz - uw_latm*uw_iparent*dmudz*t/pow(sumUW,2);
                                }
                        }
#ifdef DEBUG
      //  printf("gridx: %f  gridy: %f  gridz: %f \n",wtgradjx, wtgradjy, wtgradjz);
#endif
                        for(int latm=0; latm<devSim_dft.natom;latm++){
                                if(latm != jatm){
                                        xlatm = LOC2(devSim_dft.xyz, 0, latm, 3, devSim_dft.natom);
                                        ylatm = LOC2(devSim_dft.xyz, 1, latm, 3, devSim_dft.natom);
                                        zlatm = LOC2(devSim_dft.xyz, 2, latm, 3, devSim_dft.natom);

                                        rlg = sqrt(pow((gridx-xlatm),2) + pow((gridy-ylatm),2) + pow((gridz-zlatm),2));
                                        rjl = sqrt(pow((xjatm-xlatm),2) + pow((yjatm-ylatm),2) + pow((zjatm-zlatm),2));

                                        dmudx = (-1.0/rjl)*(1/rlg)*(xlatm-gridx) + ((rjg-rlg)/pow(rjl,3))*(xjatm-xlatm);
                                        dmudy = (-1.0/rjl)*(1/rlg)*(ylatm-gridy) + ((rjg-rlg)/pow(rjl,3))*(yjatm-ylatm);
                                        dmudz = (-1.0/rjl)*(1/rlg)*(zlatm-gridz) + ((rjg-rlg)/pow(rjl,3))*(zjatm-zlatm);

                                        u = (rjg-rlg)/rjl;
                                        t = (-35.0*pow((a+u),3)) / ((a-u)*(16.0*pow(a,3)+29.0*pow(a,2)*u+20.0*a*pow(u,2)+5.0*pow(u,3)));

                                        wtgradjx = wtgradjx + uw_jatm*uw_iparent*dmudx*t/pow(sumUW,2);
                                        wtgradjy = wtgradjy + uw_jatm*uw_iparent*dmudy*t/pow(sumUW,2);
                                        wtgradjz = wtgradjz + uw_jatm*uw_iparent*dmudz*t/pow(sumUW,2);
                                }
                        }
#ifdef DEBUG
      //  printf("gridx: %f  gridy: %f  gridz: %f \n",wtgradjx, wtgradjy, wtgradjz);
#endif

//      Now do the rotational invariance part of the derivatives.

                wtgradix = wtgradix - wtgradjx;
                wtgradiy = wtgradiy - wtgradjy;
                wtgradiz = wtgradiz - wtgradjz;

#ifdef DEBUG
        //printf("gridx: %f  gridy: %f  gridz: %f Exc: %e quadwt: %e\n",wtgradjx, wtgradjy, wtgradjz, Exc, quadwt);
#endif

//      We should now have the derivatives of the SS weights.  Now just add it.

                atomicAdd(&(exc_dev_grad[jstart+0]), wtgradjx*Exc*quadwt);
                atomicAdd(&(exc_dev_grad[jstart+1]), wtgradjy*Exc*quadwt);
                atomicAdd(&(exc_dev_grad[jstart+2]), wtgradjz*Exc*quadwt);
                }

        }

#ifdef DEBUG
        //printf("istart: %i  gridx: %f  gridy: %f  gridz: %f Exc: %e quadwt: %e\n",istart, wtgradix, wtgradiy, wtgradiz, Exc, quadwt);
#endif

        atomicAdd(&(exc_dev_grad[istart+0]), wtgradix*Exc*quadwt);
        atomicAdd(&(exc_dev_grad[istart+1]), wtgradiy*Exc*quadwt);
        atomicAdd(&(exc_dev_grad[istart+2]), wtgradiz*Exc*quadwt);


}


/*  Madu Manathunga 09/10/2019
*/
__device__ QUICKDouble get_unnormalized_weight(QUICKDouble gridx, QUICKDouble gridy, QUICKDouble gridz, int iatm){

	QUICKDouble uw = 1.0;
	QUICKDouble xiatm, yiatm, ziatm, xjatm, yjatm, zjatm;
	QUICKDouble rig, rjg, rij;

	xiatm = LOC2(devSim_dft.xyz, 0, iatm, 3, devSim_dft.natom);
	yiatm = LOC2(devSim_dft.xyz, 1, iatm, 3, devSim_dft.natom);
	ziatm = LOC2(devSim_dft.xyz, 2, iatm, 3, devSim_dft.natom);

	rig = sqrt(pow((gridx-xiatm),2) + pow((gridy-yiatm),2) + pow((gridz-ziatm),2));

	for(int jatm=0;jatm<devSim_dft.natom;jatm++){
		if(jatm != iatm){
			xjatm = LOC2(devSim_dft.xyz, 0, jatm, 3, devSim_dft.natom);
			yjatm = LOC2(devSim_dft.xyz, 1, jatm, 3, devSim_dft.natom);
			zjatm = LOC2(devSim_dft.xyz, 2, jatm, 3, devSim_dft.natom);

			rjg = sqrt(pow((gridx-xjatm),2) + pow((gridy-yjatm),2) + pow((gridz-zjatm),2));
			rij = sqrt(pow((xiatm-xjatm),2) + pow((yiatm-yjatm),2) + pow((ziatm-zjatm),2));

			QUICKDouble confocal = (rig-rjg)/rij;

			if(confocal >= 0.64){
				uw = 0.0;
			}else if(confocal >= -0.64){
				QUICKDouble frctn = confocal/0.64;
				QUICKDouble gofconfocal = (35.0*frctn-35.0*pow(frctn,3)+21.0*pow(frctn,5)-5.0*pow(frctn,7))/16.0;
				uw = uw*0.5*(1.0-gofconfocal);
			}

		}

	}

	return uw;

}

/*  Madu Manathunga 08/20/2019
*/

/*__device__ void pt2der(QUICKDouble gridx, QUICKDouble gridy, QUICKDouble gridz, QUICKDouble* dxdx, QUICKDouble* dxdy,
                QUICKDouble* dxdz, QUICKDouble* dydy, QUICKDouble* dydz, QUICKDouble* dzdz, int ibas){
		
	Given a point in space, this function calculates the value of basis
	function I and the value of its cartesian derivatives in all three derivatives.
	

	// relative coordinates between grid point and basis function I.
	QUICKDouble x1 = gridx - LOC2(devSim_dft.xyz, 0, devSim_dft.ncenter[ibas-1]-1, 3, devSim_dft.natom);
	QUICKDouble y1 = gridy - LOC2(devSim_dft.xyz, 1, devSim_dft.ncenter[ibas-1]-1, 3, devSim_dft.natom);
	QUICKDouble z1 = gridz - LOC2(devSim_dft.xyz, 2, devSim_dft.ncenter[ibas-1]-1, 3, devSim_dft.natom);	

	QUICKDouble x1i, y1i, z1i;
	QUICKDouble x1imin1, y1imin1, z1imin1, x1imin2, y1imin2, z1imin2;
	QUICKDouble x1iplus1, y1iplus1, z1iplus1, x1iplus2, y1iplus2, z1iplus2;

	*dxdx = 0.0;
	*dxdy = 0.0;
	*dxdz = 0.0;
	*dydy = 0.0;
	*dydz = 0.0;
	*dzdz = 0.0;

	int itypex = LOC2(devSim_dft.itype, 0, ibas-1, 3, devSim_dft.nbasis);
	int itypey = LOC2(devSim_dft.itype, 1, ibas-1, 3, devSim_dft.nbasis);
	int itypez = LOC2(devSim_dft.itype, 2, ibas-1, 3, devSim_dft.nbasis);

	QUICKDouble dist = x1*x1+y1*y1+z1*z1;

	if ( dist <= devSim_dft.sigrad2[ibas-1]){
		if ( itypex == 0) {
			x1imin2 = 0.0;
			x1imin1 = 0.0;
			x1i = 1.0;
			x1iplus1 = x1;
			x1iplus2 = x1*x1;
		}else if(itypex == 1){
			x1imin2 = 0.0;
			x1imin1 = 1.0;
			x1i = x1;
			x1iplus1 = x1*x1;
			x1iplus2 = x1*x1*x1;
		}else{
			x1imin2 = pow(x1, itypex-2);
			x1imin1 = x1imin2*x1;
			x1i = x1imin1 * x1;
			x1iplus1 = x1i * x1;
			x1iplus2 = x1iplus1 * x1;
		}

		if ( itypey == 0) {
			y1imin2 = 0.0;
			y1imin1 = 0.0;
			y1i = 1.0;
			y1iplus1 = y1;
			y1iplus2 = y1*y1;
		}else if(itypey == 1){
			y1imin2 = 0.0;
			y1imin1 = 1.0;
			y1i = y1;
			y1iplus1 = y1*y1;
			y1iplus2 = y1iplus1*y1;
		}else{
			y1imin2 = pow(y1, itypey-2);
			y1imin1 = y1imin2*y1; 
			y1i = y1imin1 * y1;
			y1iplus1 = y1i * y1;
			y1iplus2 = y1iplus1 * y1;
		}

		if ( itypez == 0) {
			z1imin2 = 0.0;
			z1imin1 = 0.0;
			z1i = 1.0;
			z1iplus1 = z1;
			z1iplus2 = z1*z1;
		}else if(itypez == 1){
			z1imin2 = 0.0;
			z1imin1 = 1.0;
			z1i = z1;
			z1iplus1 = z1*z1;
			z1iplus2 = z1iplus1*z1;
		}else {
			z1imin2 = pow(z1, itypez-2);
			z1imin1 = z1imin2*z1;
			z1i = z1imin1 * z1;
			z1iplus1 = z1i * z1;
			z1iplus2 = z1iplus1 * z1;
		}

		for (int i = 0; i < devSim_dft.ncontract[ibas-1]; i++) {
			QUICKDouble temp = LOC2(devSim_dft.dcoeff, i, ibas-1, devSim_dft.maxcontract, devSim_dft.nbasis) *
							exp( - LOC2(devSim_dft.aexp, i, ibas-1, devSim_dft.maxcontract, devSim_dft.nbasis) * dist);	
			QUICKDouble twoA = 2.0 * LOC2(devSim_dft.aexp, i, ibas-1, devSim_dft.maxcontract, devSim_dft.nbasis);
			QUICKDouble fourAsqr = twoA * twoA;

			*dxdx = *dxdx + temp * ((QUICKDouble)itypex * ((QUICKDouble)itypex -1.0) * x1imin2 
					- twoA * (2.0 * (QUICKDouble)itypex+1.0) * x1i
					+ fourAsqr * x1iplus2 );
			
			*dydy = *dydy + temp * ((QUICKDouble)itypey * ((QUICKDouble)itypey -1.0) * y1imin2 
					- twoA * (2.0 * (QUICKDouble)itypey+1.0) * y1i 
					+ fourAsqr * y1iplus2 );

			*dzdz = *dzdz + temp * ((QUICKDouble)itypez * ((QUICKDouble)itypez -1.0) * z1imin2
					- twoA * (2.0 * (QUICKDouble)itypez+1.0) * z1i
					+ fourAsqr * z1iplus2 );

			*dxdy = *dxdy + temp * ((QUICKDouble)itypex * (QUICKDouble)itypey * x1imin1 * y1imin1
					- twoA * (QUICKDouble)itypex * x1imin1 * y1iplus1
					- twoA * (QUICKDouble)itypey * x1iplus1 * y1imin1
					+ fourAsqr * x1iplus1 * y1iplus1);

			*dxdz = *dxdz + temp * ((QUICKDouble)itypex * (QUICKDouble)itypez * x1imin1 * z1imin1
					- twoA * (QUICKDouble)itypex * x1imin1 * z1iplus1
					- twoA * (QUICKDouble)itypez * x1iplus1 * z1imin1
					+ fourAsqr * x1iplus1 * z1iplus1);

			*dydz = *dydz + temp * ((QUICKDouble)itypey * (QUICKDouble)itypez * y1imin1 * z1imin1
					- twoA * (QUICKDouble)itypey * y1imin1 * z1iplus1
					- twoA * (QUICKDouble)itypez * y1iplus1 * z1imin1
					+ fourAsqr * y1iplus1 * z1iplus1);
		}

		*dxdx = *dxdx * y1i * z1i;
		*dydy = *dydy * x1i * z1i;
		*dzdz = *dzdz * x1i * y1i;
		*dxdy = *dxdy * z1i;
		*dxdz = *dxdz * y1i;
		*dydz = *dydz * x1i;

	}

}
*/
__device__ void pt2der_new(QUICKDouble gridx, QUICKDouble gridy, QUICKDouble gridz, QUICKDouble* dxdx, QUICKDouble* dxdy,
                QUICKDouble* dxdz, QUICKDouble* dydy, QUICKDouble* dydz, QUICKDouble* dzdz, int *primf, int *primf_counter,
		 int ibas, int ibasp){

        /*Given a point in space, this function calculates the value of basis
        function I and the value of its cartesian derivatives in all three derivatives.
        */

        // relative coordinates between grid point and basis function I.
        QUICKDouble x1 = gridx - LOC2(devSim_dft.xyz, 0, devSim_dft.ncenter[ibas]-1, 3, devSim_dft.natom);
        QUICKDouble y1 = gridy - LOC2(devSim_dft.xyz, 1, devSim_dft.ncenter[ibas]-1, 3, devSim_dft.natom);
        QUICKDouble z1 = gridz - LOC2(devSim_dft.xyz, 2, devSim_dft.ncenter[ibas]-1, 3, devSim_dft.natom);

        QUICKDouble x1i, y1i, z1i;
        QUICKDouble x1imin1, y1imin1, z1imin1, x1imin2, y1imin2, z1imin2;
        QUICKDouble x1iplus1, y1iplus1, z1iplus1, x1iplus2, y1iplus2, z1iplus2;

        *dxdx = 0.0;
        *dxdy = 0.0;
        *dxdz = 0.0;
        *dydy = 0.0;
        *dydz = 0.0;
        *dzdz = 0.0;

        int itypex = LOC2(devSim_dft.itype, 0, ibas, 3, devSim_dft.nbasis);
        int itypey = LOC2(devSim_dft.itype, 1, ibas, 3, devSim_dft.nbasis);
        int itypez = LOC2(devSim_dft.itype, 2, ibas, 3, devSim_dft.nbasis);

        QUICKDouble dist = x1*x1+y1*y1+z1*z1;
        //if ( dist <= devSim_dft.sigrad2[ibas]){
                if ( itypex == 0) {
                        x1imin2 = 0.0;
                        x1imin1 = 0.0;
                        x1i = 1.0;
                        x1iplus1 = x1;
                        x1iplus2 = x1*x1;
                }else if(itypex == 1){
                        x1imin2 = 0.0;
                        x1imin1 = 1.0;
                        x1i = x1;
                        x1iplus1 = x1*x1;
                        x1iplus2 = x1*x1*x1;
                }else{
                        x1imin2 = pow(x1, itypex-2);
                        x1imin1 = x1imin2*x1;
                        x1i = x1imin1 * x1;
                        x1iplus1 = x1i * x1;
                        x1iplus2 = x1iplus1 * x1;
                }

                if ( itypey == 0) {
                        y1imin2 = 0.0;
                        y1imin1 = 0.0;
                        y1i = 1.0;
                        y1iplus1 = y1;
                        y1iplus2 = y1*y1;
                }else if(itypey == 1){
                        y1imin2 = 0.0;
                        y1imin1 = 1.0;
                        y1i = y1;
                        y1iplus1 = y1*y1;
                        y1iplus2 = y1iplus1*y1;
                }else{
                        y1imin2 = pow(y1, itypey-2);
                        y1imin1 = y1imin2*y1;
                        y1i = y1imin1 * y1;
                        y1iplus1 = y1i * y1;
                        y1iplus2 = y1iplus1 * y1;
                }
                if ( itypez == 0) {
                        z1imin2 = 0.0;
                        z1imin1 = 0.0;
                        z1i = 1.0;
                        z1iplus1 = z1;
                        z1iplus2 = z1*z1;
                }else if(itypez == 1){
                        z1imin2 = 0.0;
                        z1imin1 = 1.0;
                        z1i = z1;
                        z1iplus1 = z1*z1;
                        z1iplus2 = z1iplus1*z1;
                }else {
                        z1imin2 = pow(z1, itypez-2);
                        z1imin1 = z1imin2*z1;
                        z1i = z1imin1 * z1;
                        z1iplus1 = z1i * z1;
                        z1iplus2 = z1iplus1 * z1;
                }

//                for (int i = 0; i < devSim_dft.ncontract[ibas-1]; i++) {
		for(int i=primf_counter[ibasp]; i< primf_counter[ibasp+1]; i++){
			int kprim = primf[i];

                        QUICKDouble temp = LOC2(devSim_dft.dcoeff, kprim, ibas, devSim_dft.maxcontract, devSim_dft.nbasis) *
                                                        exp( - LOC2(devSim_dft.aexp, kprim, ibas, devSim_dft.maxcontract, devSim_dft.nbasis) * dist);
                        QUICKDouble twoA = 2.0 * LOC2(devSim_dft.aexp, kprim, ibas, devSim_dft.maxcontract, devSim_dft.nbasis);
                        QUICKDouble fourAsqr = twoA * twoA;

                        *dxdx = *dxdx + temp * ((QUICKDouble)itypex * ((QUICKDouble)itypex -1.0) * x1imin2
                                        - twoA * (2.0 * (QUICKDouble)itypex+1.0) * x1i
                                        + fourAsqr * x1iplus2 );

                        *dydy = *dydy + temp * ((QUICKDouble)itypey * ((QUICKDouble)itypey -1.0) * y1imin2
                                        - twoA * (2.0 * (QUICKDouble)itypey+1.0) * y1i
                                        + fourAsqr * y1iplus2 );

                        *dzdz = *dzdz + temp * ((QUICKDouble)itypez * ((QUICKDouble)itypez -1.0) * z1imin2
                                        - twoA * (2.0 * (QUICKDouble)itypez+1.0) * z1i
                                        + fourAsqr * z1iplus2 );

                        *dxdy = *dxdy + temp * ((QUICKDouble)itypex * (QUICKDouble)itypey * x1imin1 * y1imin1
                                        - twoA * (QUICKDouble)itypex * x1imin1 * y1iplus1
                                        - twoA * (QUICKDouble)itypey * x1iplus1 * y1imin1
                                        + fourAsqr * x1iplus1 * y1iplus1);

                        *dxdz = *dxdz + temp * ((QUICKDouble)itypex * (QUICKDouble)itypez * x1imin1 * z1imin1
                                        - twoA * (QUICKDouble)itypex * x1imin1 * z1iplus1
                                        - twoA * (QUICKDouble)itypez * x1iplus1 * z1imin1
                                        + fourAsqr * x1iplus1 * z1iplus1);

                        *dydz = *dydz + temp * ((QUICKDouble)itypey * (QUICKDouble)itypez * y1imin1 * z1imin1
                                        - twoA * (QUICKDouble)itypey * y1imin1 * z1iplus1
                                        - twoA * (QUICKDouble)itypez * y1iplus1 * z1imin1
                                        + fourAsqr * y1iplus1 * z1iplus1);
                }

                *dxdx = *dxdx * y1i * z1i;
                *dydy = *dydy * x1i * z1i;
                *dzdz = *dzdz * x1i * y1i;
                *dxdy = *dxdy * z1i;
                *dxdz = *dxdz * y1i;
                *dydz = *dydz * x1i;

        //}

}

__device__ void pteval_new(QUICKDouble gridx, QUICKDouble gridy, QUICKDouble gridz,
            QUICKDouble* phi, QUICKDouble* dphidx, QUICKDouble* dphidy,  QUICKDouble* dphidz,
            int *primf, int *primf_counter, int ibas, int ibasp)
{

    /*
      Given a point in space, this function calculates the value of basis
      function I and the value of its cartesian derivatives in all three
      derivatives.
     */

    // relative coordinates between grid point and basis function I.
    QUICKDouble x1 = gridx - LOC2(devSim_dft.xyz, 0, devSim_dft.ncenter[ibas]-1, 3, devSim_dft.natom);
    QUICKDouble y1 = gridy - LOC2(devSim_dft.xyz, 1, devSim_dft.ncenter[ibas]-1, 3, devSim_dft.natom);
    QUICKDouble z1 = gridz - LOC2(devSim_dft.xyz, 2, devSim_dft.ncenter[ibas]-1, 3, devSim_dft.natom);


    QUICKDouble x1i, y1i, z1i;
    QUICKDouble x1imin1, y1imin1, z1imin1;
    QUICKDouble x1iplus1, y1iplus1, z1iplus1;

    *phi = 0.0;
    *dphidx = 0.0;
    *dphidy = 0.0;
    *dphidz = 0.0;

    int itypex = LOC2(devSim_dft.itype, 0, ibas, 3, devSim_dft.nbasis);
    int itypey = LOC2(devSim_dft.itype, 1, ibas, 3, devSim_dft.nbasis);
    int itypez = LOC2(devSim_dft.itype, 2, ibas, 3, devSim_dft.nbasis);

    QUICKDouble dist = x1*x1+y1*y1+z1*z1;

    if ( dist <= devSim_dft.sigrad2[ibas]){
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
     
     
//        for (int i = 0; i < devSim_dft.ncontract[ibas-1]; i++) {
	for(int i=primf_counter[ibasp]; i< primf_counter[ibasp+1]; i++){
	    int kprim = primf[i]; 
            QUICKDouble tmp = LOC2(devSim_dft.dcoeff, kprim, ibas, devSim_dft.maxcontract, devSim_dft.nbasis) * 
                              exp( - LOC2(devSim_dft.aexp, kprim, ibas, devSim_dft.maxcontract, devSim_dft.nbasis) * dist);
            *phi = *phi + tmp; 
            *dphidx = *dphidx + tmp * ( -2.0 * LOC2(devSim_dft.aexp, kprim, ibas, devSim_dft.maxcontract, devSim_dft.nbasis)* x1iplus1 + (QUICKDouble)itypex * x1imin1);
            *dphidy = *dphidy + tmp * ( -2.0 * LOC2(devSim_dft.aexp, kprim, ibas, devSim_dft.maxcontract, devSim_dft.nbasis)* y1iplus1 + (QUICKDouble)itypey * y1imin1);
            *dphidz = *dphidz + tmp * ( -2.0 * LOC2(devSim_dft.aexp, kprim, ibas, devSim_dft.maxcontract, devSim_dft.nbasis)* z1iplus1 + (QUICKDouble)itypez * z1imin1);
        }    
     
        *phi = *phi * x1i * y1i * z1i; 
        *dphidx = *dphidx * y1i * z1i;
        *dphidy = *dphidy * x1i * z1i;
        *dphidz = *dphidz * x1i * y1i;
    }
}

/*
__device__ void pteval(QUICKDouble gridx, QUICKDouble gridy, QUICKDouble gridz, 
            QUICKDouble* phi, QUICKDouble* dphidx, QUICKDouble* dphidy,  QUICKDouble* dphidz, 
            int ibas)
{
    
    
      Given a point in space, this function calculates the value of basis
      function I and the value of its cartesian derivatives in all three
      derivatives.
     
    
    // relative coordinates between grid point and basis function I.
    QUICKDouble x1 = gridx - LOC2(devSim_dft.xyz, 0, devSim_dft.ncenter[ibas-1]-1, 3, devSim_dft.natom);
    QUICKDouble y1 = gridy - LOC2(devSim_dft.xyz, 1, devSim_dft.ncenter[ibas-1]-1, 3, devSim_dft.natom);
    QUICKDouble z1 = gridz - LOC2(devSim_dft.xyz, 2, devSim_dft.ncenter[ibas-1]-1, 3, devSim_dft.natom);
    
    
    QUICKDouble x1i, y1i, z1i;
    QUICKDouble x1imin1, y1imin1, z1imin1;
    QUICKDouble x1iplus1, y1iplus1, z1iplus1;
    
    *phi = 0.0;
    *dphidx = 0.0;
    *dphidy = 0.0;
    *dphidz = 0.0;
    
    int itypex = LOC2(devSim_dft.itype, 0, ibas-1, 3, devSim_dft.nbasis);
    int itypey = LOC2(devSim_dft.itype, 1, ibas-1, 3, devSim_dft.nbasis);
    int itypez = LOC2(devSim_dft.itype, 2, ibas-1, 3, devSim_dft.nbasis);
    
    QUICKDouble dist = x1*x1+y1*y1+z1*z1;
    
    if ( dist <= devSim_dft.sigrad2[ibas-1]){
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
        
        
        for (int i = 0; i < devSim_dft.ncontract[ibas-1]; i++) {
            QUICKDouble tmp = LOC2(devSim_dft.dcoeff, i, ibas-1, devSim_dft.maxcontract, devSim_dft.nbasis) * 
                              exp( - LOC2(devSim_dft.aexp, i, ibas-1, devSim_dft.maxcontract, devSim_dft.nbasis) * dist);
            *phi = *phi + tmp;
            *dphidx = *dphidx + tmp * ( -2.0 * LOC2(devSim_dft.aexp, i, ibas-1, devSim_dft.maxcontract, devSim_dft.nbasis)* x1iplus1 + (QUICKDouble)itypex * x1imin1);
            *dphidy = *dphidy + tmp * ( -2.0 * LOC2(devSim_dft.aexp, i, ibas-1, devSim_dft.maxcontract, devSim_dft.nbasis)* y1iplus1 + (QUICKDouble)itypey * y1imin1);
            *dphidz = *dphidz + tmp * ( -2.0 * LOC2(devSim_dft.aexp, i, ibas-1, devSim_dft.maxcontract, devSim_dft.nbasis)* z1iplus1 + (QUICKDouble)itypez * z1imin1);
        }
        
        *phi = *phi * x1i * y1i * z1i;
        *dphidx = *dphidx * y1i * z1i;
        *dphidy = *dphidy * x1i * z1i;
        *dphidz = *dphidz * x1i * y1i;
    }
}
*/

//**************** Debugging **************
/*__device__ void denspt_new(QUICKDouble gridx, QUICKDouble gridy, QUICKDouble gridz, QUICKDouble* density, QUICKDouble* densityb,
            QUICKDouble* gax,   QUICKDouble* gay,   QUICKDouble* gaz,   QUICKDouble* gbx,     QUICKDouble* gby,     QUICKDouble* gbz, int gid)
{
    
     //Given a point in space, this function calculates the densities and
     //gradient  at that point.  The gradients are stored in the common block
     //three element arrays ga and gb for alpha and beta electron gradients. Thus
     //the x, y, z component of the alpha density is stored in gax, gay, gaz.
     
    *density = 0.0;
    *gax = 0.0;
    *gay = 0.0;
    *gaz = 0.0;

    for (int i = 0; i<devSim_dft.nbasis; i++) {
        if (abs(LOC2(devSim_dft.dense, i, i, devSim_dft.nbasis, devSim_dft.nbasis)) >= devSim_dft.DMCutoff ) {
            QUICKDouble phi, dphidx, dphidy, dphidz;
            pteval(gridx, gridy, gridz, &phi, &dphidx, &dphidy, &dphidz, i+1);

            if (abs(phi+dphidx+dphidy+dphidz) >= devSim_dft.DMCutoff ) {
                QUICKDouble denseii = LOC2(devSim_dft.dense, i, i, devSim_dft.nbasis, devSim_dft.nbasis) * phi;
                *density = *density + denseii * phi / 2.0;
                *gax = *gax + denseii * dphidx;
                *gay = *gay + denseii * dphidy;
                *gaz = *gaz + denseii * dphidz;
                if(gid ==5209){
                //      printf("Selected: %f %f %f i: %i densii: %10e dens: %10e  gax: %10e gay: %10e  gaz: %10e  \n", gridx, gridy, gridz, i, denseii, *density, *gax, *gay, *gaz);
                }

                for (int j = i + 1; j< devSim_dft.nbasis; j++) {
                    QUICKDouble phi2, dphidx2, dphidy2, dphidz2;
                    pteval(gridx, gridy, gridz, &phi2, &dphidx2, &dphidy2, &dphidz2, j+1);

                    QUICKDouble denseij = LOC2(devSim_dft.dense, i, j, devSim_dft.nbasis, devSim_dft.nbasis);
                    *density = *density + denseij * phi * phi2;
                    *gax = *gax + denseij * ( phi * dphidx2 + phi2 * dphidx );
                    *gay = *gay + denseij * ( phi * dphidy2 + phi2 * dphidy );
                    *gaz = *gaz + denseij * ( phi * dphidz2 + phi2 * dphidz );

                }
            }
        }
    }
    *densityb = *density;
    *gbx = *gax;
    *gby = *gay;
    *gbz = *gaz;
}
*/
//**************** End debugging **************
/*
__device__ void denspt(QUICKDouble gridx, QUICKDouble gridy, QUICKDouble gridz, QUICKDouble* density, QUICKDouble* densityb, 
            QUICKDouble* gax,   QUICKDouble* gay,   QUICKDouble* gaz,   QUICKDouble* gbx,     QUICKDouble* gby,     QUICKDouble* gbz)
{
    
     Given a point in space, this function calculates the densities and
     gradient  at that point.  The gradients are stored in the common block
     three element arrays ga and gb for alpha and beta electron gradients. Thus
     the x, y, z component of the alpha density is stored in gax, gay, gaz.
    
    *density = 0.0;
    *gax = 0.0;
    *gay = 0.0;
    *gaz = 0.0;
    
    for (int i = 0; i<devSim_dft.nbasis; i++) {
        if (abs(LOC2(devSim_dft.dense, i, i, devSim_dft.nbasis, devSim_dft.nbasis)) >= devSim_dft.DMCutoff ) { 
            QUICKDouble phi, dphidx, dphidy, dphidz;
            pteval(gridx, gridy, gridz, &phi, &dphidx, &dphidy, &dphidz, i+1);
            
            if (abs(phi+dphidx+dphidy+dphidz) >= devSim_dft.DMCutoff ) { 
                QUICKDouble denseii = LOC2(devSim_dft.dense, i, i, devSim_dft.nbasis, devSim_dft.nbasis) * phi;
                *density = *density + denseii * phi / 2.0;
                *gax = *gax + denseii * dphidx;
                *gay = *gay + denseii * dphidy;
                *gaz = *gaz + denseii * dphidz;
                for (int j = i + 1; j< devSim_dft.nbasis; j++) {
                    QUICKDouble phi2, dphidx2, dphidy2, dphidz2;
                    pteval(gridx, gridy, gridz, &phi2, &dphidx2, &dphidy2, &dphidz2, j+1);
                    
                    QUICKDouble denseij = LOC2(devSim_dft.dense, i, j, devSim_dft.nbasis, devSim_dft.nbasis);
                    *density = *density + denseij * phi * phi2;
                    *gax = *gax + denseij * ( phi * dphidx2 + phi2 * dphidx );
                    *gay = *gay + denseij * ( phi * dphidy2 + phi2 * dphidy );
                    *gaz = *gaz + denseij * ( phi * dphidz2 + phi2 * dphidz );
                    
                }
            }
        }
    }
    *densityb = *density;
    *gbx = *gax;
    *gby = *gay;
    *gbz = *gaz;
}
*/

__device__ QUICKDouble becke_e(QUICKDouble density, QUICKDouble densityb, QUICKDouble gax, QUICKDouble gay, QUICKDouble gaz,
                               QUICKDouble gbx,     QUICKDouble gby,      QUICKDouble gbz)
{
    // Given the densities and the two gradients, return the energy.
    
    /*
     Becke exchange functional
             __  ->
            |\/ rho|
     s =    ---------
            rou^(4/3)int jstart = jatm*3;
                              -
     Ex = E(LDA)x[rou(r)] -  |  Fx[s]rou^(4/3) dr
                            -
     
                    1             s^2
     Fx[s] = b 2^(- -) ------------------------
                    3      1 + 6bs sinh^(-1) s
     */
    
    QUICKDouble const b = 0.0042;
    
    
    /*
              __  ->
             |\/ rho|
     s =    ---------
            rou^(4/3)
     */
    QUICKDouble x = sqrt(gax*gax+gay*gay+gaz*gaz)/pow(density,4.0/3.0);
    
    QUICKDouble e = pow(density,4.0/3.0)*(-0.93052573634910002500-b*x*x /   \
            //                           -------------------------------------
                                         (1.0+0.0252*x*log(x+sqrt(x*x+1.0))));
    
    if (densityb != 0.0){
        QUICKDouble rhob4thirds=pow(densityb,(4.0/3.0));
        QUICKDouble xb = sqrt(gbx*gbx+gby*gby+gbz*gbz)/rhob4thirds;
        QUICKDouble gofxb = -0.93052573634910002500-b*xb*xb/(1+6.0*0.0042*xb*log(xb+sqrt(xb*xb+1.0)));
        e = e + rhob4thirds * gofxb;
    }
    
    return e;
}

__device__ void becke(QUICKDouble density, QUICKDouble gx, QUICKDouble gy, QUICKDouble gz, QUICKDouble gotherx, QUICKDouble gothery, QUICKDouble gotherz,
                      QUICKDouble* dfdr, QUICKDouble* dfdgg, QUICKDouble* dfdggo)
{
    // Given either density and the two gradients, (with gother being for
    // the spin that density is not, i.e. beta if density is alpha) return
    // the derivative of beckes 1988 functional with regard to the density
    // and the derivatives with regard to the gradient invariants.
    
    // Example:  If becke() is passed the alpha density and the alpha and beta
    // density gradients, return the derivative of f with regard to the alpha
    //denisty, the alpha-alpha gradient invariant, and the alpha beta gradient
    // invariant.
    
    /*
     Becke exchange functional
             __  ->
            |\/ rho|
        s = ---------
            rou^(4/3)
                                 -
        Ex = E(LDA)x[rou(r)] -  |  Fx[s]rou^(4/3) dr
                               -
     
                       1             s^2
        Fx[s] = b 2^(- -) ------------------------
                       3      1 + 6bs sinh^(-1) s
     */
    
    //QUICKDouble fourPi= 4*PI;
    QUICKDouble b = 0.0042;
    *dfdggo=0.0;
    
    QUICKDouble rhothirds=pow(density,(1.0/3.0));
    QUICKDouble rho4thirds=pow(rhothirds,(4.0));
    QUICKDouble x = sqrt(gx*gx+gy*gy+gz*gz)/rho4thirds;
    QUICKDouble arcsinhx = log(x+sqrt(x*x+1));
    QUICKDouble denom = 1.0 + 6.0*b*x*arcsinhx;
    QUICKDouble gofx = -0.930525736349100025000 -b*x*x/denom;
    QUICKDouble gprimeofx =(6.0*b*b*x*x*(x/sqrt(x*x+1)-arcsinhx) -2.0*b*x) /(denom*denom);
    
    *dfdr=(4.0/3.0)*rhothirds*(gofx -x*gprimeofx);
    *dfdgg= .50 * gprimeofx /(x * rho4thirds);
    return;
}

__device__ void lyp(QUICKDouble pa, QUICKDouble pb, QUICKDouble gax, QUICKDouble gay, QUICKDouble gaz, QUICKDouble gbx, QUICKDouble gby, QUICKDouble gbz,
                    QUICKDouble* dfdr, QUICKDouble* dfdgg, QUICKDouble* dfdggo)
{
    
    // Given the densites and the two gradients, (with gother being for
    // the spin that density is not, i.e. beta if density is alpha) return
    // the derivative of the lyp correlation functional with regard to the density
    // and the derivatives with regard to the gradient invariants.
    
    // Some params
    //pi=3.14159265358979323850
    
    QUICKDouble a = .049180;
    QUICKDouble b = .1320;
    QUICKDouble c = .25330;
    QUICKDouble d = .3490;
    QUICKDouble CF = .30*pow((3.0*PI*PI),(2.0/3.0));
    
    // And some required quanitities.
    
    QUICKDouble gaa = (gax*gax+gay*gay+gaz*gaz);
    QUICKDouble gbb = (gbx*gbx+gby*gby+gbz*gbz);
    QUICKDouble gab = (gax*gbx+gay*gby+gaz*gbz);
    QUICKDouble ptot = pa+pb;
    QUICKDouble ptone3rd = pow(ptot,(1.0/3.0));
    QUICKDouble third = 1.0/3.0;
    QUICKDouble third2 = 2.0/3.0;
    
    QUICKDouble w = exp(-c/ptone3rd) * pow(ptone3rd,(-11.0))/ \
    //                                -----------------------
                                        (1.0 + d/ptone3rd);
    
    
    QUICKDouble abw = a * b * w;
    QUICKDouble abwpapb = abw * pa * pb;
    QUICKDouble dabw     = (a * b * exp( -c / ptone3rd) * (c * d* pow(ptone3rd,2.0) + (c - 10.0 * d) * ptot - 11.0 * pow(ptone3rd,4.0)))/ \
                          // -----------------------------------------------------------------------------------------------------
                                             (3.0 * (pow(ptone3rd,16.0)) * pow((d+ptone3rd),2.0));
    
    QUICKDouble dabwpapb = (     a * b * exp( -c / ptone3rd) * pb * (c * d * pa * pow(ptone3rd,2.0) \
                               + (pow(ptone3rd,4.0)) * (-8.0 * pa + 3.0 * pb) \
                               + ptot * (c * pa - 7.0 * d * pa + 3.0 * d * pb))  )/ \
                          //-------------------------------------------------------------------------
                                             (3.0 * (pow(ptone3rd,16.0)) * pow((d+ptone3rd),2.0));
    
    QUICKDouble delta = c/ptone3rd + (d/ptone3rd)/ (1 + d/ptone3rd);
    
    
    *dfdr = - dabwpapb * CF * 12.6992084 * (pow(pa,(8.0/3.0)) + pow(pb,(8.0/3.0))) \
            - dabwpapb * (47.0/18.0 - 7.0 * delta/18.0) * (gaa + gbb + 2.0*gab)
            + dabwpapb * (2.50 - delta/18.0) * (gaa + gbb) \
            + dabwpapb * ((delta - 11.0)/9.0) * (pa * gaa / ptot + pb * gbb / ptot) \
            + dabw     *  (third2 * ptot * ptot * (2.0*gab) \
            + pa * pa * gbb \
            + pb * pb * gaa) \
    
            + ( -4.0 * a * pb * (3.0 * pb * pow(ptot,third) + d * (pa + 3.0*pb)))/\
            //--------------------------------------------------------------------
                    (3.0 * pow(ptot,5.0/3.0) * pow((d + pow(ptot,third)),2.0)) \
    
            + (-64.0 * pow(2.0,third2) * abwpapb * CF * pow(pa,(5.0/3.0)))/3.0
            - (7.0 * abwpapb * (gaa+2.0*gab + gbb) * (c + (d*pow(ptot,third2))/pow((d + pow(ptot,third)),2.0)))/ \
            //-----------------------------------------------------------------------------------------------------
                    (54.0*pow(ptot,4.0/3.0))\
            + (abwpapb * (gaa + gbb) * (c +(d*pow(ptot,third2))/ pow((d + pow((ptot),(third))),2.0)))/ \
            //-----------------------------------------------------------------------------------------------------
                    (54.0*pow((ptot),(4.0/3.0)))
            + (abwpapb * (-30.0 * d * d * (gaa-gbb) * pb * pow((ptot),(third)) 
                          -33.0 * (gaa - gbb) * pb * ptot
                          -c * (gaa*(pa-3.0*pb) + 4.0*gbb*pb) * pow((d + pow((ptot),(third))),2.0)
                          -d * pow((pa+pb),(third2))*(-62.0*gbb*pb+gaa*(pa+63.0*pb)))) / \
            //-----------------------------------------------------------------------------------------------------
                   (27.0 * pow((ptot),(7.0/3.0)) * pow((d + pow((ptot),(third))),2.0)) 
                  +(4.0 * abw*(gaa + 2.0*gab + gbb)*(ptot))/3.0
                  +(2.0*abw*gbb*(pa-2.0*pb))/3.0 
                  +(-4.0*abw*gaa*(ptot))/3.0;
    
    *dfdgg =  0.0 -abwpapb*(47.0/18.0 - 7.0*delta/18.0) +abwpapb*(2.50 - delta/18.0) \
    +abwpapb*((delta - 11.0)/9.0)*(pa/ptot) +abw*(2.0/3.0)*ptot*ptot -abw*((2.0/3.0)*ptot*ptot - pb*pb);
    
    *dfdggo= 0.0 -abwpapb*(47.0/18.0 - 7.0*delta/18.0)*2.0 +abw*(2.0/3.0)*ptot*ptot*2.0;
}


__device__ __forceinline__ QUICKDouble lyp_e(QUICKDouble pa, QUICKDouble pb, QUICKDouble gax, QUICKDouble gay, QUICKDouble gaz,
                               QUICKDouble gbx,     QUICKDouble gby,      QUICKDouble gbz)
{
    // Given the densities and the two gradients, return the energy, return
    // the LYP correlation energy.
    
    // Note the code is kind of garbled, as Mathematic was used to write it.
    
    // Some params:
    
    //pi=3.1415926535897932385d0
    
    
    /*
        Lee-Yang-Parr correction functional
                  _        -a
        E(LYP) = |  rou [ -----[1+b*Cf*exp(-cx)]]dr
                -         1+dx
                  _    __            exp(-cx)   1        7        dx
               + |  ab|\/rou|^2*x^5 ---------- --- [ 1 + -(cx + ------)]dr
                -                      1+dx     24       3       1+dx
                     3           2
         where Cf = --(3 PI^2)^(--)
                    10           3
               a  = 0.04918
               b  = 0.132
               c  = 0.2533
               d  = 0.349
               x  = rou^(1/3)
     */
    QUICKDouble const a = .04918;
    QUICKDouble const b = .132;
    QUICKDouble const c = .2533;
    QUICKDouble const d = .349;
    QUICKDouble const CF = .3*pow((3.0*PI*PI),2.0/3.0);
    
    // And some required quanitities.
    
    QUICKDouble gaa = (gax*gax+gay*gay+gaz*gaz);
    QUICKDouble gbb = (gbx*gbx+gby*gby+gbz*gbz);
    QUICKDouble gab = (gax*gbx+gay*gby+gaz*gbz);
    QUICKDouble ptot = pa+pb;
    QUICKDouble ptone3rd = pow(ptot,(1.0/3.0));
    
    QUICKDouble t1 = d/ptone3rd;
    QUICKDouble t2 = 1.0 + t1;
    QUICKDouble w = exp(-c/ptone3rd)*(pow(ptone3rd,-11.0))/t2;
    QUICKDouble abw = a*b*w;
    QUICKDouble abwpapb = abw*pa*pb;
    QUICKDouble delta = c/ptone3rd + t1/ (1 + t1);
    QUICKDouble const c1 = pow(2.0,(11.0/3.0));
    
    
    QUICKDouble e = - 4.0 * a * pa * pb / ( ptot * t2 )
                    - abwpapb * CF * c1 * (pow(pa,8.0/3.0) + pow(pb,(8.0/3.0))) \
                    - abwpapb * (47.0/18.0 - 7.0 * delta/18.0) * (gaa + gbb + 2.0*gab)
                    + abwpapb * (2.50   - delta/18.0)* (gaa + gbb) \
                    + abwpapb * ((delta - 11.0)/9.0) * (pa*gaa/ptot+pb*gbb/ptot) \
                    + abw     * ( 2.0/3.0) * ptot * ptot * (gaa + gbb + 2.0 * gab) \
                    - abw     * ((2.0/3.0) * ptot * ptot - pa * pa)*gbb
                    - abw     * ((2.0/3.0) * ptot * ptot - pb * pb)*gaa;
    return e;
}
                                
__device__  __forceinline__ QUICKDouble b3lyp_e(QUICKDouble rho, QUICKDouble sigma)
{
  /*
  P.J. Stephens, F.J. Devlin, C.F. Chabalowski, M.J. Frisch
  Ab initio calculation of vibrational absorption and circular
  dichroism spectra using density functional force fields
  J. Phys. Chem. 98 (1994) 11623-11627
  
  CITATION:
  Functionals were obtained from the Density Functional Repository
  as developed and distributed by the Quantum Chemistry Group,
  CCLRC Daresbury Laboratory, Daresbury, Cheshire, WA4 4AD
  United Kingdom. Contact Huub van Dam (h.j.j.vandam@dl.ac.uk) or
  Paul Sherwood for further information.
  
  COPYRIGHT:
  
  Users may incorporate the source code into software packages and
  redistribute the source code provided the source code is not
  changed in anyway and is properly cited in any documentation or
  publication related to its use.
  
  ACKNOWLEDGEMENT:
  
  The source code was generated using Maple 8 through a modified
  version of the dfauto script published in:
  
  R. Strange, F.R. Manby, P.J. Knowles
  Automatic code generation in density functional theory
  Comp. Phys. Comm. 136 (2001) 310-318.
  
  */
    QUICKDouble Eelxc = 0.0e0;
    QUICKDouble t2 = pow(rho, (1.e0/3.e0));
    QUICKDouble t3 = t2*rho;
    QUICKDouble t5 = 1/t3;
    QUICKDouble t7 = sqrt(sigma);
    QUICKDouble t8 = t7*t5;
    QUICKDouble t10 = log(0.1259921049894873e1*t8+sqrt(1+0.1587401051968199e1*t8*t8));
    QUICKDouble t17 = 1/t2;
    QUICKDouble t20 = 1/(1.e0+0.349e0*t17);
    QUICKDouble t23 = 0.2533*t17;
    QUICKDouble t24 = exp(-t23);
    QUICKDouble t26 = rho*rho;
    QUICKDouble t28 = t2*t2;
    QUICKDouble t34 = t17*t20;
    QUICKDouble t56 = 1/rho;
    QUICKDouble t57 = pow(t56,(1.e0/3.e0));
    QUICKDouble t59 = pow(t56,(1.e0/6.e0));
    QUICKDouble t62 = 1/(0.6203504908994*t57+0.1029581201158544e2*t59+0.427198e2);
    QUICKDouble t65 = log(0.6203504908994*t57*t62);
    QUICKDouble t71 = atan(0.448998886412873e-1/(0.1575246635799487e1*t59+0.13072e2));
    QUICKDouble t75 = pow((0.7876233178997433e0*t59+0.409286e0),2);
    QUICKDouble t77 = log(t75*t62);
    Eelxc = -0.5908470131056179e0*t3
            -0.3810001254882096e-2*t5*sigma/(1.e0+0.317500104573508e-1*t8*t10)
            -0.398358e-1*t20*rho 
            -0.52583256e-2*t24*t20/t28/t26/rho*(0.25e0*t26*(0.1148493600075277e2*t28*t26+(0.2611111111111111e1-0.9850555555555556e-1*t17-
    0.1357222222222222e0*t34)*sigma-0.5*(0.25e1-0.1407222222222222e-1*t17-0.1938888888888889e-1*t34)*sigma-
    0.2777777777777778e-1*(t23+0.349*t34-11.0)*sigma)-0.4583333333333333e0*t26*sigma)+
    0.19*rho*(0.310907e-1*t65+0.205219729378375e2*t71+0.4431373767749538e-2*t77);
    return Eelxc;
}
            
__device__ QUICKDouble b3lypf(QUICKDouble rho, QUICKDouble sigma, QUICKDouble* dfdr)
{
    /*
     P.J. Stephens, F.J. Devlin, C.F. Chabalowski, M.J. Frisch
     Ab initio calculation of vibrational absorption and circular
     dichroism spectra using density functional force fields
     J. Phys. Chem. 98 (1994) 11623-11627
     
     CITATION:
     Functionals were obtained from the Density Functional Repository
     as developed and distributed by the Quantum Chemistry Group,
     CCLRC Daresbury Laboratory, Daresbury, Cheshire, WA4 4AD
     United Kingdom. Contact Huub van Dam (h.j.j.vandam@dl.ac.uk) or
     Paul Sherwood for further information.
     
     COPYRIGHT:
     
     Users may incorporate the source code into software packages and
     redistribute the source code provided the source code is not
     changed in anyway and is properly cited in any documentation or
     publication related to its use.
     
     ACKNOWLEDGEMENT:
     
     The source code was generated using Maple 8 through a modified
     version of the dfauto script published in:
     
     R. Strange, F.R. Manby, P.J. Knowles
     Automatic code generation in density functional theory
     Comp. Phys. Comm. 136 (2001) 310-318.
     
     */
    QUICKDouble dot;
    QUICKDouble t2 = pow(rho, (1.e0/3.e0));
    QUICKDouble t3 = t2*rho;
    QUICKDouble t5 = 1/t3;
    QUICKDouble t6 = t5*sigma;
    QUICKDouble t7 = sqrt(sigma);
    QUICKDouble t8 = t7*t5;
    QUICKDouble t10 = log(0.1259921049894873e1*t8+sqrt(1+0.1587401051968199e1*t8*t8));
    QUICKDouble t13 = 1.0e0+0.317500104573508e-1*t8*t10;
    QUICKDouble t14 = 1.0e0/t13;
    QUICKDouble t17 = 1/t2;
    QUICKDouble t19 = 1.e0+0.349e0*t17;
    QUICKDouble t20 = 1/t19;
    QUICKDouble t23 = 0.2533e0*t17;
    QUICKDouble t24 = exp(-t23);
    QUICKDouble t25 = t24*t20;
    QUICKDouble t26 = rho*rho;
    QUICKDouble t28 = t2*t2;
    QUICKDouble t30 = 1/t28/t26/rho;
    QUICKDouble t31 = t28*t26;
    QUICKDouble t34 = t17*t20;
    
    QUICKDouble t36 = 0.2611111111111111e1-0.9850555555555556e-1*t17-0.1357222222222222e0*t34;
    QUICKDouble t44 = t23+0.349e0*t34-11.e0;
    QUICKDouble t47 = 0.1148493600075277e2*t31+t36*sigma-0.5e0*(0.25e1-0.1407222222222222e-1*t17- \
                      0.1938888888888889e-1*t34)*sigma-0.2777777777777778e-1*t44*sigma;
    
    QUICKDouble t52 = 0.25e0*t26*t47-0.4583333333333333e0*t26*sigma;
    QUICKDouble t56 = 1/rho;
    QUICKDouble t57 = pow(t56,(1.e0/3.e0));
    QUICKDouble t59 = pow(t56,(1.e0/6.e0));
    QUICKDouble t61 = 0.6203504908994e0*t57+0.1029581201158544e2*t59+0.427198e2;
    QUICKDouble t62 = 1/t61;
    QUICKDouble t65 = log(0.6203504908994e0*t57*t62);
    QUICKDouble t68 = 0.1575246635799487e1*t59+0.13072e2;
    QUICKDouble t71 = atan(0.448998886412873e-1/t68);
    QUICKDouble t74 = 0.7876233178997433e0*t59+0.409286e0;
    QUICKDouble t75 = t74*t74;
    QUICKDouble t77 = log(t75*t62);
    QUICKDouble t84 = 1/t2/t26;
    QUICKDouble t88 = t13*t13;
    QUICKDouble t89 = 1/t88;
    QUICKDouble t94 = 1/t31;
    QUICKDouble t98 = sqrt(1.e0+0.1587401051968199e1*sigma*t94);
    QUICKDouble t99 = 1/t98;
    QUICKDouble t109 = t28*rho;
    QUICKDouble t112 = t44*t56*sigma;
    QUICKDouble t117 = rho*sigma;
    QUICKDouble t123 = t19*t19;
    QUICKDouble t124 = 1/t123;
    QUICKDouble t127 = t26*t26;
    QUICKDouble t129 = 1/t127/rho;
    QUICKDouble t144 = t5*t20;
    QUICKDouble t146 = 1/t109;
    QUICKDouble t147 = t146*t124;
    QUICKDouble t175 = t57*t57;
    QUICKDouble t176 = 1/t175;
    QUICKDouble t178 = 1/t26;
    QUICKDouble t181 = t61*t61;
    QUICKDouble t182 = 1/t181;
    QUICKDouble t186 = t59*t59;
    QUICKDouble t187 = t186*t186;
    QUICKDouble t189 = 1/t187/t59;
    QUICKDouble t190 = t189*t178;
    QUICKDouble t192 = -0.2067834969664667e0*t176*t178-0.1715968668597574e1*t190;
    QUICKDouble t200 = t68*t68;
    QUICKDouble t201 = 1/t200;
    
    *dfdr=-0.7877960174741572e0*t2+0.5080001673176129e-2*t84*sigma*t14+0.1905000627441048e-2*t6*t89*
    (-0.8466669455293548e-1*t7*t84*t10-0.106673350692263e0*sigma*t30*t99)-0.398358e-1*t20-0.52583256e-2*t25*t30*(0.5e0*rho*t47+0.25e0*t26* 
    (0.3062649600200738e2*t109-0.2777777777777778e-1*t112)-0.25e0*t117)-0.46342314e-2*t124*t17-0.44397795816e-3*t129*t24*t20*t52;
    
    *dfdr= *dfdr-0.6117185448e-3*t24*t124*t129*t52+0.192805272e-1*t25/t28/t127*t52-0.52583256e-2*t25*t30*(0.25e0*t26*((0.3283518518518519e-1*t5+0.4524074074074074e-1*t144 
    -0.1578901851851852e-1*t147)*sigma-0.5e0*(0.4690740740740741e-2*t5+0.6462962962962963e-2*t144-0.2255574074074074e-2*t147)*sigma 
    -0.2777777777777778e-1*(-0.8443333333333333e-1*t5-0.1163333333333333e0*t144+0.4060033333333333e-1*t147)*sigma+0.2777777777777778e-1*t112)-0.6666666666666667e0*t117);
    
    
    *dfdr = *dfdr+0.5907233e-2*t65+0.3899174858189126e1*t71+0.8419610158724123e-3*t77+
    0.19e0*rho*(0.5011795824473985e-1*(-0.2067834969664667e0*t176*t62*t178-0.6203504908994e0*t57*t182*t192)/t57*t61
               +0.2419143800947354e0*t201*t189*t178/(1.e0+0.2016e-2*t201)+
               0.4431373767749538e-2*(-0.2625411059665811e0*t74*t62*t190-1.e0*t75*t182*t192)/t75*t61);
    dot = -0.1524000501952839e-1*t5*t14+0.3810001254882096e-2*t6*t89*(0.6350002091470161e-1/t7*t5*t10 
        +0.8000501301919725e-1*t94*t99)+0.5842584e-3*t25*t146-0.210333024e-1*t25*t30*(0.25e0*t26*t36-0.6666666666666667e0*t26);
         
    return dot;
         
}


__device__ int gen_oh(int code, int num, QUICKDouble* x, QUICKDouble* y, QUICKDouble* z, QUICKDouble* w, QUICKDouble a, QUICKDouble b, QUICKDouble v)
{
    /*
     ! vd
     ! vd   This subroutine is part of a set of subroutines that generate
     ! vd   Lebedev grids [1-6] for integration on a sphere. The original
     ! vd   C-code [1] was kindly provided by Dr. Dmitri N. Laikov and
     ! vd   translated into fortran by Dr. Christoph van Wuellen.
     ! vd   
     ! vd   Users of this code are asked to include reference [1] in their
     ! vd   publications, and in the user- and programmers-manuals
     ! vd   describing their codes.
     ! vd
     ! vd   This code was distributed through CCL (http://www.ccl.net/).
     ! vd
     ! vd   [1] V.I. Lebedev, and D.N. Laikov
     ! vd       "A quadrature formula for the sphere of the 131st
     ! vd        algebraic order of accuracy"
     ! vd       doklady Mathematics, Vol. 59, No. 3, 1999, pp. 477-481.
     ! vd
     ! vd   [2] V.I. Lebedev
     ! vd       "A quadrature formula for the sphere of 59th algebraic
     ! vd        order of accuracy"
     ! vd       Russian Acad. Sci. dokl. Math., Vol. 50, 1995, pp. 283-286.
     ! vd
     ! vd   [3] V.I. Lebedev, and A.L. Skorokhodov
     ! vd       "Quadrature formulas of orders 41, 47, and 53 for the sphere"
     ! vd       Russian Acad. Sci. dokl. Math., Vol. 45, 1992, pp. 587-592.
     ! vd
     ! vd   [4] V.I. Lebedev
     ! vd       "Spherical quadrature formulas exact to orders 25-29"
     ! vd       Siberian Mathematical Journal, Vol. 18, 1977, pp. 99-107.
     ! vd
     ! vd   [5] V.I. Lebedev
     ! vd       "Quadratures on a sphere"
     ! vd       Computational Mathematics and Mathematical Physics, Vol. 16,
     ! vd       1976, pp. 10-24.
     ! vd
     ! vd   [6] V.I. Lebedev
     ! vd       "Values of the nodes and weights of ninth to seventeenth
     ! vd        order Gauss-Markov quadrature formulae invariant under the
     ! vd        octahedron group with inversion"
     ! vd       Computational Mathematics and Mathematical Physics, Vol. 15,
     ! vd       1975, pp. 44-51.
     ! vd
     ! w
     ! w    Given a point on a sphere (specified by a and b), generate all
     ! w    the equivalent points under Oh symmetry, making grid points with
     ! w    weight v.
     ! w    The variable num is increased by the number of different points
     ! w    generated.
     ! w
     ! w    Depending on code, there are 6...48 different but equivalent
     ! w    points.
     ! w
     ! w    code=1:   (0,0,1) etc                                (  6 points)
     ! w    code=2:   (0,a,a) etc, a=1/sqrt(2)                   ( 12 points)
     ! w    code=3:   (a,a,a) etc, a=1/sqrt(3)                   (  8 points)
     ! w    code=4:   (a,a,b) etc, b=sqrt(1-2 a^2)               ( 24 points)
     ! w    code=5:   (a,b,0) etc, b=sqrt(1-a^2), a input        ( 24 points)
     ! w    code=6:   (a,b,c) etc, c=sqrt(1-a^2-b^2), a/b input  ( 48 points)
     ! w
     */
    QUICKDouble c;
    switch (code) {
        case 1:
        {
            a=1.0e0;
            x[0 + num] =      a; y[0 + num] =  0.0e0; z[0 + num] =  0.0e0; w[0 + num] =  v;
            x[1 + num] =     -a; y[1 + num] =  0.0e0; z[1 + num] =  0.0e0; w[1 + num] =  v;
            x[2 + num] =  0.0e0; y[2 + num] =      a; z[2 + num] =  0.0e0; w[2 + num] =  v;
            x[3 + num] =  0.0e0; y[3 + num] =     -a; z[3 + num] =  0.0e0; w[3 + num] =  v;
            x[4 + num] =  0.0e0; y[4 + num] =  0.0e0; z[4 + num] =      a; w[4 + num] =  v;
            x[5 + num] =  0.0e0; y[5 + num] =  0.0e0; z[5 + num] =     -a; w[5 + num] =  v;
            num=num+6;
            break;
        }
        case 2:
        {
            a=sqrt(0.5e0);
            x[0  + num] =  0e0;  y[0  + num] =    a;  z[0  + num] =    a;  w[0  + num] =  v;
            x[1  + num] =  0e0;  y[1  + num] =   -a;  z[1  + num] =    a;  w[1  + num] =  v;
            x[2  + num] =  0e0;  y[2  + num] =    a;  z[2  + num] =   -a;  w[2  + num] =  v;
            x[3  + num] =  0e0;  y[3  + num] =   -a;  z[3  + num] =   -a;  w[3  + num] =  v;
            x[4  + num] =    a;  y[4  + num] =  0e0;  z[4  + num] =    a;  w[4  + num] =  v;
            x[5  + num] =   -a;  y[5  + num] =  0e0;  z[5  + num] =    a;  w[5  + num] =  v;
            x[6  + num] =    a;  y[6  + num] =  0e0;  z[6  + num] =   -a;  w[6  + num] =  v;
            x[7  + num] =   -a;  y[7  + num] =  0e0;  z[7  + num] =   -a;  w[7  + num] =  v;
            x[8  + num] =    a;  y[8  + num] =    a;  z[8  + num] =  0e0;  w[8  + num] =  v;
            x[9  + num] =   -a;  y[9  + num] =    a;  z[9  + num] =  0e0;  w[9  + num] =  v;
            x[10 + num] =    a;  y[10 + num] =   -a;  z[10 + num] =  0e0;  w[10 + num] =  v;
            x[11 + num] =   -a;  y[11 + num] =   -a;  z[11 + num] =  0e0;  w[11 + num] =  v;
            num=num+12;
            break;
        }
        case 3:
        {
            a = sqrt(1e0/3e0);
            x[0 + num] =  a;  y[0 + num] =  a;  z[0 + num] =  a;  w[0 + num] =  v;
            x[1 + num] = -a;  y[1 + num] =  a;  z[1 + num] =  a;  w[1 + num] =  v;
            x[2 + num] =  a;  y[2 + num] = -a;  z[2 + num] =  a;  w[2 + num] =  v;
            x[3 + num] = -a;  y[3 + num] = -a;  z[3 + num] =  a;  w[3 + num] =  v;
            x[4 + num] =  a;  y[4 + num] =  a;  z[4 + num] = -a;  w[4 + num] =  v;
            x[5 + num] = -a;  y[5 + num] =  a;  z[5 + num] = -a;  w[5 + num] =  v;
            x[6 + num] =  a;  y[6 + num] = -a;  z[6 + num] = -a;  w[6 + num] =  v;  
            x[7 + num] = -a;  y[7 + num] = -a;  z[7 + num] = -a;  w[7 + num] =  v;
            num=num+8;
            break;
        }
        case 4:
        {
            b = sqrt(1e0 - 2e0*a*a);
            x[0  + num] =  a;  y[0  + num] =  a;  z[0  + num] =  b;  w[0  + num] =  v;
            x[1  + num] = -a;  y[1  + num] =  a;  z[1  + num] =  b;  w[1  + num] =  v;
            x[2  + num] =  a;  y[2  + num] = -a;  z[2  + num] =  b;  w[2  + num] =  v;
            x[3  + num] = -a;  y[3  + num] = -a;  z[3  + num] =  b;  w[3  + num] =  v;
            x[4  + num] =  a;  y[4  + num] =  a;  z[4  + num] = -b;  w[4  + num] =  v;
            x[5  + num] = -a;  y[5  + num] =  a;  z[5  + num] = -b;  w[5  + num] =  v;
            x[6  + num] =  a;  y[6  + num] = -a;  z[6  + num] = -b;  w[6  + num] =  v;
            x[7  + num] = -a;  y[7  + num] = -a;  z[7  + num] = -b;  w[7  + num] =  v;
            x[8  + num] =  a;  y[8  + num] =  b;  z[8  + num] =  a;  w[8  + num] =  v;
            x[9  + num] = -a;  y[9  + num] =  b;  z[9  + num] =  a;  w[9  + num] =  v;
            x[10 + num] =  a;  y[10 + num] = -b;  z[10 + num] =  a;  w[10 + num] =  v;
            x[11 + num] = -a;  y[11 + num] = -b;  z[11 + num] =  a;  w[11 + num] =  v;
            x[12 + num] =  a;  y[12 + num] =  b;  z[12 + num] = -a;  w[12 + num] =  v;
            x[13 + num] = -a;  y[13 + num] =  b;  z[13 + num] = -a;  w[13 + num] =  v;
            x[14 + num] =  a;  y[14 + num] = -b;  z[14 + num] = -a;  w[14 + num] =  v;
            x[15 + num] = -a;  y[15 + num] = -b;  z[15 + num] = -a;  w[15 + num] =  v;
            x[16 + num] =  b;  y[16 + num] =  a;  z[16 + num] =  a;  w[16 + num] =  v;
            x[17 + num] = -b;  y[17 + num] =  a;  z[17 + num] =  a;  w[17 + num] =  v;
            x[18 + num] =  b;  y[18 + num] = -a;  z[18 + num] =  a;  w[18 + num] =  v;
            x[19 + num] = -b;  y[19 + num] = -a;  z[19 + num] =  a;  w[19 + num] =  v;
            x[20 + num] =  b;  y[20 + num] =  a;  z[20 + num] = -a;  w[20 + num] =  v;
            x[21 + num] = -b;  y[21 + num] =  a;  z[21 + num] = -a;  w[21 + num] =  v;
            x[22 + num] =  b;  y[22 + num] = -a;  z[22 + num] = -a;  w[22 + num] =  v;
            x[23 + num] = -b;  y[23 + num] = -a;  z[23 + num] = -a;  w[23 + num] =  v;
            num = num + 24;
            break;
        }
        case 5:
        {
            b=sqrt(1e0-a*a);
            x[0  + num] =    a;  y[0  + num] =    b;  z[0  + num] =  0e0;  w[0  + num] =  v;
            x[1  + num] =   -a;  y[1  + num] =    b;  z[1  + num] =  0e0;  w[1  + num] =  v;
            x[2  + num] =    a;  y[2  + num] =   -b;  z[2  + num] =  0e0;  w[2  + num] =  v;
            x[3  + num] =   -a;  y[3  + num] =   -b;  z[3  + num] =  0e0;  w[3  + num] =  v;
            x[4  + num] =    b;  y[4  + num] =    a;  z[4  + num] =  0e0;  w[4  + num] =  v;
            x[5  + num] =   -b;  y[5  + num] =    a;  z[5  + num] =  0e0;  w[5  + num] =  v;
            x[6  + num] =    b;  y[6  + num] =   -a;  z[6  + num] =  0e0;  w[6  + num] =  v;
            x[7  + num] =   -b;  y[7  + num] =   -a;  z[7  + num] =  0e0;  w[7  + num] =  v;
            x[8  + num] =    a;  y[8  + num] =  0e0;  z[8  + num] =    b;  w[8  + num] =  v;
            x[9  + num] =   -a;  y[9  + num] =  0e0;  z[9  + num] =    b;  w[9  + num] =  v;
            x[10 + num] =    a;  y[10 + num] =  0e0;  z[10 + num] =   -b;  w[10 + num] =  v;
            x[11 + num] =   -a;  y[11 + num] =  0e0;  z[11 + num] =   -b;  w[11 + num] =  v;
            x[12 + num] =    b;  y[12 + num] =  0e0;  z[12 + num] =    a;  w[12 + num] =  v;
            x[13 + num] =   -b;  y[13 + num] =  0e0;  z[13 + num] =    a;  w[13 + num] =  v;
            x[14 + num] =    b;  y[14 + num] =  0e0;  z[14 + num] =   -a;  w[14 + num] =  v;
            x[15 + num] =   -b;  y[15 + num] =  0e0;  z[15 + num] =   -a;  w[15 + num] =  v;
            x[16 + num] =  0e0;  y[16 + num] =    a;  z[16 + num] =    b;  w[16 + num] =  v;
            x[17 + num] =  0e0;  y[17 + num] =   -a;  z[17 + num] =    b;  w[17 + num] =  v;
            x[18 + num] =  0e0;  y[18 + num] =    a;  z[18 + num] =   -b;  w[18 + num] =  v;
            x[19 + num] =  0e0;  y[19 + num] =   -a;  z[19 + num] =   -b;  w[19 + num] =  v;
            x[20 + num] =  0e0;  y[20 + num] =    b;  z[20 + num] =    a;  w[20 + num] =  v;
            x[21 + num] =  0e0;  y[21 + num] =   -b;  z[21 + num] =    a;  w[21 + num] =  v;
            x[22 + num] =  0e0;  y[22 + num] =    b;  z[22 + num] =   -a;  w[22 + num] =  v;
            x[23 + num] =  0e0;  y[23 + num] =   -b;  z[23 + num] =   -a;  w[23 + num] =  v;
            num=num+24  ;
            break  ;
        }
        case 6:
        {
            c=sqrt(1e0 - a*a - b*b);
            x[0  + num] =  a;  y[0  + num] =  b;  z[0  + num] =  c;  w[0  + num] =  v;
            x[1  + num] = -a;  y[1  + num] =  b;  z[1  + num] =  c;  w[1  + num] =  v;
            x[2  + num] =  a;  y[2  + num] = -b;  z[2  + num] =  c;  w[2  + num] =  v;
            x[3  + num] = -a;  y[3  + num] = -b;  z[3  + num] =  c;  w[3  + num] =  v;
            x[4  + num] =  a;  y[4  + num] =  b;  z[4  + num] = -c;  w[4  + num] =  v;
            x[5  + num] = -a;  y[5  + num] =  b;  z[5  + num] = -c;  w[5  + num] =  v;
            x[6  + num] =  a;  y[6  + num] = -b;  z[6  + num] = -c;  w[6  + num] =  v;
            x[7  + num] = -a;  y[7  + num] = -b;  z[7  + num] = -c;  w[7  + num] =  v;
            x[8  + num] =  a;  y[8  + num] =  c;  z[8  + num] =  b;  w[8  + num] =  v;
            x[9  + num] = -a;  y[9  + num] =  c;  z[9  + num] =  b;  w[9  + num] =  v;
            x[10 + num] =  a;  y[10 + num] = -c;  z[10 + num] =  b;  w[10 + num] =  v;
            x[11 + num] = -a;  y[11 + num] = -c;  z[11 + num] =  b;  w[11 + num] =  v;
            x[12 + num] =  a;  y[12 + num] =  c;  z[12 + num] = -b;  w[12 + num] =  v;
            x[13 + num] = -a;  y[13 + num] =  c;  z[13 + num] = -b;  w[13 + num] =  v;
            x[14 + num] =  a;  y[14 + num] = -c;  z[14 + num] = -b;  w[14 + num] =  v;
            x[15 + num] = -a;  y[15 + num] = -c;  z[15 + num] = -b;  w[15 + num] =  v;
            x[16 + num] =  b;  y[16 + num] =  a;  z[16 + num] =  c;  w[16 + num] =  v;
            x[17 + num] = -b;  y[17 + num] =  a;  z[17 + num] =  c;  w[17 + num] =  v;
            x[18 + num] =  b;  y[18 + num] = -a;  z[18 + num] =  c;  w[18 + num] =  v;
            x[19 + num] = -b;  y[19 + num] = -a;  z[19 + num] =  c;  w[19 + num] =  v;
            x[20 + num] =  b;  y[20 + num] =  a;  z[20 + num] = -c;  w[20 + num] =  v;
            x[21 + num] = -b;  y[21 + num] =  a;  z[21 + num] = -c;  w[21 + num] =  v;
            x[22 + num] =  b;  y[22 + num] = -a;  z[22 + num] = -c;  w[22 + num] =  v;
            x[23 + num] = -b;  y[23 + num] = -a;  z[23 + num] = -c;  w[23 + num] =  v;
            x[24 + num] =  b;  y[24 + num] =  c;  z[24 + num] =  a;  w[24 + num] =  v;
            x[25 + num] = -b;  y[25 + num] =  c;  z[25 + num] =  a;  w[25 + num] =  v;
            x[26 + num] =  b;  y[26 + num] = -c;  z[26 + num] =  a;  w[26 + num] =  v;
            x[27 + num] = -b;  y[27 + num] = -c;  z[27 + num] =  a;  w[27 + num] =  v;
            x[28 + num] =  b;  y[28 + num] =  c;  z[28 + num] = -a;  w[28 + num] =  v;
            x[29 + num] = -b;  y[29 + num] =  c;  z[29 + num] = -a;  w[29 + num] =  v;
            x[30 + num] =  b;  y[30 + num] = -c;  z[30 + num] = -a;  w[30 + num] =  v;
            x[31 + num] = -b;  y[31 + num] = -c;  z[31 + num] = -a;  w[31 + num] =  v;
            x[32 + num] =  c;  y[32 + num] =  a;  z[32 + num] =  b;  w[32 + num] =  v;
            x[33 + num] = -c;  y[33 + num] =  a;  z[33 + num] =  b;  w[33 + num] =  v;
            x[34 + num] =  c;  y[34 + num] = -a;  z[34 + num] =  b;  w[34 + num] =  v;
            x[35 + num] = -c;  y[35 + num] = -a;  z[35 + num] =  b;  w[35 + num] =  v;
            x[36 + num] =  c;  y[36 + num] =  a;  z[36 + num] = -b;  w[36 + num] =  v;
            x[37 + num] = -c;  y[37 + num] =  a;  z[37 + num] = -b;  w[37 + num] =  v;
            x[38 + num] =  c;  y[38 + num] = -a;  z[38 + num] = -b;  w[38 + num] =  v;
            x[39 + num] = -c;  y[39 + num] = -a;  z[39 + num] = -b;  w[39 + num] =  v;
            x[40 + num] =  c;  y[40 + num] =  b;  z[40 + num] =  a;  w[40 + num] =  v;
            x[41 + num] = -c;  y[41 + num] =  b;  z[41 + num] =  a;  w[41 + num] =  v;
            x[42 + num] =  c;  y[42 + num] = -b;  z[42 + num] =  a;  w[42 + num] =  v;
            x[43 + num] = -c;  y[43 + num] = -b;  z[43 + num] =  a;  w[43 + num] =  v;
            x[44 + num] =  c;  y[44 + num] =  b;  z[44 + num] = -a;  w[44 + num] =  v;
            x[45 + num] = -c;  y[45 + num] =  b;  z[45 + num] = -a;  w[45 + num] =  v;
            x[46 + num] =  c;  y[46 + num] = -b;  z[46 + num] = -a;  w[46 + num] =  v;
            x[47 + num] = -c;  y[47 + num] = -b;  z[47 + num] = -a;  w[47 + num] =  v;
            num=num+48;
            break;
        default:
            break;
        }
    }
    return num;
}


__device__ void LD0006(QUICKDouble* x, QUICKDouble* y, QUICKDouble* z, QUICKDouble* w, int N)
{
    /*
     ! W
     ! W    LEBEDEV    6-POINT ANGULAR GRID
     ! W
     ! vd
     ! vd   This subroutine is part of a set of subroutines that generate
     ! vd   Lebedev grids [1-6] for integration on a sphere. The original
     ! vd   C-code [1] was kindly provided by Dr. Dmitri N. Laikov and
     ! vd   translated into fortran by Dr. Christoph van Wuellen.
     ! vd   
     ! vd
     ! vd   Users of this code are asked to include reference [1] in their
     ! vd   publications, and in the user- and programmers-manuals
     ! vd   describing their codes.
     ! vd
     ! vd   This code was distributed through CCL (http://www.ccl.net/).
     ! vd
     ! vd   [1] V.I. Lebedev, and D.N. Laikov
     ! vd       "A quadrature formula for the sphere of the 131st
     ! vd        algebraic order of accuracy"
     ! vd       doklady Mathematics, Vol. 59, No. 3, 1999, pp. 477-481.
     ! vd
     ! vd   [2] V.I. Lebedev
     ! vd       "A quadrature formula for the sphere of 59th algebraic
     ! vd        order of accuracy"
     ! vd       Russian Acad. Sci. dokl. Math., Vol. 50, 1995, pp. 283-286.
     ! vd
     ! vd   [3] V.I. Lebedev, and A.L. Skorokhodov
     ! vd       "Quadrature formulas of orders 41, 47, and 53 for the sphere"
     ! vd       Russian Acad. Sci. dokl. Math., Vol. 45, 1992, pp. 587-592.
     ! vd
     ! vd   [4] V.I. Lebedev
     ! vd       "Spherical quadrature formulas exact to orders 25-29"
     ! vd       Siberian Mathematical Journal, Vol. 18, 1977, pp. 99-107.
     ! vd
     ! vd   [5] V.I. Lebedev
     ! vd       "Quadratures on a sphere"
     ! vd       Computational Mathematics and Mathematical Physics, Vol. 16,
     ! vd       1976, pp. 10-24.
     ! vd
     ! vd   [6] V.I. Lebedev
     ! vd       "Values of the nodes and weights of ninth to seventeenth
     ! vd        order Gauss-Markov quadrature formulae invariant under the
     ! vd        octahedron group with inversion"
     ! vd       Computational Mathematics and Mathematical Physics, Vol. 15,
     ! vd       1975, pp. 44-51.
     ! vd
     */
    N = 0;
    QUICKDouble a = 0;
    QUICKDouble b = 0;
    QUICKDouble v = 0.1666666666666667;
    N = gen_oh( 1, N, x, y, z, w, a, b, v);
    N = N-1;
}

__device__ void LD0014(QUICKDouble* x, QUICKDouble* y, QUICKDouble* z, QUICKDouble* w, int N)
{
    /*
     ! W
     ! W    LEBEDEV    14-POINT ANGULAR GRID
     ! W
     ! vd
     ! vd   This subroutine is part of a set of subroutines that generate
     ! vd   Lebedev grids [1-6] for integration on a sphere. The original
     ! vd   C-code [1] was kindly provided by Dr. Dmitri N. Laikov and
     ! vd   translated into fortran by Dr. Christoph van Wuellen.
     ! vd   
     ! vd
     ! vd   Users of this code are asked to include reference [1] in their
     ! vd   publications, and in the user- and programmers-manuals
     ! vd   describing their codes.
     ! vd
     ! vd   This code was distributed through CCL (http://www.ccl.net/).
     ! vd
     ! vd   [1] V.I. Lebedev, and D.N. Laikov
     ! vd       "A quadrature formula for the sphere of the 131st
     ! vd        algebraic order of accuracy"
     ! vd       doklady Mathematics, Vol. 59, No. 3, 1999, pp. 477-481.
     ! vd
     ! vd   [2] V.I. Lebedev
     ! vd       "A quadrature formula for the sphere of 59th algebraic
     ! vd        order of accuracy"
     ! vd       Russian Acad. Sci. dokl. Math., Vol. 50, 1995, pp. 283-286.
     ! vd
     ! vd   [3] V.I. Lebedev, and A.L. Skorokhodov
     ! vd       "Quadrature formulas of orders 41, 47, and 53 for the sphere"
     ! vd       Russian Acad. Sci. dokl. Math., Vol. 45, 1992, pp. 587-592.
     ! vd
     ! vd   [4] V.I. Lebedev
     ! vd       "Spherical quadrature formulas exact to orders 25-29"
     ! vd       Siberian Mathematical Journal, Vol. 18, 1977, pp. 99-107.
     ! vd
     ! vd   [5] V.I. Lebedev
     ! vd       "Quadratures on a sphere"
     ! vd       Computational Mathematics and Mathematical Physics, Vol. 16,
     ! vd       1976, pp. 10-24.
     ! vd
     ! vd   [6] V.I. Lebedev
     ! vd       "Values of the nodes and weights of ninth to seventeenth
     ! vd        order Gauss-Markov quadrature formulae invariant under the
     ! vd        octahedron group with inversion"
     ! vd       Computational Mathematics and Mathematical Physics, Vol. 15,
     ! vd       1975, pp. 44-51.
     ! vd
     */
    N = 0;
    QUICKDouble a = 0;
    QUICKDouble b = 0;
    QUICKDouble v = 0.6666666666666667e-1;
    N = gen_oh( 1, N, x, y, z, w, a, b, v);
    v = 0.7500000000000000e-1;
    N = gen_oh( 3, N, x, y, z, w, a, b, v);
    N = N-1;
}

__device__ void LD0026(QUICKDouble* x, QUICKDouble* y, QUICKDouble* z, QUICKDouble* w, int N)
{
    /*
     ! W
     ! W    LEBEDEV    26-POINT ANGULAR GRID
     ! W
     ! vd
     ! vd   This subroutine is part of a set of subroutines that generate
     ! vd   Lebedev grids [1-6] for integration on a sphere. The original
     ! vd   C-code [1] was kindly provided by Dr. Dmitri N. Laikov and
     ! vd   translated into fortran by Dr. Christoph van Wuellen.
     ! vd   
     ! vd
     ! vd   Users of this code are asked to include reference [1] in their
     ! vd   publications, and in the user- and programmers-manuals
     ! vd   describing their codes.
     ! vd
     ! vd   This code was distributed through CCL (http://www.ccl.net/).
     ! vd
     ! vd   [1] V.I. Lebedev, and D.N. Laikov
     ! vd       "A quadrature formula for the sphere of the 131st
     ! vd        algebraic order of accuracy"
     ! vd       doklady Mathematics, Vol. 59, No. 3, 1999, pp. 477-481.
     ! vd
     ! vd   [2] V.I. Lebedev
     ! vd       "A quadrature formula for the sphere of 59th algebraic
     ! vd        order of accuracy"
     ! vd       Russian Acad. Sci. dokl. Math., Vol. 50, 1995, pp. 283-286.
     ! vd
     ! vd   [3] V.I. Lebedev, and A.L. Skorokhodov
     ! vd       "Quadrature formulas of orders 41, 47, and 53 for the sphere"
     ! vd       Russian Acad. Sci. dokl. Math., Vol. 45, 1992, pp. 587-592.
     ! vd
     ! vd   [4] V.I. Lebedev
     ! vd       "Spherical quadrature formulas exact to orders 25-29"
     ! vd       Siberian Mathematical Journal, Vol. 18, 1977, pp. 99-107.
     ! vd
     ! vd   [5] V.I. Lebedev
     ! vd       "Quadratures on a sphere"
     ! vd       Computational Mathematics and Mathematical Physics, Vol. 16,
     ! vd       1976, pp. 10-24.
     ! vd
     ! vd   [6] V.I. Lebedev
     ! vd       "Values of the nodes and weights of ninth to seventeenth
     ! vd        order Gauss-Markov quadrature formulae invariant under the
     ! vd        octahedron group with inversion"
     ! vd       Computational Mathematics and Mathematical Physics, Vol. 15,
     ! vd       1975, pp. 44-51.
     ! vd
     */
    N = 0;
    QUICKDouble a = 0;
    QUICKDouble b = 0;
    QUICKDouble v = 0.4761904761904762e-1;
    N = gen_oh( 1, N, x, y, z, w, a, b, v);
    v = 0.3809523809523810e-1;
    N = gen_oh( 2, N, x, y, z, w, a, b, v);
    v=0.3214285714285714e-1;
    N = gen_oh( 3, N, x, y, z, w, a, b, v);
}

__device__ void LD0038(QUICKDouble* x, QUICKDouble* y, QUICKDouble* z, QUICKDouble* w, int N)
{
    /*
     ! W
     ! W    LEBEDEV    38-POINT ANGULAR GRID
     ! W
     ! vd
     ! vd   This subroutine is part of a set of subroutines that generate
     ! vd   Lebedev grids [1-6] for integration on a sphere. The original
     ! vd   C-code [1] was kindly provided by Dr. Dmitri N. Laikov and
     ! vd   translated into fortran by Dr. Christoph van Wuellen.
     ! vd   
     ! vd
     ! vd   Users of this code are asked to include reference [1] in their
     ! vd   publications, and in the user- and programmers-manuals
     ! vd   describing their codes.
     ! vd
     ! vd   This code was distributed through CCL (http://www.ccl.net/).
     ! vd
     ! vd   [1] V.I. Lebedev, and D.N. Laikov
     ! vd       "A quadrature formula for the sphere of the 131st
     ! vd        algebraic order of accuracy"
     ! vd       doklady Mathematics, Vol. 59, No. 3, 1999, pp. 477-481.
     ! vd
     ! vd   [2] V.I. Lebedev
     ! vd       "A quadrature formula for the sphere of 59th algebraic
     ! vd        order of accuracy"
     ! vd       Russian Acad. Sci. dokl. Math., Vol. 50, 1995, pp. 283-286.
     ! vd
     ! vd   [3] V.I. Lebedev, and A.L. Skorokhodov
     ! vd       "Quadrature formulas of orders 41, 47, and 53 for the sphere"
     ! vd       Russian Acad. Sci. dokl. Math., Vol. 45, 1992, pp. 587-592.
     ! vd
     ! vd   [4] V.I. Lebedev
     ! vd       "Spherical quadrature formulas exact to orders 25-29"
     ! vd       Siberian Mathematical Journal, Vol. 18, 1977, pp. 99-107.
     ! vd
     ! vd   [5] V.I. Lebedev
     ! vd       "Quadratures on a sphere"
     ! vd       Computational Mathematics and Mathematical Physics, Vol. 16,
     ! vd       1976, pp. 10-24.
     ! vd
     ! vd   [6] V.I. Lebedev
     ! vd       "Values of the nodes and weights of ninth to seventeenth
     ! vd        order Gauss-Markov quadrature formulae invariant under the
     ! vd        octahedron group with inversion"
     ! vd       Computational Mathematics and Mathematical Physics, Vol. 15,
     ! vd       1975, pp. 44-51.
     ! vd
     */
    N = 0;
    QUICKDouble a = 0;
    QUICKDouble b = 0;
    QUICKDouble v = 0.9523809523809524e-2;
    N = gen_oh( 1, N, x, y, z, w, a, b, v);
    v = 0.3214285714285714e-1;
    N = gen_oh( 3, N, x, y, z, w, a, b, v);
    a =0.4597008433809831e+0;
    v =0.2857142857142857e-1;
    N = gen_oh( 5, N, x, y, z, w, a, b, v);
}

__device__ void LD0050(QUICKDouble* x, QUICKDouble* y, QUICKDouble* z, QUICKDouble* w, int N)
{
    /*
     ! W
     ! W    LEBEDEV    50-POINT ANGULAR GRID
     ! W
     ! vd
     ! vd   This subroutine is part of a set of subroutines that generate
     ! vd   Lebedev grids [1-6] for integration on a sphere. The original
     ! vd   C-code [1] was kindly provided by Dr. Dmitri N. Laikov and
     ! vd   translated into fortran by Dr. Christoph van Wuellen.
     ! vd   
     ! vd
     ! vd   Users of this code are asked to include reference [1] in their
     ! vd   publications, and in the user- and programmers-manuals
     ! vd   describing their codes.
     ! vd
     ! vd   This code was distributed through CCL (http://www.ccl.net/).
     ! vd
     ! vd   [1] V.I. Lebedev, and D.N. Laikov
     ! vd       "A quadrature formula for the sphere of the 131st
     ! vd        algebraic order of accuracy"
     ! vd       doklady Mathematics, Vol. 59, No. 3, 1999, pp. 477-481.
     ! vd
     ! vd   [2] V.I. Lebedev
     ! vd       "A quadrature formula for the sphere of 59th algebraic
     ! vd        order of accuracy"
     ! vd       Russian Acad. Sci. dokl. Math., Vol. 50, 1995, pp. 283-286.
     ! vd
     ! vd   [3] V.I. Lebedev, and A.L. Skorokhodov
     ! vd       "Quadrature formulas of orders 41, 47, and 53 for the sphere"
     ! vd       Russian Acad. Sci. dokl. Math., Vol. 45, 1992, pp. 587-592.
     ! vd
     ! vd   [4] V.I. Lebedev
     ! vd       "Spherical quadrature formulas exact to orders 25-29"
     ! vd       Siberian Mathematical Journal, Vol. 18, 1977, pp. 99-107.
     ! vd
     ! vd   [5] V.I. Lebedev
     ! vd       "Quadratures on a sphere"
     ! vd       Computational Mathematics and Mathematical Physics, Vol. 16,
     ! vd       1976, pp. 10-24.
     ! vd
     ! vd   [6] V.I. Lebedev
     ! vd       "Values of the nodes and weights of ninth to seventeenth
     ! vd        order Gauss-Markov quadrature formulae invariant under the
     ! vd        octahedron group with inversion"
     ! vd       Computational Mathematics and Mathematical Physics, Vol. 15,
     ! vd       1975, pp. 44-51.
     ! vd
     */
    N = 0;
    QUICKDouble a = 0;
    QUICKDouble b = 0;
    QUICKDouble v = 0.1269841269841270e-1;
    N = gen_oh( 1, N, x, y, z, w, a, b, v);
    v = 0.2257495590828924e-1;
    N = gen_oh( 2, N, x, y, z, w, a, b, v);
    v = 0.2109375000000000e-1;
    N = gen_oh( 3, N, x, y, z, w, a, b, v);
    a = 0.3015113445777636e+0;
    v = 0.2017333553791887e-1;
    N = gen_oh( 4, N, x, y, z, w, a, b, v);
}

__device__ void LD0074(QUICKDouble* x, QUICKDouble* y, QUICKDouble* z, QUICKDouble* w, int N)
{
    /*
     ! W
     ! W    LEBEDEV    74-POINT ANGULAR GRID
     ! W
     ! vd
     ! vd   This subroutine is part of a set of subroutines that generate
     ! vd   Lebedev grids [1-6] for integration on a sphere. The original
     ! vd   C-code [1] was kindly provided by Dr. Dmitri N. Laikov and
     ! vd   translated into fortran by Dr. Christoph van Wuellen.
     ! vd   
     ! vd
     ! vd   Users of this code are asked to include reference [1] in their
     ! vd   publications, and in the user- and programmers-manuals
     ! vd   describing their codes.
     ! vd
     ! vd   This code was distributed through CCL (http://www.ccl.net/).
     ! vd
     ! vd   [1] V.I. Lebedev, and D.N. Laikov
     ! vd       "A quadrature formula for the sphere of the 131st
     ! vd        algebraic order of accuracy"
     ! vd       doklady Mathematics, Vol. 59, No. 3, 1999, pp. 477-481.
     ! vd
     ! vd   [2] V.I. Lebedev
     ! vd       "A quadrature formula for the sphere of 59th algebraic
     ! vd        order of accuracy"
     ! vd       Russian Acad. Sci. dokl. Math., Vol. 50, 1995, pp. 283-286.
     ! vd
     ! vd   [3] V.I. Lebedev, and A.L. Skorokhodov
     ! vd       "Quadrature formulas of orders 41, 47, and 53 for the sphere"
     ! vd       Russian Acad. Sci. dokl. Math., Vol. 45, 1992, pp. 587-592.
     ! vd
     ! vd   [4] V.I. Lebedev
     ! vd       "Spherical quadrature formulas exact to orders 25-29"
     ! vd       Siberian Mathematical Journal, Vol. 18, 1977, pp. 99-107.
     ! vd
     ! vd   [5] V.I. Lebedev
     ! vd       "Quadratures on a sphere"
     ! vd       Computational Mathematics and Mathematical Physics, Vol. 16,
     ! vd       1976, pp. 10-24.
     ! vd
     ! vd   [6] V.I. Lebedev
     ! vd       "Values of the nodes and weights of ninth to seventeenth
     ! vd        order Gauss-Markov quadrature formulae invariant under the
     ! vd        octahedron group with inversion"
     ! vd       Computational Mathematics and Mathematical Physics, Vol. 15,
     ! vd       1975, pp. 44-51.
     ! vd
     */
    N = 0;
    QUICKDouble a = 0;
    QUICKDouble b = 0;
    QUICKDouble v = 0.5130671797338464e-3;
    N = gen_oh( 1, N, x, y, z, w, a, b, v);
    v = 0.1660406956574204e-1;
    N = gen_oh( 2, N, x, y, z, w, a, b, v);
    v = -0.2958603896103896e-1;
    N = gen_oh( 3, N, x, y, z, w, a, b, v);
    a = 0.4803844614152614;
    v = 0.2657620708215946e-1;
    N = gen_oh( 4, N, x, y, z, w, a, b, v);
    a = 0.3207726489807764;
    v = 0.1652217099371571e-1;
    N = gen_oh( 5, N, x, y, z, w, a, b, v);
}


__device__ void LD0086(QUICKDouble* x, QUICKDouble* y, QUICKDouble* z, QUICKDouble* w, int N)
{
    /*
     ! W
     ! W    LEBEDEV    86-POINT ANGULAR GRID
     ! W
     ! vd
     ! vd   This subroutine is part of a set of subroutines that generate
     ! vd   Lebedev grids [1-6] for integration on a sphere. The original
     ! vd   C-code [1] was kindly provided by Dr. Dmitri N. Laikov and
     ! vd   translated into fortran by Dr. Christoph van Wuellen.
     ! vd   
     ! vd
     ! vd   Users of this code are asked to include reference [1] in their
     ! vd   publications, and in the user- and programmers-manuals
     ! vd   describing their codes.
     ! vd
     ! vd   This code was distributed through CCL (http://www.ccl.net/).
     ! vd
     ! vd   [1] V.I. Lebedev, and D.N. Laikov
     ! vd       "A quadrature formula for the sphere of the 131st
     ! vd        algebraic order of accuracy"
     ! vd       doklady Mathematics, Vol. 59, No. 3, 1999, pp. 477-481.
     ! vd
     ! vd   [2] V.I. Lebedev
     ! vd       "A quadrature formula for the sphere of 59th algebraic
     ! vd        order of accuracy"
     ! vd       Russian Acad. Sci. dokl. Math., Vol. 50, 1995, pp. 283-286.
     ! vd
     ! vd   [3] V.I. Lebedev, and A.L. Skorokhodov
     ! vd       "Quadrature formulas of orders 41, 47, and 53 for the sphere"
     ! vd       Russian Acad. Sci. dokl. Math., Vol. 45, 1992, pp. 587-592.
     ! vd
     ! vd   [4] V.I. Lebedev
     ! vd       "Spherical quadrature formulas exact to orders 25-29"
     ! vd       Siberian Mathematical Journal, Vol. 18, 1977, pp. 99-107.
     ! vd
     ! vd   [5] V.I. Lebedev
     ! vd       "Quadratures on a sphere"
     ! vd       Computational Mathematics and Mathematical Physics, Vol. 16,
     ! vd       1976, pp. 10-24.
     ! vd
     ! vd   [6] V.I. Lebedev
     ! vd       "Values of the nodes and weights of ninth to seventeenth
     ! vd        order Gauss-Markov quadrature formulae invariant under the
     ! vd        octahedron group with inversion"
     ! vd       Computational Mathematics and Mathematical Physics, Vol. 15,
     ! vd       1975, pp. 44-51.
     ! vd
     */
    N = 0;
    QUICKDouble a = 0;
    QUICKDouble b = 0;
    QUICKDouble v = 0.1154401154401154e-1;
    N = gen_oh( 1, N, x, y, z, w, a, b, v);
    v = 0.1194390908585628e-1;
    N = gen_oh( 3, N, x, y, z, w, a, b, v);
    a = 0.3696028464541502;
    v = 0.1111055571060340e-1;
    N = gen_oh( 4, N, x, y, z, w, a, b, v);
    a = 0.6943540066026664;
    v = 0.1187650129453714e-1;
    N = gen_oh( 4, N, x, y, z, w, a, b, v);
    a = 0.3742430390903412;
    v = 0.1181230374690448e-1;
    N = gen_oh( 5, N, x, y, z, w, a, b, v);
}

__device__ void LD0110(QUICKDouble* x, QUICKDouble* y, QUICKDouble* z, QUICKDouble* w, int N)
{
    /*
     ! W
     ! W    LEBEDEV    110-POINT ANGULAR GRID
     ! W
     ! vd
     ! vd   This subroutine is part of a set of subroutines that generate
     ! vd   Lebedev grids [1-6] for integration on a sphere. The original
     ! vd   C-code [1] was kindly provided by Dr. Dmitri N. Laikov and
     ! vd   translated into fortran by Dr. Christoph van Wuellen.
     ! vd   
     ! vd
     ! vd   Users of this code are asked to include reference [1] in their
     ! vd   publications, and in the user- and programmers-manuals
     ! vd   describing their codes.
     ! vd
     ! vd   This code was distributed through CCL (http://www.ccl.net/).
     ! vd
     ! vd   [1] V.I. Lebedev, and D.N. Laikov
     ! vd       "A quadrature formula for the sphere of the 131st
     ! vd        algebraic order of accuracy"
     ! vd       doklady Mathematics, Vol. 59, No. 3, 1999, pp. 477-481.
     ! vd
     ! vd   [2] V.I. Lebedev
     ! vd       "A quadrature formula for the sphere of 59th algebraic
     ! vd        order of accuracy"
     ! vd       Russian Acad. Sci. dokl. Math., Vol. 50, 1995, pp. 283-286.
     ! vd
     ! vd   [3] V.I. Lebedev, and A.L. Skorokhodov
     ! vd       "Quadrature formulas of orders 41, 47, and 53 for the sphere"
     ! vd       Russian Acad. Sci. dokl. Math., Vol. 45, 1992, pp. 587-592.
     ! vd
     ! vd   [4] V.I. Lebedev
     ! vd       "Spherical quadrature formulas exact to orders 25-29"
     ! vd       Siberian Mathematical Journal, Vol. 18, 1977, pp. 99-107.
     ! vd
     ! vd   [5] V.I. Lebedev
     ! vd       "Quadratures on a sphere"
     ! vd       Computational Mathematics and Mathematical Physics, Vol. 16,
     ! vd       1976, pp. 10-24.
     ! vd
     ! vd   [6] V.I. Lebedev
     ! vd       "Values of the nodes and weights of ninth to seventeenth
     ! vd        order Gauss-Markov quadrature formulae invariant under the
     ! vd        octahedron group with inversion"
     ! vd       Computational Mathematics and Mathematical Physics, Vol. 15,
     ! vd       1975, pp. 44-51.
     ! vd
     */
    N = 0;
    QUICKDouble a = 0;
    QUICKDouble b = 0;
    QUICKDouble v = 0.3828270494937162e-2;
    N = gen_oh( 1, N, x, y, z, w, a, b, v);
    
    v = 0.9793737512487512e-2;
    N = gen_oh( 3, N, x, y, z, w, a, b, v);
    
    a = 0.1851156353447362;
    v = 0.8211737283191111e-2;
    N = gen_oh( 4, N, x, y, z, w, a, b, v);
    
    a = 0.6904210483822922;
    v = 0.9942814891178103e-2;
    N = gen_oh( 4, N, x, y, z, w, a, b, v);
    
    a = 0.3956894730559419;
    v = 0.9595471336070963e-2;
    N = gen_oh( 4, N, x, y, z, w, a, b, v);
    
    a = 0.4783690288121502;
    v = 0.9694996361663028e-2;
    N = gen_oh( 5, N, x, y, z, w, a, b, v);
}

__device__ void LD0146(QUICKDouble* x, QUICKDouble* y, QUICKDouble* z, QUICKDouble* w, int N)
{
    /*
     ! W
     ! W    LEBEDEV    146-POINT ANGULAR GRID
     ! W
     ! vd
     ! vd   This subroutine is part of a set of subroutines that generate
     ! vd   Lebedev grids [1-6] for integration on a sphere. The original
     ! vd   C-code [1] was kindly provided by Dr. Dmitri N. Laikov and
     ! vd   translated into fortran by Dr. Christoph van Wuellen.
     ! vd   
     ! vd
     ! vd   Users of this code are asked to include reference [1] in their
     ! vd   publications, and in the user- and programmers-manuals
     ! vd   describing their codes.
     ! vd
     ! vd   This code was distributed through CCL (http://www.ccl.net/).
     ! vd
     ! vd   [1] V.I. Lebedev, and D.N. Laikov
     ! vd       "A quadrature formula for the sphere of the 131st
     ! vd        algebraic order of accuracy"
     ! vd       doklady Mathematics, Vol. 59, No. 3, 1999, pp. 477-481.
     ! vd
     ! vd   [2] V.I. Lebedev
     ! vd       "A quadrature formula for the sphere of 59th algebraic
     ! vd        order of accuracy"
     ! vd       Russian Acad. Sci. dokl. Math., Vol. 50, 1995, pp. 283-286.
     ! vd
     ! vd   [3] V.I. Lebedev, and A.L. Skorokhodov
     ! vd       "Quadrature formulas of orders 41, 47, and 53 for the sphere"
     ! vd       Russian Acad. Sci. dokl. Math., Vol. 45, 1992, pp. 587-592.
     ! vd
     ! vd   [4] V.I. Lebedev
     ! vd       "Spherical quadrature formulas exact to orders 25-29"
     ! vd       Siberian Mathematical Journal, Vol. 18, 1977, pp. 99-107.
     ! vd
     ! vd   [5] V.I. Lebedev
     ! vd       "Quadratures on a sphere"
     ! vd       Computational Mathematics and Mathematical Physics, Vol. 16,
     ! vd       1976, pp. 10-24.
     ! vd
     ! vd   [6] V.I. Lebedev
     ! vd       "Values of the nodes and weights of ninth to seventeenth
     ! vd        order Gauss-Markov quadrature formulae invariant under the
     ! vd        octahedron group with inversion"
     ! vd       Computational Mathematics and Mathematical Physics, Vol. 15,
     ! vd       1975, pp. 44-51.
     ! vd
     */
    N = 0;
    QUICKDouble a = 0;
    QUICKDouble b = 0;
    QUICKDouble v=0.5996313688621381e-3;
    N = gen_oh( 1, N, x, y, z, w, a, b, v);
    
    v=0.7372999718620756e-2;
    N = gen_oh( 2, N, x, y, z, w, a, b, v);
    
    v=0.7210515360144488e-2;
    N = gen_oh( 3, N, x, y, z, w, a, b, v);
    
    a=0.6764410400114264e+0;
    v=0.7116355493117555e-2;
    N = gen_oh( 4, N, x, y, z, w, a, b, v);
    
    a=0.4174961227965453e+0;
    v=0.6753829486314477e-2;
    N = gen_oh( 4, N, x, y, z, w, a, b, v);
    
    a=0.1574676672039082e+0;
    v=0.7574394159054034e-2;
    N = gen_oh( 4, N, x, y, z, w, a, b, v);
    
    a=0.1403553811713183e+0;
    b=0.4493328323269557e+0;
    v=0.6991087353303262e-2;
    N = gen_oh( 6, N, x, y, z, w, a, b, v);
}

__device__ void LD0170(QUICKDouble* x, QUICKDouble* y, QUICKDouble* z, QUICKDouble* w, int N)
{
    /*
     ! W
     ! W    LEBEDEV    170-POINT ANGULAR GRID
     ! W
     ! vd
     ! vd   This subroutine is part of a set of subroutines that generate
     ! vd   Lebedev grids [1-6] for integration on a sphere. The original
     ! vd   C-code [1] was kindly provided by Dr. Dmitri N. Laikov and
     ! vd   translated into fortran by Dr. Christoph van Wuellen.
     ! vd   
     ! vd
     ! vd   Users of this code are asked to include reference [1] in their
     ! vd   publications, and in the user- and programmers-manuals
     ! vd   describing their codes.
     ! vd
     ! vd   This code was distributed through CCL (http://www.ccl.net/).
     ! vd
     ! vd   [1] V.I. Lebedev, and D.N. Laikov
     ! vd       "A quadrature formula for the sphere of the 131st
     ! vd        algebraic order of accuracy"
     ! vd       doklady Mathematics, Vol. 59, No. 3, 1999, pp. 477-481.
     ! vd
     ! vd   [2] V.I. Lebedev
     ! vd       "A quadrature formula for the sphere of 59th algebraic
     ! vd        order of accuracy"
     ! vd       Russian Acad. Sci. dokl. Math., Vol. 50, 1995, pp. 283-286.
     ! vd
     ! vd   [3] V.I. Lebedev, and A.L. Skorokhodov
     ! vd       "Quadrature formulas of orders 41, 47, and 53 for the sphere"
     ! vd       Russian Acad. Sci. dokl. Math., Vol. 45, 1992, pp. 587-592.
     ! vd
     ! vd   [4] V.I. Lebedev
     ! vd       "Spherical quadrature formulas exact to orders 25-29"
     ! vd       Siberian Mathematical Journal, Vol. 18, 1977, pp. 99-107.
     ! vd
     ! vd   [5] V.I. Lebedev
     ! vd       "Quadratures on a sphere"
     ! vd       Computational Mathematics and Mathematical Physics, Vol. 16,
     ! vd       1976, pp. 10-24.
     ! vd
     ! vd   [6] V.I. Lebedev
     ! vd       "Values of the nodes and weights of ninth to seventeenth
     ! vd        order Gauss-Markov quadrature formulae invariant under the
     ! vd        octahedron group with inversion"
     ! vd       Computational Mathematics and Mathematical Physics, Vol. 15,
     ! vd       1975, pp. 44-51.
     ! vd
     */
    N = 0;
    QUICKDouble a = 0;
    QUICKDouble b = 0;
    QUICKDouble v=0.5544842902037365e-2;
    N = gen_oh( 1, N, x, y, z, w, a, b, v);
    v=0.6071332770670752e-2;
    N = gen_oh( 2, N, x, y, z, w, a, b, v);
    v=0.6383674773515093e-2;
    N = gen_oh( 3, N, x, y, z, w, a, b, v);
    a=0.2551252621114134e+0;
    v=0.5183387587747790e-2;
    N = gen_oh( 4, N, x, y, z, w, a, b, v);
    a=0.6743601460362766e+0;
    v=0.6317929009813725e-2;
    N = gen_oh( 4, N, x, y, z, w, a, b, v);
    a=0.4318910696719410e+0;
    v=0.6201670006589077e-2;
    N = gen_oh( 4, N, x, y, z, w, a, b, v);
    a=0.2613931360335988e+0;
    v=0.5477143385137348e-2;
    N = gen_oh( 5, N, x, y, z, w, a, b, v);
    a=0.4990453161796037e+0;
    b=0.1446630744325115e+0;
    v=0.5968383987681156e-2;
    N = gen_oh( 6, N, x, y, z, w, a, b, v);
    
}

__device__ void LD0194(QUICKDouble* x, QUICKDouble* y, QUICKDouble* z, QUICKDouble* w, int N)
{
    /*
     ! W
     ! W    LEBEDEV    194-POINT ANGULAR GRID
     ! W
     ! vd
     ! vd   This subroutine is part of a set of subroutines that generate
     ! vd   Lebedev grids [1-6] for integration on a sphere. The original
     ! vd   C-code [1] was kindly provided by Dr. Dmitri N. Laikov and
     ! vd   translated into fortran by Dr. Christoph van Wuellen.
     ! vd   
     ! vd
     ! vd   Users of this code are asked to include reference [1] in their
     ! vd   publications, and in the user- and programmers-manuals
     ! vd   describing their codes.
     ! vd
     ! vd   This code was distributed through CCL (http://www.ccl.net/).
     ! vd
     ! vd   [1] V.I. Lebedev, and D.N. Laikov
     ! vd       "A quadrature formula for the sphere of the 131st
     ! vd        algebraic order of accuracy"
     ! vd       doklady Mathematics, Vol. 59, No. 3, 1999, pp. 477-481.
     ! vd
     ! vd   [2] V.I. Lebedev
     ! vd       "A quadrature formula for the sphere of 59th algebraic
     ! vd        order of accuracy"
     ! vd       Russian Acad. Sci. dokl. Math., Vol. 50, 1995, pp. 283-286.
     ! vd
     ! vd   [3] V.I. Lebedev, and A.L. Skorokhodov
     ! vd       "Quadrature formulas of orders 41, 47, and 53 for the sphere"
     ! vd       Russian Acad. Sci. dokl. Math., Vol. 45, 1992, pp. 587-592.
     ! vd
     ! vd   [4] V.I. Lebedev
     ! vd       "Spherical quadrature formulas exact to orders 25-29"
     ! vd       Siberian Mathematical Journal, Vol. 18, 1977, pp. 99-107.
     ! vd
     ! vd   [5] V.I. Lebedev
     ! vd       "Quadratures on a sphere"
     ! vd       Computational Mathematics and Mathematical Physics, Vol. 16,
     ! vd       1976, pp. 10-24.
     ! vd
     ! vd   [6] V.I. Lebedev
     ! vd       "Values of the nodes and weights of ninth to seventeenth
     ! vd        order Gauss-Markov quadrature formulae invariant under the
     ! vd        octahedron group with inversion"
     ! vd       Computational Mathematics and Mathematical Physics, Vol. 15,
     ! vd       1975, pp. 44-51.
     ! vd
     */
    N = 0;
    QUICKDouble a = 0;
    QUICKDouble b = 0;
    QUICKDouble v=0.1782340447244611e-2;
    N = gen_oh( 1, N, x, y, z, w, a, b, v);
    v=0.5716905949977102e-2;
    N = gen_oh( 2, N, x, y, z, w, a, b, v);
    v=0.5573383178848738e-2;
    N = gen_oh( 3, N, x, y, z, w, a, b, v);
    a=0.6712973442695226e+0;
    v=0.5608704082587997e-2;
    N = gen_oh( 4, N, x, y, z, w, a, b, v);
    a=0.2892465627575439e+0;
    v=0.5158237711805383e-2;
    N = gen_oh( 4, N, x, y, z, w, a, b, v);
    a=0.4446933178717437e+0;
    v=0.5518771467273614e-2;
    N = gen_oh( 4, N, x, y, z, w, a, b, v);
    a=0.1299335447650067e+0;
    v=0.4106777028169394e-2;
    N = gen_oh( 4, N, x, y, z, w, a, b, v);
    a=0.3457702197611283e+0;
    v=0.5051846064614808e-2;
    N = gen_oh( 5, N, x, y, z, w, a, b, v);
    a=0.1590417105383530e+0;
    b=0.8360360154824589e+0;
    v=0.5530248916233094e-2;
    N = gen_oh( 6, N, x, y, z, w, a, b, v);
}
