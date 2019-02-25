// Second CUDA program
// Ping-Che Chen


#include <stdio.h>

#define BLOCK_SIZE	16
#define MAX_CYCLE   1
#define MATRIX_SIZE 1002

__global__ void matMultCUDA(const float* a, int lda, const float* b, int ldb, float* c, int ldc, int n)
{
    __shared__ float matA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float matB[BLOCK_SIZE][BLOCK_SIZE];
    const int tidc = threadIdx.x;
    const int tidr = threadIdx.y;
    const int bidc = blockIdx.x*BLOCK_SIZE;
    const int bidr = blockIdx.y*BLOCK_SIZE;
    int i,j;
    float results = 0;
    float comp = 0;
    for (j=0; j<n; j+=BLOCK_SIZE) {
        matA[tidr][tidc] = a[(tidr+bidr)*lda+tidc+j];
        matB[tidr][tidc] = b[(tidr+j)*ldb+tidc+bidc];
        
        __syncthreads();
        
        for (i=0;i<BLOCK_SIZE;i++){
            float t;
            comp -= matA[tidr][i]*matB[i][tidc];
            t=results-comp;
            comp=(t-results)+comp;
            results=t;
        }
        
        __syncthreads();
    }        
    c[(tidr+bidr)*ldc+tidc+bidc]=results;
}



clock_t matmultCUDA(const float* a, int lda, const float* b, int ldb, float* c, int ldc, int n)
{
    float *ac,*bc,*cc;
    clock_t start,end;
    size_t pitch_a,pitch_b,pitch_c;
    int blocks=(n+BLOCK_SIZE-1)/(BLOCK_SIZE);
    int newn=((n+BLOCK_SIZE-1)/BLOCK_SIZE)*BLOCK_SIZE;

    start=clock();    
    cudaMallocPitch((void**)&ac,&pitch_a,sizeof(float)*newn,newn);
    cudaMallocPitch((void**)&bc,&pitch_b,sizeof(float)*newn,newn);
    cudaMallocPitch((void**)&cc,&pitch_c,sizeof(float)*newn,newn);
    
    cudaMemset(ac, 0, pitch_a * newn);
	cudaMemset(bc, 0, pitch_b * newn);
    
    cudaMemcpy2D(ac,pitch_a, a, sizeof(float)*n,sizeof(float)*n,n,cudaMemcpyHostToDevice);
    cudaMemcpy2D(bc,pitch_b, b, sizeof(float)*n,sizeof(float)*n,n,cudaMemcpyHostToDevice);
    
    dim3 grid(blocks,blocks);
    dim3 threads(BLOCK_SIZE,BLOCK_SIZE);

    
    for (int i=0; i<MAX_CYCLE; i++) {
        matMultCUDA<<<grid,threads>>>(ac,pitch_a/sizeof(float)
            ,bc,pitch_b/sizeof(float),cc,pitch_c/sizeof(float),n);
    }

    
    cudaMemcpy2D(c,sizeof(float)*n, cc, pitch_c,sizeof(float)*n,n,cudaMemcpyDeviceToHost);    
    cudaFree(ac);
    cudaFree(bc);
    cudaFree(cc);
    end = clock();
    
    return end-start;
    
}

// local mat mult on CPU
clock_t matmult(const float* a, int lda, const float* b, int ldb, float* c, int ldc, int n)
{
	int i, j, k;
    clock_t start,end;
    start=clock();
    for (int m=0; m<MAX_CYCLE; m++) {
	for(i = 0; i < n; i++) {
		for(j = 0; j < n; j++) {
			double t = 0;
			for(k = 0; k < n; k++) {
				t += a[i * lda + k] * b[k * ldb + j];
			}
			c[i * ldc + j] = t;
		}
	}
    }
    end=clock();
    return end-start;
}

// generate matrix randomly
void matgen(float* a, int lda, int n)
{
	int i, j;

	for(i = 0; i < n; i++) {
		for(j = 0; j < n; j++) {
			a[i * lda + j] =(float) rand() / RAND_MAX;
		}
	}
}

// compare matrix
void compare_mat(const float* a, int lda, const float* b, int ldb, int n)
{
	float max_err = 0;
	float average_err = 0;
	int i, j;

	for(i = 0; i < n; i++) {
		for(j = 0; j < n; j++) {
			if(b[i * ldb + j] != 0) {
				float err = fabs((a[i * lda + j] - b[i * ldb + j]) / b[i * ldb + j]);
				if(max_err < err) max_err = err;
				average_err += err;
			}
		}
	}

	printf("Max error: %g  Average error: %g\n", max_err, average_err / (n * n));
}


bool InitCUDA()
{
	int count;

	cudaGetDeviceCount(&count);
	if(count == 0) {
		fprintf(stderr, "There is no device.\n");
		return false;
	}

	int i;
	for(i = 0; i < count; i++) {
		cudaDeviceProp prop;
		if(cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
			if(prop.major >= 1) {
				break;
			}
		}
	}

	if(i == count) {
		fprintf(stderr, "There is no device supporting CUDA 1.x.\n");
		return false;
	}

	cudaSetDevice(i);

	return true;
}

void printMatrix(const float* mat, int h, int w)
{
    int i,j;
    for (i=0; i<h; i++) {
        for (j=0; j<w; j++) {
            printf("%4.2f ",mat[i*w+j]);
        }
        printf("\n");
    }
}

extern "C" int matcuda_()
{
	float *a, *b, *c, *d;
	int n = MATRIX_SIZE;

	if(!InitCUDA()) {
		return 0;
	}

//    for (n=10; n<1000; n+=50) {
	a = (float*) malloc(sizeof(float) * n * n);
	b = (float*) malloc(sizeof(float) * n * n);
	c = (float*) malloc(sizeof(float) * n * n);
	d = (float*) malloc(sizeof(float) * n * n);

	srand(12);

	matgen(a, n, n);
	matgen(b, n, n);
    
//    printf("A\n");
//    printMatrix(a,n,n);

//    printf("B\n");
//    printMatrix(b,n,n);
	clock_t time = matmultCUDA(a, n, b, n, c, n, n);
    
    
//    printf("GPU\n");
//    printMatrix(c,n,n);
    clock_t time2 = matmult(a, n, b, n, d, n, n);
//    printf("CPU\n");
//    printMatrix(d,n,n);
        
	compare_mat(c, n, d, n, n);
    
    printf("n= %d ", n);
    double sec = (double) time / CLOCKS_PER_SEC;
	printf("GPU Time: %.4lf (%.2lf GFLOPS) \n", sec, 2.0 * n * n * n * MAX_CYCLE/ (sec * 1E9 ));
    
	sec = (double) time2 / CLOCKS_PER_SEC;
    
	printf("CPU Time: %.4lf (%.2lf GFLOPS) ", sec, 2.0 * n * n * n * MAX_CYCLE/ (sec * 1E9 ));
	printf("Speedup: %.4lf\n", (double)time2/(double)time);
    
	free(a);
	free(b);
	free(c);
	free(d);

//    }
    	return 0;
}
