#if defined(HIP) || defined(HIP_MPIV)
  #define gpuError_t hipError_t
  #define gpuSuccess hipSuccess
  #define gpuMemcpyKind hipMemcpyKind
  #define gpuMemcpyHostToDevice hipMemcpyHostToDevice
  #define gpuGetErrorString_ hipGetErrorString
#elif defined(CUDA) || defined(CUDA_MPIV)
  #define gpuError_t cudaError_t
  #define gpuSuccess cudaSuccess
  #define gpuMemcpyKind cudaMemcpyKind
  #define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
  #define gpuGetErrorString_ cudaGetErrorString
#endif


/* Safe wrapper around gpuMalloc_
 *
 * ptr: pointer to allocated device memory
 * size: reqested allocation size in bytes
 * filename: NULL-terminated source filename where function call originated
 * line: line of source file where function call originated
 */
static void gpuMalloc_(void **ptr, size_t size, const char * const filename,
        int line)
{
    gpuError_t ret;

#if defined(HIP) || defined(HIP_MPIV)
    ret = hipMalloc(ptr, size);
#elif defined(CUDA) || defined(CUDA_MPIV)
    ret = cudaMalloc(ptr, size);
#endif

    if (ret != gpuSuccess) {
        const char *str = gpuGetErrorString_(ret);

        fprintf(stderr, "[ERROR] GPU error: gpuMalloc failure\n");
        fprintf(stderr, "  [INFO] At line %d in file %.*s\n",
                line, (int) strlen(filename), filename);
        fprintf(stderr, "  [INFO] Error code: %d\n", ret);
        fprintf(stderr, "  [INFO] Error message: %.*s\n", (int) strlen(str), str);

        exit(1);
    }  
}


/* Safe wrapper around gpuMemcpy_
 *
 * dest: address to be copied to
 * src: address to be copied from
 * count: num. bytes to copy
 * dir: GPU enum specifying address types for dest and src
 * filename: NULL-terminated source filename where function call originated
 * line: line of source file where function call originated
 */
static void gpuMemcpy_(void * const dest, void const * const src, size_t count,
        gpuMemcpyKind dir, const char * const filename, int line)
{
    gpuError_t ret;

#if defined(HIP) || defined(HIP_MPIV)
    ret = hipMemcpy(dest, src, count, dir);
#elif defined(CUDA) || defined(CUDA_MPIV)
    ret = cudaMemcpy(dest, src, count, dir);
#endif

    if (ret != gpuSuccess) {
        const char *str = gpuGetErrorString_(ret);

        fprintf(stderr, "[ERROR] GPU error: gpuMemcpy failure\n");
        fprintf(stderr, "  [INFO] At line %d in file %.*s\n",
                line, (int) strlen(filename), filename);
        fprintf(stderr, "  [INFO] Error code: %d\n", ret);
        fprintf(stderr, "  [INFO] Error message: %.*s\n", (int) strlen(str), str);

        exit(1);
    }
}


//Uploads parameters required for kernels. *p is a pointer to libxc functional, gpu_work_params is a pointer
//to host memory location containing worker paramters
void* gpu_upload_maple2c_params(const xc_func_type *p)
{
    void *d_maple_params;

#ifdef DEBUG
    printf("FILE: %s, LINE: %d, FUNCTION: %s, p->params_byte_size: %d \n",
            __FILE__, __LINE__, __func__, p->params_byte_size);
#endif

    gpuMalloc_((void **) &d_maple_params, p->params_byte_size, __FILE__, __LINE__);
    gpuMemcpy_(d_maple_params, p->params, p->params_byte_size, gpuMemcpyHostToDevice, __FILE__, __LINE__);

    return d_maple_params;
}


//Uploads paramters required for kernels. *p is a pointer to libxc functional, gpu_work_params is a pointer
//to host memory location containing worker paramters
void* gpu_upload_work_params(const xc_func_type *p, void* gpu_work_params)
{
    void *d_work_params;
    int work_param_size;

    //check the family
    switch (p->info->family) {
        case XC_FAMILY_LDA:
            work_param_size = sizeof(gpu_lda_work_params);
#ifdef DEBUG
            printf("FILE: %s, LINE: %d, FUNCTION: %s, lda_work_param_size: %d \n",
                    __FILE__, __LINE__, __func__, work_param_size);
#endif
            break;

        case XC_FAMILY_HYB_GGA:
        case XC_FAMILY_GGA:
            //Now check the kind.
            switch (p->info->kind) {
                case XC_EXCHANGE:
                    work_param_size = sizeof(gpu_ggax_work_params);
                    break;

                case XC_CORRELATION:
                case XC_EXCHANGE_CORRELATION:
                    work_param_size = sizeof(gpu_ggac_work_params);
                    break;

                default:
                    break;
            }
            break;

        default:
            break;
    }

#ifdef DEBUG
    printf("FILE: %s, LINE: %d, FUNCTION: %s, gga_work_param_size: %d \n",
            __FILE__, __LINE__, __func__, work_param_size);
#endif

    gpuMalloc_((void **) &d_work_params, work_param_size, __FILE__, __LINE__);
    gpuMemcpy_(d_work_params, gpu_work_params, work_param_size, gpuMemcpyHostToDevice, __FILE__, __LINE__);

    return d_work_params;
}


//returns a pointer to an empty device array
double* gpu_upload_libxc_out_array(int size)
{
    double *d_double_arr;
    int arr_size = size * sizeof(double);

    gpuMalloc_((void **) &d_double_arr, arr_size, __FILE__, __LINE__);

    return d_double_arr;
}


//returns a pointer to a populated device array
double* gpu_upload_libxc_input_array(const double *h_input, int size)
{
    double *d_double_arr;
    int arr_size = size * sizeof(double);

    gpuMalloc_((void **) &d_double_arr, arr_size, __FILE__, __LINE__);
    gpuMemcpy_(d_double_arr, h_input, arr_size, gpuMemcpyHostToDevice, __FILE__, __LINE__);

    return d_double_arr;
}


//Returns an integer that uniquly identifies the gpu worker
int get_gpu_worker(const xc_func_type *p)
{
    int gpu_wt = 0;

    //check the family
    switch (p->info->family) {
        case XC_FAMILY_LDA:
            gpu_wt = GPU_WORK_LDA;
            break;

        case XC_FAMILY_HYB_GGA:
        case XC_FAMILY_GGA:
            //Now check the kind.
            switch (p->info->kind) {
                case XC_EXCHANGE:
                    gpu_wt = GPU_WORK_GGA_X;
                    break;

                case XC_CORRELATION:
                case XC_EXCHANGE_CORRELATION:
                    gpu_wt = GPU_WORK_GGA_C;
                    break;

                default:
                    break;
            }
            break;

        default:
            break;
    }

    return gpu_wt;

}


gpu_libxc_info* gpu_upload_libxc_info(const xc_func_type *p, void *ggwp, double mix_coeff, int np)
{
    gpu_libxc_info h_glinfo;

    h_glinfo.func_id = p->info->number;
    h_glinfo.gpu_worker = get_gpu_worker(p);
    h_glinfo.mix_coeff = mix_coeff;

    h_glinfo.d_maple2c_params = gpu_upload_maple2c_params(p);
    h_glinfo.d_worker_params = gpu_upload_work_params(p, ggwp);

    //allocate device memory for some work params required by gga_x worker.
    h_glinfo.d_gdm = gpu_upload_libxc_out_array(np);
    h_glinfo.d_ds = gpu_upload_libxc_out_array(np);
    h_glinfo.d_rhoLDA = gpu_upload_libxc_out_array(np);

    gpu_libxc_info* d_glinfo;
    gpuMalloc_((void **) &d_glinfo, sizeof(gpu_libxc_info), __FILE__, __LINE__);
    gpuMemcpy_(d_glinfo, &h_glinfo, sizeof(gpu_libxc_info), gpuMemcpyHostToDevice, __FILE__, __LINE__);

    return d_glinfo;
}


//gpu_libxc_out* gpu_upload_libxc_out(int np)
//{
//    gpu_libxc_out h_glout;
//    h_glout.d_zk = gpu_upload_libxc_out_array(np);
//    h_glout.d_vrho = gpu_upload_libxc_out_array(np);
//    h_glout.d_vsigma = gpu_upload_libxc_out_array(np);
//
//
//    gpu_libxc_out* d_glout;
//    gpuMalloc_((void **) &d_glout, sizeof(gpu_libxc_out), __FILE__, __LINE__);
//    gpuMemcpy_(d_glout, &h_glout, sizeof(gpu_libxc_out), gpuMemcpyHostToDevice, __FILE__, __LINE__);
//
//    return d_glout;
//}
//
//
//gpu_libxc_in* gpu_upload_libxc_in(const double* h_rho, const double *h_sigma, int np)
//{
//    gpu_libxc_in h_glin;
//    h_glin.d_rho = gpu_upload_libxc_input_array(h_rho, np);
//    h_glin.d_sigma = gpu_upload_libxc_input_array(h_sigma, np);
//
//    gpu_libxc_in* d_glin;
//    gpuMalloc_((void **) &d_glin, sizeof(gpu_libxc_in), __FILE__, __LINE__);
//    gpuMemcpy_(d_glin, &h_glin, sizeof(gpu_libxc_in), gpuMemcpyHostToDevice, __FILE__, __LINE__);
//
//    return d_glin;
//}
