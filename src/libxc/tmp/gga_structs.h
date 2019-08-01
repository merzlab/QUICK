
typedef struct{ 
  double A, B, c, d; 
} gga_c_lyp_params; 

typedef struct{  
  double beta, gamma, BB;  
} gga_c_pbe_params;  

typedef struct{  
  double beta, alpha;  
} gga_c_zpbeint_params;  

typedef struct{  
  double beta, alpha, omega;  
} gga_c_zvpbeint_params;  

typedef struct{  
  double aa[5], bb[5];  
} gga_k_dk_params;  

typedef struct{  
  double aa, bb, cc;  
} gga_k_ol2_params;  

typedef struct{  
  double gamma, lambda;  
} gga_k_tflw_params;  

typedef struct{  
  double beta, gamma, omega;  
} gga_x_b86_params;  

typedef struct{ 
  double beta, gamma; 
} gga_x_b88_params; 

typedef struct{  
  const double *omega;  
} gga_xc_th1_params;  

typedef struct{  
  double *omega;  
} gga_xc_th3_params;  

typedef struct{  
  double beta0, beta1, beta2;  
} gga_x_ft97_params;  

typedef struct{  
  double omega;  
} gga_x_hjs_b88_v2_params;  

typedef struct{  
  double omega;  
  
  const double *a, *b; /* pointers to the a and b parameters */  
} gga_x_hjs_params;  

typedef struct{  
  int func_id;  
  xc_gga_enhancement_t enhancement_factor;  
} gga_x_ityh_params;  

typedef struct{  
  double gamma, delta;  
} gga_x_kt_params;  

typedef struct{  
  int    modified; /* shall we use a modified version */  
  double threshold; /* when to start using the analytic form */  
  double ip;        /* ionization potential of the species */  
  double qtot;      /* total charge in the region */  
  
  double aa;     /* the parameters of LB94 */  
  double gamm;  
  
  double alpha;  
  double beta;  
} xc_gga_x_lb_params;  

typedef struct{  
  double a;  
  double c1, c2, c3;  
} gga_x_mpbe_params;  

typedef struct{  
  const double (*CC)[4];  
} gga_x_n12_params;  

typedef struct{  
  double a, b, gamma;  
} gga_x_optx_params;  

typedef struct{  
  double kappa, mu;  
  
  /* parameter used in the Odashima & Capelle versions */  
  double lambda;  
} gga_x_pbe_params;  

typedef struct{  
  double kappa, alpha, muPBE, muGE;  
} gga_x_pbeint_params;  

typedef struct{  
  double aa, bb, cc;  
} gga_x_pw86_params;  

typedef struct{  
  double a, b, c, d, f, alpha, expo;  
} gga_x_pw91_params;  

typedef struct{  
  double rpbe_kappa, rpbe_mu;  
} gga_x_rpbe_params;  

typedef struct{  
  int func_id;  
  xc_gga_enhancement_t enhancement_factor;  
} gga_x_sfat_params;  

typedef struct{  
  double kappa, mu, a[6], b[6];  
} gga_x_sogga11_params;  

typedef struct{  
  double A, B, C, D, E;  
} gga_x_ssb_sw_params;  

typedef struct{  
  double mu;  
  double alpha;  
} gga_x_vmt84_params;  

typedef struct{  
  double mu;  
  double alpha;  
} gga_x_vmt_params;  

typedef struct{  
  double omega;  
} gga_x_wpbeh_params;  

typedef struct {
  double c_x[5], c_ss[5], c_ab[5];
} gga_xc_wb97_params;

