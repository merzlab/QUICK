#define  XC_LDA_X                         1  /*Exchange                                                              */
#define  XC_LDA_C_WIGNER                  2  /*Wigner parametrization                                                */
#define  XC_LDA_C_RPA                     3  /*Random Phase Approximation                                            */
#define  XC_LDA_C_HL                      4  /*Hedin & Lundqvist                                                     */
#define  XC_LDA_C_GL                      5  /* Gunnarson & Lundqvist                                                */
#define  XC_LDA_C_XALPHA                  6  /* Slater Xalpha                                                        */
#define  XC_LDA_C_VWN                     7  /*Vosko, Wilk, & Nusair (5)                                             */
#define  XC_LDA_C_VWN_RPA                 8  /*Vosko, Wilk, & Nusair (RPA)                                           */
#define  XC_LDA_C_PZ                      9  /*Perdew & Zunger                                                       */
#define  XC_LDA_C_PZ_MOD                 10  /* Perdew & Zunger (Modified)                                           */
#define  XC_LDA_C_OB_PZ                  11  /* Ortiz & Ballone (PZ)                                                 */
#define  XC_LDA_C_PW                     12  /*Perdew & Wang                                                         */
#define  XC_LDA_C_PW_MOD                 13  /* Perdew & Wang (Modified)                                             */
#define  XC_LDA_C_OB_PW                  14  /* Ortiz & Ballone (PW)                                                 */
#define  XC_LDA_C_2D_AMGB                15  /*Attaccalite et al                                                     */
#define  XC_LDA_C_2D_PRM                 16  /*Pittalis, Rasanen & Marques correlation in 2D                         */
#define  XC_LDA_C_vBH                    17  /* von Barth & Hedin                                                    */
#define  XC_LDA_C_1D_CSC                 18  /*Casula, Sorella, and Senatore 1D correlation                          */
#define  XC_LDA_X_2D                     19  /*Exchange in 2D                                                        */
#define  XC_LDA_XC_TETER93               20  /*Teter 93 parametrization                                              */
#define  XC_LDA_X_1D                     21  /*Exchange in 1D                                                        */
#define  XC_LDA_C_ML1                    22  /*Modified LSD (version 1) of Proynov and Salahub                       */
#define  XC_LDA_C_ML2                    23  /* Modified LSD (version 2) of Proynov and Salahub                      */
#define  XC_LDA_C_GOMBAS                 24  /*Gombas parametrization                                                */
#define  XC_LDA_C_PW_RPA                 25  /* Perdew & Wang fit of the RPA                                         */
#define  XC_LDA_C_1D_LOOS                26  /*P-F Loos correlation LDA                                              */
#define  XC_LDA_C_RC04                   27  /*Ragot-Cortona                                                         */
#define  XC_LDA_C_VWN_1                  28  /*Vosko, Wilk, & Nusair (1)                                             */
#define  XC_LDA_C_VWN_2                  29  /*Vosko, Wilk, & Nusair (2)                                             */
#define  XC_LDA_C_VWN_3                  30  /*Vosko, Wilk, & Nusair (3)                                             */
#define  XC_LDA_C_VWN_4                  31  /*Vosko, Wilk, & Nusair (4)                                             */
#define  XC_LDA_XC_ZLP                   43  /*Zhao, Levy & Parr, Eq. (20)                                           */
#define  XC_LDA_K_TF                     50  /*Thomas-Fermi kinetic energy functional                                */
#define  XC_LDA_K_LP                     51  /* Lee and Parr Gaussian ansatz                                         */
#define  XC_LDA_XC_KSDT                 259  /*Karasiev et al. parametrization                                       */
#define  XC_LDA_C_CHACHIYO              287  /*Chachiyo simple 2 parameter correlation                               */
#define  XC_LDA_C_LP96                  289  /*Liu-Parr correlation                                                  */
#define  XC_LDA_X_REL                   532  /*Relativistic exchange                                                 */
#define  XC_LDA_XC_1D_EHWLRG_1          536  /*LDA constructed from slab-like systems of 1 electron                  */
#define  XC_LDA_XC_1D_EHWLRG_2          537  /* LDA constructed from slab-like systems of 2 electrons                */
#define  XC_LDA_XC_1D_EHWLRG_3          538  /* LDA constructed from slab-like systems of 3 electrons                */
#define  XC_LDA_X_ERF                   546  /*Attenuated exchange LDA (erf)                                         */
#define  XC_LDA_XC_LP_A                 547  /* Lee-Parr reparametrization B                                         */
#define  XC_LDA_XC_LP_B                 548  /* Lee-Parr reparametrization B                                         */
#define  XC_LDA_X_RAE                   549  /* Rae self-energy corrected exchange                                   */
#define  XC_LDA_K_ZLP                   550  /*kinetic energy version of ZLP                                         */
#define  XC_LDA_C_MCWEENY               551  /* McWeeny 76                                                           */
#define  XC_LDA_C_BR78                  552  /* Brual & Rothstein 78                                                 */
#define  XC_LDA_C_PK09                  554  /*Proynov and Kong 2009                                                 */
#define  XC_LDA_C_OW_LYP                573  /* Wigner with corresponding LYP parameters                             */
#define  XC_LDA_C_OW                    574  /* Optimized Wigner                                                     */
#define  XC_LDA_XC_GDSMFB               577  /* Groth et al. parametrization                                         */
#define  XC_LDA_C_GK72                  578  /*Gordon and Kim 1972                                                   */
#define  XC_LDA_C_KARASIEV              579  /* Karasiev reparameterization of Chachiyo                              */
#define  XC_LDA_K_LP96                  580  /* Liu-Parr kinetic                                                     */
#define  XC_GGA_X_GAM                    32  /* GAM functional from Minnesota                                        */
#define  XC_GGA_C_GAM                    33  /* GAM functional from Minnesota                                        */
#define  XC_GGA_X_HCTH_A                 34  /*HCTH-A                                                                */
#define  XC_GGA_X_EV93                   35  /*Engel and Vosko                                                       */
#define  XC_GGA_X_BCGP                   38  /* Burke, Cancio, Gould, and Pittalis                                   */
#define  XC_GGA_C_BCGP                   39  /*Burke, Cancio, Gould, and Pittalis                                    */
#define  XC_GGA_X_LAMBDA_OC2_N           40  /* lambda_OC2(N) version of PBE                                         */
#define  XC_GGA_X_B86_R                  41  /* Revised Becke 86 Xalpha,beta,gamma (with mod. grad. correction)      */
#define  XC_GGA_X_LAMBDA_CH_N            44  /* lambda_CH(N) version of PBE                                          */
#define  XC_GGA_X_LAMBDA_LO_N            45  /* lambda_LO(N) version of PBE                                          */
#define  XC_GGA_X_HJS_B88_V2             46  /*HJS screened exchange corrected B88 version                           */
#define  XC_GGA_C_Q2D                    47  /*Chiodo et al                                                          */
#define  XC_GGA_X_Q2D                    48  /*Chiodo et al                                                          */
#define  XC_GGA_X_PBE_MOL                49  /* Del Campo, Gazquez, Trickey and Vela (PBE-like)                      */
#define  XC_GGA_K_TFVW                   52  /*Thomas-Fermi plus von Weiszaecker correction                          */
#define  XC_GGA_K_REVAPBEINT             53  /* interpolated version of REVAPBE                                      */
#define  XC_GGA_K_APBEINT                54  /* interpolated version of APBE                                         */
#define  XC_GGA_K_REVAPBE                55  /* revised APBE                                                         */
#define  XC_GGA_X_AK13                   56  /*Armiento & Kuemmel 2013                                               */
#define  XC_GGA_K_MEYER                  57  /*Meyer,  Wang, and Young                                               */
#define  XC_GGA_X_LV_RPW86               58  /*Berland and Hyldgaard                                                 */
#define  XC_GGA_X_PBE_TCA                59  /* PBE revised by Tognetti et al                                        */
#define  XC_GGA_X_PBEINT                 60  /*PBE for hybrid interfaces                                             */
#define  XC_GGA_C_ZPBEINT                61  /*spin-dependent gradient correction to PBEint                          */
#define  XC_GGA_C_PBEINT                 62  /* PBE for hybrid interfaces                                            */
#define  XC_GGA_C_ZPBESOL                63  /* spin-dependent gradient correction to PBEsol                         */
#define  XC_GGA_XC_OPBE_D                65  /* oPBE_D functional of Goerigk and Grimme                              */
#define  XC_GGA_XC_OPWLYP_D              66  /* oPWLYP-D functional of Goerigk and Grimme                            */
#define  XC_GGA_XC_OBLYP_D               67  /*oBLYP-D functional of Goerigk and Grimme                              */
#define  XC_GGA_X_VMT84_GE               68  /* VMT{8,4} with constraint satisfaction with mu = mu_GE                */
#define  XC_GGA_X_VMT84_PBE              69  /*VMT{8,4} with constraint satisfaction with mu = mu_PBE                */
#define  XC_GGA_X_VMT_GE                 70  /* Vela, Medel, and Trickey with mu = mu_GE                             */
#define  XC_GGA_X_VMT_PBE                71  /*Vela, Medel, and Trickey with mu = mu_PBE                             */
#define  XC_GGA_C_N12_SX                 79  /* N12-SX functional from Minnesota                                     */
#define  XC_GGA_C_N12                    80  /*N12 functional from Minnesota                                         */
#define  XC_GGA_X_N12                    82  /*N12 functional from Minnesota                                         */
#define  XC_GGA_C_REGTPSS                83  /*Regularized TPSS correlation (ex-VPBE)                                */
#define  XC_GGA_C_OP_XALPHA              84  /*one-parameter progressive functional (XALPHA version)                 */
#define  XC_GGA_C_OP_G96                 85  /*one-parameter progressive functional (G96 version)                    */
#define  XC_GGA_C_OP_PBE                 86  /*one-parameter progressive functional (PBE version)                    */
#define  XC_GGA_C_OP_B88                 87  /*one-parameter progressive functional (B88 version)                    */
#define  XC_GGA_C_FT97                   88  /*Filatov & Thiel correlation                                           */
#define  XC_GGA_C_SPBE                   89  /* PBE correlation to be used with the SSB exchange                     */
#define  XC_GGA_X_SSB_SW                 90  /*Swart, Sola and Bickelhaupt correction to PBE                         */
#define  XC_GGA_X_SSB                    91  /* Swart, Sola and Bickelhaupt                                          */
#define  XC_GGA_X_SSB_D                  92  /* Swart, Sola and Bickelhaupt dispersion                               */
#define  XC_GGA_XC_HCTH_407P             93  /* HCTH/407+                                                            */
#define  XC_GGA_XC_HCTH_P76              94  /* HCTH p=7/6                                                           */
#define  XC_GGA_XC_HCTH_P14              95  /* HCTH p=1/4                                                           */
#define  XC_GGA_XC_B97_GGA1              96  /* Becke 97 GGA-1                                                       */
#define  XC_GGA_C_HCTH_A                 97  /*HCTH-A                                                                */
#define  XC_GGA_X_BPCCAC                 98  /*BPCCAC (GRAC for the energy)                                          */
#define  XC_GGA_C_REVTCA                 99  /*Tognetti, Cortona, Adamo (revised)                                    */
#define  XC_GGA_C_TCA                   100  /*Tognetti, Cortona, Adamo                                              */
#define  XC_GGA_X_PBE                   101  /*Perdew, Burke & Ernzerhof exchange                                    */
#define  XC_GGA_X_PBE_R                 102  /* Perdew, Burke & Ernzerhof exchange (revised)                         */
#define  XC_GGA_X_B86                   103  /*Becke 86 Xalpha,beta,gamma                                            */
#define  XC_GGA_X_HERMAN                104  /*Herman et al original GGA                                             */
#define  XC_GGA_X_B86_MGC               105  /* Becke 86 Xalpha,beta,gamma (with mod. grad. correction)              */
#define  XC_GGA_X_B88                   106  /*Becke 88                                                              */
#define  XC_GGA_X_G96                   107  /*Gill 96                                                               */
#define  XC_GGA_X_PW86                  108  /*Perdew & Wang 86                                                      */
#define  XC_GGA_X_PW91                  109  /*Perdew & Wang 91                                                      */
#define  XC_GGA_X_OPTX                  110  /*Handy & Cohen OPTX 01                                                 */
#define  XC_GGA_X_DK87_R1               111  /*dePristo & Kress 87 (version R1)                                      */
#define  XC_GGA_X_DK87_R2               112  /* dePristo & Kress 87 (version R2)                                     */
#define  XC_GGA_X_LG93                  113  /*Lacks & Gordon 93                                                     */
#define  XC_GGA_X_FT97_A                114  /*Filatov & Thiel 97 (version A)                                        */
#define  XC_GGA_X_FT97_B                115  /* Filatov & Thiel 97 (version B)                                       */
#define  XC_GGA_X_PBE_SOL               116  /* Perdew, Burke & Ernzerhof exchange (solids)                          */
#define  XC_GGA_X_RPBE                  117  /*Hammer, Hansen & Norskov (PBE-like)                                   */
#define  XC_GGA_X_WC                    118  /*Wu & Cohen                                                            */
#define  XC_GGA_X_MPW91                 119  /* Modified form of PW91 by Adamo & Barone                              */
#define  XC_GGA_X_AM05                  120  /*Armiento & Mattsson 05 exchange                                       */
#define  XC_GGA_X_PBEA                  121  /*Madsen (PBE-like)                                                     */
#define  XC_GGA_X_MPBE                  122  /*Adamo & Barone modification to PBE                                    */
#define  XC_GGA_X_XPBE                  123  /* xPBE reparametrization by Xu & Goddard                               */
#define  XC_GGA_X_2D_B86_MGC            124  /*Becke 86 MGC for 2D systems                                           */
#define  XC_GGA_X_BAYESIAN              125  /*Bayesian best fit for the enhancement factor                          */
#define  XC_GGA_X_PBE_JSJR              126  /* JSJR reparametrization by Pedroza, Silva & Capelle                   */
#define  XC_GGA_X_2D_B88                127  /*Becke 88 in 2D                                                        */
#define  XC_GGA_X_2D_B86                128  /*Becke 86 Xalpha,beta,gamma                                            */
#define  XC_GGA_X_2D_PBE                129  /*Perdew, Burke & Ernzerhof exchange in 2D                              */
#define  XC_GGA_C_PBE                   130  /*Perdew, Burke & Ernzerhof correlation                                 */
#define  XC_GGA_C_LYP                   131  /*Lee, Yang & Parr                                                      */
#define  XC_GGA_C_P86                   132  /*Perdew 86                                                             */
#define  XC_GGA_C_PBE_SOL               133  /* Perdew, Burke & Ernzerhof correlation SOL                            */
#define  XC_GGA_C_PW91                  134  /*Perdew & Wang 91                                                      */
#define  XC_GGA_C_AM05                  135  /*Armiento & Mattsson 05 correlation                                    */
#define  XC_GGA_C_XPBE                  136  /* xPBE reparametrization by Xu & Goddard                               */
#define  XC_GGA_C_LM                    137  /*Langreth and Mehl correlation                                         */
#define  XC_GGA_C_PBE_JRGX              138  /* JRGX reparametrization by Pedroza, Silva & Capelle                   */
#define  XC_GGA_X_OPTB88_VDW            139  /* Becke 88 reoptimized to be used with vdW functional of Dion et al    */
#define  XC_GGA_X_PBEK1_VDW             140  /* PBE reparametrization for vdW                                        */
#define  XC_GGA_X_OPTPBE_VDW            141  /* PBE reparametrization for vdW                                        */
#define  XC_GGA_X_RGE2                  142  /*Regularized PBE                                                       */
#define  XC_GGA_C_RGE2                  143  /* Regularized PBE                                                      */
#define  XC_GGA_X_RPW86                 144  /* refitted Perdew & Wang 86                                            */
#define  XC_GGA_X_KT1                   145  /*Exchange part of Keal and Tozer version 1                             */
#define  XC_GGA_XC_KT2                  146  /* Keal and Tozer version 2                                             */
#define  XC_GGA_C_WL                    147  /*Wilson & Levy                                                         */
#define  XC_GGA_C_WI                    148  /* Wilson & Ivanov                                                      */
#define  XC_GGA_X_MB88                  149  /* Modified Becke 88 for proton transfer                                */
#define  XC_GGA_X_SOGGA                 150  /* Second-order generalized gradient approximation                      */
#define  XC_GGA_X_SOGGA11               151  /*Second-order generalized gradient approximation 2011                  */
#define  XC_GGA_C_SOGGA11               152  /*Second-order generalized gradient approximation 2011                  */
#define  XC_GGA_C_WI0                   153  /*Wilson & Ivanov initial version                                       */
#define  XC_GGA_XC_TH1                  154  /* Tozer and Handy v. 1                                                 */
#define  XC_GGA_XC_TH2                  155  /*Tozer and Handy v. 2                                                  */
#define  XC_GGA_XC_TH3                  156  /*Tozer and Handy v. 3                                                  */
#define  XC_GGA_XC_TH4                  157  /* Tozer and Handy v. 4                                                 */
#define  XC_GGA_X_C09X                  158  /*C09x to be used with the VdW of Rutgers-Chalmers                      */
#define  XC_GGA_C_SOGGA11_X             159  /* To be used with HYB_GGA_X_SOGGA11_X                                  */
#define  XC_GGA_X_LB                    160  /*van Leeuwen & Baerends                                                */
#define  XC_GGA_XC_HCTH_93              161  /* HCTH functional fitted to  93 molecules                              */
#define  XC_GGA_XC_HCTH_120             162  /* HCTH functional fitted to 120 molecules                              */
#define  XC_GGA_XC_HCTH_147             163  /* HCTH functional fitted to 147 molecules                              */
#define  XC_GGA_XC_HCTH_407             164  /* HCTH functional fitted to 407 molecules                              */
#define  XC_GGA_XC_EDF1                 165  /*Empirical functionals from Adamson, Gill, and Pople                   */
#define  XC_GGA_XC_XLYP                 166  /*XLYP functional                                                       */
#define  XC_GGA_XC_KT1                  167  /* Keal and Tozer version 1                                             */
#define  XC_GGA_XC_B97_D                170  /*Grimme functional to be used with C6 vdW term                         */
#define  XC_GGA_XC_PBE1W                173  /* Functionals fitted for water                                         */
#define  XC_GGA_XC_MPWLYP1W             174  /* Functionals fitted for water                                         */
#define  XC_GGA_XC_PBELYP1W             175  /* Functionals fitted for water                                         */
#define  XC_GGA_X_LBM                   182  /* van Leeuwen & Baerends modified                                      */
#define  XC_GGA_X_OL2                   183  /*Exchange form based on Ou-Yang and Levy v.2                           */
#define  XC_GGA_X_APBE                  184  /* mu fixed from the semiclassical neutral atom                         */
#define  XC_GGA_K_APBE                  185  /* mu fixed from the semiclassical neutral atom                         */
#define  XC_GGA_C_APBE                  186  /* mu fixed from the semiclassical neutral atom                         */
#define  XC_GGA_K_TW1                   187  /* Tran and Wesolowski set 1 (Table II)                                 */
#define  XC_GGA_K_TW2                   188  /* Tran and Wesolowski set 2 (Table II)                                 */
#define  XC_GGA_K_TW3                   189  /* Tran and Wesolowski set 3 (Table II)                                 */
#define  XC_GGA_K_TW4                   190  /* Tran and Wesolowski set 4 (Table II)                                 */
#define  XC_GGA_X_HTBS                  191  /*Haas, Tran, Blaha, and Schwarz                                        */
#define  XC_GGA_X_AIRY                  192  /*Constantin et al based on the Airy gas                                */
#define  XC_GGA_X_LAG                   193  /*Local Airy Gas                                                        */
#define  XC_GGA_XC_MOHLYP               194  /* Functional for organometallic chemistry                              */
#define  XC_GGA_XC_MOHLYP2              195  /* Functional for barrier heights                                       */
#define  XC_GGA_XC_TH_FL                196  /*Tozer and Handy v. FL                                                 */
#define  XC_GGA_XC_TH_FC                197  /* Tozer and Handy v. FC                                                */
#define  XC_GGA_XC_TH_FCFO              198  /* Tozer and Handy v. FCFO                                              */
#define  XC_GGA_XC_TH_FCO               199  /* Tozer and Handy v. FCO                                               */
#define  XC_GGA_C_OPTC                  200  /*Optimized correlation functional of Cohen and Handy                   */
#define  XC_GGA_C_PBELOC                246  /*Semilocal dynamical correlation                                       */
#define  XC_GGA_XC_VV10                 255  /*Vydrov and Van Voorhis                                                */
#define  XC_GGA_C_PBEFE                 258  /* PBE for formation energies                                           */
#define  XC_GGA_C_OP_PW91               262  /*one-parameter progressive functional (PW91 version)                   */
#define  XC_GGA_X_PBEFE                 265  /* PBE for formation energies                                           */
#define  XC_GGA_X_CAP                   270  /*Correct Asymptotic Potential                                          */
#define  XC_GGA_X_EB88                  271  /* Non-empirical (excogitated) B88 functional of Becke and Elliott      */
#define  XC_GGA_C_PBE_MOL               272  /* Del Campo, Gazquez, Trickey and Vela (PBE-like)                      */
#define  XC_GGA_K_ABSP3                 277  /* gamma-TFvW form by Acharya et al [g = 1 - 1.513/N^0.35]              */
#define  XC_GGA_K_ABSP4                 278  /* gamma-TFvW form by Acharya et al [g = l = 1/(1 + 1.332/N^(1/3))]     */
#define  XC_GGA_C_BMK                   280  /* Boese-Martin for kinetics                                            */
#define  XC_GGA_C_TAU_HCTH              281  /* correlation part of tau-hcth                                         */
#define  XC_GGA_C_HYB_TAU_HCTH          283  /* correlation part of hyb_tau-hcth                                     */
#define  XC_GGA_X_BEEFVDW               285  /*BEEF-vdW exchange                                                     */
#define  XC_GGA_XC_BEEFVDW              286  /* BEEF-vdW exchange-correlation                                        */
#define  XC_GGA_X_PBETRANS              291  /*Gradient-based interpolation between PBE and revPBE                   */
#define  XC_GGA_X_CHACHIYO              298  /*Chachiyo exchange                                                     */
#define  XC_GGA_K_VW                    500  /* von Weiszaecker functional                                           */
#define  XC_GGA_K_GE2                   501  /* Second-order gradient expansion (l = 1/9)                            */
#define  XC_GGA_K_GOLDEN                502  /* TF-lambda-vW form by Golden (l = 13/45)                              */
#define  XC_GGA_K_YT65                  503  /* TF-lambda-vW form by Yonei and Tomishima (l = 1/5)                   */
#define  XC_GGA_K_BALTIN                504  /* TF-lambda-vW form by Baltin (l = 5/9)                                */
#define  XC_GGA_K_LIEB                  505  /* TF-lambda-vW form by Lieb (l = 0.185909191)                          */
#define  XC_GGA_K_ABSP1                 506  /* gamma-TFvW form by Acharya et al [g = 1 - 1.412/N^(1/3)]             */
#define  XC_GGA_K_ABSP2                 507  /* gamma-TFvW form by Acharya et al [g = 1 - 1.332/N^(1/3)]             */
#define  XC_GGA_K_GR                    508  /* gamma-TFvW form by Gazquez and Robles                                */
#define  XC_GGA_K_LUDENA                509  /* gamma-TFvW form by Ludena                                            */
#define  XC_GGA_K_GP85                  510  /* gamma-TFvW form by Ghosh and Parr                                    */
#define  XC_GGA_K_PEARSON               511  /*Pearson                                                               */
#define  XC_GGA_K_OL1                   512  /*Ou-Yang and Levy v.1                                                  */
#define  XC_GGA_K_OL2                   513  /* Ou-Yang and Levy v.2                                                 */
#define  XC_GGA_K_FR_B88                514  /* Fuentealba & Reyes (B88 version)                                     */
#define  XC_GGA_K_FR_PW86               515  /* Fuentealba & Reyes (PW86 version)                                    */
#define  XC_GGA_K_DK                    516  /*DePristo and Kress                                                    */
#define  XC_GGA_K_PERDEW                517  /* Perdew                                                               */
#define  XC_GGA_K_VSK                   518  /* Vitos, Skriver, and Kollar                                           */
#define  XC_GGA_K_VJKS                  519  /* Vitos, Johansson, Kollar, and Skriver                                */
#define  XC_GGA_K_ERNZERHOF             520  /* Ernzerhof                                                            */
#define  XC_GGA_K_LC94                  521  /* Lembarki & Chermette                                                 */
#define  XC_GGA_K_LLP                   522  /* Lee, Lee & Parr                                                      */
#define  XC_GGA_K_THAKKAR               523  /*Thakkar 1992                                                          */
#define  XC_GGA_X_WPBEH                 524  /*short-range version of the PBE                                        */
#define  XC_GGA_X_HJS_PBE               525  /*HJS screened exchange PBE version                                     */
#define  XC_GGA_X_HJS_PBE_SOL           526  /* HJS screened exchange PBE_SOL version                                */
#define  XC_GGA_X_HJS_B88               527  /* HJS screened exchange B88 version                                    */
#define  XC_GGA_X_HJS_B97X              528  /* HJS screened exchange B97x version                                   */
#define  XC_GGA_X_ITYH                  529  /*short-range recipe for exchange GGA functionals                       */
#define  XC_GGA_X_SFAT                  530  /*short-range recipe for exchange GGA functionals                       */
#define  XC_GGA_X_SG4                   533  /*Semiclassical GGA at fourth order                                     */
#define  XC_GGA_C_SG4                   534  /*Semiclassical GGA at fourth order                                     */
#define  XC_GGA_X_GG99                  535  /*Gilbert and Gill 1999                                                 */
#define  XC_GGA_X_PBEpow                539  /*PBE power                                                             */
#define  XC_GGA_X_KGG99                 544  /* Gilbert and Gill 1999 (mixed)                                        */
#define  XC_GGA_XC_HLE16                545  /* high local exchange 2016                                             */
#define  XC_GGA_C_SCAN_E0               553  /*GGA component of SCAN                                                 */
#define  XC_GGA_C_GAPC                  555  /*GapC                                                                  */
#define  XC_GGA_C_GAPLOC                556  /*Gaploc                                                                */
#define  XC_GGA_C_ZVPBEINT              557  /*another spin-dependent correction to PBEint                           */
#define  XC_GGA_C_ZVPBESOL              558  /* another spin-dependent correction to PBEsol                          */
#define  XC_GGA_C_TM_LYP                559  /* Takkar and McCarthy reparametrization                                */
#define  XC_GGA_C_TM_PBE                560  /* Thakkar and McCarthy reparametrization                               */
#define  XC_GGA_C_W94                   561  /*Wilson 94 (Eq. 25)                                                    */
#define  XC_GGA_C_CS1                   565  /*A dynamical correlation functional                                    */
#define  XC_GGA_X_B88M                  570  /* Becke 88 reoptimized to be used with mgga_c_tau1                     */
#define  XC_GGA_K_PBE3                  595  /* Three parameter PBE-like expansion                                   */
#define  XC_GGA_K_PBE4                  596  /* Four  parameter PBE-like expansion                                   */
#define  XC_GGA_K_EXP4                  597  /*Intermediate form between PBE3 and PBE4                               */
#define  XC_HYB_GGA_X_N12_SX             81  /*N12-SX functional from Minnesota                                      */
#define  XC_HYB_GGA_XC_B97_1p           266  /* version of B97 by Cohen and Handy                                    */
#define  XC_HYB_GGA_XC_PBE_MOL0         273  /* PBEmol0                                                              */
#define  XC_HYB_GGA_XC_PBE_SOL0         274  /* PBEsol0                                                              */
#define  XC_HYB_GGA_XC_PBEB0            275  /* PBEbeta0                                                             */
#define  XC_HYB_GGA_XC_PBE_MOLB0        276  /* PBEmolbeta0                                                          */
#define  XC_HYB_GGA_XC_PBE50            290  /* PBE0 with 50% exx                                                    */
#define  XC_HYB_GGA_XC_B3PW91           401  /*The original (ACM) hybrid of Becke                                    */
#define  XC_HYB_GGA_XC_B3LYP            402  /* The (in)famous B3LYP                                                 */
#define  XC_HYB_GGA_XC_B3P86            403  /* Perdew 86 hybrid similar to B3PW91                                   */
#define  XC_HYB_GGA_XC_O3LYP            404  /*hybrid using the optx functional                                      */
#define  XC_HYB_GGA_XC_MPW1K            405  /* mixture of mPW91 and PW91 optimized for kinetics                     */
#define  XC_HYB_GGA_XC_PBEH             406  /*aka PBE0 or PBE1PBE                                                   */
#define  XC_HYB_GGA_XC_B97              407  /*Becke 97                                                              */
#define  XC_HYB_GGA_XC_B97_1            408  /* Becke 97-1                                                           */
#define  XC_HYB_GGA_XC_B97_2            410  /* Becke 97-2                                                           */
#define  XC_HYB_GGA_XC_X3LYP            411  /* hybrid by Xu and Goddard                                             */
#define  XC_HYB_GGA_XC_B1WC             412  /*Becke 1-parameter mixture of WC and PBE                               */
#define  XC_HYB_GGA_XC_B97_K            413  /* Boese-Martin for Kinetics                                            */
#define  XC_HYB_GGA_XC_B97_3            414  /* Becke 97-3                                                           */
#define  XC_HYB_GGA_XC_MPW3PW           415  /* mixture with the mPW functional                                      */
#define  XC_HYB_GGA_XC_B1LYP            416  /* Becke 1-parameter mixture of B88 and LYP                             */
#define  XC_HYB_GGA_XC_B1PW91           417  /* Becke 1-parameter mixture of B88 and PW91                            */
#define  XC_HYB_GGA_XC_MPW1PW           418  /* Becke 1-parameter mixture of mPW91 and PW91                          */
#define  XC_HYB_GGA_XC_MPW3LYP          419  /* mixture of mPW and LYP                                               */
#define  XC_HYB_GGA_XC_SB98_1a          420  /* Schmider-Becke 98 parameterization 1a                                */
#define  XC_HYB_GGA_XC_SB98_1b          421  /* Schmider-Becke 98 parameterization 1b                                */
#define  XC_HYB_GGA_XC_SB98_1c          422  /* Schmider-Becke 98 parameterization 1c                                */
#define  XC_HYB_GGA_XC_SB98_2a          423  /* Schmider-Becke 98 parameterization 2a                                */
#define  XC_HYB_GGA_XC_SB98_2b          424  /* Schmider-Becke 98 parameterization 2b                                */
#define  XC_HYB_GGA_XC_SB98_2c          425  /* Schmider-Becke 98 parameterization 2c                                */
#define  XC_HYB_GGA_X_SOGGA11_X         426  /*Hybrid based on SOGGA11 form                                          */
#define  XC_HYB_GGA_XC_HSE03            427  /*the 2003 version of the screened hybrid HSE                           */
#define  XC_HYB_GGA_XC_HSE06            428  /* the 2006 version of the screened hybrid HSE                          */
#define  XC_HYB_GGA_XC_HJS_PBE          429  /* HJS hybrid screened exchange PBE version                             */
#define  XC_HYB_GGA_XC_HJS_PBE_SOL      430  /* HJS hybrid screened exchange PBE_SOL version                         */
#define  XC_HYB_GGA_XC_HJS_B88          431  /* HJS hybrid screened exchange B88 version                             */
#define  XC_HYB_GGA_XC_HJS_B97X         432  /* HJS hybrid screened exchange B97x version                            */
#define  XC_HYB_GGA_XC_CAM_B3LYP        433  /*CAM version of B3LYP                                                  */
#define  XC_HYB_GGA_XC_TUNED_CAM_B3LYP  434  /* CAM version of B3LYP tuned for excitations                           */
#define  XC_HYB_GGA_XC_BHANDH           435  /* Becke half-and-half                                                  */
#define  XC_HYB_GGA_XC_BHANDHLYP        436  /* Becke half-and-half with B88 exchange                                */
#define  XC_HYB_GGA_XC_MB3LYP_RC04      437  /* B3LYP with RC04 LDA                                                  */
#define  XC_HYB_GGA_XC_MPWLYP1M         453  /* MPW with 1 par. for metals/LYP                                       */
#define  XC_HYB_GGA_XC_REVB3LYP         454  /* Revised B3LYP                                                        */
#define  XC_HYB_GGA_XC_CAMY_BLYP        455  /*BLYP with yukawa screening                                            */
#define  XC_HYB_GGA_XC_PBE0_13          456  /* PBE0-1/3                                                             */
#define  XC_HYB_GGA_XC_B3LYPs           459  /* B3LYP* functional                                                    */
#define  XC_HYB_GGA_XC_WB97             463  /*Chai and Head-Gordon                                                  */
#define  XC_HYB_GGA_XC_WB97X            464  /* Chai and Head-Gordon                                                 */
#define  XC_HYB_GGA_XC_LRC_WPBEH        465  /* Long-range corrected functional by Rorhdanz et al                    */
#define  XC_HYB_GGA_XC_WB97X_V          466  /* Mardirossian and Head-Gordon                                         */
#define  XC_HYB_GGA_XC_LCY_PBE          467  /*PBE with yukawa screening                                             */
#define  XC_HYB_GGA_XC_LCY_BLYP         468  /*BLYP with yukawa screening                                            */
#define  XC_HYB_GGA_XC_LC_VV10          469  /*Vydrov and Van Voorhis                                                */
#define  XC_HYB_GGA_XC_CAMY_B3LYP       470  /*B3LYP with Yukawa screening                                           */
#define  XC_HYB_GGA_XC_WB97X_D          471  /* Chai and Head-Gordon                                                 */
#define  XC_HYB_GGA_XC_HPBEINT          472  /* hPBEint                                                              */
#define  XC_HYB_GGA_XC_LRC_WPBE         473  /* Long-range corrected functional by Rorhdanz et al                    */
#define  XC_HYB_GGA_XC_B3LYP5           475  /* B3LYP with VWN functional 5 instead of RPA                           */
#define  XC_HYB_GGA_XC_EDF2             476  /*Empirical functional from Lin, George and Gill                        */
#define  XC_HYB_GGA_XC_CAP0             477  /*Correct Asymptotic Potential hybrid                                   */
#define  XC_HYB_GGA_XC_LC_WPBE          478  /* Long-range corrected functional by Vydrov and Scuseria               */
#define  XC_HYB_GGA_XC_HSE12            479  /* HSE12 by Moussa, Schultz and Chelikowsky                             */
#define  XC_HYB_GGA_XC_HSE12S           480  /* Short-range HSE12 by Moussa, Schultz, and Chelikowsky                */
#define  XC_HYB_GGA_XC_HSE_SOL          481  /* HSEsol functional by Schimka, Harl, and Kresse                       */
#define  XC_HYB_GGA_XC_CAM_QTP_01       482  /* CAM-QTP(01): CAM-B3LYP retuned using ionization potentials of water  */
#define  XC_HYB_GGA_XC_MPW1LYP          483  /* Becke 1-parameter mixture of mPW91 and LYP                           */
#define  XC_HYB_GGA_XC_MPW1PBE          484  /* Becke 1-parameter mixture of mPW91 and PBE                           */
#define  XC_HYB_GGA_XC_KMLYP            485  /* Kang-Musgrave hybrid                                                 */
#define  XC_HYB_GGA_XC_B5050LYP         572  /* Like B3LYP but more exact exchange                                   */
#define  XC_MGGA_C_DLDF                  37  /* Dispersionless Density Functional                                    */
#define  XC_MGGA_XC_ZLP                  42  /*Zhao, Levy & Parr, Eq. (21)                                           */
#define  XC_MGGA_XC_OTPSS_D              64  /*oTPSS_D functional of Goerigk and Grimme                              */
#define  XC_MGGA_C_CS                    72  /*Colle and Salvetti                                                    */
#define  XC_MGGA_C_MN12_SX               73  /* MN12-SX correlation functional from Minnesota                        */
#define  XC_MGGA_C_MN12_L                74  /* MN12-L correlation functional from Minnesota                         */
#define  XC_MGGA_C_M11_L                 75  /* M11-L correlation functional from Minnesota                          */
#define  XC_MGGA_C_M11                   76  /* M11 correlation functional from Minnesota                            */
#define  XC_MGGA_C_M08_SO                77  /* M08-SO correlation functional from Minnesota                         */
#define  XC_MGGA_C_M08_HX                78  /*M08-HX correlation functional from Minnesota                          */
#define  XC_MGGA_X_LTA                  201  /*Local tau approximation of Ernzerhof & Scuseria                       */
#define  XC_MGGA_X_TPSS                 202  /*Tao, Perdew, Staroverov & Scuseria exchange                           */
#define  XC_MGGA_X_M06_L                203  /*M06-L exchange functional from Minnesota                              */
#define  XC_MGGA_X_GVT4                 204  /*GVT4 from Van Voorhis and Scuseria                                    */
#define  XC_MGGA_X_TAU_HCTH             205  /*tau-HCTH from Boese and Handy                                         */
#define  XC_MGGA_X_BR89                 206  /*Becke-Roussel 89                                                      */
#define  XC_MGGA_X_BJ06                 207  /* Becke & Johnson correction to Becke-Roussel 89                       */
#define  XC_MGGA_X_TB09                 208  /* Tran & Blaha correction to Becke & Johnson                           */
#define  XC_MGGA_X_RPP09                209  /* Rasanen, Pittalis, and Proetto correction to Becke & Johnson         */
#define  XC_MGGA_X_2D_PRHG07            210  /*Pittalis, Rasanen, Helbig, Gross Exchange Functional                  */
#define  XC_MGGA_X_2D_PRHG07_PRP10      211  /* PRGH07 with PRP10 correction                                         */
#define  XC_MGGA_X_REVTPSS              212  /* revised Tao, Perdew, Staroverov & Scuseria exchange                  */
#define  XC_MGGA_X_PKZB                 213  /*Perdew, Kurth, Zupan, and Blaha                                       */
#define  XC_MGGA_X_MS0                  221  /*MS exchange of Sun, Xiao, and Ruzsinszky                              */
#define  XC_MGGA_X_MS1                  222  /* MS1 exchange of Sun, et al                                           */
#define  XC_MGGA_X_MS2                  223  /* MS2 exchange of Sun, et al                                           */
#define  XC_MGGA_X_M11_L                226  /*M11-L exchange functional from Minnesota                              */
#define  XC_MGGA_X_MN12_L               227  /*MN12-L exchange functional from Minnesota                             */
#define  XC_MGGA_XC_CC06                229  /*Cancio and Chou 2006                                                  */
#define  XC_MGGA_X_MK00                 230  /*Exchange for accurate virtual orbital energies                        */
#define  XC_MGGA_C_TPSS                 231  /*Tao, Perdew, Staroverov & Scuseria correlation                        */
#define  XC_MGGA_C_VSXC                 232  /*VSxc from Van Voorhis and Scuseria (correlation part)                 */
#define  XC_MGGA_C_M06_L                233  /*M06-L correlation functional from Minnesota                           */
#define  XC_MGGA_C_M06_HF               234  /* M06-HF correlation functional from Minnesota                         */
#define  XC_MGGA_C_M06                  235  /* M06 correlation functional from Minnesota                            */
#define  XC_MGGA_C_M06_2X               236  /* M06-2X correlation functional from Minnesota                         */
#define  XC_MGGA_C_M05                  237  /*M05 correlation functional from Minnesota                             */
#define  XC_MGGA_C_M05_2X               238  /* M05-2X correlation functional from Minnesota                         */
#define  XC_MGGA_C_PKZB                 239  /*Perdew, Kurth, Zupan, and Blaha                                       */
#define  XC_MGGA_C_BC95                 240  /*Becke correlation 95                                                  */
#define  XC_MGGA_C_REVTPSS              241  /*revised TPSS correlation                                              */
#define  XC_MGGA_XC_TPSSLYP1W           242  /*Functionals fitted for water                                          */
#define  XC_MGGA_X_MK00B                243  /* Exchange for accurate virtual orbital energies (v. B)                */
#define  XC_MGGA_X_BLOC                 244  /* functional with balanced localization                                */
#define  XC_MGGA_X_MODTPSS              245  /* Modified Tao, Perdew, Staroverov & Scuseria exchange                 */
#define  XC_MGGA_C_TPSSLOC              247  /*Semilocal dynamical correlation                                       */
#define  XC_MGGA_X_MBEEF                249  /*mBEEF exchange                                                        */
#define  XC_MGGA_X_MBEEFVDW             250  /*mBEEF-vdW exchange                                                    */
#define  XC_MGGA_XC_B97M_V              254  /*Mardirossian and Head-Gordon                                          */
#define  XC_MGGA_X_MVS                  257  /*MVS exchange of Sun, Perdew, and Ruzsinszky                           */
#define  XC_MGGA_X_MN15_L               260  /* MN15-L exhange functional from Minnesota                             */
#define  XC_MGGA_C_MN15_L               261  /* MN15-L correlation functional from Minnesota                         */
#define  XC_MGGA_X_SCAN                 263  /*SCAN exchange of Sun, Ruzsinszky, and Perdew                          */
#define  XC_MGGA_C_SCAN                 267  /*SCAN correlation                                                      */
#define  XC_MGGA_C_MN15                 269  /* MN15 correlation functional from Minnesota                           */
#define  XC_MGGA_X_B00                  284  /* Becke 2000                                                           */
#define  XC_MGGA_XC_HLE17               288  /*high local exchange 2017                                              */
#define  XC_MGGA_C_SCAN_RVV10           292  /* SCAN correlation + rVV10 correlation                                 */
#define  XC_MGGA_X_REVM06_L             293  /* revised M06-L exchange functional from Minnesota                     */
#define  XC_MGGA_C_REVM06_L             294  /* Revised M06-L correlation functional from Minnesota                  */
#define  XC_MGGA_X_TM                   540  /*Tao and Mo 2016                                                       */
#define  XC_MGGA_X_VT84                 541  /*meta-GGA version of VT{8,4} GGA                                       */
#define  XC_MGGA_X_SA_TPSS              542  /*TPSS with correct surface asymptotics                                 */
#define  XC_MGGA_K_PC07                 543  /*Perdew and Constantin 2007                                            */
#define  XC_MGGA_C_KCIS                 562  /*Krieger, Chen, Iafrate, and Savin                                     */
#define  XC_MGGA_XC_LP90                564  /*Lee & Parr, Eq. (56)                                                  */
#define  XC_MGGA_C_B88                  571  /*Meta-GGA correlation by Becke                                         */
#define  XC_MGGA_X_GX                   575  /*GX functional of Loos                                                 */
#define  XC_MGGA_X_PBE_GX               576  /*PBE-GX functional of Loos                                             */
#define  XC_MGGA_X_REVSCAN              581  /* revised SCAN                                                         */
#define  XC_MGGA_C_REVSCAN              582  /*revised SCAN correlation                                              */
#define  XC_MGGA_C_SCAN_VV10            584  /* SCAN correlation +  VV10 correlation                                 */
#define  XC_MGGA_C_REVSCAN_VV10         585  /* revised SCAN correlation                                             */
#define  XC_MGGA_X_BR89_EXPLICIT        586  /*Becke-Roussel 89 with an explicit inversion of x(y)                   */
#define  XC_HYB_MGGA_X_DLDF              36  /*Dispersionless Density Functional                                     */
#define  XC_HYB_MGGA_X_MS2H             224  /*MS2 hybrid exchange of Sun, et al                                     */
#define  XC_HYB_MGGA_X_MN12_SX          248  /*MN12-SX hybrid exchange functional from Minnesota                     */
#define  XC_HYB_MGGA_X_SCAN0            264  /*SCAN hybrid exchange                                                  */
#define  XC_HYB_MGGA_X_MN15             268  /* MN15 hybrid exchange functional from Minnesota                       */
#define  XC_HYB_MGGA_X_BMK              279  /*Boese-Martin for kinetics                                             */
#define  XC_HYB_MGGA_X_TAU_HCTH         282  /* Hybrid version of tau-HCTH                                           */
#define  XC_HYB_MGGA_X_M08_HX           295  /*M08-HX exchange functional from Minnesota                             */
#define  XC_HYB_MGGA_X_M08_SO           296  /* M08-SO exchange functional from Minnesota                            */
#define  XC_HYB_MGGA_X_M11              297  /*M11 hybrid exchange functional from Minnesota                         */
#define  XC_HYB_MGGA_X_M05              438  /*M05 hybrid exchange functional from Minnesota                         */
#define  XC_HYB_MGGA_X_M05_2X           439  /* M05-2X hybrid exchange functional from Minnesota                     */
#define  XC_HYB_MGGA_XC_B88B95          440  /*Mixture of B88 with BC95 (B1B95)                                      */
#define  XC_HYB_MGGA_XC_B86B95          441  /* Mixture of B86 with BC95                                             */
#define  XC_HYB_MGGA_XC_PW86B95         442  /* Mixture of PW86 with BC95                                            */
#define  XC_HYB_MGGA_XC_BB1K            443  /* Mixture of B88 with BC95 from Zhao and Truhlar                       */
#define  XC_HYB_MGGA_X_M06_HF           444  /*M06-HF hybrid exchange functional from Minnesota                      */
#define  XC_HYB_MGGA_XC_MPW1B95         445  /* Mixture of mPW91 with BC95 from Zhao and Truhlar                     */
#define  XC_HYB_MGGA_XC_MPWB1K          446  /* Mixture of mPW91 with BC95 for kinetics                              */
#define  XC_HYB_MGGA_XC_X1B95           447  /* Mixture of X with BC95                                               */
#define  XC_HYB_MGGA_XC_XB1K            448  /* Mixture of X with BC95 for kinetics                                  */
#define  XC_HYB_MGGA_X_M06              449  /* M06 hybrid exchange functional from Minnesota                        */
#define  XC_HYB_MGGA_X_M06_2X           450  /* M06-2X hybrid exchange functional from Minnesota                     */
#define  XC_HYB_MGGA_XC_PW6B95          451  /* Mixture of PW91 with BC95 from Zhao and Truhlar                      */
#define  XC_HYB_MGGA_XC_PWB6K           452  /* Mixture of PW91 with BC95 from Zhao and Truhlar for kinetics         */
#define  XC_HYB_MGGA_XC_TPSSH           457  /*TPSS hybrid                                                           */
#define  XC_HYB_MGGA_XC_REVTPSSH        458  /* revTPSS hybrid                                                       */
#define  XC_HYB_MGGA_X_MVSH             474  /*MVSh hybrid                                                           */
#define  XC_HYB_MGGA_XC_WB97M_V         531  /*Mardirossian and Head-Gordon                                          */
#define  XC_HYB_MGGA_XC_B0KCIS          563  /*Hybrid based on KCIS                                                  */
#define  XC_HYB_MGGA_XC_MPW1KCIS        566  /*Modified Perdew-Wang + KCIS hybrid                                    */
#define  XC_HYB_MGGA_XC_MPWKCIS1K       567  /* Modified Perdew-Wang + KCIS hybrid with more exact exchange          */
#define  XC_HYB_MGGA_XC_PBE1KCIS        568  /* Perdew-Burke-Ernzerhof + KCIS hybrid                                 */
#define  XC_HYB_MGGA_XC_TPSS1KCIS       569  /* TPSS hybrid with KCIS correlation                                    */
#define  XC_HYB_MGGA_X_REVSCAN0         583  /* revised SCAN hybrid exchange                                         */
#define  XC_HYB_MGGA_XC_B98             598  /*Becke 98                                                              */
