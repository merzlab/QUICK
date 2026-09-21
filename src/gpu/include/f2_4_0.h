__device__ __inline__   void h2_4_0(QUICKDouble* YVerticalTemp, QUICKDouble* store,
                                   QUICKDouble Ptempx, QUICKDouble Ptempy, QUICKDouble Ptempz,  \
                                   QUICKDouble WPtempx,QUICKDouble WPtempy,QUICKDouble WPtempz, \
                                   QUICKDouble Qtempx, QUICKDouble Qtempy, QUICKDouble Qtempz,  \
                                   QUICKDouble WQtempx,QUICKDouble WQtempy,QUICKDouble WQtempz, \
                                   QUICKDouble ABCDtemp,QUICKDouble ABtemp, \
                                   QUICKDouble CDtemp, QUICKDouble ABcom, QUICKDouble CDcom)
{
    // call for L =            1  B =            0
    f_1_0_t f_1_0_0 ( VY( 0, 0, 0 ),  VY( 0, 0, 1 ), Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            1  B =            0
    f_1_0_t f_1_0_1 ( VY( 0, 0, 1 ),  VY( 0, 0, 2 ), Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            2  B =            0
    f_2_0_t f_2_0_0 ( f_1_0_0,  f_1_0_1, VY( 0, 0, 0 ), VY( 0, 0, 1 ), ABtemp, CDcom, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            1  B =            0
    f_1_0_t f_1_0_2 ( VY( 0, 0, 2 ),  VY( 0, 0, 3 ), Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            2  B =            0
    f_2_0_t f_2_0_1 ( f_1_0_1,  f_1_0_2, VY( 0, 0, 1 ), VY( 0, 0, 2 ), ABtemp, CDcom, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            3  B =            0
    f_3_0_t f_3_0_0 ( f_2_0_0,  f_2_0_1, f_1_0_0, f_1_0_1, ABtemp, CDcom, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            1  B =            0
    f_1_0_t f_1_0_3 ( VY( 0, 0, 3 ),  VY( 0, 0, 4 ), Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            2  B =            0
    f_2_0_t f_2_0_2 ( f_1_0_2,  f_1_0_3, VY( 0, 0, 2 ), VY( 0, 0, 3 ), ABtemp, CDcom, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            3  B =            0
    f_3_0_t f_3_0_1 ( f_2_0_1,  f_2_0_2, f_1_0_1, f_1_0_2, ABtemp, CDcom, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            4  B =            0
    f_4_0_t f_4_0_0 ( f_3_0_0,  f_3_0_1, f_2_0_0, f_2_0_1, ABtemp, CDcom, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // WRITE LAST FOR I =            4  J=           0
    LOCSTORE(store, 20,  0, STOREDIM, STOREDIM) = f_4_0_0.x_20_0 ;
    LOCSTORE(store, 21,  0, STOREDIM, STOREDIM) = f_4_0_0.x_21_0 ;
    LOCSTORE(store, 22,  0, STOREDIM, STOREDIM) = f_4_0_0.x_22_0 ;
    LOCSTORE(store, 23,  0, STOREDIM, STOREDIM) = f_4_0_0.x_23_0 ;
    LOCSTORE(store, 24,  0, STOREDIM, STOREDIM) = f_4_0_0.x_24_0 ;
    LOCSTORE(store, 25,  0, STOREDIM, STOREDIM) = f_4_0_0.x_25_0 ;
    LOCSTORE(store, 26,  0, STOREDIM, STOREDIM) = f_4_0_0.x_26_0 ;
    LOCSTORE(store, 27,  0, STOREDIM, STOREDIM) = f_4_0_0.x_27_0 ;
    LOCSTORE(store, 28,  0, STOREDIM, STOREDIM) = f_4_0_0.x_28_0 ;
    LOCSTORE(store, 29,  0, STOREDIM, STOREDIM) = f_4_0_0.x_29_0 ;
    LOCSTORE(store, 30,  0, STOREDIM, STOREDIM) = f_4_0_0.x_30_0 ;
    LOCSTORE(store, 31,  0, STOREDIM, STOREDIM) = f_4_0_0.x_31_0 ;
    LOCSTORE(store, 32,  0, STOREDIM, STOREDIM) = f_4_0_0.x_32_0 ;
    LOCSTORE(store, 33,  0, STOREDIM, STOREDIM) = f_4_0_0.x_33_0 ;
    LOCSTORE(store, 34,  0, STOREDIM, STOREDIM) = f_4_0_0.x_34_0 ;
}
