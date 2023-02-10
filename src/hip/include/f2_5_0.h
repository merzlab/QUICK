__device__ __inline__   void h2_5_0(QUICKDouble* YVerticalTemp, QUICKDouble* store,
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

    // call for L =            1  B =            0
    f_1_0_t f_1_0_4 ( VY( 0, 0, 4 ),  VY( 0, 0, 5 ), Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            2  B =            0
    f_2_0_t f_2_0_3 ( f_1_0_3,  f_1_0_4, VY( 0, 0, 3 ), VY( 0, 0, 4 ), ABtemp, CDcom, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            3  B =            0
    f_3_0_t f_3_0_2 ( f_2_0_2,  f_2_0_3, f_1_0_2, f_1_0_3, ABtemp, CDcom, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            4  B =            0
    f_4_0_t f_4_0_1 ( f_3_0_1,  f_3_0_2, f_2_0_1, f_2_0_2, ABtemp, CDcom, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for B =            5  L =            0
    f_5_0_t f_5_0_0 ( f_4_0_0, f_4_0_1, f_3_0_0, f_3_0_1, ABtemp, CDcom, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // WRITE LAST FOR I =            5  J=           0
    LOCSTORE(store, 35,  0, STOREDIM, STOREDIM) = f_5_0_0.x_35_0 ;
    LOCSTORE(store, 36,  0, STOREDIM, STOREDIM) = f_5_0_0.x_36_0 ;
    LOCSTORE(store, 37,  0, STOREDIM, STOREDIM) = f_5_0_0.x_37_0 ;
    LOCSTORE(store, 38,  0, STOREDIM, STOREDIM) = f_5_0_0.x_38_0 ;
    LOCSTORE(store, 39,  0, STOREDIM, STOREDIM) = f_5_0_0.x_39_0 ;
    LOCSTORE(store, 40,  0, STOREDIM, STOREDIM) = f_5_0_0.x_40_0 ;
    LOCSTORE(store, 41,  0, STOREDIM, STOREDIM) = f_5_0_0.x_41_0 ;
    LOCSTORE(store, 42,  0, STOREDIM, STOREDIM) = f_5_0_0.x_42_0 ;
    LOCSTORE(store, 43,  0, STOREDIM, STOREDIM) = f_5_0_0.x_43_0 ;
    LOCSTORE(store, 44,  0, STOREDIM, STOREDIM) = f_5_0_0.x_44_0 ;
    LOCSTORE(store, 45,  0, STOREDIM, STOREDIM) = f_5_0_0.x_45_0 ;
    LOCSTORE(store, 46,  0, STOREDIM, STOREDIM) = f_5_0_0.x_46_0 ;
    LOCSTORE(store, 47,  0, STOREDIM, STOREDIM) = f_5_0_0.x_47_0 ;
    LOCSTORE(store, 48,  0, STOREDIM, STOREDIM) = f_5_0_0.x_48_0 ;
    LOCSTORE(store, 49,  0, STOREDIM, STOREDIM) = f_5_0_0.x_49_0 ;
    LOCSTORE(store, 50,  0, STOREDIM, STOREDIM) = f_5_0_0.x_50_0 ;
    LOCSTORE(store, 51,  0, STOREDIM, STOREDIM) = f_5_0_0.x_51_0 ;
    LOCSTORE(store, 52,  0, STOREDIM, STOREDIM) = f_5_0_0.x_52_0 ;
    LOCSTORE(store, 53,  0, STOREDIM, STOREDIM) = f_5_0_0.x_53_0 ;
    LOCSTORE(store, 54,  0, STOREDIM, STOREDIM) = f_5_0_0.x_54_0 ;
    LOCSTORE(store, 55,  0, STOREDIM, STOREDIM) = f_5_0_0.x_55_0 ;
}
