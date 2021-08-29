__device__ __inline__   void h2_6_0(QUICKDouble* YVerticalTemp, QUICKDouble* store,
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

    // call for L =            1  B =            0
    f_1_0_t f_1_0_5 ( VY( 0, 0, 5 ),  VY( 0, 0, 6 ), Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            2  B =            0
    f_2_0_t f_2_0_4 ( f_1_0_4,  f_1_0_5, VY( 0, 0, 4 ), VY( 0, 0, 5 ), ABtemp, CDcom, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            3  B =            0
    f_3_0_t f_3_0_3 ( f_2_0_3,  f_2_0_4, f_1_0_3, f_1_0_4, ABtemp, CDcom, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            4  B =            0
    f_4_0_t f_4_0_2 ( f_3_0_2,  f_3_0_3, f_2_0_2, f_2_0_3, ABtemp, CDcom, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for B =            5  L =            0
    f_5_0_t f_5_0_1 ( f_4_0_1, f_4_0_2, f_3_0_1, f_3_0_2, ABtemp, CDcom, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for B =            6  L =            0
    f_6_0_t f_6_0_0 ( f_5_0_0, f_5_0_1, f_4_0_0, f_4_0_1, ABtemp, CDcom, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // WRITE LAST FOR I =            6  J=           0
    LOCSTORE(store, 56,  0, STOREDIM, STOREDIM) = f_6_0_0.x_56_0 ;
    LOCSTORE(store, 57,  0, STOREDIM, STOREDIM) = f_6_0_0.x_57_0 ;
    LOCSTORE(store, 58,  0, STOREDIM, STOREDIM) = f_6_0_0.x_58_0 ;
    LOCSTORE(store, 59,  0, STOREDIM, STOREDIM) = f_6_0_0.x_59_0 ;
    LOCSTORE(store, 60,  0, STOREDIM, STOREDIM) = f_6_0_0.x_60_0 ;
    LOCSTORE(store, 61,  0, STOREDIM, STOREDIM) = f_6_0_0.x_61_0 ;
    LOCSTORE(store, 62,  0, STOREDIM, STOREDIM) = f_6_0_0.x_62_0 ;
    LOCSTORE(store, 63,  0, STOREDIM, STOREDIM) = f_6_0_0.x_63_0 ;
    LOCSTORE(store, 64,  0, STOREDIM, STOREDIM) = f_6_0_0.x_64_0 ;
    LOCSTORE(store, 65,  0, STOREDIM, STOREDIM) = f_6_0_0.x_65_0 ;
    LOCSTORE(store, 66,  0, STOREDIM, STOREDIM) = f_6_0_0.x_66_0 ;
    LOCSTORE(store, 67,  0, STOREDIM, STOREDIM) = f_6_0_0.x_67_0 ;
    LOCSTORE(store, 68,  0, STOREDIM, STOREDIM) = f_6_0_0.x_68_0 ;
    LOCSTORE(store, 69,  0, STOREDIM, STOREDIM) = f_6_0_0.x_69_0 ;
    LOCSTORE(store, 70,  0, STOREDIM, STOREDIM) = f_6_0_0.x_70_0 ;
    LOCSTORE(store, 71,  0, STOREDIM, STOREDIM) = f_6_0_0.x_71_0 ;
    LOCSTORE(store, 72,  0, STOREDIM, STOREDIM) = f_6_0_0.x_72_0 ;
    LOCSTORE(store, 73,  0, STOREDIM, STOREDIM) = f_6_0_0.x_73_0 ;
    LOCSTORE(store, 74,  0, STOREDIM, STOREDIM) = f_6_0_0.x_74_0 ;
    LOCSTORE(store, 75,  0, STOREDIM, STOREDIM) = f_6_0_0.x_75_0 ;
    LOCSTORE(store, 76,  0, STOREDIM, STOREDIM) = f_6_0_0.x_76_0 ;
    LOCSTORE(store, 77,  0, STOREDIM, STOREDIM) = f_6_0_0.x_77_0 ;
    LOCSTORE(store, 78,  0, STOREDIM, STOREDIM) = f_6_0_0.x_78_0 ;
    LOCSTORE(store, 79,  0, STOREDIM, STOREDIM) = f_6_0_0.x_79_0 ;
    LOCSTORE(store, 80,  0, STOREDIM, STOREDIM) = f_6_0_0.x_80_0 ;
    LOCSTORE(store, 81,  0, STOREDIM, STOREDIM) = f_6_0_0.x_81_0 ;
    LOCSTORE(store, 82,  0, STOREDIM, STOREDIM) = f_6_0_0.x_82_0 ;
    LOCSTORE(store, 83,  0, STOREDIM, STOREDIM) = f_6_0_0.x_83_0 ;
}
