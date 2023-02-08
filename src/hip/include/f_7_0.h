__device__ __inline__ void h_7_0(QUICKDouble* YVerticalTemp, QUICKDouble* store,
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

    // call for L =            1  B =            0
    f_1_0_t f_1_0_6 ( VY( 0, 0, 6 ),  VY( 0, 0, 7 ), Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            2  B =            0
    f_2_0_t f_2_0_5 ( f_1_0_5,  f_1_0_6, VY( 0, 0, 5 ), VY( 0, 0, 6 ), ABtemp, CDcom, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            3  B =            0
    f_3_0_t f_3_0_4 ( f_2_0_4,  f_2_0_5, f_1_0_4, f_1_0_5, ABtemp, CDcom, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            4  B =            0
    f_4_0_t f_4_0_3 ( f_3_0_3,  f_3_0_4, f_2_0_3, f_2_0_4, ABtemp, CDcom, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for B =            5  L =            0
    f_5_0_t f_5_0_2 ( f_4_0_2, f_4_0_3, f_3_0_2, f_3_0_3, ABtemp, CDcom, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for B =            6  L =            0
    f_6_0_t f_6_0_1 ( f_5_0_1, f_5_0_2, f_4_0_1, f_4_0_2, ABtemp, CDcom, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for B =            7  L =            0
    f_7_0_t f_7_0_0 ( f_6_0_0, f_6_0_1, f_5_0_0, f_5_0_1, ABtemp, CDcom, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // WRITE LAST FOR I =            7  J=           0
    LOCSTORE(store, 84,  0, STOREDIM, STOREDIM) += f_7_0_0.x_84_0 ;
    LOCSTORE(store, 85,  0, STOREDIM, STOREDIM) += f_7_0_0.x_85_0 ;
    LOCSTORE(store, 86,  0, STOREDIM, STOREDIM) += f_7_0_0.x_86_0 ;
    LOCSTORE(store, 87,  0, STOREDIM, STOREDIM) += f_7_0_0.x_87_0 ;
    LOCSTORE(store, 88,  0, STOREDIM, STOREDIM) += f_7_0_0.x_88_0 ;
    LOCSTORE(store, 89,  0, STOREDIM, STOREDIM) += f_7_0_0.x_89_0 ;
    LOCSTORE(store, 90,  0, STOREDIM, STOREDIM) += f_7_0_0.x_90_0 ;
    LOCSTORE(store, 91,  0, STOREDIM, STOREDIM) += f_7_0_0.x_91_0 ;
    LOCSTORE(store, 92,  0, STOREDIM, STOREDIM) += f_7_0_0.x_92_0 ;
    LOCSTORE(store, 93,  0, STOREDIM, STOREDIM) += f_7_0_0.x_93_0 ;
    LOCSTORE(store, 94,  0, STOREDIM, STOREDIM) += f_7_0_0.x_94_0 ;
    LOCSTORE(store, 95,  0, STOREDIM, STOREDIM) += f_7_0_0.x_95_0 ;
    LOCSTORE(store, 96,  0, STOREDIM, STOREDIM) += f_7_0_0.x_96_0 ;
    LOCSTORE(store, 97,  0, STOREDIM, STOREDIM) += f_7_0_0.x_97_0 ;
    LOCSTORE(store, 98,  0, STOREDIM, STOREDIM) += f_7_0_0.x_98_0 ;
    LOCSTORE(store, 99,  0, STOREDIM, STOREDIM) += f_7_0_0.x_99_0 ;
    LOCSTORE(store,100,  0, STOREDIM, STOREDIM) += f_7_0_0.x_100_0 ;
    LOCSTORE(store,101,  0, STOREDIM, STOREDIM) += f_7_0_0.x_101_0 ;
    LOCSTORE(store,102,  0, STOREDIM, STOREDIM) += f_7_0_0.x_102_0 ;
    LOCSTORE(store,103,  0, STOREDIM, STOREDIM) += f_7_0_0.x_103_0 ;
    LOCSTORE(store,104,  0, STOREDIM, STOREDIM) += f_7_0_0.x_104_0 ;
    LOCSTORE(store,105,  0, STOREDIM, STOREDIM) += f_7_0_0.x_105_0 ;
    LOCSTORE(store,106,  0, STOREDIM, STOREDIM) += f_7_0_0.x_106_0 ;
    LOCSTORE(store,107,  0, STOREDIM, STOREDIM) += f_7_0_0.x_107_0 ;
    LOCSTORE(store,108,  0, STOREDIM, STOREDIM) += f_7_0_0.x_108_0 ;
    LOCSTORE(store,109,  0, STOREDIM, STOREDIM) += f_7_0_0.x_109_0 ;
    LOCSTORE(store,110,  0, STOREDIM, STOREDIM) += f_7_0_0.x_110_0 ;
    LOCSTORE(store,111,  0, STOREDIM, STOREDIM) += f_7_0_0.x_111_0 ;
    LOCSTORE(store,112,  0, STOREDIM, STOREDIM) += f_7_0_0.x_112_0 ;
    LOCSTORE(store,113,  0, STOREDIM, STOREDIM) += f_7_0_0.x_113_0 ;
    LOCSTORE(store,114,  0, STOREDIM, STOREDIM) += f_7_0_0.x_114_0 ;
    LOCSTORE(store,115,  0, STOREDIM, STOREDIM) += f_7_0_0.x_115_0 ;
    LOCSTORE(store,116,  0, STOREDIM, STOREDIM) += f_7_0_0.x_116_0 ;
    LOCSTORE(store,117,  0, STOREDIM, STOREDIM) += f_7_0_0.x_117_0 ;
    LOCSTORE(store,118,  0, STOREDIM, STOREDIM) += f_7_0_0.x_118_0 ;
    LOCSTORE(store,119,  0, STOREDIM, STOREDIM) += f_7_0_0.x_119_0 ;
}
