__device__ __inline__ void h_7_1(QUICKDouble* YVerticalTemp, QUICKDouble* store,
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

    // call for L =            1  B =            0
    f_1_0_t f_1_0_7 ( VY( 0, 0, 7 ),  VY( 0, 0, 8 ), Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            2  B =            0
    f_2_0_t f_2_0_6 ( f_1_0_6,  f_1_0_7, VY( 0, 0, 6 ), VY( 0, 0, 7 ), ABtemp, CDcom, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            3  B =            0
    f_3_0_t f_3_0_5 ( f_2_0_5,  f_2_0_6, f_1_0_5, f_1_0_6, ABtemp, CDcom, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            4  B =            0
    f_4_0_t f_4_0_4 ( f_3_0_4,  f_3_0_5, f_2_0_4, f_2_0_5, ABtemp, CDcom, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for B =            5  L =            0
    f_5_0_t f_5_0_3 ( f_4_0_3, f_4_0_4, f_3_0_3, f_3_0_4, ABtemp, CDcom, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for B =            6  L =            0
    f_6_0_t f_6_0_2 ( f_5_0_2, f_5_0_3, f_4_0_2, f_4_0_3, ABtemp, CDcom, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for B =            7  L =            0
    f_7_0_t f_7_0_1 ( f_6_0_1, f_6_0_2, f_5_0_1, f_5_0_2, ABtemp, CDcom, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for B =            7  L =            1
    f_7_1_t f_7_1_0 ( f_7_0_0,  f_7_0_1,  f_6_0_1, ABCDtemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // WRITE LAST FOR I =            7  J=           1
    LOCSTORE(store, 84,  1, STOREDIM, STOREDIM) += f_7_1_0.x_84_1 ;
    LOCSTORE(store, 84,  2, STOREDIM, STOREDIM) += f_7_1_0.x_84_2 ;
    LOCSTORE(store, 84,  3, STOREDIM, STOREDIM) += f_7_1_0.x_84_3 ;
    LOCSTORE(store, 85,  1, STOREDIM, STOREDIM) += f_7_1_0.x_85_1 ;
    LOCSTORE(store, 85,  2, STOREDIM, STOREDIM) += f_7_1_0.x_85_2 ;
    LOCSTORE(store, 85,  3, STOREDIM, STOREDIM) += f_7_1_0.x_85_3 ;
    LOCSTORE(store, 86,  1, STOREDIM, STOREDIM) += f_7_1_0.x_86_1 ;
    LOCSTORE(store, 86,  2, STOREDIM, STOREDIM) += f_7_1_0.x_86_2 ;
    LOCSTORE(store, 86,  3, STOREDIM, STOREDIM) += f_7_1_0.x_86_3 ;
    LOCSTORE(store, 87,  1, STOREDIM, STOREDIM) += f_7_1_0.x_87_1 ;
    LOCSTORE(store, 87,  2, STOREDIM, STOREDIM) += f_7_1_0.x_87_2 ;
    LOCSTORE(store, 87,  3, STOREDIM, STOREDIM) += f_7_1_0.x_87_3 ;
    LOCSTORE(store, 88,  1, STOREDIM, STOREDIM) += f_7_1_0.x_88_1 ;
    LOCSTORE(store, 88,  2, STOREDIM, STOREDIM) += f_7_1_0.x_88_2 ;
    LOCSTORE(store, 88,  3, STOREDIM, STOREDIM) += f_7_1_0.x_88_3 ;
    LOCSTORE(store, 89,  1, STOREDIM, STOREDIM) += f_7_1_0.x_89_1 ;
    LOCSTORE(store, 89,  2, STOREDIM, STOREDIM) += f_7_1_0.x_89_2 ;
    LOCSTORE(store, 89,  3, STOREDIM, STOREDIM) += f_7_1_0.x_89_3 ;
    LOCSTORE(store, 90,  1, STOREDIM, STOREDIM) += f_7_1_0.x_90_1 ;
    LOCSTORE(store, 90,  2, STOREDIM, STOREDIM) += f_7_1_0.x_90_2 ;
    LOCSTORE(store, 90,  3, STOREDIM, STOREDIM) += f_7_1_0.x_90_3 ;
    LOCSTORE(store, 91,  1, STOREDIM, STOREDIM) += f_7_1_0.x_91_1 ;
    LOCSTORE(store, 91,  2, STOREDIM, STOREDIM) += f_7_1_0.x_91_2 ;
    LOCSTORE(store, 91,  3, STOREDIM, STOREDIM) += f_7_1_0.x_91_3 ;
    LOCSTORE(store, 92,  1, STOREDIM, STOREDIM) += f_7_1_0.x_92_1 ;
    LOCSTORE(store, 92,  2, STOREDIM, STOREDIM) += f_7_1_0.x_92_2 ;
    LOCSTORE(store, 92,  3, STOREDIM, STOREDIM) += f_7_1_0.x_92_3 ;
    LOCSTORE(store, 93,  1, STOREDIM, STOREDIM) += f_7_1_0.x_93_1 ;
    LOCSTORE(store, 93,  2, STOREDIM, STOREDIM) += f_7_1_0.x_93_2 ;
    LOCSTORE(store, 93,  3, STOREDIM, STOREDIM) += f_7_1_0.x_93_3 ;
    LOCSTORE(store, 94,  1, STOREDIM, STOREDIM) += f_7_1_0.x_94_1 ;
    LOCSTORE(store, 94,  2, STOREDIM, STOREDIM) += f_7_1_0.x_94_2 ;
    LOCSTORE(store, 94,  3, STOREDIM, STOREDIM) += f_7_1_0.x_94_3 ;
    LOCSTORE(store, 95,  1, STOREDIM, STOREDIM) += f_7_1_0.x_95_1 ;
    LOCSTORE(store, 95,  2, STOREDIM, STOREDIM) += f_7_1_0.x_95_2 ;
    LOCSTORE(store, 95,  3, STOREDIM, STOREDIM) += f_7_1_0.x_95_3 ;
    LOCSTORE(store, 96,  1, STOREDIM, STOREDIM) += f_7_1_0.x_96_1 ;
    LOCSTORE(store, 96,  2, STOREDIM, STOREDIM) += f_7_1_0.x_96_2 ;
    LOCSTORE(store, 96,  3, STOREDIM, STOREDIM) += f_7_1_0.x_96_3 ;
    LOCSTORE(store, 97,  1, STOREDIM, STOREDIM) += f_7_1_0.x_97_1 ;
    LOCSTORE(store, 97,  2, STOREDIM, STOREDIM) += f_7_1_0.x_97_2 ;
    LOCSTORE(store, 97,  3, STOREDIM, STOREDIM) += f_7_1_0.x_97_3 ;
    LOCSTORE(store, 98,  1, STOREDIM, STOREDIM) += f_7_1_0.x_98_1 ;
    LOCSTORE(store, 98,  2, STOREDIM, STOREDIM) += f_7_1_0.x_98_2 ;
    LOCSTORE(store, 98,  3, STOREDIM, STOREDIM) += f_7_1_0.x_98_3 ;
    LOCSTORE(store, 99,  1, STOREDIM, STOREDIM) += f_7_1_0.x_99_1 ;
    LOCSTORE(store, 99,  2, STOREDIM, STOREDIM) += f_7_1_0.x_99_2 ;
    LOCSTORE(store, 99,  3, STOREDIM, STOREDIM) += f_7_1_0.x_99_3 ;
    LOCSTORE(store,100,  1, STOREDIM, STOREDIM) += f_7_1_0.x_100_1 ;
    LOCSTORE(store,100,  2, STOREDIM, STOREDIM) += f_7_1_0.x_100_2 ;
    LOCSTORE(store,100,  3, STOREDIM, STOREDIM) += f_7_1_0.x_100_3 ;
    LOCSTORE(store,101,  1, STOREDIM, STOREDIM) += f_7_1_0.x_101_1 ;
    LOCSTORE(store,101,  2, STOREDIM, STOREDIM) += f_7_1_0.x_101_2 ;
    LOCSTORE(store,101,  3, STOREDIM, STOREDIM) += f_7_1_0.x_101_3 ;
    LOCSTORE(store,102,  1, STOREDIM, STOREDIM) += f_7_1_0.x_102_1 ;
    LOCSTORE(store,102,  2, STOREDIM, STOREDIM) += f_7_1_0.x_102_2 ;
    LOCSTORE(store,102,  3, STOREDIM, STOREDIM) += f_7_1_0.x_102_3 ;
    LOCSTORE(store,103,  1, STOREDIM, STOREDIM) += f_7_1_0.x_103_1 ;
    LOCSTORE(store,103,  2, STOREDIM, STOREDIM) += f_7_1_0.x_103_2 ;
    LOCSTORE(store,103,  3, STOREDIM, STOREDIM) += f_7_1_0.x_103_3 ;
    LOCSTORE(store,104,  1, STOREDIM, STOREDIM) += f_7_1_0.x_104_1 ;
    LOCSTORE(store,104,  2, STOREDIM, STOREDIM) += f_7_1_0.x_104_2 ;
    LOCSTORE(store,104,  3, STOREDIM, STOREDIM) += f_7_1_0.x_104_3 ;
    LOCSTORE(store,105,  1, STOREDIM, STOREDIM) += f_7_1_0.x_105_1 ;
    LOCSTORE(store,105,  2, STOREDIM, STOREDIM) += f_7_1_0.x_105_2 ;
    LOCSTORE(store,105,  3, STOREDIM, STOREDIM) += f_7_1_0.x_105_3 ;
    LOCSTORE(store,106,  1, STOREDIM, STOREDIM) += f_7_1_0.x_106_1 ;
    LOCSTORE(store,106,  2, STOREDIM, STOREDIM) += f_7_1_0.x_106_2 ;
    LOCSTORE(store,106,  3, STOREDIM, STOREDIM) += f_7_1_0.x_106_3 ;
    LOCSTORE(store,107,  1, STOREDIM, STOREDIM) += f_7_1_0.x_107_1 ;
    LOCSTORE(store,107,  2, STOREDIM, STOREDIM) += f_7_1_0.x_107_2 ;
    LOCSTORE(store,107,  3, STOREDIM, STOREDIM) += f_7_1_0.x_107_3 ;
    LOCSTORE(store,108,  1, STOREDIM, STOREDIM) += f_7_1_0.x_108_1 ;
    LOCSTORE(store,108,  2, STOREDIM, STOREDIM) += f_7_1_0.x_108_2 ;
    LOCSTORE(store,108,  3, STOREDIM, STOREDIM) += f_7_1_0.x_108_3 ;
    LOCSTORE(store,109,  1, STOREDIM, STOREDIM) += f_7_1_0.x_109_1 ;
    LOCSTORE(store,109,  2, STOREDIM, STOREDIM) += f_7_1_0.x_109_2 ;
    LOCSTORE(store,109,  3, STOREDIM, STOREDIM) += f_7_1_0.x_109_3 ;
    LOCSTORE(store,110,  1, STOREDIM, STOREDIM) += f_7_1_0.x_110_1 ;
    LOCSTORE(store,110,  2, STOREDIM, STOREDIM) += f_7_1_0.x_110_2 ;
    LOCSTORE(store,110,  3, STOREDIM, STOREDIM) += f_7_1_0.x_110_3 ;
    LOCSTORE(store,111,  1, STOREDIM, STOREDIM) += f_7_1_0.x_111_1 ;
    LOCSTORE(store,111,  2, STOREDIM, STOREDIM) += f_7_1_0.x_111_2 ;
    LOCSTORE(store,111,  3, STOREDIM, STOREDIM) += f_7_1_0.x_111_3 ;
    LOCSTORE(store,112,  1, STOREDIM, STOREDIM) += f_7_1_0.x_112_1 ;
    LOCSTORE(store,112,  2, STOREDIM, STOREDIM) += f_7_1_0.x_112_2 ;
    LOCSTORE(store,112,  3, STOREDIM, STOREDIM) += f_7_1_0.x_112_3 ;
    LOCSTORE(store,113,  1, STOREDIM, STOREDIM) += f_7_1_0.x_113_1 ;
    LOCSTORE(store,113,  2, STOREDIM, STOREDIM) += f_7_1_0.x_113_2 ;
    LOCSTORE(store,113,  3, STOREDIM, STOREDIM) += f_7_1_0.x_113_3 ;
    LOCSTORE(store,114,  1, STOREDIM, STOREDIM) += f_7_1_0.x_114_1 ;
    LOCSTORE(store,114,  2, STOREDIM, STOREDIM) += f_7_1_0.x_114_2 ;
    LOCSTORE(store,114,  3, STOREDIM, STOREDIM) += f_7_1_0.x_114_3 ;
    LOCSTORE(store,115,  1, STOREDIM, STOREDIM) += f_7_1_0.x_115_1 ;
    LOCSTORE(store,115,  2, STOREDIM, STOREDIM) += f_7_1_0.x_115_2 ;
    LOCSTORE(store,115,  3, STOREDIM, STOREDIM) += f_7_1_0.x_115_3 ;
    LOCSTORE(store,116,  1, STOREDIM, STOREDIM) += f_7_1_0.x_116_1 ;
    LOCSTORE(store,116,  2, STOREDIM, STOREDIM) += f_7_1_0.x_116_2 ;
    LOCSTORE(store,116,  3, STOREDIM, STOREDIM) += f_7_1_0.x_116_3 ;
    LOCSTORE(store,117,  1, STOREDIM, STOREDIM) += f_7_1_0.x_117_1 ;
    LOCSTORE(store,117,  2, STOREDIM, STOREDIM) += f_7_1_0.x_117_2 ;
    LOCSTORE(store,117,  3, STOREDIM, STOREDIM) += f_7_1_0.x_117_3 ;
    LOCSTORE(store,118,  1, STOREDIM, STOREDIM) += f_7_1_0.x_118_1 ;
    LOCSTORE(store,118,  2, STOREDIM, STOREDIM) += f_7_1_0.x_118_2 ;
    LOCSTORE(store,118,  3, STOREDIM, STOREDIM) += f_7_1_0.x_118_3 ;
    LOCSTORE(store,119,  1, STOREDIM, STOREDIM) += f_7_1_0.x_119_1 ;
    LOCSTORE(store,119,  2, STOREDIM, STOREDIM) += f_7_1_0.x_119_2 ;
    LOCSTORE(store,119,  3, STOREDIM, STOREDIM) += f_7_1_0.x_119_3 ;
}
