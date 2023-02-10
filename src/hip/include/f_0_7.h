__device__ __inline__ void h_0_7(QUICKDouble* YVerticalTemp, QUICKDouble* store,
                                   QUICKDouble Ptempx, QUICKDouble Ptempy, QUICKDouble Ptempz,  \
                                   QUICKDouble WPtempx,QUICKDouble WPtempy,QUICKDouble WPtempz, \
                                   QUICKDouble Qtempx, QUICKDouble Qtempy, QUICKDouble Qtempz,  \
                                   QUICKDouble WQtempx,QUICKDouble WQtempy,QUICKDouble WQtempz, \
                                   QUICKDouble ABCDtemp,QUICKDouble ABtemp, \
                                   QUICKDouble CDtemp, QUICKDouble ABcom, QUICKDouble CDcom)
{
    // call for L =            0  B =            1
    f_0_1_t f_0_1_0 ( VY( 0, 0, 0 ), VY( 0, 0, 1 ), Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            0  B =            1
    f_0_1_t f_0_1_1 ( VY( 0, 0, 1 ), VY( 0, 0, 2 ), Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            0  B =            2
    f_0_2_t f_0_2_0 ( f_0_1_0, f_0_1_1, VY( 0, 0, 0 ), VY( 0, 0, 1 ), CDtemp, ABcom, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            0  B =            1
    f_0_1_t f_0_1_2 ( VY( 0, 0, 2 ), VY( 0, 0, 3 ), Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            0  B =            2
    f_0_2_t f_0_2_1 ( f_0_1_1, f_0_1_2, VY( 0, 0, 1 ), VY( 0, 0, 2 ), CDtemp, ABcom, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            0  B =            3
    f_0_3_t f_0_3_0 ( f_0_2_0, f_0_2_1, f_0_1_0, f_0_1_1, CDtemp, ABcom, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            0  B =            1
    f_0_1_t f_0_1_3 ( VY( 0, 0, 3 ), VY( 0, 0, 4 ), Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            0  B =            2
    f_0_2_t f_0_2_2 ( f_0_1_2, f_0_1_3, VY( 0, 0, 2 ), VY( 0, 0, 3 ), CDtemp, ABcom, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            0  B =            3
    f_0_3_t f_0_3_1 ( f_0_2_1, f_0_2_2, f_0_1_1, f_0_1_2, CDtemp, ABcom, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            0  B =            4
    f_0_4_t f_0_4_0 ( f_0_3_0, f_0_3_1, f_0_2_0, f_0_2_1, CDtemp, ABcom, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            0  B =            1
    f_0_1_t f_0_1_4 ( VY( 0, 0, 4 ), VY( 0, 0, 5 ), Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            0  B =            2
    f_0_2_t f_0_2_3 ( f_0_1_3, f_0_1_4, VY( 0, 0, 3 ), VY( 0, 0, 4 ), CDtemp, ABcom, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            0  B =            3
    f_0_3_t f_0_3_2 ( f_0_2_2, f_0_2_3, f_0_1_2, f_0_1_3, CDtemp, ABcom, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            0  B =            4
    f_0_4_t f_0_4_1 ( f_0_3_1, f_0_3_2, f_0_2_1, f_0_2_2, CDtemp, ABcom, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            0  B =            5
    f_0_5_t f_0_5_0 ( f_0_4_0, f_0_4_1, f_0_3_0, f_0_3_1, CDtemp, ABcom, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            0  B =            1
    f_0_1_t f_0_1_5 ( VY( 0, 0, 5 ), VY( 0, 0, 6 ), Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            0  B =            2
    f_0_2_t f_0_2_4 ( f_0_1_4, f_0_1_5, VY( 0, 0, 4 ), VY( 0, 0, 5 ), CDtemp, ABcom, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            0  B =            3
    f_0_3_t f_0_3_3 ( f_0_2_3, f_0_2_4, f_0_1_3, f_0_1_4, CDtemp, ABcom, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            0  B =            4
    f_0_4_t f_0_4_2 ( f_0_3_2, f_0_3_3, f_0_2_2, f_0_2_3, CDtemp, ABcom, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            0  B =            5
    f_0_5_t f_0_5_1 ( f_0_4_1, f_0_4_2, f_0_3_1, f_0_3_2, CDtemp, ABcom, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            0  B =            6
    f_0_6_t f_0_6_0 ( f_0_5_0, f_0_5_1, f_0_4_0, f_0_4_1, CDtemp, ABcom, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            0  B =            1
    f_0_1_t f_0_1_6 ( VY( 0, 0, 6 ), VY( 0, 0, 7 ), Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            0  B =            2
    f_0_2_t f_0_2_5 ( f_0_1_5, f_0_1_6, VY( 0, 0, 5 ), VY( 0, 0, 6 ), CDtemp, ABcom, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            0  B =            3
    f_0_3_t f_0_3_4 ( f_0_2_4, f_0_2_5, f_0_1_4, f_0_1_5, CDtemp, ABcom, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            0  B =            4
    f_0_4_t f_0_4_3 ( f_0_3_3, f_0_3_4, f_0_2_3, f_0_2_4, CDtemp, ABcom, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            0  B =            5
    f_0_5_t f_0_5_2 ( f_0_4_2, f_0_4_3, f_0_3_2, f_0_3_3, CDtemp, ABcom, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            0  B =            6
    f_0_6_t f_0_6_1 ( f_0_5_1, f_0_5_2, f_0_4_1, f_0_4_2, CDtemp, ABcom, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            0  B =            7
    f_0_7_t f_0_7_0 ( f_0_6_0, f_0_6_1, f_0_5_0, f_0_5_1, CDtemp, ABcom, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // WRITE LAST FOR I =            0  J=           7
    LOCSTORE(store,  0, 84, STOREDIM, STOREDIM) += f_0_7_0.x_0_84 ;
    LOCSTORE(store,  0, 85, STOREDIM, STOREDIM) += f_0_7_0.x_0_85 ;
    LOCSTORE(store,  0, 86, STOREDIM, STOREDIM) += f_0_7_0.x_0_86 ;
    LOCSTORE(store,  0, 87, STOREDIM, STOREDIM) += f_0_7_0.x_0_87 ;
    LOCSTORE(store,  0, 88, STOREDIM, STOREDIM) += f_0_7_0.x_0_88 ;
    LOCSTORE(store,  0, 89, STOREDIM, STOREDIM) += f_0_7_0.x_0_89 ;
    LOCSTORE(store,  0, 90, STOREDIM, STOREDIM) += f_0_7_0.x_0_90 ;
    LOCSTORE(store,  0, 91, STOREDIM, STOREDIM) += f_0_7_0.x_0_91 ;
    LOCSTORE(store,  0, 92, STOREDIM, STOREDIM) += f_0_7_0.x_0_92 ;
    LOCSTORE(store,  0, 93, STOREDIM, STOREDIM) += f_0_7_0.x_0_93 ;
    LOCSTORE(store,  0, 94, STOREDIM, STOREDIM) += f_0_7_0.x_0_94 ;
    LOCSTORE(store,  0, 95, STOREDIM, STOREDIM) += f_0_7_0.x_0_95 ;
    LOCSTORE(store,  0, 96, STOREDIM, STOREDIM) += f_0_7_0.x_0_96 ;
    LOCSTORE(store,  0, 97, STOREDIM, STOREDIM) += f_0_7_0.x_0_97 ;
    LOCSTORE(store,  0, 98, STOREDIM, STOREDIM) += f_0_7_0.x_0_98 ;
    LOCSTORE(store,  0, 99, STOREDIM, STOREDIM) += f_0_7_0.x_0_99 ;
    LOCSTORE(store,  0,100, STOREDIM, STOREDIM) += f_0_7_0.x_0_100 ;
    LOCSTORE(store,  0,101, STOREDIM, STOREDIM) += f_0_7_0.x_0_101 ;
    LOCSTORE(store,  0,102, STOREDIM, STOREDIM) += f_0_7_0.x_0_102 ;
    LOCSTORE(store,  0,103, STOREDIM, STOREDIM) += f_0_7_0.x_0_103 ;
    LOCSTORE(store,  0,104, STOREDIM, STOREDIM) += f_0_7_0.x_0_104 ;
    LOCSTORE(store,  0,105, STOREDIM, STOREDIM) += f_0_7_0.x_0_105 ;
    LOCSTORE(store,  0,106, STOREDIM, STOREDIM) += f_0_7_0.x_0_106 ;
    LOCSTORE(store,  0,107, STOREDIM, STOREDIM) += f_0_7_0.x_0_107 ;
    LOCSTORE(store,  0,108, STOREDIM, STOREDIM) += f_0_7_0.x_0_108 ;
    LOCSTORE(store,  0,109, STOREDIM, STOREDIM) += f_0_7_0.x_0_109 ;
    LOCSTORE(store,  0,110, STOREDIM, STOREDIM) += f_0_7_0.x_0_110 ;
    LOCSTORE(store,  0,111, STOREDIM, STOREDIM) += f_0_7_0.x_0_111 ;
    LOCSTORE(store,  0,112, STOREDIM, STOREDIM) += f_0_7_0.x_0_112 ;
    LOCSTORE(store,  0,113, STOREDIM, STOREDIM) += f_0_7_0.x_0_113 ;
    LOCSTORE(store,  0,114, STOREDIM, STOREDIM) += f_0_7_0.x_0_114 ;
    LOCSTORE(store,  0,115, STOREDIM, STOREDIM) += f_0_7_0.x_0_115 ;
    LOCSTORE(store,  0,116, STOREDIM, STOREDIM) += f_0_7_0.x_0_116 ;
    LOCSTORE(store,  0,117, STOREDIM, STOREDIM) += f_0_7_0.x_0_117 ;
    LOCSTORE(store,  0,118, STOREDIM, STOREDIM) += f_0_7_0.x_0_118 ;
    LOCSTORE(store,  0,119, STOREDIM, STOREDIM) += f_0_7_0.x_0_119 ;
}
