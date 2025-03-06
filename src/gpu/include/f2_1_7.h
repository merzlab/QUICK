__device__ __inline__ void h2_1_7(QUICKDouble* YVerticalTemp, QUICKDouble* store,
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

    // call for L =            0  B =            1
    f_0_1_t f_0_1_7 ( VY( 0, 0, 7 ), VY( 0, 0, 8 ), Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            0  B =            2
    f_0_2_t f_0_2_6 ( f_0_1_6, f_0_1_7, VY( 0, 0, 6 ), VY( 0, 0, 7 ), CDtemp, ABcom, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            0  B =            3
    f_0_3_t f_0_3_5 ( f_0_2_5, f_0_2_6, f_0_1_5, f_0_1_6, CDtemp, ABcom, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            0  B =            4
    f_0_4_t f_0_4_4 ( f_0_3_4, f_0_3_5, f_0_2_4, f_0_2_5, CDtemp, ABcom, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            0  B =            5
    f_0_5_t f_0_5_3 ( f_0_4_3, f_0_4_4, f_0_3_3, f_0_3_4, CDtemp, ABcom, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            0  B =            6
    f_0_6_t f_0_6_2 ( f_0_5_2, f_0_5_3, f_0_4_2, f_0_4_3, CDtemp, ABcom, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            0  B =            7
    f_0_7_t f_0_7_1 ( f_0_6_1, f_0_6_2, f_0_5_1, f_0_5_2, CDtemp, ABcom, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            1  B =            7
    f_1_7_t f_1_7_0 ( f_0_7_0,  f_0_7_1,  f_0_6_1, ABCDtemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // WRITE LAST FOR I =            1  J=           7
    LOCSTORE(store,  1, 84, STOREDIM, STOREDIM) = f_1_7_0.x_1_84 ;
    LOCSTORE(store,  1, 85, STOREDIM, STOREDIM) = f_1_7_0.x_1_85 ;
    LOCSTORE(store,  1, 86, STOREDIM, STOREDIM) = f_1_7_0.x_1_86 ;
    LOCSTORE(store,  1, 87, STOREDIM, STOREDIM) = f_1_7_0.x_1_87 ;
    LOCSTORE(store,  1, 88, STOREDIM, STOREDIM) = f_1_7_0.x_1_88 ;
    LOCSTORE(store,  1, 89, STOREDIM, STOREDIM) = f_1_7_0.x_1_89 ;
    LOCSTORE(store,  1, 90, STOREDIM, STOREDIM) = f_1_7_0.x_1_90 ;
    LOCSTORE(store,  1, 91, STOREDIM, STOREDIM) = f_1_7_0.x_1_91 ;
    LOCSTORE(store,  1, 92, STOREDIM, STOREDIM) = f_1_7_0.x_1_92 ;
    LOCSTORE(store,  1, 93, STOREDIM, STOREDIM) = f_1_7_0.x_1_93 ;
    LOCSTORE(store,  1, 94, STOREDIM, STOREDIM) = f_1_7_0.x_1_94 ;
    LOCSTORE(store,  1, 95, STOREDIM, STOREDIM) = f_1_7_0.x_1_95 ;
    LOCSTORE(store,  1, 96, STOREDIM, STOREDIM) = f_1_7_0.x_1_96 ;
    LOCSTORE(store,  1, 97, STOREDIM, STOREDIM) = f_1_7_0.x_1_97 ;
    LOCSTORE(store,  1, 98, STOREDIM, STOREDIM) = f_1_7_0.x_1_98 ;
    LOCSTORE(store,  1, 99, STOREDIM, STOREDIM) = f_1_7_0.x_1_99 ;
    LOCSTORE(store,  1,100, STOREDIM, STOREDIM) = f_1_7_0.x_1_100 ;
    LOCSTORE(store,  1,101, STOREDIM, STOREDIM) = f_1_7_0.x_1_101 ;
    LOCSTORE(store,  1,102, STOREDIM, STOREDIM) = f_1_7_0.x_1_102 ;
    LOCSTORE(store,  1,103, STOREDIM, STOREDIM) = f_1_7_0.x_1_103 ;
    LOCSTORE(store,  1,104, STOREDIM, STOREDIM) = f_1_7_0.x_1_104 ;
    LOCSTORE(store,  1,105, STOREDIM, STOREDIM) = f_1_7_0.x_1_105 ;
    LOCSTORE(store,  1,106, STOREDIM, STOREDIM) = f_1_7_0.x_1_106 ;
    LOCSTORE(store,  1,107, STOREDIM, STOREDIM) = f_1_7_0.x_1_107 ;
    LOCSTORE(store,  1,108, STOREDIM, STOREDIM) = f_1_7_0.x_1_108 ;
    LOCSTORE(store,  1,109, STOREDIM, STOREDIM) = f_1_7_0.x_1_109 ;
    LOCSTORE(store,  1,110, STOREDIM, STOREDIM) = f_1_7_0.x_1_110 ;
    LOCSTORE(store,  1,111, STOREDIM, STOREDIM) = f_1_7_0.x_1_111 ;
    LOCSTORE(store,  1,112, STOREDIM, STOREDIM) = f_1_7_0.x_1_112 ;
    LOCSTORE(store,  1,113, STOREDIM, STOREDIM) = f_1_7_0.x_1_113 ;
    LOCSTORE(store,  1,114, STOREDIM, STOREDIM) = f_1_7_0.x_1_114 ;
    LOCSTORE(store,  1,115, STOREDIM, STOREDIM) = f_1_7_0.x_1_115 ;
    LOCSTORE(store,  1,116, STOREDIM, STOREDIM) = f_1_7_0.x_1_116 ;
    LOCSTORE(store,  1,117, STOREDIM, STOREDIM) = f_1_7_0.x_1_117 ;
    LOCSTORE(store,  1,118, STOREDIM, STOREDIM) = f_1_7_0.x_1_118 ;
    LOCSTORE(store,  1,119, STOREDIM, STOREDIM) = f_1_7_0.x_1_119 ;
    LOCSTORE(store,  2, 84, STOREDIM, STOREDIM) = f_1_7_0.x_2_84 ;
    LOCSTORE(store,  2, 85, STOREDIM, STOREDIM) = f_1_7_0.x_2_85 ;
    LOCSTORE(store,  2, 86, STOREDIM, STOREDIM) = f_1_7_0.x_2_86 ;
    LOCSTORE(store,  2, 87, STOREDIM, STOREDIM) = f_1_7_0.x_2_87 ;
    LOCSTORE(store,  2, 88, STOREDIM, STOREDIM) = f_1_7_0.x_2_88 ;
    LOCSTORE(store,  2, 89, STOREDIM, STOREDIM) = f_1_7_0.x_2_89 ;
    LOCSTORE(store,  2, 90, STOREDIM, STOREDIM) = f_1_7_0.x_2_90 ;
    LOCSTORE(store,  2, 91, STOREDIM, STOREDIM) = f_1_7_0.x_2_91 ;
    LOCSTORE(store,  2, 92, STOREDIM, STOREDIM) = f_1_7_0.x_2_92 ;
    LOCSTORE(store,  2, 93, STOREDIM, STOREDIM) = f_1_7_0.x_2_93 ;
    LOCSTORE(store,  2, 94, STOREDIM, STOREDIM) = f_1_7_0.x_2_94 ;
    LOCSTORE(store,  2, 95, STOREDIM, STOREDIM) = f_1_7_0.x_2_95 ;
    LOCSTORE(store,  2, 96, STOREDIM, STOREDIM) = f_1_7_0.x_2_96 ;
    LOCSTORE(store,  2, 97, STOREDIM, STOREDIM) = f_1_7_0.x_2_97 ;
    LOCSTORE(store,  2, 98, STOREDIM, STOREDIM) = f_1_7_0.x_2_98 ;
    LOCSTORE(store,  2, 99, STOREDIM, STOREDIM) = f_1_7_0.x_2_99 ;
    LOCSTORE(store,  2,100, STOREDIM, STOREDIM) = f_1_7_0.x_2_100 ;
    LOCSTORE(store,  2,101, STOREDIM, STOREDIM) = f_1_7_0.x_2_101 ;
    LOCSTORE(store,  2,102, STOREDIM, STOREDIM) = f_1_7_0.x_2_102 ;
    LOCSTORE(store,  2,103, STOREDIM, STOREDIM) = f_1_7_0.x_2_103 ;
    LOCSTORE(store,  2,104, STOREDIM, STOREDIM) = f_1_7_0.x_2_104 ;
    LOCSTORE(store,  2,105, STOREDIM, STOREDIM) = f_1_7_0.x_2_105 ;
    LOCSTORE(store,  2,106, STOREDIM, STOREDIM) = f_1_7_0.x_2_106 ;
    LOCSTORE(store,  2,107, STOREDIM, STOREDIM) = f_1_7_0.x_2_107 ;
    LOCSTORE(store,  2,108, STOREDIM, STOREDIM) = f_1_7_0.x_2_108 ;
    LOCSTORE(store,  2,109, STOREDIM, STOREDIM) = f_1_7_0.x_2_109 ;
    LOCSTORE(store,  2,110, STOREDIM, STOREDIM) = f_1_7_0.x_2_110 ;
    LOCSTORE(store,  2,111, STOREDIM, STOREDIM) = f_1_7_0.x_2_111 ;
    LOCSTORE(store,  2,112, STOREDIM, STOREDIM) = f_1_7_0.x_2_112 ;
    LOCSTORE(store,  2,113, STOREDIM, STOREDIM) = f_1_7_0.x_2_113 ;
    LOCSTORE(store,  2,114, STOREDIM, STOREDIM) = f_1_7_0.x_2_114 ;
    LOCSTORE(store,  2,115, STOREDIM, STOREDIM) = f_1_7_0.x_2_115 ;
    LOCSTORE(store,  2,116, STOREDIM, STOREDIM) = f_1_7_0.x_2_116 ;
    LOCSTORE(store,  2,117, STOREDIM, STOREDIM) = f_1_7_0.x_2_117 ;
    LOCSTORE(store,  2,118, STOREDIM, STOREDIM) = f_1_7_0.x_2_118 ;
    LOCSTORE(store,  2,119, STOREDIM, STOREDIM) = f_1_7_0.x_2_119 ;
    LOCSTORE(store,  3, 84, STOREDIM, STOREDIM) = f_1_7_0.x_3_84 ;
    LOCSTORE(store,  3, 85, STOREDIM, STOREDIM) = f_1_7_0.x_3_85 ;
    LOCSTORE(store,  3, 86, STOREDIM, STOREDIM) = f_1_7_0.x_3_86 ;
    LOCSTORE(store,  3, 87, STOREDIM, STOREDIM) = f_1_7_0.x_3_87 ;
    LOCSTORE(store,  3, 88, STOREDIM, STOREDIM) = f_1_7_0.x_3_88 ;
    LOCSTORE(store,  3, 89, STOREDIM, STOREDIM) = f_1_7_0.x_3_89 ;
    LOCSTORE(store,  3, 90, STOREDIM, STOREDIM) = f_1_7_0.x_3_90 ;
    LOCSTORE(store,  3, 91, STOREDIM, STOREDIM) = f_1_7_0.x_3_91 ;
    LOCSTORE(store,  3, 92, STOREDIM, STOREDIM) = f_1_7_0.x_3_92 ;
    LOCSTORE(store,  3, 93, STOREDIM, STOREDIM) = f_1_7_0.x_3_93 ;
    LOCSTORE(store,  3, 94, STOREDIM, STOREDIM) = f_1_7_0.x_3_94 ;
    LOCSTORE(store,  3, 95, STOREDIM, STOREDIM) = f_1_7_0.x_3_95 ;
    LOCSTORE(store,  3, 96, STOREDIM, STOREDIM) = f_1_7_0.x_3_96 ;
    LOCSTORE(store,  3, 97, STOREDIM, STOREDIM) = f_1_7_0.x_3_97 ;
    LOCSTORE(store,  3, 98, STOREDIM, STOREDIM) = f_1_7_0.x_3_98 ;
    LOCSTORE(store,  3, 99, STOREDIM, STOREDIM) = f_1_7_0.x_3_99 ;
    LOCSTORE(store,  3,100, STOREDIM, STOREDIM) = f_1_7_0.x_3_100 ;
    LOCSTORE(store,  3,101, STOREDIM, STOREDIM) = f_1_7_0.x_3_101 ;
    LOCSTORE(store,  3,102, STOREDIM, STOREDIM) = f_1_7_0.x_3_102 ;
    LOCSTORE(store,  3,103, STOREDIM, STOREDIM) = f_1_7_0.x_3_103 ;
    LOCSTORE(store,  3,104, STOREDIM, STOREDIM) = f_1_7_0.x_3_104 ;
    LOCSTORE(store,  3,105, STOREDIM, STOREDIM) = f_1_7_0.x_3_105 ;
    LOCSTORE(store,  3,106, STOREDIM, STOREDIM) = f_1_7_0.x_3_106 ;
    LOCSTORE(store,  3,107, STOREDIM, STOREDIM) = f_1_7_0.x_3_107 ;
    LOCSTORE(store,  3,108, STOREDIM, STOREDIM) = f_1_7_0.x_3_108 ;
    LOCSTORE(store,  3,109, STOREDIM, STOREDIM) = f_1_7_0.x_3_109 ;
    LOCSTORE(store,  3,110, STOREDIM, STOREDIM) = f_1_7_0.x_3_110 ;
    LOCSTORE(store,  3,111, STOREDIM, STOREDIM) = f_1_7_0.x_3_111 ;
    LOCSTORE(store,  3,112, STOREDIM, STOREDIM) = f_1_7_0.x_3_112 ;
    LOCSTORE(store,  3,113, STOREDIM, STOREDIM) = f_1_7_0.x_3_113 ;
    LOCSTORE(store,  3,114, STOREDIM, STOREDIM) = f_1_7_0.x_3_114 ;
    LOCSTORE(store,  3,115, STOREDIM, STOREDIM) = f_1_7_0.x_3_115 ;
    LOCSTORE(store,  3,116, STOREDIM, STOREDIM) = f_1_7_0.x_3_116 ;
    LOCSTORE(store,  3,117, STOREDIM, STOREDIM) = f_1_7_0.x_3_117 ;
    LOCSTORE(store,  3,118, STOREDIM, STOREDIM) = f_1_7_0.x_3_118 ;
    LOCSTORE(store,  3,119, STOREDIM, STOREDIM) = f_1_7_0.x_3_119 ;
}
