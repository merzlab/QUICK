__device__ __inline__  void h_0_6(QUICKDouble* YVerticalTemp, QUICKDouble* store,
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

    // WRITE LAST FOR I =            0  J=           6
    LOC2(store,  0, 56, STOREDIM, STOREDIM) += f_0_6_0.x_0_56 ;
    LOC2(store,  0, 57, STOREDIM, STOREDIM) += f_0_6_0.x_0_57 ;
    LOC2(store,  0, 58, STOREDIM, STOREDIM) += f_0_6_0.x_0_58 ;
    LOC2(store,  0, 59, STOREDIM, STOREDIM) += f_0_6_0.x_0_59 ;
    LOC2(store,  0, 60, STOREDIM, STOREDIM) += f_0_6_0.x_0_60 ;
    LOC2(store,  0, 61, STOREDIM, STOREDIM) += f_0_6_0.x_0_61 ;
    LOC2(store,  0, 62, STOREDIM, STOREDIM) += f_0_6_0.x_0_62 ;
    LOC2(store,  0, 63, STOREDIM, STOREDIM) += f_0_6_0.x_0_63 ;
    LOC2(store,  0, 64, STOREDIM, STOREDIM) += f_0_6_0.x_0_64 ;
    LOC2(store,  0, 65, STOREDIM, STOREDIM) += f_0_6_0.x_0_65 ;
    LOC2(store,  0, 66, STOREDIM, STOREDIM) += f_0_6_0.x_0_66 ;
    LOC2(store,  0, 67, STOREDIM, STOREDIM) += f_0_6_0.x_0_67 ;
    LOC2(store,  0, 68, STOREDIM, STOREDIM) += f_0_6_0.x_0_68 ;
    LOC2(store,  0, 69, STOREDIM, STOREDIM) += f_0_6_0.x_0_69 ;
    LOC2(store,  0, 70, STOREDIM, STOREDIM) += f_0_6_0.x_0_70 ;
    LOC2(store,  0, 71, STOREDIM, STOREDIM) += f_0_6_0.x_0_71 ;
    LOC2(store,  0, 72, STOREDIM, STOREDIM) += f_0_6_0.x_0_72 ;
    LOC2(store,  0, 73, STOREDIM, STOREDIM) += f_0_6_0.x_0_73 ;
    LOC2(store,  0, 74, STOREDIM, STOREDIM) += f_0_6_0.x_0_74 ;
    LOC2(store,  0, 75, STOREDIM, STOREDIM) += f_0_6_0.x_0_75 ;
    LOC2(store,  0, 76, STOREDIM, STOREDIM) += f_0_6_0.x_0_76 ;
    LOC2(store,  0, 77, STOREDIM, STOREDIM) += f_0_6_0.x_0_77 ;
    LOC2(store,  0, 78, STOREDIM, STOREDIM) += f_0_6_0.x_0_78 ;
    LOC2(store,  0, 79, STOREDIM, STOREDIM) += f_0_6_0.x_0_79 ;
    LOC2(store,  0, 80, STOREDIM, STOREDIM) += f_0_6_0.x_0_80 ;
    LOC2(store,  0, 81, STOREDIM, STOREDIM) += f_0_6_0.x_0_81 ;
    LOC2(store,  0, 82, STOREDIM, STOREDIM) += f_0_6_0.x_0_82 ;
    LOC2(store,  0, 83, STOREDIM, STOREDIM) += f_0_6_0.x_0_83 ;
}
