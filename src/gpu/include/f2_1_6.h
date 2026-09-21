__device__ __inline__  void h2_1_6(QUICKDouble* YVerticalTemp, QUICKDouble* store,
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

    // call for L =            1  B =            6
    f_1_6_t f_1_6_0 ( f_0_6_0,  f_0_6_1,  f_0_5_1, ABCDtemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // WRITE LAST FOR I =            1  J=           6
    LOCSTORE(store,  1, 56, STOREDIM, STOREDIM) = f_1_6_0.x_1_56 ;
    LOCSTORE(store,  1, 57, STOREDIM, STOREDIM) = f_1_6_0.x_1_57 ;
    LOCSTORE(store,  1, 58, STOREDIM, STOREDIM) = f_1_6_0.x_1_58 ;
    LOCSTORE(store,  1, 59, STOREDIM, STOREDIM) = f_1_6_0.x_1_59 ;
    LOCSTORE(store,  1, 60, STOREDIM, STOREDIM) = f_1_6_0.x_1_60 ;
    LOCSTORE(store,  1, 61, STOREDIM, STOREDIM) = f_1_6_0.x_1_61 ;
    LOCSTORE(store,  1, 62, STOREDIM, STOREDIM) = f_1_6_0.x_1_62 ;
    LOCSTORE(store,  1, 63, STOREDIM, STOREDIM) = f_1_6_0.x_1_63 ;
    LOCSTORE(store,  1, 64, STOREDIM, STOREDIM) = f_1_6_0.x_1_64 ;
    LOCSTORE(store,  1, 65, STOREDIM, STOREDIM) = f_1_6_0.x_1_65 ;
    LOCSTORE(store,  1, 66, STOREDIM, STOREDIM) = f_1_6_0.x_1_66 ;
    LOCSTORE(store,  1, 67, STOREDIM, STOREDIM) = f_1_6_0.x_1_67 ;
    LOCSTORE(store,  1, 68, STOREDIM, STOREDIM) = f_1_6_0.x_1_68 ;
    LOCSTORE(store,  1, 69, STOREDIM, STOREDIM) = f_1_6_0.x_1_69 ;
    LOCSTORE(store,  1, 70, STOREDIM, STOREDIM) = f_1_6_0.x_1_70 ;
    LOCSTORE(store,  1, 71, STOREDIM, STOREDIM) = f_1_6_0.x_1_71 ;
    LOCSTORE(store,  1, 72, STOREDIM, STOREDIM) = f_1_6_0.x_1_72 ;
    LOCSTORE(store,  1, 73, STOREDIM, STOREDIM) = f_1_6_0.x_1_73 ;
    LOCSTORE(store,  1, 74, STOREDIM, STOREDIM) = f_1_6_0.x_1_74 ;
    LOCSTORE(store,  1, 75, STOREDIM, STOREDIM) = f_1_6_0.x_1_75 ;
    LOCSTORE(store,  1, 76, STOREDIM, STOREDIM) = f_1_6_0.x_1_76 ;
    LOCSTORE(store,  1, 77, STOREDIM, STOREDIM) = f_1_6_0.x_1_77 ;
    LOCSTORE(store,  1, 78, STOREDIM, STOREDIM) = f_1_6_0.x_1_78 ;
    LOCSTORE(store,  1, 79, STOREDIM, STOREDIM) = f_1_6_0.x_1_79 ;
    LOCSTORE(store,  1, 80, STOREDIM, STOREDIM) = f_1_6_0.x_1_80 ;
    LOCSTORE(store,  1, 81, STOREDIM, STOREDIM) = f_1_6_0.x_1_81 ;
    LOCSTORE(store,  1, 82, STOREDIM, STOREDIM) = f_1_6_0.x_1_82 ;
    LOCSTORE(store,  1, 83, STOREDIM, STOREDIM) = f_1_6_0.x_1_83 ;
    LOCSTORE(store,  2, 56, STOREDIM, STOREDIM) = f_1_6_0.x_2_56 ;
    LOCSTORE(store,  2, 57, STOREDIM, STOREDIM) = f_1_6_0.x_2_57 ;
    LOCSTORE(store,  2, 58, STOREDIM, STOREDIM) = f_1_6_0.x_2_58 ;
    LOCSTORE(store,  2, 59, STOREDIM, STOREDIM) = f_1_6_0.x_2_59 ;
    LOCSTORE(store,  2, 60, STOREDIM, STOREDIM) = f_1_6_0.x_2_60 ;
    LOCSTORE(store,  2, 61, STOREDIM, STOREDIM) = f_1_6_0.x_2_61 ;
    LOCSTORE(store,  2, 62, STOREDIM, STOREDIM) = f_1_6_0.x_2_62 ;
    LOCSTORE(store,  2, 63, STOREDIM, STOREDIM) = f_1_6_0.x_2_63 ;
    LOCSTORE(store,  2, 64, STOREDIM, STOREDIM) = f_1_6_0.x_2_64 ;
    LOCSTORE(store,  2, 65, STOREDIM, STOREDIM) = f_1_6_0.x_2_65 ;
    LOCSTORE(store,  2, 66, STOREDIM, STOREDIM) = f_1_6_0.x_2_66 ;
    LOCSTORE(store,  2, 67, STOREDIM, STOREDIM) = f_1_6_0.x_2_67 ;
    LOCSTORE(store,  2, 68, STOREDIM, STOREDIM) = f_1_6_0.x_2_68 ;
    LOCSTORE(store,  2, 69, STOREDIM, STOREDIM) = f_1_6_0.x_2_69 ;
    LOCSTORE(store,  2, 70, STOREDIM, STOREDIM) = f_1_6_0.x_2_70 ;
    LOCSTORE(store,  2, 71, STOREDIM, STOREDIM) = f_1_6_0.x_2_71 ;
    LOCSTORE(store,  2, 72, STOREDIM, STOREDIM) = f_1_6_0.x_2_72 ;
    LOCSTORE(store,  2, 73, STOREDIM, STOREDIM) = f_1_6_0.x_2_73 ;
    LOCSTORE(store,  2, 74, STOREDIM, STOREDIM) = f_1_6_0.x_2_74 ;
    LOCSTORE(store,  2, 75, STOREDIM, STOREDIM) = f_1_6_0.x_2_75 ;
    LOCSTORE(store,  2, 76, STOREDIM, STOREDIM) = f_1_6_0.x_2_76 ;
    LOCSTORE(store,  2, 77, STOREDIM, STOREDIM) = f_1_6_0.x_2_77 ;
    LOCSTORE(store,  2, 78, STOREDIM, STOREDIM) = f_1_6_0.x_2_78 ;
    LOCSTORE(store,  2, 79, STOREDIM, STOREDIM) = f_1_6_0.x_2_79 ;
    LOCSTORE(store,  2, 80, STOREDIM, STOREDIM) = f_1_6_0.x_2_80 ;
    LOCSTORE(store,  2, 81, STOREDIM, STOREDIM) = f_1_6_0.x_2_81 ;
    LOCSTORE(store,  2, 82, STOREDIM, STOREDIM) = f_1_6_0.x_2_82 ;
    LOCSTORE(store,  2, 83, STOREDIM, STOREDIM) = f_1_6_0.x_2_83 ;
    LOCSTORE(store,  3, 56, STOREDIM, STOREDIM) = f_1_6_0.x_3_56 ;
    LOCSTORE(store,  3, 57, STOREDIM, STOREDIM) = f_1_6_0.x_3_57 ;
    LOCSTORE(store,  3, 58, STOREDIM, STOREDIM) = f_1_6_0.x_3_58 ;
    LOCSTORE(store,  3, 59, STOREDIM, STOREDIM) = f_1_6_0.x_3_59 ;
    LOCSTORE(store,  3, 60, STOREDIM, STOREDIM) = f_1_6_0.x_3_60 ;
    LOCSTORE(store,  3, 61, STOREDIM, STOREDIM) = f_1_6_0.x_3_61 ;
    LOCSTORE(store,  3, 62, STOREDIM, STOREDIM) = f_1_6_0.x_3_62 ;
    LOCSTORE(store,  3, 63, STOREDIM, STOREDIM) = f_1_6_0.x_3_63 ;
    LOCSTORE(store,  3, 64, STOREDIM, STOREDIM) = f_1_6_0.x_3_64 ;
    LOCSTORE(store,  3, 65, STOREDIM, STOREDIM) = f_1_6_0.x_3_65 ;
    LOCSTORE(store,  3, 66, STOREDIM, STOREDIM) = f_1_6_0.x_3_66 ;
    LOCSTORE(store,  3, 67, STOREDIM, STOREDIM) = f_1_6_0.x_3_67 ;
    LOCSTORE(store,  3, 68, STOREDIM, STOREDIM) = f_1_6_0.x_3_68 ;
    LOCSTORE(store,  3, 69, STOREDIM, STOREDIM) = f_1_6_0.x_3_69 ;
    LOCSTORE(store,  3, 70, STOREDIM, STOREDIM) = f_1_6_0.x_3_70 ;
    LOCSTORE(store,  3, 71, STOREDIM, STOREDIM) = f_1_6_0.x_3_71 ;
    LOCSTORE(store,  3, 72, STOREDIM, STOREDIM) = f_1_6_0.x_3_72 ;
    LOCSTORE(store,  3, 73, STOREDIM, STOREDIM) = f_1_6_0.x_3_73 ;
    LOCSTORE(store,  3, 74, STOREDIM, STOREDIM) = f_1_6_0.x_3_74 ;
    LOCSTORE(store,  3, 75, STOREDIM, STOREDIM) = f_1_6_0.x_3_75 ;
    LOCSTORE(store,  3, 76, STOREDIM, STOREDIM) = f_1_6_0.x_3_76 ;
    LOCSTORE(store,  3, 77, STOREDIM, STOREDIM) = f_1_6_0.x_3_77 ;
    LOCSTORE(store,  3, 78, STOREDIM, STOREDIM) = f_1_6_0.x_3_78 ;
    LOCSTORE(store,  3, 79, STOREDIM, STOREDIM) = f_1_6_0.x_3_79 ;
    LOCSTORE(store,  3, 80, STOREDIM, STOREDIM) = f_1_6_0.x_3_80 ;
    LOCSTORE(store,  3, 81, STOREDIM, STOREDIM) = f_1_6_0.x_3_81 ;
    LOCSTORE(store,  3, 82, STOREDIM, STOREDIM) = f_1_6_0.x_3_82 ;
    LOCSTORE(store,  3, 83, STOREDIM, STOREDIM) = f_1_6_0.x_3_83 ;
}
