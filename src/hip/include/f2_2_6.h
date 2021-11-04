__device__ __inline__  void h2_2_6(QUICKDouble* YVerticalTemp, QUICKDouble* store,
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

    // call for L =            1  B =            6
    f_1_6_t f_1_6_1 ( f_0_6_1,  f_0_6_2,  f_0_5_2, ABCDtemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            1  B =            5
    f_1_5_t f_1_5_1 ( f_0_5_1,  f_0_5_2,  f_0_4_2, ABCDtemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            2  B =            6
    f_2_6_t f_2_6_0 ( f_1_6_0,  f_1_6_1, f_0_6_0, f_0_6_1, ABtemp, CDcom, f_1_5_1, ABCDtemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // WRITE LAST FOR I =            2  J=           6
    LOCSTORE(store,  4, 56, STOREDIM, STOREDIM) = f_2_6_0.x_4_56 ;
    LOCSTORE(store,  4, 57, STOREDIM, STOREDIM) = f_2_6_0.x_4_57 ;
    LOCSTORE(store,  4, 58, STOREDIM, STOREDIM) = f_2_6_0.x_4_58 ;
    LOCSTORE(store,  4, 59, STOREDIM, STOREDIM) = f_2_6_0.x_4_59 ;
    LOCSTORE(store,  4, 60, STOREDIM, STOREDIM) = f_2_6_0.x_4_60 ;
    LOCSTORE(store,  4, 61, STOREDIM, STOREDIM) = f_2_6_0.x_4_61 ;
    LOCSTORE(store,  4, 62, STOREDIM, STOREDIM) = f_2_6_0.x_4_62 ;
    LOCSTORE(store,  4, 63, STOREDIM, STOREDIM) = f_2_6_0.x_4_63 ;
    LOCSTORE(store,  4, 64, STOREDIM, STOREDIM) = f_2_6_0.x_4_64 ;
    LOCSTORE(store,  4, 65, STOREDIM, STOREDIM) = f_2_6_0.x_4_65 ;
    LOCSTORE(store,  4, 66, STOREDIM, STOREDIM) = f_2_6_0.x_4_66 ;
    LOCSTORE(store,  4, 67, STOREDIM, STOREDIM) = f_2_6_0.x_4_67 ;
    LOCSTORE(store,  4, 68, STOREDIM, STOREDIM) = f_2_6_0.x_4_68 ;
    LOCSTORE(store,  4, 69, STOREDIM, STOREDIM) = f_2_6_0.x_4_69 ;
    LOCSTORE(store,  4, 70, STOREDIM, STOREDIM) = f_2_6_0.x_4_70 ;
    LOCSTORE(store,  4, 71, STOREDIM, STOREDIM) = f_2_6_0.x_4_71 ;
    LOCSTORE(store,  4, 72, STOREDIM, STOREDIM) = f_2_6_0.x_4_72 ;
    LOCSTORE(store,  4, 73, STOREDIM, STOREDIM) = f_2_6_0.x_4_73 ;
    LOCSTORE(store,  4, 74, STOREDIM, STOREDIM) = f_2_6_0.x_4_74 ;
    LOCSTORE(store,  4, 75, STOREDIM, STOREDIM) = f_2_6_0.x_4_75 ;
    LOCSTORE(store,  4, 76, STOREDIM, STOREDIM) = f_2_6_0.x_4_76 ;
    LOCSTORE(store,  4, 77, STOREDIM, STOREDIM) = f_2_6_0.x_4_77 ;
    LOCSTORE(store,  4, 78, STOREDIM, STOREDIM) = f_2_6_0.x_4_78 ;
    LOCSTORE(store,  4, 79, STOREDIM, STOREDIM) = f_2_6_0.x_4_79 ;
    LOCSTORE(store,  4, 80, STOREDIM, STOREDIM) = f_2_6_0.x_4_80 ;
    LOCSTORE(store,  4, 81, STOREDIM, STOREDIM) = f_2_6_0.x_4_81 ;
    LOCSTORE(store,  4, 82, STOREDIM, STOREDIM) = f_2_6_0.x_4_82 ;
    LOCSTORE(store,  4, 83, STOREDIM, STOREDIM) = f_2_6_0.x_4_83 ;
    LOCSTORE(store,  5, 56, STOREDIM, STOREDIM) = f_2_6_0.x_5_56 ;
    LOCSTORE(store,  5, 57, STOREDIM, STOREDIM) = f_2_6_0.x_5_57 ;
    LOCSTORE(store,  5, 58, STOREDIM, STOREDIM) = f_2_6_0.x_5_58 ;
    LOCSTORE(store,  5, 59, STOREDIM, STOREDIM) = f_2_6_0.x_5_59 ;
    LOCSTORE(store,  5, 60, STOREDIM, STOREDIM) = f_2_6_0.x_5_60 ;
    LOCSTORE(store,  5, 61, STOREDIM, STOREDIM) = f_2_6_0.x_5_61 ;
    LOCSTORE(store,  5, 62, STOREDIM, STOREDIM) = f_2_6_0.x_5_62 ;
    LOCSTORE(store,  5, 63, STOREDIM, STOREDIM) = f_2_6_0.x_5_63 ;
    LOCSTORE(store,  5, 64, STOREDIM, STOREDIM) = f_2_6_0.x_5_64 ;
    LOCSTORE(store,  5, 65, STOREDIM, STOREDIM) = f_2_6_0.x_5_65 ;
    LOCSTORE(store,  5, 66, STOREDIM, STOREDIM) = f_2_6_0.x_5_66 ;
    LOCSTORE(store,  5, 67, STOREDIM, STOREDIM) = f_2_6_0.x_5_67 ;
    LOCSTORE(store,  5, 68, STOREDIM, STOREDIM) = f_2_6_0.x_5_68 ;
    LOCSTORE(store,  5, 69, STOREDIM, STOREDIM) = f_2_6_0.x_5_69 ;
    LOCSTORE(store,  5, 70, STOREDIM, STOREDIM) = f_2_6_0.x_5_70 ;
    LOCSTORE(store,  5, 71, STOREDIM, STOREDIM) = f_2_6_0.x_5_71 ;
    LOCSTORE(store,  5, 72, STOREDIM, STOREDIM) = f_2_6_0.x_5_72 ;
    LOCSTORE(store,  5, 73, STOREDIM, STOREDIM) = f_2_6_0.x_5_73 ;
    LOCSTORE(store,  5, 74, STOREDIM, STOREDIM) = f_2_6_0.x_5_74 ;
    LOCSTORE(store,  5, 75, STOREDIM, STOREDIM) = f_2_6_0.x_5_75 ;
    LOCSTORE(store,  5, 76, STOREDIM, STOREDIM) = f_2_6_0.x_5_76 ;
    LOCSTORE(store,  5, 77, STOREDIM, STOREDIM) = f_2_6_0.x_5_77 ;
    LOCSTORE(store,  5, 78, STOREDIM, STOREDIM) = f_2_6_0.x_5_78 ;
    LOCSTORE(store,  5, 79, STOREDIM, STOREDIM) = f_2_6_0.x_5_79 ;
    LOCSTORE(store,  5, 80, STOREDIM, STOREDIM) = f_2_6_0.x_5_80 ;
    LOCSTORE(store,  5, 81, STOREDIM, STOREDIM) = f_2_6_0.x_5_81 ;
    LOCSTORE(store,  5, 82, STOREDIM, STOREDIM) = f_2_6_0.x_5_82 ;
    LOCSTORE(store,  5, 83, STOREDIM, STOREDIM) = f_2_6_0.x_5_83 ;
    LOCSTORE(store,  6, 56, STOREDIM, STOREDIM) = f_2_6_0.x_6_56 ;
    LOCSTORE(store,  6, 57, STOREDIM, STOREDIM) = f_2_6_0.x_6_57 ;
    LOCSTORE(store,  6, 58, STOREDIM, STOREDIM) = f_2_6_0.x_6_58 ;
    LOCSTORE(store,  6, 59, STOREDIM, STOREDIM) = f_2_6_0.x_6_59 ;
    LOCSTORE(store,  6, 60, STOREDIM, STOREDIM) = f_2_6_0.x_6_60 ;
    LOCSTORE(store,  6, 61, STOREDIM, STOREDIM) = f_2_6_0.x_6_61 ;
    LOCSTORE(store,  6, 62, STOREDIM, STOREDIM) = f_2_6_0.x_6_62 ;
    LOCSTORE(store,  6, 63, STOREDIM, STOREDIM) = f_2_6_0.x_6_63 ;
    LOCSTORE(store,  6, 64, STOREDIM, STOREDIM) = f_2_6_0.x_6_64 ;
    LOCSTORE(store,  6, 65, STOREDIM, STOREDIM) = f_2_6_0.x_6_65 ;
    LOCSTORE(store,  6, 66, STOREDIM, STOREDIM) = f_2_6_0.x_6_66 ;
    LOCSTORE(store,  6, 67, STOREDIM, STOREDIM) = f_2_6_0.x_6_67 ;
    LOCSTORE(store,  6, 68, STOREDIM, STOREDIM) = f_2_6_0.x_6_68 ;
    LOCSTORE(store,  6, 69, STOREDIM, STOREDIM) = f_2_6_0.x_6_69 ;
    LOCSTORE(store,  6, 70, STOREDIM, STOREDIM) = f_2_6_0.x_6_70 ;
    LOCSTORE(store,  6, 71, STOREDIM, STOREDIM) = f_2_6_0.x_6_71 ;
    LOCSTORE(store,  6, 72, STOREDIM, STOREDIM) = f_2_6_0.x_6_72 ;
    LOCSTORE(store,  6, 73, STOREDIM, STOREDIM) = f_2_6_0.x_6_73 ;
    LOCSTORE(store,  6, 74, STOREDIM, STOREDIM) = f_2_6_0.x_6_74 ;
    LOCSTORE(store,  6, 75, STOREDIM, STOREDIM) = f_2_6_0.x_6_75 ;
    LOCSTORE(store,  6, 76, STOREDIM, STOREDIM) = f_2_6_0.x_6_76 ;
    LOCSTORE(store,  6, 77, STOREDIM, STOREDIM) = f_2_6_0.x_6_77 ;
    LOCSTORE(store,  6, 78, STOREDIM, STOREDIM) = f_2_6_0.x_6_78 ;
    LOCSTORE(store,  6, 79, STOREDIM, STOREDIM) = f_2_6_0.x_6_79 ;
    LOCSTORE(store,  6, 80, STOREDIM, STOREDIM) = f_2_6_0.x_6_80 ;
    LOCSTORE(store,  6, 81, STOREDIM, STOREDIM) = f_2_6_0.x_6_81 ;
    LOCSTORE(store,  6, 82, STOREDIM, STOREDIM) = f_2_6_0.x_6_82 ;
    LOCSTORE(store,  6, 83, STOREDIM, STOREDIM) = f_2_6_0.x_6_83 ;
    LOCSTORE(store,  7, 56, STOREDIM, STOREDIM) = f_2_6_0.x_7_56 ;
    LOCSTORE(store,  7, 57, STOREDIM, STOREDIM) = f_2_6_0.x_7_57 ;
    LOCSTORE(store,  7, 58, STOREDIM, STOREDIM) = f_2_6_0.x_7_58 ;
    LOCSTORE(store,  7, 59, STOREDIM, STOREDIM) = f_2_6_0.x_7_59 ;
    LOCSTORE(store,  7, 60, STOREDIM, STOREDIM) = f_2_6_0.x_7_60 ;
    LOCSTORE(store,  7, 61, STOREDIM, STOREDIM) = f_2_6_0.x_7_61 ;
    LOCSTORE(store,  7, 62, STOREDIM, STOREDIM) = f_2_6_0.x_7_62 ;
    LOCSTORE(store,  7, 63, STOREDIM, STOREDIM) = f_2_6_0.x_7_63 ;
    LOCSTORE(store,  7, 64, STOREDIM, STOREDIM) = f_2_6_0.x_7_64 ;
    LOCSTORE(store,  7, 65, STOREDIM, STOREDIM) = f_2_6_0.x_7_65 ;
    LOCSTORE(store,  7, 66, STOREDIM, STOREDIM) = f_2_6_0.x_7_66 ;
    LOCSTORE(store,  7, 67, STOREDIM, STOREDIM) = f_2_6_0.x_7_67 ;
    LOCSTORE(store,  7, 68, STOREDIM, STOREDIM) = f_2_6_0.x_7_68 ;
    LOCSTORE(store,  7, 69, STOREDIM, STOREDIM) = f_2_6_0.x_7_69 ;
    LOCSTORE(store,  7, 70, STOREDIM, STOREDIM) = f_2_6_0.x_7_70 ;
    LOCSTORE(store,  7, 71, STOREDIM, STOREDIM) = f_2_6_0.x_7_71 ;
    LOCSTORE(store,  7, 72, STOREDIM, STOREDIM) = f_2_6_0.x_7_72 ;
    LOCSTORE(store,  7, 73, STOREDIM, STOREDIM) = f_2_6_0.x_7_73 ;
    LOCSTORE(store,  7, 74, STOREDIM, STOREDIM) = f_2_6_0.x_7_74 ;
    LOCSTORE(store,  7, 75, STOREDIM, STOREDIM) = f_2_6_0.x_7_75 ;
    LOCSTORE(store,  7, 76, STOREDIM, STOREDIM) = f_2_6_0.x_7_76 ;
    LOCSTORE(store,  7, 77, STOREDIM, STOREDIM) = f_2_6_0.x_7_77 ;
    LOCSTORE(store,  7, 78, STOREDIM, STOREDIM) = f_2_6_0.x_7_78 ;
    LOCSTORE(store,  7, 79, STOREDIM, STOREDIM) = f_2_6_0.x_7_79 ;
    LOCSTORE(store,  7, 80, STOREDIM, STOREDIM) = f_2_6_0.x_7_80 ;
    LOCSTORE(store,  7, 81, STOREDIM, STOREDIM) = f_2_6_0.x_7_81 ;
    LOCSTORE(store,  7, 82, STOREDIM, STOREDIM) = f_2_6_0.x_7_82 ;
    LOCSTORE(store,  7, 83, STOREDIM, STOREDIM) = f_2_6_0.x_7_83 ;
    LOCSTORE(store,  8, 56, STOREDIM, STOREDIM) = f_2_6_0.x_8_56 ;
    LOCSTORE(store,  8, 57, STOREDIM, STOREDIM) = f_2_6_0.x_8_57 ;
    LOCSTORE(store,  8, 58, STOREDIM, STOREDIM) = f_2_6_0.x_8_58 ;
    LOCSTORE(store,  8, 59, STOREDIM, STOREDIM) = f_2_6_0.x_8_59 ;
    LOCSTORE(store,  8, 60, STOREDIM, STOREDIM) = f_2_6_0.x_8_60 ;
    LOCSTORE(store,  8, 61, STOREDIM, STOREDIM) = f_2_6_0.x_8_61 ;
    LOCSTORE(store,  8, 62, STOREDIM, STOREDIM) = f_2_6_0.x_8_62 ;
    LOCSTORE(store,  8, 63, STOREDIM, STOREDIM) = f_2_6_0.x_8_63 ;
    LOCSTORE(store,  8, 64, STOREDIM, STOREDIM) = f_2_6_0.x_8_64 ;
    LOCSTORE(store,  8, 65, STOREDIM, STOREDIM) = f_2_6_0.x_8_65 ;
    LOCSTORE(store,  8, 66, STOREDIM, STOREDIM) = f_2_6_0.x_8_66 ;
    LOCSTORE(store,  8, 67, STOREDIM, STOREDIM) = f_2_6_0.x_8_67 ;
    LOCSTORE(store,  8, 68, STOREDIM, STOREDIM) = f_2_6_0.x_8_68 ;
    LOCSTORE(store,  8, 69, STOREDIM, STOREDIM) = f_2_6_0.x_8_69 ;
    LOCSTORE(store,  8, 70, STOREDIM, STOREDIM) = f_2_6_0.x_8_70 ;
    LOCSTORE(store,  8, 71, STOREDIM, STOREDIM) = f_2_6_0.x_8_71 ;
    LOCSTORE(store,  8, 72, STOREDIM, STOREDIM) = f_2_6_0.x_8_72 ;
    LOCSTORE(store,  8, 73, STOREDIM, STOREDIM) = f_2_6_0.x_8_73 ;
    LOCSTORE(store,  8, 74, STOREDIM, STOREDIM) = f_2_6_0.x_8_74 ;
    LOCSTORE(store,  8, 75, STOREDIM, STOREDIM) = f_2_6_0.x_8_75 ;
    LOCSTORE(store,  8, 76, STOREDIM, STOREDIM) = f_2_6_0.x_8_76 ;
    LOCSTORE(store,  8, 77, STOREDIM, STOREDIM) = f_2_6_0.x_8_77 ;
    LOCSTORE(store,  8, 78, STOREDIM, STOREDIM) = f_2_6_0.x_8_78 ;
    LOCSTORE(store,  8, 79, STOREDIM, STOREDIM) = f_2_6_0.x_8_79 ;
    LOCSTORE(store,  8, 80, STOREDIM, STOREDIM) = f_2_6_0.x_8_80 ;
    LOCSTORE(store,  8, 81, STOREDIM, STOREDIM) = f_2_6_0.x_8_81 ;
    LOCSTORE(store,  8, 82, STOREDIM, STOREDIM) = f_2_6_0.x_8_82 ;
    LOCSTORE(store,  8, 83, STOREDIM, STOREDIM) = f_2_6_0.x_8_83 ;
    LOCSTORE(store,  9, 56, STOREDIM, STOREDIM) = f_2_6_0.x_9_56 ;
    LOCSTORE(store,  9, 57, STOREDIM, STOREDIM) = f_2_6_0.x_9_57 ;
    LOCSTORE(store,  9, 58, STOREDIM, STOREDIM) = f_2_6_0.x_9_58 ;
    LOCSTORE(store,  9, 59, STOREDIM, STOREDIM) = f_2_6_0.x_9_59 ;
    LOCSTORE(store,  9, 60, STOREDIM, STOREDIM) = f_2_6_0.x_9_60 ;
    LOCSTORE(store,  9, 61, STOREDIM, STOREDIM) = f_2_6_0.x_9_61 ;
    LOCSTORE(store,  9, 62, STOREDIM, STOREDIM) = f_2_6_0.x_9_62 ;
    LOCSTORE(store,  9, 63, STOREDIM, STOREDIM) = f_2_6_0.x_9_63 ;
    LOCSTORE(store,  9, 64, STOREDIM, STOREDIM) = f_2_6_0.x_9_64 ;
    LOCSTORE(store,  9, 65, STOREDIM, STOREDIM) = f_2_6_0.x_9_65 ;
    LOCSTORE(store,  9, 66, STOREDIM, STOREDIM) = f_2_6_0.x_9_66 ;
    LOCSTORE(store,  9, 67, STOREDIM, STOREDIM) = f_2_6_0.x_9_67 ;
    LOCSTORE(store,  9, 68, STOREDIM, STOREDIM) = f_2_6_0.x_9_68 ;
    LOCSTORE(store,  9, 69, STOREDIM, STOREDIM) = f_2_6_0.x_9_69 ;
    LOCSTORE(store,  9, 70, STOREDIM, STOREDIM) = f_2_6_0.x_9_70 ;
    LOCSTORE(store,  9, 71, STOREDIM, STOREDIM) = f_2_6_0.x_9_71 ;
    LOCSTORE(store,  9, 72, STOREDIM, STOREDIM) = f_2_6_0.x_9_72 ;
    LOCSTORE(store,  9, 73, STOREDIM, STOREDIM) = f_2_6_0.x_9_73 ;
    LOCSTORE(store,  9, 74, STOREDIM, STOREDIM) = f_2_6_0.x_9_74 ;
    LOCSTORE(store,  9, 75, STOREDIM, STOREDIM) = f_2_6_0.x_9_75 ;
    LOCSTORE(store,  9, 76, STOREDIM, STOREDIM) = f_2_6_0.x_9_76 ;
    LOCSTORE(store,  9, 77, STOREDIM, STOREDIM) = f_2_6_0.x_9_77 ;
    LOCSTORE(store,  9, 78, STOREDIM, STOREDIM) = f_2_6_0.x_9_78 ;
    LOCSTORE(store,  9, 79, STOREDIM, STOREDIM) = f_2_6_0.x_9_79 ;
    LOCSTORE(store,  9, 80, STOREDIM, STOREDIM) = f_2_6_0.x_9_80 ;
    LOCSTORE(store,  9, 81, STOREDIM, STOREDIM) = f_2_6_0.x_9_81 ;
    LOCSTORE(store,  9, 82, STOREDIM, STOREDIM) = f_2_6_0.x_9_82 ;
    LOCSTORE(store,  9, 83, STOREDIM, STOREDIM) = f_2_6_0.x_9_83 ;
}
