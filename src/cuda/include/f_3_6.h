__device__ __inline__   void h_3_6(QUICKDouble* YVerticalTemp, QUICKDouble* store,
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

    // call for L =            0  B =            1
    f_0_1_t f_0_1_8 ( VY( 0, 0, 8 ), VY( 0, 0, 9 ), Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            0  B =            2
    f_0_2_t f_0_2_7 ( f_0_1_7, f_0_1_8, VY( 0, 0, 7 ), VY( 0, 0, 8 ), CDtemp, ABcom, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            0  B =            3
    f_0_3_t f_0_3_6 ( f_0_2_6, f_0_2_7, f_0_1_6, f_0_1_7, CDtemp, ABcom, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            0  B =            4
    f_0_4_t f_0_4_5 ( f_0_3_5, f_0_3_6, f_0_2_5, f_0_2_6, CDtemp, ABcom, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            0  B =            5
    f_0_5_t f_0_5_4 ( f_0_4_4, f_0_4_5, f_0_3_4, f_0_3_5, CDtemp, ABcom, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            0  B =            6
    f_0_6_t f_0_6_3 ( f_0_5_3, f_0_5_4, f_0_4_3, f_0_4_4, CDtemp, ABcom, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            1  B =            6
    f_1_6_t f_1_6_2 ( f_0_6_2,  f_0_6_3,  f_0_5_3, ABCDtemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            1  B =            5
    f_1_5_t f_1_5_2 ( f_0_5_2,  f_0_5_3,  f_0_4_3, ABCDtemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            2  B =            6
    f_2_6_t f_2_6_1 ( f_1_6_1,  f_1_6_2, f_0_6_1, f_0_6_2, ABtemp, CDcom, f_1_5_2, ABCDtemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            1  B =            4
    f_1_4_t f_1_4_2 ( f_0_4_2,  f_0_4_3,  f_0_3_3, ABCDtemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            2  B =            5
    f_2_5_t f_2_5_1 ( f_1_5_1,  f_1_5_2, f_0_5_1, f_0_5_2, ABtemp, CDcom, f_1_4_2, ABCDtemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            3  B =            6
    f_3_6_t f_3_6_0 ( f_2_6_0,  f_2_6_1, f_1_6_0, f_1_6_1, ABtemp, CDcom, f_2_5_1, ABCDtemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // WRITE LAST FOR I =            3  J=           6
    LOC2(store, 10, 56, STOREDIM, STOREDIM) += f_3_6_0.x_10_56 ;
    LOC2(store, 10, 57, STOREDIM, STOREDIM) += f_3_6_0.x_10_57 ;
    LOC2(store, 10, 58, STOREDIM, STOREDIM) += f_3_6_0.x_10_58 ;
    LOC2(store, 10, 59, STOREDIM, STOREDIM) += f_3_6_0.x_10_59 ;
    LOC2(store, 10, 60, STOREDIM, STOREDIM) += f_3_6_0.x_10_60 ;
    LOC2(store, 10, 61, STOREDIM, STOREDIM) += f_3_6_0.x_10_61 ;
    LOC2(store, 10, 62, STOREDIM, STOREDIM) += f_3_6_0.x_10_62 ;
    LOC2(store, 10, 63, STOREDIM, STOREDIM) += f_3_6_0.x_10_63 ;
    LOC2(store, 10, 64, STOREDIM, STOREDIM) += f_3_6_0.x_10_64 ;
    LOC2(store, 10, 65, STOREDIM, STOREDIM) += f_3_6_0.x_10_65 ;
    LOC2(store, 10, 66, STOREDIM, STOREDIM) += f_3_6_0.x_10_66 ;
    LOC2(store, 10, 67, STOREDIM, STOREDIM) += f_3_6_0.x_10_67 ;
    LOC2(store, 10, 68, STOREDIM, STOREDIM) += f_3_6_0.x_10_68 ;
    LOC2(store, 10, 69, STOREDIM, STOREDIM) += f_3_6_0.x_10_69 ;
    LOC2(store, 10, 70, STOREDIM, STOREDIM) += f_3_6_0.x_10_70 ;
    LOC2(store, 10, 71, STOREDIM, STOREDIM) += f_3_6_0.x_10_71 ;
    LOC2(store, 10, 72, STOREDIM, STOREDIM) += f_3_6_0.x_10_72 ;
    LOC2(store, 10, 73, STOREDIM, STOREDIM) += f_3_6_0.x_10_73 ;
    LOC2(store, 10, 74, STOREDIM, STOREDIM) += f_3_6_0.x_10_74 ;
    LOC2(store, 10, 75, STOREDIM, STOREDIM) += f_3_6_0.x_10_75 ;
    LOC2(store, 10, 76, STOREDIM, STOREDIM) += f_3_6_0.x_10_76 ;
    LOC2(store, 10, 77, STOREDIM, STOREDIM) += f_3_6_0.x_10_77 ;
    LOC2(store, 10, 78, STOREDIM, STOREDIM) += f_3_6_0.x_10_78 ;
    LOC2(store, 10, 79, STOREDIM, STOREDIM) += f_3_6_0.x_10_79 ;
    LOC2(store, 10, 80, STOREDIM, STOREDIM) += f_3_6_0.x_10_80 ;
    LOC2(store, 10, 81, STOREDIM, STOREDIM) += f_3_6_0.x_10_81 ;
    LOC2(store, 10, 82, STOREDIM, STOREDIM) += f_3_6_0.x_10_82 ;
    LOC2(store, 10, 83, STOREDIM, STOREDIM) += f_3_6_0.x_10_83 ;
    LOC2(store, 11, 56, STOREDIM, STOREDIM) += f_3_6_0.x_11_56 ;
    LOC2(store, 11, 57, STOREDIM, STOREDIM) += f_3_6_0.x_11_57 ;
    LOC2(store, 11, 58, STOREDIM, STOREDIM) += f_3_6_0.x_11_58 ;
    LOC2(store, 11, 59, STOREDIM, STOREDIM) += f_3_6_0.x_11_59 ;
    LOC2(store, 11, 60, STOREDIM, STOREDIM) += f_3_6_0.x_11_60 ;
    LOC2(store, 11, 61, STOREDIM, STOREDIM) += f_3_6_0.x_11_61 ;
    LOC2(store, 11, 62, STOREDIM, STOREDIM) += f_3_6_0.x_11_62 ;
    LOC2(store, 11, 63, STOREDIM, STOREDIM) += f_3_6_0.x_11_63 ;
    LOC2(store, 11, 64, STOREDIM, STOREDIM) += f_3_6_0.x_11_64 ;
    LOC2(store, 11, 65, STOREDIM, STOREDIM) += f_3_6_0.x_11_65 ;
    LOC2(store, 11, 66, STOREDIM, STOREDIM) += f_3_6_0.x_11_66 ;
    LOC2(store, 11, 67, STOREDIM, STOREDIM) += f_3_6_0.x_11_67 ;
    LOC2(store, 11, 68, STOREDIM, STOREDIM) += f_3_6_0.x_11_68 ;
    LOC2(store, 11, 69, STOREDIM, STOREDIM) += f_3_6_0.x_11_69 ;
    LOC2(store, 11, 70, STOREDIM, STOREDIM) += f_3_6_0.x_11_70 ;
    LOC2(store, 11, 71, STOREDIM, STOREDIM) += f_3_6_0.x_11_71 ;
    LOC2(store, 11, 72, STOREDIM, STOREDIM) += f_3_6_0.x_11_72 ;
    LOC2(store, 11, 73, STOREDIM, STOREDIM) += f_3_6_0.x_11_73 ;
    LOC2(store, 11, 74, STOREDIM, STOREDIM) += f_3_6_0.x_11_74 ;
    LOC2(store, 11, 75, STOREDIM, STOREDIM) += f_3_6_0.x_11_75 ;
    LOC2(store, 11, 76, STOREDIM, STOREDIM) += f_3_6_0.x_11_76 ;
    LOC2(store, 11, 77, STOREDIM, STOREDIM) += f_3_6_0.x_11_77 ;
    LOC2(store, 11, 78, STOREDIM, STOREDIM) += f_3_6_0.x_11_78 ;
    LOC2(store, 11, 79, STOREDIM, STOREDIM) += f_3_6_0.x_11_79 ;
    LOC2(store, 11, 80, STOREDIM, STOREDIM) += f_3_6_0.x_11_80 ;
    LOC2(store, 11, 81, STOREDIM, STOREDIM) += f_3_6_0.x_11_81 ;
    LOC2(store, 11, 82, STOREDIM, STOREDIM) += f_3_6_0.x_11_82 ;
    LOC2(store, 11, 83, STOREDIM, STOREDIM) += f_3_6_0.x_11_83 ;
    LOC2(store, 12, 56, STOREDIM, STOREDIM) += f_3_6_0.x_12_56 ;
    LOC2(store, 12, 57, STOREDIM, STOREDIM) += f_3_6_0.x_12_57 ;
    LOC2(store, 12, 58, STOREDIM, STOREDIM) += f_3_6_0.x_12_58 ;
    LOC2(store, 12, 59, STOREDIM, STOREDIM) += f_3_6_0.x_12_59 ;
    LOC2(store, 12, 60, STOREDIM, STOREDIM) += f_3_6_0.x_12_60 ;
    LOC2(store, 12, 61, STOREDIM, STOREDIM) += f_3_6_0.x_12_61 ;
    LOC2(store, 12, 62, STOREDIM, STOREDIM) += f_3_6_0.x_12_62 ;
    LOC2(store, 12, 63, STOREDIM, STOREDIM) += f_3_6_0.x_12_63 ;
    LOC2(store, 12, 64, STOREDIM, STOREDIM) += f_3_6_0.x_12_64 ;
    LOC2(store, 12, 65, STOREDIM, STOREDIM) += f_3_6_0.x_12_65 ;
    LOC2(store, 12, 66, STOREDIM, STOREDIM) += f_3_6_0.x_12_66 ;
    LOC2(store, 12, 67, STOREDIM, STOREDIM) += f_3_6_0.x_12_67 ;
    LOC2(store, 12, 68, STOREDIM, STOREDIM) += f_3_6_0.x_12_68 ;
    LOC2(store, 12, 69, STOREDIM, STOREDIM) += f_3_6_0.x_12_69 ;
    LOC2(store, 12, 70, STOREDIM, STOREDIM) += f_3_6_0.x_12_70 ;
    LOC2(store, 12, 71, STOREDIM, STOREDIM) += f_3_6_0.x_12_71 ;
    LOC2(store, 12, 72, STOREDIM, STOREDIM) += f_3_6_0.x_12_72 ;
    LOC2(store, 12, 73, STOREDIM, STOREDIM) += f_3_6_0.x_12_73 ;
    LOC2(store, 12, 74, STOREDIM, STOREDIM) += f_3_6_0.x_12_74 ;
    LOC2(store, 12, 75, STOREDIM, STOREDIM) += f_3_6_0.x_12_75 ;
    LOC2(store, 12, 76, STOREDIM, STOREDIM) += f_3_6_0.x_12_76 ;
    LOC2(store, 12, 77, STOREDIM, STOREDIM) += f_3_6_0.x_12_77 ;
    LOC2(store, 12, 78, STOREDIM, STOREDIM) += f_3_6_0.x_12_78 ;
    LOC2(store, 12, 79, STOREDIM, STOREDIM) += f_3_6_0.x_12_79 ;
    LOC2(store, 12, 80, STOREDIM, STOREDIM) += f_3_6_0.x_12_80 ;
    LOC2(store, 12, 81, STOREDIM, STOREDIM) += f_3_6_0.x_12_81 ;
    LOC2(store, 12, 82, STOREDIM, STOREDIM) += f_3_6_0.x_12_82 ;
    LOC2(store, 12, 83, STOREDIM, STOREDIM) += f_3_6_0.x_12_83 ;
    LOC2(store, 13, 56, STOREDIM, STOREDIM) += f_3_6_0.x_13_56 ;
    LOC2(store, 13, 57, STOREDIM, STOREDIM) += f_3_6_0.x_13_57 ;
    LOC2(store, 13, 58, STOREDIM, STOREDIM) += f_3_6_0.x_13_58 ;
    LOC2(store, 13, 59, STOREDIM, STOREDIM) += f_3_6_0.x_13_59 ;
    LOC2(store, 13, 60, STOREDIM, STOREDIM) += f_3_6_0.x_13_60 ;
    LOC2(store, 13, 61, STOREDIM, STOREDIM) += f_3_6_0.x_13_61 ;
    LOC2(store, 13, 62, STOREDIM, STOREDIM) += f_3_6_0.x_13_62 ;
    LOC2(store, 13, 63, STOREDIM, STOREDIM) += f_3_6_0.x_13_63 ;
    LOC2(store, 13, 64, STOREDIM, STOREDIM) += f_3_6_0.x_13_64 ;
    LOC2(store, 13, 65, STOREDIM, STOREDIM) += f_3_6_0.x_13_65 ;
    LOC2(store, 13, 66, STOREDIM, STOREDIM) += f_3_6_0.x_13_66 ;
    LOC2(store, 13, 67, STOREDIM, STOREDIM) += f_3_6_0.x_13_67 ;
    LOC2(store, 13, 68, STOREDIM, STOREDIM) += f_3_6_0.x_13_68 ;
    LOC2(store, 13, 69, STOREDIM, STOREDIM) += f_3_6_0.x_13_69 ;
    LOC2(store, 13, 70, STOREDIM, STOREDIM) += f_3_6_0.x_13_70 ;
    LOC2(store, 13, 71, STOREDIM, STOREDIM) += f_3_6_0.x_13_71 ;
    LOC2(store, 13, 72, STOREDIM, STOREDIM) += f_3_6_0.x_13_72 ;
    LOC2(store, 13, 73, STOREDIM, STOREDIM) += f_3_6_0.x_13_73 ;
    LOC2(store, 13, 74, STOREDIM, STOREDIM) += f_3_6_0.x_13_74 ;
    LOC2(store, 13, 75, STOREDIM, STOREDIM) += f_3_6_0.x_13_75 ;
    LOC2(store, 13, 76, STOREDIM, STOREDIM) += f_3_6_0.x_13_76 ;
    LOC2(store, 13, 77, STOREDIM, STOREDIM) += f_3_6_0.x_13_77 ;
    LOC2(store, 13, 78, STOREDIM, STOREDIM) += f_3_6_0.x_13_78 ;
    LOC2(store, 13, 79, STOREDIM, STOREDIM) += f_3_6_0.x_13_79 ;
    LOC2(store, 13, 80, STOREDIM, STOREDIM) += f_3_6_0.x_13_80 ;
    LOC2(store, 13, 81, STOREDIM, STOREDIM) += f_3_6_0.x_13_81 ;
    LOC2(store, 13, 82, STOREDIM, STOREDIM) += f_3_6_0.x_13_82 ;
    LOC2(store, 13, 83, STOREDIM, STOREDIM) += f_3_6_0.x_13_83 ;
    LOC2(store, 14, 56, STOREDIM, STOREDIM) += f_3_6_0.x_14_56 ;
    LOC2(store, 14, 57, STOREDIM, STOREDIM) += f_3_6_0.x_14_57 ;
    LOC2(store, 14, 58, STOREDIM, STOREDIM) += f_3_6_0.x_14_58 ;
    LOC2(store, 14, 59, STOREDIM, STOREDIM) += f_3_6_0.x_14_59 ;
    LOC2(store, 14, 60, STOREDIM, STOREDIM) += f_3_6_0.x_14_60 ;
    LOC2(store, 14, 61, STOREDIM, STOREDIM) += f_3_6_0.x_14_61 ;
    LOC2(store, 14, 62, STOREDIM, STOREDIM) += f_3_6_0.x_14_62 ;
    LOC2(store, 14, 63, STOREDIM, STOREDIM) += f_3_6_0.x_14_63 ;
    LOC2(store, 14, 64, STOREDIM, STOREDIM) += f_3_6_0.x_14_64 ;
    LOC2(store, 14, 65, STOREDIM, STOREDIM) += f_3_6_0.x_14_65 ;
    LOC2(store, 14, 66, STOREDIM, STOREDIM) += f_3_6_0.x_14_66 ;
    LOC2(store, 14, 67, STOREDIM, STOREDIM) += f_3_6_0.x_14_67 ;
    LOC2(store, 14, 68, STOREDIM, STOREDIM) += f_3_6_0.x_14_68 ;
    LOC2(store, 14, 69, STOREDIM, STOREDIM) += f_3_6_0.x_14_69 ;
    LOC2(store, 14, 70, STOREDIM, STOREDIM) += f_3_6_0.x_14_70 ;
    LOC2(store, 14, 71, STOREDIM, STOREDIM) += f_3_6_0.x_14_71 ;
    LOC2(store, 14, 72, STOREDIM, STOREDIM) += f_3_6_0.x_14_72 ;
    LOC2(store, 14, 73, STOREDIM, STOREDIM) += f_3_6_0.x_14_73 ;
    LOC2(store, 14, 74, STOREDIM, STOREDIM) += f_3_6_0.x_14_74 ;
    LOC2(store, 14, 75, STOREDIM, STOREDIM) += f_3_6_0.x_14_75 ;
    LOC2(store, 14, 76, STOREDIM, STOREDIM) += f_3_6_0.x_14_76 ;
    LOC2(store, 14, 77, STOREDIM, STOREDIM) += f_3_6_0.x_14_77 ;
    LOC2(store, 14, 78, STOREDIM, STOREDIM) += f_3_6_0.x_14_78 ;
    LOC2(store, 14, 79, STOREDIM, STOREDIM) += f_3_6_0.x_14_79 ;
    LOC2(store, 14, 80, STOREDIM, STOREDIM) += f_3_6_0.x_14_80 ;
    LOC2(store, 14, 81, STOREDIM, STOREDIM) += f_3_6_0.x_14_81 ;
    LOC2(store, 14, 82, STOREDIM, STOREDIM) += f_3_6_0.x_14_82 ;
    LOC2(store, 14, 83, STOREDIM, STOREDIM) += f_3_6_0.x_14_83 ;
    LOC2(store, 15, 56, STOREDIM, STOREDIM) += f_3_6_0.x_15_56 ;
    LOC2(store, 15, 57, STOREDIM, STOREDIM) += f_3_6_0.x_15_57 ;
    LOC2(store, 15, 58, STOREDIM, STOREDIM) += f_3_6_0.x_15_58 ;
    LOC2(store, 15, 59, STOREDIM, STOREDIM) += f_3_6_0.x_15_59 ;
    LOC2(store, 15, 60, STOREDIM, STOREDIM) += f_3_6_0.x_15_60 ;
    LOC2(store, 15, 61, STOREDIM, STOREDIM) += f_3_6_0.x_15_61 ;
    LOC2(store, 15, 62, STOREDIM, STOREDIM) += f_3_6_0.x_15_62 ;
    LOC2(store, 15, 63, STOREDIM, STOREDIM) += f_3_6_0.x_15_63 ;
    LOC2(store, 15, 64, STOREDIM, STOREDIM) += f_3_6_0.x_15_64 ;
    LOC2(store, 15, 65, STOREDIM, STOREDIM) += f_3_6_0.x_15_65 ;
    LOC2(store, 15, 66, STOREDIM, STOREDIM) += f_3_6_0.x_15_66 ;
    LOC2(store, 15, 67, STOREDIM, STOREDIM) += f_3_6_0.x_15_67 ;
    LOC2(store, 15, 68, STOREDIM, STOREDIM) += f_3_6_0.x_15_68 ;
    LOC2(store, 15, 69, STOREDIM, STOREDIM) += f_3_6_0.x_15_69 ;
    LOC2(store, 15, 70, STOREDIM, STOREDIM) += f_3_6_0.x_15_70 ;
    LOC2(store, 15, 71, STOREDIM, STOREDIM) += f_3_6_0.x_15_71 ;
    LOC2(store, 15, 72, STOREDIM, STOREDIM) += f_3_6_0.x_15_72 ;
    LOC2(store, 15, 73, STOREDIM, STOREDIM) += f_3_6_0.x_15_73 ;
    LOC2(store, 15, 74, STOREDIM, STOREDIM) += f_3_6_0.x_15_74 ;
    LOC2(store, 15, 75, STOREDIM, STOREDIM) += f_3_6_0.x_15_75 ;
    LOC2(store, 15, 76, STOREDIM, STOREDIM) += f_3_6_0.x_15_76 ;
    LOC2(store, 15, 77, STOREDIM, STOREDIM) += f_3_6_0.x_15_77 ;
    LOC2(store, 15, 78, STOREDIM, STOREDIM) += f_3_6_0.x_15_78 ;
    LOC2(store, 15, 79, STOREDIM, STOREDIM) += f_3_6_0.x_15_79 ;
    LOC2(store, 15, 80, STOREDIM, STOREDIM) += f_3_6_0.x_15_80 ;
    LOC2(store, 15, 81, STOREDIM, STOREDIM) += f_3_6_0.x_15_81 ;
    LOC2(store, 15, 82, STOREDIM, STOREDIM) += f_3_6_0.x_15_82 ;
    LOC2(store, 15, 83, STOREDIM, STOREDIM) += f_3_6_0.x_15_83 ;
    LOC2(store, 16, 56, STOREDIM, STOREDIM) += f_3_6_0.x_16_56 ;
    LOC2(store, 16, 57, STOREDIM, STOREDIM) += f_3_6_0.x_16_57 ;
    LOC2(store, 16, 58, STOREDIM, STOREDIM) += f_3_6_0.x_16_58 ;
    LOC2(store, 16, 59, STOREDIM, STOREDIM) += f_3_6_0.x_16_59 ;
    LOC2(store, 16, 60, STOREDIM, STOREDIM) += f_3_6_0.x_16_60 ;
    LOC2(store, 16, 61, STOREDIM, STOREDIM) += f_3_6_0.x_16_61 ;
    LOC2(store, 16, 62, STOREDIM, STOREDIM) += f_3_6_0.x_16_62 ;
    LOC2(store, 16, 63, STOREDIM, STOREDIM) += f_3_6_0.x_16_63 ;
    LOC2(store, 16, 64, STOREDIM, STOREDIM) += f_3_6_0.x_16_64 ;
    LOC2(store, 16, 65, STOREDIM, STOREDIM) += f_3_6_0.x_16_65 ;
    LOC2(store, 16, 66, STOREDIM, STOREDIM) += f_3_6_0.x_16_66 ;
    LOC2(store, 16, 67, STOREDIM, STOREDIM) += f_3_6_0.x_16_67 ;
    LOC2(store, 16, 68, STOREDIM, STOREDIM) += f_3_6_0.x_16_68 ;
    LOC2(store, 16, 69, STOREDIM, STOREDIM) += f_3_6_0.x_16_69 ;
    LOC2(store, 16, 70, STOREDIM, STOREDIM) += f_3_6_0.x_16_70 ;
    LOC2(store, 16, 71, STOREDIM, STOREDIM) += f_3_6_0.x_16_71 ;
    LOC2(store, 16, 72, STOREDIM, STOREDIM) += f_3_6_0.x_16_72 ;
    LOC2(store, 16, 73, STOREDIM, STOREDIM) += f_3_6_0.x_16_73 ;
    LOC2(store, 16, 74, STOREDIM, STOREDIM) += f_3_6_0.x_16_74 ;
    LOC2(store, 16, 75, STOREDIM, STOREDIM) += f_3_6_0.x_16_75 ;
    LOC2(store, 16, 76, STOREDIM, STOREDIM) += f_3_6_0.x_16_76 ;
    LOC2(store, 16, 77, STOREDIM, STOREDIM) += f_3_6_0.x_16_77 ;
    LOC2(store, 16, 78, STOREDIM, STOREDIM) += f_3_6_0.x_16_78 ;
    LOC2(store, 16, 79, STOREDIM, STOREDIM) += f_3_6_0.x_16_79 ;
    LOC2(store, 16, 80, STOREDIM, STOREDIM) += f_3_6_0.x_16_80 ;
    LOC2(store, 16, 81, STOREDIM, STOREDIM) += f_3_6_0.x_16_81 ;
    LOC2(store, 16, 82, STOREDIM, STOREDIM) += f_3_6_0.x_16_82 ;
    LOC2(store, 16, 83, STOREDIM, STOREDIM) += f_3_6_0.x_16_83 ;
    LOC2(store, 17, 56, STOREDIM, STOREDIM) += f_3_6_0.x_17_56 ;
    LOC2(store, 17, 57, STOREDIM, STOREDIM) += f_3_6_0.x_17_57 ;
    LOC2(store, 17, 58, STOREDIM, STOREDIM) += f_3_6_0.x_17_58 ;
    LOC2(store, 17, 59, STOREDIM, STOREDIM) += f_3_6_0.x_17_59 ;
    LOC2(store, 17, 60, STOREDIM, STOREDIM) += f_3_6_0.x_17_60 ;
    LOC2(store, 17, 61, STOREDIM, STOREDIM) += f_3_6_0.x_17_61 ;
    LOC2(store, 17, 62, STOREDIM, STOREDIM) += f_3_6_0.x_17_62 ;
    LOC2(store, 17, 63, STOREDIM, STOREDIM) += f_3_6_0.x_17_63 ;
    LOC2(store, 17, 64, STOREDIM, STOREDIM) += f_3_6_0.x_17_64 ;
    LOC2(store, 17, 65, STOREDIM, STOREDIM) += f_3_6_0.x_17_65 ;
    LOC2(store, 17, 66, STOREDIM, STOREDIM) += f_3_6_0.x_17_66 ;
    LOC2(store, 17, 67, STOREDIM, STOREDIM) += f_3_6_0.x_17_67 ;
    LOC2(store, 17, 68, STOREDIM, STOREDIM) += f_3_6_0.x_17_68 ;
    LOC2(store, 17, 69, STOREDIM, STOREDIM) += f_3_6_0.x_17_69 ;
    LOC2(store, 17, 70, STOREDIM, STOREDIM) += f_3_6_0.x_17_70 ;
    LOC2(store, 17, 71, STOREDIM, STOREDIM) += f_3_6_0.x_17_71 ;
    LOC2(store, 17, 72, STOREDIM, STOREDIM) += f_3_6_0.x_17_72 ;
    LOC2(store, 17, 73, STOREDIM, STOREDIM) += f_3_6_0.x_17_73 ;
    LOC2(store, 17, 74, STOREDIM, STOREDIM) += f_3_6_0.x_17_74 ;
    LOC2(store, 17, 75, STOREDIM, STOREDIM) += f_3_6_0.x_17_75 ;
    LOC2(store, 17, 76, STOREDIM, STOREDIM) += f_3_6_0.x_17_76 ;
    LOC2(store, 17, 77, STOREDIM, STOREDIM) += f_3_6_0.x_17_77 ;
    LOC2(store, 17, 78, STOREDIM, STOREDIM) += f_3_6_0.x_17_78 ;
    LOC2(store, 17, 79, STOREDIM, STOREDIM) += f_3_6_0.x_17_79 ;
    LOC2(store, 17, 80, STOREDIM, STOREDIM) += f_3_6_0.x_17_80 ;
    LOC2(store, 17, 81, STOREDIM, STOREDIM) += f_3_6_0.x_17_81 ;
    LOC2(store, 17, 82, STOREDIM, STOREDIM) += f_3_6_0.x_17_82 ;
    LOC2(store, 17, 83, STOREDIM, STOREDIM) += f_3_6_0.x_17_83 ;
    LOC2(store, 18, 56, STOREDIM, STOREDIM) += f_3_6_0.x_18_56 ;
    LOC2(store, 18, 57, STOREDIM, STOREDIM) += f_3_6_0.x_18_57 ;
    LOC2(store, 18, 58, STOREDIM, STOREDIM) += f_3_6_0.x_18_58 ;
    LOC2(store, 18, 59, STOREDIM, STOREDIM) += f_3_6_0.x_18_59 ;
    LOC2(store, 18, 60, STOREDIM, STOREDIM) += f_3_6_0.x_18_60 ;
    LOC2(store, 18, 61, STOREDIM, STOREDIM) += f_3_6_0.x_18_61 ;
    LOC2(store, 18, 62, STOREDIM, STOREDIM) += f_3_6_0.x_18_62 ;
    LOC2(store, 18, 63, STOREDIM, STOREDIM) += f_3_6_0.x_18_63 ;
    LOC2(store, 18, 64, STOREDIM, STOREDIM) += f_3_6_0.x_18_64 ;
    LOC2(store, 18, 65, STOREDIM, STOREDIM) += f_3_6_0.x_18_65 ;
    LOC2(store, 18, 66, STOREDIM, STOREDIM) += f_3_6_0.x_18_66 ;
    LOC2(store, 18, 67, STOREDIM, STOREDIM) += f_3_6_0.x_18_67 ;
    LOC2(store, 18, 68, STOREDIM, STOREDIM) += f_3_6_0.x_18_68 ;
    LOC2(store, 18, 69, STOREDIM, STOREDIM) += f_3_6_0.x_18_69 ;
    LOC2(store, 18, 70, STOREDIM, STOREDIM) += f_3_6_0.x_18_70 ;
    LOC2(store, 18, 71, STOREDIM, STOREDIM) += f_3_6_0.x_18_71 ;
    LOC2(store, 18, 72, STOREDIM, STOREDIM) += f_3_6_0.x_18_72 ;
    LOC2(store, 18, 73, STOREDIM, STOREDIM) += f_3_6_0.x_18_73 ;
    LOC2(store, 18, 74, STOREDIM, STOREDIM) += f_3_6_0.x_18_74 ;
    LOC2(store, 18, 75, STOREDIM, STOREDIM) += f_3_6_0.x_18_75 ;
    LOC2(store, 18, 76, STOREDIM, STOREDIM) += f_3_6_0.x_18_76 ;
    LOC2(store, 18, 77, STOREDIM, STOREDIM) += f_3_6_0.x_18_77 ;
    LOC2(store, 18, 78, STOREDIM, STOREDIM) += f_3_6_0.x_18_78 ;
    LOC2(store, 18, 79, STOREDIM, STOREDIM) += f_3_6_0.x_18_79 ;
    LOC2(store, 18, 80, STOREDIM, STOREDIM) += f_3_6_0.x_18_80 ;
    LOC2(store, 18, 81, STOREDIM, STOREDIM) += f_3_6_0.x_18_81 ;
    LOC2(store, 18, 82, STOREDIM, STOREDIM) += f_3_6_0.x_18_82 ;
    LOC2(store, 18, 83, STOREDIM, STOREDIM) += f_3_6_0.x_18_83 ;
    LOC2(store, 19, 56, STOREDIM, STOREDIM) += f_3_6_0.x_19_56 ;
    LOC2(store, 19, 57, STOREDIM, STOREDIM) += f_3_6_0.x_19_57 ;
    LOC2(store, 19, 58, STOREDIM, STOREDIM) += f_3_6_0.x_19_58 ;
    LOC2(store, 19, 59, STOREDIM, STOREDIM) += f_3_6_0.x_19_59 ;
    LOC2(store, 19, 60, STOREDIM, STOREDIM) += f_3_6_0.x_19_60 ;
    LOC2(store, 19, 61, STOREDIM, STOREDIM) += f_3_6_0.x_19_61 ;
    LOC2(store, 19, 62, STOREDIM, STOREDIM) += f_3_6_0.x_19_62 ;
    LOC2(store, 19, 63, STOREDIM, STOREDIM) += f_3_6_0.x_19_63 ;
    LOC2(store, 19, 64, STOREDIM, STOREDIM) += f_3_6_0.x_19_64 ;
    LOC2(store, 19, 65, STOREDIM, STOREDIM) += f_3_6_0.x_19_65 ;
    LOC2(store, 19, 66, STOREDIM, STOREDIM) += f_3_6_0.x_19_66 ;
    LOC2(store, 19, 67, STOREDIM, STOREDIM) += f_3_6_0.x_19_67 ;
    LOC2(store, 19, 68, STOREDIM, STOREDIM) += f_3_6_0.x_19_68 ;
    LOC2(store, 19, 69, STOREDIM, STOREDIM) += f_3_6_0.x_19_69 ;
    LOC2(store, 19, 70, STOREDIM, STOREDIM) += f_3_6_0.x_19_70 ;
    LOC2(store, 19, 71, STOREDIM, STOREDIM) += f_3_6_0.x_19_71 ;
    LOC2(store, 19, 72, STOREDIM, STOREDIM) += f_3_6_0.x_19_72 ;
    LOC2(store, 19, 73, STOREDIM, STOREDIM) += f_3_6_0.x_19_73 ;
    LOC2(store, 19, 74, STOREDIM, STOREDIM) += f_3_6_0.x_19_74 ;
    LOC2(store, 19, 75, STOREDIM, STOREDIM) += f_3_6_0.x_19_75 ;
    LOC2(store, 19, 76, STOREDIM, STOREDIM) += f_3_6_0.x_19_76 ;
    LOC2(store, 19, 77, STOREDIM, STOREDIM) += f_3_6_0.x_19_77 ;
    LOC2(store, 19, 78, STOREDIM, STOREDIM) += f_3_6_0.x_19_78 ;
    LOC2(store, 19, 79, STOREDIM, STOREDIM) += f_3_6_0.x_19_79 ;
    LOC2(store, 19, 80, STOREDIM, STOREDIM) += f_3_6_0.x_19_80 ;
    LOC2(store, 19, 81, STOREDIM, STOREDIM) += f_3_6_0.x_19_81 ;
    LOC2(store, 19, 82, STOREDIM, STOREDIM) += f_3_6_0.x_19_82 ;
    LOC2(store, 19, 83, STOREDIM, STOREDIM) += f_3_6_0.x_19_83 ;
}
