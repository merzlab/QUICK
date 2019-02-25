__device__ __inline__  void h_2_5(QUICKDouble* YVerticalTemp, QUICKDouble* store,
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

    // call for L =            1  B =            5
    f_1_5_t f_1_5_0 ( f_0_5_0,  f_0_5_1,  f_0_4_1, ABCDtemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

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

    // call for L =            1  B =            5
    f_1_5_t f_1_5_1 ( f_0_5_1,  f_0_5_2,  f_0_4_2, ABCDtemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            1  B =            4
    f_1_4_t f_1_4_1 ( f_0_4_1,  f_0_4_2,  f_0_3_2, ABCDtemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            2  B =            5
    f_2_5_t f_2_5_0 ( f_1_5_0,  f_1_5_1, f_0_5_0, f_0_5_1, ABtemp, CDcom, f_1_4_1, ABCDtemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // WRITE LAST FOR I =            2  J=           5
    LOC2(store,  4, 35, STOREDIM, STOREDIM) += f_2_5_0.x_4_35 ;
    LOC2(store,  4, 36, STOREDIM, STOREDIM) += f_2_5_0.x_4_36 ;
    LOC2(store,  4, 37, STOREDIM, STOREDIM) += f_2_5_0.x_4_37 ;
    LOC2(store,  4, 38, STOREDIM, STOREDIM) += f_2_5_0.x_4_38 ;
    LOC2(store,  4, 39, STOREDIM, STOREDIM) += f_2_5_0.x_4_39 ;
    LOC2(store,  4, 40, STOREDIM, STOREDIM) += f_2_5_0.x_4_40 ;
    LOC2(store,  4, 41, STOREDIM, STOREDIM) += f_2_5_0.x_4_41 ;
    LOC2(store,  4, 42, STOREDIM, STOREDIM) += f_2_5_0.x_4_42 ;
    LOC2(store,  4, 43, STOREDIM, STOREDIM) += f_2_5_0.x_4_43 ;
    LOC2(store,  4, 44, STOREDIM, STOREDIM) += f_2_5_0.x_4_44 ;
    LOC2(store,  4, 45, STOREDIM, STOREDIM) += f_2_5_0.x_4_45 ;
    LOC2(store,  4, 46, STOREDIM, STOREDIM) += f_2_5_0.x_4_46 ;
    LOC2(store,  4, 47, STOREDIM, STOREDIM) += f_2_5_0.x_4_47 ;
    LOC2(store,  4, 48, STOREDIM, STOREDIM) += f_2_5_0.x_4_48 ;
    LOC2(store,  4, 49, STOREDIM, STOREDIM) += f_2_5_0.x_4_49 ;
    LOC2(store,  4, 50, STOREDIM, STOREDIM) += f_2_5_0.x_4_50 ;
    LOC2(store,  4, 51, STOREDIM, STOREDIM) += f_2_5_0.x_4_51 ;
    LOC2(store,  4, 52, STOREDIM, STOREDIM) += f_2_5_0.x_4_52 ;
    LOC2(store,  4, 53, STOREDIM, STOREDIM) += f_2_5_0.x_4_53 ;
    LOC2(store,  4, 54, STOREDIM, STOREDIM) += f_2_5_0.x_4_54 ;
    LOC2(store,  4, 55, STOREDIM, STOREDIM) += f_2_5_0.x_4_55 ;
    LOC2(store,  5, 35, STOREDIM, STOREDIM) += f_2_5_0.x_5_35 ;
    LOC2(store,  5, 36, STOREDIM, STOREDIM) += f_2_5_0.x_5_36 ;
    LOC2(store,  5, 37, STOREDIM, STOREDIM) += f_2_5_0.x_5_37 ;
    LOC2(store,  5, 38, STOREDIM, STOREDIM) += f_2_5_0.x_5_38 ;
    LOC2(store,  5, 39, STOREDIM, STOREDIM) += f_2_5_0.x_5_39 ;
    LOC2(store,  5, 40, STOREDIM, STOREDIM) += f_2_5_0.x_5_40 ;
    LOC2(store,  5, 41, STOREDIM, STOREDIM) += f_2_5_0.x_5_41 ;
    LOC2(store,  5, 42, STOREDIM, STOREDIM) += f_2_5_0.x_5_42 ;
    LOC2(store,  5, 43, STOREDIM, STOREDIM) += f_2_5_0.x_5_43 ;
    LOC2(store,  5, 44, STOREDIM, STOREDIM) += f_2_5_0.x_5_44 ;
    LOC2(store,  5, 45, STOREDIM, STOREDIM) += f_2_5_0.x_5_45 ;
    LOC2(store,  5, 46, STOREDIM, STOREDIM) += f_2_5_0.x_5_46 ;
    LOC2(store,  5, 47, STOREDIM, STOREDIM) += f_2_5_0.x_5_47 ;
    LOC2(store,  5, 48, STOREDIM, STOREDIM) += f_2_5_0.x_5_48 ;
    LOC2(store,  5, 49, STOREDIM, STOREDIM) += f_2_5_0.x_5_49 ;
    LOC2(store,  5, 50, STOREDIM, STOREDIM) += f_2_5_0.x_5_50 ;
    LOC2(store,  5, 51, STOREDIM, STOREDIM) += f_2_5_0.x_5_51 ;
    LOC2(store,  5, 52, STOREDIM, STOREDIM) += f_2_5_0.x_5_52 ;
    LOC2(store,  5, 53, STOREDIM, STOREDIM) += f_2_5_0.x_5_53 ;
    LOC2(store,  5, 54, STOREDIM, STOREDIM) += f_2_5_0.x_5_54 ;
    LOC2(store,  5, 55, STOREDIM, STOREDIM) += f_2_5_0.x_5_55 ;
    LOC2(store,  6, 35, STOREDIM, STOREDIM) += f_2_5_0.x_6_35 ;
    LOC2(store,  6, 36, STOREDIM, STOREDIM) += f_2_5_0.x_6_36 ;
    LOC2(store,  6, 37, STOREDIM, STOREDIM) += f_2_5_0.x_6_37 ;
    LOC2(store,  6, 38, STOREDIM, STOREDIM) += f_2_5_0.x_6_38 ;
    LOC2(store,  6, 39, STOREDIM, STOREDIM) += f_2_5_0.x_6_39 ;
    LOC2(store,  6, 40, STOREDIM, STOREDIM) += f_2_5_0.x_6_40 ;
    LOC2(store,  6, 41, STOREDIM, STOREDIM) += f_2_5_0.x_6_41 ;
    LOC2(store,  6, 42, STOREDIM, STOREDIM) += f_2_5_0.x_6_42 ;
    LOC2(store,  6, 43, STOREDIM, STOREDIM) += f_2_5_0.x_6_43 ;
    LOC2(store,  6, 44, STOREDIM, STOREDIM) += f_2_5_0.x_6_44 ;
    LOC2(store,  6, 45, STOREDIM, STOREDIM) += f_2_5_0.x_6_45 ;
    LOC2(store,  6, 46, STOREDIM, STOREDIM) += f_2_5_0.x_6_46 ;
    LOC2(store,  6, 47, STOREDIM, STOREDIM) += f_2_5_0.x_6_47 ;
    LOC2(store,  6, 48, STOREDIM, STOREDIM) += f_2_5_0.x_6_48 ;
    LOC2(store,  6, 49, STOREDIM, STOREDIM) += f_2_5_0.x_6_49 ;
    LOC2(store,  6, 50, STOREDIM, STOREDIM) += f_2_5_0.x_6_50 ;
    LOC2(store,  6, 51, STOREDIM, STOREDIM) += f_2_5_0.x_6_51 ;
    LOC2(store,  6, 52, STOREDIM, STOREDIM) += f_2_5_0.x_6_52 ;
    LOC2(store,  6, 53, STOREDIM, STOREDIM) += f_2_5_0.x_6_53 ;
    LOC2(store,  6, 54, STOREDIM, STOREDIM) += f_2_5_0.x_6_54 ;
    LOC2(store,  6, 55, STOREDIM, STOREDIM) += f_2_5_0.x_6_55 ;
    LOC2(store,  7, 35, STOREDIM, STOREDIM) += f_2_5_0.x_7_35 ;
    LOC2(store,  7, 36, STOREDIM, STOREDIM) += f_2_5_0.x_7_36 ;
    LOC2(store,  7, 37, STOREDIM, STOREDIM) += f_2_5_0.x_7_37 ;
    LOC2(store,  7, 38, STOREDIM, STOREDIM) += f_2_5_0.x_7_38 ;
    LOC2(store,  7, 39, STOREDIM, STOREDIM) += f_2_5_0.x_7_39 ;
    LOC2(store,  7, 40, STOREDIM, STOREDIM) += f_2_5_0.x_7_40 ;
    LOC2(store,  7, 41, STOREDIM, STOREDIM) += f_2_5_0.x_7_41 ;
    LOC2(store,  7, 42, STOREDIM, STOREDIM) += f_2_5_0.x_7_42 ;
    LOC2(store,  7, 43, STOREDIM, STOREDIM) += f_2_5_0.x_7_43 ;
    LOC2(store,  7, 44, STOREDIM, STOREDIM) += f_2_5_0.x_7_44 ;
    LOC2(store,  7, 45, STOREDIM, STOREDIM) += f_2_5_0.x_7_45 ;
    LOC2(store,  7, 46, STOREDIM, STOREDIM) += f_2_5_0.x_7_46 ;
    LOC2(store,  7, 47, STOREDIM, STOREDIM) += f_2_5_0.x_7_47 ;
    LOC2(store,  7, 48, STOREDIM, STOREDIM) += f_2_5_0.x_7_48 ;
    LOC2(store,  7, 49, STOREDIM, STOREDIM) += f_2_5_0.x_7_49 ;
    LOC2(store,  7, 50, STOREDIM, STOREDIM) += f_2_5_0.x_7_50 ;
    LOC2(store,  7, 51, STOREDIM, STOREDIM) += f_2_5_0.x_7_51 ;
    LOC2(store,  7, 52, STOREDIM, STOREDIM) += f_2_5_0.x_7_52 ;
    LOC2(store,  7, 53, STOREDIM, STOREDIM) += f_2_5_0.x_7_53 ;
    LOC2(store,  7, 54, STOREDIM, STOREDIM) += f_2_5_0.x_7_54 ;
    LOC2(store,  7, 55, STOREDIM, STOREDIM) += f_2_5_0.x_7_55 ;
    LOC2(store,  8, 35, STOREDIM, STOREDIM) += f_2_5_0.x_8_35 ;
    LOC2(store,  8, 36, STOREDIM, STOREDIM) += f_2_5_0.x_8_36 ;
    LOC2(store,  8, 37, STOREDIM, STOREDIM) += f_2_5_0.x_8_37 ;
    LOC2(store,  8, 38, STOREDIM, STOREDIM) += f_2_5_0.x_8_38 ;
    LOC2(store,  8, 39, STOREDIM, STOREDIM) += f_2_5_0.x_8_39 ;
    LOC2(store,  8, 40, STOREDIM, STOREDIM) += f_2_5_0.x_8_40 ;
    LOC2(store,  8, 41, STOREDIM, STOREDIM) += f_2_5_0.x_8_41 ;
    LOC2(store,  8, 42, STOREDIM, STOREDIM) += f_2_5_0.x_8_42 ;
    LOC2(store,  8, 43, STOREDIM, STOREDIM) += f_2_5_0.x_8_43 ;
    LOC2(store,  8, 44, STOREDIM, STOREDIM) += f_2_5_0.x_8_44 ;
    LOC2(store,  8, 45, STOREDIM, STOREDIM) += f_2_5_0.x_8_45 ;
    LOC2(store,  8, 46, STOREDIM, STOREDIM) += f_2_5_0.x_8_46 ;
    LOC2(store,  8, 47, STOREDIM, STOREDIM) += f_2_5_0.x_8_47 ;
    LOC2(store,  8, 48, STOREDIM, STOREDIM) += f_2_5_0.x_8_48 ;
    LOC2(store,  8, 49, STOREDIM, STOREDIM) += f_2_5_0.x_8_49 ;
    LOC2(store,  8, 50, STOREDIM, STOREDIM) += f_2_5_0.x_8_50 ;
    LOC2(store,  8, 51, STOREDIM, STOREDIM) += f_2_5_0.x_8_51 ;
    LOC2(store,  8, 52, STOREDIM, STOREDIM) += f_2_5_0.x_8_52 ;
    LOC2(store,  8, 53, STOREDIM, STOREDIM) += f_2_5_0.x_8_53 ;
    LOC2(store,  8, 54, STOREDIM, STOREDIM) += f_2_5_0.x_8_54 ;
    LOC2(store,  8, 55, STOREDIM, STOREDIM) += f_2_5_0.x_8_55 ;
    LOC2(store,  9, 35, STOREDIM, STOREDIM) += f_2_5_0.x_9_35 ;
    LOC2(store,  9, 36, STOREDIM, STOREDIM) += f_2_5_0.x_9_36 ;
    LOC2(store,  9, 37, STOREDIM, STOREDIM) += f_2_5_0.x_9_37 ;
    LOC2(store,  9, 38, STOREDIM, STOREDIM) += f_2_5_0.x_9_38 ;
    LOC2(store,  9, 39, STOREDIM, STOREDIM) += f_2_5_0.x_9_39 ;
    LOC2(store,  9, 40, STOREDIM, STOREDIM) += f_2_5_0.x_9_40 ;
    LOC2(store,  9, 41, STOREDIM, STOREDIM) += f_2_5_0.x_9_41 ;
    LOC2(store,  9, 42, STOREDIM, STOREDIM) += f_2_5_0.x_9_42 ;
    LOC2(store,  9, 43, STOREDIM, STOREDIM) += f_2_5_0.x_9_43 ;
    LOC2(store,  9, 44, STOREDIM, STOREDIM) += f_2_5_0.x_9_44 ;
    LOC2(store,  9, 45, STOREDIM, STOREDIM) += f_2_5_0.x_9_45 ;
    LOC2(store,  9, 46, STOREDIM, STOREDIM) += f_2_5_0.x_9_46 ;
    LOC2(store,  9, 47, STOREDIM, STOREDIM) += f_2_5_0.x_9_47 ;
    LOC2(store,  9, 48, STOREDIM, STOREDIM) += f_2_5_0.x_9_48 ;
    LOC2(store,  9, 49, STOREDIM, STOREDIM) += f_2_5_0.x_9_49 ;
    LOC2(store,  9, 50, STOREDIM, STOREDIM) += f_2_5_0.x_9_50 ;
    LOC2(store,  9, 51, STOREDIM, STOREDIM) += f_2_5_0.x_9_51 ;
    LOC2(store,  9, 52, STOREDIM, STOREDIM) += f_2_5_0.x_9_52 ;
    LOC2(store,  9, 53, STOREDIM, STOREDIM) += f_2_5_0.x_9_53 ;
    LOC2(store,  9, 54, STOREDIM, STOREDIM) += f_2_5_0.x_9_54 ;
    LOC2(store,  9, 55, STOREDIM, STOREDIM) += f_2_5_0.x_9_55 ;
}
