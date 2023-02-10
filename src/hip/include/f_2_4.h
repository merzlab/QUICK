__device__ __inline__  void h_2_4(QUICKDouble* YVerticalTemp, QUICKDouble* store,
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

    // call for L =            1  B =            4
    f_1_4_t f_1_4_0 ( f_0_4_0,  f_0_4_1,  f_0_3_1, ABCDtemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            0  B =            1
    f_0_1_t f_0_1_5 ( VY( 0, 0, 5 ), VY( 0, 0, 6 ), Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            0  B =            2
    f_0_2_t f_0_2_4 ( f_0_1_4, f_0_1_5, VY( 0, 0, 4 ), VY( 0, 0, 5 ), CDtemp, ABcom, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            0  B =            3
    f_0_3_t f_0_3_3 ( f_0_2_3, f_0_2_4, f_0_1_3, f_0_1_4, CDtemp, ABcom, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            0  B =            4
    f_0_4_t f_0_4_2 ( f_0_3_2, f_0_3_3, f_0_2_2, f_0_2_3, CDtemp, ABcom, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            1  B =            4
    f_1_4_t f_1_4_1 ( f_0_4_1,  f_0_4_2,  f_0_3_2, ABCDtemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            1  B =            3
    f_1_3_t f_1_3_1 ( f_0_3_1,  f_0_3_2,  f_0_2_2, ABCDtemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            2  B =            4
    f_2_4_t f_2_4_0 ( f_1_4_0,  f_1_4_1, f_0_4_0, f_0_4_1, ABtemp, CDcom, f_1_3_1, ABCDtemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // WRITE LAST FOR I =            2  J=           4
    LOCSTORE(store,  4, 20, STOREDIM, STOREDIM) += f_2_4_0.x_4_20 ;
    LOCSTORE(store,  4, 21, STOREDIM, STOREDIM) += f_2_4_0.x_4_21 ;
    LOCSTORE(store,  4, 22, STOREDIM, STOREDIM) += f_2_4_0.x_4_22 ;
    LOCSTORE(store,  4, 23, STOREDIM, STOREDIM) += f_2_4_0.x_4_23 ;
    LOCSTORE(store,  4, 24, STOREDIM, STOREDIM) += f_2_4_0.x_4_24 ;
    LOCSTORE(store,  4, 25, STOREDIM, STOREDIM) += f_2_4_0.x_4_25 ;
    LOCSTORE(store,  4, 26, STOREDIM, STOREDIM) += f_2_4_0.x_4_26 ;
    LOCSTORE(store,  4, 27, STOREDIM, STOREDIM) += f_2_4_0.x_4_27 ;
    LOCSTORE(store,  4, 28, STOREDIM, STOREDIM) += f_2_4_0.x_4_28 ;
    LOCSTORE(store,  4, 29, STOREDIM, STOREDIM) += f_2_4_0.x_4_29 ;
    LOCSTORE(store,  4, 30, STOREDIM, STOREDIM) += f_2_4_0.x_4_30 ;
    LOCSTORE(store,  4, 31, STOREDIM, STOREDIM) += f_2_4_0.x_4_31 ;
    LOCSTORE(store,  4, 32, STOREDIM, STOREDIM) += f_2_4_0.x_4_32 ;
    LOCSTORE(store,  4, 33, STOREDIM, STOREDIM) += f_2_4_0.x_4_33 ;
    LOCSTORE(store,  4, 34, STOREDIM, STOREDIM) += f_2_4_0.x_4_34 ;
    LOCSTORE(store,  5, 20, STOREDIM, STOREDIM) += f_2_4_0.x_5_20 ;
    LOCSTORE(store,  5, 21, STOREDIM, STOREDIM) += f_2_4_0.x_5_21 ;
    LOCSTORE(store,  5, 22, STOREDIM, STOREDIM) += f_2_4_0.x_5_22 ;
    LOCSTORE(store,  5, 23, STOREDIM, STOREDIM) += f_2_4_0.x_5_23 ;
    LOCSTORE(store,  5, 24, STOREDIM, STOREDIM) += f_2_4_0.x_5_24 ;
    LOCSTORE(store,  5, 25, STOREDIM, STOREDIM) += f_2_4_0.x_5_25 ;
    LOCSTORE(store,  5, 26, STOREDIM, STOREDIM) += f_2_4_0.x_5_26 ;
    LOCSTORE(store,  5, 27, STOREDIM, STOREDIM) += f_2_4_0.x_5_27 ;
    LOCSTORE(store,  5, 28, STOREDIM, STOREDIM) += f_2_4_0.x_5_28 ;
    LOCSTORE(store,  5, 29, STOREDIM, STOREDIM) += f_2_4_0.x_5_29 ;
    LOCSTORE(store,  5, 30, STOREDIM, STOREDIM) += f_2_4_0.x_5_30 ;
    LOCSTORE(store,  5, 31, STOREDIM, STOREDIM) += f_2_4_0.x_5_31 ;
    LOCSTORE(store,  5, 32, STOREDIM, STOREDIM) += f_2_4_0.x_5_32 ;
    LOCSTORE(store,  5, 33, STOREDIM, STOREDIM) += f_2_4_0.x_5_33 ;
    LOCSTORE(store,  5, 34, STOREDIM, STOREDIM) += f_2_4_0.x_5_34 ;
    LOCSTORE(store,  6, 20, STOREDIM, STOREDIM) += f_2_4_0.x_6_20 ;
    LOCSTORE(store,  6, 21, STOREDIM, STOREDIM) += f_2_4_0.x_6_21 ;
    LOCSTORE(store,  6, 22, STOREDIM, STOREDIM) += f_2_4_0.x_6_22 ;
    LOCSTORE(store,  6, 23, STOREDIM, STOREDIM) += f_2_4_0.x_6_23 ;
    LOCSTORE(store,  6, 24, STOREDIM, STOREDIM) += f_2_4_0.x_6_24 ;
    LOCSTORE(store,  6, 25, STOREDIM, STOREDIM) += f_2_4_0.x_6_25 ;
    LOCSTORE(store,  6, 26, STOREDIM, STOREDIM) += f_2_4_0.x_6_26 ;
    LOCSTORE(store,  6, 27, STOREDIM, STOREDIM) += f_2_4_0.x_6_27 ;
    LOCSTORE(store,  6, 28, STOREDIM, STOREDIM) += f_2_4_0.x_6_28 ;
    LOCSTORE(store,  6, 29, STOREDIM, STOREDIM) += f_2_4_0.x_6_29 ;
    LOCSTORE(store,  6, 30, STOREDIM, STOREDIM) += f_2_4_0.x_6_30 ;
    LOCSTORE(store,  6, 31, STOREDIM, STOREDIM) += f_2_4_0.x_6_31 ;
    LOCSTORE(store,  6, 32, STOREDIM, STOREDIM) += f_2_4_0.x_6_32 ;
    LOCSTORE(store,  6, 33, STOREDIM, STOREDIM) += f_2_4_0.x_6_33 ;
    LOCSTORE(store,  6, 34, STOREDIM, STOREDIM) += f_2_4_0.x_6_34 ;
    LOCSTORE(store,  7, 20, STOREDIM, STOREDIM) += f_2_4_0.x_7_20 ;
    LOCSTORE(store,  7, 21, STOREDIM, STOREDIM) += f_2_4_0.x_7_21 ;
    LOCSTORE(store,  7, 22, STOREDIM, STOREDIM) += f_2_4_0.x_7_22 ;
    LOCSTORE(store,  7, 23, STOREDIM, STOREDIM) += f_2_4_0.x_7_23 ;
    LOCSTORE(store,  7, 24, STOREDIM, STOREDIM) += f_2_4_0.x_7_24 ;
    LOCSTORE(store,  7, 25, STOREDIM, STOREDIM) += f_2_4_0.x_7_25 ;
    LOCSTORE(store,  7, 26, STOREDIM, STOREDIM) += f_2_4_0.x_7_26 ;
    LOCSTORE(store,  7, 27, STOREDIM, STOREDIM) += f_2_4_0.x_7_27 ;
    LOCSTORE(store,  7, 28, STOREDIM, STOREDIM) += f_2_4_0.x_7_28 ;
    LOCSTORE(store,  7, 29, STOREDIM, STOREDIM) += f_2_4_0.x_7_29 ;
    LOCSTORE(store,  7, 30, STOREDIM, STOREDIM) += f_2_4_0.x_7_30 ;
    LOCSTORE(store,  7, 31, STOREDIM, STOREDIM) += f_2_4_0.x_7_31 ;
    LOCSTORE(store,  7, 32, STOREDIM, STOREDIM) += f_2_4_0.x_7_32 ;
    LOCSTORE(store,  7, 33, STOREDIM, STOREDIM) += f_2_4_0.x_7_33 ;
    LOCSTORE(store,  7, 34, STOREDIM, STOREDIM) += f_2_4_0.x_7_34 ;
    LOCSTORE(store,  8, 20, STOREDIM, STOREDIM) += f_2_4_0.x_8_20 ;
    LOCSTORE(store,  8, 21, STOREDIM, STOREDIM) += f_2_4_0.x_8_21 ;
    LOCSTORE(store,  8, 22, STOREDIM, STOREDIM) += f_2_4_0.x_8_22 ;
    LOCSTORE(store,  8, 23, STOREDIM, STOREDIM) += f_2_4_0.x_8_23 ;
    LOCSTORE(store,  8, 24, STOREDIM, STOREDIM) += f_2_4_0.x_8_24 ;
    LOCSTORE(store,  8, 25, STOREDIM, STOREDIM) += f_2_4_0.x_8_25 ;
    LOCSTORE(store,  8, 26, STOREDIM, STOREDIM) += f_2_4_0.x_8_26 ;
    LOCSTORE(store,  8, 27, STOREDIM, STOREDIM) += f_2_4_0.x_8_27 ;
    LOCSTORE(store,  8, 28, STOREDIM, STOREDIM) += f_2_4_0.x_8_28 ;
    LOCSTORE(store,  8, 29, STOREDIM, STOREDIM) += f_2_4_0.x_8_29 ;
    LOCSTORE(store,  8, 30, STOREDIM, STOREDIM) += f_2_4_0.x_8_30 ;
    LOCSTORE(store,  8, 31, STOREDIM, STOREDIM) += f_2_4_0.x_8_31 ;
    LOCSTORE(store,  8, 32, STOREDIM, STOREDIM) += f_2_4_0.x_8_32 ;
    LOCSTORE(store,  8, 33, STOREDIM, STOREDIM) += f_2_4_0.x_8_33 ;
    LOCSTORE(store,  8, 34, STOREDIM, STOREDIM) += f_2_4_0.x_8_34 ;
    LOCSTORE(store,  9, 20, STOREDIM, STOREDIM) += f_2_4_0.x_9_20 ;
    LOCSTORE(store,  9, 21, STOREDIM, STOREDIM) += f_2_4_0.x_9_21 ;
    LOCSTORE(store,  9, 22, STOREDIM, STOREDIM) += f_2_4_0.x_9_22 ;
    LOCSTORE(store,  9, 23, STOREDIM, STOREDIM) += f_2_4_0.x_9_23 ;
    LOCSTORE(store,  9, 24, STOREDIM, STOREDIM) += f_2_4_0.x_9_24 ;
    LOCSTORE(store,  9, 25, STOREDIM, STOREDIM) += f_2_4_0.x_9_25 ;
    LOCSTORE(store,  9, 26, STOREDIM, STOREDIM) += f_2_4_0.x_9_26 ;
    LOCSTORE(store,  9, 27, STOREDIM, STOREDIM) += f_2_4_0.x_9_27 ;
    LOCSTORE(store,  9, 28, STOREDIM, STOREDIM) += f_2_4_0.x_9_28 ;
    LOCSTORE(store,  9, 29, STOREDIM, STOREDIM) += f_2_4_0.x_9_29 ;
    LOCSTORE(store,  9, 30, STOREDIM, STOREDIM) += f_2_4_0.x_9_30 ;
    LOCSTORE(store,  9, 31, STOREDIM, STOREDIM) += f_2_4_0.x_9_31 ;
    LOCSTORE(store,  9, 32, STOREDIM, STOREDIM) += f_2_4_0.x_9_32 ;
    LOCSTORE(store,  9, 33, STOREDIM, STOREDIM) += f_2_4_0.x_9_33 ;
    LOCSTORE(store,  9, 34, STOREDIM, STOREDIM) += f_2_4_0.x_9_34 ;
}
