__device__ __inline__  void h2_2_3(QUICKDouble* YVerticalTemp, QUICKDouble* store,
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

    // call for L =            1  B =            3
    f_1_3_t f_1_3_0 ( f_0_3_0,  f_0_3_1,  f_0_2_1, ABCDtemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            0  B =            1
    f_0_1_t f_0_1_4 ( VY( 0, 0, 4 ), VY( 0, 0, 5 ), Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            0  B =            2
    f_0_2_t f_0_2_3 ( f_0_1_3, f_0_1_4, VY( 0, 0, 3 ), VY( 0, 0, 4 ), CDtemp, ABcom, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            0  B =            3
    f_0_3_t f_0_3_2 ( f_0_2_2, f_0_2_3, f_0_1_2, f_0_1_3, CDtemp, ABcom, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            1  B =            3
    f_1_3_t f_1_3_1 ( f_0_3_1,  f_0_3_2,  f_0_2_2, ABCDtemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            1  B =            2
    f_1_2_t f_1_2_1 ( f_0_2_1,  f_0_2_2,  f_0_1_2, ABCDtemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            2  B =            3
    f_2_3_t f_2_3_0 ( f_1_3_0,  f_1_3_1, f_0_3_0, f_0_3_1, ABtemp, CDcom, f_1_2_1, ABCDtemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // WRITE LAST FOR I =            2  J=           3
    LOCSTORE(store,  4, 10, STOREDIM, STOREDIM) = f_2_3_0.x_4_10 ;
    LOCSTORE(store,  4, 11, STOREDIM, STOREDIM) = f_2_3_0.x_4_11 ;
    LOCSTORE(store,  4, 12, STOREDIM, STOREDIM) = f_2_3_0.x_4_12 ;
    LOCSTORE(store,  4, 13, STOREDIM, STOREDIM) = f_2_3_0.x_4_13 ;
    LOCSTORE(store,  4, 14, STOREDIM, STOREDIM) = f_2_3_0.x_4_14 ;
    LOCSTORE(store,  4, 15, STOREDIM, STOREDIM) = f_2_3_0.x_4_15 ;
    LOCSTORE(store,  4, 16, STOREDIM, STOREDIM) = f_2_3_0.x_4_16 ;
    LOCSTORE(store,  4, 17, STOREDIM, STOREDIM) = f_2_3_0.x_4_17 ;
    LOCSTORE(store,  4, 18, STOREDIM, STOREDIM) = f_2_3_0.x_4_18 ;
    LOCSTORE(store,  4, 19, STOREDIM, STOREDIM) = f_2_3_0.x_4_19 ;
    LOCSTORE(store,  5, 10, STOREDIM, STOREDIM) = f_2_3_0.x_5_10 ;
    LOCSTORE(store,  5, 11, STOREDIM, STOREDIM) = f_2_3_0.x_5_11 ;
    LOCSTORE(store,  5, 12, STOREDIM, STOREDIM) = f_2_3_0.x_5_12 ;
    LOCSTORE(store,  5, 13, STOREDIM, STOREDIM) = f_2_3_0.x_5_13 ;
    LOCSTORE(store,  5, 14, STOREDIM, STOREDIM) = f_2_3_0.x_5_14 ;
    LOCSTORE(store,  5, 15, STOREDIM, STOREDIM) = f_2_3_0.x_5_15 ;
    LOCSTORE(store,  5, 16, STOREDIM, STOREDIM) = f_2_3_0.x_5_16 ;
    LOCSTORE(store,  5, 17, STOREDIM, STOREDIM) = f_2_3_0.x_5_17 ;
    LOCSTORE(store,  5, 18, STOREDIM, STOREDIM) = f_2_3_0.x_5_18 ;
    LOCSTORE(store,  5, 19, STOREDIM, STOREDIM) = f_2_3_0.x_5_19 ;
    LOCSTORE(store,  6, 10, STOREDIM, STOREDIM) = f_2_3_0.x_6_10 ;
    LOCSTORE(store,  6, 11, STOREDIM, STOREDIM) = f_2_3_0.x_6_11 ;
    LOCSTORE(store,  6, 12, STOREDIM, STOREDIM) = f_2_3_0.x_6_12 ;
    LOCSTORE(store,  6, 13, STOREDIM, STOREDIM) = f_2_3_0.x_6_13 ;
    LOCSTORE(store,  6, 14, STOREDIM, STOREDIM) = f_2_3_0.x_6_14 ;
    LOCSTORE(store,  6, 15, STOREDIM, STOREDIM) = f_2_3_0.x_6_15 ;
    LOCSTORE(store,  6, 16, STOREDIM, STOREDIM) = f_2_3_0.x_6_16 ;
    LOCSTORE(store,  6, 17, STOREDIM, STOREDIM) = f_2_3_0.x_6_17 ;
    LOCSTORE(store,  6, 18, STOREDIM, STOREDIM) = f_2_3_0.x_6_18 ;
    LOCSTORE(store,  6, 19, STOREDIM, STOREDIM) = f_2_3_0.x_6_19 ;
    LOCSTORE(store,  7, 10, STOREDIM, STOREDIM) = f_2_3_0.x_7_10 ;
    LOCSTORE(store,  7, 11, STOREDIM, STOREDIM) = f_2_3_0.x_7_11 ;
    LOCSTORE(store,  7, 12, STOREDIM, STOREDIM) = f_2_3_0.x_7_12 ;
    LOCSTORE(store,  7, 13, STOREDIM, STOREDIM) = f_2_3_0.x_7_13 ;
    LOCSTORE(store,  7, 14, STOREDIM, STOREDIM) = f_2_3_0.x_7_14 ;
    LOCSTORE(store,  7, 15, STOREDIM, STOREDIM) = f_2_3_0.x_7_15 ;
    LOCSTORE(store,  7, 16, STOREDIM, STOREDIM) = f_2_3_0.x_7_16 ;
    LOCSTORE(store,  7, 17, STOREDIM, STOREDIM) = f_2_3_0.x_7_17 ;
    LOCSTORE(store,  7, 18, STOREDIM, STOREDIM) = f_2_3_0.x_7_18 ;
    LOCSTORE(store,  7, 19, STOREDIM, STOREDIM) = f_2_3_0.x_7_19 ;
    LOCSTORE(store,  8, 10, STOREDIM, STOREDIM) = f_2_3_0.x_8_10 ;
    LOCSTORE(store,  8, 11, STOREDIM, STOREDIM) = f_2_3_0.x_8_11 ;
    LOCSTORE(store,  8, 12, STOREDIM, STOREDIM) = f_2_3_0.x_8_12 ;
    LOCSTORE(store,  8, 13, STOREDIM, STOREDIM) = f_2_3_0.x_8_13 ;
    LOCSTORE(store,  8, 14, STOREDIM, STOREDIM) = f_2_3_0.x_8_14 ;
    LOCSTORE(store,  8, 15, STOREDIM, STOREDIM) = f_2_3_0.x_8_15 ;
    LOCSTORE(store,  8, 16, STOREDIM, STOREDIM) = f_2_3_0.x_8_16 ;
    LOCSTORE(store,  8, 17, STOREDIM, STOREDIM) = f_2_3_0.x_8_17 ;
    LOCSTORE(store,  8, 18, STOREDIM, STOREDIM) = f_2_3_0.x_8_18 ;
    LOCSTORE(store,  8, 19, STOREDIM, STOREDIM) = f_2_3_0.x_8_19 ;
    LOCSTORE(store,  9, 10, STOREDIM, STOREDIM) = f_2_3_0.x_9_10 ;
    LOCSTORE(store,  9, 11, STOREDIM, STOREDIM) = f_2_3_0.x_9_11 ;
    LOCSTORE(store,  9, 12, STOREDIM, STOREDIM) = f_2_3_0.x_9_12 ;
    LOCSTORE(store,  9, 13, STOREDIM, STOREDIM) = f_2_3_0.x_9_13 ;
    LOCSTORE(store,  9, 14, STOREDIM, STOREDIM) = f_2_3_0.x_9_14 ;
    LOCSTORE(store,  9, 15, STOREDIM, STOREDIM) = f_2_3_0.x_9_15 ;
    LOCSTORE(store,  9, 16, STOREDIM, STOREDIM) = f_2_3_0.x_9_16 ;
    LOCSTORE(store,  9, 17, STOREDIM, STOREDIM) = f_2_3_0.x_9_17 ;
    LOCSTORE(store,  9, 18, STOREDIM, STOREDIM) = f_2_3_0.x_9_18 ;
    LOCSTORE(store,  9, 19, STOREDIM, STOREDIM) = f_2_3_0.x_9_19 ;
}
