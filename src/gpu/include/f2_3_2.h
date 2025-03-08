__device__ __inline__  void h2_3_2(QUICKDouble* YVerticalTemp, QUICKDouble* store,
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

    // call for B =            3  L =            1
    f_3_1_t f_3_1_0 ( f_3_0_0,  f_3_0_1,  f_2_0_1, ABCDtemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            1  B =            0
    f_1_0_t f_1_0_4 ( VY( 0, 0, 4 ),  VY( 0, 0, 5 ), Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            2  B =            0
    f_2_0_t f_2_0_3 ( f_1_0_3,  f_1_0_4, VY( 0, 0, 3 ), VY( 0, 0, 4 ), ABtemp, CDcom, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            3  B =            0
    f_3_0_t f_3_0_2 ( f_2_0_2,  f_2_0_3, f_1_0_2, f_1_0_3, ABtemp, CDcom, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for B =            3  L =            1
    f_3_1_t f_3_1_1 ( f_3_0_1,  f_3_0_2,  f_2_0_2, ABCDtemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for B =            2  L =            1
    f_2_1_t f_2_1_1 ( f_2_0_1,  f_2_0_2,  f_1_0_2, ABCDtemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for B =            3  L =            2
    f_3_2_t f_3_2_0 ( f_3_1_0,  f_3_1_1, f_3_0_0, f_3_0_1, CDtemp, ABcom, f_2_1_1, ABCDtemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // WRITE LAST FOR I =            3  J=           2
    LOCSTORE(store, 10,  4, STOREDIM, STOREDIM) = f_3_2_0.x_10_4 ;
    LOCSTORE(store, 10,  5, STOREDIM, STOREDIM) = f_3_2_0.x_10_5 ;
    LOCSTORE(store, 10,  6, STOREDIM, STOREDIM) = f_3_2_0.x_10_6 ;
    LOCSTORE(store, 10,  7, STOREDIM, STOREDIM) = f_3_2_0.x_10_7 ;
    LOCSTORE(store, 10,  8, STOREDIM, STOREDIM) = f_3_2_0.x_10_8 ;
    LOCSTORE(store, 10,  9, STOREDIM, STOREDIM) = f_3_2_0.x_10_9 ;
    LOCSTORE(store, 11,  4, STOREDIM, STOREDIM) = f_3_2_0.x_11_4 ;
    LOCSTORE(store, 11,  5, STOREDIM, STOREDIM) = f_3_2_0.x_11_5 ;
    LOCSTORE(store, 11,  6, STOREDIM, STOREDIM) = f_3_2_0.x_11_6 ;
    LOCSTORE(store, 11,  7, STOREDIM, STOREDIM) = f_3_2_0.x_11_7 ;
    LOCSTORE(store, 11,  8, STOREDIM, STOREDIM) = f_3_2_0.x_11_8 ;
    LOCSTORE(store, 11,  9, STOREDIM, STOREDIM) = f_3_2_0.x_11_9 ;
    LOCSTORE(store, 12,  4, STOREDIM, STOREDIM) = f_3_2_0.x_12_4 ;
    LOCSTORE(store, 12,  5, STOREDIM, STOREDIM) = f_3_2_0.x_12_5 ;
    LOCSTORE(store, 12,  6, STOREDIM, STOREDIM) = f_3_2_0.x_12_6 ;
    LOCSTORE(store, 12,  7, STOREDIM, STOREDIM) = f_3_2_0.x_12_7 ;
    LOCSTORE(store, 12,  8, STOREDIM, STOREDIM) = f_3_2_0.x_12_8 ;
    LOCSTORE(store, 12,  9, STOREDIM, STOREDIM) = f_3_2_0.x_12_9 ;
    LOCSTORE(store, 13,  4, STOREDIM, STOREDIM) = f_3_2_0.x_13_4 ;
    LOCSTORE(store, 13,  5, STOREDIM, STOREDIM) = f_3_2_0.x_13_5 ;
    LOCSTORE(store, 13,  6, STOREDIM, STOREDIM) = f_3_2_0.x_13_6 ;
    LOCSTORE(store, 13,  7, STOREDIM, STOREDIM) = f_3_2_0.x_13_7 ;
    LOCSTORE(store, 13,  8, STOREDIM, STOREDIM) = f_3_2_0.x_13_8 ;
    LOCSTORE(store, 13,  9, STOREDIM, STOREDIM) = f_3_2_0.x_13_9 ;
    LOCSTORE(store, 14,  4, STOREDIM, STOREDIM) = f_3_2_0.x_14_4 ;
    LOCSTORE(store, 14,  5, STOREDIM, STOREDIM) = f_3_2_0.x_14_5 ;
    LOCSTORE(store, 14,  6, STOREDIM, STOREDIM) = f_3_2_0.x_14_6 ;
    LOCSTORE(store, 14,  7, STOREDIM, STOREDIM) = f_3_2_0.x_14_7 ;
    LOCSTORE(store, 14,  8, STOREDIM, STOREDIM) = f_3_2_0.x_14_8 ;
    LOCSTORE(store, 14,  9, STOREDIM, STOREDIM) = f_3_2_0.x_14_9 ;
    LOCSTORE(store, 15,  4, STOREDIM, STOREDIM) = f_3_2_0.x_15_4 ;
    LOCSTORE(store, 15,  5, STOREDIM, STOREDIM) = f_3_2_0.x_15_5 ;
    LOCSTORE(store, 15,  6, STOREDIM, STOREDIM) = f_3_2_0.x_15_6 ;
    LOCSTORE(store, 15,  7, STOREDIM, STOREDIM) = f_3_2_0.x_15_7 ;
    LOCSTORE(store, 15,  8, STOREDIM, STOREDIM) = f_3_2_0.x_15_8 ;
    LOCSTORE(store, 15,  9, STOREDIM, STOREDIM) = f_3_2_0.x_15_9 ;
    LOCSTORE(store, 16,  4, STOREDIM, STOREDIM) = f_3_2_0.x_16_4 ;
    LOCSTORE(store, 16,  5, STOREDIM, STOREDIM) = f_3_2_0.x_16_5 ;
    LOCSTORE(store, 16,  6, STOREDIM, STOREDIM) = f_3_2_0.x_16_6 ;
    LOCSTORE(store, 16,  7, STOREDIM, STOREDIM) = f_3_2_0.x_16_7 ;
    LOCSTORE(store, 16,  8, STOREDIM, STOREDIM) = f_3_2_0.x_16_8 ;
    LOCSTORE(store, 16,  9, STOREDIM, STOREDIM) = f_3_2_0.x_16_9 ;
    LOCSTORE(store, 17,  4, STOREDIM, STOREDIM) = f_3_2_0.x_17_4 ;
    LOCSTORE(store, 17,  5, STOREDIM, STOREDIM) = f_3_2_0.x_17_5 ;
    LOCSTORE(store, 17,  6, STOREDIM, STOREDIM) = f_3_2_0.x_17_6 ;
    LOCSTORE(store, 17,  7, STOREDIM, STOREDIM) = f_3_2_0.x_17_7 ;
    LOCSTORE(store, 17,  8, STOREDIM, STOREDIM) = f_3_2_0.x_17_8 ;
    LOCSTORE(store, 17,  9, STOREDIM, STOREDIM) = f_3_2_0.x_17_9 ;
    LOCSTORE(store, 18,  4, STOREDIM, STOREDIM) = f_3_2_0.x_18_4 ;
    LOCSTORE(store, 18,  5, STOREDIM, STOREDIM) = f_3_2_0.x_18_5 ;
    LOCSTORE(store, 18,  6, STOREDIM, STOREDIM) = f_3_2_0.x_18_6 ;
    LOCSTORE(store, 18,  7, STOREDIM, STOREDIM) = f_3_2_0.x_18_7 ;
    LOCSTORE(store, 18,  8, STOREDIM, STOREDIM) = f_3_2_0.x_18_8 ;
    LOCSTORE(store, 18,  9, STOREDIM, STOREDIM) = f_3_2_0.x_18_9 ;
    LOCSTORE(store, 19,  4, STOREDIM, STOREDIM) = f_3_2_0.x_19_4 ;
    LOCSTORE(store, 19,  5, STOREDIM, STOREDIM) = f_3_2_0.x_19_5 ;
    LOCSTORE(store, 19,  6, STOREDIM, STOREDIM) = f_3_2_0.x_19_6 ;
    LOCSTORE(store, 19,  7, STOREDIM, STOREDIM) = f_3_2_0.x_19_7 ;
    LOCSTORE(store, 19,  8, STOREDIM, STOREDIM) = f_3_2_0.x_19_8 ;
    LOCSTORE(store, 19,  9, STOREDIM, STOREDIM) = f_3_2_0.x_19_9 ;
}
