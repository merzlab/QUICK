__device__ __inline__  void h_3_3(QUICKDouble* YVerticalTemp, QUICKDouble* store,
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

    // call for L =            1  B =            0
    f_1_0_t f_1_0_5 ( VY( 0, 0, 5 ),  VY( 0, 0, 6 ), Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            2  B =            0
    f_2_0_t f_2_0_4 ( f_1_0_4,  f_1_0_5, VY( 0, 0, 4 ), VY( 0, 0, 5 ), ABtemp, CDcom, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            3  B =            0
    f_3_0_t f_3_0_3 ( f_2_0_3,  f_2_0_4, f_1_0_3, f_1_0_4, ABtemp, CDcom, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for B =            3  L =            1
    f_3_1_t f_3_1_2 ( f_3_0_2,  f_3_0_3,  f_2_0_3, ABCDtemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for B =            2  L =            1
    f_2_1_t f_2_1_2 ( f_2_0_2,  f_2_0_3,  f_1_0_3, ABCDtemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for B =            3  L =            2
    f_3_2_t f_3_2_1 ( f_3_1_1,  f_3_1_2, f_3_0_1, f_3_0_2, CDtemp, ABcom, f_2_1_2, ABCDtemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            0  B =            1
    f_0_1_t f_0_1_1 ( VY( 0, 0, 1 ), VY( 0, 0, 2 ), Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            0  B =            1
    f_0_1_t f_0_1_2 ( VY( 0, 0, 2 ), VY( 0, 0, 3 ), Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            0  B =            2
    f_0_2_t f_0_2_1 ( f_0_1_1, f_0_1_2, VY( 0, 0, 1 ), VY( 0, 0, 2 ), CDtemp, ABcom, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            0  B =            1
    f_0_1_t f_0_1_3 ( VY( 0, 0, 3 ), VY( 0, 0, 4 ), Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            0  B =            2
    f_0_2_t f_0_2_2 ( f_0_1_2, f_0_1_3, VY( 0, 0, 2 ), VY( 0, 0, 3 ), CDtemp, ABcom, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            1  B =            2
    f_1_2_t f_1_2_1 ( f_0_2_1,  f_0_2_2,  f_0_1_2, ABCDtemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            0  B =            1
    f_0_1_t f_0_1_4 ( VY( 0, 0, 4 ), VY( 0, 0, 5 ), Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            0  B =            2
    f_0_2_t f_0_2_3 ( f_0_1_3, f_0_1_4, VY( 0, 0, 3 ), VY( 0, 0, 4 ), CDtemp, ABcom, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            1  B =            2
    f_1_2_t f_1_2_2 ( f_0_2_2,  f_0_2_3,  f_0_1_3, ABCDtemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            1  B =            1
    f_1_1_t f_1_1_2 ( f_0_1_2,  f_0_1_3,  VY( 0, 0, 3 ), ABCDtemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            2  B =            2
    f_2_2_t f_2_2_1 ( f_1_2_1,  f_1_2_2, f_0_2_1, f_0_2_2, ABtemp, CDcom, f_1_1_2, ABCDtemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for B =            3  L =            3
    f_3_3_t f_3_3_0 ( f_3_2_0,  f_3_2_1, f_3_1_0, f_3_1_1, CDtemp, ABcom, f_2_2_1, ABCDtemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // WRITE LAST FOR I =            3  J=           3
    LOC2(store, 10, 10, STOREDIM, STOREDIM) += f_3_3_0.x_10_10 ;
    LOC2(store, 10, 11, STOREDIM, STOREDIM) += f_3_3_0.x_10_11 ;
    LOC2(store, 10, 12, STOREDIM, STOREDIM) += f_3_3_0.x_10_12 ;
    LOC2(store, 10, 13, STOREDIM, STOREDIM) += f_3_3_0.x_10_13 ;
    LOC2(store, 10, 14, STOREDIM, STOREDIM) += f_3_3_0.x_10_14 ;
    LOC2(store, 10, 15, STOREDIM, STOREDIM) += f_3_3_0.x_10_15 ;
    LOC2(store, 10, 16, STOREDIM, STOREDIM) += f_3_3_0.x_10_16 ;
    LOC2(store, 10, 17, STOREDIM, STOREDIM) += f_3_3_0.x_10_17 ;
    LOC2(store, 10, 18, STOREDIM, STOREDIM) += f_3_3_0.x_10_18 ;
    LOC2(store, 10, 19, STOREDIM, STOREDIM) += f_3_3_0.x_10_19 ;
    LOC2(store, 11, 10, STOREDIM, STOREDIM) += f_3_3_0.x_11_10 ;
    LOC2(store, 11, 11, STOREDIM, STOREDIM) += f_3_3_0.x_11_11 ;
    LOC2(store, 11, 12, STOREDIM, STOREDIM) += f_3_3_0.x_11_12 ;
    LOC2(store, 11, 13, STOREDIM, STOREDIM) += f_3_3_0.x_11_13 ;
    LOC2(store, 11, 14, STOREDIM, STOREDIM) += f_3_3_0.x_11_14 ;
    LOC2(store, 11, 15, STOREDIM, STOREDIM) += f_3_3_0.x_11_15 ;
    LOC2(store, 11, 16, STOREDIM, STOREDIM) += f_3_3_0.x_11_16 ;
    LOC2(store, 11, 17, STOREDIM, STOREDIM) += f_3_3_0.x_11_17 ;
    LOC2(store, 11, 18, STOREDIM, STOREDIM) += f_3_3_0.x_11_18 ;
    LOC2(store, 11, 19, STOREDIM, STOREDIM) += f_3_3_0.x_11_19 ;
    LOC2(store, 12, 10, STOREDIM, STOREDIM) += f_3_3_0.x_12_10 ;
    LOC2(store, 12, 11, STOREDIM, STOREDIM) += f_3_3_0.x_12_11 ;
    LOC2(store, 12, 12, STOREDIM, STOREDIM) += f_3_3_0.x_12_12 ;
    LOC2(store, 12, 13, STOREDIM, STOREDIM) += f_3_3_0.x_12_13 ;
    LOC2(store, 12, 14, STOREDIM, STOREDIM) += f_3_3_0.x_12_14 ;
    LOC2(store, 12, 15, STOREDIM, STOREDIM) += f_3_3_0.x_12_15 ;
    LOC2(store, 12, 16, STOREDIM, STOREDIM) += f_3_3_0.x_12_16 ;
    LOC2(store, 12, 17, STOREDIM, STOREDIM) += f_3_3_0.x_12_17 ;
    LOC2(store, 12, 18, STOREDIM, STOREDIM) += f_3_3_0.x_12_18 ;
    LOC2(store, 12, 19, STOREDIM, STOREDIM) += f_3_3_0.x_12_19 ;
    LOC2(store, 13, 10, STOREDIM, STOREDIM) += f_3_3_0.x_13_10 ;
    LOC2(store, 13, 11, STOREDIM, STOREDIM) += f_3_3_0.x_13_11 ;
    LOC2(store, 13, 12, STOREDIM, STOREDIM) += f_3_3_0.x_13_12 ;
    LOC2(store, 13, 13, STOREDIM, STOREDIM) += f_3_3_0.x_13_13 ;
    LOC2(store, 13, 14, STOREDIM, STOREDIM) += f_3_3_0.x_13_14 ;
    LOC2(store, 13, 15, STOREDIM, STOREDIM) += f_3_3_0.x_13_15 ;
    LOC2(store, 13, 16, STOREDIM, STOREDIM) += f_3_3_0.x_13_16 ;
    LOC2(store, 13, 17, STOREDIM, STOREDIM) += f_3_3_0.x_13_17 ;
    LOC2(store, 13, 18, STOREDIM, STOREDIM) += f_3_3_0.x_13_18 ;
    LOC2(store, 13, 19, STOREDIM, STOREDIM) += f_3_3_0.x_13_19 ;
    LOC2(store, 14, 10, STOREDIM, STOREDIM) += f_3_3_0.x_14_10 ;
    LOC2(store, 14, 11, STOREDIM, STOREDIM) += f_3_3_0.x_14_11 ;
    LOC2(store, 14, 12, STOREDIM, STOREDIM) += f_3_3_0.x_14_12 ;
    LOC2(store, 14, 13, STOREDIM, STOREDIM) += f_3_3_0.x_14_13 ;
    LOC2(store, 14, 14, STOREDIM, STOREDIM) += f_3_3_0.x_14_14 ;
    LOC2(store, 14, 15, STOREDIM, STOREDIM) += f_3_3_0.x_14_15 ;
    LOC2(store, 14, 16, STOREDIM, STOREDIM) += f_3_3_0.x_14_16 ;
    LOC2(store, 14, 17, STOREDIM, STOREDIM) += f_3_3_0.x_14_17 ;
    LOC2(store, 14, 18, STOREDIM, STOREDIM) += f_3_3_0.x_14_18 ;
    LOC2(store, 14, 19, STOREDIM, STOREDIM) += f_3_3_0.x_14_19 ;
    LOC2(store, 15, 10, STOREDIM, STOREDIM) += f_3_3_0.x_15_10 ;
    LOC2(store, 15, 11, STOREDIM, STOREDIM) += f_3_3_0.x_15_11 ;
    LOC2(store, 15, 12, STOREDIM, STOREDIM) += f_3_3_0.x_15_12 ;
    LOC2(store, 15, 13, STOREDIM, STOREDIM) += f_3_3_0.x_15_13 ;
    LOC2(store, 15, 14, STOREDIM, STOREDIM) += f_3_3_0.x_15_14 ;
    LOC2(store, 15, 15, STOREDIM, STOREDIM) += f_3_3_0.x_15_15 ;
    LOC2(store, 15, 16, STOREDIM, STOREDIM) += f_3_3_0.x_15_16 ;
    LOC2(store, 15, 17, STOREDIM, STOREDIM) += f_3_3_0.x_15_17 ;
    LOC2(store, 15, 18, STOREDIM, STOREDIM) += f_3_3_0.x_15_18 ;
    LOC2(store, 15, 19, STOREDIM, STOREDIM) += f_3_3_0.x_15_19 ;
    LOC2(store, 16, 10, STOREDIM, STOREDIM) += f_3_3_0.x_16_10 ;
    LOC2(store, 16, 11, STOREDIM, STOREDIM) += f_3_3_0.x_16_11 ;
    LOC2(store, 16, 12, STOREDIM, STOREDIM) += f_3_3_0.x_16_12 ;
    LOC2(store, 16, 13, STOREDIM, STOREDIM) += f_3_3_0.x_16_13 ;
    LOC2(store, 16, 14, STOREDIM, STOREDIM) += f_3_3_0.x_16_14 ;
    LOC2(store, 16, 15, STOREDIM, STOREDIM) += f_3_3_0.x_16_15 ;
    LOC2(store, 16, 16, STOREDIM, STOREDIM) += f_3_3_0.x_16_16 ;
    LOC2(store, 16, 17, STOREDIM, STOREDIM) += f_3_3_0.x_16_17 ;
    LOC2(store, 16, 18, STOREDIM, STOREDIM) += f_3_3_0.x_16_18 ;
    LOC2(store, 16, 19, STOREDIM, STOREDIM) += f_3_3_0.x_16_19 ;
    LOC2(store, 17, 10, STOREDIM, STOREDIM) += f_3_3_0.x_17_10 ;
    LOC2(store, 17, 11, STOREDIM, STOREDIM) += f_3_3_0.x_17_11 ;
    LOC2(store, 17, 12, STOREDIM, STOREDIM) += f_3_3_0.x_17_12 ;
    LOC2(store, 17, 13, STOREDIM, STOREDIM) += f_3_3_0.x_17_13 ;
    LOC2(store, 17, 14, STOREDIM, STOREDIM) += f_3_3_0.x_17_14 ;
    LOC2(store, 17, 15, STOREDIM, STOREDIM) += f_3_3_0.x_17_15 ;
    LOC2(store, 17, 16, STOREDIM, STOREDIM) += f_3_3_0.x_17_16 ;
    LOC2(store, 17, 17, STOREDIM, STOREDIM) += f_3_3_0.x_17_17 ;
    LOC2(store, 17, 18, STOREDIM, STOREDIM) += f_3_3_0.x_17_18 ;
    LOC2(store, 17, 19, STOREDIM, STOREDIM) += f_3_3_0.x_17_19 ;
    LOC2(store, 18, 10, STOREDIM, STOREDIM) += f_3_3_0.x_18_10 ;
    LOC2(store, 18, 11, STOREDIM, STOREDIM) += f_3_3_0.x_18_11 ;
    LOC2(store, 18, 12, STOREDIM, STOREDIM) += f_3_3_0.x_18_12 ;
    LOC2(store, 18, 13, STOREDIM, STOREDIM) += f_3_3_0.x_18_13 ;
    LOC2(store, 18, 14, STOREDIM, STOREDIM) += f_3_3_0.x_18_14 ;
    LOC2(store, 18, 15, STOREDIM, STOREDIM) += f_3_3_0.x_18_15 ;
    LOC2(store, 18, 16, STOREDIM, STOREDIM) += f_3_3_0.x_18_16 ;
    LOC2(store, 18, 17, STOREDIM, STOREDIM) += f_3_3_0.x_18_17 ;
    LOC2(store, 18, 18, STOREDIM, STOREDIM) += f_3_3_0.x_18_18 ;
    LOC2(store, 18, 19, STOREDIM, STOREDIM) += f_3_3_0.x_18_19 ;
    LOC2(store, 19, 10, STOREDIM, STOREDIM) += f_3_3_0.x_19_10 ;
    LOC2(store, 19, 11, STOREDIM, STOREDIM) += f_3_3_0.x_19_11 ;
    LOC2(store, 19, 12, STOREDIM, STOREDIM) += f_3_3_0.x_19_12 ;
    LOC2(store, 19, 13, STOREDIM, STOREDIM) += f_3_3_0.x_19_13 ;
    LOC2(store, 19, 14, STOREDIM, STOREDIM) += f_3_3_0.x_19_14 ;
    LOC2(store, 19, 15, STOREDIM, STOREDIM) += f_3_3_0.x_19_15 ;
    LOC2(store, 19, 16, STOREDIM, STOREDIM) += f_3_3_0.x_19_16 ;
    LOC2(store, 19, 17, STOREDIM, STOREDIM) += f_3_3_0.x_19_17 ;
    LOC2(store, 19, 18, STOREDIM, STOREDIM) += f_3_3_0.x_19_18 ;
    LOC2(store, 19, 19, STOREDIM, STOREDIM) += f_3_3_0.x_19_19 ;
}
