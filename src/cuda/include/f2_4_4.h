__device__ __inline__   void h2_4_4(QUICKDouble* YVerticalTemp, QUICKDouble* store,
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

    // call for L =            4  B =            0
    f_4_0_t f_4_0_0 ( f_3_0_0,  f_3_0_1, f_2_0_0, f_2_0_1, ABtemp, CDcom, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            1  B =            0
    f_1_0_t f_1_0_4 ( VY( 0, 0, 4 ),  VY( 0, 0, 5 ), Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            2  B =            0
    f_2_0_t f_2_0_3 ( f_1_0_3,  f_1_0_4, VY( 0, 0, 3 ), VY( 0, 0, 4 ), ABtemp, CDcom, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            3  B =            0
    f_3_0_t f_3_0_2 ( f_2_0_2,  f_2_0_3, f_1_0_2, f_1_0_3, ABtemp, CDcom, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            4  B =            0
    f_4_0_t f_4_0_1 ( f_3_0_1,  f_3_0_2, f_2_0_1, f_2_0_2, ABtemp, CDcom, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for B =            4  L =            1
    f_4_1_t f_4_1_0 ( f_4_0_0,  f_4_0_1,  f_3_0_1, ABCDtemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            1  B =            0
    f_1_0_t f_1_0_5 ( VY( 0, 0, 5 ),  VY( 0, 0, 6 ), Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            2  B =            0
    f_2_0_t f_2_0_4 ( f_1_0_4,  f_1_0_5, VY( 0, 0, 4 ), VY( 0, 0, 5 ), ABtemp, CDcom, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            3  B =            0
    f_3_0_t f_3_0_3 ( f_2_0_3,  f_2_0_4, f_1_0_3, f_1_0_4, ABtemp, CDcom, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            4  B =            0
    f_4_0_t f_4_0_2 ( f_3_0_2,  f_3_0_3, f_2_0_2, f_2_0_3, ABtemp, CDcom, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for B =            4  L =            1
    f_4_1_t f_4_1_1 ( f_4_0_1,  f_4_0_2,  f_3_0_2, ABCDtemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for B =            3  L =            1
    f_3_1_t f_3_1_1 ( f_3_0_1,  f_3_0_2,  f_2_0_2, ABCDtemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for B =            4  L =            2
    f_4_2_t f_4_2_0 ( f_4_1_0,  f_4_1_1, f_4_0_0, f_4_0_1, CDtemp, ABcom, f_3_1_1, ABCDtemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            1  B =            0
    f_1_0_t f_1_0_6 ( VY( 0, 0, 6 ),  VY( 0, 0, 7 ), Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            2  B =            0
    f_2_0_t f_2_0_5 ( f_1_0_5,  f_1_0_6, VY( 0, 0, 5 ), VY( 0, 0, 6 ), ABtemp, CDcom, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            3  B =            0
    f_3_0_t f_3_0_4 ( f_2_0_4,  f_2_0_5, f_1_0_4, f_1_0_5, ABtemp, CDcom, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            4  B =            0
    f_4_0_t f_4_0_3 ( f_3_0_3,  f_3_0_4, f_2_0_3, f_2_0_4, ABtemp, CDcom, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for B =            4  L =            1
    f_4_1_t f_4_1_2 ( f_4_0_2,  f_4_0_3,  f_3_0_3, ABCDtemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for B =            3  L =            1
    f_3_1_t f_3_1_2 ( f_3_0_2,  f_3_0_3,  f_2_0_3, ABCDtemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for B =            4  L =            2
    f_4_2_t f_4_2_1 ( f_4_1_1,  f_4_1_2, f_4_0_1, f_4_0_2, CDtemp, ABcom, f_3_1_2, ABCDtemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for B =            2  L =            1
    f_2_1_t f_2_1_2 ( f_2_0_2,  f_2_0_3,  f_1_0_3, ABCDtemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for B =            3  L =            2
    f_3_2_t f_3_2_1 ( f_3_1_1,  f_3_1_2, f_3_0_1, f_3_0_2, CDtemp, ABcom, f_2_1_2, ABCDtemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for B =            4  L =            3
    f_4_3_t f_4_3_0 ( f_4_2_0,  f_4_2_1, f_4_1_0, f_4_1_1, CDtemp, ABcom, f_3_2_1, ABCDtemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            1  B =            0
    f_1_0_t f_1_0_7 ( VY( 0, 0, 7 ),  VY( 0, 0, 8 ), Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            2  B =            0
    f_2_0_t f_2_0_6 ( f_1_0_6,  f_1_0_7, VY( 0, 0, 6 ), VY( 0, 0, 7 ), ABtemp, CDcom, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            3  B =            0
    f_3_0_t f_3_0_5 ( f_2_0_5,  f_2_0_6, f_1_0_5, f_1_0_6, ABtemp, CDcom, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            4  B =            0
    f_4_0_t f_4_0_4 ( f_3_0_4,  f_3_0_5, f_2_0_4, f_2_0_5, ABtemp, CDcom, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for B =            4  L =            1
    f_4_1_t f_4_1_3 ( f_4_0_3,  f_4_0_4,  f_3_0_4, ABCDtemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for B =            3  L =            1
    f_3_1_t f_3_1_3 ( f_3_0_3,  f_3_0_4,  f_2_0_4, ABCDtemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for B =            4  L =            2
    f_4_2_t f_4_2_2 ( f_4_1_2,  f_4_1_3, f_4_0_2, f_4_0_3, CDtemp, ABcom, f_3_1_3, ABCDtemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for B =            2  L =            1
    f_2_1_t f_2_1_3 ( f_2_0_3,  f_2_0_4,  f_1_0_4, ABCDtemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for B =            3  L =            2
    f_3_2_t f_3_2_2 ( f_3_1_2,  f_3_1_3, f_3_0_2, f_3_0_3, CDtemp, ABcom, f_2_1_3, ABCDtemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for B =            4  L =            3
    f_4_3_t f_4_3_1 ( f_4_2_1,  f_4_2_2, f_4_1_1, f_4_1_2, CDtemp, ABcom, f_3_2_2, ABCDtemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            0  B =            1
    f_0_1_t f_0_1_2 ( VY( 0, 0, 2 ), VY( 0, 0, 3 ), Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            0  B =            1
    f_0_1_t f_0_1_3 ( VY( 0, 0, 3 ), VY( 0, 0, 4 ), Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            0  B =            2
    f_0_2_t f_0_2_2 ( f_0_1_2, f_0_1_3, VY( 0, 0, 2 ), VY( 0, 0, 3 ), CDtemp, ABcom, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            0  B =            1
    f_0_1_t f_0_1_4 ( VY( 0, 0, 4 ), VY( 0, 0, 5 ), Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            0  B =            2
    f_0_2_t f_0_2_3 ( f_0_1_3, f_0_1_4, VY( 0, 0, 3 ), VY( 0, 0, 4 ), CDtemp, ABcom, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            1  B =            2
    f_1_2_t f_1_2_2 ( f_0_2_2,  f_0_2_3,  f_0_1_3, ABCDtemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            0  B =            1
    f_0_1_t f_0_1_5 ( VY( 0, 0, 5 ), VY( 0, 0, 6 ), Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            0  B =            2
    f_0_2_t f_0_2_4 ( f_0_1_4, f_0_1_5, VY( 0, 0, 4 ), VY( 0, 0, 5 ), CDtemp, ABcom, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            1  B =            2
    f_1_2_t f_1_2_3 ( f_0_2_3,  f_0_2_4,  f_0_1_4, ABCDtemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            1  B =            1
    f_1_1_t f_1_1_3 ( f_0_1_3,  f_0_1_4,  VY( 0, 0, 4 ), ABCDtemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            2  B =            2
    f_2_2_t f_2_2_2 ( f_1_2_2,  f_1_2_3, f_0_2_2, f_0_2_3, ABtemp, CDcom, f_1_1_3, ABCDtemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for B =            3  L =            3
    f_3_3_t f_3_3_1 ( f_3_2_1,  f_3_2_2, f_3_1_1, f_3_1_2, CDtemp, ABcom, f_2_2_2, ABCDtemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for B =            4  L =            4
    f_4_4_t f_4_4_0 ( f_4_3_0,  f_4_3_1, f_4_2_0, f_4_2_1, CDtemp, ABcom, f_3_3_1, ABCDtemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // WRITE LAST FOR I =            4  J=           4
    LOC2(store, 20, 20, STOREDIM, STOREDIM) = f_4_4_0.x_20_20 ;
    LOC2(store, 20, 21, STOREDIM, STOREDIM) = f_4_4_0.x_20_21 ;
    LOC2(store, 20, 22, STOREDIM, STOREDIM) = f_4_4_0.x_20_22 ;
    LOC2(store, 20, 23, STOREDIM, STOREDIM) = f_4_4_0.x_20_23 ;
    LOC2(store, 20, 24, STOREDIM, STOREDIM) = f_4_4_0.x_20_24 ;
    LOC2(store, 20, 25, STOREDIM, STOREDIM) = f_4_4_0.x_20_25 ;
    LOC2(store, 20, 26, STOREDIM, STOREDIM) = f_4_4_0.x_20_26 ;
    LOC2(store, 20, 27, STOREDIM, STOREDIM) = f_4_4_0.x_20_27 ;
    LOC2(store, 20, 28, STOREDIM, STOREDIM) = f_4_4_0.x_20_28 ;
    LOC2(store, 20, 29, STOREDIM, STOREDIM) = f_4_4_0.x_20_29 ;
    LOC2(store, 20, 30, STOREDIM, STOREDIM) = f_4_4_0.x_20_30 ;
    LOC2(store, 20, 31, STOREDIM, STOREDIM) = f_4_4_0.x_20_31 ;
    LOC2(store, 20, 32, STOREDIM, STOREDIM) = f_4_4_0.x_20_32 ;
    LOC2(store, 20, 33, STOREDIM, STOREDIM) = f_4_4_0.x_20_33 ;
    LOC2(store, 20, 34, STOREDIM, STOREDIM) = f_4_4_0.x_20_34 ;
    LOC2(store, 21, 20, STOREDIM, STOREDIM) = f_4_4_0.x_21_20 ;
    LOC2(store, 21, 21, STOREDIM, STOREDIM) = f_4_4_0.x_21_21 ;
    LOC2(store, 21, 22, STOREDIM, STOREDIM) = f_4_4_0.x_21_22 ;
    LOC2(store, 21, 23, STOREDIM, STOREDIM) = f_4_4_0.x_21_23 ;
    LOC2(store, 21, 24, STOREDIM, STOREDIM) = f_4_4_0.x_21_24 ;
    LOC2(store, 21, 25, STOREDIM, STOREDIM) = f_4_4_0.x_21_25 ;
    LOC2(store, 21, 26, STOREDIM, STOREDIM) = f_4_4_0.x_21_26 ;
    LOC2(store, 21, 27, STOREDIM, STOREDIM) = f_4_4_0.x_21_27 ;
    LOC2(store, 21, 28, STOREDIM, STOREDIM) = f_4_4_0.x_21_28 ;
    LOC2(store, 21, 29, STOREDIM, STOREDIM) = f_4_4_0.x_21_29 ;
    LOC2(store, 21, 30, STOREDIM, STOREDIM) = f_4_4_0.x_21_30 ;
    LOC2(store, 21, 31, STOREDIM, STOREDIM) = f_4_4_0.x_21_31 ;
    LOC2(store, 21, 32, STOREDIM, STOREDIM) = f_4_4_0.x_21_32 ;
    LOC2(store, 21, 33, STOREDIM, STOREDIM) = f_4_4_0.x_21_33 ;
    LOC2(store, 21, 34, STOREDIM, STOREDIM) = f_4_4_0.x_21_34 ;
    LOC2(store, 22, 20, STOREDIM, STOREDIM) = f_4_4_0.x_22_20 ;
    LOC2(store, 22, 21, STOREDIM, STOREDIM) = f_4_4_0.x_22_21 ;
    LOC2(store, 22, 22, STOREDIM, STOREDIM) = f_4_4_0.x_22_22 ;
    LOC2(store, 22, 23, STOREDIM, STOREDIM) = f_4_4_0.x_22_23 ;
    LOC2(store, 22, 24, STOREDIM, STOREDIM) = f_4_4_0.x_22_24 ;
    LOC2(store, 22, 25, STOREDIM, STOREDIM) = f_4_4_0.x_22_25 ;
    LOC2(store, 22, 26, STOREDIM, STOREDIM) = f_4_4_0.x_22_26 ;
    LOC2(store, 22, 27, STOREDIM, STOREDIM) = f_4_4_0.x_22_27 ;
    LOC2(store, 22, 28, STOREDIM, STOREDIM) = f_4_4_0.x_22_28 ;
    LOC2(store, 22, 29, STOREDIM, STOREDIM) = f_4_4_0.x_22_29 ;
    LOC2(store, 22, 30, STOREDIM, STOREDIM) = f_4_4_0.x_22_30 ;
    LOC2(store, 22, 31, STOREDIM, STOREDIM) = f_4_4_0.x_22_31 ;
    LOC2(store, 22, 32, STOREDIM, STOREDIM) = f_4_4_0.x_22_32 ;
    LOC2(store, 22, 33, STOREDIM, STOREDIM) = f_4_4_0.x_22_33 ;
    LOC2(store, 22, 34, STOREDIM, STOREDIM) = f_4_4_0.x_22_34 ;
    LOC2(store, 23, 20, STOREDIM, STOREDIM) = f_4_4_0.x_23_20 ;
    LOC2(store, 23, 21, STOREDIM, STOREDIM) = f_4_4_0.x_23_21 ;
    LOC2(store, 23, 22, STOREDIM, STOREDIM) = f_4_4_0.x_23_22 ;
    LOC2(store, 23, 23, STOREDIM, STOREDIM) = f_4_4_0.x_23_23 ;
    LOC2(store, 23, 24, STOREDIM, STOREDIM) = f_4_4_0.x_23_24 ;
    LOC2(store, 23, 25, STOREDIM, STOREDIM) = f_4_4_0.x_23_25 ;
    LOC2(store, 23, 26, STOREDIM, STOREDIM) = f_4_4_0.x_23_26 ;
    LOC2(store, 23, 27, STOREDIM, STOREDIM) = f_4_4_0.x_23_27 ;
    LOC2(store, 23, 28, STOREDIM, STOREDIM) = f_4_4_0.x_23_28 ;
    LOC2(store, 23, 29, STOREDIM, STOREDIM) = f_4_4_0.x_23_29 ;
    LOC2(store, 23, 30, STOREDIM, STOREDIM) = f_4_4_0.x_23_30 ;
    LOC2(store, 23, 31, STOREDIM, STOREDIM) = f_4_4_0.x_23_31 ;
    LOC2(store, 23, 32, STOREDIM, STOREDIM) = f_4_4_0.x_23_32 ;
    LOC2(store, 23, 33, STOREDIM, STOREDIM) = f_4_4_0.x_23_33 ;
    LOC2(store, 23, 34, STOREDIM, STOREDIM) = f_4_4_0.x_23_34 ;
    LOC2(store, 24, 20, STOREDIM, STOREDIM) = f_4_4_0.x_24_20 ;
    LOC2(store, 24, 21, STOREDIM, STOREDIM) = f_4_4_0.x_24_21 ;
    LOC2(store, 24, 22, STOREDIM, STOREDIM) = f_4_4_0.x_24_22 ;
    LOC2(store, 24, 23, STOREDIM, STOREDIM) = f_4_4_0.x_24_23 ;
    LOC2(store, 24, 24, STOREDIM, STOREDIM) = f_4_4_0.x_24_24 ;
    LOC2(store, 24, 25, STOREDIM, STOREDIM) = f_4_4_0.x_24_25 ;
    LOC2(store, 24, 26, STOREDIM, STOREDIM) = f_4_4_0.x_24_26 ;
    LOC2(store, 24, 27, STOREDIM, STOREDIM) = f_4_4_0.x_24_27 ;
    LOC2(store, 24, 28, STOREDIM, STOREDIM) = f_4_4_0.x_24_28 ;
    LOC2(store, 24, 29, STOREDIM, STOREDIM) = f_4_4_0.x_24_29 ;
    LOC2(store, 24, 30, STOREDIM, STOREDIM) = f_4_4_0.x_24_30 ;
    LOC2(store, 24, 31, STOREDIM, STOREDIM) = f_4_4_0.x_24_31 ;
    LOC2(store, 24, 32, STOREDIM, STOREDIM) = f_4_4_0.x_24_32 ;
    LOC2(store, 24, 33, STOREDIM, STOREDIM) = f_4_4_0.x_24_33 ;
    LOC2(store, 24, 34, STOREDIM, STOREDIM) = f_4_4_0.x_24_34 ;
    LOC2(store, 25, 20, STOREDIM, STOREDIM) = f_4_4_0.x_25_20 ;
    LOC2(store, 25, 21, STOREDIM, STOREDIM) = f_4_4_0.x_25_21 ;
    LOC2(store, 25, 22, STOREDIM, STOREDIM) = f_4_4_0.x_25_22 ;
    LOC2(store, 25, 23, STOREDIM, STOREDIM) = f_4_4_0.x_25_23 ;
    LOC2(store, 25, 24, STOREDIM, STOREDIM) = f_4_4_0.x_25_24 ;
    LOC2(store, 25, 25, STOREDIM, STOREDIM) = f_4_4_0.x_25_25 ;
    LOC2(store, 25, 26, STOREDIM, STOREDIM) = f_4_4_0.x_25_26 ;
    LOC2(store, 25, 27, STOREDIM, STOREDIM) = f_4_4_0.x_25_27 ;
    LOC2(store, 25, 28, STOREDIM, STOREDIM) = f_4_4_0.x_25_28 ;
    LOC2(store, 25, 29, STOREDIM, STOREDIM) = f_4_4_0.x_25_29 ;
    LOC2(store, 25, 30, STOREDIM, STOREDIM) = f_4_4_0.x_25_30 ;
    LOC2(store, 25, 31, STOREDIM, STOREDIM) = f_4_4_0.x_25_31 ;
    LOC2(store, 25, 32, STOREDIM, STOREDIM) = f_4_4_0.x_25_32 ;
    LOC2(store, 25, 33, STOREDIM, STOREDIM) = f_4_4_0.x_25_33 ;
    LOC2(store, 25, 34, STOREDIM, STOREDIM) = f_4_4_0.x_25_34 ;
    LOC2(store, 26, 20, STOREDIM, STOREDIM) = f_4_4_0.x_26_20 ;
    LOC2(store, 26, 21, STOREDIM, STOREDIM) = f_4_4_0.x_26_21 ;
    LOC2(store, 26, 22, STOREDIM, STOREDIM) = f_4_4_0.x_26_22 ;
    LOC2(store, 26, 23, STOREDIM, STOREDIM) = f_4_4_0.x_26_23 ;
    LOC2(store, 26, 24, STOREDIM, STOREDIM) = f_4_4_0.x_26_24 ;
    LOC2(store, 26, 25, STOREDIM, STOREDIM) = f_4_4_0.x_26_25 ;
    LOC2(store, 26, 26, STOREDIM, STOREDIM) = f_4_4_0.x_26_26 ;
    LOC2(store, 26, 27, STOREDIM, STOREDIM) = f_4_4_0.x_26_27 ;
    LOC2(store, 26, 28, STOREDIM, STOREDIM) = f_4_4_0.x_26_28 ;
    LOC2(store, 26, 29, STOREDIM, STOREDIM) = f_4_4_0.x_26_29 ;
    LOC2(store, 26, 30, STOREDIM, STOREDIM) = f_4_4_0.x_26_30 ;
    LOC2(store, 26, 31, STOREDIM, STOREDIM) = f_4_4_0.x_26_31 ;
    LOC2(store, 26, 32, STOREDIM, STOREDIM) = f_4_4_0.x_26_32 ;
    LOC2(store, 26, 33, STOREDIM, STOREDIM) = f_4_4_0.x_26_33 ;
    LOC2(store, 26, 34, STOREDIM, STOREDIM) = f_4_4_0.x_26_34 ;
    LOC2(store, 27, 20, STOREDIM, STOREDIM) = f_4_4_0.x_27_20 ;
    LOC2(store, 27, 21, STOREDIM, STOREDIM) = f_4_4_0.x_27_21 ;
    LOC2(store, 27, 22, STOREDIM, STOREDIM) = f_4_4_0.x_27_22 ;
    LOC2(store, 27, 23, STOREDIM, STOREDIM) = f_4_4_0.x_27_23 ;
    LOC2(store, 27, 24, STOREDIM, STOREDIM) = f_4_4_0.x_27_24 ;
    LOC2(store, 27, 25, STOREDIM, STOREDIM) = f_4_4_0.x_27_25 ;
    LOC2(store, 27, 26, STOREDIM, STOREDIM) = f_4_4_0.x_27_26 ;
    LOC2(store, 27, 27, STOREDIM, STOREDIM) = f_4_4_0.x_27_27 ;
    LOC2(store, 27, 28, STOREDIM, STOREDIM) = f_4_4_0.x_27_28 ;
    LOC2(store, 27, 29, STOREDIM, STOREDIM) = f_4_4_0.x_27_29 ;
    LOC2(store, 27, 30, STOREDIM, STOREDIM) = f_4_4_0.x_27_30 ;
    LOC2(store, 27, 31, STOREDIM, STOREDIM) = f_4_4_0.x_27_31 ;
    LOC2(store, 27, 32, STOREDIM, STOREDIM) = f_4_4_0.x_27_32 ;
    LOC2(store, 27, 33, STOREDIM, STOREDIM) = f_4_4_0.x_27_33 ;
    LOC2(store, 27, 34, STOREDIM, STOREDIM) = f_4_4_0.x_27_34 ;
    LOC2(store, 28, 20, STOREDIM, STOREDIM) = f_4_4_0.x_28_20 ;
    LOC2(store, 28, 21, STOREDIM, STOREDIM) = f_4_4_0.x_28_21 ;
    LOC2(store, 28, 22, STOREDIM, STOREDIM) = f_4_4_0.x_28_22 ;
    LOC2(store, 28, 23, STOREDIM, STOREDIM) = f_4_4_0.x_28_23 ;
    LOC2(store, 28, 24, STOREDIM, STOREDIM) = f_4_4_0.x_28_24 ;
    LOC2(store, 28, 25, STOREDIM, STOREDIM) = f_4_4_0.x_28_25 ;
    LOC2(store, 28, 26, STOREDIM, STOREDIM) = f_4_4_0.x_28_26 ;
    LOC2(store, 28, 27, STOREDIM, STOREDIM) = f_4_4_0.x_28_27 ;
    LOC2(store, 28, 28, STOREDIM, STOREDIM) = f_4_4_0.x_28_28 ;
    LOC2(store, 28, 29, STOREDIM, STOREDIM) = f_4_4_0.x_28_29 ;
    LOC2(store, 28, 30, STOREDIM, STOREDIM) = f_4_4_0.x_28_30 ;
    LOC2(store, 28, 31, STOREDIM, STOREDIM) = f_4_4_0.x_28_31 ;
    LOC2(store, 28, 32, STOREDIM, STOREDIM) = f_4_4_0.x_28_32 ;
    LOC2(store, 28, 33, STOREDIM, STOREDIM) = f_4_4_0.x_28_33 ;
    LOC2(store, 28, 34, STOREDIM, STOREDIM) = f_4_4_0.x_28_34 ;
    LOC2(store, 29, 20, STOREDIM, STOREDIM) = f_4_4_0.x_29_20 ;
    LOC2(store, 29, 21, STOREDIM, STOREDIM) = f_4_4_0.x_29_21 ;
    LOC2(store, 29, 22, STOREDIM, STOREDIM) = f_4_4_0.x_29_22 ;
    LOC2(store, 29, 23, STOREDIM, STOREDIM) = f_4_4_0.x_29_23 ;
    LOC2(store, 29, 24, STOREDIM, STOREDIM) = f_4_4_0.x_29_24 ;
    LOC2(store, 29, 25, STOREDIM, STOREDIM) = f_4_4_0.x_29_25 ;
    LOC2(store, 29, 26, STOREDIM, STOREDIM) = f_4_4_0.x_29_26 ;
    LOC2(store, 29, 27, STOREDIM, STOREDIM) = f_4_4_0.x_29_27 ;
    LOC2(store, 29, 28, STOREDIM, STOREDIM) = f_4_4_0.x_29_28 ;
    LOC2(store, 29, 29, STOREDIM, STOREDIM) = f_4_4_0.x_29_29 ;
    LOC2(store, 29, 30, STOREDIM, STOREDIM) = f_4_4_0.x_29_30 ;
    LOC2(store, 29, 31, STOREDIM, STOREDIM) = f_4_4_0.x_29_31 ;
    LOC2(store, 29, 32, STOREDIM, STOREDIM) = f_4_4_0.x_29_32 ;
    LOC2(store, 29, 33, STOREDIM, STOREDIM) = f_4_4_0.x_29_33 ;
    LOC2(store, 29, 34, STOREDIM, STOREDIM) = f_4_4_0.x_29_34 ;
    LOC2(store, 30, 20, STOREDIM, STOREDIM) = f_4_4_0.x_30_20 ;
    LOC2(store, 30, 21, STOREDIM, STOREDIM) = f_4_4_0.x_30_21 ;
    LOC2(store, 30, 22, STOREDIM, STOREDIM) = f_4_4_0.x_30_22 ;
    LOC2(store, 30, 23, STOREDIM, STOREDIM) = f_4_4_0.x_30_23 ;
    LOC2(store, 30, 24, STOREDIM, STOREDIM) = f_4_4_0.x_30_24 ;
    LOC2(store, 30, 25, STOREDIM, STOREDIM) = f_4_4_0.x_30_25 ;
    LOC2(store, 30, 26, STOREDIM, STOREDIM) = f_4_4_0.x_30_26 ;
    LOC2(store, 30, 27, STOREDIM, STOREDIM) = f_4_4_0.x_30_27 ;
    LOC2(store, 30, 28, STOREDIM, STOREDIM) = f_4_4_0.x_30_28 ;
    LOC2(store, 30, 29, STOREDIM, STOREDIM) = f_4_4_0.x_30_29 ;
    LOC2(store, 30, 30, STOREDIM, STOREDIM) = f_4_4_0.x_30_30 ;
    LOC2(store, 30, 31, STOREDIM, STOREDIM) = f_4_4_0.x_30_31 ;
    LOC2(store, 30, 32, STOREDIM, STOREDIM) = f_4_4_0.x_30_32 ;
    LOC2(store, 30, 33, STOREDIM, STOREDIM) = f_4_4_0.x_30_33 ;
    LOC2(store, 30, 34, STOREDIM, STOREDIM) = f_4_4_0.x_30_34 ;
    LOC2(store, 31, 20, STOREDIM, STOREDIM) = f_4_4_0.x_31_20 ;
    LOC2(store, 31, 21, STOREDIM, STOREDIM) = f_4_4_0.x_31_21 ;
    LOC2(store, 31, 22, STOREDIM, STOREDIM) = f_4_4_0.x_31_22 ;
    LOC2(store, 31, 23, STOREDIM, STOREDIM) = f_4_4_0.x_31_23 ;
    LOC2(store, 31, 24, STOREDIM, STOREDIM) = f_4_4_0.x_31_24 ;
    LOC2(store, 31, 25, STOREDIM, STOREDIM) = f_4_4_0.x_31_25 ;
    LOC2(store, 31, 26, STOREDIM, STOREDIM) = f_4_4_0.x_31_26 ;
    LOC2(store, 31, 27, STOREDIM, STOREDIM) = f_4_4_0.x_31_27 ;
    LOC2(store, 31, 28, STOREDIM, STOREDIM) = f_4_4_0.x_31_28 ;
    LOC2(store, 31, 29, STOREDIM, STOREDIM) = f_4_4_0.x_31_29 ;
    LOC2(store, 31, 30, STOREDIM, STOREDIM) = f_4_4_0.x_31_30 ;
    LOC2(store, 31, 31, STOREDIM, STOREDIM) = f_4_4_0.x_31_31 ;
    LOC2(store, 31, 32, STOREDIM, STOREDIM) = f_4_4_0.x_31_32 ;
    LOC2(store, 31, 33, STOREDIM, STOREDIM) = f_4_4_0.x_31_33 ;
    LOC2(store, 31, 34, STOREDIM, STOREDIM) = f_4_4_0.x_31_34 ;
    LOC2(store, 32, 20, STOREDIM, STOREDIM) = f_4_4_0.x_32_20 ;
    LOC2(store, 32, 21, STOREDIM, STOREDIM) = f_4_4_0.x_32_21 ;
    LOC2(store, 32, 22, STOREDIM, STOREDIM) = f_4_4_0.x_32_22 ;
    LOC2(store, 32, 23, STOREDIM, STOREDIM) = f_4_4_0.x_32_23 ;
    LOC2(store, 32, 24, STOREDIM, STOREDIM) = f_4_4_0.x_32_24 ;
    LOC2(store, 32, 25, STOREDIM, STOREDIM) = f_4_4_0.x_32_25 ;
    LOC2(store, 32, 26, STOREDIM, STOREDIM) = f_4_4_0.x_32_26 ;
    LOC2(store, 32, 27, STOREDIM, STOREDIM) = f_4_4_0.x_32_27 ;
    LOC2(store, 32, 28, STOREDIM, STOREDIM) = f_4_4_0.x_32_28 ;
    LOC2(store, 32, 29, STOREDIM, STOREDIM) = f_4_4_0.x_32_29 ;
    LOC2(store, 32, 30, STOREDIM, STOREDIM) = f_4_4_0.x_32_30 ;
    LOC2(store, 32, 31, STOREDIM, STOREDIM) = f_4_4_0.x_32_31 ;
    LOC2(store, 32, 32, STOREDIM, STOREDIM) = f_4_4_0.x_32_32 ;
    LOC2(store, 32, 33, STOREDIM, STOREDIM) = f_4_4_0.x_32_33 ;
    LOC2(store, 32, 34, STOREDIM, STOREDIM) = f_4_4_0.x_32_34 ;
    LOC2(store, 33, 20, STOREDIM, STOREDIM) = f_4_4_0.x_33_20 ;
    LOC2(store, 33, 21, STOREDIM, STOREDIM) = f_4_4_0.x_33_21 ;
    LOC2(store, 33, 22, STOREDIM, STOREDIM) = f_4_4_0.x_33_22 ;
    LOC2(store, 33, 23, STOREDIM, STOREDIM) = f_4_4_0.x_33_23 ;
    LOC2(store, 33, 24, STOREDIM, STOREDIM) = f_4_4_0.x_33_24 ;
    LOC2(store, 33, 25, STOREDIM, STOREDIM) = f_4_4_0.x_33_25 ;
    LOC2(store, 33, 26, STOREDIM, STOREDIM) = f_4_4_0.x_33_26 ;
    LOC2(store, 33, 27, STOREDIM, STOREDIM) = f_4_4_0.x_33_27 ;
    LOC2(store, 33, 28, STOREDIM, STOREDIM) = f_4_4_0.x_33_28 ;
    LOC2(store, 33, 29, STOREDIM, STOREDIM) = f_4_4_0.x_33_29 ;
    LOC2(store, 33, 30, STOREDIM, STOREDIM) = f_4_4_0.x_33_30 ;
    LOC2(store, 33, 31, STOREDIM, STOREDIM) = f_4_4_0.x_33_31 ;
    LOC2(store, 33, 32, STOREDIM, STOREDIM) = f_4_4_0.x_33_32 ;
    LOC2(store, 33, 33, STOREDIM, STOREDIM) = f_4_4_0.x_33_33 ;
    LOC2(store, 33, 34, STOREDIM, STOREDIM) = f_4_4_0.x_33_34 ;
    LOC2(store, 34, 20, STOREDIM, STOREDIM) = f_4_4_0.x_34_20 ;
    LOC2(store, 34, 21, STOREDIM, STOREDIM) = f_4_4_0.x_34_21 ;
    LOC2(store, 34, 22, STOREDIM, STOREDIM) = f_4_4_0.x_34_22 ;
    LOC2(store, 34, 23, STOREDIM, STOREDIM) = f_4_4_0.x_34_23 ;
    LOC2(store, 34, 24, STOREDIM, STOREDIM) = f_4_4_0.x_34_24 ;
    LOC2(store, 34, 25, STOREDIM, STOREDIM) = f_4_4_0.x_34_25 ;
    LOC2(store, 34, 26, STOREDIM, STOREDIM) = f_4_4_0.x_34_26 ;
    LOC2(store, 34, 27, STOREDIM, STOREDIM) = f_4_4_0.x_34_27 ;
    LOC2(store, 34, 28, STOREDIM, STOREDIM) = f_4_4_0.x_34_28 ;
    LOC2(store, 34, 29, STOREDIM, STOREDIM) = f_4_4_0.x_34_29 ;
    LOC2(store, 34, 30, STOREDIM, STOREDIM) = f_4_4_0.x_34_30 ;
    LOC2(store, 34, 31, STOREDIM, STOREDIM) = f_4_4_0.x_34_31 ;
    LOC2(store, 34, 32, STOREDIM, STOREDIM) = f_4_4_0.x_34_32 ;
    LOC2(store, 34, 33, STOREDIM, STOREDIM) = f_4_4_0.x_34_33 ;
    LOC2(store, 34, 34, STOREDIM, STOREDIM) = f_4_4_0.x_34_34 ;
}
