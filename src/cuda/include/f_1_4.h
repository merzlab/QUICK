__device__ __inline__  void h_1_4(QUICKDouble* YVerticalTemp, QUICKDouble* store,
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

    // WRITE LAST FOR I =            1  J=           4
    LOC2(store,  1, 20, STOREDIM, STOREDIM) += f_1_4_0.x_1_20 ;
    LOC2(store,  1, 21, STOREDIM, STOREDIM) += f_1_4_0.x_1_21 ;
    LOC2(store,  1, 22, STOREDIM, STOREDIM) += f_1_4_0.x_1_22 ;
    LOC2(store,  1, 23, STOREDIM, STOREDIM) += f_1_4_0.x_1_23 ;
    LOC2(store,  1, 24, STOREDIM, STOREDIM) += f_1_4_0.x_1_24 ;
    LOC2(store,  1, 25, STOREDIM, STOREDIM) += f_1_4_0.x_1_25 ;
    LOC2(store,  1, 26, STOREDIM, STOREDIM) += f_1_4_0.x_1_26 ;
    LOC2(store,  1, 27, STOREDIM, STOREDIM) += f_1_4_0.x_1_27 ;
    LOC2(store,  1, 28, STOREDIM, STOREDIM) += f_1_4_0.x_1_28 ;
    LOC2(store,  1, 29, STOREDIM, STOREDIM) += f_1_4_0.x_1_29 ;
    LOC2(store,  1, 30, STOREDIM, STOREDIM) += f_1_4_0.x_1_30 ;
    LOC2(store,  1, 31, STOREDIM, STOREDIM) += f_1_4_0.x_1_31 ;
    LOC2(store,  1, 32, STOREDIM, STOREDIM) += f_1_4_0.x_1_32 ;
    LOC2(store,  1, 33, STOREDIM, STOREDIM) += f_1_4_0.x_1_33 ;
    LOC2(store,  1, 34, STOREDIM, STOREDIM) += f_1_4_0.x_1_34 ;
    LOC2(store,  2, 20, STOREDIM, STOREDIM) += f_1_4_0.x_2_20 ;
    LOC2(store,  2, 21, STOREDIM, STOREDIM) += f_1_4_0.x_2_21 ;
    LOC2(store,  2, 22, STOREDIM, STOREDIM) += f_1_4_0.x_2_22 ;
    LOC2(store,  2, 23, STOREDIM, STOREDIM) += f_1_4_0.x_2_23 ;
    LOC2(store,  2, 24, STOREDIM, STOREDIM) += f_1_4_0.x_2_24 ;
    LOC2(store,  2, 25, STOREDIM, STOREDIM) += f_1_4_0.x_2_25 ;
    LOC2(store,  2, 26, STOREDIM, STOREDIM) += f_1_4_0.x_2_26 ;
    LOC2(store,  2, 27, STOREDIM, STOREDIM) += f_1_4_0.x_2_27 ;
    LOC2(store,  2, 28, STOREDIM, STOREDIM) += f_1_4_0.x_2_28 ;
    LOC2(store,  2, 29, STOREDIM, STOREDIM) += f_1_4_0.x_2_29 ;
    LOC2(store,  2, 30, STOREDIM, STOREDIM) += f_1_4_0.x_2_30 ;
    LOC2(store,  2, 31, STOREDIM, STOREDIM) += f_1_4_0.x_2_31 ;
    LOC2(store,  2, 32, STOREDIM, STOREDIM) += f_1_4_0.x_2_32 ;
    LOC2(store,  2, 33, STOREDIM, STOREDIM) += f_1_4_0.x_2_33 ;
    LOC2(store,  2, 34, STOREDIM, STOREDIM) += f_1_4_0.x_2_34 ;
    LOC2(store,  3, 20, STOREDIM, STOREDIM) += f_1_4_0.x_3_20 ;
    LOC2(store,  3, 21, STOREDIM, STOREDIM) += f_1_4_0.x_3_21 ;
    LOC2(store,  3, 22, STOREDIM, STOREDIM) += f_1_4_0.x_3_22 ;
    LOC2(store,  3, 23, STOREDIM, STOREDIM) += f_1_4_0.x_3_23 ;
    LOC2(store,  3, 24, STOREDIM, STOREDIM) += f_1_4_0.x_3_24 ;
    LOC2(store,  3, 25, STOREDIM, STOREDIM) += f_1_4_0.x_3_25 ;
    LOC2(store,  3, 26, STOREDIM, STOREDIM) += f_1_4_0.x_3_26 ;
    LOC2(store,  3, 27, STOREDIM, STOREDIM) += f_1_4_0.x_3_27 ;
    LOC2(store,  3, 28, STOREDIM, STOREDIM) += f_1_4_0.x_3_28 ;
    LOC2(store,  3, 29, STOREDIM, STOREDIM) += f_1_4_0.x_3_29 ;
    LOC2(store,  3, 30, STOREDIM, STOREDIM) += f_1_4_0.x_3_30 ;
    LOC2(store,  3, 31, STOREDIM, STOREDIM) += f_1_4_0.x_3_31 ;
    LOC2(store,  3, 32, STOREDIM, STOREDIM) += f_1_4_0.x_3_32 ;
    LOC2(store,  3, 33, STOREDIM, STOREDIM) += f_1_4_0.x_3_33 ;
    LOC2(store,  3, 34, STOREDIM, STOREDIM) += f_1_4_0.x_3_34 ;
}
