__device__ __inline__  void h_0_4(QUICKDouble* YVerticalTemp, QUICKDouble* store,
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

    // WRITE LAST FOR I =            0  J=           4
    LOC2(store,  0, 20, STOREDIM, STOREDIM) += f_0_4_0.x_0_20 ;
    LOC2(store,  0, 21, STOREDIM, STOREDIM) += f_0_4_0.x_0_21 ;
    LOC2(store,  0, 22, STOREDIM, STOREDIM) += f_0_4_0.x_0_22 ;
    LOC2(store,  0, 23, STOREDIM, STOREDIM) += f_0_4_0.x_0_23 ;
    LOC2(store,  0, 24, STOREDIM, STOREDIM) += f_0_4_0.x_0_24 ;
    LOC2(store,  0, 25, STOREDIM, STOREDIM) += f_0_4_0.x_0_25 ;
    LOC2(store,  0, 26, STOREDIM, STOREDIM) += f_0_4_0.x_0_26 ;
    LOC2(store,  0, 27, STOREDIM, STOREDIM) += f_0_4_0.x_0_27 ;
    LOC2(store,  0, 28, STOREDIM, STOREDIM) += f_0_4_0.x_0_28 ;
    LOC2(store,  0, 29, STOREDIM, STOREDIM) += f_0_4_0.x_0_29 ;
    LOC2(store,  0, 30, STOREDIM, STOREDIM) += f_0_4_0.x_0_30 ;
    LOC2(store,  0, 31, STOREDIM, STOREDIM) += f_0_4_0.x_0_31 ;
    LOC2(store,  0, 32, STOREDIM, STOREDIM) += f_0_4_0.x_0_32 ;
    LOC2(store,  0, 33, STOREDIM, STOREDIM) += f_0_4_0.x_0_33 ;
    LOC2(store,  0, 34, STOREDIM, STOREDIM) += f_0_4_0.x_0_34 ;
}
