__device__ __inline__  void h_0_3(QUICKDouble* YVerticalTemp, QUICKDouble* store,
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

    // WRITE LAST FOR I =            0  J=           3
    LOC2(store,  0, 10, STOREDIM, STOREDIM) += f_0_3_0.x_0_10 ;
    LOC2(store,  0, 11, STOREDIM, STOREDIM) += f_0_3_0.x_0_11 ;
    LOC2(store,  0, 12, STOREDIM, STOREDIM) += f_0_3_0.x_0_12 ;
    LOC2(store,  0, 13, STOREDIM, STOREDIM) += f_0_3_0.x_0_13 ;
    LOC2(store,  0, 14, STOREDIM, STOREDIM) += f_0_3_0.x_0_14 ;
    LOC2(store,  0, 15, STOREDIM, STOREDIM) += f_0_3_0.x_0_15 ;
    LOC2(store,  0, 16, STOREDIM, STOREDIM) += f_0_3_0.x_0_16 ;
    LOC2(store,  0, 17, STOREDIM, STOREDIM) += f_0_3_0.x_0_17 ;
    LOC2(store,  0, 18, STOREDIM, STOREDIM) += f_0_3_0.x_0_18 ;
    LOC2(store,  0, 19, STOREDIM, STOREDIM) += f_0_3_0.x_0_19 ;
}
