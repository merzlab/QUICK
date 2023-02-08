__device__ __inline__  void h_1_3(QUICKDouble* YVerticalTemp, QUICKDouble* store,
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

    // WRITE LAST FOR I =            1  J=           3
    LOCSTORE(store,  1, 10, STOREDIM, STOREDIM) += f_1_3_0.x_1_10 ;
    LOCSTORE(store,  1, 11, STOREDIM, STOREDIM) += f_1_3_0.x_1_11 ;
    LOCSTORE(store,  1, 12, STOREDIM, STOREDIM) += f_1_3_0.x_1_12 ;
    LOCSTORE(store,  1, 13, STOREDIM, STOREDIM) += f_1_3_0.x_1_13 ;
    LOCSTORE(store,  1, 14, STOREDIM, STOREDIM) += f_1_3_0.x_1_14 ;
    LOCSTORE(store,  1, 15, STOREDIM, STOREDIM) += f_1_3_0.x_1_15 ;
    LOCSTORE(store,  1, 16, STOREDIM, STOREDIM) += f_1_3_0.x_1_16 ;
    LOCSTORE(store,  1, 17, STOREDIM, STOREDIM) += f_1_3_0.x_1_17 ;
    LOCSTORE(store,  1, 18, STOREDIM, STOREDIM) += f_1_3_0.x_1_18 ;
    LOCSTORE(store,  1, 19, STOREDIM, STOREDIM) += f_1_3_0.x_1_19 ;
    LOCSTORE(store,  2, 10, STOREDIM, STOREDIM) += f_1_3_0.x_2_10 ;
    LOCSTORE(store,  2, 11, STOREDIM, STOREDIM) += f_1_3_0.x_2_11 ;
    LOCSTORE(store,  2, 12, STOREDIM, STOREDIM) += f_1_3_0.x_2_12 ;
    LOCSTORE(store,  2, 13, STOREDIM, STOREDIM) += f_1_3_0.x_2_13 ;
    LOCSTORE(store,  2, 14, STOREDIM, STOREDIM) += f_1_3_0.x_2_14 ;
    LOCSTORE(store,  2, 15, STOREDIM, STOREDIM) += f_1_3_0.x_2_15 ;
    LOCSTORE(store,  2, 16, STOREDIM, STOREDIM) += f_1_3_0.x_2_16 ;
    LOCSTORE(store,  2, 17, STOREDIM, STOREDIM) += f_1_3_0.x_2_17 ;
    LOCSTORE(store,  2, 18, STOREDIM, STOREDIM) += f_1_3_0.x_2_18 ;
    LOCSTORE(store,  2, 19, STOREDIM, STOREDIM) += f_1_3_0.x_2_19 ;
    LOCSTORE(store,  3, 10, STOREDIM, STOREDIM) += f_1_3_0.x_3_10 ;
    LOCSTORE(store,  3, 11, STOREDIM, STOREDIM) += f_1_3_0.x_3_11 ;
    LOCSTORE(store,  3, 12, STOREDIM, STOREDIM) += f_1_3_0.x_3_12 ;
    LOCSTORE(store,  3, 13, STOREDIM, STOREDIM) += f_1_3_0.x_3_13 ;
    LOCSTORE(store,  3, 14, STOREDIM, STOREDIM) += f_1_3_0.x_3_14 ;
    LOCSTORE(store,  3, 15, STOREDIM, STOREDIM) += f_1_3_0.x_3_15 ;
    LOCSTORE(store,  3, 16, STOREDIM, STOREDIM) += f_1_3_0.x_3_16 ;
    LOCSTORE(store,  3, 17, STOREDIM, STOREDIM) += f_1_3_0.x_3_17 ;
    LOCSTORE(store,  3, 18, STOREDIM, STOREDIM) += f_1_3_0.x_3_18 ;
    LOCSTORE(store,  3, 19, STOREDIM, STOREDIM) += f_1_3_0.x_3_19 ;
}
