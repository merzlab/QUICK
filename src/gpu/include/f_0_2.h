__device__ __inline__  void h_0_2(QUICKDouble* YVerticalTemp, QUICKDouble* store,
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

    // WRITE LAST FOR I =            0  J=           2
    LOCSTORE(store,  0,  4, STOREDIM, STOREDIM) += f_0_2_0.x_0_4 ;
    LOCSTORE(store,  0,  5, STOREDIM, STOREDIM) += f_0_2_0.x_0_5 ;
    LOCSTORE(store,  0,  6, STOREDIM, STOREDIM) += f_0_2_0.x_0_6 ;
    LOCSTORE(store,  0,  7, STOREDIM, STOREDIM) += f_0_2_0.x_0_7 ;
    LOCSTORE(store,  0,  8, STOREDIM, STOREDIM) += f_0_2_0.x_0_8 ;
    LOCSTORE(store,  0,  9, STOREDIM, STOREDIM) += f_0_2_0.x_0_9 ;
}
