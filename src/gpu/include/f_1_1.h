__device__ __inline__  void h_1_1(QUICKDouble* YVerticalTemp, QUICKDouble* store,
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

    // call for L =            1  B =            1
    f_1_1_t f_1_1_0 ( f_0_1_0,  f_0_1_1,  VY( 0, 0, 1 ), ABCDtemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // WRITE LAST FOR I =            1  J=           1
    LOCSTORE(store,  1,  1, STOREDIM, STOREDIM) += f_1_1_0.x_1_1 ;
    LOCSTORE(store,  1,  2, STOREDIM, STOREDIM) += f_1_1_0.x_1_2 ;
    LOCSTORE(store,  1,  3, STOREDIM, STOREDIM) += f_1_1_0.x_1_3 ;
    LOCSTORE(store,  2,  1, STOREDIM, STOREDIM) += f_1_1_0.x_2_1 ;
    LOCSTORE(store,  2,  2, STOREDIM, STOREDIM) += f_1_1_0.x_2_2 ;
    LOCSTORE(store,  2,  3, STOREDIM, STOREDIM) += f_1_1_0.x_2_3 ;
    LOCSTORE(store,  3,  1, STOREDIM, STOREDIM) += f_1_1_0.x_3_1 ;
    LOCSTORE(store,  3,  2, STOREDIM, STOREDIM) += f_1_1_0.x_3_2 ;
    LOCSTORE(store,  3,  3, STOREDIM, STOREDIM) += f_1_1_0.x_3_3 ;
}
