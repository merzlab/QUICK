__device__ __inline__  void h2_2_0(QUICKDouble* YVerticalTemp, QUICKDouble* store,
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

    // WRITE LAST FOR I =            2  J=           0
    LOC2(store,  4,  0, STOREDIM, STOREDIM) = f_2_0_0.x_4_0 ;
    LOC2(store,  5,  0, STOREDIM, STOREDIM) = f_2_0_0.x_5_0 ;
    LOC2(store,  6,  0, STOREDIM, STOREDIM) = f_2_0_0.x_6_0 ;
    LOC2(store,  7,  0, STOREDIM, STOREDIM) = f_2_0_0.x_7_0 ;
    LOC2(store,  8,  0, STOREDIM, STOREDIM) = f_2_0_0.x_8_0 ;
    LOC2(store,  9,  0, STOREDIM, STOREDIM) = f_2_0_0.x_9_0 ;
}
