__device__ __inline__  void h2_2_1(QUICKDouble* YVerticalTemp, QUICKDouble* store,
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

    // call for B =            2  L =            1
    f_2_1_t f_2_1_0 ( f_2_0_0,  f_2_0_1,  f_1_0_1, ABCDtemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // WRITE LAST FOR I =            2  J=           1
    LOC2(store,  4,  1, STOREDIM, STOREDIM) = f_2_1_0.x_4_1 ;
    LOC2(store,  4,  2, STOREDIM, STOREDIM) = f_2_1_0.x_4_2 ;
    LOC2(store,  4,  3, STOREDIM, STOREDIM) = f_2_1_0.x_4_3 ;
    LOC2(store,  5,  1, STOREDIM, STOREDIM) = f_2_1_0.x_5_1 ;
    LOC2(store,  5,  2, STOREDIM, STOREDIM) = f_2_1_0.x_5_2 ;
    LOC2(store,  5,  3, STOREDIM, STOREDIM) = f_2_1_0.x_5_3 ;
    LOC2(store,  6,  1, STOREDIM, STOREDIM) = f_2_1_0.x_6_1 ;
    LOC2(store,  6,  2, STOREDIM, STOREDIM) = f_2_1_0.x_6_2 ;
    LOC2(store,  6,  3, STOREDIM, STOREDIM) = f_2_1_0.x_6_3 ;
    LOC2(store,  7,  1, STOREDIM, STOREDIM) = f_2_1_0.x_7_1 ;
    LOC2(store,  7,  2, STOREDIM, STOREDIM) = f_2_1_0.x_7_2 ;
    LOC2(store,  7,  3, STOREDIM, STOREDIM) = f_2_1_0.x_7_3 ;
    LOC2(store,  8,  1, STOREDIM, STOREDIM) = f_2_1_0.x_8_1 ;
    LOC2(store,  8,  2, STOREDIM, STOREDIM) = f_2_1_0.x_8_2 ;
    LOC2(store,  8,  3, STOREDIM, STOREDIM) = f_2_1_0.x_8_3 ;
    LOC2(store,  9,  1, STOREDIM, STOREDIM) = f_2_1_0.x_9_1 ;
    LOC2(store,  9,  2, STOREDIM, STOREDIM) = f_2_1_0.x_9_2 ;
    LOC2(store,  9,  3, STOREDIM, STOREDIM) = f_2_1_0.x_9_3 ;
}
