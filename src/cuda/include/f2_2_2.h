__device__ __inline__  void h2_2_2(QUICKDouble* YVerticalTemp, QUICKDouble* store,
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

    // call for L =            1  B =            2
    f_1_2_t f_1_2_0 ( f_0_2_0,  f_0_2_1,  f_0_1_1, ABCDtemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            0  B =            1
    f_0_1_t f_0_1_3 ( VY( 0, 0, 3 ), VY( 0, 0, 4 ), Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            0  B =            2
    f_0_2_t f_0_2_2 ( f_0_1_2, f_0_1_3, VY( 0, 0, 2 ), VY( 0, 0, 3 ), CDtemp, ABcom, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            1  B =            2
    f_1_2_t f_1_2_1 ( f_0_2_1,  f_0_2_2,  f_0_1_2, ABCDtemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            1  B =            1
    f_1_1_t f_1_1_1 ( f_0_1_1,  f_0_1_2,  VY( 0, 0, 2 ), ABCDtemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            2  B =            2
    f_2_2_t f_2_2_0 ( f_1_2_0,  f_1_2_1, f_0_2_0, f_0_2_1, ABtemp, CDcom, f_1_1_1, ABCDtemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // WRITE LAST FOR I =            2  J=           2
    LOC2(store,  4,  4, STOREDIM, STOREDIM) = f_2_2_0.x_4_4 ;
    LOC2(store,  4,  5, STOREDIM, STOREDIM) = f_2_2_0.x_4_5 ;
    LOC2(store,  4,  6, STOREDIM, STOREDIM) = f_2_2_0.x_4_6 ;
    LOC2(store,  4,  7, STOREDIM, STOREDIM) = f_2_2_0.x_4_7 ;
    LOC2(store,  4,  8, STOREDIM, STOREDIM) = f_2_2_0.x_4_8 ;
    LOC2(store,  4,  9, STOREDIM, STOREDIM) = f_2_2_0.x_4_9 ;
    LOC2(store,  5,  4, STOREDIM, STOREDIM) = f_2_2_0.x_5_4 ;
    LOC2(store,  5,  5, STOREDIM, STOREDIM) = f_2_2_0.x_5_5 ;
    LOC2(store,  5,  6, STOREDIM, STOREDIM) = f_2_2_0.x_5_6 ;
    LOC2(store,  5,  7, STOREDIM, STOREDIM) = f_2_2_0.x_5_7 ;
    LOC2(store,  5,  8, STOREDIM, STOREDIM) = f_2_2_0.x_5_8 ;
    LOC2(store,  5,  9, STOREDIM, STOREDIM) = f_2_2_0.x_5_9 ;
    LOC2(store,  6,  4, STOREDIM, STOREDIM) = f_2_2_0.x_6_4 ;
    LOC2(store,  6,  5, STOREDIM, STOREDIM) = f_2_2_0.x_6_5 ;
    LOC2(store,  6,  6, STOREDIM, STOREDIM) = f_2_2_0.x_6_6 ;
    LOC2(store,  6,  7, STOREDIM, STOREDIM) = f_2_2_0.x_6_7 ;
    LOC2(store,  6,  8, STOREDIM, STOREDIM) = f_2_2_0.x_6_8 ;
    LOC2(store,  6,  9, STOREDIM, STOREDIM) = f_2_2_0.x_6_9 ;
    LOC2(store,  7,  4, STOREDIM, STOREDIM) = f_2_2_0.x_7_4 ;
    LOC2(store,  7,  5, STOREDIM, STOREDIM) = f_2_2_0.x_7_5 ;
    LOC2(store,  7,  6, STOREDIM, STOREDIM) = f_2_2_0.x_7_6 ;
    LOC2(store,  7,  7, STOREDIM, STOREDIM) = f_2_2_0.x_7_7 ;
    LOC2(store,  7,  8, STOREDIM, STOREDIM) = f_2_2_0.x_7_8 ;
    LOC2(store,  7,  9, STOREDIM, STOREDIM) = f_2_2_0.x_7_9 ;
    LOC2(store,  8,  4, STOREDIM, STOREDIM) = f_2_2_0.x_8_4 ;
    LOC2(store,  8,  5, STOREDIM, STOREDIM) = f_2_2_0.x_8_5 ;
    LOC2(store,  8,  6, STOREDIM, STOREDIM) = f_2_2_0.x_8_6 ;
    LOC2(store,  8,  7, STOREDIM, STOREDIM) = f_2_2_0.x_8_7 ;
    LOC2(store,  8,  8, STOREDIM, STOREDIM) = f_2_2_0.x_8_8 ;
    LOC2(store,  8,  9, STOREDIM, STOREDIM) = f_2_2_0.x_8_9 ;
    LOC2(store,  9,  4, STOREDIM, STOREDIM) = f_2_2_0.x_9_4 ;
    LOC2(store,  9,  5, STOREDIM, STOREDIM) = f_2_2_0.x_9_5 ;
    LOC2(store,  9,  6, STOREDIM, STOREDIM) = f_2_2_0.x_9_6 ;
    LOC2(store,  9,  7, STOREDIM, STOREDIM) = f_2_2_0.x_9_7 ;
    LOC2(store,  9,  8, STOREDIM, STOREDIM) = f_2_2_0.x_9_8 ;
    LOC2(store,  9,  9, STOREDIM, STOREDIM) = f_2_2_0.x_9_9 ;
}
