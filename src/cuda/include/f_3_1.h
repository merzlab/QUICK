__device__ __inline__   void h_3_1(QUICKDouble* YVerticalTemp, QUICKDouble* store,
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

    // call for B =            3  L =            1
    f_3_1_t f_3_1_0 ( f_3_0_0,  f_3_0_1,  f_2_0_1, ABCDtemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // WRITE LAST FOR I =            3  J=           1
    LOC2(store, 10,  1, STOREDIM, STOREDIM) += f_3_1_0.x_10_1 ;
    LOC2(store, 10,  2, STOREDIM, STOREDIM) += f_3_1_0.x_10_2 ;
    LOC2(store, 10,  3, STOREDIM, STOREDIM) += f_3_1_0.x_10_3 ;
    LOC2(store, 11,  1, STOREDIM, STOREDIM) += f_3_1_0.x_11_1 ;
    LOC2(store, 11,  2, STOREDIM, STOREDIM) += f_3_1_0.x_11_2 ;
    LOC2(store, 11,  3, STOREDIM, STOREDIM) += f_3_1_0.x_11_3 ;
    LOC2(store, 12,  1, STOREDIM, STOREDIM) += f_3_1_0.x_12_1 ;
    LOC2(store, 12,  2, STOREDIM, STOREDIM) += f_3_1_0.x_12_2 ;
    LOC2(store, 12,  3, STOREDIM, STOREDIM) += f_3_1_0.x_12_3 ;
    LOC2(store, 13,  1, STOREDIM, STOREDIM) += f_3_1_0.x_13_1 ;
    LOC2(store, 13,  2, STOREDIM, STOREDIM) += f_3_1_0.x_13_2 ;
    LOC2(store, 13,  3, STOREDIM, STOREDIM) += f_3_1_0.x_13_3 ;
    LOC2(store, 14,  1, STOREDIM, STOREDIM) += f_3_1_0.x_14_1 ;
    LOC2(store, 14,  2, STOREDIM, STOREDIM) += f_3_1_0.x_14_2 ;
    LOC2(store, 14,  3, STOREDIM, STOREDIM) += f_3_1_0.x_14_3 ;
    LOC2(store, 15,  1, STOREDIM, STOREDIM) += f_3_1_0.x_15_1 ;
    LOC2(store, 15,  2, STOREDIM, STOREDIM) += f_3_1_0.x_15_2 ;
    LOC2(store, 15,  3, STOREDIM, STOREDIM) += f_3_1_0.x_15_3 ;
    LOC2(store, 16,  1, STOREDIM, STOREDIM) += f_3_1_0.x_16_1 ;
    LOC2(store, 16,  2, STOREDIM, STOREDIM) += f_3_1_0.x_16_2 ;
    LOC2(store, 16,  3, STOREDIM, STOREDIM) += f_3_1_0.x_16_3 ;
    LOC2(store, 17,  1, STOREDIM, STOREDIM) += f_3_1_0.x_17_1 ;
    LOC2(store, 17,  2, STOREDIM, STOREDIM) += f_3_1_0.x_17_2 ;
    LOC2(store, 17,  3, STOREDIM, STOREDIM) += f_3_1_0.x_17_3 ;
    LOC2(store, 18,  1, STOREDIM, STOREDIM) += f_3_1_0.x_18_1 ;
    LOC2(store, 18,  2, STOREDIM, STOREDIM) += f_3_1_0.x_18_2 ;
    LOC2(store, 18,  3, STOREDIM, STOREDIM) += f_3_1_0.x_18_3 ;
    LOC2(store, 19,  1, STOREDIM, STOREDIM) += f_3_1_0.x_19_1 ;
    LOC2(store, 19,  2, STOREDIM, STOREDIM) += f_3_1_0.x_19_2 ;
    LOC2(store, 19,  3, STOREDIM, STOREDIM) += f_3_1_0.x_19_3 ;
}
