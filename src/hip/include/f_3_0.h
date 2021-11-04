__device__ __inline__   void h_3_0(QUICKDouble* YVerticalTemp, QUICKDouble* store,
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

    // WRITE LAST FOR I =            3  J=           0
    LOCSTORE(store, 10,  0, STOREDIM, STOREDIM) += f_3_0_0.x_10_0 ;
    LOCSTORE(store, 11,  0, STOREDIM, STOREDIM) += f_3_0_0.x_11_0 ;
    LOCSTORE(store, 12,  0, STOREDIM, STOREDIM) += f_3_0_0.x_12_0 ;
    LOCSTORE(store, 13,  0, STOREDIM, STOREDIM) += f_3_0_0.x_13_0 ;
    LOCSTORE(store, 14,  0, STOREDIM, STOREDIM) += f_3_0_0.x_14_0 ;
    LOCSTORE(store, 15,  0, STOREDIM, STOREDIM) += f_3_0_0.x_15_0 ;
    LOCSTORE(store, 16,  0, STOREDIM, STOREDIM) += f_3_0_0.x_16_0 ;
    LOCSTORE(store, 17,  0, STOREDIM, STOREDIM) += f_3_0_0.x_17_0 ;
    LOCSTORE(store, 18,  0, STOREDIM, STOREDIM) += f_3_0_0.x_18_0 ;
    LOCSTORE(store, 19,  0, STOREDIM, STOREDIM) += f_3_0_0.x_19_0 ;
}
