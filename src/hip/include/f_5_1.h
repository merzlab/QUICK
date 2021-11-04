__device__ __inline__   void h_5_1(QUICKDouble* YVerticalTemp, QUICKDouble* store,
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

    // call for L =            4  B =            0
    f_4_0_t f_4_0_0 ( f_3_0_0,  f_3_0_1, f_2_0_0, f_2_0_1, ABtemp, CDcom, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            1  B =            0
    f_1_0_t f_1_0_4 ( VY( 0, 0, 4 ),  VY( 0, 0, 5 ), Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            2  B =            0
    f_2_0_t f_2_0_3 ( f_1_0_3,  f_1_0_4, VY( 0, 0, 3 ), VY( 0, 0, 4 ), ABtemp, CDcom, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            3  B =            0
    f_3_0_t f_3_0_2 ( f_2_0_2,  f_2_0_3, f_1_0_2, f_1_0_3, ABtemp, CDcom, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            4  B =            0
    f_4_0_t f_4_0_1 ( f_3_0_1,  f_3_0_2, f_2_0_1, f_2_0_2, ABtemp, CDcom, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for B =            5  L =            0
    f_5_0_t f_5_0_0 ( f_4_0_0, f_4_0_1, f_3_0_0, f_3_0_1, ABtemp, CDcom, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            1  B =            0
    f_1_0_t f_1_0_5 ( VY( 0, 0, 5 ),  VY( 0, 0, 6 ), Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            2  B =            0
    f_2_0_t f_2_0_4 ( f_1_0_4,  f_1_0_5, VY( 0, 0, 4 ), VY( 0, 0, 5 ), ABtemp, CDcom, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            3  B =            0
    f_3_0_t f_3_0_3 ( f_2_0_3,  f_2_0_4, f_1_0_3, f_1_0_4, ABtemp, CDcom, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            4  B =            0
    f_4_0_t f_4_0_2 ( f_3_0_2,  f_3_0_3, f_2_0_2, f_2_0_3, ABtemp, CDcom, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for B =            5  L =            0
    f_5_0_t f_5_0_1 ( f_4_0_1, f_4_0_2, f_3_0_1, f_3_0_2, ABtemp, CDcom, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for B =            5  L =            1
    f_5_1_t f_5_1_0 ( f_5_0_0,  f_5_0_1,  f_4_0_1, ABCDtemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // WRITE LAST FOR I =            5  J=           1
    LOCSTORE(store, 35,  1, STOREDIM, STOREDIM) += f_5_1_0.x_35_1 ;
    LOCSTORE(store, 35,  2, STOREDIM, STOREDIM) += f_5_1_0.x_35_2 ;
    LOCSTORE(store, 35,  3, STOREDIM, STOREDIM) += f_5_1_0.x_35_3 ;
    LOCSTORE(store, 36,  1, STOREDIM, STOREDIM) += f_5_1_0.x_36_1 ;
    LOCSTORE(store, 36,  2, STOREDIM, STOREDIM) += f_5_1_0.x_36_2 ;
    LOCSTORE(store, 36,  3, STOREDIM, STOREDIM) += f_5_1_0.x_36_3 ;
    LOCSTORE(store, 37,  1, STOREDIM, STOREDIM) += f_5_1_0.x_37_1 ;
    LOCSTORE(store, 37,  2, STOREDIM, STOREDIM) += f_5_1_0.x_37_2 ;
    LOCSTORE(store, 37,  3, STOREDIM, STOREDIM) += f_5_1_0.x_37_3 ;
    LOCSTORE(store, 38,  1, STOREDIM, STOREDIM) += f_5_1_0.x_38_1 ;
    LOCSTORE(store, 38,  2, STOREDIM, STOREDIM) += f_5_1_0.x_38_2 ;
    LOCSTORE(store, 38,  3, STOREDIM, STOREDIM) += f_5_1_0.x_38_3 ;
    LOCSTORE(store, 39,  1, STOREDIM, STOREDIM) += f_5_1_0.x_39_1 ;
    LOCSTORE(store, 39,  2, STOREDIM, STOREDIM) += f_5_1_0.x_39_2 ;
    LOCSTORE(store, 39,  3, STOREDIM, STOREDIM) += f_5_1_0.x_39_3 ;
    LOCSTORE(store, 40,  1, STOREDIM, STOREDIM) += f_5_1_0.x_40_1 ;
    LOCSTORE(store, 40,  2, STOREDIM, STOREDIM) += f_5_1_0.x_40_2 ;
    LOCSTORE(store, 40,  3, STOREDIM, STOREDIM) += f_5_1_0.x_40_3 ;
    LOCSTORE(store, 41,  1, STOREDIM, STOREDIM) += f_5_1_0.x_41_1 ;
    LOCSTORE(store, 41,  2, STOREDIM, STOREDIM) += f_5_1_0.x_41_2 ;
    LOCSTORE(store, 41,  3, STOREDIM, STOREDIM) += f_5_1_0.x_41_3 ;
    LOCSTORE(store, 42,  1, STOREDIM, STOREDIM) += f_5_1_0.x_42_1 ;
    LOCSTORE(store, 42,  2, STOREDIM, STOREDIM) += f_5_1_0.x_42_2 ;
    LOCSTORE(store, 42,  3, STOREDIM, STOREDIM) += f_5_1_0.x_42_3 ;
    LOCSTORE(store, 43,  1, STOREDIM, STOREDIM) += f_5_1_0.x_43_1 ;
    LOCSTORE(store, 43,  2, STOREDIM, STOREDIM) += f_5_1_0.x_43_2 ;
    LOCSTORE(store, 43,  3, STOREDIM, STOREDIM) += f_5_1_0.x_43_3 ;
    LOCSTORE(store, 44,  1, STOREDIM, STOREDIM) += f_5_1_0.x_44_1 ;
    LOCSTORE(store, 44,  2, STOREDIM, STOREDIM) += f_5_1_0.x_44_2 ;
    LOCSTORE(store, 44,  3, STOREDIM, STOREDIM) += f_5_1_0.x_44_3 ;
    LOCSTORE(store, 45,  1, STOREDIM, STOREDIM) += f_5_1_0.x_45_1 ;
    LOCSTORE(store, 45,  2, STOREDIM, STOREDIM) += f_5_1_0.x_45_2 ;
    LOCSTORE(store, 45,  3, STOREDIM, STOREDIM) += f_5_1_0.x_45_3 ;
    LOCSTORE(store, 46,  1, STOREDIM, STOREDIM) += f_5_1_0.x_46_1 ;
    LOCSTORE(store, 46,  2, STOREDIM, STOREDIM) += f_5_1_0.x_46_2 ;
    LOCSTORE(store, 46,  3, STOREDIM, STOREDIM) += f_5_1_0.x_46_3 ;
    LOCSTORE(store, 47,  1, STOREDIM, STOREDIM) += f_5_1_0.x_47_1 ;
    LOCSTORE(store, 47,  2, STOREDIM, STOREDIM) += f_5_1_0.x_47_2 ;
    LOCSTORE(store, 47,  3, STOREDIM, STOREDIM) += f_5_1_0.x_47_3 ;
    LOCSTORE(store, 48,  1, STOREDIM, STOREDIM) += f_5_1_0.x_48_1 ;
    LOCSTORE(store, 48,  2, STOREDIM, STOREDIM) += f_5_1_0.x_48_2 ;
    LOCSTORE(store, 48,  3, STOREDIM, STOREDIM) += f_5_1_0.x_48_3 ;
    LOCSTORE(store, 49,  1, STOREDIM, STOREDIM) += f_5_1_0.x_49_1 ;
    LOCSTORE(store, 49,  2, STOREDIM, STOREDIM) += f_5_1_0.x_49_2 ;
    LOCSTORE(store, 49,  3, STOREDIM, STOREDIM) += f_5_1_0.x_49_3 ;
    LOCSTORE(store, 50,  1, STOREDIM, STOREDIM) += f_5_1_0.x_50_1 ;
    LOCSTORE(store, 50,  2, STOREDIM, STOREDIM) += f_5_1_0.x_50_2 ;
    LOCSTORE(store, 50,  3, STOREDIM, STOREDIM) += f_5_1_0.x_50_3 ;
    LOCSTORE(store, 51,  1, STOREDIM, STOREDIM) += f_5_1_0.x_51_1 ;
    LOCSTORE(store, 51,  2, STOREDIM, STOREDIM) += f_5_1_0.x_51_2 ;
    LOCSTORE(store, 51,  3, STOREDIM, STOREDIM) += f_5_1_0.x_51_3 ;
    LOCSTORE(store, 52,  1, STOREDIM, STOREDIM) += f_5_1_0.x_52_1 ;
    LOCSTORE(store, 52,  2, STOREDIM, STOREDIM) += f_5_1_0.x_52_2 ;
    LOCSTORE(store, 52,  3, STOREDIM, STOREDIM) += f_5_1_0.x_52_3 ;
    LOCSTORE(store, 53,  1, STOREDIM, STOREDIM) += f_5_1_0.x_53_1 ;
    LOCSTORE(store, 53,  2, STOREDIM, STOREDIM) += f_5_1_0.x_53_2 ;
    LOCSTORE(store, 53,  3, STOREDIM, STOREDIM) += f_5_1_0.x_53_3 ;
    LOCSTORE(store, 54,  1, STOREDIM, STOREDIM) += f_5_1_0.x_54_1 ;
    LOCSTORE(store, 54,  2, STOREDIM, STOREDIM) += f_5_1_0.x_54_2 ;
    LOCSTORE(store, 54,  3, STOREDIM, STOREDIM) += f_5_1_0.x_54_3 ;
    LOCSTORE(store, 55,  1, STOREDIM, STOREDIM) += f_5_1_0.x_55_1 ;
    LOCSTORE(store, 55,  2, STOREDIM, STOREDIM) += f_5_1_0.x_55_2 ;
    LOCSTORE(store, 55,  3, STOREDIM, STOREDIM) += f_5_1_0.x_55_3 ;
}
