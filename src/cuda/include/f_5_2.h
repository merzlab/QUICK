__device__ __inline__   void h_5_2(QUICKDouble* YVerticalTemp, QUICKDouble* store,
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

    // call for L =            1  B =            0
    f_1_0_t f_1_0_6 ( VY( 0, 0, 6 ),  VY( 0, 0, 7 ), Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            2  B =            0
    f_2_0_t f_2_0_5 ( f_1_0_5,  f_1_0_6, VY( 0, 0, 5 ), VY( 0, 0, 6 ), ABtemp, CDcom, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            3  B =            0
    f_3_0_t f_3_0_4 ( f_2_0_4,  f_2_0_5, f_1_0_4, f_1_0_5, ABtemp, CDcom, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            4  B =            0
    f_4_0_t f_4_0_3 ( f_3_0_3,  f_3_0_4, f_2_0_3, f_2_0_4, ABtemp, CDcom, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for B =            5  L =            0
    f_5_0_t f_5_0_2 ( f_4_0_2, f_4_0_3, f_3_0_2, f_3_0_3, ABtemp, CDcom, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for B =            5  L =            1
    f_5_1_t f_5_1_1 ( f_5_0_1,  f_5_0_2,  f_4_0_2, ABCDtemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for B =            4  L =            1
    f_4_1_t f_4_1_1 ( f_4_0_1,  f_4_0_2,  f_3_0_2, ABCDtemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for B =            5  L =            2
    f_5_2_t f_5_2_0 ( f_5_1_0,  f_5_1_1, f_5_0_0, f_5_0_1, CDtemp, ABcom, f_4_1_1, ABCDtemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // WRITE LAST FOR I =            5  J=           2
    LOC2(store, 35,  4, STOREDIM, STOREDIM) += f_5_2_0.x_35_4 ;
    LOC2(store, 35,  5, STOREDIM, STOREDIM) += f_5_2_0.x_35_5 ;
    LOC2(store, 35,  6, STOREDIM, STOREDIM) += f_5_2_0.x_35_6 ;
    LOC2(store, 35,  7, STOREDIM, STOREDIM) += f_5_2_0.x_35_7 ;
    LOC2(store, 35,  8, STOREDIM, STOREDIM) += f_5_2_0.x_35_8 ;
    LOC2(store, 35,  9, STOREDIM, STOREDIM) += f_5_2_0.x_35_9 ;
    LOC2(store, 36,  4, STOREDIM, STOREDIM) += f_5_2_0.x_36_4 ;
    LOC2(store, 36,  5, STOREDIM, STOREDIM) += f_5_2_0.x_36_5 ;
    LOC2(store, 36,  6, STOREDIM, STOREDIM) += f_5_2_0.x_36_6 ;
    LOC2(store, 36,  7, STOREDIM, STOREDIM) += f_5_2_0.x_36_7 ;
    LOC2(store, 36,  8, STOREDIM, STOREDIM) += f_5_2_0.x_36_8 ;
    LOC2(store, 36,  9, STOREDIM, STOREDIM) += f_5_2_0.x_36_9 ;
    LOC2(store, 37,  4, STOREDIM, STOREDIM) += f_5_2_0.x_37_4 ;
    LOC2(store, 37,  5, STOREDIM, STOREDIM) += f_5_2_0.x_37_5 ;
    LOC2(store, 37,  6, STOREDIM, STOREDIM) += f_5_2_0.x_37_6 ;
    LOC2(store, 37,  7, STOREDIM, STOREDIM) += f_5_2_0.x_37_7 ;
    LOC2(store, 37,  8, STOREDIM, STOREDIM) += f_5_2_0.x_37_8 ;
    LOC2(store, 37,  9, STOREDIM, STOREDIM) += f_5_2_0.x_37_9 ;
    LOC2(store, 38,  4, STOREDIM, STOREDIM) += f_5_2_0.x_38_4 ;
    LOC2(store, 38,  5, STOREDIM, STOREDIM) += f_5_2_0.x_38_5 ;
    LOC2(store, 38,  6, STOREDIM, STOREDIM) += f_5_2_0.x_38_6 ;
    LOC2(store, 38,  7, STOREDIM, STOREDIM) += f_5_2_0.x_38_7 ;
    LOC2(store, 38,  8, STOREDIM, STOREDIM) += f_5_2_0.x_38_8 ;
    LOC2(store, 38,  9, STOREDIM, STOREDIM) += f_5_2_0.x_38_9 ;
    LOC2(store, 39,  4, STOREDIM, STOREDIM) += f_5_2_0.x_39_4 ;
    LOC2(store, 39,  5, STOREDIM, STOREDIM) += f_5_2_0.x_39_5 ;
    LOC2(store, 39,  6, STOREDIM, STOREDIM) += f_5_2_0.x_39_6 ;
    LOC2(store, 39,  7, STOREDIM, STOREDIM) += f_5_2_0.x_39_7 ;
    LOC2(store, 39,  8, STOREDIM, STOREDIM) += f_5_2_0.x_39_8 ;
    LOC2(store, 39,  9, STOREDIM, STOREDIM) += f_5_2_0.x_39_9 ;
    LOC2(store, 40,  4, STOREDIM, STOREDIM) += f_5_2_0.x_40_4 ;
    LOC2(store, 40,  5, STOREDIM, STOREDIM) += f_5_2_0.x_40_5 ;
    LOC2(store, 40,  6, STOREDIM, STOREDIM) += f_5_2_0.x_40_6 ;
    LOC2(store, 40,  7, STOREDIM, STOREDIM) += f_5_2_0.x_40_7 ;
    LOC2(store, 40,  8, STOREDIM, STOREDIM) += f_5_2_0.x_40_8 ;
    LOC2(store, 40,  9, STOREDIM, STOREDIM) += f_5_2_0.x_40_9 ;
    LOC2(store, 41,  4, STOREDIM, STOREDIM) += f_5_2_0.x_41_4 ;
    LOC2(store, 41,  5, STOREDIM, STOREDIM) += f_5_2_0.x_41_5 ;
    LOC2(store, 41,  6, STOREDIM, STOREDIM) += f_5_2_0.x_41_6 ;
    LOC2(store, 41,  7, STOREDIM, STOREDIM) += f_5_2_0.x_41_7 ;
    LOC2(store, 41,  8, STOREDIM, STOREDIM) += f_5_2_0.x_41_8 ;
    LOC2(store, 41,  9, STOREDIM, STOREDIM) += f_5_2_0.x_41_9 ;
    LOC2(store, 42,  4, STOREDIM, STOREDIM) += f_5_2_0.x_42_4 ;
    LOC2(store, 42,  5, STOREDIM, STOREDIM) += f_5_2_0.x_42_5 ;
    LOC2(store, 42,  6, STOREDIM, STOREDIM) += f_5_2_0.x_42_6 ;
    LOC2(store, 42,  7, STOREDIM, STOREDIM) += f_5_2_0.x_42_7 ;
    LOC2(store, 42,  8, STOREDIM, STOREDIM) += f_5_2_0.x_42_8 ;
    LOC2(store, 42,  9, STOREDIM, STOREDIM) += f_5_2_0.x_42_9 ;
    LOC2(store, 43,  4, STOREDIM, STOREDIM) += f_5_2_0.x_43_4 ;
    LOC2(store, 43,  5, STOREDIM, STOREDIM) += f_5_2_0.x_43_5 ;
    LOC2(store, 43,  6, STOREDIM, STOREDIM) += f_5_2_0.x_43_6 ;
    LOC2(store, 43,  7, STOREDIM, STOREDIM) += f_5_2_0.x_43_7 ;
    LOC2(store, 43,  8, STOREDIM, STOREDIM) += f_5_2_0.x_43_8 ;
    LOC2(store, 43,  9, STOREDIM, STOREDIM) += f_5_2_0.x_43_9 ;
    LOC2(store, 44,  4, STOREDIM, STOREDIM) += f_5_2_0.x_44_4 ;
    LOC2(store, 44,  5, STOREDIM, STOREDIM) += f_5_2_0.x_44_5 ;
    LOC2(store, 44,  6, STOREDIM, STOREDIM) += f_5_2_0.x_44_6 ;
    LOC2(store, 44,  7, STOREDIM, STOREDIM) += f_5_2_0.x_44_7 ;
    LOC2(store, 44,  8, STOREDIM, STOREDIM) += f_5_2_0.x_44_8 ;
    LOC2(store, 44,  9, STOREDIM, STOREDIM) += f_5_2_0.x_44_9 ;
    LOC2(store, 45,  4, STOREDIM, STOREDIM) += f_5_2_0.x_45_4 ;
    LOC2(store, 45,  5, STOREDIM, STOREDIM) += f_5_2_0.x_45_5 ;
    LOC2(store, 45,  6, STOREDIM, STOREDIM) += f_5_2_0.x_45_6 ;
    LOC2(store, 45,  7, STOREDIM, STOREDIM) += f_5_2_0.x_45_7 ;
    LOC2(store, 45,  8, STOREDIM, STOREDIM) += f_5_2_0.x_45_8 ;
    LOC2(store, 45,  9, STOREDIM, STOREDIM) += f_5_2_0.x_45_9 ;
    LOC2(store, 46,  4, STOREDIM, STOREDIM) += f_5_2_0.x_46_4 ;
    LOC2(store, 46,  5, STOREDIM, STOREDIM) += f_5_2_0.x_46_5 ;
    LOC2(store, 46,  6, STOREDIM, STOREDIM) += f_5_2_0.x_46_6 ;
    LOC2(store, 46,  7, STOREDIM, STOREDIM) += f_5_2_0.x_46_7 ;
    LOC2(store, 46,  8, STOREDIM, STOREDIM) += f_5_2_0.x_46_8 ;
    LOC2(store, 46,  9, STOREDIM, STOREDIM) += f_5_2_0.x_46_9 ;
    LOC2(store, 47,  4, STOREDIM, STOREDIM) += f_5_2_0.x_47_4 ;
    LOC2(store, 47,  5, STOREDIM, STOREDIM) += f_5_2_0.x_47_5 ;
    LOC2(store, 47,  6, STOREDIM, STOREDIM) += f_5_2_0.x_47_6 ;
    LOC2(store, 47,  7, STOREDIM, STOREDIM) += f_5_2_0.x_47_7 ;
    LOC2(store, 47,  8, STOREDIM, STOREDIM) += f_5_2_0.x_47_8 ;
    LOC2(store, 47,  9, STOREDIM, STOREDIM) += f_5_2_0.x_47_9 ;
    LOC2(store, 48,  4, STOREDIM, STOREDIM) += f_5_2_0.x_48_4 ;
    LOC2(store, 48,  5, STOREDIM, STOREDIM) += f_5_2_0.x_48_5 ;
    LOC2(store, 48,  6, STOREDIM, STOREDIM) += f_5_2_0.x_48_6 ;
    LOC2(store, 48,  7, STOREDIM, STOREDIM) += f_5_2_0.x_48_7 ;
    LOC2(store, 48,  8, STOREDIM, STOREDIM) += f_5_2_0.x_48_8 ;
    LOC2(store, 48,  9, STOREDIM, STOREDIM) += f_5_2_0.x_48_9 ;
    LOC2(store, 49,  4, STOREDIM, STOREDIM) += f_5_2_0.x_49_4 ;
    LOC2(store, 49,  5, STOREDIM, STOREDIM) += f_5_2_0.x_49_5 ;
    LOC2(store, 49,  6, STOREDIM, STOREDIM) += f_5_2_0.x_49_6 ;
    LOC2(store, 49,  7, STOREDIM, STOREDIM) += f_5_2_0.x_49_7 ;
    LOC2(store, 49,  8, STOREDIM, STOREDIM) += f_5_2_0.x_49_8 ;
    LOC2(store, 49,  9, STOREDIM, STOREDIM) += f_5_2_0.x_49_9 ;
    LOC2(store, 50,  4, STOREDIM, STOREDIM) += f_5_2_0.x_50_4 ;
    LOC2(store, 50,  5, STOREDIM, STOREDIM) += f_5_2_0.x_50_5 ;
    LOC2(store, 50,  6, STOREDIM, STOREDIM) += f_5_2_0.x_50_6 ;
    LOC2(store, 50,  7, STOREDIM, STOREDIM) += f_5_2_0.x_50_7 ;
    LOC2(store, 50,  8, STOREDIM, STOREDIM) += f_5_2_0.x_50_8 ;
    LOC2(store, 50,  9, STOREDIM, STOREDIM) += f_5_2_0.x_50_9 ;
    LOC2(store, 51,  4, STOREDIM, STOREDIM) += f_5_2_0.x_51_4 ;
    LOC2(store, 51,  5, STOREDIM, STOREDIM) += f_5_2_0.x_51_5 ;
    LOC2(store, 51,  6, STOREDIM, STOREDIM) += f_5_2_0.x_51_6 ;
    LOC2(store, 51,  7, STOREDIM, STOREDIM) += f_5_2_0.x_51_7 ;
    LOC2(store, 51,  8, STOREDIM, STOREDIM) += f_5_2_0.x_51_8 ;
    LOC2(store, 51,  9, STOREDIM, STOREDIM) += f_5_2_0.x_51_9 ;
    LOC2(store, 52,  4, STOREDIM, STOREDIM) += f_5_2_0.x_52_4 ;
    LOC2(store, 52,  5, STOREDIM, STOREDIM) += f_5_2_0.x_52_5 ;
    LOC2(store, 52,  6, STOREDIM, STOREDIM) += f_5_2_0.x_52_6 ;
    LOC2(store, 52,  7, STOREDIM, STOREDIM) += f_5_2_0.x_52_7 ;
    LOC2(store, 52,  8, STOREDIM, STOREDIM) += f_5_2_0.x_52_8 ;
    LOC2(store, 52,  9, STOREDIM, STOREDIM) += f_5_2_0.x_52_9 ;
    LOC2(store, 53,  4, STOREDIM, STOREDIM) += f_5_2_0.x_53_4 ;
    LOC2(store, 53,  5, STOREDIM, STOREDIM) += f_5_2_0.x_53_5 ;
    LOC2(store, 53,  6, STOREDIM, STOREDIM) += f_5_2_0.x_53_6 ;
    LOC2(store, 53,  7, STOREDIM, STOREDIM) += f_5_2_0.x_53_7 ;
    LOC2(store, 53,  8, STOREDIM, STOREDIM) += f_5_2_0.x_53_8 ;
    LOC2(store, 53,  9, STOREDIM, STOREDIM) += f_5_2_0.x_53_9 ;
    LOC2(store, 54,  4, STOREDIM, STOREDIM) += f_5_2_0.x_54_4 ;
    LOC2(store, 54,  5, STOREDIM, STOREDIM) += f_5_2_0.x_54_5 ;
    LOC2(store, 54,  6, STOREDIM, STOREDIM) += f_5_2_0.x_54_6 ;
    LOC2(store, 54,  7, STOREDIM, STOREDIM) += f_5_2_0.x_54_7 ;
    LOC2(store, 54,  8, STOREDIM, STOREDIM) += f_5_2_0.x_54_8 ;
    LOC2(store, 54,  9, STOREDIM, STOREDIM) += f_5_2_0.x_54_9 ;
    LOC2(store, 55,  4, STOREDIM, STOREDIM) += f_5_2_0.x_55_4 ;
    LOC2(store, 55,  5, STOREDIM, STOREDIM) += f_5_2_0.x_55_5 ;
    LOC2(store, 55,  6, STOREDIM, STOREDIM) += f_5_2_0.x_55_6 ;
    LOC2(store, 55,  7, STOREDIM, STOREDIM) += f_5_2_0.x_55_7 ;
    LOC2(store, 55,  8, STOREDIM, STOREDIM) += f_5_2_0.x_55_8 ;
    LOC2(store, 55,  9, STOREDIM, STOREDIM) += f_5_2_0.x_55_9 ;
}
