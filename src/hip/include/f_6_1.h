__device__ __inline__   void h_6_1(QUICKDouble* YVerticalTemp, QUICKDouble* store,
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

    // call for B =            6  L =            0
    f_6_0_t f_6_0_0 ( f_5_0_0, f_5_0_1, f_4_0_0, f_4_0_1, ABtemp, CDcom, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

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

    // call for B =            6  L =            0
    f_6_0_t f_6_0_1 ( f_5_0_1, f_5_0_2, f_4_0_1, f_4_0_2, ABtemp, CDcom, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for B =            6  L =            1
    f_6_1_t f_6_1_0 ( f_6_0_0,  f_6_0_1,  f_5_0_1, ABCDtemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // WRITE LAST FOR I =            6  J=           1
    LOCSTORE(store, 56,  1, STOREDIM, STOREDIM) += f_6_1_0.x_56_1 ;
    LOCSTORE(store, 56,  2, STOREDIM, STOREDIM) += f_6_1_0.x_56_2 ;
    LOCSTORE(store, 56,  3, STOREDIM, STOREDIM) += f_6_1_0.x_56_3 ;
    LOCSTORE(store, 57,  1, STOREDIM, STOREDIM) += f_6_1_0.x_57_1 ;
    LOCSTORE(store, 57,  2, STOREDIM, STOREDIM) += f_6_1_0.x_57_2 ;
    LOCSTORE(store, 57,  3, STOREDIM, STOREDIM) += f_6_1_0.x_57_3 ;
    LOCSTORE(store, 58,  1, STOREDIM, STOREDIM) += f_6_1_0.x_58_1 ;
    LOCSTORE(store, 58,  2, STOREDIM, STOREDIM) += f_6_1_0.x_58_2 ;
    LOCSTORE(store, 58,  3, STOREDIM, STOREDIM) += f_6_1_0.x_58_3 ;
    LOCSTORE(store, 59,  1, STOREDIM, STOREDIM) += f_6_1_0.x_59_1 ;
    LOCSTORE(store, 59,  2, STOREDIM, STOREDIM) += f_6_1_0.x_59_2 ;
    LOCSTORE(store, 59,  3, STOREDIM, STOREDIM) += f_6_1_0.x_59_3 ;
    LOCSTORE(store, 60,  1, STOREDIM, STOREDIM) += f_6_1_0.x_60_1 ;
    LOCSTORE(store, 60,  2, STOREDIM, STOREDIM) += f_6_1_0.x_60_2 ;
    LOCSTORE(store, 60,  3, STOREDIM, STOREDIM) += f_6_1_0.x_60_3 ;
    LOCSTORE(store, 61,  1, STOREDIM, STOREDIM) += f_6_1_0.x_61_1 ;
    LOCSTORE(store, 61,  2, STOREDIM, STOREDIM) += f_6_1_0.x_61_2 ;
    LOCSTORE(store, 61,  3, STOREDIM, STOREDIM) += f_6_1_0.x_61_3 ;
    LOCSTORE(store, 62,  1, STOREDIM, STOREDIM) += f_6_1_0.x_62_1 ;
    LOCSTORE(store, 62,  2, STOREDIM, STOREDIM) += f_6_1_0.x_62_2 ;
    LOCSTORE(store, 62,  3, STOREDIM, STOREDIM) += f_6_1_0.x_62_3 ;
    LOCSTORE(store, 63,  1, STOREDIM, STOREDIM) += f_6_1_0.x_63_1 ;
    LOCSTORE(store, 63,  2, STOREDIM, STOREDIM) += f_6_1_0.x_63_2 ;
    LOCSTORE(store, 63,  3, STOREDIM, STOREDIM) += f_6_1_0.x_63_3 ;
    LOCSTORE(store, 64,  1, STOREDIM, STOREDIM) += f_6_1_0.x_64_1 ;
    LOCSTORE(store, 64,  2, STOREDIM, STOREDIM) += f_6_1_0.x_64_2 ;
    LOCSTORE(store, 64,  3, STOREDIM, STOREDIM) += f_6_1_0.x_64_3 ;
    LOCSTORE(store, 65,  1, STOREDIM, STOREDIM) += f_6_1_0.x_65_1 ;
    LOCSTORE(store, 65,  2, STOREDIM, STOREDIM) += f_6_1_0.x_65_2 ;
    LOCSTORE(store, 65,  3, STOREDIM, STOREDIM) += f_6_1_0.x_65_3 ;
    LOCSTORE(store, 66,  1, STOREDIM, STOREDIM) += f_6_1_0.x_66_1 ;
    LOCSTORE(store, 66,  2, STOREDIM, STOREDIM) += f_6_1_0.x_66_2 ;
    LOCSTORE(store, 66,  3, STOREDIM, STOREDIM) += f_6_1_0.x_66_3 ;
    LOCSTORE(store, 67,  1, STOREDIM, STOREDIM) += f_6_1_0.x_67_1 ;
    LOCSTORE(store, 67,  2, STOREDIM, STOREDIM) += f_6_1_0.x_67_2 ;
    LOCSTORE(store, 67,  3, STOREDIM, STOREDIM) += f_6_1_0.x_67_3 ;
    LOCSTORE(store, 68,  1, STOREDIM, STOREDIM) += f_6_1_0.x_68_1 ;
    LOCSTORE(store, 68,  2, STOREDIM, STOREDIM) += f_6_1_0.x_68_2 ;
    LOCSTORE(store, 68,  3, STOREDIM, STOREDIM) += f_6_1_0.x_68_3 ;
    LOCSTORE(store, 69,  1, STOREDIM, STOREDIM) += f_6_1_0.x_69_1 ;
    LOCSTORE(store, 69,  2, STOREDIM, STOREDIM) += f_6_1_0.x_69_2 ;
    LOCSTORE(store, 69,  3, STOREDIM, STOREDIM) += f_6_1_0.x_69_3 ;
    LOCSTORE(store, 70,  1, STOREDIM, STOREDIM) += f_6_1_0.x_70_1 ;
    LOCSTORE(store, 70,  2, STOREDIM, STOREDIM) += f_6_1_0.x_70_2 ;
    LOCSTORE(store, 70,  3, STOREDIM, STOREDIM) += f_6_1_0.x_70_3 ;
    LOCSTORE(store, 71,  1, STOREDIM, STOREDIM) += f_6_1_0.x_71_1 ;
    LOCSTORE(store, 71,  2, STOREDIM, STOREDIM) += f_6_1_0.x_71_2 ;
    LOCSTORE(store, 71,  3, STOREDIM, STOREDIM) += f_6_1_0.x_71_3 ;
    LOCSTORE(store, 72,  1, STOREDIM, STOREDIM) += f_6_1_0.x_72_1 ;
    LOCSTORE(store, 72,  2, STOREDIM, STOREDIM) += f_6_1_0.x_72_2 ;
    LOCSTORE(store, 72,  3, STOREDIM, STOREDIM) += f_6_1_0.x_72_3 ;
    LOCSTORE(store, 73,  1, STOREDIM, STOREDIM) += f_6_1_0.x_73_1 ;
    LOCSTORE(store, 73,  2, STOREDIM, STOREDIM) += f_6_1_0.x_73_2 ;
    LOCSTORE(store, 73,  3, STOREDIM, STOREDIM) += f_6_1_0.x_73_3 ;
    LOCSTORE(store, 74,  1, STOREDIM, STOREDIM) += f_6_1_0.x_74_1 ;
    LOCSTORE(store, 74,  2, STOREDIM, STOREDIM) += f_6_1_0.x_74_2 ;
    LOCSTORE(store, 74,  3, STOREDIM, STOREDIM) += f_6_1_0.x_74_3 ;
    LOCSTORE(store, 75,  1, STOREDIM, STOREDIM) += f_6_1_0.x_75_1 ;
    LOCSTORE(store, 75,  2, STOREDIM, STOREDIM) += f_6_1_0.x_75_2 ;
    LOCSTORE(store, 75,  3, STOREDIM, STOREDIM) += f_6_1_0.x_75_3 ;
    LOCSTORE(store, 76,  1, STOREDIM, STOREDIM) += f_6_1_0.x_76_1 ;
    LOCSTORE(store, 76,  2, STOREDIM, STOREDIM) += f_6_1_0.x_76_2 ;
    LOCSTORE(store, 76,  3, STOREDIM, STOREDIM) += f_6_1_0.x_76_3 ;
    LOCSTORE(store, 77,  1, STOREDIM, STOREDIM) += f_6_1_0.x_77_1 ;
    LOCSTORE(store, 77,  2, STOREDIM, STOREDIM) += f_6_1_0.x_77_2 ;
    LOCSTORE(store, 77,  3, STOREDIM, STOREDIM) += f_6_1_0.x_77_3 ;
    LOCSTORE(store, 78,  1, STOREDIM, STOREDIM) += f_6_1_0.x_78_1 ;
    LOCSTORE(store, 78,  2, STOREDIM, STOREDIM) += f_6_1_0.x_78_2 ;
    LOCSTORE(store, 78,  3, STOREDIM, STOREDIM) += f_6_1_0.x_78_3 ;
    LOCSTORE(store, 79,  1, STOREDIM, STOREDIM) += f_6_1_0.x_79_1 ;
    LOCSTORE(store, 79,  2, STOREDIM, STOREDIM) += f_6_1_0.x_79_2 ;
    LOCSTORE(store, 79,  3, STOREDIM, STOREDIM) += f_6_1_0.x_79_3 ;
    LOCSTORE(store, 80,  1, STOREDIM, STOREDIM) += f_6_1_0.x_80_1 ;
    LOCSTORE(store, 80,  2, STOREDIM, STOREDIM) += f_6_1_0.x_80_2 ;
    LOCSTORE(store, 80,  3, STOREDIM, STOREDIM) += f_6_1_0.x_80_3 ;
    LOCSTORE(store, 81,  1, STOREDIM, STOREDIM) += f_6_1_0.x_81_1 ;
    LOCSTORE(store, 81,  2, STOREDIM, STOREDIM) += f_6_1_0.x_81_2 ;
    LOCSTORE(store, 81,  3, STOREDIM, STOREDIM) += f_6_1_0.x_81_3 ;
    LOCSTORE(store, 82,  1, STOREDIM, STOREDIM) += f_6_1_0.x_82_1 ;
    LOCSTORE(store, 82,  2, STOREDIM, STOREDIM) += f_6_1_0.x_82_2 ;
    LOCSTORE(store, 82,  3, STOREDIM, STOREDIM) += f_6_1_0.x_82_3 ;
    LOCSTORE(store, 83,  1, STOREDIM, STOREDIM) += f_6_1_0.x_83_1 ;
    LOCSTORE(store, 83,  2, STOREDIM, STOREDIM) += f_6_1_0.x_83_2 ;
    LOCSTORE(store, 83,  3, STOREDIM, STOREDIM) += f_6_1_0.x_83_3 ;
}
