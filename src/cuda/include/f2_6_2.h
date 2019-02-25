__device__ __inline__   void h2_6_2(QUICKDouble* YVerticalTemp, QUICKDouble* store,
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

    // call for L =            1  B =            0
    f_1_0_t f_1_0_7 ( VY( 0, 0, 7 ),  VY( 0, 0, 8 ), Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            2  B =            0
    f_2_0_t f_2_0_6 ( f_1_0_6,  f_1_0_7, VY( 0, 0, 6 ), VY( 0, 0, 7 ), ABtemp, CDcom, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            3  B =            0
    f_3_0_t f_3_0_5 ( f_2_0_5,  f_2_0_6, f_1_0_5, f_1_0_6, ABtemp, CDcom, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for L =            4  B =            0
    f_4_0_t f_4_0_4 ( f_3_0_4,  f_3_0_5, f_2_0_4, f_2_0_5, ABtemp, CDcom, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for B =            5  L =            0
    f_5_0_t f_5_0_3 ( f_4_0_3, f_4_0_4, f_3_0_3, f_3_0_4, ABtemp, CDcom, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for B =            6  L =            0
    f_6_0_t f_6_0_2 ( f_5_0_2, f_5_0_3, f_4_0_2, f_4_0_3, ABtemp, CDcom, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // call for B =            6  L =            1
    f_6_1_t f_6_1_1 ( f_6_0_1,  f_6_0_2,  f_5_0_2, ABCDtemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for B =            5  L =            1
    f_5_1_t f_5_1_1 ( f_5_0_1,  f_5_0_2,  f_4_0_2, ABCDtemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for B =            6  L =            2
    f_6_2_t f_6_2_0 ( f_6_1_0,  f_6_1_1, f_6_0_0, f_6_0_1, CDtemp, ABcom, f_5_1_1, ABCDtemp, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // WRITE LAST FOR I =            6  J=           2
    LOC2(store, 56,  4, STOREDIM, STOREDIM) = f_6_2_0.x_56_4 ;
    LOC2(store, 56,  5, STOREDIM, STOREDIM) = f_6_2_0.x_56_5 ;
    LOC2(store, 56,  6, STOREDIM, STOREDIM) = f_6_2_0.x_56_6 ;
    LOC2(store, 56,  7, STOREDIM, STOREDIM) = f_6_2_0.x_56_7 ;
    LOC2(store, 56,  8, STOREDIM, STOREDIM) = f_6_2_0.x_56_8 ;
    LOC2(store, 56,  9, STOREDIM, STOREDIM) = f_6_2_0.x_56_9 ;
    LOC2(store, 57,  4, STOREDIM, STOREDIM) = f_6_2_0.x_57_4 ;
    LOC2(store, 57,  5, STOREDIM, STOREDIM) = f_6_2_0.x_57_5 ;
    LOC2(store, 57,  6, STOREDIM, STOREDIM) = f_6_2_0.x_57_6 ;
    LOC2(store, 57,  7, STOREDIM, STOREDIM) = f_6_2_0.x_57_7 ;
    LOC2(store, 57,  8, STOREDIM, STOREDIM) = f_6_2_0.x_57_8 ;
    LOC2(store, 57,  9, STOREDIM, STOREDIM) = f_6_2_0.x_57_9 ;
    LOC2(store, 58,  4, STOREDIM, STOREDIM) = f_6_2_0.x_58_4 ;
    LOC2(store, 58,  5, STOREDIM, STOREDIM) = f_6_2_0.x_58_5 ;
    LOC2(store, 58,  6, STOREDIM, STOREDIM) = f_6_2_0.x_58_6 ;
    LOC2(store, 58,  7, STOREDIM, STOREDIM) = f_6_2_0.x_58_7 ;
    LOC2(store, 58,  8, STOREDIM, STOREDIM) = f_6_2_0.x_58_8 ;
    LOC2(store, 58,  9, STOREDIM, STOREDIM) = f_6_2_0.x_58_9 ;
    LOC2(store, 59,  4, STOREDIM, STOREDIM) = f_6_2_0.x_59_4 ;
    LOC2(store, 59,  5, STOREDIM, STOREDIM) = f_6_2_0.x_59_5 ;
    LOC2(store, 59,  6, STOREDIM, STOREDIM) = f_6_2_0.x_59_6 ;
    LOC2(store, 59,  7, STOREDIM, STOREDIM) = f_6_2_0.x_59_7 ;
    LOC2(store, 59,  8, STOREDIM, STOREDIM) = f_6_2_0.x_59_8 ;
    LOC2(store, 59,  9, STOREDIM, STOREDIM) = f_6_2_0.x_59_9 ;
    LOC2(store, 60,  4, STOREDIM, STOREDIM) = f_6_2_0.x_60_4 ;
    LOC2(store, 60,  5, STOREDIM, STOREDIM) = f_6_2_0.x_60_5 ;
    LOC2(store, 60,  6, STOREDIM, STOREDIM) = f_6_2_0.x_60_6 ;
    LOC2(store, 60,  7, STOREDIM, STOREDIM) = f_6_2_0.x_60_7 ;
    LOC2(store, 60,  8, STOREDIM, STOREDIM) = f_6_2_0.x_60_8 ;
    LOC2(store, 60,  9, STOREDIM, STOREDIM) = f_6_2_0.x_60_9 ;
    LOC2(store, 61,  4, STOREDIM, STOREDIM) = f_6_2_0.x_61_4 ;
    LOC2(store, 61,  5, STOREDIM, STOREDIM) = f_6_2_0.x_61_5 ;
    LOC2(store, 61,  6, STOREDIM, STOREDIM) = f_6_2_0.x_61_6 ;
    LOC2(store, 61,  7, STOREDIM, STOREDIM) = f_6_2_0.x_61_7 ;
    LOC2(store, 61,  8, STOREDIM, STOREDIM) = f_6_2_0.x_61_8 ;
    LOC2(store, 61,  9, STOREDIM, STOREDIM) = f_6_2_0.x_61_9 ;
    LOC2(store, 62,  4, STOREDIM, STOREDIM) = f_6_2_0.x_62_4 ;
    LOC2(store, 62,  5, STOREDIM, STOREDIM) = f_6_2_0.x_62_5 ;
    LOC2(store, 62,  6, STOREDIM, STOREDIM) = f_6_2_0.x_62_6 ;
    LOC2(store, 62,  7, STOREDIM, STOREDIM) = f_6_2_0.x_62_7 ;
    LOC2(store, 62,  8, STOREDIM, STOREDIM) = f_6_2_0.x_62_8 ;
    LOC2(store, 62,  9, STOREDIM, STOREDIM) = f_6_2_0.x_62_9 ;
    LOC2(store, 63,  4, STOREDIM, STOREDIM) = f_6_2_0.x_63_4 ;
    LOC2(store, 63,  5, STOREDIM, STOREDIM) = f_6_2_0.x_63_5 ;
    LOC2(store, 63,  6, STOREDIM, STOREDIM) = f_6_2_0.x_63_6 ;
    LOC2(store, 63,  7, STOREDIM, STOREDIM) = f_6_2_0.x_63_7 ;
    LOC2(store, 63,  8, STOREDIM, STOREDIM) = f_6_2_0.x_63_8 ;
    LOC2(store, 63,  9, STOREDIM, STOREDIM) = f_6_2_0.x_63_9 ;
    LOC2(store, 64,  4, STOREDIM, STOREDIM) = f_6_2_0.x_64_4 ;
    LOC2(store, 64,  5, STOREDIM, STOREDIM) = f_6_2_0.x_64_5 ;
    LOC2(store, 64,  6, STOREDIM, STOREDIM) = f_6_2_0.x_64_6 ;
    LOC2(store, 64,  7, STOREDIM, STOREDIM) = f_6_2_0.x_64_7 ;
    LOC2(store, 64,  8, STOREDIM, STOREDIM) = f_6_2_0.x_64_8 ;
    LOC2(store, 64,  9, STOREDIM, STOREDIM) = f_6_2_0.x_64_9 ;
    LOC2(store, 65,  4, STOREDIM, STOREDIM) = f_6_2_0.x_65_4 ;
    LOC2(store, 65,  5, STOREDIM, STOREDIM) = f_6_2_0.x_65_5 ;
    LOC2(store, 65,  6, STOREDIM, STOREDIM) = f_6_2_0.x_65_6 ;
    LOC2(store, 65,  7, STOREDIM, STOREDIM) = f_6_2_0.x_65_7 ;
    LOC2(store, 65,  8, STOREDIM, STOREDIM) = f_6_2_0.x_65_8 ;
    LOC2(store, 65,  9, STOREDIM, STOREDIM) = f_6_2_0.x_65_9 ;
    LOC2(store, 66,  4, STOREDIM, STOREDIM) = f_6_2_0.x_66_4 ;
    LOC2(store, 66,  5, STOREDIM, STOREDIM) = f_6_2_0.x_66_5 ;
    LOC2(store, 66,  6, STOREDIM, STOREDIM) = f_6_2_0.x_66_6 ;
    LOC2(store, 66,  7, STOREDIM, STOREDIM) = f_6_2_0.x_66_7 ;
    LOC2(store, 66,  8, STOREDIM, STOREDIM) = f_6_2_0.x_66_8 ;
    LOC2(store, 66,  9, STOREDIM, STOREDIM) = f_6_2_0.x_66_9 ;
    LOC2(store, 67,  4, STOREDIM, STOREDIM) = f_6_2_0.x_67_4 ;
    LOC2(store, 67,  5, STOREDIM, STOREDIM) = f_6_2_0.x_67_5 ;
    LOC2(store, 67,  6, STOREDIM, STOREDIM) = f_6_2_0.x_67_6 ;
    LOC2(store, 67,  7, STOREDIM, STOREDIM) = f_6_2_0.x_67_7 ;
    LOC2(store, 67,  8, STOREDIM, STOREDIM) = f_6_2_0.x_67_8 ;
    LOC2(store, 67,  9, STOREDIM, STOREDIM) = f_6_2_0.x_67_9 ;
    LOC2(store, 68,  4, STOREDIM, STOREDIM) = f_6_2_0.x_68_4 ;
    LOC2(store, 68,  5, STOREDIM, STOREDIM) = f_6_2_0.x_68_5 ;
    LOC2(store, 68,  6, STOREDIM, STOREDIM) = f_6_2_0.x_68_6 ;
    LOC2(store, 68,  7, STOREDIM, STOREDIM) = f_6_2_0.x_68_7 ;
    LOC2(store, 68,  8, STOREDIM, STOREDIM) = f_6_2_0.x_68_8 ;
    LOC2(store, 68,  9, STOREDIM, STOREDIM) = f_6_2_0.x_68_9 ;
    LOC2(store, 69,  4, STOREDIM, STOREDIM) = f_6_2_0.x_69_4 ;
    LOC2(store, 69,  5, STOREDIM, STOREDIM) = f_6_2_0.x_69_5 ;
    LOC2(store, 69,  6, STOREDIM, STOREDIM) = f_6_2_0.x_69_6 ;
    LOC2(store, 69,  7, STOREDIM, STOREDIM) = f_6_2_0.x_69_7 ;
    LOC2(store, 69,  8, STOREDIM, STOREDIM) = f_6_2_0.x_69_8 ;
    LOC2(store, 69,  9, STOREDIM, STOREDIM) = f_6_2_0.x_69_9 ;
    LOC2(store, 70,  4, STOREDIM, STOREDIM) = f_6_2_0.x_70_4 ;
    LOC2(store, 70,  5, STOREDIM, STOREDIM) = f_6_2_0.x_70_5 ;
    LOC2(store, 70,  6, STOREDIM, STOREDIM) = f_6_2_0.x_70_6 ;
    LOC2(store, 70,  7, STOREDIM, STOREDIM) = f_6_2_0.x_70_7 ;
    LOC2(store, 70,  8, STOREDIM, STOREDIM) = f_6_2_0.x_70_8 ;
    LOC2(store, 70,  9, STOREDIM, STOREDIM) = f_6_2_0.x_70_9 ;
    LOC2(store, 71,  4, STOREDIM, STOREDIM) = f_6_2_0.x_71_4 ;
    LOC2(store, 71,  5, STOREDIM, STOREDIM) = f_6_2_0.x_71_5 ;
    LOC2(store, 71,  6, STOREDIM, STOREDIM) = f_6_2_0.x_71_6 ;
    LOC2(store, 71,  7, STOREDIM, STOREDIM) = f_6_2_0.x_71_7 ;
    LOC2(store, 71,  8, STOREDIM, STOREDIM) = f_6_2_0.x_71_8 ;
    LOC2(store, 71,  9, STOREDIM, STOREDIM) = f_6_2_0.x_71_9 ;
    LOC2(store, 72,  4, STOREDIM, STOREDIM) = f_6_2_0.x_72_4 ;
    LOC2(store, 72,  5, STOREDIM, STOREDIM) = f_6_2_0.x_72_5 ;
    LOC2(store, 72,  6, STOREDIM, STOREDIM) = f_6_2_0.x_72_6 ;
    LOC2(store, 72,  7, STOREDIM, STOREDIM) = f_6_2_0.x_72_7 ;
    LOC2(store, 72,  8, STOREDIM, STOREDIM) = f_6_2_0.x_72_8 ;
    LOC2(store, 72,  9, STOREDIM, STOREDIM) = f_6_2_0.x_72_9 ;
    LOC2(store, 73,  4, STOREDIM, STOREDIM) = f_6_2_0.x_73_4 ;
    LOC2(store, 73,  5, STOREDIM, STOREDIM) = f_6_2_0.x_73_5 ;
    LOC2(store, 73,  6, STOREDIM, STOREDIM) = f_6_2_0.x_73_6 ;
    LOC2(store, 73,  7, STOREDIM, STOREDIM) = f_6_2_0.x_73_7 ;
    LOC2(store, 73,  8, STOREDIM, STOREDIM) = f_6_2_0.x_73_8 ;
    LOC2(store, 73,  9, STOREDIM, STOREDIM) = f_6_2_0.x_73_9 ;
    LOC2(store, 74,  4, STOREDIM, STOREDIM) = f_6_2_0.x_74_4 ;
    LOC2(store, 74,  5, STOREDIM, STOREDIM) = f_6_2_0.x_74_5 ;
    LOC2(store, 74,  6, STOREDIM, STOREDIM) = f_6_2_0.x_74_6 ;
    LOC2(store, 74,  7, STOREDIM, STOREDIM) = f_6_2_0.x_74_7 ;
    LOC2(store, 74,  8, STOREDIM, STOREDIM) = f_6_2_0.x_74_8 ;
    LOC2(store, 74,  9, STOREDIM, STOREDIM) = f_6_2_0.x_74_9 ;
    LOC2(store, 75,  4, STOREDIM, STOREDIM) = f_6_2_0.x_75_4 ;
    LOC2(store, 75,  5, STOREDIM, STOREDIM) = f_6_2_0.x_75_5 ;
    LOC2(store, 75,  6, STOREDIM, STOREDIM) = f_6_2_0.x_75_6 ;
    LOC2(store, 75,  7, STOREDIM, STOREDIM) = f_6_2_0.x_75_7 ;
    LOC2(store, 75,  8, STOREDIM, STOREDIM) = f_6_2_0.x_75_8 ;
    LOC2(store, 75,  9, STOREDIM, STOREDIM) = f_6_2_0.x_75_9 ;
    LOC2(store, 76,  4, STOREDIM, STOREDIM) = f_6_2_0.x_76_4 ;
    LOC2(store, 76,  5, STOREDIM, STOREDIM) = f_6_2_0.x_76_5 ;
    LOC2(store, 76,  6, STOREDIM, STOREDIM) = f_6_2_0.x_76_6 ;
    LOC2(store, 76,  7, STOREDIM, STOREDIM) = f_6_2_0.x_76_7 ;
    LOC2(store, 76,  8, STOREDIM, STOREDIM) = f_6_2_0.x_76_8 ;
    LOC2(store, 76,  9, STOREDIM, STOREDIM) = f_6_2_0.x_76_9 ;
    LOC2(store, 77,  4, STOREDIM, STOREDIM) = f_6_2_0.x_77_4 ;
    LOC2(store, 77,  5, STOREDIM, STOREDIM) = f_6_2_0.x_77_5 ;
    LOC2(store, 77,  6, STOREDIM, STOREDIM) = f_6_2_0.x_77_6 ;
    LOC2(store, 77,  7, STOREDIM, STOREDIM) = f_6_2_0.x_77_7 ;
    LOC2(store, 77,  8, STOREDIM, STOREDIM) = f_6_2_0.x_77_8 ;
    LOC2(store, 77,  9, STOREDIM, STOREDIM) = f_6_2_0.x_77_9 ;
    LOC2(store, 78,  4, STOREDIM, STOREDIM) = f_6_2_0.x_78_4 ;
    LOC2(store, 78,  5, STOREDIM, STOREDIM) = f_6_2_0.x_78_5 ;
    LOC2(store, 78,  6, STOREDIM, STOREDIM) = f_6_2_0.x_78_6 ;
    LOC2(store, 78,  7, STOREDIM, STOREDIM) = f_6_2_0.x_78_7 ;
    LOC2(store, 78,  8, STOREDIM, STOREDIM) = f_6_2_0.x_78_8 ;
    LOC2(store, 78,  9, STOREDIM, STOREDIM) = f_6_2_0.x_78_9 ;
    LOC2(store, 79,  4, STOREDIM, STOREDIM) = f_6_2_0.x_79_4 ;
    LOC2(store, 79,  5, STOREDIM, STOREDIM) = f_6_2_0.x_79_5 ;
    LOC2(store, 79,  6, STOREDIM, STOREDIM) = f_6_2_0.x_79_6 ;
    LOC2(store, 79,  7, STOREDIM, STOREDIM) = f_6_2_0.x_79_7 ;
    LOC2(store, 79,  8, STOREDIM, STOREDIM) = f_6_2_0.x_79_8 ;
    LOC2(store, 79,  9, STOREDIM, STOREDIM) = f_6_2_0.x_79_9 ;
    LOC2(store, 80,  4, STOREDIM, STOREDIM) = f_6_2_0.x_80_4 ;
    LOC2(store, 80,  5, STOREDIM, STOREDIM) = f_6_2_0.x_80_5 ;
    LOC2(store, 80,  6, STOREDIM, STOREDIM) = f_6_2_0.x_80_6 ;
    LOC2(store, 80,  7, STOREDIM, STOREDIM) = f_6_2_0.x_80_7 ;
    LOC2(store, 80,  8, STOREDIM, STOREDIM) = f_6_2_0.x_80_8 ;
    LOC2(store, 80,  9, STOREDIM, STOREDIM) = f_6_2_0.x_80_9 ;
    LOC2(store, 81,  4, STOREDIM, STOREDIM) = f_6_2_0.x_81_4 ;
    LOC2(store, 81,  5, STOREDIM, STOREDIM) = f_6_2_0.x_81_5 ;
    LOC2(store, 81,  6, STOREDIM, STOREDIM) = f_6_2_0.x_81_6 ;
    LOC2(store, 81,  7, STOREDIM, STOREDIM) = f_6_2_0.x_81_7 ;
    LOC2(store, 81,  8, STOREDIM, STOREDIM) = f_6_2_0.x_81_8 ;
    LOC2(store, 81,  9, STOREDIM, STOREDIM) = f_6_2_0.x_81_9 ;
    LOC2(store, 82,  4, STOREDIM, STOREDIM) = f_6_2_0.x_82_4 ;
    LOC2(store, 82,  5, STOREDIM, STOREDIM) = f_6_2_0.x_82_5 ;
    LOC2(store, 82,  6, STOREDIM, STOREDIM) = f_6_2_0.x_82_6 ;
    LOC2(store, 82,  7, STOREDIM, STOREDIM) = f_6_2_0.x_82_7 ;
    LOC2(store, 82,  8, STOREDIM, STOREDIM) = f_6_2_0.x_82_8 ;
    LOC2(store, 82,  9, STOREDIM, STOREDIM) = f_6_2_0.x_82_9 ;
    LOC2(store, 83,  4, STOREDIM, STOREDIM) = f_6_2_0.x_83_4 ;
    LOC2(store, 83,  5, STOREDIM, STOREDIM) = f_6_2_0.x_83_5 ;
    LOC2(store, 83,  6, STOREDIM, STOREDIM) = f_6_2_0.x_83_6 ;
    LOC2(store, 83,  7, STOREDIM, STOREDIM) = f_6_2_0.x_83_7 ;
    LOC2(store, 83,  8, STOREDIM, STOREDIM) = f_6_2_0.x_83_8 ;
    LOC2(store, 83,  9, STOREDIM, STOREDIM) = f_6_2_0.x_83_9 ;
}
