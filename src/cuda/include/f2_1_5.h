__device__ __inline__  void h2_1_5(QUICKDouble* YVerticalTemp, QUICKDouble* store,
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

    // call for L =            0  B =            3
    f_0_3_t f_0_3_0 ( f_0_2_0, f_0_2_1, f_0_1_0, f_0_1_1, CDtemp, ABcom, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            0  B =            1
    f_0_1_t f_0_1_3 ( VY( 0, 0, 3 ), VY( 0, 0, 4 ), Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            0  B =            2
    f_0_2_t f_0_2_2 ( f_0_1_2, f_0_1_3, VY( 0, 0, 2 ), VY( 0, 0, 3 ), CDtemp, ABcom, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            0  B =            3
    f_0_3_t f_0_3_1 ( f_0_2_1, f_0_2_2, f_0_1_1, f_0_1_2, CDtemp, ABcom, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            0  B =            4
    f_0_4_t f_0_4_0 ( f_0_3_0, f_0_3_1, f_0_2_0, f_0_2_1, CDtemp, ABcom, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            0  B =            1
    f_0_1_t f_0_1_4 ( VY( 0, 0, 4 ), VY( 0, 0, 5 ), Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            0  B =            2
    f_0_2_t f_0_2_3 ( f_0_1_3, f_0_1_4, VY( 0, 0, 3 ), VY( 0, 0, 4 ), CDtemp, ABcom, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            0  B =            3
    f_0_3_t f_0_3_2 ( f_0_2_2, f_0_2_3, f_0_1_2, f_0_1_3, CDtemp, ABcom, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            0  B =            4
    f_0_4_t f_0_4_1 ( f_0_3_1, f_0_3_2, f_0_2_1, f_0_2_2, CDtemp, ABcom, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            0  B =            5
    f_0_5_t f_0_5_0 ( f_0_4_0, f_0_4_1, f_0_3_0, f_0_3_1, CDtemp, ABcom, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            0  B =            1
    f_0_1_t f_0_1_5 ( VY( 0, 0, 5 ), VY( 0, 0, 6 ), Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            0  B =            2
    f_0_2_t f_0_2_4 ( f_0_1_4, f_0_1_5, VY( 0, 0, 4 ), VY( 0, 0, 5 ), CDtemp, ABcom, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            0  B =            3
    f_0_3_t f_0_3_3 ( f_0_2_3, f_0_2_4, f_0_1_3, f_0_1_4, CDtemp, ABcom, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            0  B =            4
    f_0_4_t f_0_4_2 ( f_0_3_2, f_0_3_3, f_0_2_2, f_0_2_3, CDtemp, ABcom, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            0  B =            5
    f_0_5_t f_0_5_1 ( f_0_4_1, f_0_4_2, f_0_3_1, f_0_3_2, CDtemp, ABcom, Qtempx, Qtempy, Qtempz, WQtempx, WQtempy, WQtempz);

    // call for L =            1  B =            5
    f_1_5_t f_1_5_0 ( f_0_5_0,  f_0_5_1,  f_0_4_1, ABCDtemp, Ptempx, Ptempy, Ptempz, WPtempx, WPtempy, WPtempz);

    // WRITE LAST FOR I =            1  J=           5
    LOC2(store,  1, 35, STOREDIM, STOREDIM) = f_1_5_0.x_1_35 ;
    LOC2(store,  1, 36, STOREDIM, STOREDIM) = f_1_5_0.x_1_36 ;
    LOC2(store,  1, 37, STOREDIM, STOREDIM) = f_1_5_0.x_1_37 ;
    LOC2(store,  1, 38, STOREDIM, STOREDIM) = f_1_5_0.x_1_38 ;
    LOC2(store,  1, 39, STOREDIM, STOREDIM) = f_1_5_0.x_1_39 ;
    LOC2(store,  1, 40, STOREDIM, STOREDIM) = f_1_5_0.x_1_40 ;
    LOC2(store,  1, 41, STOREDIM, STOREDIM) = f_1_5_0.x_1_41 ;
    LOC2(store,  1, 42, STOREDIM, STOREDIM) = f_1_5_0.x_1_42 ;
    LOC2(store,  1, 43, STOREDIM, STOREDIM) = f_1_5_0.x_1_43 ;
    LOC2(store,  1, 44, STOREDIM, STOREDIM) = f_1_5_0.x_1_44 ;
    LOC2(store,  1, 45, STOREDIM, STOREDIM) = f_1_5_0.x_1_45 ;
    LOC2(store,  1, 46, STOREDIM, STOREDIM) = f_1_5_0.x_1_46 ;
    LOC2(store,  1, 47, STOREDIM, STOREDIM) = f_1_5_0.x_1_47 ;
    LOC2(store,  1, 48, STOREDIM, STOREDIM) = f_1_5_0.x_1_48 ;
    LOC2(store,  1, 49, STOREDIM, STOREDIM) = f_1_5_0.x_1_49 ;
    LOC2(store,  1, 50, STOREDIM, STOREDIM) = f_1_5_0.x_1_50 ;
    LOC2(store,  1, 51, STOREDIM, STOREDIM) = f_1_5_0.x_1_51 ;
    LOC2(store,  1, 52, STOREDIM, STOREDIM) = f_1_5_0.x_1_52 ;
    LOC2(store,  1, 53, STOREDIM, STOREDIM) = f_1_5_0.x_1_53 ;
    LOC2(store,  1, 54, STOREDIM, STOREDIM) = f_1_5_0.x_1_54 ;
    LOC2(store,  1, 55, STOREDIM, STOREDIM) = f_1_5_0.x_1_55 ;
    LOC2(store,  2, 35, STOREDIM, STOREDIM) = f_1_5_0.x_2_35 ;
    LOC2(store,  2, 36, STOREDIM, STOREDIM) = f_1_5_0.x_2_36 ;
    LOC2(store,  2, 37, STOREDIM, STOREDIM) = f_1_5_0.x_2_37 ;
    LOC2(store,  2, 38, STOREDIM, STOREDIM) = f_1_5_0.x_2_38 ;
    LOC2(store,  2, 39, STOREDIM, STOREDIM) = f_1_5_0.x_2_39 ;
    LOC2(store,  2, 40, STOREDIM, STOREDIM) = f_1_5_0.x_2_40 ;
    LOC2(store,  2, 41, STOREDIM, STOREDIM) = f_1_5_0.x_2_41 ;
    LOC2(store,  2, 42, STOREDIM, STOREDIM) = f_1_5_0.x_2_42 ;
    LOC2(store,  2, 43, STOREDIM, STOREDIM) = f_1_5_0.x_2_43 ;
    LOC2(store,  2, 44, STOREDIM, STOREDIM) = f_1_5_0.x_2_44 ;
    LOC2(store,  2, 45, STOREDIM, STOREDIM) = f_1_5_0.x_2_45 ;
    LOC2(store,  2, 46, STOREDIM, STOREDIM) = f_1_5_0.x_2_46 ;
    LOC2(store,  2, 47, STOREDIM, STOREDIM) = f_1_5_0.x_2_47 ;
    LOC2(store,  2, 48, STOREDIM, STOREDIM) = f_1_5_0.x_2_48 ;
    LOC2(store,  2, 49, STOREDIM, STOREDIM) = f_1_5_0.x_2_49 ;
    LOC2(store,  2, 50, STOREDIM, STOREDIM) = f_1_5_0.x_2_50 ;
    LOC2(store,  2, 51, STOREDIM, STOREDIM) = f_1_5_0.x_2_51 ;
    LOC2(store,  2, 52, STOREDIM, STOREDIM) = f_1_5_0.x_2_52 ;
    LOC2(store,  2, 53, STOREDIM, STOREDIM) = f_1_5_0.x_2_53 ;
    LOC2(store,  2, 54, STOREDIM, STOREDIM) = f_1_5_0.x_2_54 ;
    LOC2(store,  2, 55, STOREDIM, STOREDIM) = f_1_5_0.x_2_55 ;
    LOC2(store,  3, 35, STOREDIM, STOREDIM) = f_1_5_0.x_3_35 ;
    LOC2(store,  3, 36, STOREDIM, STOREDIM) = f_1_5_0.x_3_36 ;
    LOC2(store,  3, 37, STOREDIM, STOREDIM) = f_1_5_0.x_3_37 ;
    LOC2(store,  3, 38, STOREDIM, STOREDIM) = f_1_5_0.x_3_38 ;
    LOC2(store,  3, 39, STOREDIM, STOREDIM) = f_1_5_0.x_3_39 ;
    LOC2(store,  3, 40, STOREDIM, STOREDIM) = f_1_5_0.x_3_40 ;
    LOC2(store,  3, 41, STOREDIM, STOREDIM) = f_1_5_0.x_3_41 ;
    LOC2(store,  3, 42, STOREDIM, STOREDIM) = f_1_5_0.x_3_42 ;
    LOC2(store,  3, 43, STOREDIM, STOREDIM) = f_1_5_0.x_3_43 ;
    LOC2(store,  3, 44, STOREDIM, STOREDIM) = f_1_5_0.x_3_44 ;
    LOC2(store,  3, 45, STOREDIM, STOREDIM) = f_1_5_0.x_3_45 ;
    LOC2(store,  3, 46, STOREDIM, STOREDIM) = f_1_5_0.x_3_46 ;
    LOC2(store,  3, 47, STOREDIM, STOREDIM) = f_1_5_0.x_3_47 ;
    LOC2(store,  3, 48, STOREDIM, STOREDIM) = f_1_5_0.x_3_48 ;
    LOC2(store,  3, 49, STOREDIM, STOREDIM) = f_1_5_0.x_3_49 ;
    LOC2(store,  3, 50, STOREDIM, STOREDIM) = f_1_5_0.x_3_50 ;
    LOC2(store,  3, 51, STOREDIM, STOREDIM) = f_1_5_0.x_3_51 ;
    LOC2(store,  3, 52, STOREDIM, STOREDIM) = f_1_5_0.x_3_52 ;
    LOC2(store,  3, 53, STOREDIM, STOREDIM) = f_1_5_0.x_3_53 ;
    LOC2(store,  3, 54, STOREDIM, STOREDIM) = f_1_5_0.x_3_54 ;
    LOC2(store,  3, 55, STOREDIM, STOREDIM) = f_1_5_0.x_3_55 ;
}
