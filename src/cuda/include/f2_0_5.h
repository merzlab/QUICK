__device__ __inline__ void h2_0_5(QUICKDouble* YVerticalTemp, QUICKDouble* store,
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

    // WRITE LAST FOR I =            0  J=           5
    LOC2(store,  0, 35, STOREDIM, STOREDIM) = f_0_5_0.x_0_35 ;
    LOC2(store,  0, 36, STOREDIM, STOREDIM) = f_0_5_0.x_0_36 ;
    LOC2(store,  0, 37, STOREDIM, STOREDIM) = f_0_5_0.x_0_37 ;
    LOC2(store,  0, 38, STOREDIM, STOREDIM) = f_0_5_0.x_0_38 ;
    LOC2(store,  0, 39, STOREDIM, STOREDIM) = f_0_5_0.x_0_39 ;
    LOC2(store,  0, 40, STOREDIM, STOREDIM) = f_0_5_0.x_0_40 ;
    LOC2(store,  0, 41, STOREDIM, STOREDIM) = f_0_5_0.x_0_41 ;
    LOC2(store,  0, 42, STOREDIM, STOREDIM) = f_0_5_0.x_0_42 ;
    LOC2(store,  0, 43, STOREDIM, STOREDIM) = f_0_5_0.x_0_43 ;
    LOC2(store,  0, 44, STOREDIM, STOREDIM) = f_0_5_0.x_0_44 ;
    LOC2(store,  0, 45, STOREDIM, STOREDIM) = f_0_5_0.x_0_45 ;
    LOC2(store,  0, 46, STOREDIM, STOREDIM) = f_0_5_0.x_0_46 ;
    LOC2(store,  0, 47, STOREDIM, STOREDIM) = f_0_5_0.x_0_47 ;
    LOC2(store,  0, 48, STOREDIM, STOREDIM) = f_0_5_0.x_0_48 ;
    LOC2(store,  0, 49, STOREDIM, STOREDIM) = f_0_5_0.x_0_49 ;
    LOC2(store,  0, 50, STOREDIM, STOREDIM) = f_0_5_0.x_0_50 ;
    LOC2(store,  0, 51, STOREDIM, STOREDIM) = f_0_5_0.x_0_51 ;
    LOC2(store,  0, 52, STOREDIM, STOREDIM) = f_0_5_0.x_0_52 ;
    LOC2(store,  0, 53, STOREDIM, STOREDIM) = f_0_5_0.x_0_53 ;
    LOC2(store,  0, 54, STOREDIM, STOREDIM) = f_0_5_0.x_0_54 ;
    LOC2(store,  0, 55, STOREDIM, STOREDIM) = f_0_5_0.x_0_55 ;
}
