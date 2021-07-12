
__device__ __forceinline__ void FmT(int MaxM, QUICKDouble X, QUICKDouble* YVerticalTemp)
{

    const QUICKDouble PIE4 = (QUICKDouble) PI/4.0 ;

    const QUICKDouble XINV = (QUICKDouble) 1.0 /X;
    const QUICKDouble E = (QUICKDouble) exp(-X);
    QUICKDouble WW1, F1;
    if (X > 5.0) {
        WW1 = sqrt(PIE4 * XINV);
    }else{
        WW1 = E;
    }


    if (X > 33.0) {

    }else if( X > 15.0){
        WW1 += (( 1.9623264149430E-01 *XINV-4.9695241464490E-01 )*XINV - \
                6.0156581186481E-05 )*E;
    }else if (X > 10.0 ){
        WW1 += (((-1.8784686463512E-01 *XINV+2.2991849164985E-01 )*XINV - \
                 4.9893752514047E-01 )*XINV-2.1916512131607E-05 )*E;
    }else if (X > 5.0) {
        WW1 += (((((( 4.6897511375022E-01  *XINV-6.9955602298985E-01 )*XINV + \
                    5.3689283271887E-01 )*XINV-3.2883030418398E-01 )*XINV + \
                  2.4645596956002E-01 )*XINV-4.9984072848436E-01 )*XINV - \
                3.1501078774085E-06 )*E;
    }else if (X > 3.0){
        QUICKDouble Y = (QUICKDouble) X - 4.0 ;
        F1 = ((((((((((-2.62453564772299E-11 *Y+3.24031041623823E-10  )*Y- \
                      3.614965656163E-09 )*Y+3.760256799971E-08 )*Y- \
                    3.553558319675E-07 )*Y+3.022556449731E-06 )*Y- \
                  2.290098979647E-05 )*Y+1.526537461148E-04 )*Y- \
                8.81947375894379E-04 )*Y+4.33207949514611E-03 )*Y- \
              1.75257821619926E-02 )*Y+5.28406320615584E-02 ;
        WW1 += (X+X)*F1;
    }else if (X > 1.0){
        QUICKDouble Y = (QUICKDouble) X - 2.0 ;
        F1 = ((((((((((-1.61702782425558E-10 *Y+1.96215250865776E-09  )*Y- \
                      2.14234468198419E-08  )*Y+2.17216556336318E-07  )*Y- \
                    1.98850171329371E-06  )*Y+1.62429321438911E-05  )*Y- \
                  1.16740298039895E-04  )*Y+7.24888732052332E-04  )*Y- \
                3.79490003707156E-03  )*Y+1.61723488664661E-02  )*Y- \
              5.29428148329736E-02  )*Y+1.15702180856167E-01 ;
        WW1 += (X+X)*F1;
    }else if (X > 1.0E-1 || (X> 1.0E-4 && MaxM < 4)){

        F1 =(((((((( -8.36313918003957E-08 *X+1.21222603512827E-06  )*X- \
                   1.15662609053481E-05  )*X+9.25197374512647E-05  )*X- \
                 6.40994113129432E-04  )*X+3.78787044215009E-03  )*X- \
               1.85185172458485E-02  )*X+7.14285713298222E-02  )*X- \
             1.99999999997023E-01  )*X+3.33333333333318E-01 ;
        WW1 += (X+X)*F1;
    }else{
        WW1 = (1.0 - X)/(QUICKDouble)(2.0 * MaxM+1);
    }


    if (X > 1.0E-1 || (X> 1.0E-4 && MaxM < 4)) {
        LOC3(YVerticalTemp, 0, 0, 0, VDIM1, VDIM2, VDIM3) = WW1;
        for (int m = 1; m<= MaxM; m++) {
            LOC3(YVerticalTemp, 0, 0, m, VDIM1, VDIM2, VDIM3) = (((2*m-1)*LOC3(YVerticalTemp, 0, 0, m-1, VDIM1, VDIM2, VDIM3))- E)*0.5*XINV;
        }
    }else {
        LOC3(YVerticalTemp, 0, 0, MaxM, VDIM1, VDIM2, VDIM3) = WW1;
        for (int m = MaxM-1; m >=0; m--) {
            LOC3(YVerticalTemp, 0, 0, m, VDIM1, VDIM2, VDIM3) = (2.0 * X * LOC3(YVerticalTemp, 0, 0, m+1, VDIM1, VDIM2, VDIM3) + E) / (QUICKDouble)(m*2+1);
        }
    }
    return;
}

