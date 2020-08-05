/*
 *  gpu_type.cpp
 *  new_quick
 *
 *  Created by Yipu Miao on 6/1/11.
 *  Copyright 2011 University of Florida. All rights reserved.
 *
 */

#include "gpu_type.h"

void gpu_basis_type :: upload_all()
{

    this->ncontract->Upload();
    this->itype->Upload();
    this->aexp->Upload();
    this->dcoeff->Upload();
    this->ncenter->Upload();
/*    this->first_basis_function->Upload();
    this->last_basis_function->Upload();
    this->first_shell_basis_function->Upload();
    this->last_basis_function->Upload();
    this->kshell->Upload();
    this->ktype->Upload();
  */
    this->kstart->Upload();
    this->katom->Upload();
    this->kprim->Upload();
    this->Ksumtype->Upload();
    this->Qnumber->Upload();
    this->Qstart->Upload();
    this->Qfinal->Upload();
    this->Qsbasis->Upload();
    this->Qfbasis->Upload();
    this->gccoeff->Upload();
    this->cons->Upload();
    this->Xcoeff->Upload();
    this->gcexpo->Upload();
    this->KLMN->Upload();
    this->prim_start->Upload();
    this->Xcoeff->Upload();
    this->expoSum->Upload();
    this->weightedCenterX->Upload();
    this->weightedCenterY->Upload();
    this->weightedCenterZ->Upload();
    
    this->sorted_Q->Upload();
    this->sorted_Qnumber->Upload();
    
}
