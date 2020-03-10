/*
 Copyright (C) 2012 M.A.L. Marques, M. Oliveira

 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/


#include "xc.h"
#include "config.h"
#include <stdio.h>

static const char * libxc_version = PACKAGE_VERSION;

void xc_version(int *major, int *minor, int *micro) {

  *major = -1;
  *minor = -1;
  *micro = -1;
  sscanf(libxc_version,"%d.%d.%d",major,minor,micro);

}

const char *xc_version_string() {
  return libxc_version;
}
