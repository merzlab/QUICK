/* config.h.  Generated from config.h.in by configure.  */
/* config.h.in.  Generated from configure.ac by autoheader.  */

/* The C type of a Fortran integer */
#define CC_FORTRAN_INT int

/* Define to one of `_getb67', `GETB67', `getb67' for Cray-2 and Cray-YMP
   systems. This function is required for `alloca.c' support on those systems.
   */
/* #undef CRAY_STACKSEG_END */

/* Define to 1 if using `alloca.c'. */
/* #undef C_ALLOCA */

/* compiler supports line-number lines */
#define F90_ACCEPTS_LINE_NUMBERS 1

/* Define to dummy `main' function (if any) required to link to the Fortran
   libraries. */
/* #undef FC_DUMMY_MAIN */

/* Define if F77 and FC dummy `main' functions are identical. */
/* #undef FC_DUMMY_MAIN_EQ_F77 */

#ifdef USE_CMAKE_MANGLING
#  include <xc-fortran-mangling.h>
#  define FC_FUNC(name, NAME) FortranCInterface_GLOBAL(name, NAME)
#  define FC_FUNC_(name, NAME) FortranCInterface_GLOBAL_(name, NAME)
#else

/* Define to a macro mangling the given C identifier (in lower and upper
   case), which must not contain underscores, for linking with Fortran. */
#define FC_FUNC(name,NAME) name ## _

/* As FC_FUNC, but for C identifiers containing underscores. */
#define FC_FUNC_(name,NAME) name ## _

#endif

/* The size of a Fortran integer */
#define FC_INTEGER_SIZE 4

/* Define to 1 if you have `alloca', as a function or macro. */
#define HAVE_ALLOCA 1

/* Define to 1 if you have <alloca.h> and it should be used (not on Ultrix).
   */
#define HAVE_ALLOCA_H 1

/* libm includes cbrt */
#define HAVE_CBRT 1

/* Define to 1 if you have the <dlfcn.h> header file. */
#define HAVE_DLFCN_H 1

/* libm includes feenableexcept */
#define HAVE_FEENABLEEXCEPT 1

/* Defined if libxc is compiled with fortran support */
#define HAVE_FORTRAN 1

/* Define to 1 if you have the <inttypes.h> header file. */
#define HAVE_INTTYPES_H 1

/* Define to 1 if you have the <memory.h> header file. */
#define HAVE_MEMORY_H 1

/* Define to 1 if you have the <stdint.h> header file. */
#define HAVE_STDINT_H 1

/* Define to 1 if you have the <stdlib.h> header file. */
#define HAVE_STDLIB_H 1

/* Define to 1 if you have the <strings.h> header file. */
#define HAVE_STRINGS_H 1

/* Define to 1 if you have the <string.h> header file. */
#define HAVE_STRING_H 1

/* Define to 1 if you have the <sys/stat.h> header file. */
#define HAVE_SYS_STAT_H 1

/* Define to 1 if you have the <sys/types.h> header file. */
#define HAVE_SYS_TYPES_H 1

/* Define to 1 if you have the <unistd.h> header file. */
#define HAVE_UNISTD_H 1

/* compiler supports Fortran 2003 iso_c_binding */
#define ISO_C_BINDING 1

/* compiler supports long lines */
#define LONG_LINES 1

/* Define to the sub-directory where libtool stores uninstalled libraries. */
#define LT_OBJDIR ".libs/"

/* Name of package */
#define PACKAGE "libxc"

/* Define to the address where bug reports for this package should be sent. */
#define PACKAGE_BUGREPORT "libxc@tddft.org"

/* Define to the full name of this package. */
#define PACKAGE_NAME "libxc"

/* Define to the full name and version of this package. */
#define PACKAGE_STRING "libxc 4.3.4"

/* Define to the one symbol short name of this package. */
#define PACKAGE_TARNAME "libxc"

/* Define to the home page for this package. */
#define PACKAGE_URL "http://www.tddft.org/programs/Libxc"

/* Define to the version of this package. */
#define PACKAGE_VERSION "4.3.4"

/* The size of `void*', as computed by sizeof. */
#define SIZEOF_VOIDP 8

/* If using the C implementation of alloca, define if you know the
   direction of stack growth for your system; otherwise it will be
   automatically deduced at runtime.
	STACK_DIRECTION > 0 => grows toward higher addresses
	STACK_DIRECTION < 0 => grows toward lower addresses
	STACK_DIRECTION = 0 => direction of growth unknown */
/* #undef STACK_DIRECTION */

/* Define to 1 if you have the ANSI C header files. */
#define STDC_HEADERS 1

/* Version number of package */
#define VERSION "4.3.4"

/* Define to `__inline__' or `__inline' if that's what the C compiler
   calls it, or to nothing if 'inline' is not supported under any name.  */
#ifndef __cplusplus
/* #undef inline */
#endif

/* Define to `unsigned int' if <sys/types.h> does not define. */
/* #undef size_t */
