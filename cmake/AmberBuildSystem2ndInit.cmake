# This script handles the parts of the Amber CMake init that must happen AFTER the enable_language() command,
# because files included by this use compile tests

# standard library and should-be-in-the-standard-library includes
# --------------------------------------------------------------------

include(TargetArch)
include(ExternalProject)
include(CheckFunctionExists)
include(CheckFortranFunctionExists)
include(CheckIncludeFile)
include(CheckIncludeFileCXX)
include(CheckCSourceRuns)
include(CheckCCompilerFlag)
include(CheckCXXCompilerFlag)
include(CheckLinkerFlag)
include(CheckFortranSourceRuns)
include(CheckSymbolExists)
include(CheckConstantExists)
include(CheckLibraryExists)
include(CheckPythonPackage)
include(CheckTypeSize)
include(CMakePushCheckState)
include(TryLinkLibrary)

# Amber include files
# --------------------------------------------------------------------

include(CheckBoostWorks)
include(VerifyCompilerConfig)
include(InstallWrapped)
include(LibraryTracking)
include(DownloadHttps)
include(Replace)
include(BuildReport)
include(ConfigModuleDirs)
include(ApplyOptimizationDeclarations)
include(CompilationOptions)
include(RPATHConfig)
include(CopyTarget)
include(LibraryUtils)
