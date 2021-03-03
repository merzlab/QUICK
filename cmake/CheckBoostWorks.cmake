# Script to verify that Boost works and was built with support for zlib and bzip2 compression.
# Unfortunately, CMake's Boost packages don't generally verify that Boost is actually linkable, 
# even though there are certain issues (incompatible compilers on Windows, incompatible STLs on OSX,
# incompatible CXX ABIs on Linux) that can cause a found Boost library to not work.


function(check_boost_works RESULT_VAR)

	if(DEFINED ${RESULT_VAR})
		set(BOOST_WORKS_FIRST_RUN FALSE)
	else()
		set(BOOST_WORKS_FIRST_RUN TRUE)
	endif()

	# NOTE: it's important to attempt calling a function that takes a string argument.
	# This will catch libstdc++ c++11 ABI incompatibility issues where functions that pass strings break.
	try_link_library(${RESULT_VAR}
		LANGUAGE CXX
		FUNCTION "boost::system::system_error()"
		LIBRARIES ${Boost_LIBRARIES}
		INCLUDES ${Boost_INCLUDE_DIRS}
		FUNC_DECLARATION "
#include <boost/system/system_error.hpp>
#include <string>"
        FUNC_CALL "
boost::system::system_error testError(boost::system::error_code(), std::string(\"this seems erroneous\"))")


	if(NOT ${RESULT_VAR} AND BOOST_WORKS_FIRST_RUN)
		message(WARNING "The boost library ${Boost_SYSTEM_LIBRARY} cannot be linked.  This could be due to an incompatible compiler, STL, or ABI.")
	endif()
endfunction(check_boost_works)

function(check_boost_compression_support RESULT_VAR)

	if(DEFINED ${RESULT_VAR})
		set(COMPRESSION_CHECK_FIRST_RUN FALSE)
	else()
		set(COMPRESSION_CHECK_FIRST_RUN TRUE)
	endif()

	try_link_library(${RESULT_VAR}
		LANGUAGE CXX
		FUNCTION "boost::iostreams::bzip2_decompressor() and boost::iostreams::gzip_decompressor()"
		LIBRARIES ${Boost_LIBRARIES}
		INCLUDES ${Boost_INCLUDE_DIRS}
		FUNC_DECLARATION "
#include <boost/iostreams/filter/bzip2.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filtering_stream.hpp>"
        FUNC_CALL "
boost::iostreams::filtering_istream in;
in.push(boost::iostreams::bzip2_decompressor());
in.push(boost::iostreams::gzip_decompressor())")


	if(NOT ${RESULT_VAR} AND COMPRESSION_CHECK_FIRST_RUN)
		message(WARNING "The boost iostreams library ${Boost_IOSTREAMS_LIBRARY} was not built with libz and libbz2 support, so boost cannot be used by Amber.  If you think you have fixed this issue you can cause the check to be rerun using the -U${RESULT_VAR} option to CMake.")
	endif()
endfunction(check_boost_compression_support)