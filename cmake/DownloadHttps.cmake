# Module which uses native tools to download files using https (which is NOT supported on all platforms by CMake)
# Created Variables - HAVE_DOWNLOADER_PROGRAM -- whether or not any program capable of downloading files was found

if(UNIX)
	find_program(WGET NAMES wget DOC "Path to the wget program.  Used to download files.")
	find_program(CURL NAMES curl DOC "Path to the curl program.  Used to download files.")
endif()

if(WIN32)
	find_program(POWERSHELL NAMES powershell DOC "Path to the Windows Powershell executable.  Used to download files")
endif()

set(HAVE_DOWNLOADER_PROGRAM TRUE)

# figure out which program to use, and its respective command
if(UNIX AND CURL)
	set(DOWNLOADER_PROGRAM ${CURL})
	set(DOWNLOAD_COMMAND -L "-#" -o <destfile> <url>) 
elseif(UNIX AND WGET)
	set(DOWNLOADER_PROGRAM ${WGET})
	set(DOWNLOAD_COMMAND -O <destfile> <url>)
elseif(WIN32 AND POWERSHELL)
	set(DOWNLOADER_PROGRAM ${POWERSHELL})
	
	# from http://superuser.com/questions/25538/how-to-download-files-from-command-line-in-windows-like-wget-is-doing
	set(DOWNLOAD_COMMAND -Command "(new-object System.Net.WebClient).DownloadFile('<url>','<destfile>')")
else()
	set(DOWNLOADER_PROGRAM DOWNLOADER_PROGRAM-NOTFOUND)
	set(HAVE_DOWNLOADER_PROGRAM FALSE)
endif()

# Downloads a file from a URL.  The URL may be https, and things will still work.
# URL - the URL to dowload the file from
# DESTFILE - the location to save the download to.  The file there will be overwritten if it exists.
# FAIL_ON_ERROR - whether or not to print a FATAL_ERROR if the download fails.
function(download_file_https URL DESTFILE FAIL_ON_ERROR)
	if(NOT HAVE_DOWNLOADER_PROGRAM)
		message(FATAL_ERROR "A downloader program, either curl, wget, or powershell, is required to download files.  Please set CURL, WGET, or POWERSHELL to the location of the respective program.")
	endif()
	
	string(REPLACE "<url>" "${URL}" CONFIGURED_DOWNLOAD_COMMAND "${DOWNLOAD_COMMAND}")
	string(REPLACE "<destfile>" "${DESTFILE}" CONFIGURED_DOWNLOAD_COMMAND "${CONFIGURED_DOWNLOAD_COMMAND}")
	
	#message("Executing command: ${DOWNLOADER_PROGRAM} ${CONFIGURED_DOWNLOAD_COMMAND}")
	
	execute_process(COMMAND ${DOWNLOADER_PROGRAM} ${CONFIGURED_DOWNLOAD_COMMAND} RESULT_VARIABLE DOWNLOAD_RESULT)
	
	if((NOT ${DOWNLOAD_RESULT} EQUAL 0) AND FAIL_ON_ERROR)
		message(FATAL_ERROR "Unable to download file ${URL}")
	endif()
endfunction(download_file_https)