// This file tests that stdlib.h can be included without an error.
// It's used to auto-detect the need for a workaround for a certain compiler issue.

#include <stdlib.h>

int main(int argc, char** argv)
{
	return 0;
}