/* Test source lifted from /usr/share/autoconf/autoconf/c.m4
 * Used to check if the compiler supports the inline keyword */
typedef int foo_t;

static inline foo_t static_foo()
{
	return 0;
}

foo_t foo()
{
	return 0;
}

int main(int argc, char *argv[])
{
	return 0;
}
