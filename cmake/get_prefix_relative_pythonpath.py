#!/usr/bin/env python

# Script to find the python module install path relative to the base path that the interpreter is installed to.
import setuptools.dist
import setuptools.command.install

# use fake prefix which we can replace later, and which is unlikely to be in the real path
REPLACE_PREFIX="/placeholder_to_be_replaced_with_real_prefix"

distrib = setuptools.dist.Distribution()
install_cmd = setuptools.command.install.install(distrib)
install_cmd.prefix = REPLACE_PREFIX
install_cmd.finalize_options()
print(install_cmd.install_purelib.replace(REPLACE_PREFIX, ""))