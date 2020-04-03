# \file Makefile
# \verbatim
# Top contributors (to current version):
#   Guy Katz
# This file is part of the Reluplex project.
# Copyright (c) 2016-2017 by the authors listed in the file AUTHORS
# (in the top-level source directory) and their institutional affiliations.
# All rights reserved. See the file COPYING in the top-level source
# directory for licensing information.\endverbatim
#

ROOT_DIR = .

SUBDIRS = \
	\
	adversarial \

all:
	rm -f bin/*
	cp adversarial/adversarial.elf bin
	@echo Done

include $(ROOT_DIR)/Rules.mk

#
# Local Variables:
# compile-command: "make -C . "
# End:
#
