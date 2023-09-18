# @author  barnards (Barnards Gibbering)

#------------------------------------------------#
#   INGREDIENTS                                  #
#------------------------------------------------#
# NAME					Name of project
# P_TEST 				Test Directory
# APP_PATH 				APP Directory PATH
# INSTALL_DIR 			Install Directory
# APP 					APP Name
# P_TBLD_Report 		GcovReport Directory
# P_TBLD_Tests 			All Tests Directory
# SOURCES     			Source files

# app
NAME := MLP
P_TEST := ../tests
APP_PATH := ./view
INSTALL_DIR := ../install
APP := $(NAME).app

# Tests
P_TBLD_Report := ../tests/Report
P_TBLD_Tests := ../tests/Tests

#Utils
SOURCES = $(shell find . \( -name '*.cc' -o -name '*.h' \) ! -path './libraries/*')


#------------------------------------------------#
#   RECIPES                                      #
#------------------------------------------------#
# all install

all: install

# app 
install:
	cmake -S $(APP_PATH) -B $(INSTALL_DIR)
	cmake --build $(INSTALL_DIR) 

uninstall:
	rm -rfv $(INSTALL_DIR)

run:
	$(INSTALL_DIR)/$(APP)

# tests
tests: 
	cmake -S $(P_TEST) -B $(P_TBLD_Tests)
	cmake --build $(P_TBLD_Tests) --config Debug --target Model
	$(P_TBLD_Tests)/Model

coverage: 
	cmake -S $(P_TEST) -B $(P_TBLD_Report)
	cmake --build $(P_TBLD_Report) --target coverage

# Uttils
clean: uninstall
	rm -rf $(P_TBLD)
	rm -rf /report
	rm -rf $(BUILD_DIR_MODEL)
	rm -rf text
	rm -rf file
	rm -rf *.tar.gz
	rm -rf $(P_TBLD_Tests)
	rm -rf $(P_TBLD_Report)

dvi:
	pdflatex docs.tex
	rm -f *.toc *.log *.aux *.dvi coverage.info

dist:
	cp -r $(INSTALL_DIR)/$(APP) ./$(APP)
	-tar -czvf $(NAME).tar.gz $(APP) */ *.txt Makefile
	rm -rf $(APP)

linter:
	clang-format -i -style=google -verbose $(SOURCES)
	clang-format -n -style=google -verbose $(SOURCES)

cppcheck:
	cppcheck --language=c++ --std=c++17 \
	--enable=all \
	--suppress=missingInclude \
	--suppress=ctuOneDefinitionRuleViolation \
	$(SOURCES) tests/main.cc

#------------------------------------------------#
#   SPEC                                         #
#------------------------------------------------#

.PHONY: clean tests linter cppcheck leaks install uninstall dist dvi
.SILENT:
