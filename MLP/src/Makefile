#Tests
P_TEST := ../tests
P_TBLD_Hash := ../tests/HashTableTests
P_TBLD_Report := ../tests/Report
P_TBLD_Tests := ../tests/Tests
P_TBLD_MODEL := ../tests/ModelTests

#Targets
BUILD_DIR_MODEL := ../Model

.PHONY: install uninstall clean style style-fix cppcheck dvi dist

all: install

#app 
install:

uninstall:

dvi: 

dist:

tests: 
	cmake -S $(P_TEST) -B $(P_TBLD_Tests)
	cmake --build $(P_TBLD_Tests) --target Model
	../tests/Tests/Model

coverage: 
	cmake -S $(P_TEST) -B $(P_TBLD_Report)
	cmake --build $(P_TBLD_Report) --target coverage

#Uttils
clean:
	rm -rf $(P_TBLD)
	rm -rf /report
	rm -rf $(BUILD_DIR_MODEL)
	rm -rf text
	rm -rf file
	rm -rf $(P_TBLD_Tests)

style:
	clang-format -n 
style-fix:
	clang-format -i 

cppcheck:
	cppcheck --language=c++ --std=c++17 --suppress=missingIncludeSystem --suppress=unusedFunction .
