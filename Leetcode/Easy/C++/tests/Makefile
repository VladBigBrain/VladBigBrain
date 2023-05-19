.PHONY: test clean

P_TEST := .
P_TBLD := ../tests/build
TESTDIR := ../tests
test: clean
	cmake -S $(P_TEST) -B $(P_TBLD)
	cmake --build $(P_TBLD)
	cd $(P_TBLD); ctest

cppcheck:
	cppcheck --language=c++ --std=c++20 .

clean:
	rm -rf $(TESTDIR)
	rm -rf $(P_TBLD)
	rm -rf /report
	rm -rf /build
	rm -rf $(BUILD_DIR)
	rm -rf text
	rm -rf file
