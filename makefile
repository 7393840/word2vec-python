PYTHONINCLUDEPATH = /usr/include/python3.5m
PYTHONLIBPATH = /usr/lib/x86_64-linux-gnu
BOOSTLIBPATH = /usr/local/lib
PYTHONLIB = python3.5m
BOOSTPYTHONLIB = boost_python3
BOOSTNUMPYLIB = boost_numpy3
CXX = g++
skip_gram.so: skip_gram.o
	$(CXX) -shared -fPIC -L$(PYTHONLIBPATH) -L$(BOOSTLIBPATH) skip_gram.o -o skip_gram.so -l$(PYTHONLIB) -l$(BOOSTPYTHONLIB) -l$(BOOSTNUMPYLIB)
skip_gram.o: skip_gram.cpp
	$(CXX) -fPIC -O3 -march=native -Wall -funroll-loops -Wno-unused-result -I$(PYTHONINCLUDEPATH) -c skip_gram.cpp -o skip_gram.o -std=c++11
clean:
	rm -rf *.o *.so
