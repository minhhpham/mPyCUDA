#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <Python.h>
#include <boost/python.hpp>
#include <PyDataParse.h>

// include methods 
#include <deviceQuery.cuh>
#include <reduce.cuh>


BOOST_PYTHON_MODULE(mPyCUDA)
{
    namespace python = boost::python;

    // Register interable conversions from Python
    IterableConverter().from_python<std::vector<std::vector<double> > >();
    IterableConverter().from_python<std::vector<double> >();

    // register functions in this module
    python::def("deviceQuery", &runDeviceQuery, 
        "Print information for all GPUs \n"
        "Return Nothing, print info to stdout"
    );

    python::def("sumReduce", &sumReduce,
        "perform sum reduce on GPU \n "
        "   arg1: a list of float/int"
    );
}