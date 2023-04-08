---
layout: home
title:  "Learning Python internals - Building a C Extension"
date:   2021-06-26
permalink: /python-c-extension/
categories: c python 
image: python-extension.png
---

# Introduction

In this post we'll be reviewing Python internals by making a simple Python extension in C using the C python API.
We'll explore things like module definition, function creation, garbage collection (memory management) and error treatment.
As an example we'll write a **matrix multiplication** module and attempt to use it in Python.

> :speaker: **A note on python extensions**: While normally forgotten, when writing CPU bound applications, i.e for ML programs
> 90% of the work is orchestrating C(or any other lower level language) function calls with Python. Python extensions can come very
> handy for performance critical operations whenever they are executed millions of times which can boost application performance.


## The extension


Let's start by creating a `matmul.c` file where we'll define our matrix multiplication extension.
The first thing to do is including the python header file which will include the Python C API and defining `PY_SSIZE_T_CLEAN` before including it.
```c
#define PY_SSIZE_T_CLEAN // For all # variants of formats (s#, y#, etc.), the macro PY_SSIZE_T_CLEAN must be defined before including Python.h. Using Py_SSIZT_T instead of int whenever we parse # arguments.
#include <Python.h>
```

Below you can see the module initialization, where we define a custom error `InvalidMatSize`.
In python, there is a global, per-thread, error indicator, which states the last error that happened in the program.
This error is created with `PyErr_NewException` under `matmul.error` which creates a new exception class (which is also an object for garbage collection).

More on this in the [documentation](https://docs.python.org/3/c-api/exceptions.html#exception-handling).

```c
static PyObject* InvalidMatSize;

PyMODINIT_FUNC PyInit_matmul(void){    
  PyObject* m = PyModule_Create(&matmulmodule);    
  if (m == NULL){    
    return NULL;    
  }    
    
  InvalidMatSize = PyErr_NewException("matmul.error", NULL, NULL);    
  Py_XINCREF(InvalidMatSize);    
    
  if (PyModule_AddObject(m, "error", InvalidMatSize) < 0) {    
    Py_XDECREF(InvalidMatSize);    
    Py_CLEAR(InvalidMatSize);    
    Py_DECREF(m);    
    return NULL;    
  }    
    
    
  return m;    
} 
```

In the snippet above you can see a `matmulmodule` being initialized and created through `PyModule_Create`.
This variable is a struct definition `PyModuleDef` defined below.

1. The first `m_base` argument is always initialized to `PyModuleDef_HEAD_INIT`.
2. The second `const char*` argument is the name of the module, which we named *"matmul"*
3. The third `const char*` argument is the module documentation, which we will carefully ignore (:stuck_out_tongue:).
4. The fourth argument `Py_ssize_t` is the module size which is allocated on module creation and freed when the module object is deallocated. This is useful if the module is supposed to work for multiple sub-interpreters, but we initialize this to -1 since we don't want to keep any state.
5. The final provided argument are the `PyMehtodDef MatMulMethods` that are our module functions, which we will see later in the document.

[Documentation](https://docs.python.org/3/c-api/module.html#c.PyModuleDef).


```c
static struct PyModuleDef matmulmodule = {
  PyModuleDef_HEAD_INIT,
  "matmul", /*name of the module*/
  NULL, /* module documentation, may be NULL */
  -1, /*  size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
  MatMulMethods
};
```


We also need to define our module functions in a `PyMethodDef` static variable array.
Each element is a definiton of a:
1. `const char*` function name. We defined 2 functions, *"matmul"* and a dummy one called "list_copy".
2. `PyCFunction` ml_meth which is the function implementation. We referenced the function definitions which you'll see later.
3. `int` the flah indicating how the function call will be constructed. We initialized to `MET_VARGARGS` since we only want positional arguments, not named arguments (simplicity).
4. `const char*` Function documentation.

[Documentation](https://docs.python.org/3/c-api/structures.html#c.PyMethodDef)


```c
static PyMethodDef MatMulMethods[]= {
  {"matmul", matmul, METH_VARARGS, "Matrix Multiplication"},
  {"list_copy", listCopy, METH_VARARGS, "List Copy"},
  {NULL, NULL, 0, NULL} /* Sentinel to check if the definition has finished*/
};
```


Below you can check the dummy `listCopy` function which takes as an argument a list and returns a new list with the same contents
as the original one.
Notice that everything is treated as a `PyObject`, and it doesn't really matter the type of variable we're dealing with. We need to treat it as a `PyObject`. Luckily for us, the C Python API implements multiple sub APIS for dealing with specific variable types. I.e take a look at the one we're going to use the most in here, the [C Python list API](https://docs.python.org/3/c-api/list.html).
Since we defined `METH_VARARGS`, we need use `PyArg_ParseTuple` to parse the `args` variable. Notice how we pass a *"O"* string and a pointer of a pointer to where we want to store the argument list.
Since we wanted to deal with lists we needed to pass "O" as an argument as you can check int the [documentation](https://docs.python.org/3/c-api/arg.html#other-objects).
Then we used `PyList_Size` to find the length of the passed list, and we create a new one with the same size, using `PyList_New` which will 
return a new `list` reference.
After that we loop over the length of the original list, we fetch the current index position of the original list using `PyList_GetItem` and set it
in the same index in the list we wish to return using `PyList_SetItem`.
We can stop now to explain Python reference counting.
The **first part** of the Python garbage collection is through the reference counting mechanism.
Each variable has a reference count, which is the total number of references for the current variable. If the count reaches 0, the object is garbage collected.

From the [documentation](https://pythonextensionpatterns.readthedocs.io/en/latest/refcount.html#pyobjects-and-reference-counting):

> In Python C extensions you always create and deallocate these PyObjects indirectly. Creation is via Python’s C API and destruction is done by decrementing the reference count. If this count hits zero then CPython will free all the resources used by the object.

You can check the reference count of a variable in Python as well.

```python
import sys
a = "Test"
sys.getrefcount(a)
```

Notice the `Py_INCREF(value)` call which increments the reference count of the returned value from the `PyList_GetItem` function call above. 
We need to do this because the `PyList_GetItem` call returns a [*"borrowed"*](https://pythonextensionpatterns.readthedocs.io/en/latest/refcount.html#borrowed-references) reference, and so if we don't increment the reference count, all the items set to the `result` list would be garbage collected if the original list was garbage collected. And so, in order for this to work our function has the responsibility to increment the reference count.


```c
static PyObject* listCopy(PyObject* self, PyObject* args){
  PyObject* original;
  
  if(!PyArg_ParseTuple(args, "O", &original)){
    return NULL;
  }
  
  Py_ssize_t matSize = PyList_Size(original);
  PyObject* result = PyList_New(matSize);

  for(Py_ssize_t i = 0; i < matSize; i++){
    PyObject* value = PyList_GetItem(original, i);
    Py_INCREF(value);
    PyList_SetItem(result, i, value);
  }

  return result;
}
```

Below you can check the matrix multiplication function.
Differences from the `list_copy` include we need now 2 lists, not only one, and so we add an extra "O".
We also raise an error with `PyErr_SetString` if the matrix lengths are not the same. The raised error is the one
we created above `InvalidMatSize`.
This sets the error indicator.

Inside the `matrixMultiplication` function, you'll see the usage of `PyNumber_Multiply` to multiply the elements of the current row and column
and `PyNumber_Add` add the result to the total.

```c
static PyObject* matmul(PyObject* self, PyObject* args){    
  PyObject* firstMat;    
  PyObject* secondMat;    
    
  if(!PyArg_ParseTuple(args, "OO", &firstMat, &secondMat)){    
    return NULL;    
  }    
    
  Py_ssize_t matSize = PyList_Size(firstMat);    
  Py_ssize_t secondmatSize = PyList_Size(secondMat);    
    
  if(matSize != secondmatSize){    
    PyErr_SetString(InvalidMatSize, "Matrix len(s) should be the same.");    
    return NULL;    
  }    
    
  PyObject* returnMat = matrixMultiplication(firstMat, secondMat, matSize);    
    
  return returnMat;    
} 


PyObject* matrixMultiplication(PyObject* first, PyObject* second, Py_ssize_t matSize){
    PyObject* result = PyList_New(matSize);
    for(Py_ssize_t i = 0; i < matSize ; ++i){
        PyObject* row = PyList_New(matSize);
        for(Py_ssize_t j = 0; j < matSize; ++j){
          PyObject* zero = Py_BuildValue("i", 0);
          PyList_SetItem(row, j, zero);
        }
      PyList_SetItem(result, i, row);
    }

    for(Py_ssize_t i = 0; i < matSize ; ++i){
        PyObject* firstRow = PyList_GetItem(first, i);
        PyObject* targetRow = PyList_GetItem(result, i);
  
        for(Py_ssize_t j = 0; j < matSize; ++j){
            PyObject* initialValue = PyList_GetItem(targetRow, j);
            for(Py_ssize_t k  = 0; k < matSize; ++k){
                PyObject* firstValue = PyList_GetItem(firstRow, k);
                PyObject* secondRow = PyList_GetItem(second, k);
                PyObject* secondValue = PyList_GetItem(secondRow, j);    
                PyObject* multValues = PyNumber_Multiply(firstValue, secondValue);
                //result[i*matSize + j] += first[i*matSize + k] * second[k*matSize + j];
                initialValue = PyNumber_Add(initialValue, multValues);
            } 
            PyList_SetItem(targetRow, j, initialValue);
        }
    }
    return result;
}
```

## Building the extension with Python

Now we are ready to create and build the extension with python.
We create a `extension.py` file with the following code.
We need only to create an `Extension` object, whose documentation you can check [here](https://docs.python.org/3/distutils/apiref.html#distutils.core.Extension).

1. `python extension.py build` - To compile the extension.
2. `python extension.py install` - To install the extension as a python package.


```python
from distutils.core import setup, Extension
import os

os.environ['CFLAGS'] = '-mavx'

module1 = Extension(
    'matmul',
    define_macros = [('MAJOR_VERSION', '1'),
                     ('MINOR_VERSION', '0')],
    include_dirs = ['/usr/local/include'],
    library_dirs = ['/usr/local/lib'],
    sources = ['matmul.c'],
    extra_compile_args=["-mavx", "-O3"]
)

setup (name = 'matmul',
       version = '1.0',
       description = 'This is a demo package',
       author = 'Andre Fernandes',
       author_email = 'fernandoandre49@gmail.com',
       url = 'https://docs.python.org/extending/building',
       long_description = '''
            This is really just a demo package.
        ''',
       ext_modules = [module1],
)
```

# Using the extension

We'll initialize 2 random squared matrices with 200*200 elements.
We also created a function in python to compute a matrix multiplication of the previous matrices.
Then, we time the 2 exections, the C extension and the standard python one.


```python
import matmul
import timeit
import random

MAT_SIZE = 200
first = [[random.randint(1, 100)] * MAT_SIZE for _ in range(MAT_SIZE)]
second = [[random.randint(1, 100)] * MAT_SIZE for _ in range(MAT_SIZE)]

def python_matmul(X, Y):
    result = [
        [0]*len(X)
        for _ in range(len(X))
    ]
    for i in range(len(X)):
        for j in range(len(Y[0])):
           for k in range(len(Y)):
               result[i][j] += X[i][k] * Y[k][j]
    return result


time_took_python_ms = timeit.timeit(lambda: python_matmul(first, second), number=5) * 1000
time_took_matmul_ms = timeit.timeit(lambda: matmul.matmul(first, second), number=5) * 1000

print(f"Python took {time_took_python_ms} ms.")
print(f"Simple C extension took {time_took_matmul_ms} ms."
```

> Python took 4549.059671999203 ms.
> Simple C extension took 1533.0800699994143 ms.
