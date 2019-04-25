#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarraytypes.h>
#include <numpy/ufuncobject.h>

#define LT(n) n, n, n, n, n, n, n, n, n, n, n, n, n, n, n, n
static const char log_table_256[256] = 
{
    -1, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
    LT(4), LT(5), LT(5), LT(6), LT(6), LT(6), LT(6),
    LT(7), LT(7), LT(7), LT(7), LT(7), LT(7), LT(7), LT(7)
};

static PyObject* log2_fast_lookup_table(PyObject *self, PyObject *args) {
    const long v; // pointer to the unsigned 32-bit word passed from Python
    uint8_t r; // result
    uint32_t t, tt; // temporaries stored in the CPU registers

    if (!PyArg_ParseTuple(args, "l", &v))
        return NULL;

    /* Raise a Python ValueError exception if the input is < 0 */
    if (v < 0) {
        PyErr_SetString(PyExc_ValueError, "The input must be > 0");
        return NULL;
    }

    // Take the 32 first bit of the long as the value to compute the log base 2
    /*uint32_t v = in & 0xffffffff;*/

    if ((tt = (v >> 16))) {
        r = (t = (tt >> 8)) ? (24 + log_table_256[t]) : (16 + log_table_256[tt]);
    } else {
        r = (t = (v >> 8)) ? (8 + log_table_256[t]) : log_table_256[v];
    }

    return PyLong_FromLong(r);
}

static PyMethodDef log2_methods[] = {
    /* List the C methods here */
    {"fast_lookup_table_c",  log2_fast_lookup_table, METH_VARARGS,
     "Compute the log base 2 of a unsigned 32-bit word"},
    /* Sentinel */
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef log2module = {
    PyModuleDef_HEAD_INIT,
    "_core",   /* name of module */
    NULL,      /* module documentation, may be NULL */
    -1,        /* size of per-interpreter state of the module,
                  or -1 if the module keeps state in global variables. */
    log2_methods
};

/* log2_fast_lookup_table numpy UFUNC */
static void log2_fast_lookup_table_ufunc(
    char **args,
    npy_intp *dimensions,
    npy_intp* steps,
    void* data) {

    npy_intp i;
    npy_intp n = dimensions[0];
    char *in = args[0], *out = args[1];
    npy_intp in_step = steps[0], out_step = steps[1];

    uint32_t v;
    uint32_t t, tt;

    for (i = 0; i < n; i++) {
        /*BEGIN main ufunc computation*/
        v = *(uint32_t *)in;
        if ((tt = (v >> 16))) {
            *((uint8_t *)out) = (t = (tt >> 8)) ? (24 + log_table_256[t]) : (16 + log_table_256[tt]);
        } else {
            *((uint8_t *)out) = (t = (v >> 8)) ? (8 + log_table_256[t]) : log_table_256[v];
        }
        /*END main ufunc computation*/

        in += in_step;
        out += out_step;
    }
}

/*This a pointer to the above function*/
PyUFuncGenericFunction funcs[1] = {&log2_fast_lookup_table_ufunc};

/* These are the input and return dtypes of logit.*/
static char types[2] = {NPY_UINT32, NPY_UINT8};
static void *data[1] = {NULL};

/* PyMODINIT_FUNC is a macro equivalent to an extern "C". 
This should be the only non-static object declared in the
extension file */
PyMODINIT_FUNC PyInit__core(void) {
    PyObject *module, *log2_fast_lt_ufunc_obj, *d;

    module = PyModule_Create(&log2module);
    if (module == NULL)
        return NULL;

    import_array();
    import_umath();
    
    log2_fast_lt_ufunc_obj = PyUFunc_FromFuncAndData(funcs, data, types, 1, 1, 1,
                                    PyUFunc_None, "log2_fast_lookup_table_c_ufunc",
                                    "Compute the log base 2 of a unsigned 32-bit word (UFUNC)", 0);

    d = PyModule_GetDict(module);

    PyDict_SetItemString(d, "log2_fast_lookup_table_c_ufunc", log2_fast_lt_ufunc_obj);
    Py_DECREF(log2_fast_lt_ufunc_obj);

    return module;
}
