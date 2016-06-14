#include "Python.h"
#include "numpy/arrayobject.h"
#include <fcntl.h>
#include <math.h>
#include <omp.h>

#define VERSION "0.1"

static PyObject *
slopes(const double *x, const double *y, const double *z, const unsigned int n, const unsigned int *tri, const unsigned int m) {
	PyArrayObject *net, *slp;
	npy_intp *dim;
	unsigned int i, j, k, l;
	unsigned int *lst[n], *lp, *e;
	unsigned int u, v, w, vex, wex;
	
	for(i = 0; i < n; i++) {
		lp = malloc(8 * sizeof(unsigned int));
		if(!lp) {
			PyErr_SetString(PyExc_MemoryError, "...");
			return NULL;
		}
		lp[0] = 8;
		lp[1] = 2;
		lst[i] = lp;
	}
	l = 0;
	for(i = 0; i < 3*m; i += 3) {
		for(j = 0; j < 3; j++) {
			u = tri[i+j];
			v = tri[i+(j+1)%3];
			w = tri[i+(j+2)%3];
			if(z[v] > z[u] && z[w] > z[u])
				continue;
			lp = lst[u];
			vex = wex = 0;
			for(k = 2; k < lp[1]; k++) {
				if(lp[k] == v)
					vex = 1;
				if(lp[k] == w)
					wex = 1;
			}
			if(!vex && z[v] <= z[u]) {
				lp[lp[1]++] = v;
				l++;
			}
			if(!wex && z[w] <= z[u]) {
				lp[lp[1]++] = w;
				l++;
			}
			if(lp[1] >= lp[0]) {
				lp[0] = lp[1] + 2;
				lp = realloc(lp, lp[0] * sizeof(unsigned int));
			}
		}
	}
	dim = malloc(sizeof(npy_intp));
	dim[0] = l + n + 1;
	net = (PyArrayObject *) PyArray_ZEROS(1, dim, PyArray_UINT, 0);
	slp = (PyArrayObject *) PyArray_ZEROS(1, dim, PyArray_UINT, 0);
	free(dim);
	if(!net || !slp) {
		PyErr_SetString(PyExc_MemoryError, "...");
		return NULL;
	}
	e = (unsigned int *) net->data;
	j = n + 1;
	for(i = 0; i < n; i++) {
		lp = lst[i];
		e[i] = j;
		for(k = 2; k < lp[1]; k++) {
			e[j++] = lp[k];
		}
		free(lp);
	}
	e[n] = j;
	return Py_BuildValue("(OO)", net, slp);
}

static PyObject *
DemNets_Slopes(PyObject *self, PyObject* args) {
	PyObject *xarg, *yarg, *zarg, *triarg, *SlopeNet;
	PyArrayObject *x, *y, *z, *tri;
	unsigned int n;

	if(!PyArg_ParseTuple(args, "OOOO", &xarg, &yarg, &zarg, &triarg))
		return NULL;
	x = (PyArrayObject *) PyArray_ContiguousFromObject(xarg, PyArray_DOUBLE, 1, 1);
	y = (PyArrayObject *) PyArray_ContiguousFromObject(yarg, PyArray_DOUBLE, 1, 1);
	z = (PyArrayObject *) PyArray_ContiguousFromObject(zarg, PyArray_DOUBLE, 1, 1);
	tri = (PyArrayObject *) PyArray_ContiguousFromObject(triarg, PyArray_UINT, 2, 2);
	if(!x || !y || !z || !tri)
		return NULL;
	n = x->dimensions[0];
	if(n != y->dimensions[0])
		return NULL;
	if(n != z->dimensions[0])
		return NULL;
	
	SlopeNet = slopes((double *)x->data, (double *)y->data, (double *)z->data, n,
					  (unsigned int *)tri->data, tri->dimensions[0]);
	Py_DECREF(x);
	Py_DECREF(y);
	Py_DECREF(z);
	Py_DECREF(tri);
	return SlopeNet;
}

static PyMethodDef DemNets_methods[] = {
	{"Slopes", DemNets_Slopes, METH_VARARGS,
	 "..."},
	{NULL, NULL, 0, NULL}
};

void
initDemNets(void) {
	PyObject *m;
	PyObject *v;

	v = Py_BuildValue("s", VERSION);
	PyImport_AddModule("DemNets");
	m = Py_InitModule3("DemNets", DemNets_methods,
	"Digital Elevation Model Networks");
	PyModule_AddObject(m, "__version__", v);
	import_array();
}

int
main(int argc, char **argv) {
	Py_SetProgramName(argv[0]);
	Py_Initialize();
	initDemNets();
	Py_Exit(0);
	return 0;
}
