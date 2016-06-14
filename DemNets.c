#include "Python.h"
#include "numpy/arrayobject.h"
#include <fcntl.h>
#include <math.h>
#include <omp.h>

#define VERSION "0.1"

static PyArrayObject *
network(const double *z, const unsigned int n, const unsigned int *tri, const unsigned int m) {
	PyArrayObject *net;
	npy_intp *dim;
	unsigned int i, j, k, l;
	unsigned int *lst[n], *lp, *e;
	unsigned int u, v, w, vex, wex;
	
	// alloc list of arrays
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

	// retrieve downwards pointing links from the triangulation
	l = 0;
	for(i = 0; i < 3*m; i += 3) {
		for(j = 0; j < 3; j++) {
			u = tri[i+j];
			v = tri[i+(j+1)%3];
			w = tri[i+(j+2)%3];
			if(z[v] >= z[u] && z[w] >= z[u])
				continue;
			lp = lst[u];
			vex = wex = 0;
			for(k = 2; k < lp[1]; k++) {
				if(lp[k] == v)
					vex = 1;
				if(lp[k] == w)
					wex = 1;
			}
			if(!vex && z[v] < z[u]) {
				lp[lp[1]++] = v;
				l++;
			}
			if(!wex && z[w] < z[u]) {
				lp[lp[1]++] = w;
				l++;
			}
			if(lp[1] >= lp[0]) {
				lp[0] = lp[1] + 2;
				lp = realloc(lp, lp[0] * sizeof(unsigned int));
			}
		}
	}

	// alloc numpy array
	dim = malloc(sizeof(npy_intp));
	dim[0] = l + n + 1;
	net = (PyArrayObject *) PyArray_ZEROS(1, dim, PyArray_UINT, 0);
	free(dim);
	if(!net) {
		PyErr_SetString(PyExc_MemoryError, "...");
		return NULL;
	}
	e = (unsigned int *) net->data;

	// store in compressed row format
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
	return net;
}

static PyArrayObject *
slopes(const double *x, const double *y, const double *z, const unsigned int *net) {
	PyArrayObject *slp;
	npy_intp *dim;
	unsigned i, j, k, n;
	double *s, dx, dy;

	// alloc numpy array
	n = net[0] - 1;
	dim = malloc(sizeof(npy_intp));
	dim[0] = net[n];
	slp = (PyArrayObject *) PyArray_ZEROS(1, dim, PyArray_DOUBLE, 0);
	free(dim);
	if(!slp) {
		PyErr_SetString(PyExc_MemoryError, "...");
		return NULL;
	}
	s = (double *) slp->data;

	// estimate slopes along links
	for(i = 0; i < n; i++) {
		for(k = net[i]; k < net[i+1]; k++) {
			j = net[k];
			dx = x[i] - x[j];
			dy = y[i] - y[j];
			s[k] = (z[i] - z[j]) / sqrt(dx*dx + dy*dy);
		}
	}
	return slp;
}

static PyObject *
DemNets_FlowNetwork(PyObject *self, PyObject* args) {
	PyObject *elearg, *triarg;
	PyArrayObject *ele, *tri, *net;
	unsigned int n;

	// parse input
	if(!PyArg_ParseTuple(args, "OO", &elearg, &triarg))
		return NULL;
	ele = (PyArrayObject *) PyArray_ContiguousFromObject(elearg, PyArray_DOUBLE, 1, 1);
	tri = (PyArrayObject *) PyArray_ContiguousFromObject(triarg, PyArray_UINT, 2, 2);
	if(!ele || !tri)
		return NULL;

	// retrieve flow network from the triangulation and elevation
	net = network((double *)ele->data, ele->dimensions[0], (unsigned int *)tri->data, tri->dimensions[0]);
	Py_DECREF(ele);
	Py_DECREF(tri);
	return PyArray_Return(net);
}

static PyObject *
DemNets_Slopes(PyObject *self, PyObject* args) {
	PyObject *xarg, *yarg, *zarg, *netarg;
	PyArrayObject *x, *y, *z;
	PyArrayObject *net, *slp;
	unsigned int *e, n;

	// parse input
	if(!PyArg_ParseTuple(args, "OOOO", &xarg, &yarg, &zarg, &netarg))
		return NULL;
	x = (PyArrayObject *) PyArray_ContiguousFromObject(xarg, PyArray_DOUBLE, 1, 1);
	y = (PyArrayObject *) PyArray_ContiguousFromObject(yarg, PyArray_DOUBLE, 1, 1);
	z = (PyArrayObject *) PyArray_ContiguousFromObject(zarg, PyArray_DOUBLE, 1, 1);
	net = (PyArrayObject *) PyArray_ContiguousFromObject(netarg, PyArray_UINT, 1, 1);
	if(!x || !y || !z || !net)
		return NULL;

	// check input
	n = x->dimensions[0];
	if(n != y->dimensions[0]) {
		PyErr_SetString(PyExc_IndexError, "(x, y) not of the same dimension.");
		return NULL;
	}
	if(n != z->dimensions[0]) {
		PyErr_SetString(PyExc_IndexError, "(x, y, z) not of the same dimension.");
		return NULL;
	}
	e = (unsigned int *) net->data;
	if(n != e[0] - 1) {
		PyErr_SetString(PyExc_IndexError, "(x, y, z) does not match with the network.");
		return NULL;
	}
	if(e[n] != net->dimensions[0]) {
		PyErr_SetString(PyExc_IndexError, "corrupted network format.");
		return NULL;
	}

	// get slopes
	slp = slopes((double *)x->data, (double *)y->data, (double *)z->data, e);

	Py_DECREF(x);
	Py_DECREF(y);
	Py_DECREF(z);
	Py_DECREF(net);
	return PyArray_Return(slp);
}

static PyMethodDef DemNets_methods[] = {
	{"FlowNetwork", DemNets_FlowNetwork, METH_VARARGS, "..."},
	{"Slopes", DemNets_Slopes, METH_VARARGS, "..."},
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
