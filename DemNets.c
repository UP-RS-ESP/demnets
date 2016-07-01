#include "Python.h"
#include "numpy/arrayobject.h"
#include <fcntl.h>
#include <math.h>
#include <omp.h>

#define VERSION "0.1"

typedef struct Stack Stack;
typedef struct Queue Queue;
typedef struct Node Node;

struct Node {
    Node *next;
    int i;
    unsigned int d;
};

struct Queue {
    Node *first;
    Node *last;
};

struct Stack {
    Stack *head;
    int i;
};

int
initrng(void) {
    int i, rdev;
    long int seed;

    rdev = open("/dev/urandom", O_RDONLY);
    if(!rdev) {
        PyErr_SetString(PyExc_RuntimeError, "error: cannot open /dev/urandom\n");
        return 0;
    }
    read(rdev, &seed, sizeof(long int));
    close(rdev);
    srand48(seed);
    for(i = 0; i < 10; i++)
        drand48();
    return 1;
}

int
put(Queue *q,
    const unsigned int i,
    const unsigned int d) {
    Node *n;
    n = malloc(sizeof(Node));
    if(!n)
        return 1;
    n->i = i;
    n->d = d;
    if(!q->first) {
        q->first = q->last = n;
    } else {
        q->last->next = n;
        q->last = n;
    }
    n->next = NULL;
    return 0;
}

int
get(Queue *q,
    unsigned int *i,
    unsigned int *d) {
    Node *tmp;
    if(!q->first)
        return 1;
    *i = q->first->i;
    *d = q->first->d;
    tmp = q->first;
    q->first = q->first->next;
    free(tmp);
    return 0;
}

static PyArrayObject *
betweenness(const unsigned int *net,
            const double *lw,
            const double *nw) {
    PyArrayObject *betw;
    npy_intp *dim;
    double *b;

    unsigned int i, j, k, l, m, n;
    unsigned int d, *dd, oset;
    double *r, *db, *no, fact;
    Queue *open;
    Stack *done, *work, *tmp;

    double lwmax;

    n = net[0] - 1;
#pragma omp parallel
    m = omp_get_num_threads();

    r = calloc(n * m, sizeof(double));
    if(!r) {
        PyErr_SetString(PyExc_MemoryError, "...");
        return NULL;
    }
#pragma omp parallel for private(d,i,j,k,l,oset,fact,open,done,work,tmp,db,no,dd)
    for(j = 0; j < n; j++) {
        open = malloc(sizeof(Queue));
        done = malloc(sizeof(Stack));
        open->first = open->last = NULL;
        done->head = NULL;
        db = calloc(n, sizeof(double));
        no = calloc(n, sizeof(double));
        dd = calloc(n, sizeof(unsigned int));
        if(!db || !no || !dd) {
            PyErr_SetString(PyExc_MemoryError, "...");
            exit(EXIT_FAILURE);
        }
        no[j] = nw[j];
        dd[j] = 0;
        // put root into queue
        if(put(open, j, 0)) {
            PyErr_SetString(PyExc_MemoryError, "failed to fill queue ..");
            exit(EXIT_FAILURE);
        }
        while(!get(open, &i, &d)) {
            // add n to stack with increasing distance d
            work = malloc(sizeof(Stack));
            work->i = i;
            work->head = done->head;
            done->head = work;
            d++;
            // all neighbors l of i
            lwmax = 0;
            for(k = net[i]; k < net[i+1]; k++)
                if(lw[k] > lwmax)
                    lwmax = lw[k];
            for(k = net[i]; k < net[i+1]; k++) {
                if(lw[k] < 0.3333 * lwmax)
                    continue;
                l = net[k];
                if(no[l]) {
                    // we have seen l - more shortest paths
                    if(dd[l] == d)
                        no[l] += nw[l] * no[i];
                } else {
                    // first time at l
                    no[l] = nw[l] * no[i];
                    dd[l] = d;
                    if(put(open, l, d)) {
                        PyErr_SetString(PyExc_MemoryError, "failed to fill queue ..");
                        exit(EXIT_FAILURE);
                    }
                }
            }
        }
        // empty stack
        for(work = done->head; work->head != NULL;) {
            i = work->i;
            tmp = work;
            work = work->head;
            free(tmp);
            if(dd[i] <= 1)
                continue;
            fact = nw[i] * (nw[i] + db[i]) / no[i];
            // all neighbors l of i
            lwmax = 0;
            for(k = net[i]; k < net[i+1]; k++)
                if(lw[k] > lwmax)
                    lwmax = lw[k];
            for(k = net[i]; k < net[i+1]; k++) {
                if(lw[k] < 0.3333 * lwmax)
                    continue;
                l = net[k];
                // follow only shortest paths
                if(dd[l] == dd[i] - 1)
                    db[l] += no[l] * fact;
            }
        }
        free(open);
        free(done);
        free(no);
        free(dd);
        oset = n * omp_get_thread_num();
        for(k = 0; k < n; k++)
            r[k + oset] += nw[j] * db[k];
        free(db);
    }

    // alloc numpy array
    dim = malloc(sizeof(npy_intp));
    dim[0] = n;
    betw = (PyArrayObject *) PyArray_ZEROS(1, dim, PyArray_DOUBLE, 0);
    free(dim);
    if(!betw) {
        PyErr_SetString(PyExc_MemoryError, "...");
        return NULL;
    }
    b = (double *) betw->data;

    /* reduction of threaded array */
    for(i = 0; i < m; i++) {
        oset = i * n;
        for(j = 0; j < n; j++)
            b[j] += r[j + oset];
    }
    free(r);
    for(j = 0; j < n; j++)
        b[j] /= nw[j];

    return betw;
}

static PyArrayObject *
linklengths(const double *x,
            const double *y,
            const double *z,
            const unsigned int *net) {
    PyArrayObject *lln;
    npy_intp *dim;
    unsigned int i, j, k, n;
    double *l;
    double dx, dy, dz;
    double xi, yi, zi;

    // alloc numpy array
    n = net[0] - 1;
    dim = malloc(sizeof(npy_intp));
    dim[0] = net[n];
    lln = (PyArrayObject *) PyArray_ZEROS(1, dim, PyArray_DOUBLE, 0);
    free(dim);
    if(!lln) {
        PyErr_SetString(PyExc_MemoryError, "...");
        return NULL;
    }
    l = (double *) lln->data;

    // estimate 2-norm link distances i->j
    for(i = 0; i < n; i++) {
        xi = x[i];
        yi = y[i];
        zi = z[i];
        for(k = net[i]; k < net[i+1]; k++) {
            j = net[k];
            dx = xi - x[j];
            dy = yi - y[j];
            dz = zi - z[j];
            l[k] = sqrt(dx*dx + dy*dy + dz*dz);
        }
    }
    return lln;
}

static PyArrayObject *
linksinks(const unsigned int *net,
          const unsigned int *ten,
          const double *z) {
    PyArrayObject *new;
    npy_intp *dim;
    double zj;
    unsigned int i, j, k, l, n, m;
    unsigned int len, d, *e;
    unsigned int *seen, *list, link;
    Queue *que;

    n = net[0] - 1;
    m = 0;
    len = 2 + n / 10;
    list = malloc(len * sizeof(unsigned int));
    if(!list) {
        PyErr_SetString(PyExc_MemoryError, "...");
        exit(EXIT_FAILURE);
    }
    for(j = 0; j < n; j++) {
        if(net[j] == net[j+1]) {
            zj = z[j];
            que = malloc(sizeof(Queue));
            que->first = que->last = NULL;
            seen = calloc(n, sizeof(unsigned int));
            if(!seen) {
                PyErr_SetString(PyExc_MemoryError, "...");
                exit(EXIT_FAILURE);
            }
            // put sink into queue
            if(put(que, j, 0)) {
                PyErr_SetString(PyExc_MemoryError, "failed to fill queue ..");
                exit(EXIT_FAILURE);
            }
            link = 0;
            while(!get(que, &i, &d) && !link) {
                d++;
                seen[i]++;
                //fprintf(stderr, "%i ", d);
                if(len < m + 2) {
                    len += 24;
                    list = realloc(list, len * sizeof(unsigned int));
                    if(!list) {
                        PyErr_SetString(PyExc_MemoryError, "...");
                        exit(EXIT_FAILURE);
                    }
                }
                // all neighbors l of i
                for(k = net[i]; k < net[i+1]; k++) {
                    l = net[k];
                    if(seen[l])
                        continue;
                    if(z[l] <= zj) {
                        list[m++] = j;
                        list[m++] = l;
                        link = 1;
                        break;
                    }
                    put(que, l, d);
                }
                if(link)
                    break;
                for(k = ten[i]; k < ten[i+1]; k++) {
                    l = ten[k];
                    if(seen[l])
                        continue;
                    if(z[l] <= zj) {
                        list[m++] = j;
                        list[m++] = l;
                        link = 1;
                        break;
                    }
                    put(que, l, d);
                }
            }
            free(seen);
            free(que);
        }
    }

    // alloc numpy array
    dim = malloc(sizeof(npy_intp));
    dim[0] = n + m + 1;
    new = (PyArrayObject *) PyArray_ZEROS(1, dim, PyArray_UINT, 0);
    free(dim);
    if(!new) {
        PyErr_SetString(PyExc_MemoryError, "...");
        return NULL;
    }
    e = (unsigned int *) new->data;

    // store in compressed row format
    l = 0;
    j = n + 1;
    for(i = 0; i < n; i++) {
        e[i] = j;
        if(list[l] == i) {
            e[j++] = list[++l];
            l++;
        }
    }
    e[n] = j;
    return new;
}

static PyArrayObject *
flowdistance(const unsigned int *net,
             const double *lw,
             const double *ld,
             const unsigned int iter) {
    PyArrayObject *dst;
    npy_intp *dim;
    unsigned int i, k, l, u, v;
    unsigned int n, m, o, oset, nbrs;
    double *cum, p;
    double *c, *d;

    if(!initrng())
        return NULL;

    n = net[0] - 1;
#pragma omp parallel
    m = omp_get_num_threads();

    c = calloc(n * m, sizeof(double));
    if(!c) {
        PyErr_SetString(PyExc_MemoryError, "...");
        return NULL;
    }

#pragma omp parallel for private(i,k,l,u,v,p,cum,oset,nbrs)
    for(i = 0; i < n * iter; i++) {
        oset = n * omp_get_thread_num();
        o = i % n;
        u = o;
        nbrs = net[u+1] - net[u];
        while(nbrs) {
            cum = malloc(nbrs * sizeof(double));
            l = 0;
            cum[l++] = lw[net[u]];
            for(k = net[u] + 1; k < net[u+1]; k++) {
                cum[l] = cum[l-1] + lw[k];
                l++;
            }
            p = cum[nbrs-1] * drand48();
            l = 0;
            for(k = net[u]; k < net[u+1]; k++) {
                if(p < cum[l++]) {
                    v = net[k];
                    break;
                }
            }
            free(cum);
            nbrs = net[v+1] - net[v];
            c[o + oset] += ld[k];
            u = v;
        }
    }
    /* end parallel for */

    // alloc numpy array
    dim = malloc(sizeof(npy_intp));
    dim[0] = n;
    dst = (PyArrayObject *) PyArray_ZEROS(1, dim, PyArray_DOUBLE, 0);
    free(dim);
    if(!dst) {
        PyErr_SetString(PyExc_MemoryError, "...");
        return NULL;
    }
    d = (double *) dst->data;

    /* reduction of threaded array into numpy */
    for(i = 0; i < m; i++) {
        oset = i * n;
        for(k = 0; k < n; k++)
            d[k] += c[k + oset];
    }
    free(c);
    // average
    for(k = 0; k < n; k++)
        d[k] /= iter;

    return dst;
}

static PyArrayObject *
network(const double *z,
        const unsigned int n,
        const unsigned int *tri,
        const unsigned int m,
        const unsigned int flow) {
    PyArrayObject *net;
    npy_intp *dim;
    unsigned int i, j, k, l;
    unsigned int *lst[n], *e;
    unsigned int u, v, w, vex, wex;

    // alloc list of arrays
    for(i = 0; i < n; i++) {
        lst[i] = malloc(8 * sizeof(unsigned int));
        if(!lst[i]) {
            PyErr_SetString(PyExc_MemoryError, "malloc of list of arrays failed.");
            return NULL;
        }
        lst[i][0] = 8;
        lst[i][1] = 2;
    }

    if(flow) {
        // retrieve downwards pointing links from the triangulation
        l = 0;
        for(i = 0; i < 3*m; i += 3) {
            for(j = 0; j < 3; j++) {
                u = tri[i+j];
                v = tri[i+(j+1)%3];
                w = tri[i+(j+2)%3];
                if(z[v] > z[u] && z[w] > z[u])
                    continue;
                vex = wex = 0;
                for(k = 2; k < lst[u][1]; k++) {
                    if(lst[u][k] == v)
                        vex = 1;
                    if(lst[u][k] == w)
                        wex = 1;
                }
                if(!vex && z[v] <= z[u]) {
                    lst[u][lst[u][1]++] = v;
                    l++;
                }
                if(!wex && z[w] <= z[u]) {
                    lst[u][lst[u][1]++] = w;
                    l++;
                }
                if(lst[u][0] < lst[u][1] + 2) {
                    lst[u][0] = lst[u][1] + 4;
                    lst[u] = realloc(lst[u], lst[u][0] * sizeof(unsigned int));
                    if(!lst[u]) {
                        PyErr_SetString(PyExc_MemoryError, "realloc of array in list failed.");
                        return NULL;
                    }
                }
            }
        }
    } else {
        // retrieve upwards pointing links from the triangulation
        l = 0;
        for(i = 0; i < 3*m; i += 3) {
            for(j = 0; j < 3; j++) {
                u = tri[i+j];
                v = tri[i+(j+1)%3];
                w = tri[i+(j+2)%3];
                if(z[v] <= z[u] && z[w] <= z[u])
                    continue;
                vex = wex = 0;
                for(k = 2; k < lst[u][1]; k++) {
                    if(lst[u][k] == v)
                        vex = 1;
                    if(lst[u][k] == w)
                        wex = 1;
                }
                if(!vex && z[v] > z[u]) {
                    lst[u][lst[u][1]++] = v;
                    l++;
                }
                if(!wex && z[w] > z[u]) {
                    lst[u][lst[u][1]++] = w;
                    l++;
                }
                if(lst[u][0] < lst[u][1] + 2) {
                    lst[u][0] = lst[u][1] + 4;
                    lst[u] = realloc(lst[u], lst[u][0] * sizeof(unsigned int));
                    if(!lst[u]) {
                        PyErr_SetString(PyExc_MemoryError, "realloc of array in list failed.");
                        return NULL;
                    }
                }
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
        e[i] = j;
        for(k = 2; k < lst[i][1]; k++) {
            e[j++] = lst[i][k];
        }
        free(lst[i]);
    }
    e[n] = j;
    return net;
}

static PyArrayObject *
slopes(const double *x,
       const double *y,
       const double *z,
       const unsigned int *net) {
    PyArrayObject *slp;
    npy_intp *dim;
    unsigned int i, j, k, n;
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

static PyArrayObject *
throughput(const unsigned int *net,
           const double *lw,
           const double *nw,
           const unsigned int iter) {
    PyArrayObject *tput;
    npy_intp *dim;
    unsigned int i, k, l, u, v;
    unsigned int n, m, oset, nbrs;
    unsigned long *c, *t;
    double *cum, p;

    if(!initrng())
        return NULL;

    n = net[0] - 1;
#pragma omp parallel
    m = omp_get_num_threads();

    c = calloc(n * m, sizeof(unsigned long));
    if(!c) {
        PyErr_SetString(PyExc_MemoryError, "...");
        return NULL;
    }

#pragma omp parallel for private(i,k,l,u,v,p,cum,oset,nbrs)
    for(i = 0; i < n * iter; i++) {
        oset = n * omp_get_thread_num();
        u = i % n;
        nbrs = net[u+1] - net[u];
        while(nbrs) {
            cum = malloc(nbrs * sizeof(double));
            l = 0;
            cum[l++] = lw[net[u]];
            for(k = net[u] + 1; k < net[u+1]; k++) {
                cum[l] = cum[l-1] + lw[k];
                l++;
            }
            p = cum[nbrs-1] * drand48();
            l = 0;
            for(k = net[u]; k < net[u+1]; k++) {
                if(p < cum[l++]) {
                    v = net[k];
                    break;
                }
            }
            free(cum);
            nbrs = net[v+1] - net[v];
            c[v + oset] += 1;
            u = v;
        }
    }
    /* end parallel for */

    // alloc numpy array
    dim = malloc(sizeof(npy_intp));
    dim[0] = n;
    tput = (PyArrayObject *) PyArray_ZEROS(1, dim, PyArray_ULONG, 0);
    free(dim);
    if(!tput) {
        PyErr_SetString(PyExc_MemoryError, "...");
        return NULL;
    }
    t = (unsigned long *) tput->data;

    /* reduction of threaded array into numpy */
    for(i = 0; i < m; i++) {
        oset = i * n;
        for(k = 0; k < n; k++)
            t[k] += c[k + oset];
    }
    free(c);
    return tput;
}

static PyObject *
DemNets_Betweenness(PyObject *self, PyObject* args) {
    PyObject *netarg, *lwarg, *nwarg;
    PyArrayObject *wlinks, *wnodes, *net, *b;
    unsigned int *e;

    // parse input
    if(!PyArg_ParseTuple(args, "OOO", &netarg, &lwarg, &nwarg))
        return NULL;
    net = (PyArrayObject *) PyArray_ContiguousFromObject(netarg, PyArray_UINT, 1, 1);
    wlinks = (PyArrayObject *) PyArray_ContiguousFromObject(lwarg, PyArray_DOUBLE, 1, 1);
    wnodes = (PyArrayObject *) PyArray_ContiguousFromObject(nwarg, PyArray_DOUBLE, 1, 1);
    if(!net || !wlinks || !wnodes)
        return NULL;

    // check input
    e = (unsigned int *) net->data;
    if(e[e[0] - 1] != net->dimensions[0]) {
        PyErr_SetString(PyExc_IndexError, "corrupted network format.");
        return NULL;
    }
    if(e[e[0] - 1] != wlinks->dimensions[0]) {
        PyErr_SetString(PyExc_IndexError, "link-weight array does not match network.");
        return NULL;
    }
    if(e[0] - 1 != wnodes->dimensions[0]) {
        PyErr_SetString(PyExc_IndexError, "node-weight array does not match network.");
        return NULL;
    }

    // get shortest-path betweenness
    b = betweenness(e, (double *)wlinks->data, (double *)wnodes->data);

    Py_DECREF(net);
    Py_DECREF(wlinks);
    Py_DECREF(wnodes);
    return PyArray_Return(b);
}

static PyObject *
DemNets_LinkLengths(PyObject *self, PyObject* args) {
    PyObject *xarg, *yarg, *zarg, *netarg;
    PyArrayObject *x, *y, *z;
    PyArrayObject *net, *lln;
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

    // get link distances
    lln = linklengths((double *)x->data, (double *)y->data, (double *)z->data, e);

    Py_DECREF(x);
    Py_DECREF(y);
    Py_DECREF(z);
    Py_DECREF(net);
    return PyArray_Return(lln);
}

static PyObject *
DemNets_FlowDistance(PyObject *self, PyObject* args) {
    PyObject *netarg, *lwarg, *ldarg;
    PyArrayObject *wlinks, *dlinks, *net, *d;
    unsigned int *e, iter;

    // parse input
    iter = 100;
    if(!PyArg_ParseTuple(args, "OOO|I", &netarg, &lwarg, &ldarg, &iter))
        return NULL;
    net = (PyArrayObject *) PyArray_ContiguousFromObject(netarg, PyArray_UINT, 1, 1);
    wlinks = (PyArrayObject *) PyArray_ContiguousFromObject(lwarg, PyArray_DOUBLE, 1, 1);
    dlinks = (PyArrayObject *) PyArray_ContiguousFromObject(ldarg, PyArray_DOUBLE, 1, 1);
    if(!net || !wlinks || !dlinks)
        return NULL;

    // check input
    e = (unsigned int *) net->data;
    if(e[e[0] - 1] != net->dimensions[0]) {
        PyErr_SetString(PyExc_IndexError, "corrupted network format.");
        return NULL;
    }
    if(e[e[0] - 1] != wlinks->dimensions[0]) {
        PyErr_SetString(PyExc_IndexError, "link-weight array does not match network.");
        return NULL;
    }
    if(e[e[0] - 1] != dlinks->dimensions[0]) {
        PyErr_SetString(PyExc_IndexError, "link-length array does not match network.");
        return NULL;
    }

    // get node throughput
    d = flowdistance(e, (double *)wlinks->data, (double *)dlinks->data, iter);

    Py_DECREF(net);
    Py_DECREF(wlinks);
    Py_DECREF(dlinks);
    return PyArray_Return(d);
}

static PyObject *
DemNets_FlowNetwork(PyObject *self, PyObject* args) {
    PyObject *elearg, *triarg;
    PyArrayObject *ele, *tri, *net;
    unsigned int flow;

    // parse input
    flow = 1;
    if(!PyArg_ParseTuple(args, "OO|I", &elearg, &triarg, &flow))
        return NULL;
    ele = (PyArrayObject *) PyArray_ContiguousFromObject(elearg, PyArray_DOUBLE, 1, 1);
    tri = (PyArrayObject *) PyArray_ContiguousFromObject(triarg, PyArray_UINT, 2, 2);
    if(!ele || !tri)
        return NULL;

    // retrieve flow network from the triangulation and elevation
    net = network((double *)ele->data, ele->dimensions[0], (unsigned int *)tri->data, tri->dimensions[0], flow);
    Py_DECREF(ele);
    Py_DECREF(tri);
    return PyArray_Return(net);
}

static PyObject *
DemNets_LinkSinks(PyObject *self, PyObject* args) {
    PyObject *elearg, *netarg, *tenarg;
    PyArrayObject *ele, *net, *ten, *new;
    unsigned int *x, n;

    // parse input
    if(!PyArg_ParseTuple(args, "OOO", &netarg, &tenarg, &elearg))
        return NULL;
    net = (PyArrayObject *) PyArray_ContiguousFromObject(netarg, PyArray_UINT, 1, 1);
    ten = (PyArrayObject *) PyArray_ContiguousFromObject(tenarg, PyArray_UINT, 1, 1);
    ele = (PyArrayObject *) PyArray_ContiguousFromObject(elearg, PyArray_DOUBLE, 1, 1);
    if(!ele || !net || !ten)
        return NULL;

    // check input
    n = ele->dimensions[0];
    x = (unsigned int *) net->data;
    if(n != x[0] - 1) {
        PyErr_SetString(PyExc_IndexError, "elevation measures do not match with the flow network.");
        return NULL;
    }
    if(x[n] != net->dimensions[0]) {
        PyErr_SetString(PyExc_IndexError, "corrupted network format for the flow network.");
        return NULL;
    }
    x = (unsigned int *) ten->data;
    if(x[n] != ten->dimensions[0]) {
        PyErr_SetString(PyExc_IndexError, "corrupted network format for the reverse flow network.");
        return NULL;
    }
    new = linksinks((unsigned int *)net->data, (unsigned int *)ten->data, (double *)ele->data);
    Py_DECREF(ele);
    Py_DECREF(net);
    Py_DECREF(ten);
    return PyArray_Return(new);
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

static PyObject *
DemNets_Throughput(PyObject *self, PyObject* args) {
    PyObject *netarg, *lwarg, *nwarg;
    PyArrayObject *wlinks, *wnodes, *net, *t;
    unsigned int *e, iter;

    // parse input
    iter = 100;
    if(!PyArg_ParseTuple(args, "OOO|I", &netarg, &lwarg, &nwarg, &iter))
        return NULL;
    net = (PyArrayObject *) PyArray_ContiguousFromObject(netarg, PyArray_UINT, 1, 1);
    wlinks = (PyArrayObject *) PyArray_ContiguousFromObject(lwarg, PyArray_DOUBLE, 1, 1);
    wnodes = (PyArrayObject *) PyArray_ContiguousFromObject(nwarg, PyArray_DOUBLE, 1, 1);
    if(!net || !wlinks || !wnodes)
        return NULL;

    // check input
    e = (unsigned int *) net->data;
    if(e[e[0] - 1] != net->dimensions[0]) {
        PyErr_SetString(PyExc_IndexError, "corrupted network format.");
        return NULL;
    }
    if(e[e[0] - 1] != wlinks->dimensions[0]) {
        PyErr_SetString(PyExc_IndexError, "link-weight array does not match network.");
        return NULL;
    }
    if(e[0] - 1 != wnodes->dimensions[0]) {
        PyErr_SetString(PyExc_IndexError, "node-weight array does not match network.");
        return NULL;
    }

    // get node throughput
    t = throughput(e, (double *)wlinks->data, (double *)wnodes->data, iter);

    Py_DECREF(net);
    Py_DECREF(wlinks);
    Py_DECREF(wnodes);
    return PyArray_Return(t);
}

static PyMethodDef DemNets_methods[] = {
    {"Betweenness", DemNets_Betweenness, METH_VARARGS, "..."},
    {"FlowNetwork", DemNets_FlowNetwork, METH_VARARGS, "..."},
    {"FlowDistance", DemNets_FlowDistance, METH_VARARGS, "..."},
    {"LinkLengths", DemNets_LinkLengths, METH_VARARGS, "..."},
    {"Slopes", DemNets_Slopes, METH_VARARGS, "..."},
    {"LinkSinks", DemNets_LinkSinks, METH_VARARGS, "..."},
    {"Throughput", DemNets_Throughput, METH_VARARGS, "..."},
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
