#include "Python.h"
#include "numpy/arrayobject.h"
#include <fcntl.h>
#include <math.h>
#include <omp.h>

#define VERSION "0.1"

#define MINELEVATION -999
#define MAXPATHLEN 10000000

typedef struct Stack Stack;
typedef struct Queue Queue;
typedef struct Node Node;

struct Node {
    Node *next;
    unsigned int i;
    double d;
};

struct Queue {
    Node *first;
    Node *last;
};

struct Stack {
    Stack *head;
    unsigned int i;
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
    const double d) {
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
    double *d) {
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
components(const unsigned int *net,
           const unsigned int *seeds, const unsigned int nseeds) {
    PyArrayObject *com;
    npy_intp *dim;
    unsigned int i, j, k, l;
    unsigned int *c, n;
    unsigned int *seen;
	double d;
    Queue *que;

    n = net[0] - 1;

    // alloc numpy array
    dim = malloc(sizeof(npy_intp));
    dim[0] = n;
    com = (PyArrayObject *) PyArray_ZEROS(1, dim, PyArray_UINT, 0);
    free(dim);
    if(!com) {
        PyErr_SetString(PyExc_MemoryError, "...");
        return NULL;
    }
    c = (unsigned int *) com->data;

    for(j = 0; j < nseeds; j++) {
        que = malloc(sizeof(Queue));
        que->first = que->last = NULL;
        seen = calloc(n, sizeof(unsigned int));
        if(!seen) {
            PyErr_SetString(PyExc_MemoryError, "...");
            exit(EXIT_FAILURE);
        }
        // put sink into queue
        if(put(que, seeds[j], 0)) {
            PyErr_SetString(PyExc_MemoryError, "failed to fill queue ..");
            exit(EXIT_FAILURE);
        }
        seen[seeds[j]] = 1;
        while(!get(que, &i, &d)) {
            c[i]++;
            // all neighbors l of i
            for(k = net[i]; k < net[i+1]; k++) {
                l = net[k];
                if(!seen[l]) {
                    seen[l]++;
                    put(que, l, d+1);
                }
            }
        }
        free(seen);
        free(que);
    }
    return com;
}

static PyArrayObject *
sinkdistance(const unsigned int *net,
             const unsigned int sink) {
    PyArrayObject *dist;
    npy_intp *dim;
    unsigned int i, k, l;
    unsigned int *seen, *d;
	double di;
    Queue *que;

    // alloc numpy array
    dim = malloc(sizeof(npy_intp));
    dim[0] = net[0] - 1;
    dist = (PyArrayObject *) PyArray_ZEROS(1, dim, PyArray_UINT, 0);
    free(dim);
    if(!dist) {
        PyErr_SetString(PyExc_MemoryError, "...");
        return NULL;
    }
    d = (unsigned int *) dist->data;

    que = malloc(sizeof(Queue));
    que->first = que->last = NULL;
    seen = calloc(net[0] - 1, sizeof(unsigned int));
    if(!seen) {
        PyErr_SetString(PyExc_MemoryError, "...");
        exit(EXIT_FAILURE);
    }
    // put sink into queue
    if(put(que, sink, 0)) {
        PyErr_SetString(PyExc_MemoryError, "failed to fill queue ..");
        exit(EXIT_FAILURE);
    }
    seen[sink] = 1;
    while(!get(que, &i, &di)) {
        d[i] = di;
        // all neighbors l of i
        for(k = net[i]; k < net[i+1]; k++) {
            l = net[k];
            if(!seen[l]) {
                seen[l]++;
                put(que, l, di+1);
            }
        }
    }
    free(seen);
    free(que);
    return dist;
}

static PyArrayObject *
linklengths(const double *x,
            const double *y,
            //const double *z,
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
    dim[0] = net[n] - n - 1;
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
        //zi = z[i];
        for(k = net[i]; k < net[i+1]; k++) {
            j = net[k];
            dx = xi - x[j];
            dy = yi - y[j];
            //dz = zi - z[j];
            l[k - n - 1] = sqrt(dx*dx + dy*dy);
        }
    }
    return lln;
}

static PyArrayObject *
linkangles(const double *x,
           const double *y,
           const unsigned int *net) {
    PyArrayObject *phi;
    npy_intp *dim;
    unsigned int i, j, k, n;
    double *ph;
    double xi, yi;

    // alloc numpy array
    n = net[0] - 1;
    dim = malloc(sizeof(npy_intp));
    dim[0] = net[n] - n - 1;
    phi = (PyArrayObject *) PyArray_ZEROS(1, dim, PyArray_DOUBLE, 0);
    free(dim);
    if(!phi) {
        PyErr_SetString(PyExc_MemoryError, "...");
        return NULL;
    }
    ph = (double *) phi->data;

    // estimate link i->j angular directions
    for(i = 0; i < n; i++) {
        xi = x[i];
        yi = y[i];
        for(k = net[i]; k < net[i+1]; k++) {
            j = net[k];
            ph[k - n - 1] = atan2(y[j] - yi, x[j] - xi);
        }
    }
    return phi;
}

static PyArrayObject *
voronoi(const unsigned int *net,
        const unsigned int *ten,
        const double *x,
        const double *y) {
    PyArrayObject *area;
    npy_intp *dim;
    unsigned int i, j, k, n;
    double *a;
    double dx, dy;
    double xi, yi;

    // alloc numpy array
    n = net[0] - 1;
    dim = malloc(sizeof(npy_intp));
    dim[0] = n;
    area = (PyArrayObject *) PyArray_ZEROS(1, dim, PyArray_DOUBLE, 0);
    free(dim);
    if(!area) {
        PyErr_SetString(PyExc_MemoryError, "...");
        return NULL;
    }
    a = (double *) area->data;

    for(i = 0; i < n; i++) {
        xi = x[i];
        yi = y[i];
        //zi = z[i];
        for(k = net[i]; k < net[i+1]; k++) {
            j = net[k];
            dx = xi - x[j];
            dy = yi - y[j];
            //dz = zi - z[j];
            //l[k - n - 1] = sqrt(dx*dx + dy*dy);
        }
    }
    return area;
}

static PyObject *
linksinks(const unsigned int *net,
          const unsigned int *ten,
          const double *x,
          const double *y,
          const double *z,
          const double maxdistance) {
    PyArrayObject *tubes, *sinks;
    npy_intp *dim;
    double xj, yj, zj, d, dx, dy, dthr;
    unsigned int i, j, k, l, n, m, q, p;
    unsigned int len, *e, *s;
    unsigned int *seen, *list, link;
    Queue *que;

    n = net[0] - 1;
    m = 0;
    list = NULL;
    len = 0;
    for(j = 0; j < n; j++) {
        if(net[j] == net[j+1]) {
            if(len < m + 2) {
                len += 1024;
                list = realloc(list, len * sizeof(unsigned int));
                if(!list) {
                    PyErr_SetString(PyExc_MemoryError, "...");
                    exit(EXIT_FAILURE);
                }
            }
            list[m++] = j;
            list[m++] = n+1;
        }
    }
    if(!m) {
        PyErr_SetString(PyExc_RuntimeError, "no dead ends in the network?");
        return NULL;
    }
    dthr = maxdistance * maxdistance;

#pragma omp parallel for private(q,j,xj,yj,zj,que,seen,link,i,d,dx,dy,k,l) schedule(guided)
    for(q = 0; q < m; q += 2) {
        j = list[q];
        xj = x[j];
        yj = y[j];
        zj = z[j];
		if(zj < MINELEVATION)
			continue;
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
        seen[j] = 1;
        link = 0;
        while(!get(que, &i, &d) && !link) {
            if(d > 100)
                break;
            // all neighbors l of i
            for(k = net[i]; k < net[i+1]; k++) {
                l = net[k];
                if(seen[l])
                    continue;
                seen[l]++;
				dx = xj - x[l];
                dy = yj - y[l];
                if(z[l] < zj && dx*dx+dy*dy < dthr) {
                    list[q+1] = l;
                    link = 1;
                    break;
                }
                put(que, l, d+1);
            }
            if(link)
                break;
            for(k = ten[i]; k < ten[i+1]; k++) {
                l = ten[k];
                if(seen[l])
                    continue;
                seen[l]++;
				dx = xj - x[l];
                dy = yj - y[l];
				if(z[l] < zj && dx*dx+dy*dy < dthr) {
                    list[q+1] = l;
                    link = 1;
                    break;
                }
                put(que, l, d+1);
            }
        }
        while(!get(que, &i, &d));
        free(seen);
        free(que);
    }

    // store in compressed row format
    l = k = 0;
    j = n + 1;
    e = malloc((n + m/2 + 1) * sizeof(unsigned int));
    s = malloc(m / 2 * sizeof(unsigned int));
    if(!e || !s) {
        PyErr_SetString(PyExc_MemoryError, "...");
        return NULL;
    }
    for(i = 0; i < n; i++) {
        e[i] = j;
        if(i == list[l]) {
            l++;
            if(list[l] < n)
                e[j++] = list[l];
            else
                s[k++] = list[l-1];
            l++;
        }
    }
    e[n] = j;
    free(list);

    // alloc numpy array
    dim = malloc(sizeof(npy_intp));
    dim[0] = j;
    tubes = (PyArrayObject *) PyArray_ZEROS(1, dim, PyArray_UINT, 0);
    if(!tubes) {
        PyErr_SetString(PyExc_MemoryError, "...");
        return NULL;
    }
    memcpy(tubes->data, e, j * sizeof(unsigned int));
    free(e);
    dim[0] = k;
    sinks = (PyArrayObject *) PyArray_ZEROS(1, dim, PyArray_UINT, 0);
    free(dim);
    if(!sinks) {
        PyErr_SetString(PyExc_MemoryError, "...");
        return NULL;
    }
    memcpy(sinks->data, s, k * sizeof(unsigned int));
    free(s);

	return Py_BuildValue("(OO)", tubes, sinks);
}

static PyArrayObject *
flowdistance(const unsigned int *net,
             const double *lw,
             const double *ld,
             const unsigned int iter,
             const unsigned int plen) {
    PyArrayObject *dst;
    npy_intp *dim;
    unsigned int i, k, l, u, v;
    unsigned int n, m, o, oset, nbrs;
    double *cum, p, q;
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

#pragma omp parallel for private(i,k,l,o,u,v,p,q,cum,oset,nbrs)
    for(i = 0; i < n * iter; i++) {
        oset = n * omp_get_thread_num();
        o = i % n;
        u = o;
        q = 0;
        nbrs = net[u+1] - net[u];
        while(nbrs && q < plen) {
            cum = malloc(nbrs * sizeof(double));
            l = 0;
            cum[l++] = lw[net[u] - n - 1] + 0.0001;
            for(k = net[u] + 1; k < net[u+1]; k++) {
                cum[l] = cum[l-1] + lw[k - n - 1] + 0.0001;
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
            q++;
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
            d[k] += (double)c[k + oset] / iter;
    }
    free(c);

    return dst;
}

static PyArrayObject *
Simplicies(const unsigned int *tri,
           const unsigned int m,
           const unsigned int n) {
    PyArrayObject *net;
    npy_intp dim[1];
    unsigned int i, j, k, l, o;
    unsigned int *lst[n], *e, ex;
    // node to simplicies map

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

    l = 0;
    for(i = 0; i < m; i++) {
        for(j = 0; j < 3; j++) {
            k = tri[i*3+j];
            ex = 0;
            for(o = 2; o < lst[k][1]; o++) {
                if(lst[k][o] == i) {
                    ex = 1;
                    break;
                }
            }
            if(!ex) {
                lst[k][lst[k][1]++] = i;
                l++;
            }
            if(lst[k][0] < lst[k][1] + 2) {
                lst[k][0] = lst[k][1] + 4;
                lst[k] = realloc(lst[k], lst[k][0] * sizeof(unsigned int));
                if(!lst[k]) {
                    PyErr_SetString(PyExc_MemoryError, "realloc of array in list failed.");
                    return NULL;
                }
            }
        }
    }

    // alloc numpy array
    dim[0] = l + n + 1;
    net = (PyArrayObject *) PyArray_ZEROS(1, dim, PyArray_UINT, 0);
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
                if(z[v] >= z[u] && z[w] >= z[u])
                    continue;
                vex = wex = 0;
                for(k = 2; k < lst[u][1]; k++) {
                    if(lst[u][k] == v)
                        vex = 1;
                    if(lst[u][k] == w)
                        wex = 1;
                }
                if(!vex && z[v] < z[u]) {
                    lst[u][lst[u][1]++] = v;
                    l++;
                }
                if(!wex && z[w] < z[u]) {
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
                if(z[v] < z[u] && z[w] < z[u])
                    continue;
                vex = wex = 0;
                for(k = 2; k < lst[u][1]; k++) {
                    if(lst[u][k] == v)
                        vex = 1;
                    if(lst[u][k] == w)
                        wex = 1;
                }
                if(!vex && z[v] >= z[u]) {
                    lst[u][lst[u][1]++] = v;
                    l++;
                }
                if(!wex && z[w] >= z[u]) {
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
gridnetwork(const double *z,
            const unsigned int xn,
            const unsigned int yn,
            const unsigned int flow,
            const unsigned int halo) {
    PyArrayObject *net;
    npy_intp *dim;
    unsigned int *gn, x, y;
    unsigned int i, n, m;
    int u, v, w;
    int uu, ud, vl, vr;
    double h;

    n = xn * yn;
    gn = malloc(n * (2*halo+1) * (2*halo+1) * sizeof(unsigned int));
    if(!gn) {
        PyErr_SetString(PyExc_MemoryError, "...");
        return NULL;
    }
    m = n + 1;
    if(flow) {
        for(y = 0; y < yn; y++) {
            if(y < halo)
                uu = -y;
            else
                uu = -halo;
            if(y >= yn - halo)
                ud = yn - y;
            else
                ud = halo + 1;
            for(x = 0; x < xn; x++) {
                if(x < halo)
                    vl = -x;
                else
                    vl = -halo;
                if(x >= xn - halo)
                    vr = xn - x;
                else
                    vr = halo + 1;
                i = y * xn + x;
                gn[i] = m;
                h = z[i];
                if(h < MINELEVATION)
                    continue;
                for(u = uu; u < ud; u++) {
                    for(v = vl; v < vr; v++) {
                        if(sqrt(u*u + v*v) > (double)halo) continue;
                        w = i + u * xn + v;
                        if(MINELEVATION < z[w] && z[w] < h && w != i)
                            gn[m++] = w;
                    }
                }
            }
        }
        gn[n] = m;
    } else {
        i = 0;
        gn[i] = m;
        h = z[i];
        if(h > MINELEVATION) {
            if(z[i + 1] >= h)
                gn[m++] = i + 1;
            if(z[i + xn] >= h)
                gn[m++] = i + xn;
        }
        for(i = 1; i < xn - 1; i++) {
            gn[i] = m;
            h = z[i];
            if(h < MINELEVATION)
                continue;
            if(z[i - 1] >= h)
                gn[m++] = i - 1;
            if(z[i + 1] >= h)
                gn[m++] = i + 1;
            if(z[i + xn] >= h)
                gn[m++] = i + xn;
        }
        i = xn - 1;
        gn[i] = m;
        h = z[i];
        if(h > MINELEVATION) {
            if(z[i - 1] >= h)
                gn[m++] = i - 1;
            if(z[i + xn] >= h)
                gn[m++] = i + xn;
        }
        for(y = 1; y < yn - 1; y++) {
            i = y * xn;
            gn[i] = m;
            h = z[i];
            if(h > MINELEVATION) {
                if(z[i - xn] >= h)
                    gn[m++] = i - xn;
                if(z[i + 1] >= h)
                    gn[m++] = i + 1;
                if(z[i + xn] >= h)
                    gn[m++] = i + xn;
            }
            for(x = 1; x < xn - 1; x++) {
                i = y * xn + x;
                gn[i] = m;
                h = z[i];
                if(h < MINELEVATION)
                    continue;
                if(z[i - xn] >= h)
                    gn[m++] = i - xn;
                if(z[i - 1] >= h)
                    gn[m++] = i - 1;
                if(z[i + 1] >= h)
                    gn[m++] = i + 1;
                if(z[i + xn] >= h)
                    gn[m++] = i + xn;
            }
            i = y * xn + xn - 1;
            gn[i] = m;
            h = z[i];
            if(h > MINELEVATION) {
                if(z[i - xn] >= h)
                    gn[m++] = i - xn;
                if(z[i - 1] >= h)
                    gn[m++] = i - 1;
                if(z[i + xn] >= h)
                    gn[m++] = i + xn;
            }
        }
        i = n - xn;
        gn[i] = m;
        h = z[i];
        if(h > MINELEVATION) {
            if(z[i - xn] >= h)
                gn[m++] = i - xn;
            if(z[i + 1] >= h)
                gn[m++] = i + 1;
        }
        for(i = n - xn + 1; i < n - 1; i++) {
            gn[i] = m;
            h = z[i];
            if(h < MINELEVATION)
                continue;
            if(z[i - xn] >= h)
                gn[m++] = i - xn;
            if(z[i - 1] >= h)
                gn[m++] = i - 1;
            if(z[i + 1] >= h)
                gn[m++] = i + 1;
        }
        i = n - 1;
        gn[i] = m;
        h = z[i];
        if(h > MINELEVATION) {
            if(z[i - xn] >= h)
                gn[m++] = i - xn;
            if(z[i - 1] >= h)
                gn[m++] = i - 1;
        }
        gn[n] = m;
    }
/*
    if(flow) {
        i = 0;
        gn[i] = m;
        h = z[i];
        if(h > MINELEVATION) {
            if(MINELEVATION < z[i + 1] && z[i + 1] < h)
                gn[m++] = i + 1;
            if(MINELEVATION < z[i + xn] && z[i + xn] < h)
                gn[m++] = i + xn;
        }
        for(i = 1; i < xn - 1; i++) {
            gn[i] = m;
            h = z[i];
            if(h < MINELEVATION)
                continue;
            if(MINELEVATION < z[i - 1] && z[i - 1] < h)
                gn[m++] = i - 1;
            if(MINELEVATION < z[i + 1] && z[i + 1] < h)
                gn[m++] = i + 1;
            if(MINELEVATION < z[i + xn] && z[i + xn] < h)
                gn[m++] = i + xn;
        }
        i = xn - 1;
        gn[i] = m;
        h = z[i];
        if(h > MINELEVATION) {
            if(MINELEVATION < z[i - 1] && z[i - 1] < h)
                gn[m++] = i - 1;
            if(MINELEVATION < z[i + xn] && z[i + xn] < h)
                gn[m++] = i + xn;
        }
        for(y = 1; y < yn - 1; y++) {
            i = y * xn;
            gn[i] = m;
            h = z[i];
            if(h > MINELEVATION) {
                if(MINELEVATION < z[i - xn] && z[i - xn] < h)
                    gn[m++] = i - xn;
                if(MINELEVATION < z[i + 1] && z[i + 1] < h)
                    gn[m++] = i + 1;
                if(MINELEVATION < z[i + xn] && z[i + xn] < h)
                    gn[m++] = i + xn;
            }
            for(x = 1; x < xn - 1; x++) {
                i = y * xn + x;
                gn[i] = m;
                h = z[i];
                if(h < MINELEVATION)
                    continue;
                if(MINELEVATION < z[i - xn] && z[i - xn] < h)
                    gn[m++] = i - xn;
                if(MINELEVATION < z[i - 1] && z[i - 1] < h)
                    gn[m++] = i - 1;
                if(MINELEVATION < z[i + 1] && z[i + 1] < h)
                    gn[m++] = i + 1;
                if(MINELEVATION < z[i + xn] && z[i + xn] < h)
                    gn[m++] = i + xn;
            }
            i = y * xn + xn - 1;
            gn[i] = m;
            h = z[i];
            if(h > MINELEVATION) {
                if(MINELEVATION < z[i - xn] && z[i - xn] < h)
                    gn[m++] = i - xn;
                if(MINELEVATION < z[i - 1] && z[i - 1] < h)
                    gn[m++] = i - 1;
                if(MINELEVATION < z[i + xn] && z[i + xn] < h)
                    gn[m++] = i + xn;
            }
        }
        i = n - xn;
        gn[i] = m;
        h = z[i];
        if(h > MINELEVATION) {
            if(MINELEVATION < z[i - xn] && z[i - xn] < h)
                gn[m++] = i - xn;
            if(MINELEVATION < z[i + 1] && z[i + 1] < h)
                gn[m++] = i + 1;
        }
        for(i = n - xn + 1; i < n - 1; i++) {
            gn[i] = m;
            h = z[i];
            if(h < MINELEVATION)
                continue;
            if(MINELEVATION < z[i - xn] && z[i - xn] < h)
                gn[m++] = i - xn;
            if(MINELEVATION < z[i - 1] && z[i - 1] < h)
                gn[m++] = i - 1;
            if(MINELEVATION < z[i + 1] && z[i + 1] < h)
                gn[m++] = i + 1;
        }
        i = n - 1;
        gn[i] = m;
        h = z[i];
        if(h > MINELEVATION) {
            if(MINELEVATION < z[i - xn] && z[i - xn] < h)
                gn[m++] = i - xn;
            if(MINELEVATION < z[i - 1] && z[i - 1] < h)
                gn[m++] = i - 1;
        }
        gn[n] = m;
    } else {
        i = 0;
        gn[i] = m;
        h = z[i];
        if(h > MINELEVATION) {
            if(z[i + 1] >= h)
                gn[m++] = i + 1;
            if(z[i + xn] >= h)
                gn[m++] = i + xn;
        }
        for(i = 1; i < xn - 1; i++) {
            gn[i] = m;
            h = z[i];
            if(h < MINELEVATION)
                continue;
            if(z[i - 1] >= h)
                gn[m++] = i - 1;
            if(z[i + 1] >= h)
                gn[m++] = i + 1;
            if(z[i + xn] >= h)
                gn[m++] = i + xn;
        }
        i = xn - 1;
        gn[i] = m;
        h = z[i];
        if(h > MINELEVATION) {
            if(z[i - 1] >= h)
                gn[m++] = i - 1;
            if(z[i + xn] >= h)
                gn[m++] = i + xn;
        }
        for(y = 1; y < yn - 1; y++) {
            i = y * xn;
            gn[i] = m;
            h = z[i];
            if(h > MINELEVATION) {
                if(z[i - xn] >= h)
                    gn[m++] = i - xn;
                if(z[i + 1] >= h)
                    gn[m++] = i + 1;
                if(z[i + xn] >= h)
                    gn[m++] = i + xn;
            }
            for(x = 1; x < xn - 1; x++) {
                i = y * xn + x;
                gn[i] = m;
                h = z[i];
                if(h < MINELEVATION)
                    continue;
                if(z[i - xn] >= h)
                    gn[m++] = i - xn;
                if(z[i - 1] >= h)
                    gn[m++] = i - 1;
                if(z[i + 1] >= h)
                    gn[m++] = i + 1;
                if(z[i + xn] >= h)
                    gn[m++] = i + xn;
            }
            i = y * xn + xn - 1;
            gn[i] = m;
            h = z[i];
            if(h > MINELEVATION) {
                if(z[i - xn] >= h)
                    gn[m++] = i - xn;
                if(z[i - 1] >= h)
                    gn[m++] = i - 1;
                if(z[i + xn] >= h)
                    gn[m++] = i + xn;
            }
        }
        i = n - xn;
        gn[i] = m;
        h = z[i];
        if(h > MINELEVATION) {
            if(z[i - xn] >= h)
                gn[m++] = i - xn;
            if(z[i + 1] >= h)
                gn[m++] = i + 1;
        }
        for(i = n - xn + 1; i < n - 1; i++) {
            gn[i] = m;
            h = z[i];
            if(h < MINELEVATION)
                continue;
            if(z[i - xn] >= h)
                gn[m++] = i - xn;
            if(z[i - 1] >= h)
                gn[m++] = i - 1;
            if(z[i + 1] >= h)
                gn[m++] = i + 1;
        }
        i = n - 1;
        gn[i] = m;
        h = z[i];
        if(h > MINELEVATION) {
            if(z[i - xn] >= h)
                gn[m++] = i - xn;
            if(z[i - 1] >= h)
                gn[m++] = i - 1;
        }
        gn[n] = m;
    }
*/
    // alloc numpy array
    dim = malloc(sizeof(npy_intp));
    dim[0] = m;
    net = (PyArrayObject *) PyArray_ZEROS(1, dim, PyArray_UINT, 0);
    free(dim);
    if(!net) {
        PyErr_SetString(PyExc_MemoryError, "...");
        return NULL;
    }
    memcpy(net->data, gn, m * sizeof(unsigned int));
    free(gn);

    return net;
}

static PyArrayObject *
fusenetworks(const unsigned int *u,
             const unsigned int *v) {
    PyArrayObject *new;
    npy_intp *dim;
    unsigned int i, j, k, n;
    unsigned int *w;

    n = u[0] - 1;

    // alloc numpy array
    dim = malloc(sizeof(npy_intp));
    dim[0] = u[n] + v[n] - n - 1;
    new = (PyArrayObject *) PyArray_ZEROS(1, dim, PyArray_UINT, 0);
    free(dim);
    if(!new) {
        PyErr_SetString(PyExc_MemoryError, "...");
        return NULL;
    }
    w = (unsigned int *) new->data;

    // store in compressed row format
    j = n + 1;
    for(i = 0; i < n; i++) {
        w[i] = j;
        for(k = u[i]; k < u[i+1]; k++)
            w[j++] = u[k];
        for(k = v[i]; k < v[i+1]; k++)
            w[j++] = v[k];
    }
    w[n] = j;
    return new;
}

static PyArrayObject *
removelinks(const unsigned int *tmp,
            const unsigned int *ten,
            const unsigned int *rmn,
            const unsigned int nrmn) {
    PyArrayObject *new;
    npy_intp *dim;
    unsigned int i, k, m, n, d;
    unsigned int *w, *net;
    Stack *stack, *workr, *p;

    // remove links from nodes
    n = tmp[0] - 1;
    m = tmp[n];
    net = malloc(m * sizeof(unsigned int));
    if(!net) {
        PyErr_SetString(PyExc_MemoryError, "...");
        return NULL;
    }
    memcpy(net, tmp, m * sizeof(unsigned int));
    stack = malloc(sizeof(Stack));
    if(!stack) {
        PyErr_SetString(PyExc_MemoryError, "...");
        return NULL;
    }
    stack->head = NULL;
    // fill stack with nodes of which links have to be removed
    for(i = 0; i < nrmn; i++) {
        workr = malloc(sizeof(Stack));
        workr->i = rmn[i];
        workr->head = stack->head;
        stack->head = workr;
    }
    // empty stack
    for(workr = stack->head; workr->head != NULL;) {
        i = workr->i;
        p = workr;
        workr = workr->head;
        free(p);
        d = net[i+1] - net[i];
        if(d) {
            for(k = net[i]; k < m - d; k++)
                net[k] = net[k+d];
            for(k = i+1; k <= n; k++)
                net[k] -= d;
            m -= d;
        }
    }
    free(stack);

    // alloc numpy array
    dim = malloc(sizeof(npy_intp));
    dim[0] = m;
    new = (PyArrayObject *) PyArray_ZEROS(1, dim, PyArray_UINT, 0);
    free(dim);
    if(!new) {
        PyErr_SetString(PyExc_MemoryError, "...");
        return NULL;
    }
    w = (unsigned int *) new->data;
    for(i = 0; i < m; i++)
        w[i] = net[i];

    return new;
}

static PyArrayObject *
reverselinks(const unsigned int *net) {
    PyArrayObject *new;
    npy_intp *dim;
    unsigned int i, n;
    unsigned int *w;

    n = net[0] - 1;

    // alloc numpy array
    dim = malloc(sizeof(npy_intp));
    dim[0] = n;
    new = (PyArrayObject *) PyArray_ZEROS(1, dim, PyArray_UINT, 0);
    free(dim);
    if(!new) {
        PyErr_SetString(PyExc_MemoryError, "...");
        return NULL;
    }
    w = (unsigned int *) new->data;
    for(i = 0; i < n; i++)
        w[i] = net[i];

    return new;
}

static PyArrayObject *
slopes(const double *x,
       const double *y,
       const double *z,
       const unsigned int *net) {
    PyArrayObject *slp;
    npy_intp *dim;
    unsigned int i, j, k, n;
    double *s;
    double dx, dy;

    // alloc numpy array
    n = net[0] - 1;
    dim = malloc(sizeof(npy_intp));
    dim[0] = net[n] - n - 1;
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
            s[k - n - 1] = (z[i] - z[j]) / sqrt(dx*dx + dy*dy);
        }
    }
    return slp;
}

static PyArrayObject *
rwthroughput(const unsigned int *net,
           const double *lw,
           const unsigned int iter,
           const unsigned int plen) {
    PyArrayObject *tput;
    npy_intp *dim;
    unsigned int i, k, l, u, v, o;
    unsigned int n, m, oset, nbrs;
    unsigned long *c;
    double *cum, *t, p;

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

	p = 0;
	for(i = 0 ; i < n ; i++)
		for(k = net[i]; k < net[i+1]; k++)
			if(lw[k - n - 1] < p)
                p = lw[k - n - 1];
    if(p < 0) {
        PyErr_SetString(PyExc_IndexError, "negative link weights are not allowed");
        return NULL;
    }
    v = 0;
#pragma omp parallel for private(i,k,l,o,u,v,p,cum,oset,nbrs) schedule(guided)
    for(i = 0; i < n * iter; i++) {
        oset = n * omp_get_thread_num();
        u = i % n;
		c[u + oset] += 1;
        o = 0;
        nbrs = net[u+1] - net[u];
        while(nbrs && o < plen) {
            cum = malloc(nbrs * sizeof(double));
            l = 0;
            cum[l++] = lw[net[u] - n - 1] + 0.0001;
            for(k = net[u] + 1; k < net[u+1]; k++) {
                cum[l] = cum[l-1] + lw[k - n - 1] + 0.0001;
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
            o++;
        }
    }
    /* end parallel for */

    // alloc numpy array
    dim = malloc(sizeof(npy_intp));
    dim[0] = n;
    tput = (PyArrayObject *) PyArray_ZEROS(1, dim, PyArray_DOUBLE, 0);
    free(dim);
    if(!tput) {
        PyErr_SetString(PyExc_MemoryError, "...");
        return NULL;
    }
    t = (double *) tput->data;

    /* reduction of threaded array into numpy */
    for(i = 0; i < m; i++) {
        oset = i * n;
        for(k = 0; k < n; k++)
            t[k] += (double)c[k + oset] / iter;
    }
    free(c);
    return tput;
}

void
AggregateLinkThroughput(double *rho, double *phi,
        const unsigned int *net,
        const double *ltput,
        const double *theta) {
    unsigned int i, j, k, n, c, *deg;
    double x, y, rh, ph, d;
    Queue *que;

    n = net[0] - 1;
    que = malloc(sizeof(Queue));
    deg = malloc(n * sizeof(unsigned int));
    if(!que || !deg) {
        PyErr_SetString(PyExc_MemoryError, "failed to allocate queue ..");
        exit(EXIT_FAILURE);
    }
    for(i = 0 ; i < n ; i++)
        deg[i] = net[i+1] - net[i];
    for(j = 0 ; j < n ; j++) {
        que->first = que->last = NULL;
        if(put(que, j, 0)) {
            PyErr_SetString(PyExc_MemoryError, "failed to fill queue ..");
            exit(EXIT_FAILURE);
        }
        c = 0;
        x = y = 0;
        while(!get(que, &i, &d)) {
            if(d > 8)
                break;
            for(k = net[i]; k < net[i+1]; k++) {
                rh = ltput[k - n - 1];
                ph = theta[k - n - 1];
                x += rh * cos(ph);
                y += rh * sin(ph);
                c++;
                if(put(que, net[k], d+1)) {
                    PyErr_SetString(PyExc_MemoryError, "failed to fill queue ..");
                    exit(EXIT_FAILURE);
                }
            }
        }
        x /= c;
        y /= c;
        //x *= deg[j];
        //y *= deg[j];
        rho[j] = sqrt(x*x + y*y);
        phi[j] = atan2(y, x);
    }
    free(que);
    free(deg);
}

unsigned int
SimplexOfNodes(const unsigned int *net,
               const unsigned int a,
               const unsigned int b,
               const unsigned int x,
               const unsigned int m) {
    unsigned int i, k;

    // find the not-x simplex of two nodes a and b
    for(i = net[a]; i < net[a+1]; i++)
        for(k = net[b]; k < net[b+1]; k++)
            if(net[i] == net[k] && net[i] != x)
                return net[i];
    return m;
}

unsigned int
NodeOfSimplicies(const unsigned int *tri,
                 const unsigned int a,
                 const unsigned int b,
                 const double *z) {
    unsigned int i, j, k, p, q;
    double zmin;

    // get the lowest node j of two facets a and b
    p = a*3;
    q = b*3;
    zmin = 1E99;
    j = 0;
    for(i = 0; i < 3; i++)
        for(k = 0; k < 3; k++)
            if(tri[p+i] == tri[q+k]) {
                if(z[tri[p+i]] < zmin) {
                    zmin = z[tri[p+i]];
                    j = tri[p+i];
                }
            }
    return j;
}

double
HeronsTriangle(double a, double b, double c) {
    double d;
    // return the area of a facet

    // ! a >= b >= c
    if(a < b) {
        d = b;
        b = a;
        a = d;
    }
    if(b < c) {
        d = c;
        c = b;
        b = d;
    }
    if(a < b) {
        d = b;
        b = a;
        a = d;
    }
    return sqrt((a+(b+c))*(c-(a-b))*(c+(a-b))*(a+(b-c))) / 4;
}

void
FacetFlowThroughput(double *ltp,
                    const unsigned int *spx,
                    const double *spw,
                    const double *spa,
                    const unsigned int m) {
    double ltpi;
    unsigned int i, j, k;
    unsigned int *seen, *ideg, itr;
    Queue *que;

    // initialize
    seen = calloc(m, sizeof(unsigned int));
    ideg = calloc(m, sizeof(unsigned int));
    que = malloc(sizeof(Queue));
    if(!que || !ideg || !seen) {
        PyErr_SetString(PyExc_MemoryError, "...");
        exit(EXIT_FAILURE);
    }
    que->first = que->last = NULL;

    // get in-degree
    for(i = 0; i < m; i++) {
        itr = i * 2;
        for(j = 0; j < 2; j++) {
            k = itr + j;
            if(m > spx[k])
                ideg[spx[k]]++;
        }
    }
    ltpi = 0;
    // start at facets without in-degree
    for(i = 0; i < m; i++) {
        if(!ideg[i]) {
            itr = i * 2;
            seen[i]++;
            for(j = 0; j < 2; j++) {
                k = itr + j;
                if(m > spx[k]) {
                    ltp[k] = spa[k];
                    if(put(que, spx[k], ltp[k])) {
                        PyErr_SetString(PyExc_MemoryError, "failed to fill queue ..");
                        exit(EXIT_FAILURE);
                    }
                }
            }
        }
    }
    // work the queue
    while(!get(que, &i, &ltpi)) {
        itr = i * 2;
        seen[i]++;
        ltp[itr] += ltpi;
        if(seen[i] == ideg[i]) {
            // we collected all input for node i
            ltpi = ltp[itr];
            ltp[itr] = 0;
            for(j = 0; j < 2; j++) {
                k = itr + j;
                if(m > spx[k]) {
                    // link throughput
                    ltp[k] = ltpi * spw[k] + spa[k];
                    if(put(que, spx[k], ltp[k])) {
                        PyErr_SetString(PyExc_MemoryError, "failed to fill queue ..");
                        exit(EXIT_FAILURE);
                    }
                }
            }
        }
	}
}

void
FacetFlowNetwork(unsigned int *spx, double *spw, double *spa, double *spd, double *phi,
           const unsigned int *net,
           const unsigned int *tri,
           const unsigned int m,
           const double *x,
           const double *y,
           const double *z) {
    int sgn, l;
    double du, dv, dw, a, b, c;
    double xx, yy, slp, frc;
    double dx, dy, dz, dn, s, t;
    double xa, xb, xc, ya, yb, yc;
    double aa, ab, ac, bb, bc;
    double zmin, phii, beta;
    unsigned int i, j, k, o;
    unsigned int u, v, w, q, p;
    unsigned int dest;

    for(i = 0; i < m; i++) {
        // at p, q we store the pos of children
        p = i * 2;
        q = i * 2 + 1;
        for(j = 0; j < 3; j++) {
            u = tri[i*3 + j];
            v = tri[i*3 + (j+1)%3];
            w = tri[i*3 + (j+2)%3];
            // grad (dx,dy) of three point plane
            dz = ((x[w]-x[u])*(y[v]-y[u]) - (y[w]-y[u])*(x[v]-x[u]));
            dy = ((z[w]-z[u])*(x[v]-x[u]) - (x[w]-x[u])*(z[v]-z[u])) / dz;
            dx = ((y[w]-y[u])*(z[v]-z[u]) - (z[w]-z[u])*(y[v]-y[u])) / dz;

            // tri sides vs grad
            xa = x[w] - x[u];
            ya = y[w] - y[u];
            xb = x[v] - x[u];
            yb = y[v] - y[u];
            
            // dot products
            aa = xa*xa + ya*ya;
            ab = xa*xb + ya*yb;
            bb = xb*xb + yb*yb;
            dn = 1. / (aa*bb - ab*ab);
            for(sgn = -1; sgn <= 1; sgn += 2) {
                xc = sgn * dx;
                yc = sgn * dy;
                ac = xa*xc + ya*yc;
                bc = xb*xc + yb*yc;
                s = (bb*ac - ab*bc) * dn;
                t = (aa*bc - ab*ac) * dn;
                if(s >= 0 && t >= 0) {
                    phii = atan2(dy, dx);
                    phi[i] = phii; 
                    if(phii < 0)
                        phii += M_PI;
                    a = sqrt(xa*xa + ya*ya);
                    b = sqrt(xb*xb + yb*yb);
                    if(sgn > 0) {
                        spx[p] = SimplexOfNodes(net, w, v, i, m);
                        spx[q] = m;
                        spw[p] = 1;
                        spw[q] = 0;
                        c = sqrt((x[v]-x[w])*(x[v]-x[w])+(y[v]-y[w])*(y[v]-y[w]));
                        spa[p] = HeronsTriangle(a, b, c);
                        spa[q] = 0;
                        beta = atan2(y[w]-y[v], x[w]-x[v]);
                        if(beta < 0)
                            beta += M_PI;
                        beta -= phii;
                        if(beta > M_PI / 2)
                            beta = M_PI - beta;
                        spd[i] = c * fabs(sin(beta));
                    } else {
                        slp = dy / dx;
                        frc = (y[w] - y[v]) / (x[w] - x[v]);
                        if(dx) {
                            if(x[w] != x[v])
                                xx = (yb + x[u]*slp - x[v]*frc) / (slp - frc);
                            else
                                xx = x[w];
                            yy = (xx - x[u])*slp + y[u];
                        } else {
                            xx = x[u];
                            yy = (xx - x[w])*frc + y[w];
                        }
                        if(isinf(yy)) {
                            fprintf(stderr, "flat triangle %i (u:%.2f v:%.2f w:%.2f)\n", i, z[u], z[v], z[w]);
                            spw[p] = 0.5;
                            spw[q] = 0.5;
                            c = sqrt((x[v]-x[w])*(x[v]-x[w])+(y[v]-y[w])*(y[v]-y[w]));
                            spa[p] = HeronsTriangle(a, b, c) / 2.0;
                            spa[q] = spa[p];
                        } else {
                            du = sqrt((xx-x[u])*(xx-x[u])+(yy-y[u])*(yy-y[u]));
                            dv = sqrt((xx-x[v])*(xx-x[v])+(yy-y[v])*(yy-y[v]));
                            dw = sqrt((xx-x[w])*(xx-x[w])+(yy-y[w])*(yy-y[w]));
                            spw[p] = dv / (dv+dw);
                            spw[q] = dw / (dv+dw);
                            spa[p] = HeronsTriangle(b, dv, du);
                            spa[q] = HeronsTriangle(a, dw, du);
                        }
                        spx[p] = SimplexOfNodes(net, u, v, i, m);
                        spx[q] = SimplexOfNodes(net, u, w, i, m);
                        beta = atan2(yb, xb);
                        if(beta < 0)
                            beta += M_PI;
                        beta -= phii;
                        if(beta > M_PI / 2)
                            beta = M_PI - beta;
                        spd[i] = b * fabs(sin(beta));
                        beta = atan2(ya, xa);
                        if(beta < 0)
                            beta += M_PI;
                        beta -= phii;
                        if(beta > M_PI / 2)
                            beta = M_PI - beta;
                        spd[i] += a * fabs(sin(beta));
                    }
                    j = 3;
                    break;
                }
            }
        }
    }

    // fix lose ends
    for(i = 0; i < m; i++) {
        p = i * 2;
        for(j = 0; j < 2; j++) {
            q = p + j;
            l = spx[q];
            if(l == m)
                continue;
            // check whether two neighboring facets flow into each other
            if(spx[l*2] == i || spx[l*2+1] == i) {
                // get lowest node of these two facets
                u = NodeOfSimplicies(tri, i, l, z);
                zmin = z[u];
                dest = m;
                for(k = net[u]; k < net[u+1]; k++) {
                    if(net[k] == i || net[k] == l)
                        continue;
                    for(o = 0; o < 2; o++) {
                        // find nodes of neighboring facets which are lower
                        if(z[tri[net[k]*2 + o]] < zmin) {
                            zmin = z[tri[net[k]*2 + o]];
                            dest = net[k];
                        }
                    }
                }
                // drain into the lower facet dest (i->dest)
                spx[q] = dest;
                // rewire also the other facet to that lower facet (l->dest)
                for(o = 0; o < 2; o++)
                    if(spx[l*2+o] == i)
                        spx[l*2+o] = dest;
            }
        }
    }
}

void
SpecificNodeThroughput(double *rho, double *phi,
        const unsigned int *net,
        const double *nw,
        const double *lw,
        const double *ld,
        const double *theta) {
    unsigned int i, k;
    unsigned int n;
    unsigned int *seen, *deg;
    double ltp, lws, lds, psum;
    double *xtp, *ytp, degr;
    Queue *que;

    n = net[0] - 1;
    seen = calloc(n, sizeof(unsigned int));
    que = malloc(sizeof(Queue));
    deg = calloc(n, sizeof(unsigned int));
    xtp = calloc(n, sizeof(double));
    ytp = calloc(n, sizeof(double));
    if(!deg || !que || !seen || !xtp || !ytp) {
        PyErr_SetString(PyExc_MemoryError, "...");
        exit(EXIT_FAILURE);
    }
    
	// in-degree deg
    for(i = 0 ; i < n ; i++) {
        for(k = net[i]; k < net[i+1]; k++)
            deg[net[k]]++;
    }
    que->first = que->last = NULL;
    for(i = 0 ; i < n ; i++) {
        // hill tops
        if(!deg[i] && net[i] < net[i+1]) {
            lws = lds = psum = 0;
            for(k = net[i]; k < net[i+1]; k++) {
                lws += lw[k-n-1];
                lds += ld[k-n-1];
                //if(lw[k-n-1] < 1E-8) fprintf(stderr, "lw %i ->%i : %.3e\n", i, net[k], lw[k-n-1]);
                //if(ld[k-n-1] < 1E-8) fprintf(stderr, "ld %i ->%i : %.3e\n", i, net[k], ld[k-n-1]);
            }
            for(k = net[i]; k < net[i+1]; k++)
                psum += lw[k-n-1] * ld[k-n-1] / lws / lds;
            //ltpmax = 0;
            for(k = net[i]; k < net[i+1]; k++) {
                ltp = nw[i] * lw[k-n-1] * ld[k-n-1] / lws / lds / psum;
                //ltp = nw[i] * lw[k-n-1] * ld[k-n-1] / psum;
                //ltp = nw[i] * lw[k-n-1] / lws;
                // put children j = net[k] of hill tops i into queue
                if(put(que, net[k], ltp)) {
                    PyErr_SetString(PyExc_MemoryError, "failed to fill queue ..");
                    exit(EXIT_FAILURE);
                }
                /*if(ltp > ltpmax) {
                    ltpmax = ltp;
                    phi[i] = theta[k-n-1];
                }*/
                xtp[i] += ltp * cos(theta[k-n-1]) / ld[k-n-1];
                ytp[i] += ltp * sin(theta[k-n-1]) / ld[k-n-1];
                xtp[net[k]] += ltp * cos(theta[k-n-1]) / ld[k-n-1];
                ytp[net[k]] += ltp * sin(theta[k-n-1]) / ld[k-n-1];
            }
            //rho[i] = ltpmax;
            degr = deg[i] + net[i+1] - net[i];
            if(degr)
                rho[i] = 2.0 * sqrt(xtp[i]*xtp[i] + ytp[i]*ytp[i]) / degr;
            phi[i] = atan2(ytp[i], xtp[i]);
            seen[i] = 1;
        }
    }
    while(!get(que, &i, &ltp)) {
        rho[i] += ltp;
        if(seen[i] == deg[i] - 1) {
            lws = lds = psum = 0;
            for(k = net[i]; k < net[i+1]; k++) {
                lws += lw[k-n-1];
                lds += ld[k-n-1];
                //if(lw[k-n-1] < 1E-8) fprintf(stderr, "lw %i ->%i : %.3e\n", i, net[k], lw[k-n-1]);
                //if(ld[k-n-1] < 1E-8) fprintf(stderr, "ld %i ->%i : %.3e\n", i, net[k], ld[k-n-1]);
            }
            for(k = net[i]; k < net[i+1]; k++)
                psum += lw[k-n-1] * ld[k-n-1] / lws / lds;
            //ltpmax = 0;
            for(k = net[i]; k < net[i+1]; k++) {
                ltp = (rho[i] + nw[i]) * lw[k-n-1] * ld[k-n-1] / lws / lds / psum;
                //ltp = (rho[i] + nw[i]) * lw[k-n-1] * ld[k-n-1] / psum;
                //ltp = (rho[i] + nw[i]) * lw[k-n-1] / lws;
                if(put(que, net[k], ltp)) {
                    PyErr_SetString(PyExc_MemoryError, "failed to fill queue ..");
                    exit(EXIT_FAILURE);
                }
                /*if(ltp > ltpmax) {
                    ltpmax = ltp;
                    phi[i] = theta[k-n-1];
                }*/
                xtp[i] += ltp * cos(theta[k-n-1]) / ld[k-n-1];
                ytp[i] += ltp * sin(theta[k-n-1]) / ld[k-n-1];
                xtp[net[k]] += ltp * cos(theta[k-n-1]) / ld[k-n-1];
                ytp[net[k]] += ltp * sin(theta[k-n-1]) / ld[k-n-1];
            }
            //rho[i] = ltpmax;
            //rho[i] = cos(M_PI / 2. + atan2(ytp, xtp) - atan2(y[i], x[i])) * sqrt(xtp*xtp + ytp*ytp);
            degr = deg[i] + net[i+1] - net[i];
            if(degr)
                rho[i] = 2.0 * sqrt(xtp[i]*xtp[i] + ytp[i]*ytp[i]) / degr;
            phi[i] = atan2(ytp[i], xtp[i]);
        }
        seen[i]++;
    }
    free(seen);
    free(que);
    free(deg);
}

void
NodeThroughput(double *rho, double *phi,
        const unsigned int *net,
        const double *nw,
        const double *lw,
        const double *ld,
        const double *theta) {
    unsigned int i, k;
    unsigned int u, v;
    unsigned int n;
    unsigned int *seen, *deg;
    double ltp, lws, lds, psum;
    double *xtp, *ytp, degr;
    Queue *que;

    n = net[0] - 1;
    seen = calloc(n, sizeof(unsigned int));
    que = malloc(sizeof(Queue));
    deg = calloc(n, sizeof(unsigned int));
    xtp = calloc(n, sizeof(double));
    ytp = calloc(n, sizeof(double));
    if(!deg || !que || !seen || !xtp || !ytp) {
        PyErr_SetString(PyExc_MemoryError, "...");
        exit(EXIT_FAILURE);
    }
    
	// in-degree deg
    for(i = 0 ; i < n ; i++) {
        for(k = net[i]; k < net[i+1]; k++)
            deg[net[k]]++;
    }
    que->first = que->last = NULL;
    for(i = 0 ; i < n ; i++) {
        // hill tops
        if(!deg[i] && net[i] < net[i+1]) {
            lws = lds = psum = 0;
            for(k = net[i]; k < net[i+1]; k++) {
                u = net[k];
                //for(v = net[u]; v < net[u+1]; v++)
                //    lws += lw[k-n-1] * lw[v-n-1];
                lws += lw[k-n-1];
                lds += ld[k-n-1];
                //if(lw[k-n-1] < 1E-8) fprintf(stderr, "lw %i ->%i : %.3e\n", i, net[k], lw[k-n-1]);
                //if(ld[k-n-1] < 1E-8) fprintf(stderr, "ld %i ->%i : %.3e\n", i, net[k], ld[k-n-1]);
            }
            for(k = net[i]; k < net[i+1]; k++) {
                u = net[k];
                psum += ld[k-n-1] * lw[k-n-1];
                //for(v = net[u]; v < net[u+1]; v++)
                //    psum += lw[k-n-1] * lw[v-n-1] * ld[k-n-1] / lws / lds;
            }
            //ltpmax = 0;
            for(k = net[i]; k < net[i+1]; k++) {
                u = net[k];
                ltp = 0;
                //for(v = net[u]; v < net[u+1]; v++)
                //    ltp += nw[i] * lw[k-n-1] * lw[v-n-1] * ld[k-n-1] / lws / lds / psum;
                ltp = nw[i] * ld[k-n-1] * lw[k-n-1] / psum;
                //ltp = nw[i] * lw[k-n-1] / lws;
                // put children j = net[k] of hill tops i into queue
                if(put(que, net[k], ltp)) {
                    PyErr_SetString(PyExc_MemoryError, "failed to fill queue ..");
                    exit(EXIT_FAILURE);
                }
                /*if(ltp > ltpmax) {
                    ltpmax = ltp;
                    phi[i] = theta[k-n-1];
                }*/
                xtp[i] += ltp * cos(theta[k-n-1]) / ld[k-n-1];
                ytp[i] += ltp * sin(theta[k-n-1]) / ld[k-n-1];
                xtp[net[k]] += ltp * cos(theta[k-n-1]) / ld[k-n-1];
                ytp[net[k]] += ltp * sin(theta[k-n-1]) / ld[k-n-1];
            }
            //rho[i] = ltpmax;
            degr = deg[i] + net[i+1] - net[i];
            if(degr)
                rho[i] = 2.0 * sqrt(xtp[i]*xtp[i] + ytp[i]*ytp[i]) / degr;
            phi[i] = atan2(ytp[i], xtp[i]);
            seen[i] = 1;
        }
    }
    while(!get(que, &i, &ltp)) {
        rho[i] += ltp;
        if(seen[i] == deg[i] - 1) {
            lws = lds = psum = 0;
            for(k = net[i]; k < net[i+1]; k++) {
                u = net[k];
                for(v = net[u]; v < net[u+1]; v++)
                    lws += lw[k-n-1] * lw[v-n-1];
                lds += ld[k-n-1];
                //if(lw[k-n-1] < 1E-8) fprintf(stderr, "lw %i ->%i : %.3e\n", i, net[k], lw[k-n-1]);
                //if(ld[k-n-1] < 1E-8) fprintf(stderr, "ld %i ->%i : %.3e\n", i, net[k], ld[k-n-1]);
            }
            for(k = net[i]; k < net[i+1]; k++) {
                u = net[k];
                psum += ld[k-n-1] * lw[k-n-1];
                //for(v = net[u]; v < net[u+1]; v++)
                //    psum += lw[k-n-1] * lw[v-n-1] * ld[k-n-1] / lws / lds;
            }
            //ltpmax = 0;
            for(k = net[i]; k < net[i+1]; k++) {
                u = net[k];
                ltp = 0;
                //for(v = net[u]; v < net[u+1]; v++)
                //    ltp += (rho[i] + nw[i]) * lw[k-n-1] * lw[v-n-1] * ld[k-n-1] / lws / lds / psum;
                ltp = (rho[i] + nw[i]) * ld[k-n-1] * lw[k-n-1] / psum;
                //ltp = (rho[i] + nw[i]) * lw[k-n-1] / lws;
                if(put(que, net[k], ltp)) {
                    PyErr_SetString(PyExc_MemoryError, "failed to fill queue ..");
                    exit(EXIT_FAILURE);
                }
                /*if(ltp > ltpmax) {
                    ltpmax = ltp;
                    phi[i] = theta[k-n-1];
                }*/
                xtp[i] += ltp * cos(theta[k-n-1]) / ld[k-n-1];
                ytp[i] += ltp * sin(theta[k-n-1]) / ld[k-n-1];
                xtp[net[k]] += ltp * cos(theta[k-n-1]) / ld[k-n-1];
                ytp[net[k]] += ltp * sin(theta[k-n-1]) / ld[k-n-1];
            }
            //rho[i] = ltpmax;
            //rho[i] = cos(M_PI / 2. + atan2(ytp, xtp) - atan2(y[i], x[i])) * sqrt(xtp*xtp + ytp*ytp);
            degr = deg[i] + net[i+1] - net[i];
            if(degr)
                rho[i] = 2.0 * sqrt(xtp[i]*xtp[i] + ytp[i]*ytp[i]) / degr;
            phi[i] = atan2(ytp[i], xtp[i]);
        }
        seen[i]++;
    }
    free(seen);
    free(que);
    free(deg);
}

void
SteepThroughput(double *rho, double *phi,
        const unsigned int *net,
        const double *nw,
        const double *lw,
        const double *ld,
        const double *theta) {
    unsigned int i, k;
    unsigned int u, v;
    unsigned int n;
    unsigned int *seen, *deg;
    double ltp, lws, lds, psum;
    double *xtp, *ytp, degr;
    Queue *que;

    n = net[0] - 1;
    seen = calloc(n, sizeof(unsigned int));
    que = malloc(sizeof(Queue));
    deg = calloc(n, sizeof(unsigned int));
    xtp = calloc(n, sizeof(double));
    ytp = calloc(n, sizeof(double));
    if(!deg || !que || !seen || !xtp || !ytp) {
        PyErr_SetString(PyExc_MemoryError, "...");
        exit(EXIT_FAILURE);
    }
    
	// in-degree deg
    for(i = 0 ; i < n ; i++) {
        for(k = net[i]; k < net[i+1]; k++)
            deg[net[k]]++;
    }
    que->first = que->last = NULL;
    for(i = 0 ; i < n ; i++) {
        // hill tops
        if(!deg[i] && net[i] < net[i+1]) {
            lws = lds = psum = 0;
            for(k = net[i]; k < net[i+1]; k++) {
                if(lw[k-n-1] > lws) {
                    lws = lw[k-n-1];
                    u = k;
                }
            }
            ltp = nw[i];
            // put children j = net[k] of hill tops i into queue
            if(put(que, net[u], nw[i])) {
                PyErr_SetString(PyExc_MemoryError, "failed to fill queue ..");
                exit(EXIT_FAILURE);
            }
            xtp[i] += ltp * cos(theta[k-n-1]) / ld[k-n-1];
            ytp[i] += ltp * sin(theta[k-n-1]) / ld[k-n-1];
            xtp[net[u]] += ltp * cos(theta[k-n-1]) / ld[k-n-1];
            ytp[net[u]] += ltp * sin(theta[k-n-1]) / ld[k-n-1];
            rho[i] = sqrt(xtp[i]*xtp[i] + ytp[i]*ytp[i]);
            phi[i] = atan2(ytp[i], xtp[i]);
            seen[i] = 1;
        }
    }
    while(!get(que, &i, &ltp)) {
        rho[i] += ltp;
        if(seen[i] == deg[i] - 1) {
            lws = lds = psum = 0;
            for(k = net[i]; k < net[i+1]; k++) {
                if(lw[k-n-1] > lws) {
                    lws = lw[k-n-1];
                    u = k;
                }
            }
            ltp = rho[i] + nw[i];
            if(put(que, net[u], ltp)) {
                PyErr_SetString(PyExc_MemoryError, "failed to fill queue ..");
                exit(EXIT_FAILURE);
            }
            xtp[i] += ltp * cos(theta[k-n-1]) / ld[k-n-1];
            ytp[i] += ltp * sin(theta[k-n-1]) / ld[k-n-1];
            xtp[net[k]] += ltp * cos(theta[k-n-1]) / ld[k-n-1];
            ytp[net[k]] += ltp * sin(theta[k-n-1]) / ld[k-n-1];
            rho[i] = sqrt(xtp[i]*xtp[i] + ytp[i]*ytp[i]);
            phi[i] = atan2(ytp[i], xtp[i]);
        }
        seen[i]++;
    }
    free(seen);
    free(que);
    free(deg);
}

static PyArrayObject *
LinkThroughput(const unsigned int *net,
           const double *nw,
           const double *lw,
           const double *ld) {
    npy_intp *dim;
    PyArrayObject *ltput;
    unsigned int i, j, k;
    unsigned int n, *c;
    unsigned int *seen;
    double buf, lws, lds, psum;
    double *t, *ltp;
    Queue *que;

    n = net[0] - 1;
    seen = calloc(n, sizeof(unsigned int));
    t = calloc(n, sizeof(double));
    que = malloc(sizeof(Queue));
    c = calloc(n, sizeof(unsigned int));
    dim = malloc(sizeof(npy_intp));
    dim[0] = net[n] - n - 1;
    ltput = (PyArrayObject *) PyArray_ZEROS(1, dim, PyArray_DOUBLE, 0);
    free(dim);
    ltp = (double *)ltput->data;
    if(!ltput || !c || !que || !t || !seen) {
        PyErr_SetString(PyExc_MemoryError, "...");
        exit(EXIT_FAILURE);
    }
    
	// in-degree c
    for(i = 0 ; i < n ; i++) {
        for(k = net[i]; k < net[i+1]; k++)
            c[net[k]]++;
    }
    que->first = que->last = NULL;
    for(i = 0 ; i < n ; i++) {
        // hill tops
        if(!c[i] && net[i] < net[i+1]) {
            lws = lds = psum = 0;
            for(k = net[i]; k < net[i+1]; k++) {
                j = k - n - 1;
                lws += lw[j];
                lds += ld[j];
            }
            for(k = net[i]; k < net[i+1]; k++) {
                j = k - n - 1;
                psum += lw[j] * ld[j] / lws / lds;
            }
            for(k = net[i]; k < net[i+1]; k++) {
                j = k - n - 1;
                ltp[j] = nw[i] * lw[j] * ld[j] / lws / lds / psum;
                //ltp[j] = nw[i] * lw[j] * ld[j] / psum;
                //ltp[j] = nw[i] * lw[j] / lws;
                // put children j = net[k] of hill tops i into queue
                if(put(que, net[k], ltp[j])) {
                    PyErr_SetString(PyExc_MemoryError, "failed to fill queue ..");
                    exit(EXIT_FAILURE);
                }
                ltp[j] /= ld[j];
            }
            seen[i] = 1;
        }
    }
    while(!get(que, &i, &buf)) {
        t[i] += buf;
        if(seen[i] == c[i] - 1) {
            lws = lds = psum = 0;
            for(k = net[i]; k < net[i+1]; k++) {
                j = k - n - 1;
                lws += lw[j];
                lds += ld[j];
            }
            for(k = net[i]; k < net[i+1]; k++) {
                j = k - n - 1;
                psum += lw[j] * ld[j] / lws / lds;
            }
            for(k = net[i]; k < net[i+1]; k++) {
                j = k - n - 1;
                ltp[j] = (t[i] + nw[i]) * lw[j] * ld[j] / lws / lds / psum;
                //ltp[j] = (t[i] + nw[i]) * lw[j] * ld[j] / psum;
                //ltp[j] = (t[i] + nw[i]) * lw[j] / lws;
                if(put(que, net[k], ltp[j])) {
                    PyErr_SetString(PyExc_MemoryError, "failed to fill queue ..");
                    exit(EXIT_FAILURE);
                }
                ltp[j] /= ld[j];
            }
        }
        seen[i]++;
    }
    free(seen);
    free(t);
    free(que);
    free(c);
    return ltput;
}

static PyArrayObject *
pathlengths(const unsigned int *net,
           const double *lw,
           const unsigned int iter,
           const unsigned int plen) {
    PyArrayObject *lens;
    npy_intp *dim;
    unsigned int i, k, l, u, v, o;
    unsigned int n, m, oset, nbrs;
    unsigned long *c;
    double *cum, *pl, p;

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

#pragma omp parallel for private(i,k,l,o,u,v,p,cum,oset,nbrs) schedule(guided)
    for(i = 0; i < n * iter; i++) {
        oset = n * omp_get_thread_num();
        u = i % n;
        v = u;
        o = 0;
        nbrs = net[u+1] - net[u];
        while(nbrs && o < plen) {
            c[v + oset] += 1;
            cum = malloc(nbrs * sizeof(double));
            l = 0;
            cum[l++] = lw[net[u] - n - 1] + 0.0001;
            for(k = net[u] + 1; k < net[u+1]; k++) {
                cum[l] = cum[l-1] + lw[k - n - 1] + 0.0001;
                l++;
            }
            p = cum[nbrs-1] * drand48();
            l = 0;
            for(k = net[u]; k < net[u+1]; k++) {
                if(p < cum[l++]) {
                    u = net[k];
                    break;
                }
            }
            free(cum);
            nbrs = net[u+1] - net[u];
            o++;
        }
    }
    /* end parallel for */

    // alloc numpy array
    dim = malloc(sizeof(npy_intp));
    dim[0] = n;
    lens = (PyArrayObject *) PyArray_ZEROS(1, dim, PyArray_DOUBLE, 0);
    free(dim);
    if(!lens) {
        PyErr_SetString(PyExc_MemoryError, "...");
        return NULL;
    }
    pl = (double *) lens->data;

    /* reduction of threaded array into numpy */
    for(i = 0; i < m; i++) {
        oset = i * n;
        for(k = 0; k < n; k++)
            pl[k] += (double)c[k + oset] / iter;
    }
    free(c);
    return lens;
}

static PyArrayObject *
revisits(const unsigned int *net,
           const double *lw,
           const unsigned int iter,
           const unsigned int plen) {
    PyArrayObject *revisits;
    npy_intp *dim;
    unsigned int i, k, l, u, o;
    unsigned int n, m, oset, nbrs;
    unsigned int *c, *r;
    double *cum, p;

    if(!initrng())
        return NULL;

    n = net[0] - 1;
    
    // alloc numpy array
    dim = malloc(sizeof(npy_intp));
    dim[0] = n;
    revisits = (PyArrayObject *) PyArray_ZEROS(1, dim, PyArray_UINT, 0);
    free(dim);
    if(!revisits) {
        PyErr_SetString(PyExc_MemoryError, "...");
        return NULL;
    }
    r = (unsigned int *) revisits->data;

#pragma omp parallel for private(i,k,l,o,u,p,cum,oset,nbrs,c) schedule(guided)
    for(i = 0; i < n * iter; i++) {
        c = calloc(n, sizeof(unsigned int));
        if(!c) {
            PyErr_SetString(PyExc_MemoryError, "...");
            exit(EXIT_FAILURE);
        }
        oset = n * omp_get_thread_num();
        u = i % n;
        o = 0;
        nbrs = net[u+1] - net[u];
        while(nbrs && o < plen) {
            c[u]++;
            cum = malloc(nbrs * sizeof(double));
            l = 0;
            cum[l++] = lw[net[u] - n - 1] + 0.0001;
            for(k = net[u] + 1; k < net[u+1]; k++) {
                cum[l] = cum[l-1] + lw[k - n - 1] + 0.0001;
                l++;
            }
            p = cum[nbrs-1] * drand48();
            l = 0;
            for(k = net[u]; k < net[u+1]; k++) {
                if(p < cum[l++]) {
                    u = net[k];
                    break;
                }
            }
            free(cum);
            nbrs = net[u+1] - net[u];
            o++;
        }
        for(k = 0; k < n; k++) {
            if(c[k] > 1) {
#pragma omp critical
                r[k] += c[k];
            }
        }
        free(c);
    }
    /* end parallel for */

    return revisits;
}

static PyObject *
derivatives(const unsigned int *net,
           const double *lw,
           const double *z,
           const unsigned int iter) {
    PyArrayObject *velo, *accl;
    npy_intp *dim;
    unsigned int *x, i, j, l, k;
    unsigned int n, m, oset, nbrs;
    unsigned long *c;
    double *v, *a, *ve, *ac;
    double *cum, p;

    if(!initrng())
        return NULL;

    n = net[0] - 1;
#pragma omp parallel
    m = omp_get_num_threads();

    c = calloc(n * m, sizeof(unsigned long));
    v = calloc(n * m, sizeof(double));
    a = calloc(n * m, sizeof(double));
    if(!c || !v || !a) {
        PyErr_SetString(PyExc_MemoryError, "...");
        return NULL;
    }

#pragma omp parallel for private(i,j,k,l,x,nbrs,cum,p,oset) schedule(guided)
    for(i = 0; i < n * iter; i++) {
        x = malloc(6 * sizeof(unsigned int));
        j = 0;
        x[j] = i % n;
        nbrs = net[x[j]+1] - net[x[j]];
        while(nbrs > 0 && z[x[j]] > 0 && j < 5) {
            cum = malloc(nbrs * sizeof(double));
            l = 0;
            cum[l++] = lw[net[x[j]] - n - 1] + 0.0001;
            for(k = net[x[j]] + 1; k < net[x[j]+1]; k++) {
                cum[l] = cum[l-1] + lw[k - n - 1] + 0.0001;
                l++;
            }
            p = cum[nbrs-1] * drand48();
            l = 0;
            for(k = net[x[j]]; k < net[x[j]+1]; k++) {
                if(p < cum[l++]) {
                    x[++j] = net[k];
                    break;
                }
            }
            free(cum);
            nbrs = net[x[j]+1] - net[x[j]];
        }
        if(j == 5 && z[x[4]] > 0) {
            oset = n * omp_get_thread_num();
            c[x[2]+oset] += 1;
            // three point derivatives
            //v[x[2]+oset] += z[x[3]] - z[x[1]];
            //a[x[2]+oset] += z[x[3]] - 2*z[x[2]] + z[x[1]];
            // five point derivatives
            v[x[2]+oset] += z[x[0]]-8*z[x[1]]+8*z[x[3]]-z[x[4]];
            a[x[2]+oset] += -z[x[0]]+16*z[x[1]]-30*z[x[2]]+16*z[x[3]]-z[x[4]];
        }
        free(x);
    }

    // alloc numpy array
    dim = malloc(sizeof(npy_intp));
    dim[0] = n;
    velo = (PyArrayObject *) PyArray_ZEROS(1, dim, PyArray_DOUBLE, 0);
    accl = (PyArrayObject *) PyArray_ZEROS(1, dim, PyArray_DOUBLE, 0);
    free(dim);
    if(!velo || !accl) {
        PyErr_SetString(PyExc_MemoryError, "...");
        return NULL;
    }
    ve = (double *) velo->data;
    ac = (double *) accl->data;

    // reduction of threaded array into numpy
    for(k = 0; k < n; k++) {
        ve[k] = v[k];
        ac[k] = a[k];
    }
    for(i = 1; i < m; i++) {
        oset = i * n;
        for(k = 0; k < n; k++) {
            c[k] += c[k + oset];
            ve[k] += v[k + oset];
            ac[k] += a[k + oset];
        }
    }
    free(v);
    free(a);
    // average over all paths
    for(k = 0; k < n; k++) {
        if(c[k]) {
            ve[k] /= c[k] * 12.0;
            ac[k] /= c[k] * 12.0;
        } else {
            ve[k] = 0;
            ac[k] = 0;
        }
    }
    free(c);
	return Py_BuildValue("(OO)", velo, accl);
}

static PyObject *
walk(const unsigned int *net,
     const double *lw,
     const int start,
     const unsigned int plen) {
    PyArrayObject *trace, *tracei;
    npy_intp *dim;
    unsigned int i, k, l, o, u, v;
    unsigned int n, nbrs;
    unsigned int *t, *ti, tlen;
    double *cum, p;

    if(!initrng())
        return NULL;

    // alloc numpy array
    n = net[0] - 1;
    dim = malloc(sizeof(npy_intp));
    dim[0] = n;
    trace = (PyArrayObject *) PyArray_ZEROS(1, dim, PyArray_UINT, 0);
    free(dim);
    tlen = 1024;
    ti = malloc(tlen * sizeof(unsigned int));
    if(!trace || !ti) {
        PyErr_SetString(PyExc_MemoryError, "trace");
        return NULL;
    }
    t = (unsigned int *) trace->data;

    // start random walk at node start
    if(start < 0)
        u = (unsigned int)(n * drand48());
    else
        u = start;
    nbrs = net[u+1] - net[u];
    i = 0;
    ti[i++] = u;
    o = 0;
    while(nbrs && o < plen) {
        cum = malloc(nbrs * sizeof(double));
        l = 0;
        cum[l++] = lw[net[u] - n - 1] + 0.0001;
        for(k = net[u] + 1; k < net[u+1]; k++) {
            cum[l] = cum[l-1] + lw[k - n - 1] + 0.0001;
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
        t[v]++;
        ti[i++] = v;
        if(i >= tlen) {
            tlen += 1024;
            ti = realloc(ti, tlen * sizeof(unsigned int));
            if(!ti) {
                PyErr_SetString(PyExc_MemoryError, "trace to long");
                return NULL;
            }
        }
        u = v;
        o++;
    }
    tlen = i;
    ti = realloc(ti, tlen * sizeof(unsigned int));

    // alloc numpy array
    dim = malloc(sizeof(npy_intp));
    dim[0] = tlen;
    tracei = (PyArrayObject *) PyArray_ZEROS(1, dim, PyArray_UINT, 0);
    free(dim);
    if(!tracei) {
        PyErr_SetString(PyExc_MemoryError, "...");
        return NULL;
    }
    memcpy(tracei->data, ti, tlen * sizeof(unsigned int));
    free(ti);
    return Py_BuildValue("(OO)", trace, tracei);
}

static PyArrayObject *
walks(const unsigned int *net,
      const double *lw,
      const unsigned int dst,
      const unsigned int *src,
      const unsigned int nsrc,
      const unsigned int iter,
      const unsigned int plen) {
    PyArrayObject *traces;
    npy_intp *dim;
    unsigned int i, j, k, l, o, u, v;
    unsigned int h, hh, n, nbrs;
    unsigned int *t, *tail, *head;
    unsigned int tlen, tllen;
    double *cum, p;

    if(!initrng())
        return NULL;

    tllen = 2048;
    n = iter * nsrc;
    head = malloc((1+n) * sizeof(unsigned int));
    tail = malloc(tllen * sizeof(unsigned int));
    if(!head || !tail) {
        PyErr_SetString(PyExc_MemoryError, "...");
        return NULL;
    }
    h = 0;
    head[h] = 0;
#pragma omp parallel for private(i,j,k,l,o,p,t,u,v,tlen,nbrs,cum) schedule(guided)
    for(i = 0; i < iter * nsrc; i++) {
        tlen = 1024;
        t = malloc(tlen * sizeof(unsigned int));
        if(!t) {
            PyErr_SetString(PyExc_MemoryError, "cannot start a new trace");
            exit(EXIT_FAILURE);
        }
        u = src[i % nsrc];
        j = 0;
        t[j++] = u;
        nbrs = net[u+1] - net[u];
        o = 0;
        while(nbrs && o < plen) {
            cum = malloc(nbrs * sizeof(double));
            l = 0;
            cum[l++] = lw[net[u] - n - 1] + 0.0001;
            for(k = net[u] + 1; k < net[u+1]; k++) {
                cum[l] = cum[l-1] + lw[k - n - 1] + 0.0001;
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
            t[j++] = v;
            if(j >= tlen) {
                tlen += 1024;
                t = realloc(t, tlen * sizeof(unsigned int));
                if(!t) {
                    PyErr_SetString(PyExc_MemoryError, "trace to long");
                    exit(EXIT_FAILURE);
                }
            }
            if(v == dst) {
                tlen = j;
#pragma omp critical
{
                hh = head[h++];
                head[h] = hh + tlen;
                if(tllen <= head[h]) {
                    tllen = head[h] + tlen;
                    tail = realloc(tail, tllen * sizeof(unsigned int));
                    if(!tail) {
                        PyErr_SetString(PyExc_MemoryError, "cannot store more traces");
                        exit(EXIT_FAILURE);
                    }
                }
                for(j = 0; j < tlen; j++)
                    tail[j + hh] = t[j];
}
                free(t);
                break;
            }
            u = v;
            o++;
        }
    }

    // alloc numpy array
    dim = malloc(sizeof(npy_intp));
    dim[0] = 1 + h + head[h];
    traces = (PyArrayObject *) PyArray_ZEROS(1, dim, PyArray_UINT, 0);
    free(dim);
    if(!traces) {
        PyErr_SetString(PyExc_MemoryError, "not enough memory for the numpy array");
        return NULL;
    }
    t = (unsigned int *)traces->data;
    for(i = 0; i <= h; i++)
        t[i] = head[i] + h + 1;
    for(i = 0; i < head[h]; i++)
        t[i+h+1] = tail[i];
    free(head);
    free(tail);

    return traces;
}

static PyArrayObject *
hillslopelengths(const unsigned int *net,
      const double *lw,
      const double *z,
      const double zthr,
      const unsigned int *src,
      const unsigned int nsrc,
      const unsigned int iter,
      const unsigned int plen) {
    PyArrayObject *lengths;
    npy_intp *dim;
    unsigned long *c;
    unsigned int i, x, o, l, k, m, n, oset, nbrs;
    double *h, *cum, p;

    if(!initrng())
        return NULL;

    n = net[0] - 1;
//#pragma omp parallel
    m = omp_get_num_threads();

    c = calloc(n * m, sizeof(unsigned long));
    if(!c) {
        PyErr_SetString(PyExc_MemoryError, "...");
        return NULL;
    }

//#pragma omp parallel for private(i,x,o,l,k,nbrs,cum,p) schedule(guided)
    for(i = 0; i < iter * nsrc; i++) {
        x = src[i % nsrc];
        o = x + n * omp_get_thread_num();
        nbrs = net[x+1] - net[x];
        while(nbrs && z[x] <= zthr && c[o] < plen) {
            c[o]++;
            cum = malloc(nbrs * sizeof(double));
            l = 0;
            cum[l++] = lw[net[x] - n - 1] + 0.0001;
            for(k = net[x] + 1; k < net[x+1]; k++) {
                cum[l] = cum[l-1] + lw[k - n - 1] + 0.0001;
                l++;
            }
            p = cum[nbrs-1] * drand48();
            l = 0;
            for(k = net[x]; k < net[x+1]; k++) {
                if(p < cum[l++]) {
                    x = net[k];
                    break;
                }
            }
            free(cum);
            nbrs = net[x+1] - net[x];
        }
    }

    // alloc numpy output array
    dim = malloc(sizeof(npy_intp));
    dim[0] = net[0] - 1;
    lengths = (PyArrayObject *) PyArray_ZEROS(1, dim, PyArray_DOUBLE, 0);
    free(dim);
    if(!lengths) {
        PyErr_SetString(PyExc_MemoryError, "not enough memory for the numpy array");
        return NULL;
    }
    h = (double *)lengths->data;
    for(i = 0; i < m; i++) {
        oset = i * n;
        for(k = 0; k < n; k++)
            h[k] += c[k + oset] / (double)iter;
    }
    return lengths;
}

static PyObject *
DemNets_Components(PyObject *self, PyObject* args) {
    PyObject *netarg, *seedsarg;
    PyArrayObject *net, *com, *seeds;
    unsigned int *x;

    // parse input
    if(!PyArg_ParseTuple(args, "OO", &netarg, &seedsarg))
        return NULL;
    net = (PyArrayObject *) PyArray_ContiguousFromObject(netarg, PyArray_UINT, 1, 1);
    seeds = (PyArrayObject *) PyArray_ContiguousFromObject(seedsarg, PyArray_UINT, 1, 1);
    if(!net || !seeds)
        return NULL;

    // check input
    x = (unsigned int *) net->data;
    if(x[x[0] - 1] != net->dimensions[0]) {
        PyErr_SetString(PyExc_IndexError, "corrupted network format.");
        return NULL;
    }

    // get components
    com = components(x, (unsigned int *)seeds->data, seeds->dimensions[0]);
    Py_DECREF(net);
    Py_DECREF(seeds);
    return PyArray_Return(com);
}

static PyObject *
DemNets_LinkLengths(PyObject *self, PyObject* args) {
    PyObject *xarg, *yarg, *zarg, *netarg;
    PyArrayObject *x, *y, *z;
    PyArrayObject *net, *lln;
    unsigned int *e, n;

    // parse input
    if(!PyArg_ParseTuple(args, "OOO", &xarg, &yarg, &netarg))
        return NULL;
    x = (PyArrayObject *) PyArray_ContiguousFromObject(xarg, PyArray_DOUBLE, 1, 1);
    y = (PyArrayObject *) PyArray_ContiguousFromObject(yarg, PyArray_DOUBLE, 1, 1);
    //z = (PyArrayObject *) PyArray_ContiguousFromObject(zarg, PyArray_DOUBLE, 1, 1);
    net = (PyArrayObject *) PyArray_ContiguousFromObject(netarg, PyArray_UINT, 1, 1);
    if(!x || !y || !net)
        return NULL;

    // check input
    n = x->dimensions[0];
    if(n != y->dimensions[0]) {
        PyErr_SetString(PyExc_IndexError, "(x, y) not of the same dimension.");
        return NULL;
    }
    //if(n != z->dimensions[0]) {
    //    PyErr_SetString(PyExc_IndexError, "(x, y, z) not of the same dimension.");
    //    return NULL;
    //}
    e = (unsigned int *) net->data;
    if(n != e[0] - 1) {
        PyErr_SetString(PyExc_IndexError, "(x, y) does not match with the network.");
        return NULL;
    }
    if(e[n] != net->dimensions[0]) {
        PyErr_SetString(PyExc_IndexError, "corrupted network format.");
        return NULL;
    }

    // get link distances
    lln = linklengths((double *)x->data, (double *)y->data, e);

    Py_DECREF(x);
    Py_DECREF(y);
    //Py_DECREF(z);
    Py_DECREF(net);
    return PyArray_Return(lln);
}

static PyObject *
DemNets_LinkAngles(PyObject *self, PyObject* args) {
    PyObject *xarg, *yarg, *netarg;
    PyArrayObject *x, *y;
    PyArrayObject *net, *phi;
    unsigned int *e, n;

    // parse input
    if(!PyArg_ParseTuple(args, "OOO", &xarg, &yarg, &netarg))
        return NULL;
    x = (PyArrayObject *) PyArray_ContiguousFromObject(xarg, PyArray_DOUBLE, 1, 1);
    y = (PyArrayObject *) PyArray_ContiguousFromObject(yarg, PyArray_DOUBLE, 1, 1);
    net = (PyArrayObject *) PyArray_ContiguousFromObject(netarg, PyArray_UINT, 1, 1);
    if(!x || !y || !net)
        return NULL;

    // check input
    n = x->dimensions[0];
    if(n != y->dimensions[0]) {
        PyErr_SetString(PyExc_IndexError, "(x, y) not of the same dimension.");
        return NULL;
    }
    e = (unsigned int *) net->data;
    if(n != e[0] - 1) {
        PyErr_SetString(PyExc_IndexError, "(x, y) does not match with the network.");
        return NULL;
    }
    if(e[n] != net->dimensions[0]) {
        PyErr_SetString(PyExc_IndexError, "corrupted network format.");
        return NULL;
    }

    // get link direction angles phi = atan2(y, x)
    phi = linkangles((double *)x->data, (double *)y->data, e);

    Py_DECREF(x);
    Py_DECREF(y);
    Py_DECREF(net);
    return PyArray_Return(phi);
}

static PyObject *
DemNets_FuseNetworks(PyObject *self, PyObject* args) {
    PyObject *a, *b;
    PyArrayObject *u, *v, *w;
    unsigned int *x, *y;

    // parse input
    if(!PyArg_ParseTuple(args, "OO", &a, &b))
        return NULL;
    u = (PyArrayObject *) PyArray_ContiguousFromObject(a, PyArray_UINT, 1, 1);
    v = (PyArrayObject *) PyArray_ContiguousFromObject(b, PyArray_UINT, 1, 1);
    if(!u || !v)
        return NULL;

    // check input
    x = (unsigned int *) u->data;
    y = (unsigned int *) v->data;
    if(x[0] != y[0]) {
        PyErr_SetString(PyExc_IndexError, "networks do not match.");
        return NULL;
    }
    if(x[x[0]-1] != u->dimensions[0]) {
        PyErr_SetString(PyExc_IndexError, "corrupted format for the first network.");
        return NULL;
    }
    if(y[y[0]-1] != v->dimensions[0]) {
        PyErr_SetString(PyExc_IndexError, "corrupted format for the second network.");
        return NULL;
    }
    w = fusenetworks(x, y);
    Py_DECREF(u);
    Py_DECREF(v);
    return PyArray_Return(w);
}

static PyObject *
DemNets_FlowDistance(PyObject *self, PyObject* args) {
    PyObject *netarg, *lwarg, *ldarg;
    PyArrayObject *wlinks, *dlinks, *net, *d;
    unsigned int *e, iter, plen;

    // parse input
    iter = 100;
    plen = MAXPATHLEN;
    if(!PyArg_ParseTuple(args, "OOO|II", &netarg, &lwarg, &ldarg, &iter, &plen))
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

    // get nodes average flow distance (path lengths from source to current node)
    d = flowdistance(e, (double *)wlinks->data, (double *)dlinks->data, iter, plen);

    Py_DECREF(net);
    Py_DECREF(wlinks);
    Py_DECREF(dlinks);
    return PyArray_Return(d);
}

static PyObject *
DemNets_Simplicies(PyObject *self, PyObject* args) {
    PyObject *triarg;
    PyArrayObject *tri, *net;
    unsigned int n;

    // parse input
    if(!PyArg_ParseTuple(args, "OI", &triarg, &n))
        return NULL;
    tri = (PyArrayObject *) PyArray_ContiguousFromObject(triarg, PyArray_UINT, 2, 2);
    if(!tri)
        return NULL;

    // get simplicies for a given node
    net = Simplicies((unsigned int *)tri->data, tri->dimensions[0], n);
    Py_DECREF(tri);
    return PyArray_Return(net);
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
DemNets_FlowNetworkFromGrid(PyObject *self, PyObject* args) {
    PyObject *elearg;
    PyArrayObject *ele, *net;
    unsigned int flow, halo;

    // parse input
    flow = halo = 1;
    if(!PyArg_ParseTuple(args, "O|II", &elearg, &flow, &halo))
        return NULL;
    ele = (PyArrayObject *) PyArray_ContiguousFromObject(elearg, PyArray_DOUBLE, 2, 2);
    if(!ele)
        return NULL;

    // retrieve flow network from elevation grid
    net = gridnetwork((double *)ele->data, ele->dimensions[1], ele->dimensions[0], flow, halo);
    Py_DECREF(ele);
    return PyArray_Return(net);
}

static PyObject *
DemNets_LinkSinks(PyObject *self, PyObject* args) {
    PyObject *xarg, *yarg, *zarg, *netarg, *tenarg, *obj;
    PyArrayObject *x, *y, *z, *net, *ten;
    unsigned int *e, n;
    double maxdistance;

    maxdistance = 50;

    // parse input
    if(!PyArg_ParseTuple(args, "OOOOO|d", &netarg, &tenarg, &xarg, &yarg, &zarg, &maxdistance))
        return NULL;
    net = (PyArrayObject *) PyArray_ContiguousFromObject(netarg, PyArray_UINT, 1, 1);
    ten = (PyArrayObject *) PyArray_ContiguousFromObject(tenarg, PyArray_UINT, 1, 1);
    x = (PyArrayObject *) PyArray_ContiguousFromObject(xarg, PyArray_DOUBLE, 1, 1);
    y = (PyArrayObject *) PyArray_ContiguousFromObject(yarg, PyArray_DOUBLE, 1, 1);
    z = (PyArrayObject *) PyArray_ContiguousFromObject(zarg, PyArray_DOUBLE, 1, 1);
    if(!net || !ten || !x || !y || !z)
        return NULL;

    // check input
    n = x->dimensions[0];
    e = (unsigned int *) net->data;
    if(n != e[0] - 1) {
        PyErr_SetString(PyExc_IndexError, "elevation measures do not match with the flow network.");
        return NULL;
    }
    if(e[n] != net->dimensions[0]) {
        PyErr_SetString(PyExc_IndexError, "corrupted network format for the flow network.");
        return NULL;
    }
    e = (unsigned int *) ten->data;
    if(e[n] != ten->dimensions[0]) {
        PyErr_SetString(PyExc_IndexError, "corrupted network format for the reverse flow network.");
        return NULL;
    }
    obj = linksinks((unsigned int *)net->data, (unsigned int *)ten->data, (double *)x->data, (double *)y->data, (double *)z->data, maxdistance);
    Py_DECREF(x);
    Py_DECREF(y);
    Py_DECREF(z);
    Py_DECREF(net);
    Py_DECREF(ten);
    return obj;
}

static PyObject *
DemNets_VoronoiArea(PyObject *self, PyObject* args) {
    PyObject *xarg, *yarg, *netarg, *tenarg;
    PyArrayObject *x, *y, *net, *ten, *area;
    unsigned int *e, n;

    // parse input
    if(!PyArg_ParseTuple(args, "OOOO", &netarg, &tenarg, &xarg, &yarg))
        return NULL;
    net = (PyArrayObject *) PyArray_ContiguousFromObject(netarg, PyArray_UINT, 1, 1);
    ten = (PyArrayObject *) PyArray_ContiguousFromObject(tenarg, PyArray_UINT, 1, 1);
    x = (PyArrayObject *) PyArray_ContiguousFromObject(xarg, PyArray_DOUBLE, 1, 1);
    y = (PyArrayObject *) PyArray_ContiguousFromObject(yarg, PyArray_DOUBLE, 1, 1);
    if(!net || !ten || !x || !y)
        return NULL;

    // check input
    n = x->dimensions[0];
    e = (unsigned int *) net->data;
    if(n != e[0] - 1) {
        PyErr_SetString(PyExc_IndexError, "coordinates do not match with the networks.");
        return NULL;
    }
    if(e[n] != net->dimensions[0]) {
        PyErr_SetString(PyExc_IndexError, "corrupted network format for the flow network.");
        return NULL;
    }
    e = (unsigned int *) ten->data;
    if(e[n] != ten->dimensions[0]) {
        PyErr_SetString(PyExc_IndexError, "corrupted network format for the reverse flow network.");
        return NULL;
    }
    area = voronoi((unsigned int *)net->data, (unsigned int *)ten->data, (double *)x->data, (double *)y->data);
    Py_DECREF(x);
    Py_DECREF(y);
    Py_DECREF(net);
    Py_DECREF(ten);
    return PyArray_Return(area);
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
DemNets_RemoveLinks(PyObject *self, PyObject* args) {
    PyObject *netarg, *tenarg, *rmnarg;
    PyArrayObject *net, *ten, *rmn, *new;
    unsigned int *x, *y;

    // parse input
    if(!PyArg_ParseTuple(args, "OOO", &netarg, &tenarg, &rmnarg))
        return NULL;
    net = (PyArrayObject *) PyArray_ContiguousFromObject(netarg, PyArray_UINT, 1, 1);
    ten = (PyArrayObject *) PyArray_ContiguousFromObject(tenarg, PyArray_UINT, 1, 1);
    rmn = (PyArrayObject *) PyArray_ContiguousFromObject(rmnarg, PyArray_UINT, 1, 1);
    if(!net || !ten || !rmn)
        return NULL;

    // check network format
    x = (unsigned int *) net->data;
    y = (unsigned int *) ten->data;
    if(x[0] != y[0]) {
        PyErr_SetString(PyExc_IndexError, "networks do not match.");
        return NULL;
    }
    if(x[x[0]-1] != net->dimensions[0]) {
        PyErr_SetString(PyExc_IndexError, "corrupted format for the flow network.");
        return NULL;
    }
    if(y[y[0]-1] != ten->dimensions[0]) {
        PyErr_SetString(PyExc_IndexError, "corrupted format for the reverse-flow network.");
        return NULL;
    }

    // remove nodes and links from network
    new = removelinks(x, y, (unsigned int *)rmn->data, rmn->dimensions[0]);

    Py_DECREF(net);
    Py_DECREF(ten);
    Py_DECREF(rmn);
    return PyArray_Return(new);
}

static PyObject *
DemNets_ReverseLinks(PyObject *self, PyObject* args) {
    PyObject *netarg;
    PyArrayObject *net, *new;
    unsigned int *x;

    // parse input
    if(!PyArg_ParseTuple(args, "O", &netarg))
        return NULL;
    net = (PyArrayObject *) PyArray_ContiguousFromObject(netarg, PyArray_UINT, 1, 1);
    if(!net)
        return NULL;

    // check network format
    x = (unsigned int *) net->data;
    if(x[x[0] - 1] != net->dimensions[0]) {
        PyErr_SetString(PyExc_IndexError, "corrupted network.");
        return NULL;
    }

    // reverse links, swap direction of each link
    new = reverselinks(x);

    Py_DECREF(net);
    return PyArray_Return(new);
}

static PyObject *
DemNets_AggregateLinkThroughput(PyObject *self, PyObject* args) {
    PyObject *thetaarg, *ltputarg, *netarg;
    PyArrayObject *rho, *phi, *theta, *ltput, *net;
    npy_intp dim[1];
    unsigned int *e, n;
    double *rh, *ph;

    // parse input
    if(!PyArg_ParseTuple(args, "OOO", &netarg, &ltputarg, &thetaarg))
        return NULL;
    net = (PyArrayObject *) PyArray_ContiguousFromObject(netarg, PyArray_UINT, 1, 1);
    ltput = (PyArrayObject *) PyArray_ContiguousFromObject(ltputarg, PyArray_DOUBLE, 1, 1);
    theta = (PyArrayObject *) PyArray_ContiguousFromObject(thetaarg, PyArray_DOUBLE, 1, 1);
    if(!net || !ltput || !theta)
        return NULL;

    // check input
    e = (unsigned int *) net->data;
    n = e[0] - 1;
    dim[0] = n;
    if(e[n] != net->dimensions[0]) {
        PyErr_SetString(PyExc_IndexError, "corrupted network format.");
        return NULL;
    }
    n = e[n] - n - 1;
    if(n != ltput->dimensions[0]) {
        PyErr_SetString(PyExc_IndexError, "link throughput does not match network.");
        return NULL;
    }
    if(n != theta->dimensions[0]) {
        PyErr_SetString(PyExc_IndexError, "link-angle array does not match network.");
        return NULL;
    }

    // allocate output arrays
    rho = (PyArrayObject *) PyArray_ZEROS(1, dim, PyArray_DOUBLE, 0);
    phi = (PyArrayObject *) PyArray_ZEROS(1, dim, PyArray_DOUBLE, 0);
    if(!rho || !phi) {
        PyErr_SetString(PyExc_MemoryError,
        "Cannot allocate enough memory for output.");
        return NULL;
    }

    rh = (double *)rho->data;
    ph = (double *)phi->data;
    AggregateLinkThroughput(rh, ph, e,
                            (double *)ltput->data,
                            (double *)theta->data);

    Py_DECREF(net);
    Py_DECREF(ltput);
    Py_DECREF(theta);
    return Py_BuildValue("(OO)", rho, phi);
}

static PyObject *
DemNets_NodeThroughput(PyObject *self, PyObject* args) {
    PyObject *thetaarg, *netarg, *ldarg, *lwarg, *nwarg;
    PyArrayObject *rho, *phi, *theta, *ld, *lw, *nw, *net;
    unsigned int *e, n;
    double *rh, *ph;

    // parse input
    if(!PyArg_ParseTuple(args, "OOOOO", &netarg, &nwarg, &lwarg, &ldarg, &thetaarg))
        return NULL;
    net = (PyArrayObject *) PyArray_ContiguousFromObject(netarg, PyArray_UINT, 1, 1);
    nw = (PyArrayObject *) PyArray_ContiguousFromObject(nwarg, PyArray_DOUBLE, 1, 1);
    lw = (PyArrayObject *) PyArray_ContiguousFromObject(lwarg, PyArray_DOUBLE, 1, 1);
    ld = (PyArrayObject *) PyArray_ContiguousFromObject(ldarg, PyArray_DOUBLE, 1, 1);
    theta = (PyArrayObject *) PyArray_ContiguousFromObject(thetaarg, PyArray_DOUBLE, 1, 1);
    if(!net || !nw || !lw || !ld || !theta)
        return NULL;

    // check input
    e = (unsigned int *) net->data;
    n = e[0] - 1;
    if(e[n] != net->dimensions[0]) {
        PyErr_SetString(PyExc_IndexError, "corrupted network format.");
        return NULL;
    }
    if(n != nw->dimensions[0]) {
        PyErr_SetString(PyExc_IndexError, "node-weight array does not match network.");
        return NULL;
    }
    n = e[n] - n - 1;
    if(n != lw->dimensions[0]) {
        PyErr_SetString(PyExc_IndexError, "link-weight array does not match network.");
        return NULL;
    }
    if(n != ld->dimensions[0]) {
        PyErr_SetString(PyExc_IndexError, "link-width array does not match network.");
        return NULL;
    }
    if(n != theta->dimensions[0]) {
        PyErr_SetString(PyExc_IndexError, "link-angle array does not match network.");
        return NULL;
    }

    // allocate output arrays
    rho = (PyArrayObject *) PyArray_ZEROS(1, nw->dimensions, PyArray_DOUBLE, 0);
    phi = (PyArrayObject *) PyArray_ZEROS(1, nw->dimensions, PyArray_DOUBLE, 0);
    if(!rho || !phi) {
        PyErr_SetString(PyExc_MemoryError,
        "Cannot allocate enough memory for output.");
        return NULL;
    }

    // get node throughput
    rh = (double *)rho->data;
    ph = (double *)phi->data;
    NodeThroughput(rh, ph, e,
            (double *)nw->data,
            (double *)lw->data,
            (double *)ld->data,
            (double *)theta->data);

    Py_DECREF(net);
    Py_DECREF(nw);
    Py_DECREF(lw);
    Py_DECREF(ld);
    Py_DECREF(theta);
    return Py_BuildValue("(OO)", rho, phi);
}

static PyObject *
DemNets_SteepThroughput(PyObject *self, PyObject* args) {
    PyObject *thetaarg, *netarg, *ldarg, *lwarg, *nwarg;
    PyArrayObject *rho, *phi, *theta, *ld, *lw, *nw, *net;
    unsigned int *e, n;
    double *rh, *ph;

    // parse input
    if(!PyArg_ParseTuple(args, "OOOOO", &netarg, &nwarg, &lwarg, &ldarg, &thetaarg))
        return NULL;
    net = (PyArrayObject *) PyArray_ContiguousFromObject(netarg, PyArray_UINT, 1, 1);
    nw = (PyArrayObject *) PyArray_ContiguousFromObject(nwarg, PyArray_DOUBLE, 1, 1);
    lw = (PyArrayObject *) PyArray_ContiguousFromObject(lwarg, PyArray_DOUBLE, 1, 1);
    ld = (PyArrayObject *) PyArray_ContiguousFromObject(ldarg, PyArray_DOUBLE, 1, 1);
    theta = (PyArrayObject *) PyArray_ContiguousFromObject(thetaarg, PyArray_DOUBLE, 1, 1);
    if(!net || !nw || !lw || !ld || !theta)
        return NULL;

    // check input
    e = (unsigned int *) net->data;
    n = e[0] - 1;
    if(e[n] != net->dimensions[0]) {
        PyErr_SetString(PyExc_IndexError, "corrupted network format.");
        return NULL;
    }
    if(n != nw->dimensions[0]) {
        PyErr_SetString(PyExc_IndexError, "node-weight array does not match network.");
        return NULL;
    }
    n = e[n] - n - 1;
    if(n != lw->dimensions[0]) {
        PyErr_SetString(PyExc_IndexError, "link-weight array does not match network.");
        return NULL;
    }
    if(n != ld->dimensions[0]) {
        PyErr_SetString(PyExc_IndexError, "link-width array does not match network.");
        return NULL;
    }
    if(n != theta->dimensions[0]) {
        PyErr_SetString(PyExc_IndexError, "link-angle array does not match network.");
        return NULL;
    }

    // allocate output arrays
    rho = (PyArrayObject *) PyArray_ZEROS(1, nw->dimensions, PyArray_DOUBLE, 0);
    phi = (PyArrayObject *) PyArray_ZEROS(1, nw->dimensions, PyArray_DOUBLE, 0);
    if(!rho || !phi) {
        PyErr_SetString(PyExc_MemoryError,
        "Cannot allocate enough memory for output.");
        return NULL;
    }

    // get node throughput
    rh = (double *)rho->data;
    ph = (double *)phi->data;
    SteepThroughput(rh, ph, e,
            (double *)nw->data,
            (double *)lw->data,
            (double *)ld->data,
            (double *)theta->data);

    Py_DECREF(net);
    Py_DECREF(nw);
    Py_DECREF(lw);
    Py_DECREF(ld);
    Py_DECREF(theta);
    return Py_BuildValue("(OO)", rho, phi);
}

static PyObject *
DemNets_SpecificNodeThroughput(PyObject *self, PyObject* args) {
    PyObject *thetaarg, *netarg, *ldarg, *lwarg, *nwarg;
    PyArrayObject *rho, *phi, *theta, *ld, *lw, *nw, *net;
    unsigned int *e, n;
    double *rh, *ph;

    // parse input
    if(!PyArg_ParseTuple(args, "OOOOO", &netarg, &nwarg, &lwarg, &ldarg, &thetaarg))
        return NULL;
    net = (PyArrayObject *) PyArray_ContiguousFromObject(netarg, PyArray_UINT, 1, 1);
    nw = (PyArrayObject *) PyArray_ContiguousFromObject(nwarg, PyArray_DOUBLE, 1, 1);
    lw = (PyArrayObject *) PyArray_ContiguousFromObject(lwarg, PyArray_DOUBLE, 1, 1);
    ld = (PyArrayObject *) PyArray_ContiguousFromObject(ldarg, PyArray_DOUBLE, 1, 1);
    theta = (PyArrayObject *) PyArray_ContiguousFromObject(thetaarg, PyArray_DOUBLE, 1, 1);
    if(!net || !nw || !lw || !ld || !theta)
        return NULL;

    // check input
    e = (unsigned int *) net->data;
    n = e[0] - 1;
    if(e[n] != net->dimensions[0]) {
        PyErr_SetString(PyExc_IndexError, "corrupted network format.");
        return NULL;
    }
    if(n != nw->dimensions[0]) {
        PyErr_SetString(PyExc_IndexError, "node-weight array does not match network.");
        return NULL;
    }
    n = e[n] - n - 1;
    if(n != lw->dimensions[0]) {
        PyErr_SetString(PyExc_IndexError, "link-weight array does not match network.");
        return NULL;
    }
    if(n != ld->dimensions[0]) {
        PyErr_SetString(PyExc_IndexError, "link-width array does not match network.");
        return NULL;
    }
    if(n != theta->dimensions[0]) {
        PyErr_SetString(PyExc_IndexError, "link-angle array does not match network.");
        return NULL;
    }

    // allocate output arrays
    rho = (PyArrayObject *) PyArray_ZEROS(1, nw->dimensions, PyArray_DOUBLE, 0);
    phi = (PyArrayObject *) PyArray_ZEROS(1, nw->dimensions, PyArray_DOUBLE, 0);
    if(!rho || !phi) {
        PyErr_SetString(PyExc_MemoryError,
        "Cannot allocate enough memory for output.");
        return NULL;
    }

    // get node throughput
    rh = (double *)rho->data;
    ph = (double *)phi->data;
    SpecificNodeThroughput(rh, ph, e,
            (double *)nw->data,
            (double *)lw->data,
            (double *)ld->data,
            (double *)theta->data);

    Py_DECREF(net);
    Py_DECREF(nw);
    Py_DECREF(lw);
    Py_DECREF(ld);
    Py_DECREF(theta);
    return Py_BuildValue("(OO)", rho, phi);
}

static PyObject *
DemNets_FacetFlowThroughput(PyObject *self, PyObject* args) {
    PyObject *spxarg, *spwarg, *spaarg;
    PyArrayObject *spx, *spw, *spa, *ltp;
    char errstr[30];
    unsigned int i;

    // parse input
    if(!PyArg_ParseTuple(args, "OOO", &spxarg, &spwarg, &spaarg))
        return NULL;
    spx = (PyArrayObject *) PyArray_ContiguousFromObject(spxarg, PyArray_UINT, 2, 2);
    spw = (PyArrayObject *) PyArray_ContiguousFromObject(spwarg, PyArray_DOUBLE, 2, 2);
    spa = (PyArrayObject *) PyArray_ContiguousFromObject(spaarg, PyArray_DOUBLE, 2, 2);
    if(!spx || !spw || !spa)
        return NULL;

    // check input
    for(i = 0; i < 2; i++) {
        if(spx->dimensions[i] != spw->dimensions[i]) {
            snprintf(errstr, 30 * sizeof(char), "spx.shape[%i] != spw.shape[%i]", i, i);
            PyErr_SetString(PyExc_IndexError, errstr);
            return NULL;
        }
        if(spx->dimensions[i] != spa->dimensions[i]) {
            snprintf(errstr, 30 * sizeof(char), "spx.shape[%i] != spa.shape[%i]", i, i);
            PyErr_SetString(PyExc_IndexError, errstr);
            return NULL;
        }
    }

    // allocate output arrays
    ltp = (PyArrayObject *) PyArray_ZEROS(2, spx->dimensions, PyArray_DOUBLE, 0);
    if(!ltp) {
        PyErr_SetString(PyExc_MemoryError,
        "Cannot allocate enough memory for output.");
        return NULL;
    }

    // get node throughput
    FacetFlowThroughput((double *)ltp->data,
                           (int *)spx->data,
                        (double *)spw->data,
                        (double *)spa->data,
                          ltp->dimensions[0]);
    Py_DECREF(spx);
    Py_DECREF(spw);
    Py_DECREF(spa);
    return PyArray_Return(ltp);
}

static PyObject *
DemNets_FacetFlowNetwork(PyObject *self, PyObject* args) {
    PyObject *netarg, *triarg, *xarg, *yarg, *zarg;
    PyArrayObject *spx, *spw, *spa, *spd, *phi;
    PyArrayObject *x, *y, *z, *net, *tri;
    npy_intp dim[2];
    unsigned int *e, n;

    // parse input
    if(!PyArg_ParseTuple(args, "OOOOO", &triarg, &netarg, &xarg, &yarg, &zarg))
        return NULL;
    tri = (PyArrayObject *) PyArray_ContiguousFromObject(triarg, PyArray_UINT, 2, 2);
    net = (PyArrayObject *) PyArray_ContiguousFromObject(netarg, PyArray_UINT, 1, 1);
    x = (PyArrayObject *) PyArray_ContiguousFromObject(xarg, PyArray_DOUBLE, 1, 1);
    y = (PyArrayObject *) PyArray_ContiguousFromObject(yarg, PyArray_DOUBLE, 1, 1);
    z = (PyArrayObject *) PyArray_ContiguousFromObject(zarg, PyArray_DOUBLE, 1, 1);
    if(!tri || !net || !x || !y || !z)
        return NULL;

    // check input
    e = (unsigned int *) net->data;
    n = e[0] - 1;
    if(e[n] != net->dimensions[0]) {
        PyErr_SetString(PyExc_IndexError, "corrupted network format.");
        return NULL;
    }
    if(n != x->dimensions[0]) {
        PyErr_SetString(PyExc_IndexError, "x array does not match network.");
        return NULL;
    }
    if(n != y->dimensions[0]) {
        PyErr_SetString(PyExc_IndexError, "y array does not match network.");
        return NULL;
    }
    if(n != z->dimensions[0]) {
        PyErr_SetString(PyExc_IndexError, "z array does not match network.");
        return NULL;
    }

    // allocate output arrays
    dim[0] = tri->dimensions[0];
    dim[1] = 2;
    spx = (PyArrayObject *) PyArray_ZEROS(2, dim, PyArray_UINT, 0);
    spw = (PyArrayObject *) PyArray_ZEROS(2, dim, PyArray_DOUBLE, 0);
    spa = (PyArrayObject *) PyArray_ZEROS(2, dim, PyArray_DOUBLE, 0);
    spd = (PyArrayObject *) PyArray_ZEROS(1, dim, PyArray_DOUBLE, 0);
    phi = (PyArrayObject *) PyArray_ZEROS(1, dim, PyArray_DOUBLE, 0);
    if(!spx || !spw || !spa || !spd || !phi) {
        PyErr_SetString(PyExc_MemoryError,
        "Cannot allocate enough memory for output.");
        return NULL;
    }

    // get node throughput
    FacetFlowNetwork((unsigned int *)spx->data,
                  (double *)spw->data,
                  (double *)spa->data,
                  (double *)spd->data,
                  (double *)phi->data,
            (unsigned int *)net->data,
            (unsigned int *)tri->data,
            dim[0],
            (double *)x->data,
            (double *)y->data,
            (double *)z->data);

    Py_DECREF(net);
    Py_DECREF(x);
    Py_DECREF(y);
    Py_DECREF(z);
    Py_DECREF(tri);
    return Py_BuildValue("(OOOOO)", spx, spw, spa, spd, phi);
}

static PyObject *
DemNets_LinkThroughput(PyObject *self, PyObject* args) {
    PyObject *netarg, *ldarg, *lwarg, *nwarg;
    PyArrayObject *ltput, *ld, *lw, *nw, *net;
    unsigned int *e, n;

    // parse input
    nwarg = NULL;
    if(!PyArg_ParseTuple(args, "OOOO", &netarg, &nwarg, &lwarg, &ldarg))
        return NULL;
    net = (PyArrayObject *) PyArray_ContiguousFromObject(netarg, PyArray_UINT, 1, 1);
    nw = (PyArrayObject *) PyArray_ContiguousFromObject(nwarg, PyArray_DOUBLE, 1, 1);
    lw = (PyArrayObject *) PyArray_ContiguousFromObject(lwarg, PyArray_DOUBLE, 1, 1);
    ld = (PyArrayObject *) PyArray_ContiguousFromObject(ldarg, PyArray_DOUBLE, 1, 1);
    if(!net || !nw || !lw || !ld)
        return NULL;

    // check input
    e = (unsigned int *) net->data;
    n = e[0] - 1;
    if(e[n] != net->dimensions[0]) {
        PyErr_SetString(PyExc_IndexError, "corrupted network format.");
        return NULL;
    }
    if(n != nw->dimensions[0]) {
        PyErr_SetString(PyExc_IndexError, "node-weight array does not match network.");
        return NULL;
    }
    n = e[n] - n - 1;
    if(n != lw->dimensions[0]) {
        PyErr_SetString(PyExc_IndexError, "link-weight array does not match network.");
        return NULL;
    }
    if(n != ld->dimensions[0]) {
        PyErr_SetString(PyExc_IndexError, "link-width array does not match network.");
        return NULL;
    }

    // get link throughput
    ltput = LinkThroughput(e,(double *)nw->data,
                             (double *)lw->data,
                             (double *)ld->data);

    Py_DECREF(net);
    Py_DECREF(nw);
    Py_DECREF(lw);
    Py_DECREF(ld);
    return PyArray_Return(ltput);
}

static PyObject *
DemNets_RandomWalkThroughput(PyObject *self, PyObject* args) {
    PyObject *netarg, *lwarg, *nwarg;
    PyArrayObject *wlinks, *wnodes, *net, *t;
    unsigned int *e, n, iter, plen;

    // parse input
    iter = 100;
    plen = MAXPATHLEN;
    if(!PyArg_ParseTuple(args, "OO|II", &netarg, &lwarg, &iter, &plen))
        return NULL;
    net = (PyArrayObject *) PyArray_ContiguousFromObject(netarg, PyArray_UINT, 1, 1);
    wlinks = (PyArrayObject *) PyArray_ContiguousFromObject(lwarg, PyArray_DOUBLE, 1, 1);
    if(!net || !wlinks)
        return NULL;

    // check input
    e = (unsigned int *) net->data;
    n = e[0] - 1;
    if(e[n] != net->dimensions[0]) {
        PyErr_SetString(PyExc_IndexError, "corrupted network format.");
        return NULL;
    }
    if(e[n] - n - 1 != wlinks->dimensions[0]) {
        PyErr_SetString(PyExc_IndexError, "link-weight array does not match network.");
        return NULL;
    }

    // get random walk node throughput
    t = rwthroughput(e, (double *)wlinks->data, iter, plen);

    Py_DECREF(net);
    Py_DECREF(wlinks);
    return PyArray_Return(t);
}

static PyObject *
DemNets_AveragePathLengths(PyObject *self, PyObject* args) {
    PyObject *netarg, *lwarg, *nwarg;
    PyArrayObject *wlinks, *net, *lens;
    unsigned int *e, iter, plen;

    // parse input
    iter = 100;
    plen = MAXPATHLEN;
    if(!PyArg_ParseTuple(args, "OO|II", &netarg, &lwarg, &iter, &plen))
        return NULL;
    net = (PyArrayObject *) PyArray_ContiguousFromObject(netarg, PyArray_UINT, 1, 1);
    wlinks = (PyArrayObject *) PyArray_ContiguousFromObject(lwarg, PyArray_DOUBLE, 1, 1);
    if(!net || !wlinks)
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

    // get node's average path length
    lens = pathlengths(e, (double *)wlinks->data, iter, plen);

    Py_DECREF(net);
    Py_DECREF(wlinks);
    return PyArray_Return(lens);
}

static PyObject *
DemNets_Revisits(PyObject *self, PyObject* args) {
    PyObject *netarg, *lwarg, *nwarg;
    PyArrayObject *wlinks, *net, *r;
    unsigned int *e, iter, plen;

    // parse input
    iter = 1;
    plen = MAXPATHLEN;
    if(!PyArg_ParseTuple(args, "OO|II", &netarg, &lwarg, &iter, &plen))
        return NULL;
    net = (PyArrayObject *) PyArray_ContiguousFromObject(netarg, PyArray_UINT, 1, 1);
    wlinks = (PyArrayObject *) PyArray_ContiguousFromObject(lwarg, PyArray_DOUBLE, 1, 1);
    if(!net || !wlinks)
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

    // get node's average path length
    r = revisits(e, (double *)wlinks->data, iter, plen);

    Py_DECREF(net);
    Py_DECREF(wlinks);
    return PyArray_Return(r);
}

static PyObject *
DemNets_OutDegree(PyObject *self, PyObject* args) {
    PyObject *netarg;
    PyArrayObject *net, *deg;
    npy_intp dim[1];
    unsigned int *e, i, n;
    double *d;

    // parse input
    if(!PyArg_ParseTuple(args, "O", &netarg))
        return NULL;
    net = (PyArrayObject *) PyArray_ContiguousFromObject(netarg, PyArray_UINT, 1, 1);
    if(!net)
        return NULL;

    // check input
    e = (unsigned int *) net->data;
    n = e[0] - 1;
    if(e[n] != net->dimensions[0]) {
        PyErr_SetString(PyExc_IndexError, "corrupted network format.");
        return NULL;
    }

    //npy_intp dim[1];
    dim[0] = n;
    deg = (PyArrayObject *) PyArray_ZEROS(1, dim, PyArray_DOUBLE, 0);
    if(!deg) {
        PyErr_SetString(PyExc_MemoryError,
        "Cannot allocate enough memory for output.");
        return NULL;
    }

    d = (double *) deg->data;
    for(i = 0; i < n; i++)
        d[i] = e[i+1] - e[i];

    Py_DECREF(net);
    return PyArray_Return(deg);
}

static PyObject *
DemNets_Derivatives(PyObject *self, PyObject* args) {
    PyObject *netarg, *lwarg, *zarg, *vds;
    PyArrayObject *wlinks, *z, *net;
    unsigned int *e, n, iter, plen;

    // parse input
    iter = 100;
    if(!PyArg_ParseTuple(args, "OOO|I", &netarg, &lwarg, &zarg, &iter))
        return NULL;
    net = (PyArrayObject *) PyArray_ContiguousFromObject(netarg, PyArray_UINT, 1, 1);
    wlinks = (PyArrayObject *) PyArray_ContiguousFromObject(lwarg, PyArray_DOUBLE, 1, 1);
    z = (PyArrayObject *) PyArray_ContiguousFromObject(zarg, PyArray_DOUBLE, 1, 1);
    if(!net || !wlinks || !z)
        return NULL;

    // check input
    e = (unsigned int *) net->data;
    n = e[0] - 1;
    if(e[n] != net->dimensions[0]) {
        PyErr_SetString(PyExc_IndexError, "corrupted network format.");
        return NULL;
    }
    if(e[n] - n - 1 != wlinks->dimensions[0]) {
        PyErr_SetString(PyExc_IndexError, "link-weight array does not match network.");
        return NULL;
    }
    if(n != z->dimensions[0]) {
        PyErr_SetString(PyExc_IndexError, "height array does not match network.");
        return NULL;
    }

    // get vertical velocity and acceleration of paths at each node
    vds = derivatives(e, (double *)wlinks->data, (double *)z->data, iter);

    Py_DECREF(net);
    Py_DECREF(wlinks);
    Py_DECREF(z);
    return vds;
}

static PyObject *
DemNets_RandomWalk(PyObject *self, PyObject* args) {
    PyObject *netarg, *lwarg, *trace;
    PyArrayObject *net, *wlinks;
    unsigned int *e, plen;
    int start;

    // parse input
    plen = MAXPATHLEN;
    start = -1;
    if(!PyArg_ParseTuple(args, "OO|I", &netarg, &lwarg, &start))//, &plen))
        return NULL;
    net = (PyArrayObject *) PyArray_ContiguousFromObject(netarg, PyArray_UINT, 1, 1);
    wlinks = (PyArrayObject *) PyArray_ContiguousFromObject(lwarg, PyArray_DOUBLE, 1, 1);
    if(!net || !wlinks)
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
    if(start < -1 && start >= e[0] - 1) {
        PyErr_SetString(PyExc_IndexError, "start position not available.");
        return NULL;
    }

    // get random walk trace
    trace = walk(e, (double *)wlinks->data, start, plen);

    Py_DECREF(net);
    Py_DECREF(wlinks);
    return trace;
}

static PyObject *
DemNets_RandomWalks(PyObject *self, PyObject* args) {
    PyObject *netarg, *lwarg, *srcarg;
    PyArrayObject *net, *wlinks, *src, *traces;
    unsigned int *e, dst, iter, plen;

    // parse input
    iter = 1;
    plen = MAXPATHLEN;
    if(!PyArg_ParseTuple(args, "OOOI|II", &netarg, &lwarg, &srcarg, &dst, &iter, &plen))
        return NULL;
    net = (PyArrayObject *) PyArray_ContiguousFromObject(netarg, PyArray_UINT, 1, 1);
    wlinks = (PyArrayObject *) PyArray_ContiguousFromObject(lwarg, PyArray_DOUBLE, 1, 1);
    src = (PyArrayObject *) PyArray_ContiguousFromObject(srcarg, PyArray_UINT, 1, 1);
    if(!net || !wlinks || !src)
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
    if(dst > -1 && dst >= e[0] - 1) {
        PyErr_SetString(PyExc_IndexError, "destination not available.");
        return NULL;
    }

    // get random walk traces
    traces = walks(e, (double *)wlinks->data, dst, (unsigned int *)src->data, src->dimensions[0], iter, plen);

    Py_DECREF(net);
    Py_DECREF(src);
    Py_DECREF(wlinks);
    return PyArray_Return(traces);
}

static PyObject *
DemNets_HillSlopeLengths(PyObject *self, PyObject* args) {
    PyObject *netarg, *lwarg, *zarg, *srcarg;
    PyArrayObject *net, *wlinks, *src, *z, *lengths;
    double zthr;
    unsigned int *e, iter, plen;

    // parse input
    iter = 100;
    plen = MAXPATHLEN;
    if(!PyArg_ParseTuple(args, "OOOdO|I", &netarg, &lwarg, &zarg, &zthr, &srcarg, &iter))
        return NULL;
    net = (PyArrayObject *) PyArray_ContiguousFromObject(netarg, PyArray_UINT, 1, 1);
    wlinks = (PyArrayObject *) PyArray_ContiguousFromObject(lwarg, PyArray_DOUBLE, 1, 1);
    z = (PyArrayObject *) PyArray_ContiguousFromObject(zarg, PyArray_DOUBLE, 1, 1);
    src = (PyArrayObject *) PyArray_ContiguousFromObject(srcarg, PyArray_UINT, 1, 1);
    if(!net || !wlinks || !src || !z)
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

    lengths = hillslopelengths(e, (double *)wlinks->data, (double *)z->data, zthr, (unsigned int *)src->data, src->dimensions[0], iter, plen);

    Py_DECREF(z);
    Py_DECREF(net);
    Py_DECREF(src);
    Py_DECREF(wlinks);
    return PyArray_Return(lengths);
}

static PyObject *
DemNets_SinkDistance(PyObject *self, PyObject* args) {
    PyObject *netarg;
    PyArrayObject *net, *d;
    unsigned int *e, sink;

    // parse input
    if(!PyArg_ParseTuple(args, "OI", &netarg, &sink))
        return NULL;
    net = (PyArrayObject *) PyArray_ContiguousFromObject(netarg, PyArray_UINT, 1, 1);
    if(!net)
        return NULL;

    // check input
    e = (unsigned int *) net->data;
    if(e[e[0] - 1] != net->dimensions[0]) {
        PyErr_SetString(PyExc_IndexError, "corrupted network format.");
        return NULL;
    }
    if(sink >= e[0] - 1) {
        PyErr_SetString(PyExc_IndexError, "sink node not available.");
        return NULL;
    }

    // get upstream distances from sink node
    d = sinkdistance(e, sink);

    Py_DECREF(net);
    return PyArray_Return(d);
}

static PyMethodDef DemNets_methods[] = {
    {"Components", DemNets_Components, METH_VARARGS, "..."},
    {"Derivatives", DemNets_Derivatives, METH_VARARGS, "..."},
    {"OutDegree", DemNets_OutDegree, METH_VARARGS, "..."},
    {"FuseNetworks", DemNets_FuseNetworks, METH_VARARGS, "..."},
    {"Simplicies", DemNets_Simplicies, METH_VARARGS, "..."},
    {"FlowNetwork", DemNets_FlowNetwork, METH_VARARGS, "..."},
    {"FlowNetworkFromGrid", DemNets_FlowNetworkFromGrid, METH_VARARGS, "..."},
    {"FlowDistance", DemNets_FlowDistance, METH_VARARGS, "..."},
    {"LinkLengths", DemNets_LinkLengths, METH_VARARGS, "..."},
    {"LinkAngles", DemNets_LinkAngles, METH_VARARGS, "..."},
    {"HillSlopeLengths", DemNets_HillSlopeLengths, METH_VARARGS, "..."},
    {"Slopes", DemNets_Slopes, METH_VARARGS, "..."},
    {"SinkDistance", DemNets_SinkDistance, METH_VARARGS, "..."},
    {"LinkSinks", DemNets_LinkSinks, METH_VARARGS, "..."},
    {"RandomWalk", DemNets_RandomWalk, METH_VARARGS, "..."},
    {"RandomWalks", DemNets_RandomWalks, METH_VARARGS, "..."},
    {"RemoveLinks", DemNets_RemoveLinks, METH_VARARGS, "..."},
    {"ReverseLinks", DemNets_ReverseLinks, METH_VARARGS, "..."},
    {"Revisits", DemNets_Revisits, METH_VARARGS, "..."},
    {"AveragePathLengths", DemNets_AveragePathLengths, METH_VARARGS, "..."},
    {"VoronoiArea", DemNets_VoronoiArea, METH_VARARGS, "..."},
    {"FacetFlowNetwork", DemNets_FacetFlowNetwork, METH_VARARGS, "..."},
    {"FacetFlowThroughput", DemNets_FacetFlowThroughput, METH_VARARGS, "..."},
    {"NodeThroughput", DemNets_NodeThroughput, METH_VARARGS, "..."},
    {"SteepThroughput", DemNets_SteepThroughput, METH_VARARGS, "..."},
    {"SpecificNodeThroughput", DemNets_SpecificNodeThroughput, METH_VARARGS, "..."},
    {"LinkThroughput", DemNets_LinkThroughput, METH_VARARGS, "..."},
    {"AggregateLinkThroughput", DemNets_AggregateLinkThroughput, METH_VARARGS, "..."},
    {"RandomWalkThroughput", DemNets_RandomWalkThroughput, METH_VARARGS, "..."},
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
