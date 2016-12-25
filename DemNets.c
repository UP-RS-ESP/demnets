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
    double *p;
    double dx, dy;
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
    p = (double *) phi->data;

    // estimate link i->j angular directions
    for(i = 0; i < n; i++) {
        xi = x[i];
        yi = y[i];
        for(k = net[i]; k < net[i+1]; k++) {
            j = net[k];
            dx = xi - x[j];
            dy = yi - y[j];
            p[k - n - 1] = atan2(dy, dx);
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

static PyArrayObject *
throughput(const unsigned int *net,
           const double *nw,
           const double *lw,
           const double *ld,
           const double *phi) {
    PyArrayObject *tput;
    npy_intp *dim;
    unsigned int i, k;
    unsigned int n, *c;
    unsigned int *seen;
    double *t, ltp, lws, lds, psum;
    double xtp, ytp;
    Queue *que;

    // alloc numpy array for output
    n = net[0] - 1;
    dim = malloc(sizeof(npy_intp));
    dim[0] = n;
    tput = (PyArrayObject *) PyArray_ZEROS(1, dim, PyArray_DOUBLE, 0);
    free(dim);
    if(!tput) {
        PyErr_SetString(PyExc_MemoryError, "...");
        return NULL;
    }
    t = (double *) tput->data;

    seen = calloc(n, sizeof(unsigned int));
    que = malloc(sizeof(Queue));
    c = calloc(n, sizeof(unsigned int));
    if(!c || !que || !seen) {
        PyErr_SetString(PyExc_MemoryError, "...");
        return NULL;
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
                lws += lw[k - n - 1];
                lds += ld[k - n - 1];
            }
            for(k = net[i]; k < net[i+1]; k++)
                psum += lw[k - n - 1] * ld[k - n - 1] / lws / lds;
            xtp = ytp = 0;
            for(k = net[i]; k < net[i+1]; k++) {
                ltp = nw[i] * lw[k-n-1] / lws;
                // put children j = net[k] of hill tops i into queue
                if(put(que, net[k], ltp)) {
                    PyErr_SetString(PyExc_MemoryError, "failed to fill queue ..");
                    exit(EXIT_FAILURE);
                }
                xtp += ltp * cos(phi[k-n-1]) / ld[k-n-1];
                ytp += ltp * sin(phi[k-n-1]) / ld[k-n-1];
            }
            t[i] = sqrt(xtp*xtp + ytp*ytp);
            seen[i] = 1;
        }
    }
    while(!get(que, &i, &ltp)) {
        t[i] += ltp;
        if(seen[i] == c[i] - 1) {
            lws = lds = psum = 0;
            for(k = net[i]; k < net[i+1]; k++) {
                lws += lw[k - n - 1];
                lds += ld[k - n - 1];
            }
            for(k = net[i]; k < net[i+1]; k++)
                psum += lw[k - n - 1] * ld[k - n - 1] / lws / lds;
            xtp = ytp = 0;
            for(k = net[i]; k < net[i+1]; k++) {
                ltp = (t[i] + nw[i]) * lw[k-n-1] / lws;
                if(put(que, net[k], ltp)) {
                    PyErr_SetString(PyExc_MemoryError, "failed to fill queue ..");
                    exit(EXIT_FAILURE);
                }
                xtp += ltp * cos(phi[k-n-1]) / ld[k-n-1];
                ytp += ltp * sin(phi[k-n-1]) / ld[k-n-1];
            }
            t[i] = sqrt(xtp*xtp + ytp*ytp);
        }
        seen[i]++;
    }
    free(seen);
    free(que);
    free(c);
    return tput;
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
DemNets_Throughput(PyObject *self, PyObject* args) {
    PyObject *phiarg, *netarg, *ldarg, *lwarg, *nwarg;
    PyArrayObject *phi, *ld, *lw, *nw, *net, *t;
    unsigned int *e, n, i;
    npy_intp *dim;
    double *w;

    // parse input
    nwarg = NULL;
    if(!PyArg_ParseTuple(args, "OOOO|O", &netarg, &lwarg, &ldarg, &phiarg, &nwarg))
        return NULL;
    net = (PyArrayObject *) PyArray_ContiguousFromObject(netarg, PyArray_UINT, 1, 1);
    lw = (PyArrayObject *) PyArray_ContiguousFromObject(lwarg, PyArray_DOUBLE, 1, 1);
    ld = (PyArrayObject *) PyArray_ContiguousFromObject(ldarg, PyArray_DOUBLE, 1, 1);
    phi = (PyArrayObject *) PyArray_ContiguousFromObject(phiarg, PyArray_DOUBLE, 1, 1);
    if(!net || !lw || !ld || !phi)
        return NULL;
    if(!nwarg) {
        e = (unsigned int *) net->data;
        dim = malloc(sizeof(npy_intp));
        n = e[0] - 1;
        dim[0] = n;
        nw = (PyArrayObject *) PyArray_ZEROS(1, dim, PyArray_DOUBLE, 0);
        free(dim);
        w = (double *) nw->data;
        for(i = 0; i < n; i++)
            w[i] = 1.0;
    } else {
        nw = (PyArrayObject *) PyArray_ContiguousFromObject(nwarg, PyArray_DOUBLE, 1, 1);
    }
    if(!nw)
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
    if(n != phi->dimensions[0]) {
        PyErr_SetString(PyExc_IndexError, "link-angle array does not match network.");
        return NULL;
    }

    // get node throughput
    t = throughput(e, (double *)nw->data, (double *)lw->data, (double *)ld->data, (double *)phi->data);

    Py_DECREF(net);
    Py_DECREF(nw);
    Py_DECREF(lw);
    Py_DECREF(ld);
    Py_DECREF(phi);
    return PyArray_Return(t);
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
    {"FuseNetworks", DemNets_FuseNetworks, METH_VARARGS, "..."},
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
    {"Throughput", DemNets_Throughput, METH_VARARGS, "..."},
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
