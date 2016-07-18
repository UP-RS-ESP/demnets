#include "Python.h"
#include "numpy/arrayobject.h"
#include <fcntl.h>
#include <math.h>
#include <omp.h>

#define VERSION "0.1"

#define MINELEVATION -999

typedef struct Stack Stack;
typedef struct Queue Queue;
typedef struct Node Node;

struct Node {
    Node *next;
    unsigned int i;
    unsigned int d;
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
            // add nodes to stack with increasing distance d
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
components(const unsigned int *net,
           const unsigned int *seeds, const unsigned int nseeds) {
    PyArrayObject *com;
    npy_intp *dim;
    unsigned int i, j, k, l;
    unsigned int *c, n, d;
    unsigned int *seen;
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
    unsigned int *seen, *d, di;
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
        l[i] = net[i];
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
    l[n] = net[n];
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
    list = NULL;
    len = 0;
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
            seen[j] = 1;
            link = 0;
            while(!get(que, &i, &d) && !link) {
                //if(d > 5)
                //    break;
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
                    seen[l]++;
                    if(z[l] <= zj) {
                        list[m++] = j;
                        list[m++] = l;
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
                    if(z[l] <= zj) {
                        list[m++] = j;
                        list[m++] = l;
                        link = 1;
                        break;
                    }
                    put(que, l, d+1);
                }
            }
            free(seen);
            free(que);
        }
    }
    if(!m) {
        free(list);
        PyErr_SetString(PyExc_RuntimeError, "no sink was linked.");
        return NULL;
    }

    // alloc numpy array
    dim = malloc(sizeof(npy_intp));
    dim[0] = n + m/2 + 1;
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
        while(list[l] == i) {
            e[j++] = list[++l];
            l++;
        }
    }
    e[n] = j;
    free(list);
    return new;
}

static PyArrayObject *
linksinksr(const unsigned int *net,
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
    list = NULL;
    len = 0;
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
            seen[j] = 1;
            link = 0;
            while(!get(que, &i, &d) && !link) {
                //if(d > 5)
                //    break;
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
                    seen[l]++;
                    if(z[l] <= zj) {
                        list[m++] = l;
                        list[m++] = j;
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
                    if(z[l] <= zj) {
                        list[m++] = l;
                        list[m++] = j;
                        link = 1;
                        break;
                    }
                    put(que, l, d+1);
                }
            }
            free(seen);
            free(que);
        }
    }
    if(!m) {
        free(list);
        PyErr_SetString(PyExc_RuntimeError, "no sink was linked.");
        return NULL;
    }
    for(i = 0; i < m; i+=2)
        fprintf(stderr, "%i %i\n", list[i], list[i+1]);

    // alloc numpy array
    dim = malloc(sizeof(npy_intp));
    dim[0] = n + m/2 + 1;
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
        while(list[l] == i) {
            e[j++] = list[++l];
            l++;
        }
    }
    e[n] = j;
    free(list);
    return new;
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
            cum[l++] = lw[net[u]] + 1;
            for(k = net[u] + 1; k < net[u+1]; k++) {
                cum[l] = cum[l-1] + lw[k] + 1;
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
gridnetwork(const double *z,
            const unsigned int xn,
            const unsigned int yn,
            const unsigned int flow) {
    PyArrayObject *net;
    npy_intp *dim;
    unsigned int *gn;
    unsigned int i, j, k, n, m;
    double h;

    n = xn * yn;
    gn = malloc(n * 9 * sizeof(unsigned int));
    if(!gn) {
        PyErr_SetString(PyExc_MemoryError, "...");
        return NULL;
    }
    m = n + 1;
    for(k = 0; k < xn; k++)
        gn[k] = m;
    if(flow) {
        for(i = 1; i < yn; i++) {
            j = i * xn;
            gn[j] = m;
            h = z[j];
            if(h >= MINELEVATION) {
                if(z[j - xn] <= h)
                    gn[m++] = j - xn;
                if(z[j - xn + 1] <= h)
                    gn[m++] = j - xn + 1;
                if(z[j + 1] <= h)
                    gn[m++] = j + 1;
                if(z[j + xn] <= h)
                    gn[m++] = j + xn;
                if(z[j + xn + 1] <= h)
                    gn[m++] = j + xn + 1;
            }
            for(k = 1; k < xn - 1; k++) {
                j = i * xn + k;
                gn[j] = m;
                h = z[j];
                if(h < MINELEVATION)
                    continue;
                if(z[j - xn - 1] <= h)
                    gn[m++] = j - xn - 1;
                if(z[j - xn] <= h)
                    gn[m++] = j - xn;
                if(z[j - xn + 1] <= h)
                    gn[m++] = j - xn + 1;
                if(z[j - 1] <= h)
                    gn[m++] = j - 1;
                if(z[j + 1] <= h)
                    gn[m++] = j + 1;
                if(z[j + xn - 1] <= h)
                    gn[m++] = j + xn - 1;
                if(z[j + xn] <= h)
                    gn[m++] = j + xn;
                if(z[j + xn + 1] <= h)
                    gn[m++] = j + xn + 1;
            }
            j = (i + 1) * xn - 1;
            gn[j] = m;
            h = z[j];
            if(h >= MINELEVATION) {
                if(z[j - xn - 1] <= h)
                    gn[m++] = j - xn - 1;
                if(z[j - xn] <= h)
                    gn[m++] = j - xn;
                if(z[j - 1] <= h)
                    gn[m++] = j - 1;
                if(z[j + xn - 1] <= h)
                    gn[m++] = j + xn - 1;
                if(z[j + xn] <= h)
                    gn[m++] = j + xn;
            }
        }
    } else {
        for(i = 1; i < yn; i++) {
            j = i * xn;
            gn[j] = m;
            h = z[j];
            if(h >= MINELEVATION) {
                if(z[j - xn] >= h)
                    gn[m++] = j - xn;
                if(z[j - xn + 1] >= h)
                    gn[m++] = j - xn + 1;
                if(z[j + 1] >= h)
                    gn[m++] = j + 1;
                if(z[j + xn] >= h)
                    gn[m++] = j + xn;
                if(z[j + xn + 1] >= h)
                    gn[m++] = j + xn + 1;
            }
            for(k = 1; k < xn - 1; k++) {
                j = i * xn + k;
                gn[j] = m;
                h = z[j];
                if(h < MINELEVATION)
                    continue;
                if(z[j - xn - 1] >= h)
                    gn[m++] = j - xn - 1;
                if(z[j - xn] >= h)
                    gn[m++] = j - xn;
                if(z[j - xn + 1] >= h)
                    gn[m++] = j - xn + 1;
                if(z[j - 1] >= h)
                    gn[m++] = j - 1;
                if(z[j + 1] >= h)
                    gn[m++] = j + 1;
                if(z[j + xn - 1] >= h)
                    gn[m++] = j + xn - 1;
                if(z[j + xn] >= h)
                    gn[m++] = j + xn;
                if(z[j + xn + 1] >= h)
                    gn[m++] = j + xn + 1;
            }
            j = (i + 1) * xn - 1;
            gn[j] = m;
            h = z[j];
            if(h >= MINELEVATION) {
                if(z[j - xn - 1] >= h)
                    gn[m++] = j - xn - 1;
                if(z[j - xn] >= h)
                    gn[m++] = j - xn;
                if(z[j - 1] >= h)
                    gn[m++] = j - 1;
                if(z[j + xn - 1] >= h)
                    gn[m++] = j + xn - 1;
                if(z[j + xn] >= h)
                    gn[m++] = j + xn;
            }
        }
    }
    for(k = n - xn; k <= n; k++)
        gn[k] = m;

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
       const unsigned int *net,
       const double scale) {
    PyArrayObject *slp;
    npy_intp *dim;
    unsigned int i, j, k, n;
    unsigned int *s;
    double dx, dy;

    // alloc numpy array
    n = net[0] - 1;
    dim = malloc(sizeof(npy_intp));
    dim[0] = net[n];
    slp = (PyArrayObject *) PyArray_ZEROS(1, dim, PyArray_UINT, 0);
    free(dim);
    if(!slp) {
        PyErr_SetString(PyExc_MemoryError, "...");
        return NULL;
    }
    s = (unsigned int *) slp->data;

    // estimate slopes along links
    for(i = 0; i < n; i++) {
        s[i] = net[i];
        for(k = net[i]; k < net[i+1]; k++) {
            j = net[k];
            dx = x[i] - x[j];
            dy = y[i] - y[j];
            s[k] = scale * fabs(z[i] - z[j]) / sqrt(dx*dx + dy*dy);
        }
    }
    s[n] = net[n];
    return slp;
}

static PyArrayObject *
throughput(const unsigned int *net,
           const double *lw,
           const double *nw,
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

    v = 0;
#pragma omp parallel for private(i,k,l,o,u,v,p,cum,oset,nbrs) schedule(guided)
    for(i = 0; i < n * iter; i++) {
        oset = n * omp_get_thread_num();
        u = i % n;
        o = 0;
        nbrs = net[u+1] - net[u];
        while(nbrs && o < plen) {
            cum = malloc(nbrs * sizeof(double));
            l = 0;
            cum[l++] = lw[net[u]] + 1;
            for(k = net[u] + 1; k < net[u+1]; k++) {
                cum[l] = cum[l-1] + lw[k] + 1;
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

static PyObject *
derivatives(const unsigned int *net,
           const double *lw,
           const double *z,
           const unsigned int iter,
           const unsigned int plen) {
    PyArrayObject *velo, *accl;
    npy_intp *dim;
    unsigned int i, k, l, o;
    unsigned int w, u, t, s, r, x, y;
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

#pragma omp parallel for private(i,k,l,o,w,u,t,s,r,x,y,p,cum,oset,nbrs) schedule(guided)
    for(i = 0; i < n * iter; i++) {
        s = t = u = w = n + 1;
        x = y = o = l = 0;
        oset = n * omp_get_thread_num();
        r = i % n;
        nbrs = net[r+1] - net[r];
        if(!nbrs)
            continue;
        cum = malloc(nbrs * sizeof(double));
        cum[l++] = lw[net[r]] + 1;
        for(k = net[r] + 1; k < net[r+1]; k++) {
            cum[l] = cum[l-1] + lw[k] + 1;
            l++;
        }
        p = cum[nbrs-1] * drand48();
        l = 0;
        for(k = net[r]; k < net[r+1]; k++) {
            if(p < cum[l++]) {
                x = net[k];
                break;
            }
        }
        free(cum);
        nbrs = net[x+1] - net[x];
        while(nbrs && o < plen) {
            cum = malloc(nbrs * sizeof(double));
            l = 0;
            cum[l++] = lw[net[x]] + 1;
            for(k = net[x] + 1; k < net[x+1]; k++) {
                cum[l] = cum[l-1] + lw[k] + 1;
                l++;
            }
            p = cum[nbrs-1] * drand48();
            l = 0;
            for(k = net[x]; k < net[x+1]; k++) {
                if(p < cum[l++]) {
                    y = net[k];
                    break;
                }
            }
            free(cum);
            nbrs = net[y+1] - net[y];
            c[x + oset] += 1;
            v[x + oset] += z[r] - z[y];
            //a[x + oset] += z[y] + z[r] - z[x]*2;
            if(w < n)
                a[s + oset] += z[y]+z[w]+2*(z[x]+z[u])-z[r]-z[t]-4*z[s];
            w = u;
            u = t;
            t = s;
            s = r;
            r = x;
            x = y;
            o++;
        }
    }
    // end parallel for

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
        ve[k] += v[k];
        ac[k] += a[k];
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
        ve[k] /= c[k] * 2.0;
        ac[k] /= c[k] * 16.;
    }
    free(c);
	return Py_BuildValue("(OO)", velo, accl);
}

static PyArrayObject *
walk(const unsigned int *net,
     const double *lw,
     const int start,
     const unsigned int plen) {
    PyArrayObject *trace;
    npy_intp *dim;
    unsigned int k, l, o, u, v;
    unsigned int n, nbrs;
    unsigned int *t;
    double *cum, p;

    if(!initrng())
        return NULL;

    // alloc numpy array
    n = net[0] - 1;
    dim = malloc(sizeof(npy_intp));
    dim[0] = n;
    trace = (PyArrayObject *) PyArray_ZEROS(1, dim, PyArray_UINT, 0);
    free(dim);
    if(!trace) {
        PyErr_SetString(PyExc_MemoryError, "...");
        return NULL;
    }
    t = (unsigned int *) trace->data;

    // start random walk at node start
    if(start < 0)
        u = (unsigned int)(n * drand48());
    else
        u = start;
    nbrs = net[u+1] - net[u];
    o = 0;
    while(nbrs && o < plen) {
        cum = malloc(nbrs * sizeof(double));
        l = 0;
        cum[l++] = lw[net[u]] + 1;
        for(k = net[u] + 1; k < net[u+1]; k++) {
            cum[l] = cum[l-1] + lw[k] + 1;
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
        u = v;
        o++;
    }
    printf("reached node %i after %i steps.\n", u, o);
    return trace;
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
    plen = 1000;
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
    unsigned int flow;

    // parse input
    flow = 1;
    if(!PyArg_ParseTuple(args, "O|I", &elearg, &flow))
        return NULL;
    ele = (PyArrayObject *) PyArray_ContiguousFromObject(elearg, PyArray_DOUBLE, 2, 2);
    if(!ele)
        return NULL;

    // retrieve flow network from elevation grid
    net = gridnetwork((double *)ele->data, ele->dimensions[1], ele->dimensions[0], flow);
    Py_DECREF(ele);
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
DemNets_LinkSinks_r(PyObject *self, PyObject* args) {
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
    new = linksinksr((unsigned int *)net->data, (unsigned int *)ten->data, (double *)ele->data);
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
    double scale;

    // parse input
    scale = 100000;
    if(!PyArg_ParseTuple(args, "OOOO|d", &xarg, &yarg, &zarg, &netarg, &scale))
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
    slp = slopes((double *)x->data, (double *)y->data, (double *)z->data, e, scale);

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
    PyObject *netarg, *lwarg, *nwarg;
    PyArrayObject *wlinks, *wnodes, *net, *t;
    unsigned int *e, iter, plen;

    // parse input
    iter = 1;
    plen = 1000000;
    if(!PyArg_ParseTuple(args, "OOO|II", &netarg, &lwarg, &nwarg, &iter, &plen))
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
    t = throughput(e, (double *)wlinks->data, (double *)wnodes->data, iter, plen);

    Py_DECREF(net);
    Py_DECREF(wlinks);
    Py_DECREF(wnodes);
    return PyArray_Return(t);
}

static PyObject *
DemNets_Derivatives(PyObject *self, PyObject* args) {
    PyObject *netarg, *lwarg, *zarg, *vds;
    PyArrayObject *wlinks, *z, *net;
    unsigned int *e, iter, plen;

    // parse input
    iter = 1;
    plen = 1000000;
    if(!PyArg_ParseTuple(args, "OOO|II", &netarg, &lwarg, &zarg, &iter, &plen))
        return NULL;
    net = (PyArrayObject *) PyArray_ContiguousFromObject(netarg, PyArray_UINT, 1, 1);
    wlinks = (PyArrayObject *) PyArray_ContiguousFromObject(lwarg, PyArray_DOUBLE, 1, 1);
    z = (PyArrayObject *) PyArray_ContiguousFromObject(zarg, PyArray_DOUBLE, 1, 1);
    if(!net || !wlinks || !z)
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
    if(e[0] - 1 != z->dimensions[0]) {
        PyErr_SetString(PyExc_IndexError, "height array does not match network.");
        return NULL;
    }

    // get vertical velocity and acceleration of paths at each node
    vds = derivatives(e, (double *)wlinks->data, (double *)z->data, iter, plen);

    Py_DECREF(net);
    Py_DECREF(wlinks);
    Py_DECREF(z);
    return vds;
}

static PyObject *
DemNets_RandomWalk(PyObject *self, PyObject* args) {
    PyObject *netarg, *lwarg;
    PyArrayObject *net, *wlinks, *t;
    unsigned int *e, plen;
    int start;

    // parse input
    start = -1;
    if(!PyArg_ParseTuple(args, "OO|II", &netarg, &lwarg, &start, &plen))
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
    if(start > -1 && start >= e[0] - 1) {
        PyErr_SetString(PyExc_IndexError, "start position not available.");
        return NULL;
    }

    // get random walk trace
    t = walk(e, (double *)wlinks->data, start, plen);

    Py_DECREF(net);
    return PyArray_Return(t);
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
    {"Betweenness", DemNets_Betweenness, METH_VARARGS, "..."},
    {"Components", DemNets_Components, METH_VARARGS, "..."},
    {"Derivatives", DemNets_Derivatives, METH_VARARGS, "..."},
    {"FuseNetworks", DemNets_FuseNetworks, METH_VARARGS, "..."},
    {"FlowNetwork", DemNets_FlowNetwork, METH_VARARGS, "..."},
    {"FlowNetworkFromGrid", DemNets_FlowNetworkFromGrid, METH_VARARGS, "..."},
    {"FlowDistance", DemNets_FlowDistance, METH_VARARGS, "..."},
    {"LinkLengths", DemNets_LinkLengths, METH_VARARGS, "..."},
    {"Slopes", DemNets_Slopes, METH_VARARGS, "..."},
    {"SinkDistance", DemNets_SinkDistance, METH_VARARGS, "..."},
    {"LinkSinks", DemNets_LinkSinks, METH_VARARGS, "..."},
    {"LinkSinks_r", DemNets_LinkSinks_r, METH_VARARGS, "..."},
    {"RandomWalk", DemNets_RandomWalk, METH_VARARGS, "..."},
    {"RemoveLinks", DemNets_RemoveLinks, METH_VARARGS, "..."},
    {"ReverseLinks", DemNets_ReverseLinks, METH_VARARGS, "..."},
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
