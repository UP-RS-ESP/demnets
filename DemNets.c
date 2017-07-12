#include "Python.h"
#include "numpy/arrayobject.h"
#include <fcntl.h>
#include <math.h>
#include <omp.h>
#include <sys/param.h>

#define VERSION "0.2"

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
UpstreamNetwork(const unsigned int *spx,
    const unsigned int m) {
    PyArrayObject *net;
    npy_intp dim[1];
    unsigned int i, j, k, l, o;
    unsigned int *lst[m], *e, ex, itr;
    // reverse facet flow network spx

    // alloc list of arrays
    for(i = 0; i < m; i++) {
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
        itr = i * 2;
        for(j = 0; j < 2; j++) {
            k = spx[itr + j];
            if(k == m)
                continue;
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
    dim[0] = l + m + 1;
    net = (PyArrayObject *) PyArray_ZEROS(1, dim, PyArray_UINT, 0);
    if(!net) {
        PyErr_SetString(PyExc_MemoryError, "...");
        return NULL;
    }
    e = (unsigned int *) net->data;

    // store in compressed row format
    j = m + 1;
    for(i = 0; i < m; i++) {
        e[i] = j;
        for(k = 2; k < lst[i][1]; k++) {
            e[j++] = lst[i][k];
        }
        free(lst[i]);
    }
    e[m] = j;
    return net;
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
    unsigned int i, j, k, l;
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
            l = spx[k];
            if(m > l)
                ideg[l]++;
        }
    }
    // start at facets without in-degree draining into l
    for(i = 0; i < m; i++) {
        if(!ideg[i]) {
            itr = i * 2;
            for(j = 0; j < 2; j++) {
                k = itr + j;
                l = spx[k];
                ltp[k] = spa[k];
                if(m > l) {
                    if(put(que, l, ltp[k])) {
                        PyErr_SetString(PyExc_MemoryError, "failed to fill queue ..");
                        exit(EXIT_FAILURE);
                    }
                }
            }
        }
    }
    // work the queue
    while(!get(que, &i, &ltpi)) {
        seen[i]++;
        itr = i * 2;
        ltp[itr] += ltpi;
        if(seen[i] == ideg[i]) {
            // we collected all input for node i
            ltpi = ltp[itr];
            ltp[itr] = 0;
            for(j = 0; j < 2; j++) {
                k = itr + j;
                l = spx[k];
                // link throughput
                ltp[k] = ltpi * spw[k] + spa[k];
                if(m > l) {
                    if(put(que, l, ltp[k])) {
                        PyErr_SetString(PyExc_MemoryError, "failed to fill queue ..");
                        exit(EXIT_FAILURE);
                    }
                }
            }
        }
	}
    //for(i = 0; i < m; i++) {
    //    ltp[i*2] = ideg[i] - seen[i];
    //    ltp[i*2+1] = 0;
    //}
}

void
FacetFlowNetwork(unsigned int *spx, double *spw, double *spa, double *spd, double *phi,
           const unsigned int *net,
           const unsigned int *tri,
           const unsigned int m,
           const double *x,
           const double *y,
           const double *z) {
    int sgn;
    double du, dv, dw, a, b, c;
    double xx, yy, slp, frc;
    double dx, dy, dz, dn, s, t;
    double xa, xb, xc, ya, yb, yc;
    double aa, ab, ac, bb, bc;
    double zf, zmin, phii, beta;
    unsigned int i, j, k, l;
    unsigned int u, v, w, q, p;
    unsigned int dest, f, g, h;
    unsigned int *seen;
    Queue *que;

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

    // fix sinks
#pragma omp parallel for private(i,j,k,l,p,q,u,du,zmin,zf,dest,g,f,h,seen,que,xx,yy,dx,dy) schedule(guided)
    for(i = 0; i < m; i++) {
        p = i * 2;
        for(j = 0; j < 2; j++) {
            q = p + j;
            l = spx[q];
            //l = m;
            if(l == m)
                continue;
            // check whether two neighboring facets flow into each other
            if(spx[l*2] == i || spx[l*2+1] == i) {
                // get lowest node of these two facets
                que = malloc(sizeof(Queue));
                seen = calloc(m, sizeof(unsigned int));
                if(!que || !seen) {
                    PyErr_SetString(PyExc_MemoryError, "...");
                    exit(EXIT_FAILURE);
                }
                que->first = que->last = NULL;
                u = NodeOfSimplicies(tri, i, l, z);
                zmin = z[u];
                dest = m;
                for(k = net[u]; k < net[u+1]; k++) {
                    g = net[k];
                    seen[g]++;
                    if(put(que, g, 0)) {
                        PyErr_SetString(PyExc_MemoryError, "failed to fill queue ..");
                        exit(EXIT_FAILURE);
                    }
                }
                xx = (x[tri[i*3]]+x[tri[i*3+1]]+x[tri[i*3+2]]) / 3.;
                yy = (y[tri[i*3]]+y[tri[i*3+1]]+y[tri[i*3+2]]) / 3.;
                while(!get(que, &f, &du) && dest == m) {
                    // don't look forever
                    if(du > 50)
                        break;
                    // don't tube to where we came from
                    if(f != i && f != l) {
                        // zf = z of facet, the highest point of the current facet
                        zf = z[tri[f*3]];
                        if(z[tri[f*3+1]] > zf) zf = z[tri[f*3+1]];
                        if(z[tri[f*3+2]] > zf) zf = z[tri[f*3+2]];
                        /*
                        flow = 0;
                        if(du) {
                            if(zf < zmin)
                               flow = 1;
                        } else {
                            if(zf <= zmin)
                                flow = 1;
                        }
                        if(flow) {
                        */
                        if(zf < zmin) {
                            if(spx[f*2] != i && spx[f*2] != l && spx[f*2+1] != i && spx[f*2+1] != l) {
                                dx = xx - (x[tri[f*3]]+x[tri[f*3+1]]+x[tri[f*3+2]]) / 3.;
                                dy = yy - (y[tri[f*3]]+y[tri[f*3+1]]+y[tri[f*3+2]]) / 3.;
                                if(sqrt(dx*dx+dy*dy) < 50) {
                                    zmin = zf;
                                    dest = f;
                                    break;
                                }
                            }
                        }
                    }
                    for(h = 0; h < 3; h++) {
                        u = tri[f*3 + h];
                        for(k = net[u]; k < net[u+1]; k++) {
                            g = net[k];
                            if(seen[g])
                                continue;
                            seen[g]++;
                            if(put(que, g, du+1)) {
                                PyErr_SetString(PyExc_MemoryError, "failed to fill queue ..");
                                exit(EXIT_FAILURE);
                            }
                        }
                    }
                }
                //fprintf(stderr, "\n");
                while(!get(que, &f, &du));
                free(seen);
                free(que);
                //if(dest == m) fprintf(stderr, "sink at %i %.3f %.0f\n", i, zmin, du);
                // drain into the facet dest (i->dest)
                spx[q] = dest;
                // rewire also the other facet to that lower facet (l->dest)
                for(h = 0; h < 2; h++)
                    if(spx[l*2+h] == i)
                        spx[l*2+h] = dest;
            }
        }
    }

    // remove very long tubes ..
#pragma omp parallel for private(i,j,p,q,l,xx,yy,dx,dy)
    for(i = 0; i < m; i++) {
        xx = (x[tri[i*3]]+x[tri[i*3+1]]+x[tri[i*3+2]]) / 3.;
        yy = (y[tri[i*3]]+y[tri[i*3+1]]+y[tri[i*3+2]]) / 3.;
        p = i * 2;
        for(j = 0; j < 2; j++) {
            q = p + j;
            l = spx[q];
            if(l == m)
                continue;
            dx = xx - (x[tri[l*3]]+x[tri[l*3+1]]+x[tri[l*3+2]]) / 3.;
            dy = yy - (y[tri[l*3]]+y[tri[l*3+1]]+y[tri[l*3+2]]) / 3.;
            if(sqrt(dx*dx+dy*dy) > 50)
                spx[q] = m;
        }
    }

    // fix spw and spa
#pragma omp parallel for private(i,p)
    for(i = 0; i < m; i++) {
        p = i * 2;
        if(spx[p] == m && spx[p+1] < m) {
            spa[p+1] += spa[p];
            spw[p+1] = 1;
            spw[p] = 0;
        } else if(spx[p] < m && spx[p+1] == m) {
            spa[p] += spa[p+1];
            spw[p] = 1;
            spw[p+1] = 0;
        }
    }
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
DemNets_FacetUpstreamNetwork(PyObject *self, PyObject* args) {
    PyObject *spxarg;
    PyArrayObject *spx, *net;

    // parse input
    if(!PyArg_ParseTuple(args, "O", &spxarg))
        return NULL;
    spx = (PyArrayObject *) PyArray_ContiguousFromObject(spxarg, PyArray_UINT, 2, 2);
    if(!spx)
        return NULL;

    // reverse the flow network
    net = UpstreamNetwork((unsigned int *)spx->data, spx->dimensions[0]);
    Py_DECREF(spx);
    return PyArray_Return(net);
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
                        (unsigned int *)spx->data,
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

static PyMethodDef DemNets_Methods[] = {
    {"Simplicies", DemNets_Simplicies, METH_VARARGS, "..."},
    {"FacetUpstreamNetwork", DemNets_FacetUpstreamNetwork, METH_VARARGS, "..."},
    {"FacetFlowNetwork", DemNets_FacetFlowNetwork, METH_VARARGS, "..."},
    {"FacetFlowThroughput", DemNets_FacetFlowThroughput, METH_VARARGS, "..."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef ModDef = {
    PyModuleDef_HEAD_INIT,
    "DemNets",
    NULL,
    -1,
    DemNets_Methods
};

PyMODINIT_FUNC
PyInit_DemNets(void) {
    PyObject *mod;

    mod = PyModule_Create(&ModDef);
    PyModule_AddStringConstant(mod, "__author__", "Aljoscha Rheinwalt <aljoscha.rheinwalt@uni-potsdam.de>");
    PyModule_AddStringConstant(mod, "__version__", VERSION);
    import_array();

    return mod;
}

int
main(int argc, char **argv) {
    wchar_t pname[255];

    PyImport_AppendInittab("DemNets", PyInit_DemNets);
    mbstowcs(pname, argv[0], strlen(argv[0])+1);
    Py_SetProgramName(pname);
    Py_Initialize();
    PyImport_ImportModule("DemNets");
    PyMem_RawFree(argv[0]);
    return 0;
}
