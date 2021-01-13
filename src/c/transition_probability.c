#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <stdio.h>

#include "transition_probability.h"

/* Matrix */

Matrix* matrix_new(unsigned int rows, unsigned int cols) {
    Matrix* self = malloc(sizeof(Matrix));
    self->data = calloc(rows * cols, sizeof(double*));
    self->rows = rows;
    self->cols = cols;
    return self;
}

double matrix_get(Matrix* self, unsigned int i, unsigned int j) {
    return self->data[(i * self->cols) + j];
}

void matrix_set(Matrix* self, unsigned int i, unsigned int j, double v) {
    self->data[(i * self->cols) + j] = v;
}

void matrix_fill(Matrix* self, double v) {
    unsigned int i, j;
    for (i = 0; i < self->rows; i++) {
        for (j = 0; j < self->cols; j++) {
            XMEM(self, i, j) = v;
        }
    }
} 

void matrix_print(Matrix* self) {
    unsigned int i, j;
    printf("[");
    for (i = 0; i < self->rows; i++) {
        for (j = 0; j < self->cols; j++) {
            printf("%.3e ", XMEM(self, i, j));
        }
        if (i == self->rows - 1) {
            printf("]\n");
        } else {
            printf("\n");
        }
    }
}

void matrix_del(Matrix* self) {
    free(self->data);
    free(self);
}

/* Tile */

Tile* tile_new(unsigned int rows, unsigned int cols) {
    Tile* self = malloc(sizeof(Tile));
    self->data = malloc(rows * cols * sizeof(Matrix*));
    self->rows = rows;
    self->cols = cols;

    unsigned int i, j;
    for (i = 0; i < self->rows; i++) {
        for (j = 0; j < self->cols; j++) {
            Matrix* M = matrix_new(i+1, j+1);
            matrix_fill(M, NAN);
            XMEM(self, i, j) = M;
        }
    }

    return self;
}

void tile_print(Tile* self) {
    unsigned int i, j;
    for (i = 0; i < self->rows; i++) {
        for (j = 0; j < self->cols; j++) {
            matrix_print(XMEM(self, i, j));
        }
    }
}

void tile_del(Tile* self) {

    unsigned int i, j;
    for (i = 0; i < self->rows; i++) {
        for (j = 0; j < self->cols; j++) {
            free(XMEM(self, i, j));
        }
    }

    free(self->data);
    free(self);
}

double basecases(int io, int no, int ic, int nc, int t) {
    if ((no < 0) || (nc < 0) || (io < 0) || (ic < 0) || (io > no) || (ic > nc) || 
            ((no > 0) && (nc == 0)) || ((io > 0) && (ic == 0)) || ((io < no) && (ic == nc))) {
        return 0;
    } else if ((io == 0) && (no == 0) && (ic == 0) && (nc == 0)) {
        if (t <= 0) {
            return 1;
        } else {
            return 0;
        }
    } else {
        return NAN;
    }
}


double Pf(int io, int no, int ic, int nc, double s, int N, int t, int max_t, Tile** cache) {
    double v = basecases(io, no, ic, nc, t);
    if (isnan(v)) {
        v = XMEM(XMEM(cache[t], nc, no), ic, io);
        if(isnan(v)) {
            if (t <= 0) {
                v = P0(io, no, ic, nc, s, N, max_t, cache);
            } else {
                double dnc = (double)nc;
                double dic = (double)ic;
                double dN = (double)N;
                double oos = 1.0-((dnc-1.0)/dN);
                double b = oos * (dic / dnc) * s * Pf(io, no, ic-1, nc-1, s, N, t-1, max_t, cache);
                double c = (dic/dN) * s * Pf(io, no, ic, nc, s, N, t-1, max_t, cache);
                v = b + c;
            }
        }
    }
    if ((nc >= 0) && (no >= 0) && (ic >= 0) && (io >= 0) && (ic <= nc) && (io <= no) && (t >= 0)) {
        XMEM(XMEM(cache[t], nc, no), ic, io) = v;
    }
    return v;
}

double P0(int io, int no, int ic, int nc, double s, int N, int max_t, Tile** cache) {
    double v = basecases(io, no, ic, nc, max_t);
    if (isnan(v)) {
        v = XMEM(XMEM(cache[0], nc, no), ic, io);
        if(isnan(v)) {
            double dnc = (double)nc;
            double dic = (double)ic;
            double dN = (double)N;
            double oos = 1.0-((dnc-1.0)/dN);
            double sel = 1.0-s;
            int t;

            double af = 0;
            for (t = max_t; t >= 0; t--) {
                af += Pf(io, no-1, ic, nc-1, s, N, t, max_t, cache);
            }
            double a = oos * ((dnc-dic)/dnc) * af;

            double bf = 0;
            for (t = max_t; t >= 0; t--) {
                bf += Pf(io-1, no-1, ic-1, nc-1, s, N, t, max_t, cache);
            }
            double b = oos * (dic/dnc) * sel * bf;

            double cf = 0;
            for (t = max_t; t >= 0; t--) {
                cf += Pf(io-1, no-1, ic, nc, s, N, t, max_t, cache);
            }
            double c = (dic/dN) * sel * cf;

            double df = 0;
            for (t = max_t; t >= 0; t--) {
                df += Pf(io, no-1, ic, nc, s, N, t, max_t, cache);
            }
            double d = ((dnc-dic)/dN) * df;

            v = a + b + c + d;
            v += (oos*(dic/dnc) * s * Pf(io-1, no-1, ic-1, nc-1, s, N, max_t, max_t, cache)) + (
                (dic/dN) * s * Pf(io-1, no-1, ic, nc, s, N, max_t, max_t, cache));
        }
    }
    if ((nc >= 0) && (no >= 0) && (ic >= 0) && (io >= 0) && (ic <= nc) && (io <= no)) {
        /* printf("%e ", v); */
        XMEM(XMEM(cache[0], nc, no), ic, io) = v;
    }
    return v;
}

