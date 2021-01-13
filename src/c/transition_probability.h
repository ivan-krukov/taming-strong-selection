// Access matrix (2D) memory
#ifndef XMEM
#define XMEM(M,i,j) (M)->data[(((i) * (M)->cols)) + (j)]
#endif

/* Matrix */
typedef struct  {
    double *data;
    unsigned int rows;
    unsigned int cols;
} Matrix;
Matrix* matrix_new(unsigned int rows, unsigned int cols);
double matrix_get(Matrix* self, unsigned int i, unsigned int j);
void matrix_set(Matrix* self, unsigned int i, unsigned int j, double v);
void matrix_fill(Matrix* self, double v);
void matrix_print(Matrix* self);
void matrix_del(Matrix* self);

/* Tile */
typedef struct {
    Matrix** data;
    unsigned int rows;
    unsigned int cols;
} Tile;
Tile* tile_new(unsigned int rows, unsigned int cols);
void tile_print(Tile* self);
void tile_del(Tile* self);

double basecases(int io, int no, int ic, int nc, int t);
double Pf(int io, int no, int ic, int nc, double s, int N, int t, int max_t, Tile** cache);
double P0(int io, int no, int ic, int nc, double s, int N, int max_t, Tile** cache);
