cdef extern from "c/transition_probability.h":
    ctypedef struct Matrix:
        double *data
        unsigned int rows
        unsigned int cols

    ctypedef struct Tile:
        Matrix** data
        unsigned int rows
        unsigned int cols

    Tile* tile_new(int rows, int cols)
    Matrix* tile_get(Tile* self, int rows, int cols)
    void tile_del(Tile* self)
    double P0(int io, int no, int ic, int nc, double s, int N, int max_t, Tile** cache)
