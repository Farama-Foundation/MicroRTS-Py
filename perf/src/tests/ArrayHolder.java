package tests;

public class ArrayHolder {

    final int dim1, dim2, dim3, dim4;
    public final int[][][][] data;

    public ArrayHolder(int dim1, int dim2, int dim3, int dim4) {
        this.dim1 = dim1;
        this.dim2 = dim2;
        this.dim3 = dim3;
        this.dim4 = dim4;
        data = new int[dim1][dim2][dim3][dim4];
    }

    public void fillIn(int value) {
        for (int i1 = 0; i1 < dim1; i1++) {
            for (int i2 = 0; i2 < dim2; i2++) {
                for (int i3 = 0; i3 < dim3; i3++) {
                    for (int i4 = 0; i4 < dim4; i4++) {
                        data[i1][i2][i3][i4] = value;
                    }
                }
            }
        }
    }

}