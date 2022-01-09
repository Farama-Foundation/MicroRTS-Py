package tests;

import java.nio.IntBuffer;

public class ArrayPerf {

    public static final long sumOfArray(int[][][] x) {
        long total = 0;
        for (int[][] x1: x) {
            for (int[] x2: x1) {
                for (int x3: x2) {
                    total += x3;
                }
            }
        }
        return total;
    }

    public static final long sumOfFlatArray(int[] x) {
        long total = 0;
        for (int elem: x) {
            total += elem;
        }
        return total;
    }

    public static final long sumOfBuffer(IntBuffer buffer) {
        long total = 0;
        while (buffer.hasRemaining()) {
            total += buffer.get();
        }
        return total;
    }

    public static final void fillBuffer(IntBuffer buffer, int ntimes) {
        int cursor = 0;
        while (ntimes > cursor) {
            buffer.put(cursor, cursor % 64);
            cursor += 1;
        }
        buffer.rewind();
    }

    public static final int[] fillArray(int ntimes) {
        final int[] buffer = new int[ntimes];
        int cursor = 0;
        while (ntimes > cursor) {
            buffer[cursor] = cursor % 64;
            cursor += 1;
        }
        return buffer;
    }

}