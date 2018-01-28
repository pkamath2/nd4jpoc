package com.pk.nd4j;

public class MNistBO {

    private int num_of_samples;

    private int[] data_arr;

    public int getNum_of_samples() {
        return num_of_samples;
    }

    public void setNum_of_samples(int num_of_samples) {
        this.num_of_samples = num_of_samples;
    }

    public int[] getData_arr() {
        return data_arr;
    }

    public void setData_arr(int[] data_arr) {
        this.data_arr = data_arr;
    }
}
