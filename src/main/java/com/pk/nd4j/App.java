package com.pk.nd4j;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.Arrays;

import static org.nd4j.linalg.ops.transforms.Transforms.sigmoid;

public class App 
{
    public static void main( String[] args )
    {
        /*INDArray sigmoid = sigmoid(Nd4j.create(new double[]{3}));
        System.out.println(sigmoid);

        INDArray numArray = Nd4j.create(new double[]{1,2,3});
        INDArray numArraySigmoid = sigmoid(numArray);
        System.out.println("Num Array = " + numArray);
        System.out.println("Sigmoid of Num Array = " + numArraySigmoid);

        //System.out.println(numArray.add(3));

        INDArray sigmoid_derivative = numArraySigmoid.mul(numArraySigmoid.mul(-1).add(1));
        System.out.println("Derivative of Sigmoid = " + sigmoid_derivative);

        *//*
        * Shapre & Reshape
         *//*
        System.out.println(Arrays.toString(numArraySigmoid.shape()));

        INDArray numArrReshaped = numArray.reshape(3,1);
        System.out.println("Num Arr reshaped = " + Arrays.toString(numArrReshaped.shape()));

        double[] imageArrDouble = new double[]{
                0.67826139,  0.29380381,
                0.90714982,  0.52835647,
                0.4215251 ,  0.45017551,
                0.92814219,  0.96677647,
                0.85304703,  0.52351845,
                0.19981397,  0.27417313,
                0.60659855,  0.00533165,
                0.10820313,  0.49978937,
                0.34144279,  0.94630077
        };

        INDArray imageArr = Nd4j.create(imageArrDouble, new int[]{3,3,2});
        System.out.println(imageArr);
        System.out.println("***");

        System.out.println(imageArr.reshape(18,1));


        *//*
            Normalization
         *//*
        System.out.println("***");
        INDArray unNormalizedArr = Nd4j.create(new double[][]{{0,3,4},{2,6,4}});
        System.out.println(unNormalizedArr);
        System.out.println("Normalized : " + Transforms.normalizeZeroMeanAndUnitVariance(unNormalizedArr));


        System.out.println("***");
        INDArray unNormalizedArr1 = Nd4j.create(new double[]{0,3,4});
        System.out.println(unNormalizedArr1);
        System.out.println("Normalized : " + Transforms.normalizeZeroMeanAndUnitVariance(unNormalizedArr1));*/

        //System.out.println(convertNumLabelsToSoftmaxLabels(Nd4j.create(new double[]{5,0,7})));

        INDArray arr = Nd4j.create(new double[]{7,2,-1,1,8,3,4,2});
        System.out.println(arr);
        arr = arr.reshape(2,4);
        System.out.println(arr);
        arr = arr.transpose();
        System.out.println(arr);
        System.out.println(Transforms.softmax(arr));


    }

    private static INDArray convertNumLabelsToSoftmaxLabels(INDArray labels){//Labels is 1Xm array

        INDArray label_arr = Nd4j.create(labels.shape()[1],10);
        INDArray label_lookup= Nd4j.eye(10);

        for(int i=0;i<labels.shape()[1];i++){
            INDArray label = label_lookup.getRow(labels.getInt(i));
            label_arr.putRow(i, label);

        }
        return label_arr.transpose();
    }


}
