package com.pk.nd4j;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Tanh;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.io.*;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

import static com.pk.nd4j.MNistReader.*;

@SuppressWarnings("Duplicates")
public class TwoLayerLogisticRegression {


    private static final String MNIST_LOCATION_TRAIN_LABELS = "/Users/purnimakamath/appdir/datasets/MNIST/train-labels-idx1-ubyte";
    private static final String MNIST_LOCATION_TRAIN_IMAGES = "/Users/purnimakamath/appdir/datasets/MNIST/train-images-idx3-ubyte";
    private static final String MNIST_LOCATION_TEST_LABELS = "/Users/purnimakamath/appdir/datasets/MNIST/t10k-labels-idx1-ubyte";
    private static final String MNIST_LOCATION_TEST_IMAGES = "/Users/purnimakamath/appdir/datasets/MNIST/t10k-images-idx3-ubyte";

    private static int num_of_train_samples;
    private static int num_of_test_samples;

    private static int[] train_labels = null;
    private static int[] test_labels = null;

    private static int[] train_images = null;
    private static int[] test_images = null;

    private static INDArray train_images_flatten = null;
    private static INDArray test_images_flatten = null;

    private static INDArray train_labels_flatten = null;
    private static INDArray test_labels_flatten = null;

    private static INDArray train_images_normalized = null;
    private static INDArray test_images_normalized = null;

    private static INDArray train_softmax_labels = null;

    private static INDArray weights = null;
    private static double bias = 0;

    static boolean regression_or_predict = true;

    static int whichImage = 19000;

    public static void main(String[] args) {

        MNistBO mNistBO = createLabelArray(MNIST_LOCATION_TRAIN_LABELS);
        train_labels = mNistBO.getData_arr();
        num_of_train_samples = mNistBO.getNum_of_samples();
        mNistBO = createImageArray(MNIST_LOCATION_TRAIN_IMAGES);
        train_images = mNistBO.getData_arr();

        mNistBO = createLabelArray(MNIST_LOCATION_TEST_LABELS);
        test_labels = mNistBO.getData_arr();
        num_of_test_samples = mNistBO.getNum_of_samples();
        mNistBO = createImageArray(MNIST_LOCATION_TEST_IMAGES);
        test_images = mNistBO.getData_arr();


        //For Test Images
        test_images_flatten = createImageNDArray(test_images, num_of_test_samples);
        test_labels_flatten = Nd4j.create(Arrays.stream(test_labels).asDoubleStream().toArray());

        train_images_flatten = createImageNDArray(train_images, num_of_train_samples);
        train_labels_flatten = Nd4j.create(Arrays.stream(train_labels).asDoubleStream().toArray());

        System.out.println("Shape of test images => "+ Arrays.toString(test_images_flatten.shape()));
        System.out.println("Shape of test labels => "+ Arrays.toString(test_labels_flatten.shape()));
        System.out.println("Shape of train images => "+ Arrays.toString(train_images_flatten.shape()));
        System.out.println("Shape of train labels => "+ Arrays.toString(train_labels_flatten.shape()));



        //

        test_images_normalized = test_images_flatten.div(255);//255
        train_images_normalized = train_images_flatten.div(255);//255

        //System.out.println(train_images_normalized.getColumn(9000));


        if(regression_or_predict) {
            /*//Initialize weights & bias
            weights = Nd4j.rand(new int[]{28 * 28, 10});
            //System.out.println("Initial Weights :: "+weights);
            //weights = Nd4j.zeros(new int[]{28*28, 10});
            bias = 0;*/


            //originalArray.get(NDArrayIndex.all(), NDArrayIndex.interval(0,3))
            HashMap<String, Object> params = forward_propogate(train_images_normalized, train_labels_flatten, num_of_train_samples, 2, 1.2);
            //HashMap<String, Object> params = forward_propogate(train_images_normalized.get(NDArrayIndex.all(), NDArrayIndex.interval(0,10000)), train_labels_flatten.get(NDArrayIndex.all(), NDArrayIndex.interval(0,10000)), 20, 100, 0.01);

            try (FileOutputStream fos_w1 = new FileOutputStream("W1");
                 FileOutputStream fos_w2 = new FileOutputStream("W2");
                 FileOutputStream fos_b1 = new FileOutputStream("b1");
                 FileOutputStream fos_b2 = new FileOutputStream("b2");) {
                Nd4j.write(fos_w1, (INDArray) params.get("W1"));
                Nd4j.write(fos_w2, (INDArray) params.get("W2"));
                Nd4j.write(fos_b1, (INDArray) params.get("b1"));
                Nd4j.write(fos_b2, (INDArray) params.get("b2"));
            } catch (IOException ioe) {
                ioe.printStackTrace();
            }

        }else{

            drawImage(train_images_flatten.getColumn(whichImage), train_labels_flatten.getInt(whichImage), whichImage);

            try (FileInputStream fis_w1 = new FileInputStream("W1");
                 FileInputStream fis_w2 = new FileInputStream("W2");
                 FileInputStream fis_b1 = new FileInputStream("b1");
                 FileInputStream fis_b2 = new FileInputStream("b2");){
                INDArray w1s = Nd4j.read(fis_w1);
                INDArray w2s = Nd4j.read(fis_w2);
                INDArray b1s = Nd4j.read(fis_b1);
                INDArray b2s = Nd4j.read(fis_b2);

                System.out.println("Predicting: " + train_labels_flatten.getColumn(whichImage));
                predict(train_images_normalized.getColumn(whichImage), w1s, w2s, b1s, b2s);

                System.out.println("********************************");
                System.out.println("********************************");
                System.out.println("********************************");

                System.out.println("Calculating accuracy");

                calculate_accuracy(test_images_normalized, test_labels_flatten, w1s,w2s,b1s,b2s,num_of_test_samples);
            } catch (IOException ioe) {
                ioe.printStackTrace();
            }


        }


    }

    private static HashMap forward_propogate(INDArray images, INDArray labels, int num_samples, int epoch, double learning_rate){

        System.out.println("*****&&&&&&&&&&&******");
        labels = convertNumLabelsToSoftmaxLabels(labels);
//        Nd4j.copy(labels, train_softmax_labels);
        HashMap<String, Object> params = new HashMap<>();

        //images: 784 X m
        //labels: 10 X m

        //W1: 1 X 784
        //b1: 1 X 1
        //Z1: 1 X m
        //A1: 1 X m

        //W2: 10 X 1
        //b2: 10 X 1
        //Z2: 10 X m
        //A2: 10 X m
        INDArray W1 = Nd4j.rand(new int[]{1, 28 * 28});
        INDArray b1 = Nd4j.zeros(1, 1);

        INDArray W2 = Nd4j.rand(new int[]{10, 1});
        INDArray b2 = Nd4j.zeros(10, 1);

        System.out.println("W1: "+W1);
        System.out.println("W2: "+W2);
        System.out.println("b1: "+b1);
        System.out.println("b1: "+b2);

        List<Double> costList = new ArrayList<>();
        while(epoch > 0) {
            INDArray Z1 = W1.mmul(images).add(b1);
            INDArray A1 = Transforms.hardTanh(Z1.transpose());
            A1 = A1.transpose();

            System.out.println("Z1 Shape => " + Arrays.toString(Z1.shape()));
            System.out.println("A1 Shape => " + Arrays.toString(A1.shape()));

            INDArray Z2 = W2.mmul(A1).addColumnVector(b2);
            INDArray A2 = Transforms.softmax(Z2.transpose());
            A2 = A2.transpose();

            System.out.println("Z2 Shape => " + Arrays.toString(Z2.shape()));
            System.out.println("A2 Shape => " + Arrays.toString(A2.shape()));

            double cost = (-1) * Nd4j.mean(Transforms.log(A2).mul(labels), 0).getDouble(0);
            costList.add(cost);
            INDArray dZ2 = A2.sub(labels);
            INDArray dW2 = (dZ2.mmul(A1.transpose())).mul(1 / num_samples);
            //INDArray db2 = Nd4j.create(new double[] {dZ2.sumNumber().doubleValue()/num_samples});
            INDArray db2 = Nd4j.mean(dZ2, 1);

//            System.out.println("dZ2 Shape => " + Arrays.toString(dZ2.shape()));
//            System.out.println("dW2 Shape => " + Arrays.toString(dW2.shape()));
//            System.out.println("db2 Shape => " + Arrays.toString(db2.shape()));

            INDArray dZ1 = W2.transpose().mmul(dZ2).mul(Transforms.pow(A1, 2).mul(-1).add(1));
            INDArray dW1 = dZ1.mmul(images.transpose()).mul(1 / num_samples);
            INDArray db1 = Nd4j.mean(dZ1, 1);

//            System.out.println("dZ1 Shape => " + Arrays.toString(dZ1.shape()));
//            System.out.println("dW1 Shape => " + Arrays.toString(dW1.shape()));
//            System.out.println("db1 Shape => " + Arrays.toString(db1.shape()));
            W1 = W1.sub(dW1.mul(learning_rate));
            b1 = b1.sub(db1.mul(learning_rate));

            W2 = W2.sub(dW2.mul(learning_rate));
            b2 = b2.sub(db2.mul(learning_rate));


//            System.out.println("W1 Shape => " + Arrays.toString(W1.shape()));
//            System.out.println("b1 Shape => " + Arrays.toString(b1.shape()));
//            System.out.println("W2 Shape => " + Arrays.toString(W2.shape()));
//            System.out.println("b2 Shape => " + Arrays.toString(b2.shape()));

            System.out.println("Cost after epoch: " + epoch+" = "+cost);
            epoch--;
        }


        System.out.println(costList);
        params.put("W1", W1);
        params.put("b1",b1);
        params.put("W2", W2);
        params.put("b2",b2);
        return params;

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

    public static void predict(INDArray input, INDArray W1, INDArray W2, INDArray b1, INDArray b2){

        INDArray labels = convertNumLabelsToSoftmaxLabels(train_labels_flatten);
        System.out.println(labels.getColumn(whichImage));
        INDArray Z1 = W1.mmul(input).add(b1);
        INDArray A1 = Transforms.tanh(Z1.transpose());
        A1 = A1.transpose();


        System.out.println("A1: "+A1);
        System.out.println(Arrays.toString(A1.shape()));
        INDArray Z2 = W2.mmul(A1).addColumnVector(b2);
        System.out.println("Z2: "+Z2);
        INDArray A2 = Transforms.softmax((Z2.mul(100)).transpose());
        A2 = A2.transpose();
        System.out.println(Arrays.toString(A2.shape()));
        System.out.println("Prediction:: "+A2);

    }

    //print ('Accuracy: %d' % float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100) + '%')

    private static void calculate_accuracy(INDArray input, INDArray labels, INDArray W1, INDArray W2, INDArray b1, INDArray b2, int num_of_samples){

        System.out.println("W1: "+W1);
        System.out.println("W2: "+W2);
        System.out.println("b1: "+b1);
        System.out.println("b1: "+b2);

        INDArray Z1 = W1.mmul(input).add(b1);
        INDArray A1 = Transforms.tanh(Z1.transpose());
        A1 = A1.transpose();


        System.out.println("A1: "+A1);
        System.out.println(Arrays.toString(A1.shape()));
        INDArray Z2 = W2.mmul(A1).addColumnVector(b2);
        System.out.println("Z2: "+Z2);
        //INDArray A2 = Transforms.softmax((Z2.mul(100)).transpose());
        INDArray A2 = Transforms.softmax(Z2.transpose());
        A2 = A2.transpose();
        System.out.println("A2:: "+A2);


        INDArray first_part = labels.mmul(A2.transpose());
        INDArray second_part = (labels.mul(-1).add(1)).mmul((A2.mul(-1).add(1)).transpose());

        INDArray total = (first_part.sum(second_part)).div(num_of_samples);

        System.out.println("Accuracy: "+total);
    }

    /*
    W1: [0.63,  0.47,  0.08,  0.86,  0.08,  0.67,  0.97,  0.43,  0.30,  0.93,  0.08,  0.45,  0.13,  0.02,  0.13,  0.32,  0.18,  0.22,  0.36,  0.08,  0.58,  1.00,  0.30,  0.48,  0.82,  0.02,  0.50,  0.75,  0.33,  0.24,  0.19,  0.58,  0.80,  0.91,  0.64,  0.24,  0.02,  0.91,  0.52,  0.52,  0.62,  0.37,  0.36,  0.21,  0.58,  0.73,  0.85,  0.81,  0.87,  0.71,  0.44,  0.47,  0.46,  0.79,  0.09,  0.79,  0.71,  0.08,  0.95,  0.39,  0.67,  0.49,  0.68,  0.64,  0.33,  0.36,  0.44,  0.64,  0.44,  0.09,  0.44,  0.85,  0.12,  0.72,  0.59,  0.04,  0.99,  0.02,  1.00,  0.09,  0.69,  0.32,  0.47,  0.54,  0.79,  0.46,  0.32,  0.39,  0.33,  0.65,  0.04,  0.43,  0.99,  0.45,  0.75,  0.86,  0.64,  0.92,  0.40,  0.51,  0.79,  0.79,  0.46,  0.36,  0.36,  0.00,  0.58,  0.23,  0.09,  0.93,  0.00,  0.65,  0.43,  0.19,  0.50,  0.33,  0.09,  0.59,  0.47,  0.17,  0.71,  0.57,  0.77,  0.67,  0.49,  0.42,  0.91,  0.77,  0.48,  0.86,  0.17,  0.25,  0.81,  0.73,  0.16,  0.59,  0.92,  0.06,  0.52,  0.60,  0.17,  0.94,  0.04,  0.03,  0.55,  0.98,  0.56,  0.68,  0.45,  0.84,  0.78,  0.56,  0.58,  0.48,  0.09,  0.61,  0.81,  0.82,  0.44,  0.10,  0.67,  0.47,  0.67,  0.95,  0.01,  0.86,  0.49,  0.30,  0.76,  0.43,  0.55,  0.50,  0.95,  0.90,  0.25,  0.14,  0.78,  0.09,  0.88,  0.43,  0.91,  0.27,  0.48,  0.30,  0.13,  0.43,  0.43,  0.27,  0.07,  0.04,  0.84,  0.06,  0.10,  0.85,  0.45,  0.83,  0.59,  0.46,  0.99,  0.60,  0.46,  0.99,  0.81,  0.91,  0.52,  0.55,  0.88,  0.59,  1.00,  0.76,  0.81,  0.16,  0.44,  0.47,  0.38,  0.79,  0.89,  0.31,  0.60,  0.52,  0.91,  0.86,  0.50,  0.41,  0.27,  0.04,  0.50,  0.82,  0.93,  0.72,  0.76,  0.07,  0.21,  0.61,  0.89,  0.61,  0.07,  0.42,  0.01,  0.40,  0.38,  0.11,  0.75,  0.86,  0.01,  0.21,  0.80,  0.36,  0.58,  0.07,  0.99,  0.50,  0.90,  0.14,  0.24,  0.88,  0.47,  0.55,  0.59,  0.09,  0.19,  0.93,  0.24,  0.37,  0.96,  0.46,  0.82,  0.19,  0.72,  0.12,  0.19,  0.08,  1.00,  0.46,  0.18,  0.08,  0.53,  0.15,  0.70,  0.21,  0.58,  0.93,  0.84,  0.15,  0.35,  0.83,  0.20,  0.41,  0.42,  0.43,  0.65,  0.89,  0.69,  0.63,  0.62,  0.11,  0.07,  0.06,  0.91,  0.21,  0.23,  0.30,  0.45,  0.66,  0.36,  0.20,  0.81,  0.94,  0.20,  0.80,  0.40,  0.47,  0.46,  0.29,  0.16,  0.72,  0.71,  0.10,  0.50,  0.06,  0.83,  0.92,  0.59,  0.04,  0.29,  0.94,  0.49,  0.68,  0.46,  0.60,  0.19,  0.20,  0.28,  0.64,  0.30,  0.70,  0.42,  0.37,  0.90,  0.40,  0.34,  0.78,  0.19,  0.33,  0.89,  0.42,  0.22,  0.28,  0.52,  0.23,  0.41,  0.19,  0.71,  0.87,  0.80,  0.23,  0.50,  0.10,  0.46,  0.11,  0.18,  0.82,  0.07,  0.19,  0.32,  0.13,  0.56,  0.56,  0.02,  0.49,  0.64,  0.22,  0.53,  0.58,  0.76,  0.68,  0.48,  0.34,  0.49,  0.36,  0.09,  0.71,  0.57,  0.69,  0.22,  0.03,  0.33,  0.53,  0.31,  0.50,  0.89,  0.66,  0.30,  0.07,  0.57,  0.47,  0.26,  0.65,  0.23,  0.85,  0.25,  0.18,  0.56,  0.24,  0.53,  0.90,  0.01,  0.29,  0.45,  0.41,  0.93,  0.34,  0.28,  0.30,  0.15,  0.27,  0.74,  0.55,  0.88,  0.14,  0.11,  0.53,  0.22,  0.85,  0.63,  0.01,  0.12,  0.87,  0.70,  0.70,  0.51,  0.32,  0.10,  0.34,  0.15,  0.45,  0.71,  0.68,  0.94,  0.14,  0.68,  0.05,  0.07,  0.66,  0.33,  0.21,  0.77,  0.92,  0.96,  0.42,  0.50,  0.10,  0.55,  0.19,  0.63,  0.16,  0.87,  0.48,  0.33,  0.67,  0.39,  0.22,  0.20,  0.74,  0.19,  0.18,  0.60,  0.09,  0.39,  0.78,  0.03,  0.93,  0.81,  0.77,  0.38,  0.62,  0.42,  0.21,  0.57,  0.23,  0.25,  0.17,  0.53,  0.69,  0.77,  0.67,  0.70,  0.86,  0.53,  0.57,  0.27,  0.42,  0.08,  0.38,  0.98,  0.33,  0.61,  0.92,  0.98,  0.18,  0.95,  0.22,  0.92,  0.98,  0.47,  0.68,  0.37,  0.72,  0.03,  0.33,  0.67,  0.56,  0.21,  0.31,  0.35,  0.19,  0.04,  0.90,  0.72,  0.63,  0.95,  0.59,  0.07,  0.03,  0.02,  0.38,  0.47,  0.53,  0.34,  0.12,  0.20,  0.26,  0.94,  0.04,  0.98,  0.65,  0.73,  0.59,  0.96,  0.43,  0.55,  0.25,  0.85,  0.33,  0.67,  0.07,  0.88,  0.58,  0.60,  0.68,  0.42,  0.54,  0.59,  0.03,  0.25,  0.89,  0.29,  0.88,  0.92,  0.30,  0.64,  0.72,  0.12,  0.82,  0.42,  0.17,  0.46,  0.39,  0.72,  0.67,  0.97,  0.41,  0.73,  0.14,  0.68,  0.67,  0.92,  0.16,  0.85,  0.60,  0.49,  0.82,  0.82,  0.38,  0.59,  0.27,  0.45,  0.53,  0.74,  0.45,  0.52,  0.95,  0.33,  0.37,  0.39,  0.91,  0.23,  0.59,  0.39,  0.98,  0.48,  0.94,  0.69,  0.84,  0.09,  0.29,  0.97,  0.31,  0.51,  0.99,  0.62,  0.09,  0.67,  0.52,  0.78,  0.63,  0.31,  0.73,  0.80,  0.95,  0.78,  0.70,  0.53,  0.72,  0.78,  0.76,  0.63,  0.94,  0.97,  0.83,  0.27,  0.04,  0.94,  0.94,  0.38,  0.10,  0.75,  0.87,  0.66,  0.08,  0.48,  0.76,  0.82,  0.78,  0.51,  0.82,  0.11,  0.98,  0.72,  0.21,  0.41,  0.51,  0.33,  0.14,  0.10,  0.07,  0.54,  0.38,  0.81,  0.28,  0.23,  0.13,  0.17,  0.46,  0.30,  0.38,  0.35,  0.43,  0.44,  0.66,  0.86,  0.24,  0.92,  0.44,  0.19,  0.96,  0.06,  0.31,  0.66,  0.44,  0.36,  0.07,  0.44,  0.44,  0.23,  0.57,  0.52,  0.35,  0.26,  0.41,  0.74,  0.11,  0.67,  0.87,  0.26,  0.07,  0.49,  0.37,  0.10,  0.78,  0.28,  0.99,  0.32,  0.12,  0.73,  0.58,  0.17,  0.96,  0.18,  0.31,  0.63,  0.23,  0.41,  0.49,  0.19,  0.06,  0.29,  0.32,  0.29,  0.92,  0.84,  0.73,  0.44,  0.20,  0.09,  0.02,  0.15,  0.50,  0.78,  0.94,  0.45,  0.07,  0.77,  0.49,  0.28,  0.53,  0.05,  0.60,  0.50,  0.85,  0.86,  0.91,  0.52,  0.61,  0.23,  0.99,  0.58,  0.90,  0.25,  0.13,  0.76,  0.60,  0.33,  0.72,  0.62,  0.63,  0.69,  0.52,  0.83,  0.90,  0.02,  0.65,  0.60,  0.57,  0.99,  0.68,  0.93,  0.68,  0.88,  0.81,  0.80,  0.78,  0.79,  0.56,  0.27,  0.02,  0.97,  0.73,  0.17,  0.24,  0.25,  0.14,  0.03,  0.13]
W2: [0.35,  0.50,  0.88,  0.21,  0.96,  0.59,  0.89,  0.63,  0.98,  0.45]
b1: 0.00
b1: [0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00]
     */

}
