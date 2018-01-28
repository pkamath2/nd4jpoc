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

    private static INDArray weights = null;
    private static double bias = 0;

    static boolean regression_or_predict = true;

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


        int whichImage = 1500;
        //

        test_images_normalized = test_images_flatten.div(255);//255
        train_images_normalized = train_images_flatten.div(255);//255

        System.out.println(train_images_normalized.getColumn(9000));


        if(regression_or_predict) {
            /*//Initialize weights & bias
            weights = Nd4j.rand(new int[]{28 * 28, 10});
            //System.out.println("Initial Weights :: "+weights);
            //weights = Nd4j.zeros(new int[]{28*28, 10});
            bias = 0;*/


            //originalArray.get(NDArrayIndex.all(), NDArrayIndex.interval(0,3))
            HashMap<String, Object> params = forward_propogate(train_images_normalized, train_labels_flatten, num_of_train_samples, 100, 0.001);
            //HashMap<String, Object> params = forward_propogate(train_images_normalized.get(NDArrayIndex.all(), NDArrayIndex.interval(0,10000)), train_labels_flatten.get(NDArrayIndex.all(), NDArrayIndex.interval(0,10000)), 20, 100, 0.01);

            /*try (FileOutputStream fos = new FileOutputStream("weights");
                 FileOutputStream fos_bais = new FileOutputStream("bias");) {
                Nd4j.write(fos, (INDArray) params.get("weights"));
                fos_bais.write((bias+"").getBytes());
            } catch (IOException ioe) {
                ioe.printStackTrace();
            }


            System.out.println("---------------------------------------------------------------");
            System.out.println("---------------------------------------------------------------");

            System.out.println("Predicting: " + test_labels_flatten.getColumn(whichImage));
            predict(test_images_normalized.getColumn(whichImage), (INDArray) params.get("weights"), (Double) params.get("bias"));*/
        }else{

            drawImage(train_images_flatten.getColumn(whichImage), train_labels_flatten.getInt(whichImage), whichImage);

            try (FileInputStream fis_wts = new FileInputStream("weights");
                 BufferedReader bufferedReader = new BufferedReader(new FileReader("bias"));){
                INDArray wts = Nd4j.read(fis_wts);
                double b = new Double(bufferedReader.readLine()).doubleValue();

                System.out.println("Predicting: " + train_labels_flatten.getColumn(whichImage));
                predict(train_images_normalized.getColumn(whichImage), wts, b);
            } catch (IOException ioe) {
                ioe.printStackTrace();
            }

        }


    }

    private static HashMap forward_propogate(INDArray images, INDArray labels, int num_samples, int epoch, double learning_rate){

        System.out.println("*****&&&&&&&&&&&******");
        labels = convertNumLabelsToSoftmaxLabels(labels);
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

        List<Double> costList = new ArrayList<>();
        while(epoch > 0) {
            INDArray Z1 = W1.mmul(images).add(b1);
            INDArray A1 = Transforms.tanh(Z1.transpose());
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

            System.out.println("dZ2 Shape => " + Arrays.toString(dZ2.shape()));
            System.out.println("dW2 Shape => " + Arrays.toString(dW2.shape()));
            System.out.println("db2 Shape => " + Arrays.toString(db2.shape()));

            INDArray dZ1 = W2.transpose().mmul(dZ2).mul(Transforms.pow(A1, 2).mul(-1).add(1));
            INDArray dW1 = dZ1.mmul(images.transpose()).mul(1 / num_samples);
            INDArray db1 = Nd4j.mean(dZ1, 1);

            System.out.println("dZ1 Shape => " + Arrays.toString(dZ1.shape()));
            System.out.println("dW1 Shape => " + Arrays.toString(dW1.shape()));
            System.out.println("db1 Shape => " + Arrays.toString(db1.shape()));
            System.out.println("BEFORE W1 Shape => " + Arrays.toString(W1.shape()));
            W1 = W1.sub(dW1.mul(learning_rate));
            System.out.println("AFTER W1 Shape => " + Arrays.toString(W1.shape()));
            b1 = b1.sub(db1.mul(learning_rate));

            W2 = W2.sub(dW2.mul(learning_rate));
            b2 = b2.sub(db2.mul(learning_rate));


            System.out.println("W1 Shape => " + Arrays.toString(W1.shape()));
            System.out.println("b1 Shape => " + Arrays.toString(b1.shape()));
            System.out.println("W2 Shape => " + Arrays.toString(W2.shape()));
            System.out.println("b2 Shape => " + Arrays.toString(b2.shape()));

            System.out.println("Cost after epoch: " + cost);
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

    public static void predict(INDArray input, INDArray weights, double bias){

        INDArray A = weights.transpose().mmul(input).add(bias);
        A = Transforms.softmax(A.transpose());
        System.out.println(Arrays.toString(A.shape()));
        System.out.println("Prediction:: "+A);

    }

}
