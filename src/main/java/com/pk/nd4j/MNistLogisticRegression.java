package com.pk.nd4j;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.io.*;
import java.util.Arrays;
import java.util.HashMap;

import static com.pk.nd4j.MNistReader.*;

@SuppressWarnings("Duplicates")
public class MNistLogisticRegression {

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

    static boolean regression_or_predict = false;

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
            //Initialize weights & bias
            weights = Nd4j.rand(new int[]{28 * 28, 10});
            //System.out.println("Initial Weights :: "+weights);
            //weights = Nd4j.zeros(new int[]{28*28, 10});
            bias = 0;


            //originalArray.get(NDArrayIndex.all(), NDArrayIndex.interval(0,3))
            HashMap<String, Object> params = forward_propogate(train_images_normalized, train_labels_flatten, num_of_train_samples, 100, 0.001);
            //HashMap<String, Object> params = forward_propogate(train_images_normalized.get(NDArrayIndex.all(), NDArrayIndex.interval(0,20)), train_labels_flatten.get(NDArrayIndex.all(), NDArrayIndex.interval(0,20)), 20, 3, 0.01);

            try (FileOutputStream fos = new FileOutputStream("weights");
                 FileOutputStream fos_bais = new FileOutputStream("bias");) {
                Nd4j.write(fos, (INDArray) params.get("weights"));
                fos_bais.write((bias+"").getBytes());
            } catch (IOException ioe) {
                ioe.printStackTrace();
            }


            System.out.println("---------------------------------------------------------------");
            System.out.println("---------------------------------------------------------------");

            System.out.println("Predicting: " + test_labels_flatten.getColumn(whichImage));
            predict(test_images_normalized.getColumn(whichImage), (INDArray) params.get("weights"), (Double) params.get("bias"));
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

        //images: 784 X 20
        //weights: 784 X 10
        //labels: 1 X 20
        //A: 1 X 20
        //first_part y(log(a)) : 784 X 10

        while(epoch > 0) {
            INDArray A = weights.transpose().mmul(images).add(bias);
            A = Transforms.softmax(A.transpose());
            A = A.transpose();

//            System.out.println("Shape of Images -> "+Arrays.toString(images.shape()));
//            System.out.println("Shape of A -> "+Arrays.toString(A.shape()));
//            System.out.println("Shape of Labels -> "+Arrays.toString(labels.shape()));

            //This is for Softmax classification
            double cost = (-1) * Nd4j.mean(labels.mul(Transforms.log(A)), 0).getDouble(0);

            // This is for Sigmoid activation
             //double cost = (-1 * total_part.sumNumber().doubleValue() / num_samples);

            System.out.println("Cost after " + epoch +" iteration => " + cost);

            INDArray dWeight = images.mmul(A.sub(labels).transpose()).mul(1 / num_samples);
            double dBias = (A.sub(labels).sumNumber()).doubleValue() / num_samples;

            weights = weights.sub(dWeight.mul(learning_rate));
            bias = bias - (dBias * learning_rate);



            epoch--;
        }

        params.put("weights", weights);
        params.put("bias",bias);
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
        System.out.println("Prediction:: "+A);

    }
}
