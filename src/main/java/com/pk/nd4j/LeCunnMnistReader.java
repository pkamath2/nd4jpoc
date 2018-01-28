package com.pk.nd4j;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;
import java.nio.ByteBuffer;
import java.util.Arrays;

public class LeCunnMnistReader {

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

        System.out.println("------------------------------------------------");

        //For Test Images
        test_images_flatten = createImageNDArray(test_images, num_of_test_samples);
        test_labels_flatten = Nd4j.create(Arrays.stream(test_labels).asDoubleStream().toArray());

        train_images_flatten = createImageNDArray(train_images, num_of_train_samples);
        train_labels_flatten = Nd4j.create(Arrays.stream(train_labels).asDoubleStream().toArray());

        System.out.println("Shape of test images => "+ Arrays.toString(test_images_flatten.shape()));
        System.out.println("Shape of test labels => "+ Arrays.toString(test_labels_flatten.shape()));
        System.out.println("Shape of train images => "+ Arrays.toString(train_images_flatten.shape()));
        System.out.println("Shape of train labels => "+ Arrays.toString(train_labels_flatten.shape()));


        int whichImage = 9000;
        drawImage(train_images_flatten.getColumn(whichImage), train_labels_flatten.getInt(whichImage), whichImage);

        test_images_normalized = test_images_flatten.div(255);
        train_images_normalized = train_images_flatten.div(255);

    }

    private static void drawImage(INDArray imageArr, int label, int index){

        imageArr = imageArr.reshape(28,28);
        try {
            BufferedImage image = new BufferedImage(28,28, BufferedImage.TYPE_INT_RGB);
            for(int i=0;i<28;i++){
                INDArray row = imageArr.getColumn(i);
                for(int j=0;j<28;j++){
                    int pixel = row.getInt(j);
                    image.setRGB(i,j,pixel);
                }
            }
            ImageIO.write(image, "png", new File("image-"+label+"-"+index+".png"));
        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    private static INDArray createImageNDArray(int[] imageArr, int num_of_samples){//28*28*num of samples

        INDArray indArray = Nd4j.create(Arrays.stream(imageArr).asDoubleStream().toArray());
        System.out.println(num_of_samples);
        System.out.println(Arrays.toString(indArray.shape()));

        indArray = indArray.reshape(num_of_samples, 28*28);
        indArray = indArray.transpose();

        return indArray;
    }


    private static MNistBO createImageArray(String imageFileLocation){
        int[] images = null;
        int sample_size = 0;
        MNistBO mNistBO = new MNistBO();
        try(ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
            FileInputStream fileInputStream = new FileInputStream(imageFileLocation);){

            byte[] tempByteArr = new byte[255];

            int readStatus = fileInputStream.read(tempByteArr);
            while(readStatus != -1){
                byteArrayOutputStream.write(tempByteArr);
                readStatus = fileInputStream.read(tempByteArr);
            }

            byte[] ba = byteArrayOutputStream.toByteArray();
            ByteBuffer byteBuffer = ByteBuffer.wrap(ba);

            int magic_number = byteBuffer.getInt();
            sample_size = byteBuffer.getInt();
            int num_of_rows = byteBuffer.getInt();
            int num_of_columns = byteBuffer.getInt();

            System.out.println("Magic Number => "+magic_number+" & Sample Size =>"+sample_size+" & Number Of Rows =>"+num_of_rows+" & Number Of Columns =>"+num_of_columns);

            images = new int[sample_size*num_of_columns*num_of_rows];
            for (int i=0 ;i<sample_size*num_of_columns*num_of_rows;i++) {
                images[i] = byteBuffer.get() & 0xff;//Unsigned ints
            }

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e){
            e.printStackTrace();
        }

        mNistBO.setNum_of_samples(sample_size);
        mNistBO.setData_arr(images);
        return mNistBO;
    }

    private static MNistBO createLabelArray(String labelFileLocation) {

        int[] labels = null;
        int sample_size = 0;
        MNistBO mNistBO = new MNistBO();
        try(ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
            FileInputStream fileInputStream = new FileInputStream(labelFileLocation);){

            byte[] tempByteArr = new byte[255];

            int readStatus = fileInputStream.read(tempByteArr);
            while(readStatus != -1){
                byteArrayOutputStream.write(tempByteArr);
                readStatus = fileInputStream.read(tempByteArr);
            }

            byte[] ba = byteArrayOutputStream.toByteArray();
            ByteBuffer byteBuffer = ByteBuffer.wrap(ba);

            int magic_number = byteBuffer.getInt();
            sample_size = byteBuffer.getInt();

            System.out.println("Magic Number => "+magic_number+" & Sample Size =>"+sample_size);

            labels = new int[sample_size];
            for (int i=0 ;i<sample_size;i++){
                labels[i] = byteBuffer.get() & 0xff;//Unsigned ints
            }

            System.out.println("Labels =>"+ Arrays.toString(labels));


        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e){
            e.printStackTrace();
        }

        mNistBO.setNum_of_samples(sample_size);
        mNistBO.setData_arr(labels);
        return mNistBO;
    }


}
