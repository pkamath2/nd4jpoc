package com.pk.nd4j;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;
import java.nio.ByteBuffer;
import java.util.Arrays;

@SuppressWarnings("Duplicates")
public class MNistReader {

    public static void drawImage(INDArray imageArr, int label, int index){

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

    public static INDArray createImageNDArray(int[] imageArr, int num_of_samples){//28*28*num of samples

        INDArray indArray = Nd4j.create(Arrays.stream(imageArr).asDoubleStream().toArray());
        System.out.println(num_of_samples);
        System.out.println(Arrays.toString(indArray.shape()));

        indArray = indArray.reshape(num_of_samples, 28*28);
        indArray = indArray.transpose();

        return indArray;
    }


    public static MNistBO createImageArray(String imageFileLocation){
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

    public static MNistBO createLabelArray(String labelFileLocation) {

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
