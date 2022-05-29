package com.amaan.aslr;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

public class ObjectDetector {
    
    private final Interpreter interp; // we create an Interpreter to encapsulate a pre-trained TensorFlow Lite model, in this case hand_model.tflite
    private final Interpreter interp2;     // we create another interpreter for sign language model
    
    // store all label in array
    private final List<String> labelList;
    private final int INPUT_SIZE;
    private final int PIXEL_SIZE = 3; // for RGB
    private final int IMAGE_MEAN = 0;
    private final float IMAGE_STD = 255.0f;
    
    // use to initialize gpu in app
    private final GpuDelegate graphicsProcessingUnitDelegate;
    private int height = 0;
    private int width = 0;
    private int Classification_Input_Size = 0;

    //                                          hand_model                    input size of hand_model  Sign_lang_model          Input size of Sign_lang_model
    ObjectDetector(AssetManager assetManager, String modelPath, String labelPath, int inputSize, String classification_model, int classification_input_size) throws IOException {
        INPUT_SIZE = inputSize;
        Classification_Input_Size = classification_input_size;
        
        // use to define gpu or cpu // no. of threads
        Interpreter.Options options = new Interpreter.Options();
        graphicsProcessingUnitDelegate = new GpuDelegate();
        options.addDelegate(graphicsProcessingUnitDelegate);
        options.setNumThreads(4); // set it according to your phone
        
        // loading hand_model
        interp = new Interpreter(loadModelFile(assetManager, modelPath), options);
        // load labelmap
        labelList = loadLabelList(assetManager, labelPath);

        //code for loading sign language model
        Interpreter.Options options2 = new Interpreter.Options();
        //add 2 threads to this interpreter
        options2.setNumThreads(2);
        //load model
        interp2 = new Interpreter(loadModelFile(assetManager, classification_model), options2);


    }

    private List<String> loadLabelList(AssetManager assetManager, String labelPath) throws IOException {
        
        // to store label
        List<String> labelList = new ArrayList<>();
        
        // create a new reader
        BufferedReader reader = new BufferedReader(new InputStreamReader(assetManager.open(labelPath)));
        String line;
        
        // loop through each line and store it to labelList
        while ((line = reader.readLine()) != null) {
            labelList.add(line);
        }
        reader.close();
        return labelList;
    }

    private ByteBuffer loadModelFile(AssetManager assetManager, String modelPath) throws IOException {
        // use to get description of file
        AssetFileDescriptor fileDescriptor = assetManager.openFd(modelPath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();

        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    // create new Mat function
    public Mat recognizeImage(Mat mat_image) {
        
        // Rotate original image by 90 degree get get portrait frame
        Mat rotated_mat_image = new Mat();
        Mat a = mat_image.t();
        Core.flip(a, rotated_mat_image, 1);
        // Release mat
        a.release();
        // if you do not do this process you will get improper prediction, less no. of object
        
        
        // now convert it to bitmap
        Bitmap bitmap = null;
        bitmap = Bitmap.createBitmap(rotated_mat_image.cols(), rotated_mat_image.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(rotated_mat_image, bitmap);
        
        // define height and width
        height = bitmap.getHeight();
        width = bitmap.getWidth();

        // scale the bitmap to input size of model
        Bitmap scaledBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, false);

        // convert bitmap to bytebuffer as model input should be in it
        ByteBuffer byteBuffer = convertBitmapToByteBuffer(scaledBitmap);
        

        // defining output
        // 10: top 10 object detected
        // 4: their coordinate in image
        //  float[][][]result=new float[1][10][4];
        Object[] input = new Object[1];
        input[0] = byteBuffer;

        Map<Integer, Object> output_map = new TreeMap<>();
        // we create treemap of three array (boxes,score,classes)

        float[][][] boxes = new float[1][10][4];
        // 10: top 10 object detected
        // 4: their coordinate in image
        
        
        float[][] scores = new float[1][10];
        // stores scores of 10 object
        
        
        float[][] classes = new float[1][10];
        // stores class of object

        
        // add it to object_map;
        output_map.put(0, boxes);
        output_map.put(1, classes);
        output_map.put(2, scores);

        // now we predict
        interp.runForMultipleInputsOutputs(input, output_map);
        Object value = output_map.get(0);
        Object Object_class = output_map.get(1);
        Object score = output_map.get(2);

        
        // loop through each object
        // as output has only 10 boxes
        for (int i = 0; i < 10; i++) {
            //here we are looping through each hand which is detected
            float class_value = (float) Array.get(Array.get(Object_class, 0), i);
            float score_value = (float) Array.get(Array.get(score, 0), i);
            // define threshold for score
            
            // Here you can change threshold according to your model
            if (score_value > 0.5) {
                Object box1 = Array.get(Array.get(value, 0), i);
            
                
                // we are multiplying box1 with Original height and width of frame
                //(x1,y1) is the starting point of the hand
                //(x2,y2) is the end point of the hand
                float y1 = (float) Array.get(box1, 0) * height;
                float x1 = (float) Array.get(box1, 1) * width;
                float y2 = (float) Array.get(box1, 2) * height;
                float x2 = (float) Array.get(box1, 3) * width;

                //setting boundary limit
                if (y1 < 0) 
                {
                    y1 = 0;
                }
                if (x1 < 0) 
                {
                    x1 = 0;
                }
                if (x2 > width) 
                {
                    x2 = width;
                }
                if (y2 > height) 
                {
                    y2 = height;
                }
                
                
                //now we set height and width of box
                //since, (x1,y1) is the starting point of the hand
                //(x2,y2) is the end point of the hand, the width and height will be:
                float w1 = x2 - x1;
                float h1 = y2 - y1;


                //crop hand image from original frame
                Rect cropped_roi = new Rect((int) x1, (int) y1, (int) w1, (int) h1);
                Mat cropped = new Mat(rotated_mat_image, cropped_roi).clone();

                
                //Now convert this cropped Mat to Bitmap
                Bitmap bitmap1 = null;
                bitmap1 = Bitmap.createBitmap(cropped.cols(), cropped.rows(), Bitmap.Config.ARGB_8888);
                Utils.matToBitmap(cropped, bitmap1);


                //Resize bitmap to classification input size=96
                Bitmap scaledBitmap1 = Bitmap.createScaledBitmap(bitmap1, Classification_Input_Size, Classification_Input_Size, false);
                
                //convert scaled bitmap to byte buffer
                ByteBuffer byteBuffer1 = convertBitmapToByteBuffer1(scaledBitmap1);

                //create an array of output of interpr2
                float[][] output_class_value = new float[1][1];

                //predict output for bytebuffer1
                interp2.run(byteBuffer1, output_class_value);

                //Just to check the output_class_value
                Log.d("objectDetectionClass", "output_class_value: " + output_class_value[0][0]);
                
                //now we need to convert output_class_value to alphabets
                //to do that we will create get_alphabets function
                String sign_value = get_alphabets(output_class_value[0][0]);
                
                //use puttext to add Class name in image
                //              input/output      text            starting point                font size              text color:white
                Imgproc.putText(rotated_mat_image, "" + sign_value, new Point(x1 + 10, y1 + 40), 2, 1.5, new Scalar(255, 255, 255, 255), 2);

                // draw rectangle in Original frame //  starting point    // ending point of box  // color of box   thickness
                Imgproc.rectangle(rotated_mat_image, new Point(x1, y1), new Point(x2, y2), new Scalar(0, 255, 0, 255), 2);
            }

        }

        // after prediction, rotate back by -90 degree
        Mat b = rotated_mat_image.t();
        Core.flip(b, mat_image, 0);
        b.release();
        return mat_image;
    }
    
    //function get_alphabets, to convert the number received in line 241 and convert it into its corresponding alphabet
    private String get_alphabets(float sig_v) {
        String val = "";
        if (sig_v >= -0.5 & sig_v < 0.5) {
            val = "A";
        } else if (sig_v >= 0.5 & sig_v < 1.5) {
            val = "B";
        } else if (sig_v >= 1.5 & sig_v < 2.5) {
            val = "C";
        } else if (sig_v >= 2.5 & sig_v < 3.5) {
            val = "D";
        } else if (sig_v >= 3.5 & sig_v < 4.5) {
            val = "E";
        } else if (sig_v >= 4.5 & sig_v < 5.5) {
            val = "F";
        } else if (sig_v >= 5.5 & sig_v < 6.5) {
            val = "G";
        } else if (sig_v >= 6.5 & sig_v < 7.5) {
            val = "H";
        } else if (sig_v >= 7.5 & sig_v < 8.5) {
            val = "I";
        } else if (sig_v >= 8.5 & sig_v < 9.5) {
            val = "J";
        } else if (sig_v >= 9.5 & sig_v < 10.5) {
            val = "K";
        } else if (sig_v >= 10.5 & sig_v < 11.5) {
            val = "L";
        } else if (sig_v >= 11.5 & sig_v < 12.5) {
            val = "M";
        } else if (sig_v >= 12.5 & sig_v < 13.5) {
            val = "N";
        } else if (sig_v >= 13.5 & sig_v < 14.5) {
            val = "O";
        } else if (sig_v >= 14.5 & sig_v < 15.5) {
            val = "P";
        } else if (sig_v >= 15.5 & sig_v < 16.5) {
            val = "Q";
        } else if (sig_v >= 16.5 & sig_v < 17.5) {
            val = "R";
        } else if (sig_v >= 17.5 & sig_v < 18.5) {
            val = "S";
        } else if (sig_v >= 18.5 & sig_v < 19.5) {
            val = "T";
        } else if (sig_v >= 19.5 & sig_v < 20.5) {
            val = "U";
        } else if (sig_v >= 20.5 & sig_v < 21.5) {
            val = "V";
        } else if (sig_v >= 21.5 & sig_v < 22.5) {
            val = "W";
        } else if (sig_v >= 22.5 & sig_v < 23.5) {
            val = "X";
        } else {
            val = "Y";
        }

        return val;
    }
    
    //bytebuffer converter for hand_model
    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
        ByteBuffer byteBuffer;
        int quant = 1;
        int size_images = INPUT_SIZE;
        if (quant == 0) {
            byteBuffer = ByteBuffer.allocateDirect(1 * size_images * size_images * 3);
        } else {
            byteBuffer = ByteBuffer.allocateDirect(4 * 1 * size_images * size_images * 3);
        }
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues = new int[size_images * size_images];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        int pixel = 0;
        for (int i = 0; i < size_images; ++i) {
            for (int j = 0; j < size_images; ++j) {
                final int val = intValues[pixel++];
                if (quant == 0) {
                    byteBuffer.put((byte) ((val >> 16) & 0xFF));
                    byteBuffer.put((byte) ((val >> 8) & 0xFF));
                    byteBuffer.put((byte) (val & 0xFF));
                } else {
                    // paste this
                    byteBuffer.putFloat((((val >> 16) & 0xFF)) / 255.0f);
                    byteBuffer.putFloat((((val >> 8) & 0xFF)) / 255.0f);
                    byteBuffer.putFloat((((val) & 0xFF)) / 255.0f);
                }
            }
        }
        return byteBuffer;
    }


    
    //we create another bytebuffer converter for sign language model
    private ByteBuffer convertBitmapToByteBuffer1(Bitmap bitmap) {
        ByteBuffer byteBuffer;

        int quant = 1;
        //setting input size of the sign language model
        int size_images = Classification_Input_Size;
        if (quant == 0) {
            byteBuffer = ByteBuffer.allocateDirect(1 * size_images * size_images * 3);
        } else {
            byteBuffer = ByteBuffer.allocateDirect(4 * 1 * size_images * size_images * 3);
        }
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues = new int[size_images * size_images];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        int pixel = 0;
        

        for (int i = 0; i < size_images; ++i) {
            for (int j = 0; j < size_images; ++j) {
                final int val = intValues[pixel++];
                if (quant == 0) {
                    byteBuffer.put((byte) ((val >> 16) & 0xFF));
                    byteBuffer.put((byte) ((val >> 8) & 0xFF));
                    byteBuffer.put((byte) (val & 0xFF));
                } else {
                    byteBuffer.putFloat((((val >> 16) & 0xFF)));
                    byteBuffer.putFloat((((val >> 8) & 0xFF)));
                    byteBuffer.putFloat((((val) & 0xFF)));
                }
            }
        }
        return byteBuffer;
    }

}
