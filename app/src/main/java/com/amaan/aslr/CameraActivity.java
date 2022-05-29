package com.amaan.aslr;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;
import android.view.Window;
import android.view.WindowManager;

import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

import java.io.IOException;

//to inherit the CameraBridgeView from OpenCV library we extend and implement it here
public class CameraActivity extends Activity implements CameraBridgeViewBase.CvCameraViewListener2 
{
    private static final String TAG = "MainActivity";

    private Mat mRgba;
    private Mat mGray;
    private CameraBridgeViewBase mOpenCvCameraView;
    private ObjectDetector objectDetectorClass;
    private final BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) 
    {
        @Override
        public void onManagerConnected(int status) 
        {
            switch (status) 
            {
                case LoaderCallbackInterface
                        .SUCCESS: 
                        {
                    Log.i(TAG, "OpenCv Is loaded");
                            //upon successfully laoding the OpenCV library, we enable the CameraView
                    mOpenCvCameraView.enableView();
                }
                default: 
                {
                    super.onManagerConnected(status);

                }
                break;
            }
        }
    };

    public CameraActivity() 
    {
        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) 
    {
        super.onCreate(savedInstanceState);
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        int MY_PERMISSIONS_REQUEST_CAMERA = 0;
        
        //Ask for camera permission on device, if already not given
        if (ContextCompat.checkSelfPermission(CameraActivity.this, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_DENIED) {
            ActivityCompat.requestPermissions(CameraActivity.this, new String[]{Manifest.permission.CAMERA}, MY_PERMISSIONS_REQUEST_CAMERA);
        }

        //After camera permission is granted, we change from that activity to the new activity, thus changing the layout from activity_camera.xml
        setContentView(R.layout.activity_camera);

        mOpenCvCameraView = findViewById(R.id.frame_Surface);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
        try 
        {
            
            // now we call the ObjectDetector class here
            // the first parameter passed here will be getAssets(), to access the assets folder and all the models present in it
            // then we pass the hand_model.tflite, which predicts whether the object present in the camera frame is a hand or not
            // we pass the label for the prediction and the input size of the hand_model, 300
            // then we pass the Sign_lan_model.tflite for sign language alphabet detection
            // Lastly we pass the input size of the Sign_Lang_model, 96
            
            objectDetectorClass = new ObjectDetector(getAssets(), "hand_model.tflite", "custom_label.txt", 300, "Sign_lang_model.tflite", 96);
            Log.d("MainActivity", "Model is successfully loaded");
        } catch (IOException e) {
            Log.d("MainActivity", "Getting some error");
            e.printStackTrace();
        }
    }

    @Override
    protected void onResume() 
    {
        super.onResume();
        if (OpenCVLoader.initDebug()) 
        {
            //if the load is successful
            Log.d(TAG, "Opencv initialization is done");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        } 
        else 
        {
            //if the load is unsuccessful
            Log.d(TAG, "Opencv is not loaded. try again");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, mLoaderCallback);
        }
    }

    @Override
    protected void onPause() 
    {
        super.onPause();
        if (mOpenCvCameraView != null) 
        {
            mOpenCvCameraView.disableView();
        }
    }

    public void onDestroy() 
    {
        super.onDestroy();
        if (mOpenCvCameraView != null) 
        {
            mOpenCvCameraView.disableView();
        }

    }

    public void onCameraViewStarted(int width, int height) 
    {
        mRgba = new Mat(height, width, CvType.CV_8UC4);
        mGray = new Mat(height, width, CvType.CV_8UC1);
    }

    public void onCameraViewStopped() 
    {
        mRgba.release();
    }

    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) 
    {
        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();

        Mat out = new Mat();
        out = objectDetectorClass.recognizeImage(mRgba);

        return out;
    }

}
