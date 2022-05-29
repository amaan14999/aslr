package com.amaan.aslr;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;
import android.view.Window;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.TextView;

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
public class CombineLettersActivity extends Activity implements CameraBridgeViewBase.CvCameraViewListener2 
{
    private static final String TAG = "MainActivity";

    private Mat mRgba;
    private Mat mGray;
    private CameraBridgeViewBase mOpenCvCameraView;
    private SignLanguage signLanguageClass;

    // We define the Clear and Add Button here
    // Add button will append each alphabet after the previous alphabet
    // Clear button will clear the String that is currently being displayed on the Screen.
    private Button clear_button;
    private Button add_button;
    
    // We define textview to display the String on the top of the screen
    private TextView change_text;

    //We define a button to convert the string displayed as text, to speech
    private Button text_speech_button;


    private final BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) 
    {
        @Override
        public void onManagerConnected(int status) 
        {
            switch (status) 
            {
                case LoaderCallbackInterface
                        .SUCCESS: {
                    Log.i(TAG, "OpenCv Is loaded");
                            //upon successfully laoding the OpenCV library, we enable the CameraView
                    mOpenCvCameraView.enableView();
                }
                default: {
                    super.onManagerConnected(status);

                }
                break;
            }
        }
    };

    public CombineLettersActivity() 
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
        if (ContextCompat.checkSelfPermission(CombineLettersActivity.this, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_DENIED) {
            ActivityCompat.requestPermissions(CombineLettersActivity.this, new String[]{Manifest.permission.CAMERA}, MY_PERMISSIONS_REQUEST_CAMERA);
        }

        // After camera permission is granted, we change from that activity to the new activity, thus changing the layout from activity_combine_letters.xml
        setContentView(R.layout.activity_combine_letters);


        mOpenCvCameraView = findViewById(R.id.frame_Surface);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
        
        // we assign the values to each of the buttons and textview created by finding their IDs from the xml file.
        clear_button = findViewById(R.id.clear_button);
        add_button = findViewById(R.id.add_button);
        change_text = findViewById(R.id.change_text);
        text_speech_button = findViewById(R.id.text_speech_button);


        try {
            
            // we pass the clear_button, add_button, and change_text in signLanguageClass
            // we will also pass text_speech_button through signLanguageClass
            // we will aslo need context in this class as well
            // now we can define this button and text in signlanguageClass and can use it.
            // the remaining paramets will be same from the CameraActivity class
            // i.e., hand_model.tflite, its size,300
            // Sign_lang_model, its size, 96
            // we are doing this as this class also does the same function CameraActivity, it just adds extra features
            // of adding and forming words and converting them from text to speech.
            
            signLanguageClass = new SignLanguage(CombineLettersActivity.this, clear_button, add_button, change_text, text_speech_button, getAssets(), "hand_model.tflite", "custom_label.txt", 300, "Sign_lang_model.tflite", 96);
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
        out = signLanguageClass.recognizeImage(mRgba);

        return out;
    }

}
