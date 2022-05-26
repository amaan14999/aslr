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

public class CombineLettersActivity extends Activity implements CameraBridgeViewBase.CvCameraViewListener2 {
    private static final String TAG = "MainActivity";

    private Mat mRgba;
    private Mat mGray;
    private CameraBridgeViewBase mOpenCvCameraView;
    private SignLanguage signLanguageClass;

    //now define Add ad Clear Button here
    private Button clear_button;
    private Button add_button;
    //define textview also
    private TextView change_text;

    //define button for tts
    private Button text_speech_button;


    private final BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface
                        .SUCCESS: {
                    Log.i(TAG, "OpenCv Is loaded");
                    mOpenCvCameraView.enableView();
                }
                default: {
                    super.onManagerConnected(status);

                }
                break;
            }
        }
    };

    public CombineLettersActivity() {
        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        int MY_PERMISSIONS_REQUEST_CAMERA = 0;
        // if camera permission is not given it will ask for it on device
        if (ContextCompat.checkSelfPermission(CombineLettersActivity.this, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_DENIED) {
            ActivityCompat.requestPermissions(CombineLettersActivity.this, new String[]{Manifest.permission.CAMERA}, MY_PERMISSIONS_REQUEST_CAMERA);
        }

        setContentView(R.layout.activity_combine_letters);


        mOpenCvCameraView = findViewById(R.id.frame_Surface);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
        clear_button = findViewById(R.id.clear_button);
        add_button = findViewById(R.id.add_button);
        change_text = findViewById(R.id.change_text);
        text_speech_button = findViewById(R.id.text_speech_button);


        try {
            // so we have to us clear_button, ad_button , change_text in signlanguageclass
            //due to shortage of time we haven't found any other method and will use this method
            //we will just pass clear_button, add_button, change_text here

            //we will pass text_speech_button through signLanguageClass
            //we will aslo need context in this class as well

            //now I can define this button and text in signlanguageClass and can use it.
            signLanguageClass = new SignLanguage(CombineLettersActivity.this, clear_button, add_button, change_text, text_speech_button, getAssets(), "hand_model.tflite", "custom_label.txt", 300, "Sign_lang_model.tflite", 96);
            Log.d("MainActivity", "Model is successfully loaded");
        } catch (IOException e) {
            Log.d("MainActivity", "Getting some error");
            e.printStackTrace();
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (OpenCVLoader.initDebug()) {
            //if load success
            Log.d(TAG, "Opencv initialization is done");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        } else {
            //if not loaded
            Log.d(TAG, "Opencv is not loaded. try again");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, mLoaderCallback);
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null) {
            mOpenCvCameraView.disableView();
        }
    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null) {
            mOpenCvCameraView.disableView();
        }

    }

    public void onCameraViewStarted(int width, int height) {
        mRgba = new Mat(height, width, CvType.CV_8UC4);
        mGray = new Mat(height, width, CvType.CV_8UC1);
    }

    public void onCameraViewStopped() {
        mRgba.release();
    }

    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();
        // Before watching this video please watch previous video of loading tensorflow lite model

        // now call that function
        Mat out = new Mat();
        out = signLanguageClass.recognizeImage(mRgba);

        return out;
    }

}