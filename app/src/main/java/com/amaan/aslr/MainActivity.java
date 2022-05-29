package com.amaan.aslr;

import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;

import androidx.appcompat.app.AppCompatActivity;

import org.opencv.android.OpenCVLoader;

public class MainActivity extends AppCompatActivity 
{
    static 
    {
        if (OpenCVLoader.initDebug()) 
        {
            Log.d("MainActivity: ", "Opencv is loaded");
        } 
        else 
        {
            Log.d("MainActivity: ", "Opencv failed to load");
        }
    }
    
    // to display button at homescreen for the 2 features.
    private Button camera_button;
    private Button combine_letter_button;

    @Override
    protected void onCreate(Bundle savedInstanceState) 
    {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        
        camera_button = findViewById(R.id.camera_button); // to find the camera button's id from the activity_main.xml
        
        // now, on clicking the camera_button, the activity will redirect from main activity to cameraActivity class
        camera_button.setOnClickListener(new View.OnClickListener() 
        {
            @Override
            public void onClick(View v) 
            {
                //when this button is clicked, navigate to CameraActivity
                startActivity(new Intent(MainActivity.this, CameraActivity.class).addFlags(Intent.FLAG_ACTIVITY_CLEAR_TASK | Intent.FLAG_ACTIVITY_CLEAR_TOP));
            }
        });


        combine_letter_button = findViewById(R.id.combine_letter_button); // to find the camera button's id from the activity_main.xml
        
         // now, on clicking the combine_letter_button, the activity will redirect from main activity to CombineLettersActivity class
        combine_letter_button.setOnClickListener(new View.OnClickListener() 
        {
            @Override
            public void onClick(View view) 
            {
                //when this button is clicked, navigate to CombineLettersActivity
                startActivity(new Intent(MainActivity.this, CombineLettersActivity.class).addFlags(Intent.FLAG_ACTIVITY_CLEAR_TASK | Intent.FLAG_ACTIVITY_CLEAR_TOP));
            }
        });
    }
}
