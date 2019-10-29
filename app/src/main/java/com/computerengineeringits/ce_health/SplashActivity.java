package com.computerengineeringits.ce_health;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.content.Context;

public class SplashActivity extends AppCompatActivity {

    private static final int MY_PERMISSIONS_REQUEST_CAMERA = 1;
    int PERMISSION_ALL = 1;
    String[] PERMISSIONS = {
            Manifest.permission.CAMERA,
            Manifest.permission.WRITE_EXTERNAL_STORAGE,
            Manifest.permission.INTERNET
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);


        if(!hasPermissions(this, PERMISSIONS)){
            ActivityCompat.requestPermissions(this, PERMISSIONS, PERMISSION_ALL);
        } else {
            // langsung pindah ke MainActivity atau activity lain
            // begitu memasuki splash screen ini
            //Intent intent = new Intent(this, CameraActivity.class);
            Intent intent = new Intent(this, DetectionActivity.class);
            startActivity(intent);
            finish();
        }
    }

    public static boolean hasPermissions(Context context, String... permissions) {
        if (context != null && permissions != null) {
            for (String permission : permissions) {
                if (ActivityCompat.checkSelfPermission(context, permission) != PackageManager.PERMISSION_GRANTED) {
                    return false;
                }
            }
        }
        return true;
    }

    @Override
    public void onRequestPermissionsResult(int requestCode,
                                           String[] permissions, int[] grantResults) {
        if (requestCode == PERMISSION_ALL){
            // langsung pindah ke MainActivity atau activity lain
            // begitu memasuki splash screen ini
            //Intent intent = new Intent(this, CameraActivity.class);
            Intent intent = new Intent(this, DetectionActivity.class);
            startActivity(intent);
            finish();
            // permission was granted, yay!
        }
        super.onRequestPermissionsResult(requestCode,permissions,grantResults);
    }
}