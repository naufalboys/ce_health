package com.computerengineeringits.ce_health;

import android.content.res.AssetFileDescriptor;
import android.graphics.Canvas;
import android.graphics.Bitmap;
import android.graphics.Bitmap.CompressFormat;
import android.graphics.Color;
import android.app.Activity;
import android.content.pm.ActivityInfo;
import android.content.SharedPreferences;
import android.nfc.Tag;
import android.os.*;
import android.os.Bundle;
import android.os.SystemClock;
import android.preference.PreferenceManager;
import android.text.SpannableString;
import android.text.style.ForegroundColorSpan;
import android.util.Log;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Date;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Timer;
import java.util.Calendar;

import ch.zhaw.facerecognitionlibrary.Helpers.CustomCameraView;
import ch.zhaw.facerecognitionlibrary.Helpers.MatOperation;
import ch.zhaw.facerecognitionlibrary.PreProcessor.PreProcessorFactory;

public class DetectionActivity extends Activity implements CameraBridgeViewBase.CvCameraViewListener2 {
    private boolean night_portrait;
    private int exposure_compensation;
    private CustomCameraView mDetectionView;
    private PreProcessorFactory ppF;
    private boolean isFrontCamera;
    private Mat imgRgba;
    private TextView textView,textView2;

    Interpreter tensorflow;

    /** Tag for the {@link Log}. */
    private static final String TAG = "CE-Health";

    /** Name of the model file stored in Assets. */
    private static final String MODEL_PATH = "graph.lite";

    /** Name of the label file stored in Assets. */
    private static final String LABEL_PATH = "labels.txt";

    /** Number of results to show in the UI. */
    private static final int RESULTS_TO_SHOW = 3;

    /** Dimensions of inputs. */
    private static final int DIM_BATCH_SIZE = 1;

    private static final int DIM_PIXEL_SIZE = 3;

    static final int DIM_IMG_SIZE_X = 224;
    static final int DIM_IMG_SIZE_Y = 224;

    private static final int IMAGE_MEAN = 128;
    private static final float IMAGE_STD = 128.0f;

    /** Preallocated buffers for storing image data in. */
    private int[] intValues = new int[DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y];

    /** Labels corresponding to the output of the vision model. */
    private List<String> labelList;

    /** A ByteBuffer to hold image data, to be feed into Tensorflow Lite as inputs. */
    private ByteBuffer imgData = null;

    /** An array to hold inference results, to be feed into Tensorflow Lite as outputs. */
    private float[][] labelProbArray = null;
    /** multi-stage low pass filter **/
    private float[][] filterLabelProbArray = null;
    private static final int FILTER_STAGES = 3;
    private static final float FILTER_FACTOR = 0.4f;

    private PriorityQueue<Map.Entry<String, Float>> sortedLabels =
            new PriorityQueue<>(
                    RESULTS_TO_SHOW,
                    new Comparator<Map.Entry<String, Float>>() {
                        @Override
                        public int compare(Map.Entry<String, Float> o1, Map.Entry<String, Float> o2) {
                            return (o1.getValue()).compareTo(o2.getValue());
                        }
                    });

    static {
        if (!OpenCVLoader.initDebug()) {
            // Handle initialization error
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        //setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_PORTRAIT);
        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);
        setContentView(R.layout.activity_detection);
        mDetectionView = (CustomCameraView) findViewById(R.id.DetectionView);
        mDetectionView.setVisibility(SurfaceView.VISIBLE);
        mDetectionView.setCvCameraViewListener(this);
        mDetectionView.enableView();
        ImageView switchCamera = (ImageView) findViewById(R.id.imageViewSwitchCamera2);
        ImageView screen_capture = (ImageView) findViewById(R.id.screencapt);

        textView = (TextView) findViewById(R.id.textdetection);
        textView.setText("");
        textView.setVisibility(View.GONE);
        textView2 = (TextView) findViewById(R.id.textdetection2);

        try {
            tensorflow = new Interpreter(loadModelFile(this, MODEL_PATH));
            labelList = loadLabelList(this);
            imgData =
                    ByteBuffer.allocateDirect(
                            4 * DIM_BATCH_SIZE * DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y * DIM_PIXEL_SIZE);
            imgData.order(ByteOrder.nativeOrder());
            labelProbArray = new float[1][labelList.size()];
            filterLabelProbArray = new float[FILTER_STAGES][labelList.size()];
            Log.i(TAG, "Created a Tensorflow Lite Image Classifier.");
        } catch (IOException e) {
            e.printStackTrace();
        }

        switchCamera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                switch (v.getId()){
                    case R.id.imageViewSwitchCamera2:
                        mDetectionView.disableView();
                        if (isFrontCamera){
                            mDetectionView.setCameraIndex(CameraBridgeViewBase.CAMERA_ID_BACK);
                            isFrontCamera = false;
                        }
                        else {
                            mDetectionView.setCameraIndex(CameraBridgeViewBase.CAMERA_ID_FRONT);
                            isFrontCamera = true;
                        }
                        mDetectionView.enableView();
                        break;
                    default:
                }
            }
        });

        screen_capture.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Bitmap screenshot = takeScreenshot();
                saveBitmap(screenshot);
            }
        });

        // Use camera which is selected in settings
        SharedPreferences sharedPref = PreferenceManager.getDefaultSharedPreferences(this);
        night_portrait = sharedPref.getBoolean("key_night_portrait", false);
        exposure_compensation = Integer.valueOf(sharedPref.getString("key_exposure_compensation", "60"));
        int maxCameraViewWidth = Integer.parseInt(sharedPref.getString("key_maximum_camera_view_width", "320"));
        int maxCameraViewHeight = Integer.parseInt(sharedPref.getString("key_maximum_camera_view_height", "240"));
        mDetectionView.setMaxFrameSize(maxCameraViewWidth, maxCameraViewHeight);
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        if (night_portrait) {
            mDetectionView.setNightPortrait();
        }
        if (exposure_compensation != 50 && 0 <= exposure_compensation && exposure_compensation <= 100)
            mDetectionView.setExposure(exposure_compensation);
        /*if (isFrontCamera) {
            mDetectionView.setExposure(30);
            Log.i(TAG,"Masuk Sini : 30");
        } else {
            mDetectionView.setExposure(40);
            Log.i(TAG,"Masuk Sini : 40");
        }*/
        imgRgba = new Mat(width,height,CvType.CV_8UC4);
    }

    @Override
    public void onCameraViewStopped() {
        imgRgba.release();
        //tensorflow.close();
        //tensorflow = null;
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

        Mat imgRgba = inputFrame.rgba();
        Mat img = new Mat();
        imgRgba.copyTo(img);
        List<Mat> images = ppF.getCroppedImage(img);
        Rect[] faces = ppF.getFacesForRecognition();

        // Selfie / Mirror mode
        if(isFrontCamera){
            //rotate(imgRgba,-90);
            Core.flip(imgRgba,imgRgba,1);
            //Core.rotate(imgRgba, imgRgba, Core.ROTATE_90_CLOCKWISE);
        }

        if(images == null || images.size() == 0 || faces == null || faces.length == 0 || ! (images.size() == faces.length)){
            // skip
            textView2.setText("Tidak Terdeteksi Orang ! ");
            Log.i(TAG,"Tidak Terdeteksi Orang ! ");
            showToast("");
            return imgRgba;
        } else {
            faces = MatOperation.rotateFaces(imgRgba, faces, ppF.getAngleForRecognition());
            //MatOperation.drawRectangleAndLabelOnPreview(imgRgba, faces[i], "", isFrontCamera);
            for(int i = 0; i<faces.length; i++)
                if (i == 0 && (faces.length == 1)) {
                    Log.i(TAG, "Terdeteksi 1 Orang ! ");
                    MatOperation.drawRectangleAndLabelOnPreview(imgRgba, faces[i], "", isFrontCamera);
                    Mat m = new Mat();
                    m = imgRgba.submat(faces[0]);
                    Bitmap mBitmap = Bitmap.createBitmap(m.width(), m.height(), Bitmap.Config.ARGB_8888);
                    Utils.matToBitmap(m, mBitmap);
                    Bitmap kecil = Bitmap.createScaledBitmap(mBitmap, 224, 224, false);
                    String textToShow = Fix(kecil);
                    mBitmap.recycle();
                    showToast(textToShow);
                    textView2.setText("Terdeteksi 1 Orang !");
                    //classifyFrame(imgRgba);
                } else {
                    Log.i(TAG, "Terdeteksi Lebih Dari 1 Orang ! ");
                    MatOperation.drawRectangleAndLabelOnPreview(imgRgba, faces[i], "", isFrontCamera);
                    textView2.setText("Terdeteksi Lebih dari 1 orang !");
                    showToast("");
                }
            return imgRgba;
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        ppF = new PreProcessorFactory(getApplicationContext());
        mDetectionView.enableView();
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (mDetectionView != null) {
            mDetectionView.disableView();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        mDetectionView.disableView();
        tensorflow.close();
        tensorflow = null;
    }

    public Bitmap takeScreenshot() {
        View rootView = findViewById(android.R.id.content).getRootView();
        Bitmap bitmap = Bitmap.createBitmap(rootView.getWidth(), rootView.getHeight(), Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(bitmap);
        rootView.draw(canvas);
        return bitmap;
    }

    public void saveBitmap(Bitmap bitmap) {
        Date currentTime = Calendar.getInstance().getTime();
        String hasilnya = Environment.getExternalStorageDirectory().toString() + "/" + currentTime + "-screenshot.jpg";
        File imagePath = new File(hasilnya);
        FileOutputStream fos;
        try {
            fos = new FileOutputStream(imagePath);
            bitmap.compress(CompressFormat.JPEG, 90, fos);
            fos.flush();
            fos.close();
        } catch (FileNotFoundException e) {
            Log.e(TAG, e.getMessage(), e);
        } catch (IOException e) {
            Log.e(TAG, e.getMessage(), e);
        }
    }

    public static Mat rotate(Mat src, double angle)
    {
        Mat dst = new Mat();
        if(angle == 180 || angle == -180) {
            Core.flip(src, dst, -1);
        } else if(angle == 90 || angle == -270) {
            Core.flip(src.t(), dst, 1);
        } else if(angle == 270 || angle == -90) {
            Core.flip(src.t(), dst, 0);
        }

        return dst;
    }

    /**
     * Shows a {@link Toast} on the UI thread for the classification results.
     *
     * @param text The message to show
     */
    private void showToast(final String text){
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                if (text == null || text.length() == 0) {
                    textView.setVisibility(View.GONE);
                    textView.setText("");
                } else {
                    textView.setVisibility(View.VISIBLE);
                    textView.setText(text);
                }
            }
        });
    }

    /** Classifies a frame from the preview stream. */
    private void classifyFrame(Mat sumber){
        if (tensorflow == null || DetectionActivity.this == null) {
            showToast("Loading Process . . .");
            //showToast("Uninitialized Classifier or invalid context.");
            try {
                tensorflow = new Interpreter(loadModelFile(this, MODEL_PATH));
                labelList = loadLabelList(this);
                imgData =
                        ByteBuffer.allocateDirect(
                                4 * DIM_BATCH_SIZE * DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y * DIM_PIXEL_SIZE);
                imgData.order(ByteOrder.nativeOrder());
                labelProbArray = new float[1][labelList.size()];
                filterLabelProbArray = new float[FILTER_STAGES][labelList.size()];
                Log.i(TAG, "Created a Tensorflow Lite Image Classifier.");
            } catch (IOException e) {
                e.printStackTrace();
            }
            return;
        }
        Bitmap myBitmap = Bitmap.createBitmap(sumber.cols(), sumber.rows(), Bitmap.Config.ARGB_8888);
        //Bitmap myBitmap = Bitmap.createBitmap(sumber.cols(), sumber.rows(), Bitmap.Config.RGB_565);
        Utils.matToBitmap(sumber,myBitmap);
        Bitmap perkecil = Bitmap.createScaledBitmap(myBitmap,224,224,false);
        String textToShow = Fix(perkecil);
        myBitmap.recycle();
        showToast(textToShow);
    }

    /** Classifies a frame from the preview stream. */
    String Fix(Bitmap bitmap){
        if (tensorflow == null || DetectionActivity.this == null) {
            showToast("Loading Process . . .");
            //showToast("Uninitialized Classifier or invalid context.");
            try {
                tensorflow = new Interpreter(loadModelFile(this, MODEL_PATH));
                labelList = loadLabelList(this);
                imgData =
                        ByteBuffer.allocateDirect(
                                4 * DIM_BATCH_SIZE * DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y * DIM_PIXEL_SIZE);
                imgData.order(ByteOrder.nativeOrder());
                labelProbArray = new float[1][labelList.size()];
                filterLabelProbArray = new float[FILTER_STAGES][labelList.size()];
                Log.i(TAG, "Created a Tensorflow Lite Image Classifier.");
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        convertBitmapToByteBuffer(bitmap);
        // Here's where the magic happens!!!
        long startTime = SystemClock.uptimeMillis();
        tensorflow.run(imgData, labelProbArray);
        long endTime = SystemClock.uptimeMillis();
        Log.i(TAG, "Timecost to run model inference: " + Long.toString(endTime - startTime));

        // smooth the results
        applyFilter();

        // print the results
        String textToShow = String.format("Aplikasi Pendeteksi Kesehatan Seseorang\n");
        SpannableString string1 = new SpannableString(textToShow);
        string1.setSpan(new ForegroundColorSpan(Color.rgb(178,235,242)),0,string1.length(),0);
        String runningTime = String.format("\nRunning Time : ") + Long.toString(endTime - startTime) + "ms";
        SpannableString string2 = new SpannableString(runningTime);
        string2.setSpan(new ForegroundColorSpan(Color.rgb(244,199,195)),0,string2.length(),0);
        textToShow = textToShow + runningTime;
        textToShow = textToShow + printTopKLabels();
        /*textToShow = Long.toString(endTime - startTime) + " ms" + textToShow;*/

        return textToShow;
    }

    void applyFilter(){
        int num_labels =  labelList.size();

        // Low pass filter `labelProbArray` into the first stage of the filter.
        for(int j=0; j<num_labels; ++j){
            filterLabelProbArray[0][j] += FILTER_FACTOR*(labelProbArray[0][j] -
                    filterLabelProbArray[0][j]);
        }
        // Low pass filter each stage into the next.
        for (int i=1; i<FILTER_STAGES; ++i){
            for(int j=0; j<num_labels; ++j){
                filterLabelProbArray[i][j] += FILTER_FACTOR*(
                        filterLabelProbArray[i-1][j] -
                                filterLabelProbArray[i][j]);

            }
        }

        // Copy the last stage filter output back to `labelProbArray`.
        for(int j=0; j<num_labels; ++j){
            labelProbArray[0][j] = filterLabelProbArray[FILTER_STAGES-1][j];
        }
    }

    /** Memory-map the model file in Assets. */
    private MappedByteBuffer loadModelFile(Activity activity,String MODEL_FILE) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(MODEL_FILE);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    /** Reads label list from Assets. */
    private List<String> loadLabelList(Activity activity) throws IOException {
        List<String> labelList = new ArrayList<String>();
        BufferedReader reader =
                new BufferedReader(new InputStreamReader(activity.getAssets().open(LABEL_PATH)));
        String line;
        while ((line = reader.readLine()) != null) {
            labelList.add(line);
        }
        reader.close();
        return labelList;
    }

    /** Writes Image data into a {@code ByteBuffer}. */
    private void convertBitmapToByteBuffer(Bitmap bitmap) {
        if (imgData == null) {
            return;
        }
        imgData.rewind();
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        // Convert the image to floating point.
        int pixel = 0;
        long startTime = SystemClock.uptimeMillis();
        for (int i = 0; i < DIM_IMG_SIZE_X; ++i) {
            for (int j = 0; j < DIM_IMG_SIZE_Y; ++j) {
                final int val = intValues[pixel++];
                imgData.putFloat((((val >> 16) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
                imgData.putFloat((((val >> 8) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
                imgData.putFloat((((val) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
            }
        }
        long endTime = SystemClock.uptimeMillis();
        Log.d(TAG, "Timecost to put values into ByteBuffer: " + Long.toString(endTime - startTime));
    }

    /** Prints top-K labels, to be shown in UI as the results. */
    private String printTopKLabels() {
        for (int i = 0; i < labelList.size(); ++i) {
            sortedLabels.add(
                    new AbstractMap.SimpleEntry<>(labelList.get(i), labelProbArray[0][i]));
            if (sortedLabels.size() > RESULTS_TO_SHOW) {
                sortedLabels.poll();
            }
        }
        String textToShow = "";
        final int size = sortedLabels.size();
        for (int i = 0; i < size; ++i) {
            Map.Entry<String, Float> label = sortedLabels.poll();
            textToShow = String.format("\n%s : %4.2f",label.getKey(),label.getValue()) + textToShow;
        }
        if (labelProbArray[0][0] > labelProbArray[0][1]) {
            textToShow = textToShow + String.format("\nKondisi anda Sakit");
        }
        if (labelProbArray[0][0] < labelProbArray[0][1]) {
            textToShow = textToShow + String.format("\nKondisi anda Tidak Sakit");
        }
        return textToShow;
    }
}
