package com.example.toyapplicationv2;

import androidx.appcompat.app.AppCompatActivity;

import android.annotation.SuppressLint;
import android.content.Context;
import android.graphics.Color;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;

import com.github.mikephil.charting.charts.LineChart;
import com.github.mikephil.charting.components.Legend;
import com.github.mikephil.charting.components.XAxis;
import com.github.mikephil.charting.components.YAxis;
import com.github.mikephil.charting.data.Entry;
import com.github.mikephil.charting.data.LineData;
import com.github.mikephil.charting.data.LineDataSet;
import com.github.mikephil.charting.interfaces.datasets.ILineDataSet;

import java.util.List;

public class MainActivity extends AppCompatActivity implements SensorEventListener {
    // Constants
    private static final String TAG = "MainActivity";
    private static final String[] LABELS = new String[]{"x-value", "y-value", "z-value"};
    private static final int[] COLORS = new int[]{Color.RED, Color.GREEN, Color.BLUE};

    private boolean connected = false;
    private String IP = "";
    private int PORT = 0;

    private static final int COUNTER_LIMIT = 10;
    private String messageBuffer = "";
    int messageCounter = 0;


    // Sensors
    private SensorManager mSensorManager;
    private Sensor mAccelerometer;
    private Sensor sensors;

    // Plots
    private LineChart mChart;
    private Thread thread;
    private boolean plotData = true;

    // TextViews
    TextView zValue, status;

    // Buttons
    Button connButton, disconnButton;

    // EditText
    EditText ipText, portText;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        // Default
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Initialize sensors
        mSensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
        mAccelerometer = mSensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION);
        List<Sensor> sensors = mSensorManager.getSensorList(Sensor.TYPE_ALL);
        for(int i=0; i<sensors.size(); i++){
            Log.d(TAG, "onCreate: Sensor "+ i + ": " + sensors.get(i).toString());
        }

        // Register accelerometer listener
        if (mAccelerometer != null) {
            mSensorManager.registerListener(this, mAccelerometer, SensorManager.SENSOR_DELAY_GAME);
        }

        // Initialize TextViews
        zValue = (TextView) findViewById(R.id.zValue);
        status = (TextView) findViewById(R.id.status);

        // Initialize Buttons
        connButton = (Button) findViewById(R.id.connectButton);
        disconnButton = (Button) findViewById(R.id.disconnectButton);
        connButton.setOnClickListener(new View.OnClickListener() {

            @SuppressLint("SetTextI18n")
            @Override
            public void onClick(View view) {
                connected = true;
                status.setText("Status: connected");
            }
        });
        disconnButton.setOnClickListener(new View.OnClickListener() {

            @SuppressLint("SetTextI18n")
            @Override
            public void onClick(View view) {
                connected = false;
                status.setText("Status: disconnected");
            }
        });

        // Initialize EditText
        ipText = (EditText) findViewById(R.id.ip);
        portText = (EditText) findViewById(R.id.port);
        IP = ipText.getText().toString();
        PORT = Integer.parseInt(portText.getText().toString(), 10);

        // Initialize chart
        mChart = (LineChart) findViewById(R.id.accXYZChart);
        configureChart();

        // Initialize data
        LineData data = new LineData();
        data.setValueTextColor(Color.WHITE); // make data invisible
        mChart.setData(data);

        // Configure legends and axis (only possible after setting data)
        configureLegendsAndAxis();

        // Start plotting
        startPlot();
    }

    @Override
    protected void onPause() {
        super.onPause();

        if (thread != null) {
            thread.interrupt();
        }
        mSensorManager.unregisterListener(this);
    }

    @Override
    public final void onAccuracyChanged(Sensor sensor, int accuracy) {
        // Do something here if sensor accuracy changes.
    }

    @SuppressLint("SetTextI18n")
    @Override
    public final void onSensorChanged(SensorEvent event) {
        zValue.setText("Timestamp: " + System.currentTimeMillis());
        // Plot XYZ values
        if(plotData){
            addEntry(event);
            plotData = false;
        }
        // Send data
        String message = System.currentTimeMillis() + ":" + event.values[0] + "," + event.values[1] + "," + event.values[2] + "\n";
        if (connected) {
            messageBuffer = messageBuffer + message;
            messageCounter += 1;
            if (messageCounter > COUNTER_LIMIT) {
                send(messageBuffer);
                messageBuffer = "";
                messageCounter = 0;
            }
        } else {
            if (messageCounter > 0) {
                send(messageBuffer);
                messageBuffer = "";
                messageCounter = 0;
            }
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        mSensorManager.registerListener(this, mAccelerometer, SensorManager.SENSOR_DELAY_GAME);
    }

    @Override
    protected void onDestroy() {
        mSensorManager.unregisterListener(MainActivity.this);
        thread.interrupt();
        super.onDestroy();
    }

    ////////////////////////////////////////////////////////
    ////              Helper Functions              ////////
    ////////////////////////////////////////////////////////

    public void send(String message) {
        MessageSender messageSender = new MessageSender(IP, PORT);
        messageSender.execute(message);
    }

    private void configureChart() {
        // enable description text
        mChart.getDescription().setEnabled(true);
        mChart.getDescription().setText("Accelerometer XYZ-Values Plot");

        // enable touch gestures
        mChart.setTouchEnabled(true);

        // enable scaling and dragging
        mChart.setDragEnabled(true);
        mChart.setScaleEnabled(true);
        mChart.setDrawGridBackground(false);

        // if disabled, scaling can be done on x- and y-axis separately
        mChart.setPinchZoom(true);

        // set an alternative background color
        mChart.setBackgroundColor(Color.WHITE);
    }

    private void configureLegendsAndAxis() {
        Legend l = mChart.getLegend();

        // modify the legend ...
        l.setForm(Legend.LegendForm.LINE);
        l.setTextColor(Color.WHITE);

        XAxis xl = mChart.getXAxis();
        xl.setTextColor(Color.WHITE);
        xl.setDrawGridLines(true);
        xl.setAvoidFirstLastClipping(true);
        xl.setEnabled(true);

        YAxis leftAxis = mChart.getAxisLeft();
        leftAxis.setTextColor(Color.WHITE);
        leftAxis.setDrawGridLines(false);
        leftAxis.setAxisMaximum(10f);
        leftAxis.setAxisMinimum(0f);
        leftAxis.setDrawGridLines(true);

        YAxis rightAxis = mChart.getAxisRight();
        rightAxis.setEnabled(false);

        mChart.getAxisLeft().setDrawGridLines(false);
        mChart.getXAxis().setDrawGridLines(false);
        mChart.setDrawBorders(false);
    }

    private void addEntry(SensorEvent event) {

        LineData data = mChart.getData();

        if (data != null) {
            // Get data set for x, y, z values
            for (int i = 0; i < 3; i++) {
                ILineDataSet set = data.getDataSetByIndex(i);
                // Create a new set if empty
                if (set == null) {
                    set = createSet(i);
                    data.addDataSet(set);
                }
                // Update data set
                data.addEntry(new Entry(set.getEntryCount(), event.values[i] + 5), i);
            }

            // Update data changes
            data.notifyDataChanged();
            mChart.notifyDataSetChanged();
            mChart.setVisibleXRangeMaximum(150);
            mChart.moveViewToX(data.getEntryCount());
        }
    }

    private LineDataSet createSet(int setIndex) {
        String valueLabel = LABELS[setIndex];
        LineDataSet set = new LineDataSet(null, valueLabel);
        set.setAxisDependency(YAxis.AxisDependency.LEFT);
        set.setLineWidth(3f);
        set.setColor(COLORS[setIndex]);
        set.setHighlightEnabled(false);
        set.setDrawValues(false);
        set.setDrawCircles(false);
        // Tunable parameters
        set.setMode(LineDataSet.Mode.CUBIC_BEZIER);
        set.setCubicIntensity(0.2f);
        return set;
    }

    private void startPlot() {

        if (thread != null){
            thread.interrupt();
        }

        thread = new Thread(new Runnable() {
            @Override
            public void run() {
                while (true){
                    plotData = true;
                    try {
                        Thread.sleep(10); // sampling rate 100/s
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }
            }
        });

        thread.start();
    }
}