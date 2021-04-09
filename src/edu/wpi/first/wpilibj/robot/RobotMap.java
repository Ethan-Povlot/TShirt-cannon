
package edu.wpi.first.wpilibj.robot;

public class RobotMap {
    //Ports for the drive motors
    public static final int leftDrive = 1;
    public static final int rightDrive = 2;
    
    //Ports for compressor
    public static final int compressorOne_relay = 3; //Spike relay input for controlling the compressor
    public static final int compressorOne_pressure_port = 3; //Digital input for checking the accumulator pressure
    public static final int compressorTwo_pressure_port = 5; //Digital input for checking the accumulator pressure
    public static final int compressorTwo_relay = 4;
    
    //Ports for solenoids -- SOLENOID NUMBERS ARE FROM RIGHT TO LEFT ON THE ACTUAL CANNON
    public static final int soleOne = 1; //Port on breakout board for solenoid one
    public static final int soleTwo = 2; //Port on breakout board for solenoid two
    public static final int soleThree = 2; //Port on breakout board for solenoid three
    public static final int soleFour = 2; //Port on breakout board for solenoid four
    
    //Ports for turrett
    public static final int turrett_elevate = 5;
    public static final int turrett_turn = 6;
    public static final int turrett_turn_right_limit = 7;
    public static final int turrett_turn_left_limit = 8;
    public static final int turrett_upper_limit = 9;
    public static final int turrett_lower_limit = 10;
}