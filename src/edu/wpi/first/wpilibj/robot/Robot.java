/*----------------------------------------------------------------------------*/
/* Copyright (c) FIRST 2008. All Rights Reserved.                             */
/* Open Source Software - may be modified and shared by FRC teams. The code   */
/* must be accompanied by the FIRST BSD license file in the root directory of */
/* the project.                                                               */
/*----------------------------------------------------------------------------*/

package edu.wpi.first.wpilibj.robot;


import edu.wpi.first.wpilibj.DigitalInput;
import edu.wpi.first.wpilibj.IterativeRobot;
import edu.wpi.first.wpilibj.Joystick;
import edu.wpi.first.wpilibj.Relay;
import edu.wpi.first.wpilibj.Timer;
import edu.wpi.first.wpilibj.Victor;
import edu.wpi.first.wpilibj.buttons.JoystickButton;


public class Robot extends IterativeRobot {
    //Operator Interface
    public Joystick leftDrive = new Joystick(1); //left-hand joystick
    public Joystick rightDrive = new Joystick(2); //right-hand joystick
    public Joystick turretDrive = new Joystick(3);
    public JoystickButton fireOne = new JoystickButton(turretDrive, 2);
    public JoystickButton fireTwo = new JoystickButton(turretDrive, 3);
    public JoystickButton fireThree = new JoystickButton(turretDrive, 4);
    public JoystickButton fireFour = new JoystickButton(turretDrive, 5);
    public JoystickButton Safety = new JoystickButton(turretDrive, 1);
    
    //Compressors
    DigitalInput pressure = new DigitalInput(1); //Pressure Switch object
    Relay compressors = new Relay(1); //Relay that controls both compressors
    
    //DriveTrain
    Victor leftMotor = new Victor(1); //leftDrive
    Victor rightMotor = new Victor(2); //rightDrive
    
    //turret subsystem
    public Victor elevation = new Victor(3); //motor that raises the turret
    public Victor turn = new Victor(4); //motor that lowers the turret
    public DigitalInput upperLinit = new DigitalInput(2); //elevator upper limit switch
    public DigitalInput lowerLimit = new DigitalInput(3); //elevator lower limit switch
    public DigitalInput leftLimit = new DigitalInput(4); //turret turn axis left limit switch
    public DigitalInput rightLimit = new DigitalInput(5); //turret turn axis right limit switch
    
    //Solenoid Subsystem
    public Relay soleOne = new Relay(2);
    public Relay soleTwo = new Relay(3);
    public Relay soleThree = new Relay(4);
    public Relay soleFour = new Relay(5);
    
    public void robotInit() {

    }


    public void autonomousPeriodic() {

    }

    public void teleopPeriodic() {
        double delay = 0.15; //timer delay
        //Drive code
        leftMotor.set(leftDrive.getY()); //set the left motor to the leftSpeed
        rightMotor.set(rightDrive.getY()); //set the right motor to the rightSpeed
        
        if (Safety.get() && fireOne.get()) {
            soleOne.set(Relay.Value.kOn);
            Timer.delay(delay);
        }
        else if (Safety.get() && fireTwo.get()) {
            soleTwo.set(Relay.Value.kOn);
            Timer.delay(delay);
        }
        else if (Safety.get() && fireThree.get()) {
            soleThree.set(Relay.Value.kOn);
            Timer.delay(delay);
        }
        else if (Safety.get() && fireFour.get()) {
            soleFour.set(Relay.Value.kOn);
            Timer.delay(delay);
        }
        else {
            soleOne.set(Relay.Value.kOff);
            soleTwo.set(Relay.Value.kOff);
            soleThree.set(Relay.Value.kOff);
            soleFour.set(Relay.Value.kOff);
        }
    }
    
    public void testPeriodic() {
    
    }
    
}
