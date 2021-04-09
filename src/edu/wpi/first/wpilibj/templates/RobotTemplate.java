
package edu.wpi.first.wpilibj.templates;

import edu.wpi.first.wpilibj.DigitalInput;
import edu.wpi.first.wpilibj.IterativeRobot;
import edu.wpi.first.wpilibj.Joystick;
import edu.wpi.first.wpilibj.Relay;
import edu.wpi.first.wpilibj.Timer;
import edu.wpi.first.wpilibj.Victor;
import edu.wpi.first.wpilibj.buttons.JoystickButton;
 
import java.lang.Math;

public class RobotTemplate extends IterativeRobot {
    
    //Define Compressor stuff
    public Relay Compressors = new Relay(1); //Relay for controlling BOTH COMPRESSORS
    public DigitalInput Pressure = new DigitalInput(1); //Pressure Switch
    
    //Define OperatorInterface
    public Joystick rightStick = new Joystick(1);
    public Joystick leftStick = new Joystick(2);
    public JoystickButton fireOne = new JoystickButton(rightStick, 2);
    public JoystickButton fireTwo = new JoystickButton(rightStick, 3);
    public JoystickButton fireThree = new JoystickButton(rightStick, 4);
    public JoystickButton fireFour = new JoystickButton(rightStick, 5);
    public JoystickButton Safety = new JoystickButton(rightStick, 1);
    
    //Define Drive Stuff
    public Victor left = new Victor(1); //Left Motor on PWM 1
    public Victor right = new Victor(2); //Right Motor on PWM 2
    
    
    //Define Solenoid stuff
    public Relay soleOne = new Relay(2); //Solenoid 1
    public Relay soleTwo = new Relay(3); //Solenoid 2
    public Relay soleThree = new Relay(4); //Solenoid 3
    public Relay soleFour = new Relay(5); //Solenoid 4
    
    //Define Turret Stuff
    public Victor elevateTurret = new Victor(3); //Turret elevator
    public Victor turnTurret = new Victor(4); //turret turn motor
    
    //Define motor ramps; cannot use Math.pow() cause 2014 WPILib has conflict
    public double throttlePositive = leftStick.getY() * leftStick.getY() * leftStick.getY();
    public double turnPositive = leftStick.getX() * leftStick.getX() * leftStick.getX();
    
    public void robotInit() {
    }

    public void autonomousPeriodic() {
    }

    public void teleopPeriodic() {
        turnTurret.set(rightStick.getX());
        elevateTurret.set(rightStick.getX());
        if (leftStick.getY() > 0 && leftStick.getX() > 0) {
            left.set(throttlePositive + turnPositive); 
            right.set(throttlePositive - turnPositive);
        }
        else if (leftStick.getY() < 0 && leftStick.getX() > 0) {
            left.set((throttlePositive * -1) + turnPositive); 
            right.set((throttlePositive * -1) - turnPositive);
        }
        else if (leftStick.getY() < 0 && leftStick.getX() < 0) {
            left.set((throttlePositive * -1) + (turnPositive * -1)); 
            right.set((throttlePositive * -1) - (turnPositive * -1));
        }
        else if (leftStick.getY() < 0 && leftStick.getX() < 0) {
            left.set(throttlePositive + (turnPositive * -1)); 
            right.set(throttlePositive - (turnPositive * -1));
        }
        
        elevateTurret.set(rightStick.getY());
        turnTurret.set(rightStick.getX());
      
        if (Pressure.get()) {
            Compressors.set(Relay.Value.kOn);
        } 
        else {
            Compressors.set(Relay.Value.kOff);
        }
        if (fireOne.get() && Safety.get()) {
            soleOne.set(Relay.Value.kOn);
            Timer.delay(0.5);
        }
        else if (fireTwo.get() && Safety.get()) {
            soleTwo.set(Relay.Value.kOn);
            Timer.delay(0.5);
        }
        else if (fireThree.get() && Safety.get()) {
            soleThree.set(Relay.Value.kOn);
            Timer.delay(0.5);
        }
        else if (fireFour.get() && Safety.get()) {
            soleFour.set(Relay.Value.kOn);
            Timer.delay(0.5);
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