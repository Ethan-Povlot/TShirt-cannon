
package edu.wpi.first.wpilibj.robot;

import edu.wpi.first.wpilibj.Joystick;
import edu.wpi.first.wpilibj.buttons.JoystickButton;

public class OI {
    //Joysticks
    public static Joystick leftStick = new Joystick(1);
    public static Joystick rightStick = new Joystick(2);
    
    //Joystick Buttons
    public static JoystickButton soleOne = new JoystickButton(leftStick, 1); //Button One on the lefthand joystick controls solenoid #1
    public static JoystickButton soleTwo = new JoystickButton(leftStick, 2); //Button Two on the lefthand joystick controls solenoid #2
    public static JoystickButton soleThree = new JoystickButton(leftStick, 3); //Button Three on the lefthand joystick controls solenoid #3
    public static JoystickButton soleFour = new JoystickButton(leftStick, 4); //Button Four on the lefthand joystick controls solenid #4
    public static JoystickButton turrettControl = new JoystickButton(rightStick, 1); //Button one on rightstick for controling turrett
    public static JoystickButton safety = new JoystickButton(leftStick, 0); //safety for firing shirts on the left joystick
}
