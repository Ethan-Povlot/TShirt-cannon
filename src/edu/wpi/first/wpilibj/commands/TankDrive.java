
package edu.wpi.first.wpilibj.commands;

import edu.wpi.first.wpilibj.Victor;
import edu.wpi.first.wpilibj.robot.OI;
import edu.wpi.first.wpilibj.robot.Robot;
import edu.wpi.first.wpilibj.subsystems.DriveTrain;

public class TankDrive {
    //create a drive function
    public static void drive () {
        DriveTrain.leftDrive.set(OI.leftStick.getY()); //set the left victor to the y axis if left joystick
        DriveTrain.rightDrive.set(OI.rightStick.getY()); //set the right victor to the y axis of the right joystick
    }
}
