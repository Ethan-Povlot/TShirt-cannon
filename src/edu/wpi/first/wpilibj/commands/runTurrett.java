
package edu.wpi.first.wpilibj.commands;

import edu.wpi.first.wpilibj.Timer;
import edu.wpi.first.wpilibj.robot.Robot;
import edu.wpi.first.wpilibj.subsystems.Turrett;

public class runTurrett {
    
    public static void run () {
        while(Robot.OI.turrettControl.get() == true) { //while the turrett control button is pressed
            Robot.turrett.elevation.set(Robot.OI.rightStick.getY()); //set the height to the y axis of the rightstick
            Robot.turrett.turn.set(Robot.OI.rightStick.getX()); //set the turnn to the x axis of the rightstick
            if (Robot.turrett.upperLimit.get()) {
                Robot.turrett.elevation.set(-.15);
                    Timer.delay(0.25);
            }
            if (Robot.turrett.lowerLimit.get()) {
                Robot.turrett.elevation.set(.15);
                    Timer.delay(0.25);
            }
            if (Robot.turrett.rightLimit.get()) {
                Robot.turrett.turn.set(-.15);
                    Timer.delay(0.25);
            }
            if (Robot.turrett.leftLimit.get()) {
                Robot.turrett.turn.set(0.15);
                    Timer.delay(.25);
            }
            //set the robot not to drive
            Robot.driveTrain.leftDrive.set(0.0);
            Robot.driveTrain.rightDrive.set(0.0);
            Robot.driveTrain.leftDrive.set(0.0);
        }
    }
}
