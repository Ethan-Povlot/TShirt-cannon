
package edu.wpi.first.wpilibj.commands;

import edu.wpi.first.wpilibj.robot.Robot;
import edu.wpi.first.wpilibj.subsystems.Pneumatics;

public class Solenoids {
    public boolean isRunning;
    public static void solenoidFire() { //Command to fire the solenoids based on button input
        if(Robot.OI.soleOne.get() == true) { //if button #1 is being pressed
            Pneumatics.fireSolenoid(Robot.pneumatics.one, 1); //fireone() command from the pneumatics subsystem
        }
        if(Robot.OI.soleTwo.get() == true) { //if button #2 is being pressed
            Pneumatics.fireSolenoid(Robot.pneumatics.two, 2); //firetwo() command from pneumatics subsystem
        }
        if(Robot.OI.soleThree.get() == true) { //if button #3 is being pressed
            Pneumatics.fireSolenoid(Robot.pneumatics.three, 3); //firethree() command from pneumatics subsystem
        }
        if(Robot.OI.soleFour.get() == true) { //if button #4 is being pressed
            Pneumatics.fireSolenoid(Robot.pneumatics.four, 4); //firefour() command from pneumatics subsystem
        }
    }
}
