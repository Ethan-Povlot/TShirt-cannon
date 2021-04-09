
package edu.wpi.first.wpilibj.subsystems;

import edu.wpi.first.wpilibj.Victor;
import edu.wpi.first.wpilibj.robot.RobotMap;

public class DriveTrain {
    //Drive Motors
    public static Victor leftDrive = new Victor(RobotMap.leftDrive);
    public static Victor rightDrive = new Victor(RobotMap.rightDrive);
}
