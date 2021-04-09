
package edu.wpi.first.wpilibj.subsystems;

import edu.wpi.first.wpilibj.DigitalInput;
import edu.wpi.first.wpilibj.Victor;
import edu.wpi.first.wpilibj.robot.RobotMap;


public class Turrett {
    //motors
    public static Victor elevation = new Victor(RobotMap.turrett_elevate); //height motor on the turrett
    public static Victor turn = new Victor(RobotMap.turrett_turn); //turn motor on the turrett
    
    //Limit Switches
    public static DigitalInput leftLimit = new DigitalInput(RobotMap.turrett_turn_left_limit);
    public static DigitalInput rightLimit = new DigitalInput(RobotMap.turrett_turn_right_limit);
    public static DigitalInput upperLimit = new DigitalInput(RobotMap.turrett_upper_limit);
    public static DigitalInput lowerLimit = new DigitalInput(RobotMap.turrett_lower_limit);
    
}
