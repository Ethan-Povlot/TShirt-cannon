/*----------------------------------------------------------------------------*/
/* Copyright (c) 2017-2018 FIRST. All Rights Reserved.                        */
/* Open Source Software - may be modified and shared by FRC teams. The code   */
/* must be accompanied by the FIRST BSD license file in the root directory of */
/* the project.                                                               */
/*----------------------------------------------------------------------------*/

package frc.robot;

import edu.wpi.first.wpilibj.Joystick;
import edu.wpi.first.wpilibj.Relay;
import edu.wpi.first.wpilibj.SpeedControllerGroup;
import edu.wpi.first.wpilibj.Victor;
import edu.wpi.first.wpilibj.Relay.Direction;
import edu.wpi.first.wpilibj.buttons.JoystickButton;

public class OI {
	public static Victor leftFront = new Victor(0);
	public static Victor leftBack = new Victor(1);
	public static Victor rightFront = new Victor(2);
	public static Victor rightBack = new Victor(3);
	public static Victor turretRotation = new Victor(4);
	public static Victor turretElevation = new Victor(5);
	public static SpeedControllerGroup left = new SpeedControllerGroup(leftFront, leftBack);
	public static SpeedControllerGroup right = new SpeedControllerGroup(rightFront, rightBack);

	public static Relay compressors = new Relay(0);
	public static Relay relay1 = new Relay(1, Direction.kBoth);
	public static Relay relay2 = new Relay(2, Direction.kBoth);
	
	public static Joystick leftDrive = new Joystick(0);
	public static Joystick rightDrive = new Joystick(1);
	public static Joystick turret = new Joystick(2);

	public static JoystickButton barrelOne = new JoystickButton(turret, 5);
	public static JoystickButton barrelTwo = new JoystickButton(turret, 3);
	public static JoystickButton barrelThree = new JoystickButton(turret, 6);
	public static JoystickButton barrelFour = new JoystickButton(turret, 4);
	public static JoystickButton trigger = new JoystickButton(turret, 0);
}
