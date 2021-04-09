/*----------------------------------------------------------------------------*/
/* Copyright (c) 2017-2018 FIRST. All Rights Reserved.                        */
/* Open Source Software - may be modified and shared by FRC teams. The code   */
/* must be accompanied by the FIRST BSD license file in the root directory of */
/* the project.                                                               */
/*----------------------------------------------------------------------------*/

package frc.robot;

import edu.wpi.first.cameraserver.CameraServer;
import edu.wpi.first.wpilibj.TimedRobot;
import edu.wpi.first.wpilibj.Relay.Value;
import frc.robot.OI;

public class Robot extends TimedRobot {
	public static void fire() {
		if ((OI.barrelOne.get() == true) && (OI.trigger.get() == true)) {
			OI.relay1.set(Value.kForward);
		} else if ((OI.barrelTwo.get() == true) && (OI.trigger.get() == true)) {
			OI.relay1.set(Value.kReverse);
		} else if ((OI.barrelThree.get() == true) && (OI.trigger.get() == true)) {
			OI.relay2.set(Value.kForward);
		} else if ((OI.barrelThree.get() == true) && (OI.trigger.get() == true)) {
			OI.relay2.set(Value.kReverse);
		}
 	}

    @Override
    public void robotInit() {
	   OI.compressors.set(Value.kOn);
    }

    
    @Override
    public void robotPeriodic() {}

    
    @Override
    public void disabledInit() {}

    @Override
    public void disabledPeriodic() {}

    
    @Override
    public void autonomousInit() {}

    
    @Override
    public void autonomousPeriodic() {}

    @Override
    public void teleopInit() {
		CameraServer.getInstance().startAutomaticCapture();
	}

  
    @Override
    public void teleopPeriodic() {
		OI.left.set(OI.leftDrive.getY());
		OI.right.set(OI.rightDrive.getY());
		OI.turretElevation.set(OI.turret.getY());
		OI.turretRotation.set(OI.turret.getX());


    }

    @Override
    public void testPeriodic() {
    }
}
