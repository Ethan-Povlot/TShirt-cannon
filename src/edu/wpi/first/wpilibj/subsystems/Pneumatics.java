
package edu.wpi.first.wpilibj.subsystems;

import edu.wpi.first.wpilibj.Compressor;
import edu.wpi.first.wpilibj.Solenoid;
import edu.wpi.first.wpilibj.Timer;
import edu.wpi.first.wpilibj.robot.RobotMap;

public class Pneumatics {
   //Define Objects
   public static Compressor compressorOne = new Compressor(RobotMap.compressorOne_pressure_port, RobotMap.compressorOne_relay); //Create a compressor object with the ports defined in RobotMap()
   public static Compressor compressorTwo = new Compressor(RobotMap.compressorTwo_pressure_port, RobotMap.compressorTwo_relay); //Create a compressor object with the ports defined in RobotMap()
   public static Solenoid one = new Solenoid(RobotMap.soleOne); //Define Solenoid #1
   public static Solenoid two = new Solenoid(RobotMap.soleTwo); //Define Solenoid #2
   public static Solenoid three = new Solenoid(RobotMap.soleThree); //Define Solenoid #3
   public static Solenoid four = new Solenoid(RobotMap.soleFour); //Define Solenoid #4
   public static Timer soleTimer = new Timer();
   public static boolean isRunning = false;
   public static boolean oneFired;
   public static boolean twoFired;
   public static boolean threeFired;
   public static boolean fourFired;
   
//Void for running the compressor
   public static void compressorStart() {
       compressorOne.start(); //start the compressor, which creates instances of the relay and the digital port. Will automatically check if pressure exceeds limit set by the high pressure switch, it will shut off.
       compressorTwo.start(); //start the compressor, which creates instances of the relay and the digital port. Will automatically check if pressure exceeds limit set by the high pressure switch, it will shut off.
   }
   public static void compressorStop() {
       compressorOne.stop(); //turn the compressor off
       compressorTwo.stop(); //turn the compressor off
   }
   
   public static void fireSolenoid(Solenoid solenoid, int i){ //Command to fire any solenoid
       soleTimer.start(); //start the timer
       //set all solenoids to be off
       setSolenoids(false, false, false, false);
       while (soleTimer.get() <= 0.5) { 
           solenoid.set(true);
       }
       isRunning = true;
       switch(i){
           case 1:
               oneFired = true;
               break;
           case 2:
               twoFired = true;
               break;
           case 3:
               threeFired = true;
               break;
           case 4:
               fourFired = true;
           default:
               break;
       }
       //oneFired, etc can be replaced by just using each solenoid's get function when needed
   }
   
   public static void setSolenoids(boolean value_1, boolean value_2, boolean value_3, boolean value_4){ //Set solenoid values
       one.set(value_1);
       two.set(value_2);
       three.set(value_3);
       four.set(value_4);
    }
}
