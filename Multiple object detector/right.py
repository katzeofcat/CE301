from AriaPy import *
robot = Aria.connectToRobot()
robot.enableMotors()
robot.runAsync()

def turnRight():
  robot.setVel2(200,0)


robot.addUserTask(turnRight, 'drive')

ArUtil.sleep(2000);

Aria.exit(0)
