from AriaPy import *
robot = Aria.connectToRobot()
robot.enableMotors()
robot.runAsync()
#set right wheel speed 200 and left 0.
def turnleft():
  robot.setVel2(0,200)


robot.addUserTask(turnleft, 'drive')

ArUtil.sleep(2000);

Aria.exit(0)
