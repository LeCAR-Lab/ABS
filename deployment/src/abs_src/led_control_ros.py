import rospy
from std_msgs.msg import Float64MultiArray
import serial

def led_control_cb(ser, state):
    rospy.loginfo("Received state: " + str(state.data[0]))
    if state.data[0] > -0.05:
        ser.write(b'\xA0\x01\x00\xA1')
    else:
        ser.write(b'\xA0\x01\x01\xA2')
        

def shutdown_handler(ser):
    ser.close()

if __name__ == "__main__":
    PORT = "/dev/state_led"
    rospy.init_node('led_controller_node')
    rospy.loginfo("Starting led_controller_node")
    
    # Open the COM port using PySerial
    ser = serial.Serial(PORT, baudrate=9600)  # Modify the baudrate accordinglyr
    rospy.loginfo("Opened serial port: " + PORT)
    
    
    rospy.on_shutdown(lambda : shutdown_handler(ser))
    rospy.loginfo("Registered shutdown handler")
    
    rospy.Subscriber("RA_value", Float64MultiArray, lambda state: led_control_cb(ser, state))
    rospy.loginfo("Subscribed to /RA_value")
    
    rospy.spin()
