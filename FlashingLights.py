#!/usr/bin/env Python
# This is a hacked-together port of the C code I wrote for the Arduino. Thus it is fairly non-Pythonic. I'll do some legitimate porting later when it actually matters. It does not have the blazing speed of the C code, but it does its best
# -SCT (09/27/14)

# ACTUALLY possibly too slow to use for 5 ms pulses. This is a shame. digitalWrite takes like 8 or 9 ms

# nanpy is the package with the Arduino control via python
from nanpy import (ArduinoApi, SerialManager)
import time
import datetime
import multiprocessing
import os
import wiringpi2 as wp
# will automatically find the arduino
# if there's more than one Arduino and you want a specific location, write SerialManager('location') (not as a string)
connection = SerialManager()
# a is the name of our Arduino at location connection
a = ArduinoApi(connection=connection)
# Define the names of the pins that we're using to control the lights
indicator = 9
pinStart = 10
pinEnd = 14
numPins = pinEnd-pinStart

# timescale is 1000 if we're using milliseconds, 1000000 if we're using microseconds (this is just a quality of life change to make the code easy to deal with if the implementation changes; thanks Stephen from a year ago!)
timescale = 1

# startTime is how long from program initation until it should start (this is a holdover from the C code, since the RasPi should be able to do all the clockwork on its own. I may change this eventually)
startTime = 0
duration = 1200 #20 minutes
endOfDays = startTime + duration
# Frequency for each light, organized as an array. The first entry corresponds to the "first" set of lights
freq = [5,0,5,0]
shuffle = [1, 0, 3, 2]
# Similar for pulse width (except this is in milliseconds)
pulseWidth = [5,0,5,0]
flipTime = 600 # 10 minutes
flip = True
hasStarted = False
# This is like the void setup() method in C
for pin in range(pinStart,pinEnd):
    a.pinMode(pin, a.OUTPUT)
    a.digitalWrite(pin,a.HIGH)
a.pinMode(indicator,a.OUTPUT)

# Define the process by which lights flick on and off
# number = pin number
# freq = frequency for that pin (in Hz)
# width = duration of a single flash
# cycles = number of times you want to repeat this
def flashLights(number, freq, width, cycles):
    if not freq == 0:
        for i in range(0,cycles):
            a.digitalWrite(number,a.LOW)
	    t_ = time.time()
#            wp.delay( float(width/timescale) )
	    wp.delayMicroseconds( 1000*(width/timescale) )
            a.digitalWrite(number,a.HIGH)
	    t = time.time()
#	    wp.delay( float((1-width)/(timescale*freq)) )
	    wp.delayMicroseconds( 1000*(1000 - width)/(timescale*freq) )
	    print(t-t_)

# Here's a function for running the pin processes
def runPinProcesses():
    for n in range(0,numPins):
        if not freq[n] == 0:
            p = multiprocessing.Process(target=flashLights, args=(n+pinStart, freq[n], pulseWidth[n],flipTime*freq[n]))
	    p.start()
	# Okay, I'm not proud of this part. It's a little bit hacky. Basically I tell it to wait 1 second in between starting each type of light. The Arduino gets confused if I manage to sync up the processes too closely
	    time.sleep(1)

# This hideous non-Pythonic port is of the loop() method from the Arduino
time.sleep(startTime)
print("Starting the loop")
numCycs = int(duration/flipTime)    
for cyc in range(0,numCycs):    
    runPinProcesses()
    time.sleep(flipTime)
    freq = [ freq[i] for i in shuffle]
    pulseWidth = [ pulseWidth[i] for i in shuffle]
    runPinProcesses()
    time.sleep(flipTime)
    freq = [ freq[i] for i in shuffle]
    pulseWidth = [ pulseWidth[i] for i in shuffle]

