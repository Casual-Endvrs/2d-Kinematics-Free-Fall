import streamlit as st


def instructions_app():
    instructions_txt = """
    This experiment can be used to measure the acceleration due to gravity.

    ## Equipment requirements:
    1. Phone to record video
    2. Ruler to determine video scale
    3. A ball to toss

    ## Proceedure - Recording (Not required if you use the available default video):
    1. Set the phone up 3-4 meters (12-16 feet) away from a wall
    2. Tape the ruler to the wall
    3. Start recording a video
    4. Toss (or bounce) a ball in an arc as close as possible to the wall
    
    ## Proceedure - Data Processing:
    1. Go to "Data Point Selection" in the left menu
    2. Load the video into the program
    3. Find the first frame with the ball in free flight
    4. Click the center of the ball for each frame while the ball is in free flight, the program will automatically proceed to the next frame once a ball has been selected
    5. From the action list at the top, select "Set Length Markers"
    \t If you're using the default video, the grid in the background is 5cm by 5cm squares.
    6. Click on two points on the ruler where you know the distance between them
    7. Enter the distance between the two points, in centimeters, and hit "Enter"
    8. Click to go to the Visualisation page in the menu on the left
    9. The acceleration due to gravity is listed in the fit report at the bottom of the page

    ## Helpful Hints - To get best results:
    1. Calculated values for the acceleration due to gravity will be higher than expected (9.8 m/s$^2$) if the phone is too close to the wall or the ball is too far from the wall.
    2. Reduce the camera shutter speed to reduce blur from the ball while it is in flight.
    3. The video may be dark as a result of using a short shutter speed. The image brightness can be increased to compensate for this.
    """

    st.write(instructions_txt)
