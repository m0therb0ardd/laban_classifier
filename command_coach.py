def loop():
    try:
        with open("command.txt", "r") as f:
            command = f.read().strip()
    except:
        command = "NONE"

    if command == "LEFT":
        robot.set_wheel_speeds(-0.2, 0.2)  # move left by rotating in place
        robot.sleep(1.0)
        robot.stop()

    elif command == "RIGHT":
        robot.set_wheel_speeds(0.2, -0.2)  # rotate right
        robot.sleep(1.0)
        robot.stop()

    else:
        # Idle or do something cute
        robot.set_wheel_speeds(0, 0)
