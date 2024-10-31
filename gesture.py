class Pose:
    def __init__(self, name, angles):
        self.name = name
        self.angles = angles

    def check(self, angles):
        for i in range(len(self.angles)):
            angle_min = self.angles[i][0]
            angle_max = self.angles[i][1]
            # Angles range from -180 to 180
            if (angles[i] < angle_min or angles[i] > angle_max) and angle_min < angle_max:
                return False
            # Switching the order of min and max inverts the region
            if (angles[i] < angle_min and angles[i] > angle_max) and angle_min > angle_max:
                return False
        return True

    def show(self):
        elbow_l, elbow_r, shoulder_l, shoulder_r = self.angles
        print("Pose          :", self.name)
        print("Left  Elbow   :", elbow_l)
        print("Right Elbow   :", elbow_r)
        print("Left  Shoulder:", shoulder_l)
        print("Right Shoulder:", shoulder_r)

class Gesture:
    def __init__(self, name, poses, action = None):
        self.name = name
        self.poses = poses
        self.action = action
        self.index = 0

    def checkPrev(self, angles):
        if self.index is 0:
            return True
        return self.poses[self.index - 1].check(angles)

    def check(self, angles):
        return self.poses[self.index].check(angles)

    def isDone(self):
        return self.index == len(self.poses)

    def step(self):
        if self.index < len(self.poses):
            self.index += 1

    def reset(self):
        self.index = 0

    def show(self):
        print("Gesture:", self.name)
        print(", ".join([pose.name for pose in self.poses]))


class GestureManager:
    def __init__(self, gestures):
        self.gestures = gestures

    def match(self, angles):
        do_reset = False
        to_reset = []
        matched = None

        for gesture in self.gestures:
            if gesture.check(angles):
                do_reset = True
                gesture.step()
            elif not gesture.checkPrev(angles):
                to_reset.append(gesture)

            if gesture.isDone():
                matched = gesture
                to_reset = self.gestures # Reset everything
                break

        if do_reset:
            for gesture in to_reset:
                gesture.reset()

        return matched
