Keyframe function:

self.keyframes = []
    add keyframe at current timePosition of timeline (button)
        self.keyframes["valueType"]["timePosition"] = value
    del keyframe at current timePosition of timeline
        try del self.keyframes["valueType"]["timePosition"]

keyframe draw function:
    for i, item in self.keyframes:
    print i
    print item


