class physics:

    class rigidBody:
        def __init__(self, attribute):
            self.attribute = attribute

    class satellite:
        def __init__(self, attribute):
            self.attribute = attribute

    def animate(obj):
        print(obj.attribute)

    def getAttribute(obj):
        return obj.attribute

    def setAttribute(obj, newAttribute):
        obj.attribute = newAttribute


goob1 = physics.rigidBody(6)
goob2 = physics.satellite(7)

physics.setAttribute(goob1, 7)

physics.animate(goob1)
