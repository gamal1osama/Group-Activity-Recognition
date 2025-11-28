


class BoxInfo:
    def __init__(self, line):
        elements = line.strip().split()
        self.category = elements.pop()

        elements = [int(e) for e in elements]
        
        self.player_id = elements[0]
        del elements[0]

        x1, x2, y1, y2, frame_id, lost, grouping, generated = elements

        self.box = (x1, y1, x2, y2)
        self.frame_id = frame_id
        self.lost = lost
        self.grouping = grouping
        self.generated = generated
        