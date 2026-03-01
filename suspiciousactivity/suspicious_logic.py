import numpy as np

class SuspiciousDetector:
    def __init__(self, loiter_threshold=30, speed_threshold=10):
        self.positions = {}  # person_id: [last 30 positions]
        self.loiter_threshold = loiter_threshold
        self.speed_threshold = speed_threshold

    def update(self, person_id, position):
        if person_id not in self.positions:
            self.positions[person_id] = []
        self.positions[person_id].append(position)
        if len(self.positions[person_id]) > 30:
            self.positions[person_id].pop(0)

    def check(self, person_id):
        positions = self.positions.get(person_id, [])
        if len(positions) < 5:
            return None  # Not enough data

        # Compute average speed
        diffs = [np.linalg.norm(np.array(positions[i+1]) - np.array(positions[i])) for i in range(len(positions)-1)]
        avg_speed = sum(diffs) / len(diffs)

        if avg_speed > self.speed_threshold:
            return "running"
        elif avg_speed < 1.0 and len(positions) >= self.loiter_threshold:
            return "loitering"
        else:
            return None
