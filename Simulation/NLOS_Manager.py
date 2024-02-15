class NLOS_Manager:
    def __init__(self, nlos_bias=2.):
        self.nlos_bias = nlos_bias

    def los(self, i, uwb_measurement):
        los_state = True
        return uwb_measurement, los_state

    def nlos_direct(self, i, uwb_measurement):
        los_state = True
        if i < 100:
            uwb_measurement = uwb_measurement + self.nlos_bias
            los_state = False
        return uwb_measurement, los_state

    def nlos_120(self, i, uwb_measurement):
        los_state = True
        if 120 < i < 310:
            uwb_measurement = uwb_measurement + self.nlos_bias
            los_state = False
        return uwb_measurement, los_state

    def nlos_from_120(self, i, uwb_measurement):
        los_state = True
        if i > 120:
            uwb_measurement = uwb_measurement + self.nlos_bias
            los_state = False
        return uwb_measurement, los_state

    def nlos_from_60(self, i, uwb_measurement):
        los_state = True
        if i > 60:
            uwb_measurement = uwb_measurement + self.nlos_bias
            los_state = False
        return uwb_measurement, los_state

    def nlos_300_600(self, i, uwb_measurement):
        los_state = True
        if 100 < i < 200:
            uwb_measurement = uwb_measurement + self.nlos_bias
            los_state = False
        return uwb_measurement, los_state

    def nlos_50(self, i, uwb_measurement):
        los_state = True
        if i > 200 and (i % 50) == (i % 100):
            uwb_measurement = uwb_measurement + self.nlos_bias
            los_state = False
        return uwb_measurement, los_state

    def nlos_2(self, i, uwb_measurement):
        los_state = True
        if i > 10 and (i % 2 == 0):
            uwb_measurement = uwb_measurement + self.nlos_bias
            los_state = False
        return uwb_measurement, los_state

    def nlos_2_a(self, i, uwb_measurement):
        los_state = True
        if i > 200 and (i % 5 == 0 or (i - 1) % 5 == 0):
            uwb_measurement = uwb_measurement + self.nlos_bias
            los_state = False
        return uwb_measurement, los_state
