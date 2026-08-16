from scipy import randn
import numpy as np


class NLOS_Manager:
    def __init__(self, nlos_bias=2.):
        self.nlos_bias = np.random.uniform(1, 5)
        self.nlos_dc_bias = np.random.uniform(0, 1)
        self.nlos_dc_frequency = np.random.uniform(0.01, 0.001)
        self.nlos_sigma = np.random.uniform(0.1, 0.5)
        self.los_state = True

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

    def nlos_10(self, i, uwb_measurement):
        los_state = True
        if i > 100 and (i % 100 == 0):
            uwb_measurement = uwb_measurement + self.nlos_bias
            los_state = False
        return uwb_measurement, los_state

    def set_nlos_variables(self):
        self.nlos_bias = np.random.uniform(1, 5)
        self.nlos_dc_bias = np.random.uniform(0, 1)
        self.nlos_dc_frequency = np.random.uniform(0.01, 0.001)
        self.nlos_sigma = np.random.uniform(0.1, 0.5)

    def nlos_dc(self, i, uwb_measurement):
        los_state = True

        uwb_measurement = uwb_measurement + self.nlos_bias + self.nlos_dc_bias * np.sin(2 * np.pi * i * self.nlos_dc_frequency) + np.random.randn(1)[0] * self.nlos_sigma
        los_state = False
        return uwb_measurement, los_state

    def nlos_1(self, i, uwb_measurement):

        if (i % 2 == 0 ):
            uwb_measurement = uwb_measurement + self.nlos_bias
            self.los_state = False
        else:
            self.los_state = True
            self.set_nlos_variables()
        return uwb_measurement, self.los_state

    def nlos_hz(self, i, uwb_measurement, number):
        if i % number == 0:
            self.los_state = not self.los_state
        if self.los_state:
            self.set_nlos_variables()
        else:
            uwb_measurement = uwb_measurement + self.nlos_bias + self.nlos_dc_bias * np.sin(2 * np.pi * i * self.nlos_dc_frequency) + np.random.randn(1)[0] * self.nlos_sigma

        return uwb_measurement, self.los_state


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    t = np.linspace(0, 1000, 1000)  # 0 to 10 seconds, 1000 points
    nlos_man = NLOS_Manager()
    nlos_man.set_nlos_variables()
    measurement = []
    for i in range(1000):
        mes,_   = nlos_man.nlos_hz(i, 0, 4)
        measurement.append(mes)




    # Plotting the signal
    plt.figure(figsize=(10, 6))
    plt.plot(t, measurement, label="DC + Variable AC")
    # plt.plot(t, dc, label="DC Component (slow change)", linestyle='--', color='orange')
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Slow Changing DC + Variable AC")
    plt.legend()
    plt.grid()
    plt.show()