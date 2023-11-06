import numpy as np
from typing import List


class Meter:
    def __init__(self, name: str = "meter", mul_factor=1.0, unit=""):
        self.name = name
        self.mul_factor = mul_factor
        self.unit = unit
        self.values = []

    def update(self, value):
        if isinstance(value, List):
            assert all(isinstance(item, float) for item in value), "All values in the list must be of type 'float'"
            self.values.extend(value)
        elif isinstance(value, float):
            self.values.append([value])
        else:
            raise ValueError("Argument 'value' needs to be a list of floats or float.")

    def mean(self):
        values = np.array(self.values)
        return values.mean()

    def std(self):
        values = np.array(self.values)
        return values.std()

    def median(self):
        values = np.array(self.values)
        return np.median(values)

    def errorbar(self):
        return self.std() / np.sqrt(len(self.values))

    def report(self):
        print(
            f"Metric: {self.name}; Mean: {self.mean() * self.mul_factor:.2f} +/- {self.errorbar() * self.mul_factor:.2f} {self.unit}; Median: {self.median() * self.mul_factor:.2f} {self.unit}"
        )



