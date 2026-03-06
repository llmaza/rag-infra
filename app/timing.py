from time import perf_counter

class Timer:
    def __init__(self):
        self.t0 = perf_counter()

    def ms(self) -> float:
        return (perf_counter() - self.t0) * 1000.0