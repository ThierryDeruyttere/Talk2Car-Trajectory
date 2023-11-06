class GTProbScheduler:
    def __init__(self, thresholds, additive_factor=0.0, multiplicative_factor=1.0):
        self.thresholds = sorted(thresholds)
        self.additive_factor = additive_factor
        self.multiplicative_factor = multiplicative_factor

        self.threshold_passed_ind = 0
        self.gt_prob = 1.0

    def update(self, current_epoch):
        if self.threshold_passed_ind < len(self.thresholds):
            if current_epoch >= self.thresholds[self.threshold_passed_ind]:
                self.gt_prob = (self.gt_prob + self.additive_factor) * self.multiplicative_factor
                self.threshold_passed_ind += 1
        return self.gt_prob
