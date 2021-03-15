from scipy import interpolate


class ExtrapolateFitting(object):
    def __init__(self):
        self.fit_function = None
        self.first_derivative = None
        self.second_derivative = None
        self.third_derivative = None
        self.fourth_derivative = None

    def fit(self, x_axis, y_axis):
        self.fit_function = interpolate.interp1d(x_axis, y_axis, fill_value="extrapolate")
        return self.fit_function

    def get_first_derivative(self):
        self.first_derivative = self.fit_function.derivative()

    def get_second_derivative(self):
        self.second_derivative = self.first_derivative.derivative()

    def get_third_derivative(self):
        self.third_derivative = self.second_derivative.derivative()

    def get_fourth_derivative(self):
        self.fourth_derivative = self.third_derivative.derivative()

    def get_derivatives(self):
        self.get_first_derivative()
        self.get_second_derivative()
        self.get_third_derivative()
        self.get_fourth_derivative()
