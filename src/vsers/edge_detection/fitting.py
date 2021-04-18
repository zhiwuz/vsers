from scipy import interpolate


class ExtrapolateFitting(object):
    def __init__(self, fit_method='UnivariateSpline'):
        self.order = 4
        self.fit_method = fit_method
        self.fit_function = None
        self.first_derivative = None
        self.second_derivative = None
        self.third_derivative = None
        self.fourth_derivative = None

    def fit(self, x_axis, y_axis):
        if self.fit_method == 'UnivariateSpline':
            self.fit_function = interpolate.UnivariateSpline(x_axis, y_axis, k=self.order)
        elif self.fit_method == 'interp1d':
            self.fit_function = interpolate.interp1d(x_axis, y_axis, fill_value="extrapolate")
        return self.fit_function

    def get_first_derivative(self):
        if self.fit_method == 'UnivariateSpline':
            self.first_derivative = self.fit_function.derivative()

    def get_second_derivative(self):
        if self.fit_method == 'UnivariateSpline':
            self.second_derivative = self.first_derivative.derivative()

    def get_third_derivative(self):
        if self.fit_method == 'UnivariateSpline':
            self.third_derivative = self.second_derivative.derivative()

    def get_fourth_derivative(self):
        if self.fit_method == 'UnivariateSpline':
            self.fourth_derivative = self.third_derivative.derivative()

    def get_derivatives(self):
        self.get_first_derivative()
        self.get_second_derivative()
        self.get_third_derivative()
        self.get_fourth_derivative()
