from matplotlib.scale import ScaleBase,register_scale
from matplotlib.ticker import FuncFormatter,AutoLocator,Locator
from matplotlib.transforms import Transform
import numpy as np

class InverseSquareScale(ScaleBase):
    name = 'isqr'
    """
    A matplotlib Scale object which applies the inverse
    square transform to an axis. 
    """
    def get_transform(self):
        return self.InverseSquareTransform()

    def set_default_locators_and_formatters(self, axis):
        formatter = FuncFormatter(
            lambda x, pos=None: f"{x:0.2f}"
        )
        locator = InverseSquareScale.InverseSquareTickLocator()
        axis.set(
            major_locator=locator,
            major_formatter=formatter,
            minor_formatter=formatter,
        )

    class InverseSquareTransform(Transform):
        input_dims = 1
        output_dims = 1
        eps = 1e-300

        def transform_non_affine(self, values):
            out = values.copy()
            idx = np.abs(out) > self.eps
            pos = (out > 0) & idx
            out[pos] = np.reciprocal(np.square(out[pos]))
            neg = (out < 0) & idx
            out[neg] = -np.reciprocal(np.square(-out[neg]))
            return out

        def inverted(self):
            return InverseSquareScale.InverseSqrtTransform()

    class InverseSqrtTransform(Transform):
        input_dims = 1
        output_dims = 1
        eps = 1e-300

        def transform_non_affine(self, values):
            out = values.copy()
            idx = np.abs(out) > self.eps
            pos = (out > 0) & idx
            out[pos] = np.reciprocal(np.sqrt(out[pos]))
            neg = (out < 0) & idx
            out[neg] = -np.reciprocal(np.sqrt(-out[neg]))
            return out

        def inverted(self):
            return InverseSquareScale.InverseSquareTransform()

    class InverseSquareTickLocator(Locator):
        nticks = 6
        eps = 1e-5
        def tick_values(self, vmin, vmax, nticks=None):
            vmin,vmax = sorted([vmin, vmax])
            if nticks is None:
                nticks = self.nticks
            if vmin < 0.:
                vmin = 100
            ticks = np.linspace(vmin**-2.0, vmax**-2.0, self.nticks+2)[1:-1]
            ticks = ticks**-0.5
            return ticks

        def __call__(self, *args, **kwargs):
            vmin, vmax = self.axis.get_view_interval()
            out = self.tick_values(vmin, vmax)
            return out


if __name__=='__main__':
    from matplotlib.pylab import *
    x = y = np.linspace(25.**-2, 2**-2, 100)**-0.5
    y = np.abs(y)

    f, ax = plt.subplots()
    ax.scatter(x, y, c=y)
    ax.set_xscale(InverseSquareScale(ax))
    plt.show()

