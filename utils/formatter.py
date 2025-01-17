from matplotlib.ticker import Formatter
import datetime as dt
import time

class DateFormatter(Formatter):
    """
    Formats dates for plotting
    """
    def __init__(self, dates, fmt='%Y-%m-%d'):
        self.dates = dates
        self.fmt = fmt

    def __call__(self, x, pos=0):
        'Return the label for time x at position pos'
        ind = int(x)
        if ind >= len(self.dates) or ind < 0:
            return ''
        else:
            return self.dates[ind].strftime(self.fmt)

