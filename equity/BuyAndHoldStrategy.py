DATA_DIR = '../data'
import MomentumStrategy
from utils import MyFormatter

class BuyandHold:
	def __init__(self, principal=1000000):
		self.prices = None
		self.dates = None
		self._load_data

		self.principal=principal
		self.price_data = self.prices.iloc[:, [2,4,6]].values

	def calc_strategy(self):
		mut = MomentumStrategy.momentum_signal(prices, 1, False)+1
		mut[0] = 1

		pnl_hold = np.sum((np.cumprod(mut, axis=0)-1) / 3*self.principal, axis=1)
		pnl_hold = pnl_hold - np.roll(pnl_hold, 1)
		pnl_hold[0] = 0


	def _load_data(self):
        self.prices = pd.read_csv(DATA_DIR, parse_dates=[0])
        self.dates = self.prices.iloc[:,0].apply(lambda x: pd.to_datetime(x))

if __name__ == '__main__':
	# Testing class
	plt.style.use("ggplot")
	formatter = MyFormatter(dates)
	fig, ax = plt.subplots(figsize=(11, 7))
	ax.xaxis.set_major_formatter(formatter)
	ax.plot(np.arange(pnl_hold.shape[0]), np.cumsum(pnl_hold))
	fig.autofmt_xdate()
	plt.show()

	print(annual_sharpe(pnl_hold))