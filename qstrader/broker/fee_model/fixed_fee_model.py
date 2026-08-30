from qstrader.broker.fee_model.fee_model import FeeModel


class FixedFeeModel(FeeModel):
    """
    A FeeModel subclass that produces a percentage cost
    for tax and commission.

    Parameters
    ----------
    commission_pct : `float`, optional
        The percentage commission applied to the consideration.
        0-100% is in the range [0.0, 1.0]. Hence, e.g. 0.1% is 0.001
    tax_pct : `float`, optional
        The percentage tax applied to the consideration.
        0-100% is in the range [0.0, 1.0]. Hence, e.g. 0.1% is 0.001
    """

    def __init__(self, commission=0.0):
        super().__init__()
        self.commission = commission

    def _calc_commission(self, asset=None, quantity=None, consideration=None, broker=None):
        return self.commission

    def _calc_tax(self, asset, quantity, consideration, broker=None):
        return 0

    def calc_total_cost(self, asset, quantity, consideration, broker=None):
        return self._calc_commission()
