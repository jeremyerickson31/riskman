
import unittest
from dashboard.utils import engines
from dashboard.utils import common


class TestEngines(unittest.TestCase):

    def test_forward_interest_rate_repricing_3Pct_1Yr_AA_to_AA(self):
        provider = "Credit Metrics"
        forward_rates = common.get_interest_rate_curves()
        bond_props = {
            "bond_name": "bond_15",
            "par": 100.0,
            "coupon": 0.03,
            "maturity": 1,
            "notional": 11000000.0,
            "rating": "AA",
            "seniority": "Senior Unsecured"}

        bond = engines.Bond(bond_props)  # initialize the bond object
        bond.get_transition_probabilities(provider)  # fetch transition probs for given provider and self.rating
        bond.calc_prices_under_forwards(forward_rates)  # use provided forward rates to do re-pricing
        price = bond.rating_level_prices_pct["AA"]
        self.assertEqual(round(103.0, 10), round(price, 10))

    def test_forward_interest_rate_repricing_3Pct_2Yr_AA_to_AA(self):
        provider = "Credit Metrics"
        forward_rates = common.get_interest_rate_curves()
        bond_props = {
            "bond_name": "bond_15",
            "par": 100.0,
            "coupon": 0.03,
            "maturity": 2,
            "notional": 11000000.0,
            "rating": "AA",
            "seniority": "Senior Unsecured"}

        bond = engines.Bond(bond_props)  # initialize the bond object
        bond.get_transition_probabilities(provider)  # fetch transition probs for given provider and self.rating
        bond.calc_prices_under_forwards(forward_rates)  # use provided forward rates to do re-pricing
        price = bond.rating_level_prices_pct["AA"]
        self.assertEqual(round(102.37288953207911, 10), round(price, 10))

    def test_forward_interest_rate_repricing_3Pct_3Yr_AA_to_AA(self):
        provider = "Credit Metrics"
        forward_rates = common.get_interest_rate_curves()
        bond_props = {
            "bond_name": "bond_15",
            "par": 100.0,
            "coupon": 0.03,
            "maturity": 3,
            "notional": 11000000.0,
            "rating": "AA",
            "seniority": "Senior Unsecured"}

        bond = engines.Bond(bond_props)  # initialize the bond object
        bond.get_transition_probabilities(provider)  # fetch transition probs for given provider and self.rating
        bond.calc_prices_under_forwards(forward_rates)  # use provided forward rates to do re-pricing
        price = bond.rating_level_prices_pct["AA"]
        self.assertEqual(round(100.72202761155926, 10), round(price, 10))


if __name__ == '__main__':
    unittest.main()
