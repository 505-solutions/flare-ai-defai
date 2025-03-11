import time
import requests
import pandas as pd

class FTSOFeed:
    def __init__(self):

        self.feed_list = {
            "FLR/USD": "0x01464c522f55534400000000000000000000000000",
            "SGB/USD": "0x015347422f55534400000000000000000000000000",
            "BTC/USD": "0x014254432f55534400000000000000000000000000",
            "XRP/USD": "0x015852502f55534400000000000000000000000000",
            "LTC/USD": "0x014c54432f55534400000000000000000000000000",
            "XLM/USD": "0x01584c4d2f55534400000000000000000000000000",
            "DOGE/USD": "0x01444f47452f555344000000000000000000000000",
            "ADA/USD": "0x014144412f55534400000000000000000000000000",
            "ALGO/USD": "0x01414c474f2f555344000000000000000000000000",
            "ETH/USD": "0x014554482f55534400000000000000000000000000",
            "FIL/USD": "0x0146494c2f55534400000000000000000000000000",
            "ARB/USD": "0x014152422f55534400000000000000000000000000",
            "AVAX/USD": "0x01415641582f555344000000000000000000000000",
            "BNB/USD": "0x01424e422f55534400000000000000000000000000",
            "POL/USD": "0x01504f4c2f55534400000000000000000000000000",
            "SOL/USD": "0x01534f4c2f55534400000000000000000000000000",
            "USDC/USD": "0x01555344432f555344000000000000000000000000",
            "USDT/USD": "0x01555344542f555344000000000000000000000000",
            "XDC/USD": "0x015844432f55534400000000000000000000000000",
            "TRX/USD": "0x015452582f55534400000000000000000000000000",
            "LINK/USD": "0x014c494e4b2f555344000000000000000000000000",
            "ATOM/USD": "0x0141544f4d2f555344000000000000000000000000",
            "DOT/USD": "0x01444f542f55534400000000000000000000000000",
            "TON/USD": "0x01544f4e2f55534400000000000000000000000000",
            "ICP/USD": "0x014943502f55534400000000000000000000000000",
            "SHIB/USD": "0x01534849422f555344000000000000000000000000",
            "DAI/USD": "0x014441492f55534400000000000000000000000000",
            "BCH/USD": "0x014243482f55534400000000000000000000000000",
            "NEAR/USD": "0x014e4541522f555344000000000000000000000000",
            "LEO/USD": "0x014c454f2f55534400000000000000000000000000",
            "UNI/USD": "0x01554e492f55534400000000000000000000000000",
            "ETC/USD": "0x014554432f55534400000000000000000000000000",
            "WIF/USD": "0x015749462f55534400000000000000000000000000",
            "BONK/USD": "0x01424f4e4b2f555344000000000000000000000000",
            "JUP/USD": "0x014a55502f55534400000000000000000000000000",
            "ETHFI/USD": "0x0145544846492f5553440000000000000000000000",
            "ENA/USD": "0x01454e412f55534400000000000000000000000000",
            "PYTH/USD": "0x01505954482f555344000000000000000000000000",
            "HNT/USD": "0x01484e542f55534400000000000000000000000000",
            "SUI/USD": "0x015355492f55534400000000000000000000000000",
            "PEPE/USD": "0x01504550452f555344000000000000000000000000",
            "QNT/USD": "0x01514e542f55534400000000000000000000000000",
            "AAVE/USD": "0x01414156452f555344000000000000000000000000",
            "FTM/USD": "0x0146544d2f55534400000000000000000000000000",
            "ONDO/USD": "0x014f4e444f2f555344000000000000000000000000",
            "TAO/USD": "0x0154414f2f55534400000000000000000000000000",
            "FET/USD": "0x014645542f55534400000000000000000000000000",
            "RENDER/USD": "0x0152454e4445522f55534400000000000000000000",
            "NOT/USD": "0x014e4f542f55534400000000000000000000000000",
            "RUNE/USD": "0x0152554e452f555344000000000000000000000000",
            "TRUMP/USD": "0x015452554d502f5553440000000000000000000000",
            "USDX/USD": "0x01555344582f555344000000000000000000000000",
            "JOULE/USD": "0x014a4f554c452f5553440000000000000000000000",
            "HBAR/USD": "0x01484241522f555344000000000000000000000000",
            "PENGU/USD": "0x0150454e47552f5553440000000000000000000000",
            "HYPE/USD": "0x01485950452f555344000000000000000000000000",
            "APT/USD": "0x014150542f55534400000000000000000000000000",
            "PAXG/USD": "0x01504158472f555344000000000000000000000000",
            "BERA/USD": "0x01424552412f555344000000000000000000000000",
        }

    def _fetch_flare_feed(self, feed_name, from_ts, to_ts):
        url = f"https://flare-systems-explorer.flare.network/backend-url/api/v0/fast_updates_feed"
        params = {
            "feed_name": self.feed_list[feed_name][2:], # if feed name is not in the list we should do 2 fetches and then convert eg PENGU/USDC = PENGU/USD * USD/USDC
            "from_ts": from_ts,
            "to_ts": to_ts,
        }
        headers = {
            "Accept": "application/json, text/plain, */*",
        }
        response = requests.get(url, params=params, headers=headers)

        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Request failed with status code {response.status_code}"}

    def get_feed_analytics(self, feed_name, from_ts, to_ts):
        feed = self._fetch_flare_feed(feed_name, from_ts, to_ts)

        df = pd.DataFrame(feed)

        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')

        min_price = df['value'].min()
        max_price = df['value'].max()
        mean_price = df['value'].mean()
        median_price = df['value'].median()
        std_dev = df['value'].std()
        price_range = max_price - min_price
        price_volatility = std_dev / mean_price * 100

        df['price_change'] = df['value'].diff()
        df['percent_change'] = df['price_change'] / df['value'].shift(1) * 100

        max_positive_change = df['percent_change'].max()
        max_negative_change = df['percent_change'].min()

        time_span = (df['timestamp'].max() - df['timestamp'].min()) / 60
        avg_time_between_updates = df['timestamp'].diff().mean()

        window_size = 5
        df['sma'] = df['value'].rolling(window=window_size).mean()

        first_price = df['value'].iloc[0]
        last_price = df['value'].iloc[-1]
        overall_change = ((last_price - first_price) / first_price) * 100

        if overall_change > 0.1:
            trend = "Upward"
        elif overall_change < -0.1:
            trend = "Downward"
        else:
            trend = "Sideways"

        return {
            "asset": "FLR/USD",
            "time_period": {
                "start": df['datetime'].min().strftime('%Y-%m-%d %H:%M:%S'),
                "end": df['datetime'].max().strftime('%Y-%m-%d %H:%M:%S'),
                "duration_minutes": time_span
            },
            "price_statistics": {
                "min": min_price,
                "max": max_price,
                "mean": mean_price,
                "median": median_price,
                "std_dev": std_dev,
                "range": price_range,
                "volatility_percent": price_volatility
            },
            "price_movements": {
                "max_positive_change_percent": max_positive_change,
                "max_negative_change_percent": max_negative_change,
                "overall_trend": trend,
                "overall_change_percent": overall_change
            },
            "feed_metrics": {
                "data_points": len(df),
                "avg_seconds_between_updates": avg_time_between_updates,
                "block_range": [df['block'].min(), df['block'].max()]
            }
        }


from_ts = int(time.time()) - 3600
to_ts = int(time.time())

feed = FTSOFeed()
print(feed.get_feed_analytics("FLR/USD", from_ts, to_ts))