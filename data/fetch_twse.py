import yfinance as yf
import ta

def get_stock_features(stock_code):
    df = yf.download(f"{stock_code}.TW", period="3mo")
    df = ta.add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume")
    return df.tail(1).dropna(axis=1)

def get_historical_data(stock_code):
    df = yf.download(f"{stock_code}.TW", period="6mo")
    df = ta.add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume")
    df["1D"] = df["Close"].pct_change().shift(-1)
    df["3D"] = df["Close"].pct_change(3).shift(-3)
    df["5D"] = df["Close"].pct_change(5).shift(-5)
    df = df.dropna()
    return df
