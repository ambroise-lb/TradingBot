import numpy as np
import pandas as pd


def features_actif(d):
    """Uniquement du passe : chaque ligne n'utilise que des barres jusqu'a elle-meme."""
    f = pd.DataFrame(index=d.index)
    c, v = d["close"], d["volume"]
    r = c.pct_change(fill_method=None)

    delta = 2 * d["taker_buy_base"] - v
    for n in [1, 6, 12, 48]:
        f[f"imb{n}"] = delta.rolling(n).sum() / v.rolling(n).sum()

    for n in [6, 12, 48, 288]:
        f[f"mom{n}"] = c.pct_change(n, fill_method=None)

    f["vol12"] = r.rolling(12).std()
    f["vol288"] = r.rolling(288).std()
    f["ratio_vol"] = f["vol12"] / f["vol288"]
    f["vol_rel"] = v.rolling(12).mean() / v.rolling(288).mean()
    f["trades_rel"] = d["n_trades"].rolling(12).mean() / d["n_trades"].rolling(288).mean()

    typ = (d["high"] + d["low"] + d["close"]) / 3
    jour = d.index.date
    vwap = (typ * v).groupby(jour).cumsum() / v.groupby(jour).cumsum()
    ecart = c / vwap - 1
    f["z_vwap"] = ecart / ecart.rolling(288).std()

    f["dist_haut48"] = c / d["high"].rolling(48).max() - 1
    f["dist_bas48"] = c / d["low"].rolling(48).min() - 1
    return f


def construire_features(btc, eth):
    """Features du BTC, avec les variables croisees ETH. Index = heure de CLOTURE de la barre."""
    idx = btc.index.intersection(eth.index)
    b, e = btc.loc[idx], eth.loc[idx]

    f = features_actif(b)
    fo = features_actif(e)
    for col in ["mom12", "mom48", "imb12", "imb48", "vol12"]:
        f[f"x_{col}"] = fo[col]
    f["ecart_mom12"] = f["mom12"] - fo["mom12"]
    f["ecart_mom48"] = f["mom48"] - fo["mom48"]

    f.index = f.index + pd.Timedelta("5min")          # meme convention que l'entrainement
    heure = f.index.hour + f.index.minute / 60
    f["h_sin"] = np.sin(2 * np.pi * heure / 24)
    f["h_cos"] = np.cos(2 * np.pi * heure / 24)
    f["jour_semaine"] = f.index.dayofweek
    return f
