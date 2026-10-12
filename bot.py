import json
import os
import time

import lightgbm as lgb
import pandas as pd
import requests

from features import construire_features

HOTES = ["https://data-api.binance.vision", "https://api.binance.com",
         "https://api1.binance.com", "https://api2.binance.com"]
SYMBOLE, AUTRE = "BTCUSDT", "ETHUSDT"
SIGNAL_PERIME_MIN = 30          # au-dela, on ne trade pas sur un signal trop vieux
COLONNES = ["open_time", "open", "high", "low", "close", "volume", "close_time",
            "quote_volume", "n_trades", "taker_buy_base", "taker_buy_quote", "ignore"]


def telecharger_klines(symbole, limite=1000):
    """1000 barres de 5 min (environ 3,5 jours), avec plusieurs adresses de secours."""
    erreur = None
    for hote in HOTES:
        try:
            r = requests.get(f"{hote}/api/v3/klines", timeout=20,
                             params={"symbol": symbole, "interval": "5m", "limit": limite})
            if r.status_code == 200:
                return r.json()
            erreur = f"{hote} -> HTTP {r.status_code}"
        except requests.RequestException as e:
            erreur = f"{hote} -> {e}"
    raise RuntimeError(f"Binance inaccessible pour {symbole} : {erreur}")


def vers_barres(klines):
    df = pd.DataFrame(klines, columns=COLONNES).drop(columns="ignore")
    for c in df.columns:
        df[c] = pd.to_numeric(df[c])
    df["time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df.set_index("time")


def etat_initial(prix, maintenant, frais):
    return {"debut": str(maintenant), "prix_debut": prix, "derniere_decision": None,
            "en_position": 0, "dernier_prix": prix, "equity_bh": 1.0,
            "scenarios": {str(f): {"equity": 1.0, "trades": 0} for f in frais}}


def main():
    config = json.load(open("config.json"))
    FEATURES = config["features"]
    HAUT, BAS = config["seuil_entree"], config["seuil_sortie"]
    FRAIS = config["frais_simules"]
    booster = lgb.Booster(model_file="model.txt")

    # ---- 1) donnees ----
    maintenant = pd.Timestamp.now(tz="UTC")
    now_ms = int(maintenant.timestamp() * 1000)
    kb, ke = vers_barres(telecharger_klines(SYMBOLE)), vers_barres(telecharger_klines(AUTRE))
    prix = float(kb["close"].iloc[-1])                      # dernier prix connu (barre en cours)
    btc, eth = kb[kb["close_time"] < now_ms], ke[ke["close_time"] < now_ms]   # barres terminees

    # ---- 2) features et dernier moment de decision (multiple de 15 min) ----
    feats = construire_features(btc, eth)
    candidats = feats[feats.index.minute % 15 == 0].dropna(subset=FEATURES)
    if candidats.empty:
        raise RuntimeError("Aucune ligne de features exploitable")
    t_sig = candidats.index[-1]
    age_min = (maintenant - t_sig).total_seconds() / 60

    # ---- 3) etat du portefeuille simule ----
    if os.path.exists("state.json"):
        etat = json.load(open("state.json"))
    else:
        etat = etat_initial(prix, maintenant, FRAIS)

    ratio = prix / etat["dernier_prix"]                     # evolution du prix depuis le dernier passage
    etat["equity_bh"] *= ratio
    if etat["en_position"]:
        for s in etat["scenarios"].values():
            s["equity"] *= ratio

    # ---- 4) decision (seulement si c'est un nouveau moment de decision et qu'il est frais) ----
    proba = None
    nouvelle = (etat["derniere_decision"] is None
                or t_sig > pd.Timestamp(etat["derniere_decision"]))
    if nouvelle and age_min <= SIGNAL_PERIME_MIN:
        proba = float(booster.predict(candidats.loc[[t_sig], FEATURES])[0])
        ancienne = etat["en_position"]
        if proba > HAUT:
            nouvelle_pos = 1
        elif proba < BAS:
            nouvelle_pos = 0
        else:
            nouvelle_pos = ancienne                         # entre les deux seuils : on ne bouge pas
        if nouvelle_pos != ancienne:
            for f, s in etat["scenarios"].items():
                s["equity"] *= (1 - float(f))               # frais payes a chaque entree ou sortie
                s["trades"] += 1
        etat["en_position"] = nouvelle_pos
        etat["derniere_decision"] = str(t_sig)
    elif nouvelle:
        print(f"Signal {t_sig} perime ({age_min:.0f} min) : pas de decision ce tour-ci")

    etat["dernier_prix"] = prix
    json.dump(etat, open("state.json", "w"), indent=2)

    # ---- 5) journal ----
    ligne = {"run": str(maintenant), "signal": str(t_sig), "age_min": round(age_min, 1),
             "prix": prix, "proba": proba, "en_position": etat["en_position"],
             "equity_buy_hold": etat["equity_bh"]}
    for f, s in etat["scenarios"].items():
        ligne[f"equity_frais_{f}"] = s["equity"]
    pd.DataFrame([ligne]).to_csv("journal.csv", mode="a", index=False,
                                 header=not os.path.exists("journal.csv"))
    print(ligne)


if __name__ == "__main__":
    main()
