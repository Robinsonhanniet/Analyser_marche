"""
Analyseur de marchés — actions, indices, crypto + actualité
-----------------------------------------------------------
Lance :  pip install streamlit yfinance pandas numpy vaderSentiment
         streamlit run app.py

Ce que fait l'app :
  1. Télécharge 2 ans de cours pour chaque titre des marchés choisis
  2. Calcule des indicateurs techniques (tendance, RSI, MACD, momentum, volatilité)
  3. Lit les titres d'actualité des 14 derniers jours, mesure leur ton et repère les gros événements
     (patron viré, scandale, enquête, faillite, résultats, rachat, sanctions…) avec une alerte 🚨
  4. Regarde si le marché global est en tendance haussière ou baissière
  5. Combine tout en une note de -100 à +100 et un signal ACHAT / ATTENDRE / VENTE
  6. Propose un stop-loss, un objectif et une taille de position selon ton capital
  7. Teste la stratégie sur le passé (backtest) pour voir si elle aurait vraiment marché

ATTENTION : aucun outil ne prédit la bourse avec précision. Ces signaux sont une aide
à la décision, pas une garantie. Teste d'abord sans argent réel.
"""
import re
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ------------------------------------------------------------------ marchés
MARCHES = {
    "France (CAC 40)": {"indice": "^FCHI", "titres": {
        "LVMH": "MC.PA", "TotalEnergies": "TTE.PA", "Airbus": "AIR.PA", "L'Oréal": "OR.PA",
        "Schneider": "SU.PA", "BNP Paribas": "BNP.PA", "Hermès": "RMS.PA", "Sanofi": "SAN.PA",
        "Safran": "SAF.PA", "Stellantis": "STLAP.PA"}},
    "USA (Nasdaq / S&P 500)": {"indice": "^GSPC", "titres": {
        "Apple": "AAPL", "Microsoft": "MSFT", "Nvidia": "NVDA", "Amazon": "AMZN",
        "Alphabet": "GOOGL", "Meta": "META", "Tesla": "TSLA", "JPMorgan": "JPM",
        "Netflix": "NFLX", "AMD": "AMD"}},
    "Allemagne (DAX)": {"indice": "^GDAXI", "titres": {
        "SAP": "SAP.DE", "Siemens": "SIE.DE", "Allianz": "ALV.DE", "Mercedes": "MBG.DE",
        "Deutsche Telekom": "DTE.DE", "Adidas": "ADS.DE"}},
    "Crypto": {"indice": "BTC-USD", "titres": {
        "Bitcoin": "BTC-USD", "Ethereum": "ETH-USD", "Solana": "SOL-USD"}},
}

POIDS = {"tech": 0.60, "news": 0.25, "marche": 0.15}
SEUIL_ACHAT, SEUIL_VENTE = 40, -40


# ------------------------------------------------------------------ données
@st.cache_data(ttl=900, show_spinner=False)
def cours(tickers: tuple) -> dict:
    data = yf.download(list(tickers), period="2y", interval="1d",
                       group_by="ticker", auto_adjust=True, progress=False, threads=True)
    out = {}
    for t in tickers:
        try:
            df = data[t] if len(tickers) > 1 else data
            df = df.dropna(subset=["Close"])
            if len(df) > 220:
                out[t] = df
        except KeyError:
            pass
    return out


@st.cache_data(ttl=900, show_spinner=False)
def actus(ticker: str, jours: int = 14) -> list:
    """Titres d'actualité récents (format yfinance ancien et nouveau)."""
    try:
        brut = yf.Ticker(ticker).news or []
    except Exception:
        return []
    limite = datetime.now(timezone.utc) - timedelta(days=jours)
    res = []
    for n in brut:
        c = n.get("content", n)
        titre = c.get("title") or ""
        date = c.get("pubDate") or c.get("providerPublishTime")
        try:
            d = (datetime.fromtimestamp(date, timezone.utc) if isinstance(date, (int, float))
                 else datetime.fromisoformat(str(date).replace("Z", "+00:00")))
        except Exception:
            d = None
        lien = (c.get("canonicalUrl") or {}).get("url") if isinstance(c.get("canonicalUrl"), dict) else c.get("link")
        if titre and (d is None or d >= limite):
            res.append({"titre": titre, "date": d, "lien": lien})
    return res


# ------------------------------------------------------------------ indicateurs
def indicateurs(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    c = d["Close"]
    d["SMA20"], d["SMA50"], d["SMA200"] = c.rolling(20).mean(), c.rolling(50).mean(), c.rolling(200).mean()
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    perte = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    d["RSI"] = 100 - 100 / (1 + gain / perte.replace(0, np.nan))
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    d["MACD"] = ema12 - ema26
    d["MACDsig"] = d["MACD"].ewm(span=9, adjust=False).mean()
    tr = pd.concat([d["High"] - d["Low"], (d["High"] - c.shift()).abs(), (d["Low"] - c.shift()).abs()], axis=1).max(axis=1)
    d["ATR"] = tr.rolling(14).mean()
    d["Mom3M"] = c.pct_change(63)
    d["Vol"] = c.pct_change().rolling(20).std() * np.sqrt(252)
    d["VolumeRel"] = d["Volume"] / d["Volume"].rolling(20).mean()
    return d


def note_technique(d: pd.DataFrame) -> pd.Series:
    """Note de -100 à +100 calculée pour chaque jour (sert aussi au backtest)."""
    s = pd.Series(0.0, index=d.index)
    s += np.where(d["Close"] > d["SMA200"], 25, -25)            # tendance longue
    s += np.where(d["SMA50"] > d["SMA200"], 15, -15)            # croisement doré / de la mort
    s += np.where(d["Close"] > d["SMA20"], 10, -10)             # tendance courte
    s += np.where(d["MACD"] > d["MACDsig"], 15, -15)            # élan
    s += np.clip(d["Mom3M"] * 100, -20, 20)                     # performance 3 mois
    rsi = d["RSI"]
    s += np.select([rsi < 30, rsi > 75, rsi.between(50, 70)], [10, -15, 5], 0)  # survente / surachat
    s += np.where((d["VolumeRel"] > 1.5) & (d["Close"] > d["Close"].shift()), 5, 0)  # achat avec volume
    return s.clip(-100, 100)


_vader = SentimentIntensityAnalyzer()

# --------------------------------------------- événements majeurs (anglais + français)
# (nom, impact de -100 à +100, motifs recherchés dans le titre)
EVENEMENTS = [
    ("Départ / renvoi du patron", -70, r"^(?=.*\b(ceo|chief executive|founder|chairman|patron|pdg|directeur général|fondateur)\b).*\b(fire[sd]?|ousted?|resign\w*|steps? down|stepping down|exit\w*|departure|leav(es|ing)|quits?|renvoy\w*|limog\w*|démission\w*|quitte|départ|évincé)\b"),
    ("Scandale / enquête", -60, r"\b(scandal\w*|probe|investigation|subpoena|espionage|spy|fraud|indicted|charged|sec (probe|investigation)|enquête|scandale|espionnage|fraude|mis en examen)\b"),
    ("Procès / amende", -40, r"\b(lawsuit|sued|class action|fine[ds]?\b|penalt\w+|antitrust|procès|amende|poursuivi\w*)\b"),
    ("Faillite / dette", -90, r"\b(bankrupt\w*|chapter 11|insolven\w*|default\w*|faillite|redressement judiciaire|cessation de paiement)\b"),
    ("Résultats décevants", -50, r"\b(miss(es|ed)? (estimates|expectations)|cuts? (guidance|forecast|outlook)|profit warning|lowers? (guidance|forecast)|résultats décevants|abaisse ses (prévisions|objectifs)|avertissement sur résultats)\b"),
    ("Licenciements", -25, r"\b(layoffs?|job cuts|cuts? \d[\d,]* jobs|licenciement\w*|plan social|suppression\w* de postes)\b"),
    ("Rappel produit / panne", -35, r"\b(recall\w*|outage|data breach|hack(ed|ers)?|cyberattack|rappel de produits?|cyberattaque|fuite de données)\b"),
    ("Sanctions / interdiction export", -45, r"\b(export (ban|controls?|restrictions?)|sanction\w*|tariff\w*|ban(s|ned)? (on|from)|droits de douane|embargo)\b"),
    ("Note abaissée", -30, r"\b(downgrade[sd]?|cut to (sell|underperform)|dégrad\w*|abaiss\w* (sa )?recommandation)\b"),
    ("Résultats au-dessus des attentes", 50, r"\b(beats? (estimates|expectations)|tops? (estimates|expectations)|record (revenue|profit|sales|quarter)|raises? (guidance|forecast|outlook)|dépasse les attentes|relève ses (prévisions|objectifs)|résultats record)\b"),
    ("Rachat / fusion", 35, r"\b(acquir\w+|acquisition|takeover|merger|to buy .{0,30} for \$|rachat de|rachète|fusion|offre publique)\b"),
    ("Rachat d'actions / dividende", 25, r"\b(buyback|share repurchase|raises? dividend|special dividend|rachat d'actions|hausse du dividende)\b"),
    ("Note relevée", 30, r"\b(upgrade[sd]?|raised to (buy|outperform)|relève (sa )?recommandation|passe à l'achat)\b"),
    ("Gros contrat / partenariat", 30, r"\b(wins? .{0,20}contract|partnership|deal with|strikes? deal|approval|approved by|fda approv\w*|contrat|partenariat|feu vert|autorisation)\b"),
    ("Nouveau patron", -15, r"\b(names? new ceo|appoints? .{0,30}ceo|interim ceo|nouveau (pdg|directeur général)|pdg par intérim)\b"),
]
_EV = [(n, imp, re.compile(p, re.I)) for n, imp, p in EVENEMENTS]
SEUIL_MAJEUR = 60


def detecter_evenements(titre: str) -> list:
    return [(n, imp) for n, imp, rx in _EV if rx.search(titre)]


def note_news(items: list) -> tuple:
    """Renvoie (note actu, items annotés, événements détectés).
    La note mélange le ton des titres (VADER) et l'impact des événements repérés,
    les plus récents comptant davantage."""
    if not items:
        return 0.0, [], []
    maintenant = datetime.now(timezone.utc)
    tons, impacts, evs = [], [], []
    for it in items:
        it["sentiment"] = _vader.polarity_scores(it["titre"])["compound"]
        it["evenements"] = detecter_evenements(it["titre"])
        age = (maintenant - it["date"]).days if it["date"] else 3
        poids = 0.5 ** (age / 4)  # une news perd la moitié de son poids tous les 4 jours
        tons.append(it["sentiment"] * poids)
        for n, imp in it["evenements"]:
            impacts.append(imp * poids)
            evs.append({"evenement": n, "impact": imp, "titre": it["titre"], "date": it["date"], "lien": it["lien"]})
    ton = np.clip(np.mean(tons) * 150, -100, 100)
    choc = np.clip(sum(impacts), -100, 100) if impacts else 0
    note = 0.4 * ton + 0.6 * choc if impacts else ton
    return float(np.clip(note, -100, 100)), items, evs


def note_marche(df_indice: pd.DataFrame) -> float:
    d = indicateurs(df_indice)
    last = d.iloc[-1]
    s = (40 if last["Close"] > last["SMA200"] else -40) + (30 if last["SMA50"] > last["SMA200"] else -30) \
        + (30 if last["MACD"] > last["MACDsig"] else -30)
    return float(s)


# ------------------------------------------------------------------ backtest
def backtest(d: pd.DataFrame, score: pd.Series, frais=0.001) -> dict:
    """Entre si note >= seuil, sort si note <= 0 ou stop à 2 ATR. Uniquement la partie technique
    (l'historique des actus n'est pas disponible)."""
    en_pos, entree, stop, trades = False, 0.0, 0.0, []
    equity = [1.0]
    c, atr = d["Close"].values, d["ATR"].values
    for i in range(200, len(d) - 1):
        if not en_pos and score.iloc[i] >= SEUIL_ACHAT:
            en_pos, entree, stop = True, c[i + 1], c[i + 1] - 2 * atr[i]
        elif en_pos and (score.iloc[i] <= 0 or c[i] < stop):
            sortie = c[i + 1]
            trades.append(sortie / entree - 1 - 2 * frais)
            en_pos = False
    if en_pos:
        trades.append(c[-1] / entree - 1 - 2 * frais)
    for r in trades:
        equity.append(equity[-1] * (1 + r))
    bh = c[-1] / c[200] - 1
    return {"trades": len(trades), "gagnants": float(np.mean([r > 0 for r in trades])) if trades else np.nan,
            "strategie": equity[-1] - 1, "achat_conserve": bh}


# ------------------------------------------------------------------ interface
st.set_page_config(page_title="Analyseur de marchés", page_icon="📈", layout="wide")
st.title("📈 Analyseur de marchés")
st.caption("Aide à la décision seulement — aucun signal n'est une garantie. Teste toujours sans argent réel d'abord.")

with st.sidebar:
    choix = st.multiselect("Marchés", list(MARCHES), default=["France (CAC 40)", "USA (Nasdaq / S&P 500)"])
    perso = st.text_input("Ajouter des tickers (séparés par des virgules)", placeholder="ex : ORA.PA, KO")
    capital = st.number_input("Capital (€)", 100, 10_000_000, 1000, step=100)
    risque = st.slider("Risque max par position (% du capital)", 0.5, 5.0, 1.0, 0.5)
    avec_news = st.toggle("Analyser l'actualité", True)
    st.button("🔄 Actualiser", on_click=st.cache_data.clear)

if not choix and not perso:
    st.info("Choisis au moins un marché.")
    st.stop()

noms, indices = {}, {}
for m in choix:
    for n, t in MARCHES[m]["titres"].items():
        noms[t] = n
        indices[t] = MARCHES[m]["indice"]
for t in [x.strip().upper() for x in perso.split(",") if x.strip()]:
    noms[t], indices[t] = t, "^GSPC"

with st.spinner("Téléchargement des cours…"):
    data = cours(tuple(sorted(set(noms) | set(indices.values()))))

regime = {i: note_marche(data[i]) for i in set(indices.values()) if i in data}

lignes, details, alertes = [], {}, []
prog = st.progress(0.0, "Analyse…")
for k, (t, nom) in enumerate(noms.items()):
    prog.progress((k + 1) / len(noms), f"Analyse de {nom}…")
    if t not in data:
        continue
    d = indicateurs(data[t])
    sc = note_technique(d)
    tech = float(sc.iloc[-1])
    news_score, news, evs = note_news(actus(t)) if avec_news else (0.0, [], [])
    mk = regime.get(indices[t], 0.0)
    poids_news = POIDS["news"] if news else 0
    total_poids = POIDS["tech"] + poids_news + POIDS["marche"]
    note = (tech * POIDS["tech"] + news_score * poids_news + mk * POIDS["marche"]) / total_poids
    signal = "🟢 ACHAT" if note >= SEUIL_ACHAT else "🔴 VENTE" if note <= SEUIL_VENTE else "⚪ ATTENDRE"
    majeur_neg = [e for e in evs if e["impact"] <= -SEUIL_MAJEUR]
    if majeur_neg and signal == "🟢 ACHAT":
        signal = "⚠️ PRUDENCE"  # un gros événement négatif bloque l'achat, même si le graphique est bon
    for e in evs:
        alertes.append({**e, "titre_action": nom, "ticker": t})
    last = d.iloc[-1]
    stop, objectif = last["Close"] - 2 * last["ATR"], last["Close"] + 3 * last["ATR"]
    nb = int((capital * risque / 100) // (2 * last["ATR"])) if last["ATR"] > 0 else 0
    nb = min(nb, int(capital // last["Close"]))
    bt = backtest(d, sc)
    lignes.append({"Titre": nom, "Ticker": t, "Signal": signal, "Note": round(note),
                   "Technique": round(tech), "Actu": round(news_score) if news else None,
                   "Marché": round(mk), "Événements": ", ".join(sorted({e["evenement"] for e in evs})) or "—", "Cours": round(last["Close"], 2), "RSI": round(last["RSI"], 1),
                   "Perf 3 mois %": round(last["Mom3M"] * 100, 1), "Stop": round(stop, 2),
                   "Objectif": round(objectif, 2), "Quantité": nb,
                   "Backtest strat. %": round(bt["strategie"] * 100, 1),
                   "Achat-conservé %": round(bt["achat_conserve"] * 100, 1),
                   "Trades gagnants %": round(bt["gagnants"] * 100) if bt["trades"] else None})
    details[t] = (d, news, bt)
prog.empty()

if not lignes:
    st.error("Aucune donnée reçue. Vérifie ta connexion internet ou les tickers.")
    st.stop()

tab = pd.DataFrame(lignes).sort_values("Note", ascending=False)
c1, c2, c3 = st.columns(3)
c1.metric("Signaux ACHAT", int((tab["Note"] >= SEUIL_ACHAT).sum()))
c2.metric("Signaux VENTE", int((tab["Note"] <= SEUIL_VENTE).sum()))
c3.metric("Titres analysés", len(tab))

if alertes:
    st.subheader("🚨 Événements détectés (14 derniers jours)")
    vus = set()
    for a in sorted(alertes, key=lambda x: (abs(x["impact"]) < SEUIL_MAJEUR, x["date"] is None, -(x["date"].timestamp() if x["date"] else 0))):
        cle = (a["ticker"], a["evenement"], a["titre"])
        if cle in vus:
            continue
        vus.add(cle)
        icone = "🚨" if abs(a["impact"]) >= SEUIL_MAJEUR else ("🔺" if a["impact"] > 0 else "🔻")
        quand = a["date"].strftime("%d/%m") if a["date"] else ""
        texte = f"{icone} **{a['titre_action']}** — {a['evenement']} ({a['impact']:+d}) · {quand} · "
        texte += f"[{a['titre']}]({a['lien']})" if a["lien"] else a["titre"]
        (st.error if a["impact"] <= -SEUIL_MAJEUR else st.success if a["impact"] >= SEUIL_MAJEUR else st.info)(texte)

st.subheader("Classement")
st.dataframe(tab, hide_index=True, use_container_width=True,
             column_config={"Note": st.column_config.ProgressColumn("Note", min_value=-100, max_value=100, format="%d")})
st.caption("Stop = cours − 2 ATR, Objectif = cours + 3 ATR. Quantité = nombre d'actions pour ne pas perdre plus que "
           f"{risque} % de ton capital si le stop est touché. Backtest sur ~1 an et demi, sans l'actualité, frais 0,1 % inclus.")

st.subheader("Détail d'un titre")
sel = st.selectbox("Titre", tab["Ticker"], format_func=lambda t: f"{noms[t]} ({t})")
d, news, bt = details[sel]
st.line_chart(d[["Close", "SMA50", "SMA200"]].tail(250))
a, b, c = st.columns(3)
a.metric("RSI", f"{d['RSI'].iloc[-1]:.0f}", help="< 30 survendu, > 70 suracheté")
b.metric("Volatilité annuelle", f"{d['Vol'].iloc[-1] * 100:.0f} %")
c.metric("Stratégie vs achat-conservé", f"{bt['strategie'] * 100:.1f} %", f"{(bt['strategie'] - bt['achat_conserve']) * 100:+.1f} pts")
if news:
    st.markdown("**Actualité des 14 derniers jours**")
    for n in sorted(news, key=lambda x: x.get("sentiment", 0)):
        pastille = "🟢" if n["sentiment"] > 0.2 else "🔴" if n["sentiment"] < -0.2 else "⚪"
        quand = n["date"].strftime("%d/%m") if n["date"] else ""
        tag = " · ".join(f"**{e[0]}**" for e in n.get("evenements", []))
        tag = f" — {tag}" if tag else ""
        st.markdown((f"{pastille} {quand} — [{n['titre']}]({n['lien']})" if n["lien"] else f"{pastille} {quand} — {n['titre']}") + tag)
else:
    st.caption("Pas d'actualité récente trouvée pour ce titre.")
