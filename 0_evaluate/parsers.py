import pandas as pd
import numpy as np

def sampleBounds(bounds, goLow):
    if bounds[0] != bounds[1] and bounds[0] > 0:
        l, b = bounds
        s = np.random.uniform(l, b)
    elif bounds[0] == 0 and bounds[0] > 0:
        l, b = np.log(bounds)
        s = np.random.uniform(l, b)
    elif bounds[0]==bounds[1] and bounds[0]==0: 
        s = 1e-14
    else:
        l, b = np.log(bounds)
        s = np.random.uniform(l, b)
    return s


def getSpecies(path):
    df_ = pd.read_csv(path)
    df_ = df_[~df_['reactions'].str.contains('REV')]
    ss = df_['reactions'].str.split(r"[^a-zA-Z0-9-_.\s]").explode().unique()
    species = np.array(ss[ss != ''], dtype=str)
    return species


def generateBounds(df):
    bounds = dict()
    for index, row in df.iterrows():
        lb = row.minconc
        ub = row.maxconc2
        if type(val) is str:
            aux = val.split("-")
            lb = float(aux[0])*10**(-12)
            ub = float(aux[1])*10**(-12)
        else:
            lb = float(val)*10**(-12)
            ub = float(val)*10**(-12)
        bounds[row[0]] = [lb, ub]
    return bounds


def generateICs(species, bounds, goLow=False):
    to_file = dict()
    for s in species:
        if s not in bounds.keys():
            to_file[s] = 0
            continue
    # listából kinézi, hogy van e IC-je
    for s in species:
        if s in bounds.keys():
            ic = sampleBounds(bounds[s], goLow)
            to_file[s] = ic
        # ezeknek a kezdeti értéke nem változik
        # itt kérdés, hogy legyen e continuity a caspase és caspasea között
    return to_file


def compileDataRow(variables, dataPoints):
    meas = ""
    for v in variables:
        meas = meas+"<%s>" % v + "{:.2e}" + "</%s>" % v
    start = "<dataPoint>"
    close = "</dataPoint>"
    row = start+meas.format(*dataPoints)+close
    return row
