"""RibbandsRapidplay functions.
These are the core functions that allow computation of rating gains, using fixed assigned
Tournament Ratings (TournamentRating) and for a tournament cross-table to be generated that shows
the rating gains (instead of just +1, +0.5 or 0) as the score for each match.
"""


import pandas as pd
import numpy as np
import requests
import json
import math as m
import copy
import string

from io import StringIO
from functools import reduce


def getGameData(code, event=None, flds=None, timeCtrl = 'Rapid', limit=50):
    qq = "http://www.ecfrating.org.uk/v2/new/api.php?v2/games/"+timeCtrl+"/player/"+code+"/limit/"+str(limit)
    response = requests.get(qq)
    games = response.json()['games']
        
    if event is None:
        if flds is None:
            data = [i for i in games]
        else:
            data = [[i.get(k) for k in flds] for i in games] 
    else:
        if flds is None:
            data = [i for i in games if i['event_code'] == event]
        else:
            data = [[i.get(k) for k in flds] for i in games if i['event_code'] == event]

    return data

def getTPR(code, event, TPRdata):
    opponents = getGameData(code, event, ['opponent_no','score'])
    
    wins = len([i[1] for i in opponents if i[1]==1])
    losses = len([i[1] for i in opponents if i[1]==0])
    nGames = len(opponents)

    opponents = pd.DataFrame(opponents, columns=['BCFCode','score'])
    opponents.index = opponents['BCFCode']

    opponents.loc[opponents['score']==5,'score'] = 0.5

    avgOppRating = np.mean(TPRdata.loc[opponents['BCFCode'],'TournamentRating'])
    TPR = avgOppRating + 400*(wins-losses)/nGames

    player = TPRdata.set_index('BCFCode').loc[code,'TournamentRating']

    opponents['TournamentRating']=TPRdata.loc[opponents.index,'TournamentRating']
    
    gains = opponents.apply(lambda x: calcECFgain(player,x['TournamentRating'],x['score']),axis=1)

    avgGain = round(sum(gains)/nGames,1)

    return (TPR, avgGain, avgOppRating, wins, losses, nGames)


def calcECFgain(player,opponent,score=1.0,K=20):

    tbl = pd.DataFrame([[0,3,0],
                        [4,10,0.2],
                        [11,17,0.4],
                        [18,25,0.6],
                        [26,32,0.8],
                        [33,39,1.0],
                        [40,46,1.2],
                        [47,53,1.4],
                        [54,61,1.6],
                        [62,68,1.8],
                        [69,76,2.0],
                        [77,83,2.2],
                        [84,91,2.4],
                        [92,98,2.6],
                        [99,106,2.8],
                        [107,113,3.0],
                        [114,121,3.2],
                        [122,129,3.4],
                        [130,137,3.6],
                        [138,145,3.8],
                        [146,153,4.0],
                        [154,162,4.2],
                        [163,170,4.4],
                        [171,179,4.6],
                        [180,188,4.8],
                        [189,197,5.0],
                        [198,206,5.2],
                        [207,215,5.4],
                        [216,225,5.6],
                        [226,235,5.8],
                        [236,245,6.0],
                        [246,256,6.2],
                        [257,267,6.4],
                        [268,278,6.6],
                        [279,289,6.8],
                        [290,302,7.0],
                        [303,315,7.2],
                        [316,328,7.4],
                        [329,344,7.6],
                        [345,357,7.8],
                        [358,374,8.0],
                        [375,391,8.2],
                        [392,4000,8.4]], columns=['DMin','DMax','DOff'])
    
    D = opponent-player
    DOff = tbl.loc[(tbl.DMin<=abs(D)) & (tbl.DMax>=abs(D)),'DOff']
    DOff = DOff.to_numpy()[0]
    ELOgain = (20*score-10+m.copysign(1,D)*DOff)*K/20.0

    return ELOgain

def scorePlayer(i, results):

    whites = results[results['PIN1']==i]
    blacks = results[results['PIN2']==i]

    if len(whites.index)==0:
        wGains=[]
        wOppRatings=[]
    else:
        wGains = whites.apply(lambda x: calcECFgain(x['RATING1'],x['RATING2'],x['SCORE1']), axis=1)
        wOppRatings=whites['RATING2']

    if len(blacks.index)==0:
        bGains=[]
        bOppRatings=[]
    else:
        bGains = blacks.apply(lambda x: calcECFgain(x['RATING2'],x['RATING1'],x['SCORE2']), axis=1)
        bOppRatings=blacks['RATING1']

    gameScore = sum(whites['SCORE1'])+sum(blacks['SCORE2'])

    out=list(wGains)+list(bGains)
    oppRatings = list(wOppRatings)+list(bOppRatings)
    
    avgOppRating=(round(np.mean(oppRatings),0)).astype(int)

    nGames = len(out)
    avgGain = round(np.mean(out),1)

    return gameScore, nGames, avgOppRating, avgGain

def compileStandings(f, event='LN00003877'):
    """
    Compiles standings for default event LN00003877 (Ribbands Rapidplay 2023).
    Pulls results direct from ECF website. Needs an input file that has a sheet called Player_List
    that holds two columns. BCFCode, and TournamentRating.
    The only parameters required here are f - the filename, and optionally event if to be used
    to compute rating gains for results from a different event.
    """

    TPRdata = pd.read_excel(f, sheet_name='Player_List')
    TPRdata.index = [i[:6] for i in TPRdata['BCFCode']]
    TPRdata.index = TPRdata.index.astype(int)

    TPR = pd.DataFrame([getTPR(i,event,TPRdata) for i in TPRdata['BCFCode']], columns=['RibbandsTPR','avgGain',
        'avgOpp','wins','losses','nGames'])
    TPR.index = TPRdata.index
    results = pd.concat([TPRdata, TPR], axis=1)
    
    results['Ribbands'] = results['RibbandsTPR']/results['TournamentRating']
    results['Ribbands'] = results.apply(lambda x: -1000+x['avgGain'] if x['nGames']<8 else x['avgGain'], axis=1)

    results.sort_values('Ribbands', ascending=False, inplace=True)
    results['Score'] = results['wins']+(results['nGames']-results['wins']-results['losses'])*0.5

    displayOrder = ['NAME', 'Score','nGames','avgGain', 'TournamentRating', 'avgOpp']
    
    results = results[displayOrder]

    return results

def compileStandingsFromGradingData(f):
    """
    Alternative function for compiling standings, based on data from the grading data
    generated from the LMS website. Needs an additional column added to the Player_List
    sheet, headed TournamentRating. This should hold the fixed assigned tournament ratings
    to be used for the rating gain computations.
    This function allows standings to be compiled even if the grading data hasn't yet been
    submitted to the Ratings website.
    """
    
    players = pd.read_excel(f, sheet_name='Player_List')
    results = pd.read_excel(f, sheet_name='Results_List')

    players['PIN'] = players['PIN'].astype(int)
    results['PIN1'] = results['PIN1'].astype(int)
    results['PIN2'] = results['PIN2'].astype(int)

    players.set_index('PIN', inplace=True)

    results['RATING1'] = list(players.loc[results['PIN1'],'TournamentRating'])
    results['RATING2'] = list(players.loc[results['PIN2'],'TournamentRating'])
    results['SCORE1'] = [1 if i==10.0 else 0.5 if i==55.0 else 0 for i in results['Result']]
    results['SCORE2'] = [0 if i==10.0 else 0.5 if i==55.0 else 1 for i in results['Result']]

    df = pd.DataFrame([scorePlayer(i, results) for i in players.index],
                        columns=['Score','Played','avgOppRating', 'Gain'])
    df.index=players.index

    df.insert(2,'Rating',players.apply(lambda x: x['TournamentRating'], axis=1))

    out=players[['BCFCode','NAME']].merge(df, left_index=True, right_index=True)
    out['Ribbands']=out.apply(lambda x: x['Gain'] if x['Played']>=8 else -1000+x['Gain'], axis=1)
    out.sort_values('Ribbands',ascending=False, inplace=True)

    out.drop('Ribbands', axis=1, inplace=True)

    out = out[['NAME', 'Score', 'Played', 'Gain', 'Rating', 'avgOppRating']]

    return out

def xtableResults(f):
    """
    Compiles the cross-table based on a file in the grading data format
    generated from the LMS website. Additional column headed TournamentRating
    needs to be added to the Player_List sheet.
    """

    players = pd.read_excel(f, sheet_name='Player_List')
    results = pd.read_excel(f, sheet_name='Results_List')

    players['PIN'] = players['PIN'].astype(int)
    results['PIN1'] = results['PIN1'].astype(int)
    results['PIN2'] = results['PIN2'].astype(int)

    players.set_index('PIN', inplace=True)

    results['RATING1'] = list(players.loc[results['PIN1'],'TournamentRating'])
    results['RATING2'] = list(players.loc[results['PIN2'],'TournamentRating'])
    results['SCORE1'] = [1 if i==10.0 else 0.5 if i==55.0 else 0 for i in results['Result']]
    results['SCORE2'] = [0 if i==10.0 else 0.5 if i==55.0 else 1 for i in results['Result']]

    results['Round']=results['Round'].astype(int)
    results.set_index(['Round','PIN1'], inplace=True)

    nRounds = max(results.index.get_level_values(0))
    out = pd.DataFrame(index=players.index)

    # first process the Whites
    for j in range(nRounds):
        R = results.loc[j+1,:]

        v = ['('+str(1+players.index.get_loc(R.loc[i,'PIN2']))+'W'
                +('+,' if R.loc[i,'SCORE1']==1.0 else '-,' if R.loc[i,'SCORE1']==0.0 else '=,')
                +str(round(calcECFgain(R.loc[i,'RATING1'], R.loc[i,'RATING2'], R.loc[i,'SCORE1']),2))
                +')'
                if i in R.index else np.nan for i in players.index]
       
        out[j] = v
    
    # now the Blacks
    for j in range(nRounds):
        R = results.reset_index().set_index(['Round','PIN2']).loc[j+1,:]
        for i in R.index:
            game = R.loc[i,:]
            
            G = '(' + str(1+players.index.get_loc(game['PIN1']))+'B'+('+,' if game['SCORE2']==1.0 else '-,' if game['SCORE2']==0.0 else '=,') + str(round(calcECFgain(game['RATING2'],game['RATING1'],game['SCORE2']),2))+')'

            out.loc[i,j]=G

    out.columns=['R'+str(1+i) for i in out.columns]
    out.insert(0,'IDX',[1+players.index.get_loc(i) for i in players.index])
    out.insert(0,'NAME',players.loc[out.index,'NAME'])

    return out

