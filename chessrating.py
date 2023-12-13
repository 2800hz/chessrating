import pandas as pd
import numpy as np
import requests
import json
import math as m
import copy
import string

from io import StringIO
from functools import reduce

def getRatingGains(code, event):
    qq = "http://www.ecfrating.org.uk/v2/new/api.php?v2/games/Rapid/player/"+code+"/limit/100"
    response = requests.get(qq)
    games = response.json()['games']
    data = [i['increment'] for i in games if i['event_code'] == event]
    return data

def getRatingList():
    qq = "https://www.ecfrating.org.uk/v2/new/api.php?v2/rating_list_csv"
    response = requests.get(qq)
    players = pd.read_csv(StringIO(response.text))
    
    return players

def getRatingDataForClubs(q):
    """Return active players.
    q: list of text to search for
    """
    players = getRatingList()
    out = pd.concat([players[players['club_name'].str.contains(club)] for club in q])

    return out

def getRatingDataBACLPlayers(cols=['full_name','member_no','FIDE_no','gender','club_name','club_code','revised_standard','revised_rapid']):
    players = getRatingDataForClubs(['Linton','Bury','Cambridge','Ely','Stowmarket','Newmarket','Sudbury'])
    
    if cols is not None:
        players = players[cols]

    return players

def regionalClubs():
    clubs1 = ['Linton','Cambridge','Bury','Ely','Stowmarket','Newmarket','Sudbury']
    clubs2 = ['Broadland','Norwich','Norfolk','Essex']
    clubs = clubs1+clubs2

    return clubs

def getRatingDataForLocalClubs(cols=['full_name','member_no','FIDE_no','gender','club_name','club_code','revised_standard','revised_rapid']):
    
    clubs = regionalClubs()
    players = getRatingDataForClubs(clubs)

    if cols is not None:
        players = players[cols]

    players['member_no'] = players['member_no'].astype(str)
    return players


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

def LintonPlayers():
    df = pd.read_csv('RibbandsRapidplay.csv')
    members = df['Membership'].dropna()
    res = [requests.get('http://www.ecfrating.org.uk/v2/new/api.php?v2/players/mid/'+i) for i in members]
    res = pd.DataFrame([i.json() for i in res])
    res.set_index('ECF_code', inplace=True)
    
    return res

def getMemberNumbersForClubs(clubs):
    players = getClubPlayers(clubs)
    active = players[players['member_no'].notna()]

    membernos = [('ME'+('%06d'%i)) for i in active['member_no']]

    return membernos

def getMemberInfo(members):
    res = [requests.get('http://www.ecfrating.org.uk/v2/new/api.php?v2/players/mid/'+i) for i in members]
    res = pd.DataFrame([i.json() for i in res])
#    res.set_index('ECF_code', inplace=True)
    
    return res

def getMemberInfoForClubs(clubs):
    membernos = getMemberNumbersForClubs(clubs)
    info = getMemberInfo(membernos)

    return info.iloc[:,:-5]

def compileOfficialGain():
    players = pd.read_csv('RibbandsRapidplay.csv')
    D = pd.DataFrame(getRatingGains(i, 'LN00003877') for i in players['Grade Code'])
    R = pd.concat([players, D], axis=1)
    R.to_csv('results.csv')
    print(R)

def getRating(code, dt, type='revised'):
    API = "http://www.ecfrating.org.uk/v2/new/api.php?v2/ratings/"
    qq = API+"R/"+code+"/"+dt  # date in yyyy-mm-dd format
    response = requests.get(qq)  
    data = response.json()
    rapidRating = str(data[type+"_rating"])+data[type+"_category"] if data['success'] else None

    qq = API+"S/"+code+"/"+dt  # date in yyyy-mm-dd format
    response = requests.get(qq)
    data = response.json()
    stdRating = str(data[type+"_rating"])+data[type+"_category"] if data['success'] else None

    return (rapidRating, stdRating)

def getFullRatingDetails(code, date, type='revised_rating'):
    API = "http://www.ecfrating.org.uk/v2/new/api.php?v2/ratings/"
    qq = API+"R/"+code+"/"+date  # date in yyyy-mm-dd format
    response = requests.get(qq)  
    data = response.json()
    rapidDetail = data

    qq = API+"S/"+code+"/"+date  # date in yyyy-mm-dd format
    response = requests.get(qq)
    data = response.json()
    stdDetail = data

    return (rapidDetail, stdDetail)

def getRibbandsRating(player_code, opponent_code, date):
    playerStats = getFullRatingDetails(player_code)
    opponentStats = getFullRatingDetails(opponent_code)


def calcELOgain(player, opponent, score, K=20):
    gain = K*(score-1/(1+m.pow(10,max(-400,min(-(player-opponent),400))/400))) 
    
    return gain 

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

    print(gains) 
    print(type(gains))

    # gains = [calcELOgain(player, opponents.loc[i,'TournamentRating'],opponents.loc[i,'score']) 
    #                for i in opponents.index]
    
    avgGain = round(sum(gains)/nGames,1)

    return (TPR, avgGain, avgOppRating, wins, losses, nGames)


def compileStandings(f, event='LN00003877'):
    TPRdata = pd.read_excel(f, sheet_name='Players')
    TPRdata.index = [i[:6] for i in TPRdata['BCFCode']]
    TPRdata.index = TPRdata.index.astype(int)

    TPR = pd.DataFrame([getTPR(i,event,TPRdata) for i in TPRdata['BCFCode']], columns=['RibbandsTPR','avgGain',
        'avgOpp','wins','losses','nGames'])
    TPR.index = TPRdata.index
    results = pd.concat([TPRdata, TPR], axis=1)
    results['RibbandsIdx'] = results['RibbandsTPR']/results['TournamentRating']

    results['RibbandsIdx'] = results.apply(lambda x: -1000+x['avgGain'] if x['nGames']<8 else x['avgGain'], axis=1)

    results.sort_values('RibbandsIdx', ascending=False, inplace=True)
    results['Score'] = results['wins']+(results['nGames']-results['wins']-results['losses'])*0.5

    displayOrder = ['NAME', 'Score','nGames','avgGain', 'TournamentRating', 'avgOpp']
    
    results = results[displayOrder]

    return results

def compileRatings():
    players = pd.read_csv('RRperformance.csv')
    dt1 = '2023-05-01'
    dt2 = '2023-08-01'

    ratings1 = [getRating(code, dt1) for code in players['Grade Code']]
    ratings1 = pd.DataFrame(ratings1, columns=['RapidMay','StandardMay'])
   
    ratings2 = [getRating(code, dt2) for code in players['Grade Code']]
    ratings2 = pd.DataFrame(ratings2, columns=['RapidAug','StandardAug'])
     
    R = pd.concat([players, ratings1, ratings2], axis=1)
    return R

def getECFcode(name):
    bbl = {'Anthony':'154753H','Vikram':'302840K','Rob':'173087D','Paul':'127446G'}

    return bbl[name]

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

def getPlayerDetails(i, results, K=20):

    whites = results[results['PIN1']==i]
    blacks = results[results['PIN2']==i]

    if len(whites.index)==0:
        wGains=[]
        wOppRatings=[]
    else:
        wGains = whites.apply(lambda x: calcECFgain(x['RATING1'],x['RATING2'],x['SCORE1'],K=K), axis=1)
        wOppRatings=whites['RATING2']

    if len(blacks.index)==0:
        bGains=[]
        bOppRatings=[]
    else:
        bGains = blacks.apply(lambda x: calcECFgain(x['RATING2'],x['RATING1'],x['SCORE2'],K=K), axis=1)
        bOppRatings=blacks['RATING1']

    gameScore = sum(whites['SCORE1'])+sum(blacks['SCORE2'])

    out=list(wGains)+list(bGains)
    oppRatings = list(wOppRatings)+list(bOppRatings)
    
    avgOppRating=(round(np.mean(oppRatings),0)).astype(int)

    nGames = len(out)
    # avgGain = round(np.mean(out),1)
    Gains = round(np.sum(out),0)
    return gameScore, nGames, avgOppRating, Gains


def scorePlayer(i, results,K=20):

    whites = results[results['PIN1']==i]
    blacks = results[results['PIN2']==i]

    if len(whites.index)==0:
        wGains=[]
        wOppRatings=[]
    else:
        wGains = whites.apply(lambda x: calcECFgain(x['RATING1'],x['RATING2'],x['SCORE1'],K=K), axis=1)
        wOppRatings=whites['RATING2']

    if len(blacks.index)==0:
        bGains=[]
        bOppRatings=[]
    else:
        bGains = blacks.apply(lambda x: calcECFgain(x['RATING2'],x['RATING1'],x['SCORE2'],K=K), axis=1)
        bOppRatings=blacks['RATING1']

    gameScore = sum(whites['SCORE1'])+sum(blacks['SCORE2'])

    out=list(wGains)+list(bGains)
    oppRatings = list(wOppRatings)+list(bOppRatings)
    
    avgOppRating=(round(np.mean(oppRatings),0)).astype(int)

    nGames = len(out)
    # avgGain = round(np.mean(out),1)
    Gains = round(np.sum(out),0)
    return gameScore, nGames, avgOppRating, Gains

def getPlayersResults(f, pList='Player_List', rList='Results_List', seedRating=None):
    players = pd.read_excel(f, sheet_name=pList)
    results = pd.read_excel(f, sheet_name=rList)

    players.set_index('NAME', inplace=True)  # to allow setting of custom seed rating by Name

    if seedRating is not None:
        players['TournamentRating'] = seedRating

    players.reset_index(inplace = True)

    players['PIN'] = players['PIN'].astype(int)
    results['PIN1'] = results['PIN1'].astype(int)
    results['PIN2'] = results['PIN2'].astype(int)

    players.set_index('PIN', inplace=True)

    results['RATING1'] = list(players.loc[results['PIN1'],'TournamentRating'])
    results['RATING2'] = list(players.loc[results['PIN2'],'TournamentRating'])
    results['SCORE1'] = [1 if i==10.0 else 0.5 if i==55.0 else 0 for i in results['Result']]
    results['SCORE2'] = [0 if i==10.0 else 0.5 if i==55.0 else 1 for i in results['Result']]

    return players, results

def compileStandingsFromGradingData(f, K=20, pList = 'Player_List', rList='Results_List', seedRating = None):

    players, results = getPlayersResults(f, pList, rList, seedRating = seedRating)

    # whoPlayed = pd.concat([results.PIN1, results.PIN2]).unique()

    df = pd.DataFrame([scorePlayer(i, results, K=K) for i in players.index],
                        columns=['Score','Played','avgOppRating', 'Gain'])
    
    df.index = players.index
    # df.index = whoPlayed

    df.insert(2,'Rating',players.loc[df.index,'TournamentRating'])

    out=players[['BCFCode','NAME']].merge(df, left_index=True, right_index=True)

    out = out[['NAME', 'Score', 'Played', 'Gain', 'Rating', 'avgOppRating']]
    out['TPR']=round(out['avgOppRating']+400*(2*out['Score']-out['Played'])/out['Played'],0).astype(int)

    return out

def xtableACA(f, K=20, pList = 'Player_List', rList='Results_List', seedRating=None):
    players, results = getPlayersResults(f, pList, rList, seedRating = seedRating)
    
    summ = compileStandingsFromGradingData(f, K=K, pList=pList, rList=rList, seedRating = seedRating)

    summ.insert(3,'Perf',round(summ['Score']/summ['Played'],2))
    # summ.loc[summ['Played']<3,'Perf']=-1
    # summ.sort_values('Perf', ascending=False, inplace=True)

    summ['liveMMR'] = summ['Rating']+summ['Gain']
    summ['Rank'] = summ.apply(lambda x: x['liveMMR'] if x['Played']>=6 else -5000+x['liveMMR'], axis=1)
    summ.sort_values('Rank', ascending=False, inplace=True)
    summ.drop('Rank',axis=1, inplace=True)

    out = pd.DataFrame(index=summ.index, columns=summ.index)
    out = out.apply(pd.to_numeric)

    for i in results.index:
        white = results.loc[i,'PIN1']
        black = results.loc[i,'PIN2']

        if pd.isna(out.loc[white,black]):
            out.loc[white,black] = results.loc[i,'SCORE1']
            out.loc[black,white] = results.loc[i,'SCORE2']
        else:
            out.loc[white,black]=out.loc[white,black]+results.loc[i,'SCORE1']
            out.loc[black,white]=out.loc[black,white]+results.loc[i,'SCORE2']

    out['Name'] = players.loc[out.index,'NAME']
     
    out.fillna(' ', inplace=True)


    out.insert(0,'Score',summ.loc[out.index,'Score'])
    out.insert(1,'Played',summ.loc[out.index,'Played'])
    out.insert(2,'AvgOpponent',summ.loc[out.index,'avgOppRating']) 
    out.insert(3,'StartMMR',summ.loc[out.index,'Rating'])
    out.insert(4,'XP',summ.loc[out.index,'Gain'])
    out.insert(5,'LiveMMR',summ.loc[out.index,'liveMMR'])
    out.insert(6,'TPR',summ.loc[out.index,'TPR'])
 #   out.insert(7,'House',players.loc[out.index,'House'])
    out.reset_index(inplace=True)
    
    tmp = out.pop('PIN')
    out.insert(7,'IDX',tmp)

    out.set_index('Name', inplace=True)

    out['IDX']=[i+1 for i in range(out.shape[0])]
    out.columns=list(out.columns[:8]) + list(out['IDX'])

    return out



def xtableResults(f):
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


def compileTPRs(f,event):
    TPRdata = pd.read_csv(f)
    TPRdata.index = [i[:6] for i in TPRdata['Grade Code']]
    TPRdata.index = TPRdata.index.astype(int)

    TPRs = [getTPR(i, event,TPRdata) for i in TPRdata['Grade Code']]
    TPRdata.insert(2,'RibbandsTPR',TPRs)

    TPRdata.insert(3,'RibbandsScore',TPRdata['RibbandsTPR']-TPRdata['TournamentRating'])

    return(TPRdata)

def getPlayerPairings(pairings, player):
    pairs = copy.deepcopy(pairings)

    return pairs[(pairs['White']==player) | (pairs['Black']==player)]

def calcPlayerStats(pairings, nTeams):
    """
    Runs player stats for a given player
    """
   
    pairs = copy.deepcopy(pairings)

    pairs['WhiteTeam'] = [i[0] for i in pairs['White']]
    pairs['BlackTeam'] = [i[0] for i in pairs['Black']]

    pairs['WhiteBoard'] = [int(i[-1:]) for i in pairs['White']]
    pairs['BlackBoard'] = [int(i[-1:]) for i in pairs['Black']]
    
    players = list(set(list(pairs['White'])+list(pairs['Black'])))
    
    # initialise the table that will store everything
    Teams = list(string.ascii_uppercase[:nTeams])

    out = pd.DataFrame(index=players,
            columns=['Whites','Blacks']+Teams)
 
    out.iloc[:,2:] = 0

    for player in players:
        # first just extract the match ups for the given player
        games = copy.deepcopy(pairs[(pairs['White']==player) | (pairs['Black']==player)])
        
        nRounds = games.shape[0] 
        games['Colour'] = games.apply(lambda x: 'W' if x['White']==player else 'B', axis=1)

        # number of Whites and Blacks
        nWhites = sum(games['Colour']=='W')
        nBlacks = sum(games['Colour']=='B')

        # extract list of teams the player faces
        teams = list(games['WhiteTeam'])+list(games['BlackTeam'])
        teamsFaced=[i[0] for i in teams if i != player]
        
        out.loc[player,'Whites'] = nWhites
        out.loc[player,'Blacks'] = nBlacks
        
        for i in teamsFaced:
            out.loc[player,i] = out.loc[player,i]+1
        out.loc[player,player[0]] = 0

    out = out.apply(pd.to_numeric)
    out.insert(0,'Team',[i[0] for i in out.index])

    return out.groupby('Team').sum(), out.sort_index()

def calcTeamStatsRoundByRound(pairings, nTeams):
    out = [calcPlayerStats(pairings[pairings['Round']==i+1], nTeams) for i in range(3)]
    teamStats = [i[0] for i in out]

    return teamStats

def checkPairings(df):
    """
    Check jamboree pariings.
    """

    # first check number of Whites and Blacks
    # let's work out the number of teams in the pairings

    pairs=copy.deepcopy(df)
    pairs['WhiteTeam'] = [i[0] for i in pairs['White']]
    pairs['BlackTeam'] = [i[0] for i in pairs['Black']]

    pairs['WhiteBoard'] = [int(i[-1:]) for i in pairs['White']]
    pairs['BlackBoard'] = [int(i[-1:]) for i in pairs['Black']]
    
    # not totally robust, but assume each distinct team gets at least one White
    nTeams = pairs['WhiteTeam'].unique()
    
    # now assign a list of the team letters
    Teams = list(string.ascii_uppercase[:nTeams])


    return pairs


