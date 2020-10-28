import json
import copy
def insertdata(frame):
    myframe={}
    hands = frame['hands']
    temhand=copy.copy(hands[0])
    if(temhand['type']=='left'):
        temhand['type']='right'
    else:
        temhand['type'] = 'left'
    temhand['id'] = temhand['id'] + 1
    hands.append(temhand)
    pointposition=frame['pointables']
    for i in range(len(pointposition)):
        tempointposition=copy.copy(pointposition[i])
        tempointposition['handId']=tempointposition['handId']+1
        pointposition.append(tempointposition)
    for k,v in frame.items():
        myframe[k]=v
        if(k=='hands'):
            myframe[k]=hands
        if(k=='pointables'):
            myframe[k]=pointposition
    return myframe
