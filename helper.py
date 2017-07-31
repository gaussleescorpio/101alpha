import numpy as np
import pandas as pd
import talib
import os
import matplotlib.pyplot as plt
import datetime
import pytz
def getTick(shareFolder,exc,ins,date,session,extension):
    filename =os.path.join(shareFolder,exc,ins,ins+"-"+str(date)+"-"+session+"."+extension)
    if os.path.isfile(filename):
        data = pd.read_csv(filename)
        if 'turnOver' in data.columns:
            data.rename(columns = {'turnOver':'turnover'}, inplace=True)
        if ' volumeAcc' in data.columns:
            data.rename(columns = {' volumeAcc':'volumeAcc'}, inplace=True)
        #print(data.columns)
        data = data[1:-1]
        data.reset_index(inplace=True)
        return data
    return None
def getInstrumentInfo(ins):
    product_name=ins[:-4].upper()
    instrument_infos= pd.read_csv("InstrumentInfo.txt")
    instrument_infos['InstrumentID']=list(map(str.upper, instrument_infos['InstrumentID']))
    indexed_instrument_infos=instrument_infos.set_index('InstrumentID')
    found_instrument_info=indexed_instrument_infos.loc[product_name]
    return found_instrument_info
def getOHLC(close,n):
    a = float('nan')
    data_open=close[:-n+1]
    data_open=np.pad(data_open,(n-1,0),'constant',constant_values=(a,0))
    high = pd.Series(close).rolling(n).max()
    low = pd.Series(close).rolling(n).min()
    df = pd.DataFrame(dict(open=data_open,high=high,low=low,close=close))
    return df
def TSI(df, r=25, s=13):  
    M = df.diff(1)  
    aM = abs(M)  
    EMA1 = M.ewm(span = r, min_periods = r - 1).mean()
    aEMA1 = aM.ewm(span = r, min_periods = r - 1).mean()
    EMA2 = pd.Series(EMA1.ewm(span = s, min_periods = s - 1).mean())
    aEMA2 = pd.Series(aEMA1.ewm(span = s, min_periods = s - 1).mean())  
    TSI = pd.Series(EMA2 / aEMA2, name = 'TSI_' + str(r) + '_' + str(s))  
    return TSI
def STO(df, n):  
    avg = (df['high']+df['low'])/2.0
    SOk = pd.Series((df['close'] - avg) / ((df['high'] - df['low'])/2), name = 'SO%k')  
    SOd = pd.Series(SOk.ewm(span = n, min_periods = n - 1).mean(), name = 'SO%d_' + str(n))   
    return SOd
def CCI(df,n):
    average= pd.Series(df).rolling(n).mean()
    mad = lambda x: np.fabs(x - x.mean()).mean()
    meanDeviation=pd.Series(df).rolling(n).apply(mad)
    result=np.where(meanDeviation!=0,(df-average)/(10*meanDeviation) ,0)
    return result
def convertTick(row):
    bidMap={row.bb1:row.bbz1,row.bb2:row.bbz2,row.bb3:row.bbz3,row.bb4:row.bbz4,row.bb5:row.bbz5}
           #row.bb6:row.bbz6,row.bb7:row.bbz7,row.bb8:row.bbz8,row.bb9:row.bbz9,row.bb10:row.bbz10}
    askMap={row.ba1:row.baz1,row.ba2:row.baz2,row.ba3:row.baz3,row.ba4:row.baz4,row.ba5:row.baz5}
           #row.ba6:row.baz6,row.ba7:row.baz7,row.ba8:row.baz8,row.ba9:row.baz9,row.ba10:row.baz10}
    return bidMap,askMap
def estimateTrade(lastTick,newTick,contractMultiplier):
    #trade = newTick.volume / 2.0
    trade = (newTick.volumeAcc - lastTick.volumeAcc) / 2.0
    amount = (newTick.turnover- lastTick.turnover)/ 2.0
    if trade == 0 :
        return (0,0, (newTick.bb1+newTick.ba1)/2,0,0)
    
    accuratedMidPrice = (newTick.turnover- lastTick.turnover)/(trade * 2 *contractMultiplier)
    estiAvgAskPrice,estiAvgBidPrice,TOA,TOB =0.0,0.0,0.0,0.0
    lastbb,lastba = convertTick(lastTick)
    newbb,newba = convertTick(newTick)
    
    if accuratedMidPrice > lastTick.ba1 and (newTick.bb1 > lastTick.bb1 or 
                                              (newTick.bb1 == lastTick.bb1 and newTick.bbz1 > lastTick.bbz1)):
        #assume all on ask
        return (trade ,0, accuratedMidPrice,accuratedMidPrice,0)
    
    if accuratedMidPrice < lastTick.bb1 and (newTick.ba1 < lastTick.ba1 or 
                                              (newTick.ba1 == lastTick.ba1 and newTick.baz1 > lastTick.baz1)):
        #assume all on ask
        return (0, trade, accuratedMidPrice ,0 ,accuratedMidPrice)
    
    
    if newTick.ba1 >= lastTick.ba1:
        totalTradeOnAsk =0.0
        turnoverOnAsk =0.0
        for (k,v) in lastba.items() :
            trades =0
            if k < newTick.ba1:
                trades = v
            elif k == newTick.ba1 and v > newTick.baz1:
                trades = v-newTick.baz1
            if trades > (trade -totalTradeOnAsk):
                trades = trade -totalTradeOnAsk
            totalTradeOnAsk += trades
            turnoverOnAsk += trades * k
        if totalTradeOnAsk > 0 :
            estiAvgAskPrice =turnoverOnAsk/totalTradeOnAsk
    elif lastTick.ba1 in newba:
        if newba[lastTick.ba1] < lastTick.baz1 :
            estiAvgAskPrice = lastTick.ba1
            
    if newTick.bb1 <= lastTick.bb1:
        totalTradeOnBid =0.0
        turnoverOnBid =0.0
        for (k,v) in lastbb.items() :
            bidTrades =0
            if k > newTick.bb1:
                bidTrades = v
            elif k == newTick.bb1 and v > newTick.bbz1:
                bidTrades=v-newTick.bbz1
            if bidTrades > (trade -totalTradeOnBid):
                bidTrades = trade -totalTradeOnBid
            totalTradeOnBid += bidTrades
            turnoverOnBid += bidTrades * k
        if totalTradeOnBid > 0 :
            estiAvgBidPrice =turnoverOnBid/totalTradeOnBid
    elif lastTick.bb1 in newbb:
        if newbb[lastTick.bb1] < lastTick.bbz1 :
            estiAvgBidPrice = lastTick.bb1
    if estiAvgBidPrice == 0 and estiAvgAskPrice == 0:
        estiAvgAskPrice = newTick.ba1
        estiAvgBidPrice = newTick.bb1
        
    TOA=round((amount/contractMultiplier - trade * estiAvgBidPrice)/(estiAvgAskPrice-estiAvgBidPrice))
    if TOA > trade:
        TOA = trade
    elif TOA < 0:
        TOA = 0
    TOB = trade - TOA
    
    if TOA == 0:
        estiAvgBidPrice = accuratedMidPrice
    if TOB == 0:
        estiAvgAskPrice = accuratedMidPrice
    
    return (TOA ,TOB, accuratedMidPrice,estiAvgAskPrice,estiAvgBidPrice)

def getTickDetailByTurnover(lastTick,newTick,instrument_info):
    TOA ,TOB ,accuratedMidPrice, askAveragePrice, bidAveragePrice = estimateTrade(lastTick,newTick,instrument_info.ContractMultiplier)
    unknownTrade, cancelOnBid, cancelOnAsk, changeOnBid, changeOnAsk, newOnBid, newOnAsk, newFrontBid, newFrontAsk = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    lastbb,lastba = convertTick(lastTick)
    newbb,newba = convertTick(newTick)
    newFrontAsk +=sum({k:v for (k,v) in newba.items() if k<lastTick.ba1}.values())
    newFrontBid +=sum({k:v for (k,v) in newbb.items() if k>lastTick.bb1}.values())
    changeOnAsk -=sum({k:v for (k,v) in lastba.items() if k<newTick.ba1}.values())
    changeOnBid -=sum({k:v for (k,v) in lastbb.items() if k>newTick.bb1}.values())
    if lastTick.ba1 in newba:
        changeOnAsk += (newba[lastTick.ba1] - lastTick.baz1)
    elif newTick.ba1 in lastba:
        changeOnAsk += (newTick.baz1 - lastba[newTick.ba1])
        
    if lastTick.bb1 in newbb:
        changeOnBid += (newbb[lastTick.bb1] - lastTick.bbz1)
    elif newTick.bb1 in lastbb:
        changeOnBid += (newTick.bbz1 - lastbb[newTick.bb1])
        
    if changeOnBid < 0:
        if TOB > -1.0* changeOnBid:
            newOnBid = TOB +changeOnBid
            cancelOnBid = 0
        else:
            newOnBid =0
            cancelOnBid = -1 * changeOnBid -TOB
    else:
        newOnBid = changeOnBid
    if changeOnAsk < 0:
        if TOA > -1.0* changeOnAsk:
            newOnAsk = TOA +changeOnAsk
            cancelOnAsk = 0
        else:
            newOnAsk =0
            cancelOnAsk = -1 * changeOnAsk -TOA
    else:
        newOnAsk = changeOnAsk
    newOnBid += newFrontBid
    newOnAsk += newFrontAsk
    return TOA,TOB,cancelOnBid,cancelOnAsk,newOnBid,newOnAsk, askAveragePrice, bidAveragePrice

def getBernardDetail(shareFolder,exc,ins,date,session):
    data=getTick(shareFolder,exc,ins,date,session,"tick")
    if data is None:
        return None
    numberOfRows= len(data)
    instrument_info=getInstrumentInfo(ins)
    lastRow= None
    res=[]
    for row in data.itertuples():
        
        vwap=0.0
        flow_d3=0.0
        TOA,TOB,cancelOnBid,cancelOnAsk,newOnBid,newOnAsk, askAveragePrice, bidAveragePrice=0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
        bb,ba=convertTick(row)
        bidCut = row.bb1 - 2*instrument_info.TickStep
        askCut = row.ba1 + 2*instrument_info.TickStep
        midPx=(row.ba1+row.bb1)/2
        if(lastRow is None):
            vwap=midPx
        else:
            currentVolume = row.volumeAcc - lastRow.volumeAcc
            vwap=(row.turnover-lastRow.turnover)/(currentVolume *instrument_info.ContractMultiplier) if currentVolume!=0 else midPx
            lastbb,lastba=convertTick(lastRow)
            filtered_lastbb = {k:v for (k,v) in lastbb.items() if k>=bidCut}
            filtered_bb = {k:v for (k,v) in bb.items() if k>=bidCut}
            filtered_ba= {k:v for (k,v) in ba.items() if k<=askCut}
            filtered_lastba = {k:v for (k,v) in lastba.items() if k<=askCut}
            flow_d3 = sum(filtered_bb.values())-sum(filtered_lastbb.values())-sum(filtered_ba.values())+sum(filtered_lastba.values())
            TOA,TOB,cancelOnBid,cancelOnAsk,newOnBid,newOnAsk, askAveragePrice, bidAveragePrice =getTickDetailByTurnover(lastRow,row,instrument_info)
        imba4_d3 = (row.bbz1+row.bbz2+row.bbz3-row.baz1-row.baz2-row.baz3)/(row.bbz1+row.bbz2+row.bbz3+row.baz1+row.baz2+row.baz3)
        imba4_d2=  (row.bbz1+row.bbz2-row.baz1-row.baz2)/(row.bbz1+row.bbz2+row.baz1+row.baz2)
        adjPrice = (row.bb1*row.baz1 +row.ba1*row.bbz1)/(row.bbz1 + row.baz1)
        res.append({'vwap':vwap,'flow_d3':flow_d3,'imba4_d2':imba4_d2,'imba4_d3':imba4_d3,'adjPx':adjPrice,
                    'midPx':midPx,'TOA':TOA,'TOB':TOB,'COA':cancelOnAsk,'COB':cancelOnBid,
                     'NOA':newOnAsk,'NOB':newOnBid, 'askAveragePrice': askAveragePrice, 'bidAveragePrice':bidAveragePrice})
        lastRow= row
    newDF = pd.DataFrame(res)
    filterDataFrame = data[["utcReceiveTime",'lastPrice','volume','volumeAcc','bbz1','bb1','ba1','baz1','utcQuoteTime','turnover']]
    mergedDF= pd.concat([filterDataFrame,newDF],axis=1)
    return mergedDF

    
def getTechnicalData(shareFolder,exc,ins,date,session,drop_na = False):    
    def get_log_div(data1, data2, multiplier=1):
        data1 = data1.fillna(0)
        data2 = data2.fillna(0)
        cond = (data2 != 0) & (data1 != 0)
        res = pd.Series(np.zeros(len(data1)), index=data1.index)
        res[cond] = np.log( (data1[cond] / data2[cond]) * multiplier )
        return res
    data=getBernardDetail(shareFolder,exc,ins,date,session)
    if data is None:
        return None
    fvolume=np.array(data.volume.as_matrix(),dtype=float)
    instrument_info=getInstrumentInfo(ins)
    
    ACCVOL1M = pd.Series((data.volumeAcc.diff(120)).fillna(0),name="ACCVOL1M")
    ACCVOL1MDIF = pd.Series((ACCVOL1M.diff(120)).fillna(0),name="ACCVOL1MDIF")    
    targetSeries = None
    #for i in range(0,120):
    #    newVolumeList = ACCVOL1M[lambda x:x.index%120==i]        
    #    emaVolume =talib.EMA(newVolumeList.as_matrix())
    #    returnSeries = pd.Series(emaVolume,index=newVolumeList.index)        
    #    if targetSeries is None:
    #        targetSeries = returnSeries
    #    else:
    #        targetSeries = targetSeries.append(returnSeries)
    #ACCVOL1MEMA = pd.Series(targetSeries.sort_index(),name = "ACCVOL1MEMA")
    
    numToFill =len(ACCVOL1M)%120
    padded=np.pad(ACCVOL1M.values,(0,120-numToFill),'constant',constant_values=(0,0))
    newShape= padded.reshape(int(len(padded)/120),120)
    newEMA=np.apply_along_axis(talib.EMA,0,newShape)
    ACCVOL1MEMA =pd.Series(newEMA.reshape(1,int(len(padded)))[0][ :len(padded) + numToFill - 120],name="ACCVOL1MEMA")
    
    ACCVOL1MDIFRATIO =  pd.Series((ACCVOL1MDIF/ACCVOL1MEMA).fillna(0),name="ACCVOL1MDIFRATIO")

    NEXT=pd.Series(np.pad(data.vwap.diff(1).dropna(),(0,1),'constant',constant_values=(0,0)),name="NEXT")
    NEXT2=pd.Series(np.pad(data.vwap.diff(2).dropna(),(0,1),'constant',constant_values=(0,0)),name="NEXT2")
    NEXT3=pd.Series(np.pad(data.vwap.diff(3).dropna(),(0,1),'constant',constant_values=(0,0)),name="NEXT3")
    NEXT4=pd.Series(np.pad(data.vwap.diff(4).dropna(),(0,1),'constant',constant_values=(0,0)),name="NEXT4")
    NEXT5=pd.Series(np.pad(data.vwap.diff(5).dropna(),(0,1),'constant',constant_values=(0,0)),name="NEXT5")
    NEXT10=pd.Series(np.pad(data.vwap.diff(10).dropna(),(0,10),'constant',constant_values=(0,0)),name="NEXT10") 
    NEXT30=pd.Series(np.pad(data.vwap.diff(30).dropna(),(0,30),'constant',constant_values=(0,0)),name="NEXT30") 
    NEXT120=pd.Series(np.pad(data.vwap.diff(120).dropna(),(0,120),'constant',constant_values=(0,0)),name="NEXT120") 
    vwap_price_diff =pd.Series((data.vwap-data.midPx),name="vwap_mid_diff")
    adjpx_mid_diff = pd.Series((data.adjPx -data.midPx),name="adjpx_mid_diff")
    estimateTrade
    ROC_VWAP_1200 = pd.Series( (get_log_div( data.vwap, data.vwap.shift(1200) )).fillna(0),name="ROC-VWAP-1200")
    ROC_VWAP_180 = pd.Series((get_log_div(data.vwap, data.vwap.shift(180) )).fillna(0),name="ROC-VWAP-180")
    ROC_VWAP_90 = pd.Series((get_log_div(data.vwap, data.vwap.shift(90) )).fillna(0),name="ROC-VWAP-90")
    ROC_VWAP_30 = pd.Series((get_log_div(data.vwap, data.vwap.shift(30) )).fillna(0),name="ROC-VWAP-30")
    ROC_VWAP_10 = pd.Series((get_log_div(data.vwap, data.vwap.shift(10) )).fillna(0),name="ROC-VWAP-10")
    
    VWAP_DIFF_1200 = pd.Series((data.vwap - data.vwap.shift(1200)).fillna(0),name="VWAP-DIFF-1200")
    VWAP_DIFF_180 = pd.Series((data.vwap - data.vwap.shift(180)).fillna(0),name="VWAP-DIFF-180")
    VWAP_DIFF_100 = pd.Series((data.vwap - data.vwap.shift(100)).fillna(0),name="VWAP-DIFF-100")
    VWAP_DIFF_90 = pd.Series((data.vwap - data.vwap.shift(90)).fillna(0),name="VWAP-DIFF-90")
    VWAP_DIFF_80 = pd.Series((data.vwap - data.vwap.shift(80)).fillna(0),name="VWAP-DIFF-80")
    VWAP_DIFF_70 = pd.Series((data.vwap - data.vwap.shift(70)).fillna(0),name="VWAP-DIFF-70")
    VWAP_DIFF_60 = pd.Series((data.vwap - data.vwap.shift(60)).fillna(0),name="VWAP-DIFF-60")
    VWAP_DIFF_50 = pd.Series((data.vwap - data.vwap.shift(50)).fillna(0),name="VWAP-DIFF-50")
    VWAP_DIFF_40 = pd.Series((data.vwap - data.vwap.shift(40)).fillna(0),name="VWAP-DIFF-40")
    VWAP_DIFF_30 = pd.Series((data.vwap - data.vwap.shift(30)).fillna(0),name="VWAP-DIFF-30")
    VWAP_DIFF_20 = pd.Series((data.vwap - data.vwap.shift(20)).fillna(0),name="VWAP-DIFF-20")
    VWAP_DIFF_10 = pd.Series((data.vwap - data.vwap.shift(10)).fillna(0),name="VWAP-DIFF-10")
    
    BAZ_RATIO = pd.Series(data.bbz1 / data.baz1, name="BAZ_RATIO")
    
    
    
    macd,macdsignal,macdhist =  talib.MACD(data.vwap.as_matrix(),11,23,5) 
    MACD_VWAP = pd.Series(macd-macdsignal,name="MACD-VWAP")
    macd,macdsignal,macdhist =  talib.MACD(fvolume,11,23,5) 
    MACD_VOLUMN = pd.Series(macd-macdsignal,name="MACD-VOLUMN")
    macd,macdsignal,macdhist =  talib.MACD(data.imba4_d3.as_matrix(),11,23,5) 
    MACD_IMBA = pd.Series(macd-macdsignal,name="MACD-IMBA")
    macd,macdsignal,macdhist =  talib.MACD(data.flow_d3.as_matrix(),11,23,5) 
    MACD_FLOW = pd.Series(macd-macdsignal,name="MACD-FLOW")
    STOC_DIF_15_RAW = STO(getOHLC(adjpx_mid_diff.rolling(15).sum(),30),30)
    STOC_DIF_15 = pd.Series(STOC_DIF_15_RAW,name="STOC-DIF-15")
    STOC_IMBA_RAW = STO(getOHLC(data.imba4_d3,30),30)
    STOC_IMBA = pd.Series(STOC_IMBA_RAW,name="STOC-IMBA")
    STOC_FLOW_RAW = STO(getOHLC(data.flow_d3,30),30)
    STOC_FLOW = pd.Series(STOC_FLOW_RAW,name="STOC-FLOW")
    STOC_VWAP1_RAW = STO(getOHLC(data.vwap,10),10)
    STOC_VWAP1 = pd.Series(STOC_VWAP1_RAW,name="STOC-VWAP1")
    STOC_VWAP2_RAW = STO(getOHLC(data.vwap,30),30)
    STOC_VWAP2 = pd.Series(STOC_VWAP2_RAW,name="STOC-VWAP2")
    STOC_EMA_VOLUME_RAW = STO(getOHLC(talib.EMA(fvolume),30),30)
    STOC_EMA_VOLUME = pd.Series(STOC_EMA_VOLUME_RAW,name="STOC-EMA-VOLUME")
    TSI_DIF_15_RAW = TSI(adjpx_mid_diff.rolling(15).sum(),39,39)
    TSI_DIF_15 = pd.Series(TSI_DIF_15_RAW,name="TSI-DIF-15")
    TSI_IMBA_RAW = TSI(data.imba4_d3,39,39)
    TSI_IMBA = pd.Series(TSI_IMBA_RAW,name="TSI-IMBA")
    TSI_FLOW_RAW = TSI(data.flow_d3,39,39)
    TSI_FLOW = pd.Series(TSI_FLOW_RAW,name="TSI-FLOW")
    TSI_VWAP_RAW = TSI(data.vwap,39,39)
    TSI_VWAP = pd.Series(TSI_VWAP_RAW,name="TSI-VWAP")
    EMAV = pd.Series(talib.EMA(fvolume),name="EMAV")
    TSI_EMAV_RAW = TSI(pd.Series(talib.EMA(fvolume)),39,39)
    TSI_EMAV = pd.Series(TSI_EMAV_RAW,name="TSI-EMAV")
    CCI2_EMAV_RAW = CCI(talib.EMA(fvolume),20)
    CCI2_EMAV = pd.Series(CCI2_EMAV_RAW,name="CCI-EMAV")
    CCI2_DIF_15_RAW = CCI(adjpx_mid_diff.rolling(15).sum(),20)
    CCI2_DIF_15 = pd.Series(CCI2_DIF_15_RAW,name="CCI-DIF-15")
    CCI2_IMBA_RAW = CCI(data.imba4_d3,20)
    CCI2_IMBA = pd.Series(CCI2_IMBA_RAW,name="CCI-IMBA")
    CCI2_FLOW_RAW = CCI(data.flow_d3,20)
    CCI2_FLOW = pd.Series(CCI2_FLOW_RAW,name="CCI-FLOW")
    CCI2_VWAP_RAW = CCI(data.vwap,20)
    CCI2_VWAP = pd.Series(CCI2_VWAP_RAW,name="CCI-VWAP")
    # newly added features
    VOL_DIFF_30 = data.volumeAcc - data.volumeAcc.shift(30)
    V_30 = pd.Series(VOL_DIFF_30, name="V_30")
    
    
    V_30_DIFF = pd.Series(get_log_div(V_30 , V_30.shift(30)), name="ROC-V30")
    
    V_10_DIFF = pd.Series(get_log_div(V_30 , V_30.shift(10)), name="ROC-V10")
    
    V_300_DIFF = pd.Series(get_log_div(V_30 , V_30.shift(300)), name="ROC-V300")
    
    VWAP_10 = data.turnover - data.turnover.shift(10)
    VWAP_10.loc[VWAP_10 != 0] = VWAP_10.loc[VWAP_10 != 0] / (data.volumeAcc - data.volumeAcc.shift(10)).loc[VWAP_10 != 0] / 100
    VWAP_10.loc[VWAP_10 == 0] = data.loc[VWAP_10 == 0, "vwap"]
    VWAP_10_VAL = pd.Series(VWAP_10, name="vwap10")
    
    VWAP_30 = data.turnover - data.turnover.shift(30)
    VWAP_30.loc[VWAP_30 != 0] = VWAP_30.loc[VWAP_30 != 0] / (data.volumeAcc - data.volumeAcc.shift(30)).loc[VWAP_30 != 0] / 100
    VWAP_30.loc[VWAP_30 == 0] = data.loc[VWAP_30 == 0, "vwap"]
    VWAP_30_VAL = pd.Series(VWAP_30, name="vwap30")
    
    VWAP_100 = data.turnover - data.turnover.shift(100)
    VWAP_100.loc[VWAP_100 != 0] = VWAP_100.loc[VWAP_100 != 0] / (data.volumeAcc - data.volumeAcc.shift(100)).loc[VWAP_100 != 0] / 100
    VWAP_100.loc[VWAP_100 == 0] = data.loc[VWAP_100 == 0, "vwap"]
    VWAP_100_VAL = pd.Series(VWAP_100, name="vwap100")
    
    V10DIFF = pd.Series(VWAP_10_VAL - data.vwap, name="VWAP10DIFF")
    V30DIFF = pd.Series(VWAP_30_VAL - data.vwap, name="VWAP30DIFF")
    V100DIFF = pd.Series(VWAP_100_VAL - data.vwap, name="VWAP100DIFF")
    
    TOAT = pd.Series(data.TOA * data.askAveragePrice, name="TOAT").fillna(0)
    TOBT = pd.Series(data.TOB * data.bidAveragePrice, name="TOBT").fillna(0)
    AT30 = pd.Series(TOAT.rolling(30, 30).sum(), name="AT30")
    BT30 = pd.Series(TOBT.rolling(30, 30).sum(), name="BT30")
    
    AT30.loc[AT30==0] = 1
    AT30.loc[AT30<=1] = 1
    BT30.loc[BT30==0] = 1
    BT30.loc[BT30<=1] = 1
    
    ROC_AT30 = pd.Series(get_log_div(AT30 , AT30.shift(30)), name="ROC-AT30")
    ROC_BT30 = pd.Series(get_log_div(BT30 , BT30.shift(30)), name="ROC-BT30")
    
    FLOW30D = pd.Series(ROC_BT30 - ROC_AT30, name="FLOW30D")
    
    
    
    TOA_SUM = data.TOA.fillna(0).rolling(30, 30).sum()
    TOA_SUM.loc[TOA_SUM!=0] = TOAT.rolling(30,30).sum().loc[TOA_SUM!=0]/TOA_SUM.loc[TOA_SUM!=0]
    TOA_SUM.loc[TOA_SUM==0] = data.vwap.loc[TOA_SUM==0]
    TA30 = pd.Series(TOA_SUM, name="ta30")
    
    TOB_SUM = data.TOB.fillna(0).rolling(30, 30).sum()
    TOB_SUM.loc[TOB_SUM!=0] = TOBT.rolling(30,30).sum().loc[TOB_SUM!=0]/TOB_SUM.loc[TOB_SUM!=0]
    TOB_SUM.loc[TOB_SUM==0] = data.vwap.loc[TOB_SUM==0]
    TB30 = pd.Series(TOB_SUM, name="tb30")
    
    TATBDIF30 = pd.Series(TA30 - TB30 - 0.5, name="TATBDIF30")
    
    
    df =pd.concat([data,vwap_price_diff,adjpx_mid_diff,NEXT,NEXT10,NEXT30,NEXT120,
                   ROC_VWAP_180,ROC_VWAP_90,ROC_VWAP_30,ROC_VWAP_10,ROC_VWAP_1200,
                   MACD_VWAP,MACD_VOLUMN,MACD_IMBA,MACD_FLOW,STOC_DIF_15,STOC_IMBA,STOC_FLOW,STOC_VWAP1,STOC_VWAP2,STOC_EMA_VOLUME,
                   TSI_DIF_15,TSI_IMBA,TSI_FLOW,TSI_VWAP,TSI_EMAV,CCI2_EMAV,CCI2_DIF_15,CCI2_IMBA,CCI2_FLOW,CCI2_VWAP,EMAV,
                   ACCVOL1M,ACCVOL1MDIF,ACCVOL1MEMA,ACCVOL1MDIFRATIO,V_30,
                  V_30_DIFF, V_10_DIFF, V_300_DIFF, VWAP_10_VAL, V10DIFF, V30DIFF, V100DIFF,
                  TOAT,TOBT,AT30, BT30,ROC_AT30,ROC_BT30, FLOW30D, TA30, TB30, TATBDIF30, VWAP_DIFF_30,
                  VWAP_DIFF_90,VWAP_DIFF_180,VWAP_DIFF_1200,  VWAP_DIFF_100 , VWAP_DIFF_80 ,VWAP_DIFF_70 ,VWAP_DIFF_60,
                   VWAP_DIFF_50 ,VWAP_DIFF_40 , VWAP_DIFF_20 ,VWAP_DIFF_10, BAZ_RATIO, NEXT2, NEXT3, NEXT4, NEXT5],axis=1)
    df= df.rename(columns={'midPx': 'mid', 'lastPrice': 'price','bbz1':'bestBidSize','baz1':'bestAskSize','ba1':'bestAsk','bb1':'bestBid'})
    if drop_na:
        return df.dropna()
    else:
        return df.fillna(0)
def getOverDayData(shareFolder,exc,ins,session,startDate ,endDate,drop_na=False):
    data=getTechnicalData(shareFolder,exc,ins,startDate,"D",drop_na=drop_na)
    data2 =getTechnicalData(shareFolder,exc,ins,startDate,"N",drop_na=drop_na)
    data =data.append(data2) if not(data is None) else data2
    for date in range(startDate+1 ,endDate+1):
        data3=getTechnicalData(shareFolder,exc,ins,date,"D",drop_na=drop_na)
        data=data.append(data3) if not(data is None) else data3
        if date != 20170222:
            data4 =getTechnicalData(shareFolder,exc,ins,date,"N",drop_na=drop_na)
            data =data.append(data4) if not(data is None) else data4
    return data
def getBar(data,n,byTick=True):
    if byTick:
        BarData = data.groupby((data.index/n).astype(int))
        open =pd.Series(BarData.price.first(),name="Open")
        high =pd.Series(BarData.price.max(),name="High")
        low =pd.Series(BarData.price.min(),name="Low")
        close =pd.Series(BarData.price.last(), name ="Close")
        volume = pd.Series(BarData.volume.sum(), name="Volume")
        turnover = pd.Series(BarData.turnover.sum(), name="turnover")
        df =pd.concat([open,high,low,close, volume, turnover],axis=1)
        return df
    else:
        BarData = data.groupby((data.utcQuoteTime/n).astype(int)*n)
        open =pd.Series(BarData.price.first(),name="Open")
        high =pd.Series(BarData.price.max(),name="High")
        low =pd.Series(BarData.price.min(),name="Low")
        close =pd.Series(BarData.price.last(),name ="Close")
        volume = pd.Series(BarData.volume.sum(), name="Volume")
        turnover = pd.Series(BarData.turnover.sum(), name="turnover")
        df =pd.concat([open,high,low,close, volume, turnover],axis=1)
        return df
def getLastTradingDay(date):
    date_object = datetime.datetime.strptime(str(date),'%Y%m%d')
    date_weekday = date_object.isoweekday()
    if date_weekday > 5:
        return date
    elif date_weekday ==1:
        return (date_object - datetime.timedelta(days=3)).strftime('%Y%m%d')
    else:
        return (date_object - datetime.timedelta(days=1)).strftime('%Y%m%d')
    
def _change_fromUTCtime(df, keys, rf_date, time_zone=None):
    if df is None:
        return
    assert isinstance(df, pd.DataFrame), "input must be a pandas dataframe"
    # Trans date to datetime
    date = datetime.datetime.strptime(str(rf_date), "%Y%m%d")
    # Adapt the timezone
    if time_zone is None:
        date = date.replace(tzinfo=pytz.utc).astimezone(pytz.timezone("Asia/Taipei"))
    else:
        date = date.replace(tzinfo=pytz.utc).astimezone(pytz.timezone(time_zone))
    # Change to full datetime with reso in seconds
    epoch = datetime.datetime.utcfromtimestamp(0)
    epoch = epoch.replace(tzinfo=pytz.utc).astimezone(pytz.timezone("Europe/London"))
    for key in keys:
        time_temp = df[key].astype("int64") + (date - epoch + date.utcoffset()).total_seconds() * 1000
        df.loc[:, key] = pd.to_datetime(time_temp, unit="ms")

def getOverDayData2(shareFolder,exc,ins,session,startDate ,endDate,drop_na=False):
    data=getTechnicalData(shareFolder,exc,ins,startDate,"D",drop_na=drop_na)
    _change_fromUTCtime(data, ["utcQuoteTime", "utcReceiveTime"], startDate)
    data2 =getTechnicalData(shareFolder,exc,ins,startDate,"N",drop_na=drop_na)
    _change_fromUTCtime(data2, ["utcQuoteTime", "utcReceiveTime"], startDate)
    data =data.append(data2) if not(data is None) else data2
    for date in range(startDate+1 ,endDate+1):
        data3=getTechnicalData(shareFolder,exc,ins,date,"D",drop_na=drop_na)
        _change_fromUTCtime(data3, ["utcQuoteTime", "utcReceiveTime"], date)
        data=data.append(data3) if not(data is None) else data3
        if date != 20170222:
            data4 =getTechnicalData(shareFolder,exc,ins,date,"N",drop_na=drop_na)
            _change_fromUTCtime(data4, ["utcQuoteTime", "utcReceiveTime"], date)
            data =data.append(data4) if not(data is None) else data4
    return data
