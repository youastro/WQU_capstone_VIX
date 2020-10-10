from clr import AddReference # .NET Common Language Runtime (CLR) <- http://pythonnet.github.io/
AddReference("System")
AddReference("QuantConnect.Algorithm") # to load an assembly use AddReference
AddReference("QuantConnect.Common")

from System import * # CLR namespaces to be treatedas Python packages
from QuantConnect import *
from QuantConnect.Algorithm import *

from QuantConnect.Python import PythonQuandl
from QuantConnect.Data.Custom import *
from QuantConnect.Data.Custom.CBOE import CBOE

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.optimize import minimize
import decimal
from io import StringIO
import bisect

class VIXTermStructure(QCAlgorithm):

    def Initialize(self):

        self.SetStartDate(2015, 1, 1)   # Set Start Date
        self.SetEndDate(2017, 12, 31)     # Set End Date
        self.SetCash(100000000)          # Set Strategy Cash
        
        self.thresholdToPlaceOrder = 0.001
        self.multipler = 1000 # VIX multipler is $1000
        self.fee = 0.05 
        self.vixvolColPrefix = "ivcall" #ivmean doesn't always exist, e.g. 2014/02/14
        
        holidays = self.TradingCalendar.GetDaysByType(TradingDayType.PublicHoliday, self.StartDate, self.EndDate)
        self.holidays = [i.Date.date() for i in holidays]
        
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)

        # self.vix = self.AddData(QuandlVix, "CBOE/VIX", Resolution.Daily).Symbol              # Add Quandl VIX price (daily), no longer works

        Quandl.SetAuthCode("WFW5X2RC9YiGoVUbbsX_")
        self.vxxvol = self.AddData(QuandlVxx, "VOL/VXX", Resolution.Daily, TimeZones.NewYork).Symbol
        self.vixvol = self.AddData(QuandlVxx, "VOL/VIX", Resolution.Daily, TimeZones.NewYork).Symbol

        self.Download("http://cache.quantconnect.com/alternative/cboe/vix.csv")
        self.vix = self.AddData(CBOE,"VIX").Symbol

        # need 8 monthly futures because when calculating the weights I will exclude the current front month
        # and the second month seems to have problem when computing PNL at the expiry date
        self.nfut = 8
        self.nexclude = 1
        self.VIX_futures_names = ["VX" + str(i) for i in range(1, 1 + self.nfut)]
        self.VIX_symbols =[]
        for vname in self.VIX_futures_names:
            # VX9 misses 01/23 - 03/23 in 2015, exclude vx9 from modeling
            self.VIX_symbols.append( self.AddData(QuandlFutures, "CHRIS/CBOE_" + vname, Resolution.Daily).Symbol )
          
        # the data of VXX or vxx.1 don't match with online sources (e.g. yahoo) after 2018
        # https://www.quantconnect.com/forum/discussion/7975/can-039-t-load-historical-vxx-data-even-though-it-looks-like-it-exists/p1
        # use this ticker before 2019
        self.vxx = self.AddEquity("vxx.1",  Resolution.Daily).Symbol                           # Add VXX, vxx.1 is the permtick according to the link
        # use this ticker after 2018
        #self.vxx = self.AddEquity("VXX",  Resolution.Daily).Symbol           
        
        #self.Securities[self.vxx].SetDataNormalizationMode(DataNormalizationMode.Raw);
        
        # 2009-01-30 to 2020-08-10
        #self.vxx = self.AddData(VXXData, "VXX", Resolution.Daily).Symbol

        # Add VIX futures contract data 
        # this is strange, if I set the time resolution to daily, i don't get any future contracts from data.FutureChains
        # even with minute resolution, i don't get future contracts the first time I call data.FutureChains
        vixfuture = self.AddFuture(Futures.Indices.VIX)
        vixfuture.SetFilter(timedelta(0), timedelta(300))

        # 2011 to present
        expiry_f = self.Download("https://www.dropbox.com/s/ny8nxqcp6u76igw/expiry.csv?dl=1")
        self.expiry = pd.read_csv(StringIO(expiry_f), names=["expiry"], parse_dates=["expiry"], infer_datetime_format=True)["expiry"].tolist()

        #stores the map from expiry to the contract
        self.contracts = {}
        
        # index of the front contract
        self.nextRebalanceIndex = bisect.bisect_left(self.expiry, self.StartDate)

        # weight for vix future contracts and vxx (the last one), initialize to 0
        self.curWeight = np.array([0.] * (self.nfut - self.nexclude + 1))
        self.curWeight[0] = -0.1
        self.curWeight[-1] = 0.1

        # we trade at 10am ET every day
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.At(10,0), self.TryTrade)

    def OnData(self, data):
        pass
    
    def TryTrade(self):

        # for some reason the contracts are not necessary available at the first few calls
        # so I have accumulate the contracts in this way
        for chain in self.CurrentSlice.FutureChains.Values:
            contracts = chain.Contracts
            for contract in contracts.Values:
                if contract.Expiry.date() in self.contracts:
                    continue
                self.contracts[contract.Expiry.date()] = contract
                self.Log(str(contract.Symbol) + " expiry=" +  str(contract.Expiry.date()))

        #skip weekends and holidays
        if self.Time.date() in self.holidays:
            self.Log(str(self.Time.date()) + " is a holiday or weekend, skipping ")
            return

        # if it is a expiry day, we liquidate the first contract we trade (not the front month contract), 
        # and shift the weights, but we don't trade
        if self.Time.date() in self.expiry:
            expiry = self.expiry[self.nextRebalanceIndex + self.nexclude].date()
            sec = self.Portfolio[self.contracts[expiry].Symbol]
            if sec.Invested:
                holding = sec.Quantity
                if sec.IsLong:
                    self.MarketOrder(self.contracts[expiry].Symbol, -holding)
                else:
                    self.MarketOrder(self.contracts[expiry].Symbol, holding)
            self.curWeight = self.curWeight[1:] # remove the first element
            self.curWeight = np.append(self.curWeight, 0) # append another one at the end
            self.nextRebalanceIndex += 1
            return

        # now if it is not an expiry date
        weights = self.CalCoeff()
        if weights is None:
            self.Log("failed to calculate weight for " + str(self.Time.date()))
            return
        self.Log(" new weights: " + str(weights) + ", current weights : " + str(self.curWeight))

        self.PrintPort("before trading")
        totalMargin = self.Portfolio.TotalPortfolioValue
        for i in range(self.nfut - self.nexclude):
            expiry = self.expiry[self.nextRebalanceIndex + i + self.nexclude].date()
            if not expiry in self.contracts:
                self.Error("how come the contract with this expiry doesn't exist: " + \
                            str(expiry))
                continue
            
            if (np.abs(weights[i] - self.curWeight[i]) > self.thresholdToPlaceOrder):
                qty = int(totalMargin * (weights[i] - self.curWeight[i]) / \
                    self.Securities[self.contracts[expiry].Symbol].Price / self.multipler)
                orderticket = self.MarketOrder(self.contracts[expiry].Symbol, qty)
                self.Log("traded " + str(self.contracts[expiry].Symbol) + " " + str(qty) + " " \
                    + str(expiry) + " " + str(orderticket.QuantityFilled) + "@" + str(orderticket.AverageFillPrice))
                self.PrintOrderStatus(orderticket)
            else:
                # if not traded because of too small of difference, we need to update in the new weights
                weights[i] = self.curWeight[i]

        qty = int(totalMargin * (weights[-1] - self.curWeight[-1]) / self.Securities[self.vxx].Price)
        orderticket = self.MarketOrder(self.vxx, qty)
        self.Log("traded VXX" + str(orderticket.QuantityFilled) + "@" + str(orderticket.AverageFillPrice) + \
                " status: " + str(orderticket.Status) + " requested qty: " + str(qty) )
        self.PrintOrderStatus(orderticket)

        self.curWeight = weights
        self.PrintPort("after trading")
        
    def CalCoeff(self):
        
        try:
            corr_M = self.CalCorrelationM()
            self.Log("corr:")
            self.Log(corr_M)
            
            exret = self.ExRet()
            self.Log("ExRet")
            self.Log(exret)
            
            vixvol_df = self.VIXVol()
            self.Log("vixvol_df")
            self.Log(vixvol_df)

            # contractIndex =1 is the front month
            days2exp = []
            for contractIndex in range(self.nexclude+1, self.nfut+1):
                days2exp.append( self.DaysToExpire(self.Time.date(), contractIndex) )
            
            vixVols = self.VIXVol_interp(vixvol_df, days2exp)
            self.Log("vixvol")
            self.Log(vixVols)
    
            vxxvol = self.VXXVol()

            vols = vixVols.copy()
            vols.append(vxxvol)
        except:
            self.Log("error in processing")
            return None

        # now set up the objective and constraints for optimization
        def objective(weights):
            wv = pd.Series(weights).reset_index(drop=True) * \
                pd.Series(vols).reset_index(drop=True)
            wv = pd.DataFrame(wv).T
            
            wv.columns = corr_M.columns
            firstTerm = 0.5 * (wv @ corr_M @ wv.T).iloc[0].tolist()[0]
            secondTerm = pd.Series(weights[:-1]).reset_index(drop=True).dot(exret.reset_index(drop=True))
            thirdterm = self.CalCost(weights[:-1])

            return  firstTerm + secondTerm + thirdterm

        def contrain(weights):
            return 1 - sum(map(abs,weights[:-1]))

        cons = [{"type" : "ineq", "fun" : contrain}]
        w0 = [0.1] * (self.nfut - self.nexclude + 1)
        res = minimize(objective, w0, method="SLSQP", constraints = cons, options={"disp" : True})

        return res.x


    def CalCorrelationM(self):
        
        window = 35
        
        # excluding the front month and second month
        vxhist = self.History(self.VIX_symbols[self.nexclude:], timedelta(days=window), Resolution.Daily)
        # should we use settlement price or close price here?
        vxhist = vxhist['settle'].unstack(level=0)
        vxhist.columns = self.VIX_futures_names[self.nexclude:]

        vxxhist = self.History([self.vxx], timedelta(days=window), Resolution.Daily)
        vxxhist = vxxhist['close'].unstack(level=0)
        vxxhist.columns = ["VXX"]

        fdf = pd.merge(vxhist, vxxhist, right_index=True, left_index= True)

        fdf = fdf.shift(1)  # to avoid the look-ahead bias
        fdf.dropna(how="any", inplace = True)

        # calculate the daily return
        returndf = []
        for col in fdf.columns:
            returndf.append(fdf[col].pct_change(1))
            
        returndf = pd.concat(returndf, axis=1)
        returndf.dropna(how="any", inplace = True)

        return returndf.corr(method="pearson")

    def VXXVol(self):
        vxxvolhist = self.History([self.vxxvol], 5, Resolution.Daily)
        vxxvolhist.index = vxxvolhist.index.droplevel()
        vxxvolhist = vxxvolhist.shift(1)  # to avoid the look-ahead bias

        # sometimes the current date is not in the historical data, not sure why,
        # this will throw an exception which will be caught in the upper level
        vxxvolhist = pd.DataFrame(vxxvolhist.loc[self.Time.date()]).T
        # vxx term is 30 days
        # divide by 100 is arbitrary, just a test to keep the first term on the same order of other terms
        return vxxvolhist[self.vixvolColPrefix + "30"].tolist()[0] / 10.

    def VIXVol(self):
        vixvolhist = self.History([self.vixvol], 5, Resolution.Daily)
        
        # this sometimes (e.g. 02/05/2015) return index with less than 2 levels, 
        # I cannot debug because i run out of log limit today
        # it will throw exception
        vixvolhist.index = vixvolhist.index.droplevel()
        vixvolhist = vixvolhist.shift(1)  # to avoid the look-ahead bias

        # sometimes the current date is not in the historical data, not sure why,
        # this will throw an exception which will be caught in the upper level
        vixvolhist = pd.DataFrame(vixvolhist.loc[self.Time.date()]).T

        # only use the self.vixvolColPrefix columns
        vixvolhist = vixvolhist[vixvolhist.columns[vixvolhist.columns.to_series().str.contains( self.vixvolColPrefix + '[0-9]+')]]
        vixvolhist.columns = [int(x[6:]) for x in vixvolhist.columns]
        vixvolhist = vixvolhist.T
        vixvolhist.columns = ["VIXVol"]
        # divide by 100 is arbitrary, just a test to keep the first term on the same order of other terms
        return vixvolhist/10.
        
    def VIXVol_interp(self, vixvol_df, numOfDaysArr):
        for numOfDays in numOfDaysArr:
            vixvol_df.loc[numOfDays] = np.nan
        vixvol_df = vixvol_df.sort_index()
        vixvol_df = vixvol_df.interpolate(method='polynomial', order=2)
        return (vixvol_df.loc[numOfDaysArr])[vixvol_df.columns[0]].tolist()

    def ExRet(self):
        
        #get the close price from yesterday
        vxhist = self.History(self.VIX_symbols[self.nexclude:], timedelta(days=5), Resolution.Daily)
        # should we use settlement price or close price here?
        vxhist = vxhist['settle'].unstack(level=0)
        vxhist.columns = self.VIX_futures_names[self.nexclude:]

        vixhist = self.History([self.vix], 5, Resolution.Daily)
        vixhist = vixhist['close'].unstack(level=0)

        vixhist.columns = ["VIX"]
        vixhist.index = vixhist.index.date

        fdf = pd.merge(vxhist, vixhist, right_index=True, left_index= True)

        fdf = fdf.shift(1)  # to avoid the look-ahead bias
        fdf.dropna(how="any", inplace = True)

        for col in fdf.columns:
            if col == "VIX":
                continue
            contractIndex = int(col[-1])
            days2exp = self.DaysToExpire(self.Time.date(), contractIndex)
            fdf[col] = (fdf[col] - fdf["VIX"]) / fdf[col] / days2exp

        fdf = fdf.drop("VIX", axis=1)
        
        # sometimes the current date (e.g.2014.02.18) is not in the historical data, not sure why,
        # this will throw an exception which will be caught in the upper level
        return fdf.loc[self.Time.date()]

    # contractIndex=1 is the front month
    def DaysToExpire(self, today, contractIndex):
        index = bisect.bisect_left(self.expiry, today)
        return (self.expiry[index + contractIndex].date() - today).days

    def CalCost(self, weight):
        vxhist = self.History(self.VIX_symbols[self.nexclude:], timedelta(days=6), Resolution.Daily)
        # should we use settlement price or close price here?
        vxhist = vxhist['settle'].unstack(level=0)
        vxhist.columns = self.VIX_futures_names[self.nexclude:]
        vxhist = vxhist.shift(1)  # to avoid the look-ahead bias
        vxhist_pct_change = []
        for col in vxhist.columns:
            vxhist_pct_change.append(vxhist[col].pct_change(1))
            
        vxhist_pct_change = pd.concat(vxhist_pct_change, axis=1)
        vxhist_pct_change.dropna(how="any", inplace = True)
        
        pct_change = vxhist_pct_change.loc[self.Time.date()] + 1

        weight_with_return = pd.Series(self.curWeight).reset_index(drop=True) * pct_change.reset_index(drop=True)
        # if self.Time.date() in self.expiry:
        #     weight_with_return = weight_with_return.shift(-1, fill_value = 0)
    
        costRatio = (self.fee / vxhist.loc[self.Time.date()]).reset_index(drop=True) * \
            ((pd.Series(weight) - weight_with_return).abs()).reset_index(drop=True)

        return costRatio.sum()
        

    def OnEndOfAlgorithm(self):
        self.Liquidate()

    def PrintOrderStatus(self,orderticket):
        if orderticket.Status == OrderStatus.New:
            self.Log("order is new")
        elif orderticket.Status == OrderStatus.Submitted:
            self.Log("order is Submitted")
        elif orderticket.Status == OrderStatus.PartiallyFilled:
            self.Log("order is PartiallyFilled")
        elif orderticket.Status == OrderStatus.Filled:
            self.Log("order is Filled")
        elif orderticket.Status == OrderStatus.Canceled:
            self.Log("order is Canceled")
        elif orderticket.Status == OrderStatus.Invalid:
            self.Log("order is Invalid")
        elif orderticket.Status == OrderStatus.CancelPending:
            self.Log("order is CancelPending")
        elif orderticket.Status == OrderStatus.UpdateSubmitted:
            self.Log("order is UpdateSubmitted")
        else:
            self.Log("order is None")

    def PrintPort(self, status):
        invested = [x.Key for x in self.Portfolio if x.Value.Invested]
        for symbol in invested:
            security_holding = self.Portfolio[symbol]
            quantity = security_holding.Quantity
            price = security_holding.AveragePrice
            self.Log("holding " + str(symbol) + " " + str(quantity) + "@" + str(price) + \
                " current Px=" + str(self.Securities[symbol].Price) + \
                " UnrealizedProfit=" + str(security_holding.UnrealizedProfit) + \
                " TotalFees=" + str(security_holding.TotalFees) )

        self.Log(status + ": "
                 " Cash: " + str(self.Portfolio.Cash) +\
                 " UnsettledCash: " + str(self.Portfolio.UnsettledCash) +\
                 " TotalFees: " + str(self.Portfolio.TotalFees) +\
                 " TotalHoldingsValue: " + str(self.Portfolio.TotalHoldingsValue) +\
                 " MarginRemaining: " + str(self.Portfolio.MarginRemaining) +\
                 " TotalMarginUsed: " + str(self.Portfolio.TotalMarginUsed) +\
                 " TotalPortfolioValue: " + str(self.Portfolio.TotalPortfolioValue) +\
                 " TotalProfit: " + str(self.Portfolio.TotalProfit) +\
                 " TotalUnrealizedProfit: " + str(self.Portfolio.TotalUnrealizedProfit) 
                 )

class QuandlVix(PythonQuandl):
    def __init__(self):
        self.ValueColumnName = "vix Close"

class QuandlFutures(PythonQuandl):
    def __init__(self):
        self.ValueColumnName = "settle"
        
class QuandlVxx(PythonQuandl):
    def __init__(self):
        self.ValueColumnName =  "ivcall10"

class VXXData(PythonData):

    def GetSource(self, config, date, isLive):
        source = "https://www.dropbox.com/s/ux8ctn5xd3rfwur/vxx.csv?dl=1"
        return SubscriptionDataSource(source, SubscriptionTransportMedium.RemoteFile);


    def Reader(self, config, line, date, isLive):

        if not (line.strip() and line[0].isdigit()): return None

        data = line.split(',')
        vxx = VXXData()
        vxx.Symbol = config.Symbol
        vxx.Time = datetime.strptime(data[0], '%m-%d-%y')
        vxx.Value = data[1]
        vxx["close"] = float(data[1])

        return vxx