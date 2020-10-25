from clr import AddReference # .NET Common Language Runtime (CLR) <- http://pythonnet.github.io/
AddReference("System")
AddReference("QuantConnect.Algorithm") # to load an assembly use AddReference
AddReference("QuantConnect.Common")

from System import * # CLR namespaces to be treatedas Python packages
from QuantConnect import *
from QuantConnect.Algorithm import *

# from QuantConnect.Python import PythonQuandl # quandl data not CLOSE
# from QuantConnect.Python import PythonData # custom data

from QuantConnect.Python import PythonQuandl
from QuantConnect.Data.Custom import *

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date
from scipy.optimize import minimize
import decimal
from io import StringIO
import bisect

class SimpleVIX(QCAlgorithm):

    def Initialize(self):

        self.SetStartDate(2011, 1, 1)   # Set Start Date
        self.SetEndDate(2019, 12, 31)     # Set End Date
        self.SetCash(10000000)          # Set Strategy Cash
        
        # if the weight of a futures contract is lower than this, we will not trade
        self.thresholdToPlaceOrder = 0.000001
        self.multipler = 1000 # VIX multipler is $1000

        holidays = self.TradingCalendar.GetDaysByType(TradingDayType.PublicHoliday, self.StartDate, self.EndDate)
        self.holidays = [i.Date.date() for i in holidays]
    
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)


        # VX9 misses 01/23 - 03/23 in 2015, exclude vx9 from modeling
        # need 8 monthly futures because when calculating the weights I will exclude the current front month
        # and the second month seems to have problem when computing PNL at the expiry date
        self.nfut = 8
        self.nexclude = 2
        self.VIX_futures_names = ["VX" + str(i) for i in range(1, 1 + self.nfut)]
        self.VIX_symbols =[]
        for vname in self.VIX_futures_names:
            self.VIX_symbols.append( self.AddData(QuandlFutures, "CHRIS/CBOE_" + vname, Resolution.Daily).Symbol )
          
        # the data of VXX or vxx.1 don't match with online sources (e.g. yahoo) after 2018
        # https://www.quantconnect.com/forum/discussion/7975/can-039-t-load-historical-vxx-data-even-though-it-looks-like-it-exists/p1
        # use this ticker before 2019
        self.vxx = self.AddEquity("vxx.1",  Resolution.Daily).Symbol                           # Add VXX, vxx.1 is the permtick according to the link
        # use this ticker after 2018
        #self.vxx = self.AddEquity("VXX",  Resolution.Daily).Symbol           
        
        vixfuture = self.AddFuture(Futures.Indices.VIX)
        vixfuture.SetFilter(timedelta(0), timedelta(300))

        # 2011 to present
        expiry_f = self.Download("https://www.dropbox.com/s/ny8nxqcp6u76igw/expiry.csv?dl=1")
        self.expiry = pd.read_csv(StringIO(expiry_f), names=["expiry"], parse_dates=["expiry"], infer_datetime_format=True)["expiry"].tolist()

        self.contracts = {}
        
        index = bisect.bisect_left(self.expiry, self.StartDate)
        # this is the expiration date of the front contract, market opens at 6pm ET. The platform default timezone is ET
        self.nextRebalanceDT = self.expiry[index] + timedelta(hours=18)
        self.nextRebalanceIndex = index

        # some dates have bad data, and mess up the backtesting, exclude them in the list
        # the hacky way to get around these bad days is to liquidate before these days, and 
        # then buy back after these days
        self.bad_dates = [date(2012,11,5), date(2012,11,6), date(2012,11,7),\
                          date(2013,10,10),date(2013,10,28),date(2013,10,29), date(2013,10,30),date(2013,10,31), date(2014,10,15)]

        # we trade at 10am ET next day the front month expires.
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.At(10,0), self.tryTrade)

    def OnData(self, data):
        pass
    
    def tryTrade(self):
        
        # for some reason the contracts are not necessary available at the first few calls
        # so I have accumulate the contracts in this way
        for chain in self.CurrentSlice.FutureChains.Values:
            contracts = chain.Contracts
            for contract in contracts.Values:
                if contract.Expiry.date() in self.contracts:
                    continue
                self.contracts[contract.Expiry.date()] = contract
                self.Log(str(contract.Symbol) + " expiry=" +  str(contract.Expiry.date()))

        # if next business day is a bad date, I will liquidate today
        nextBday = self.Time.date() + timedelta(days=1)
        while nextBday in self.holidays:
            nextBday += timedelta(days=1)
        if nextBday in self.bad_dates:
            self.Liquidate()

        if self.Time.date() in self.bad_dates:
            return
          
        if self.Time < self.nextRebalanceDT:
            return
        
        try:
            # calculate the weights of each VIX futures contract
            coeffs = self.cal_coeff()
        except:
            self.Log("error in calculating coeff on " + str(self.Time))
            return

        self.print_port("before liquidation")
        self.Liquidate()
        self.print_port("before trading")
        totalMargin = self.Portfolio.MarginRemaining / 2.0
        for i in range(self.nexclude, self.nfut):
            nextExpiry = self.expiry[self.nextRebalanceIndex + i].date()
            if not nextExpiry in self.contracts:
                self.Error("how come the contract with this expiry doesn't exist: " + \
                            str(nextExpiry))
                continue
            
            if (np.abs(coeffs[i-self.nexclude]) > self.thresholdToPlaceOrder):
                qty = int(totalMargin * coeffs[i-self.nexclude] / self.Securities[self.contracts[nextExpiry].Symbol].Price / self.multipler)
                # send a market order of VIX futures contract
                orderticket = self.MarketOrder(self.contracts[nextExpiry].Symbol, qty)

        qty = int(totalMargin / self.Securities[self.vxx].Price )
        # send a markt order of VXX 
        orderticket = self.MarketOrder(self.vxx, -qty)

        self.print_port("after trading")
        
        self.nextRebalanceIndex += 1
        self.nextRebalanceDT = self.expiry[self.nextRebalanceIndex] + timedelta(hours=18)


    def cal_coeff(self):
        
        window = 35
        
        # excluding the front month and second month
        vxhist = self.History(self.VIX_symbols[self.nexclude:], timedelta(days=window), Resolution.Daily)
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

        # now set up the objective and constraints for optimization
        def objective(weights):
            returndf["diff"] = returndf["VXX"]
            for i in range(len(weights)):
                returndf["diff"] -= weights[i] * returndf[returndf.columns[i]]
            return (returndf["diff"]**2).sum()

        def contrain(weights):
            return sum(weights) - 1

        # setup the contraints
        cons = [{"type" : "eq", "fun" : contrain}]

        #setup the boundaries
        bnd = (0, np.inf)
        bounds = tuple([bnd] * (self.nfut - self.nexclude))

        # setup the init guess
        w0 = [1/(self.nfut - self.nexclude)] * (self.nfut -self.nexclude)

        # now run the optimization
        res = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints = cons, options={"disp" : True})

        return res.x

    def OnEndOfAlgorithm(self):
        self.Liquidate()

    def print_order_status(self,orderticket):
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

    def print_port(self, status):
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