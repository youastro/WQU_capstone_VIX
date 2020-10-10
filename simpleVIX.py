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
from datetime import datetime, timedelta
from scipy.optimize import minimize
import decimal
from io import StringIO
import bisect

class SimpleVIX(QCAlgorithm):

    def Initialize(self):

        self.SetStartDate(2014, 2, 1)   # Set Start Date
        self.SetEndDate(2018, 12, 31)     # Set End Date
        self.SetCash(10000000)          # Set Strategy Cash
        
        self.thresholdToPlaceOrder = 0.000001
        self.multipler = 1000 # VIX multipler is $1000
        
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage, AccountType.Margin)

        # self.vix = self.AddData(QuandlVix, "CBOE/VIX", Resolution.Daily).Symbol              # Add Quandl VIX price (daily)
        # self.es1 = self.AddData(QuandlFutures, "CHRIS/CME_ES1", Resolution.Daily).Symbol     # Add Quandl E-mini S&P500 front month futures data (daily)

        # need 8 monthly futures because when calculating the weights I will exclude the current front month
        # and the second month seems to have problem when computing PNL at the expiry date
        self.nfut = 8
        self.nexclude = 2
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

        self.contracts = {}
        
        index = bisect.bisect_left(self.expiry, self.StartDate)
        # this is the expiration date of the front contract, market opens at 6pm ET. The platform default timezone is ET
        self.nextRebalanceDT = self.expiry[index] + timedelta(hours=18)
        self.nextRebalanceIndex = index

        self.count = 1

        # we trade at 10am ET next day the front month expires.
        # if we don't liquidate the front month before it expires, quantconnect seems to have some problem to compute the correct PNL
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

        # self.count += 1
        # if self.count > 100:
        #     self.Quit()
                
        if self.Time < self.nextRebalanceDT:
            return
        
        coeffs = self.cal_coeff()

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
                orderticket = self.MarketOrder(self.contracts[nextExpiry].Symbol, qty)
                # self.Log("traded " + str(self.contracts[nextExpiry].Symbol) + " " + str(coeffs[i-1]) + " " \
                #     + str(nextExpiry) + " " + str(orderticket.QuantityFilled) + "@" + str(orderticket.AverageFillPrice))
                #self.print_order_status(orderticket)

        qty = int(totalMargin / self.Securities[self.vxx].Price )
        orderticket = self.MarketOrder(self.vxx, -qty)
        # self.Log("traded VXX" + str(orderticket.QuantityFilled) + "@" + str(orderticket.AverageFillPrice) + \
        #         " status: " + str(orderticket.Status) + " requested qty: " + str(qty) )
                
        #self.print_order_status(orderticket)

        self.print_port("after trading")
        
        self.nextRebalanceIndex += 1
        self.nextRebalanceDT = self.expiry[self.nextRebalanceIndex] + timedelta(hours=18)


    def cal_coeff(self):
        
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

        # now set up the objective and constraints for optimization
        def objective(weights):
            returndf["diff"] = returndf["VXX"]
            for i in range(len(weights)):
                returndf["diff"] -= weights[i] * returndf[returndf.columns[i]]
            return (returndf["diff"]**2).sum()

        def contrain(weights):
            return sum(weights) - 1

        cons = [{"type" : "eq", "fun" : contrain}]
        bnd = (0, np.inf)
        bounds = tuple([bnd] * (self.nfut - self.nexclude))
        w0 = [1/(self.nfut - self.nexclude)] * (self.nfut -self.nexclude)
        res = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints = cons, options={"disp" : True})
        #res = minimize(objective, w0, method="SLSQP")

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