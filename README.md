# Community-Microgrid
## 主要任務
本次任務進行電價媒合，因此最重要的任務就是先預測出 **「用電量」** 與 **「產電量」**
接著就根據前一次的媒合結果，做出我們要賣或是要買的滾動式調整的結果。
Design an agent for bidding power to minimize your electricity bill
## 模型架構
本次模型架構使用 LSTM ，以下為模型的詳細分層
<center><img src=https://i.imgur.com/ze8CLN2.png width=30% /></center>

## 預測結果
![](https://i.imgur.com/vh9lzbM.png)

## 模型策略
1. 先整理原始資料的日期結構
2. 使用 **[Gaussian Filter](https://medium.com/@bob800530/python-gaussian-filter-%E6%A6%82%E5%BF%B5%E8%88%87%E5%AF%A6%E4%BD%9C-676aac52ea17)** 當作是處理資料雜訊的工具
3. 餵入 guassianBlur() dataframe 包含兩串數值( generation and consumption )、還有 size
4. 根據 24\*7 個時段，決定下一個時段的 consumption 與 generation 
5. **train_x.shape = (198205,168,5)**
    - 198205 約表示 : 5833(個小時) * 49(組帳號) * 0.7(的比例要拿來做 training)
    - 168 表示 : 過去 24\*7 hour data 要拿來預測下一個階段的
    - 5 表示 : 餵進去的 features 有`gen`、`con`、`month`、`day`、`hour`，共5筆 features
6. 最後可以得出預估
    - **Gy** generate 
    - **Cy** consumption 
    - **buyP** buying price 
    - **sellP** selling price 
    - **balance** Gy - Gc
## 買賣策略
>根據上次的交易結果率，先訂定最低的價格，再根據上次的結果，依層次依序訂出本次交易的價格
- 買入
```python
sellP = sLast
if sSuccessRatio < 0.1 and sellP - 0.05 > sMin:
    sellP = sellP - 0.05
elif sSuccessRatio < 0.3 and sellP - 0.02 > sMin:
    sellP = sellP - 0.02
elif sSuccessRatio < 0.5 and sellP - 0.01 > sMin:
    sellP = sellP - 0.01
elif sSuccessRatio > 0.95:
    sellP = sellP + 0.01
```
- 賣出
```python
if bSuccessRatio < 0.0001 and bMean < 2.53:
    buyP = bMean if 2.51 < bMean else 2.51
elif bSuccessRatio < 0.1 and buyP + 0.05 < 2.53:
    buyP = buyP + 0.05 
elif bSuccessRatio < 0.3 and buyP + 0.02 < 2.53:
    buyP = buyP + 0.02
elif bSuccessRatio < 0.5 and buyP + 0.01 < 2.53:
    buyP = buyP + 0.01
elif bSuccessRatio > 0.95:
    buyP = buyP - 0.01
```

