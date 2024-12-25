import matplotlib.pyplot as plt
import pandas as pd
df=pd.read_csv("data.csv")
def mean_squared_error(m,b,points):
    total_error=0
    for i in range(len(points)):
        x=points.iloc[i].m2
        y=points.iloc[i].price
        total_error+=(y-(m*x+b))**2
    MSE=total_error/len(points)
    return MSE
def gradient_calculation(Learning_rate,m_now,b_now,points):
    m_gradient=0
    b_gradient=0
    N=len(points)
    for i in range(N):
        x=points.iloc[i].m2
        y=points.iloc[i].price
        m_gradient+=(-2/N)*(x*(y-(m_now*x+b_now)))
        b_gradient+=(-2/N)*((y-(m_now*x+b_now)))#değişim hızı
    new_m=m_now-Learning_rate*m_gradient
    new_b=b_now-Learning_rate*b_gradient#learning_rate for minizmiing instant changes in gradient
    return new_m,new_b
epochs=1000
m_now=0.1
b_now=0.1   
L=0.00001
for i in range(epochs):

    m_now,b_now=gradient_calculation(L,m_now,b_now,df)
    mse=mean_squared_error(m_now,b_now,df)
    if i%25==0:
        print(f"Epoch: {i}, MSE: {mse:.2f}")
print(f"slope: {m_now:.2f} constant: {b_now:2f}")
plt.scatter(df.m2, df.price, color="black", label="Data points")
plt.plot(df.m2, m_now * df.m2 + b_now, color="red", label="Regression line")
plt.xlabel("m2 (area)")
plt.ylabel("price")
plt.title("Linear Regression model)")
plt.legend()
plt.show()
