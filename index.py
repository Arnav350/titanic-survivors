import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('survivors.csv')
df_dropped = df.drop(['PassengerId', 'Name', 'Ticket', 'SibSp', 'Parch', 'Cabin'], axis=1)

leSex = LabelEncoder()
leEmbarked = LabelEncoder()
df_dropped['Le_Sex'] = leSex.fit_transform(df_dropped['Sex'])
df_dropped['Le_Embarked'] = leEmbarked.fit_transform(df_dropped['Embarked'])
df_dropped_label = df_dropped.drop(['Sex', 'Embarked'], axis=1)

df_dropped_na = df_dropped_label.dropna()

new_df = df_dropped_na.drop('Survived', axis=1)
target = df_dropped_na['Survived']

X_train, X_test, y_train, y_test = train_test_split(new_df, target, test_size=0.2)
model = RandomForestClassifier(n_estimators=20)
model.fit(X_train, y_train)

print(X_test[0:1])
data = [['2', 5, 000, 0, 0]]
  
df = pd.DataFrame(data, columns=['Pclass', 'Age', 'Fare', 'Le_Sex', 'Le_Embarked'])
  

st.title('Survivors Python App')
st.write('Fill out the inputs below and click submit')
passenger = st.radio('Passenger Class', ('1st Class', '2nd Class', '3rd Class'))
sex = st.radio('Male or Female', ('Male', 'Female'))
age = st.slider('Age', 0, 80)
fare = st.number_input('Fare in $', 0, 512)
embarked = st.radio('Port of Embarkment', ('Cherbourg', 'Queenstown', 'Southampton'))
submit = st.button('Submit')
if submit:
    passenger = (passenger[0])
    sex = (0 if sex == 'Male' else 1)
    if embarked == 'Cherbourg':
        embarked = 0
    elif embarked == 'Queenstown':
        embarked = 1
    elif embarked == 'Southampton':
        embarked = 2

    predict = model.predict(pd.DataFrame([[passenger, sex, age, fare, embarked]], columns=['Pclass', 'Age', 'Fare', 'Le_Sex', 'Le_Embarked']))
    st.write('You Survived' if predict == 1 else 'You did not survive')