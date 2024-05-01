import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import NearMiss
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout



df=pd.read_csv('data\hepatitis_csv.csv')

df = df.copy()

# Identify the continuous numeric features
continuous_features = ['age', 'bilirubin', 'alk_phosphate', 'sgot', 'albumin', 'protime']

gender_dict = {"male":1, "female":0}
feature_dict = {"No":0, "Yes":1}
def get_value(val,my_dict):
    for key,value in my_dict.items():
        if val == key:
            return value

def get_key(val,my_dict):
    for key,value in my_dict.items():
        if val == key:
            return key

def get_fvalue(val):
    feature_dict = {"No":0, "Yes":1}
    for key,value in feature_dict.items():
        if val == key:
            return value

# Fill missing values
for column in continuous_features:
    df[column] = df[column].fillna(df[column].mean())

for column in df.columns.drop(continuous_features):
    df[column] = df[column].fillna(df[column].mode().sample(1, random_state=1).values[0])

# Convert the booleans columns into integer columns
    for column in df.select_dtypes('bool'):
        df[column] = df[column].astype(np.int64)

# Encode the sex column as a binary feature
df['sex'] = df['sex'].replace({
    'female': 0,
    'male': 1
})

# Encode the class column as a binary feature
df['class'] = df['class'].replace({
    'live': 0,
    'die': 1
})

xfeatures =df[['age', 'sex', 'steroid', 'antivirals', 'fatigue', 'spiders',
       'ascites', 'varices', 'bilirubin', 'alk_phosphate', 'sgot', 'albumin',
       'protime', 'histology']]
ylabels = df['class']

smk = SMOTETomek(random_state=42)
xfeatures_new, ylabels_new = smk.fit_resample(xfeatures, ylabels)
pd.Series(ylabels_new).value_counts()

df_new = xfeatures_new.join(ylabels_new)

skb = SelectKBest(score_func=chi2,k=12)
best_feature_fit = skb.fit(xfeatures_new,ylabels_new)

feature_scores = pd.DataFrame(best_feature_fit.scores_,columns=['Feature_Scores'])
feature_column_names = pd.DataFrame(xfeatures_new.columns,columns=['Feature_name'])
best_feat_df = pd.concat([feature_scores,feature_column_names],axis=1)
best_feat_df.nlargest(12,'Feature_Scores')



df_b=df_new[['class','protime','age','bilirubin','alk_phosphate','sgot','albumin','spiders','ascites','fatigue','varices']]

xfeatures_b = df_b[['class','protime','age','bilirubin','alk_phosphate','sgot','albumin','spiders','ascites','fatigue','varices']]
ylabels_b = df_b[['class']]

# Lấy cột 'age' và 'class' từ DataFrame
X = df_new.drop(columns=['class']).values
#X = data[['bilirubin', 'alk_phosphate','sgot']].values
y = df_new['class'].values

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

n_features = x_train_scaled.shape[1]
timesteps = 1
# Reshape training and testing features for LSTM input
x_train_reshaped = x_train_scaled.reshape(-1, timesteps, n_features)
x_test_reshaped = x_test_scaled.reshape(-1, timesteps, n_features)

# Xây dựng mô hình BiLSTM
model = Sequential([
    Bidirectional(LSTM(64, return_sequences=True), input_shape=(timesteps, n_features)),
    Dropout(0.5),
    Bidirectional(LSTM(32)),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Biên dịch mô hình
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(x_train_reshaped, y_train, epochs=5, batch_size=32, validation_split=0.2)


y_pred_prob = model.predict(x_test_reshaped)
y_pred = (y_pred_prob > 0.5).astype(int)







age = st.number_input("Age",7,80)
sex = st.radio("Sex",tuple(gender_dict.keys()))
steroid = st.radio("Do You Take Steroids?",tuple(feature_dict.keys()))
antivirals = st.radio("Do You Take Antivirals?",tuple(feature_dict.keys()))
fatigue = st.radio("Do You Have Fatigue",tuple(feature_dict.keys()))
spiders = st.radio("Presence of Spider Naeve",tuple(feature_dict.keys()))
ascites = st.selectbox("Ascities",tuple(feature_dict.keys()))
varices = st.selectbox("Presence of Varices",tuple(feature_dict.keys()))
bilirubin = st.number_input("bilirubin Content",0.0,8.0)
alk_phosphate = st.number_input("Alkaline Phosphate Content",0.0,296.0)
sgot = st.number_input("Sgot",0.0,648.0)
albumin = st.number_input("Albumin",0.0,6.4)
protime = st.number_input("Prothrombin Time",0.0,100.0)
histology = st.selectbox("Histology",tuple(feature_dict.keys()))

new_data = [age, sex, steroid, antivirals, fatigue, 
                spiders, ascites, varices, bilirubin, alk_phosphate, sgot, albumin, protime, histology]
# Encode the sex column as a binary feature
# Convert the booleans columns into integer columns

feature_list = [age,get_value(sex,gender_dict),get_fvalue(steroid),get_fvalue(antivirals),get_fvalue(fatigue),get_fvalue(spiders),get_fvalue(ascites),get_fvalue(varices),bilirubin,alk_phosphate,sgot,albumin,int(protime),get_fvalue(histology)]
st.write(feature_list)
pretty_result = {"age":age,"sex":sex,"steroid":steroid,"antivirals":antivirals,"fatigue":fatigue,"spiders":spiders,"ascites":ascites,"varices":varices,"bilirubin":bilirubin,"alk_phosphate":alk_phosphate,"sgot":sgot,"albumin":albumin,"protime":protime,"histolog":histology}
st.json(pretty_result)
single_sample = np.array(feature_list).reshape(1,-1)



def predict_new_data(model, scaler, new_data, threshold):
    #new_data_scaled = scaler.transform([new_data])
    new_data_reshaped = new_data.reshape(-1, timesteps, n_features)
    predictions_prob = model.predict(new_data_reshaped)
    predictions_binary = (predictions_prob >= threshold).astype(int)
    return predictions_binary

def print_prediction(prediction):
    if prediction == 0:
        st.write("Predict_patient_ survivability: Die")
    else:
        st.write("Predict_patient_ survivability: Live")


threshold = 0.5
prediction_binary = predict_new_data(model, scaler, single_sample, threshold)
print_prediction(prediction_binary)
